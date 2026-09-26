"""Reproducible warm latency/profile harness for the four new pyramid metrics.

The baseline directory contains byte-for-byte copies of the user's initial
untracked metric modules. Run this script under the repository's Python with
``--baseline-dir`` and save its JSON lines output for comparison.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

# Running ``python bench/bench_new_multiscale.py`` otherwise puts ``bench/``
# first on sys.path and can silently import an installed, stale package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_analytics import functional


FUNCTIONS = {
    "ms_gmsd": ("ms_gmsd", "ms_gmsd"),
    "vifp": ("vifp", "vifp"),
    "iwssim": ("iwssim", "iw_ssim"),
    "nlpd": ("nlpd", "nlpd"),
}
WORKLOADS = {"primary": (1, 3, 256, 256), "heldout": (1, 1, 192, 256)}


def load_snapshot(path: Path, metric: str, prefix: str):
    module_name = f"frame_analytics._{prefix}_{metric}"
    spec = importlib.util.spec_from_file_location(module_name, path / f"{metric}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The shared compile cache is keyed by a string, so namespace snapshot
    # epilogues to prevent one arm from accidentally running the other's code.
    mod._maybe_compile = lambda fn, key, dynamic=None: functional._maybe_compile(
        fn, f"{prefix}:{key}", dynamic)
    return getattr(mod, FUNCTIONS[metric][1])


def load_baseline(path: Path, metric: str):
    return load_snapshot(path, metric, "baseline")


def data(device: str, shape: tuple[int, ...]):
    gen = torch.Generator(device=device).manual_seed(20260923)
    cases = []
    for _ in range(4):
        x = torch.rand(shape, generator=gen, device=device)
        noise = torch.randn(shape, generator=gen, device=device)
        y = (x + 0.04 * noise).clamp(0, 1)
        cases.append((x, y))
    return cases


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def timed(fn, cases, device: str, index: int):
    x, y = cases[index % len(cases)]
    synchronize(device)
    begin = time.perf_counter_ns()
    with torch.no_grad():
        result = fn(x, y, data_range=1.0, reduction="none")
    synchronize(device)
    elapsed = (time.perf_counter_ns() - begin) / 1e6
    return elapsed, float(result.sum().item())


def run(args):
    torch.set_num_threads(args.threads)
    functional.set_compile_enabled(True)
    shape = WORKLOADS[args.workload]
    cases = data(args.device, shape)
    baseline = load_baseline(args.baseline_dir, args.metric)
    functions = {"baseline": baseline}
    if args.phase == "compare":
        if args.candidate_dir is None:
            module = importlib.import_module(f"frame_analytics.{args.metric}")
            functions["candidate"] = getattr(module, FUNCTIONS[args.metric][1])
        else:
            functions["candidate"] = load_snapshot(args.candidate_dir,
                                                   args.metric, "candidate")
    for label, fn in functions.items():
        for i in range(args.warmup):
            timed(fn, cases, args.device, i)
    rng = random.Random(20260923)
    rows = []
    for i in range(args.samples):
        labels = list(functions)
        rng.shuffle(labels)
        for label in labels:
            ms, checksum = timed(functions[label], cases, args.device, i)
            rows.append({"sample": i, "arm": label, "ms": ms, "checksum": checksum})
    stats = {label: {"median_ms": statistics.median(
        row["ms"] for row in rows if row["arm"] == label),
        "min_ms": min(row["ms"] for row in rows if row["arm"] == label)}
        for label in functions}
    print(json.dumps({"metric": args.metric, "phase": args.phase,
                      "device": args.device, "shape": shape,
                      "threads": args.threads, "workload": args.workload,
                      "compile_enabled": True, "warmup": args.warmup,
                      "samples": args.samples, "rows": rows, "stats": stats}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--metric", choices=FUNCTIONS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--workload", choices=WORKLOADS, default="primary")
    parser.add_argument("--phase", choices=("baseline", "compare"), default="baseline")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
