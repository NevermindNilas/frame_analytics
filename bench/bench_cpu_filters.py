"""Paired public-API benchmark against a saved package directory.

    python bench/bench_cpu_filters.py --baseline build/optimization/baseline \
        --output build/optimization/confirmation.json

Setup is excluded; each sample includes three complete calls, including
autograd.grad for training cases. Both arms receive the same three frame pairs.
Use a quiet machine. Compilation is disabled unless --compiled is specified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frame_analytics as candidate


def load_baseline(path):
    spec = importlib.util.spec_from_file_location(
        "frame_analytics_baseline", Path(path) / "__init__.py",
        submodule_search_locations=[str(Path(path).resolve())])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()
    baseline = load_baseline(args.baseline)
    torch.set_num_threads(16)
    for module in (baseline, candidate):
        module.set_compile_enabled(args.compiled)
    order = random.Random(20260912)
    rng = np.random.default_rng(20260912)
    report = {"python": sys.version, "torch": torch.__version__,
              "platform": platform.platform(), "threads": torch.get_num_threads(),
              "compiled": args.compiled, "samples": 15, "calls_per_sample": 3,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": []}
    sizes = [(1, 512, 512), (1, 720, 1280), (2, 257, 389)]
    metrics = [("ssim", False), ("ssim", True), ("ms_ssim", False),
               ("ms_ssim", True), ("ssimulacra2", False)]
    if args.compiled:
        sizes, metrics = sizes[:1], metrics[:2]
    for n, h, w in sizes:
        generator = torch.Generator().manual_seed(827 + h)
        pairs = []
        for _ in range(3):
            x = torch.rand(n, 3, h, w, generator=generator)
            y = (x + 0.05 * torch.randn(x.shape, generator=generator)).clamp(0, 1)
            pairs.append((x, y))
        for metric, backward in metrics:
            inputs = [(x.requires_grad_(backward), y.requires_grad_(backward))
                      for x, y in pairs]
            kwargs = {} if metric == "ssimulacra2" else {"backend_hint": "torch"}

            def call(module, pair):
                out = getattr(module, metric)(*pair, **kwargs)
                if backward:
                    return out, torch.autograd.grad(out, pair)
                return out

            for module in (baseline, candidate):
                for pair in inputs:
                    call(module, pair)
            timings = {"baseline": [], "candidate": []}
            for _ in range(15):
                arms = [("baseline", baseline), ("candidate", candidate)]
                order.shuffle(arms)
                for name, module in arms:
                    start = time.perf_counter_ns()
                    for pair in inputs:
                        result = call(module, pair)
                    timings[name].append((time.perf_counter_ns() - start) / 3e6)
                    del result
            # Resample paired blocks, never the dependent inner calls.
            a, b = np.array(timings["baseline"]), np.array(timings["candidate"])
            indices = rng.integers(0, 15, size=(10000, 15))
            effects = 100 * (1 - np.median(b[indices], axis=1)
                             / np.median(a[indices], axis=1))
            case = {"metric": metric, "backward": backward, "shape": [n, 3, h, w],
                    "baseline_ms": statistics.median(a),
                    "candidate_ms": statistics.median(b),
                    "improvement_pct": 100 * (1 - statistics.median(b) / statistics.median(a)),
                    "bootstrap_95_pct": np.quantile(effects, [0.025, 0.975]).tolist(),
                    "raw_ms": timings}
            report["cases"].append(case)
            print(json.dumps({k: v for k, v in case.items() if k != "raw_ms"}), flush=True)
            Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
