"""Paired 20-call steady-stream benchmark for the four pyramid metrics.

The earlier one-call end-to-end CUDA timings had 95% intervals spanning large
gains and regressions while desktop applications used 19--43% of the GPU.
This protocol keeps the same four seeded fixtures and ten paired samples, but
each arm performs twenty serial calls before one device synchronization. It
reports amortized milliseconds per call for a warm stream. This is a distinct
metric from synchronized single-call latency; both arms use identical work.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from time import perf_counter_ns
from pathlib import Path

import torch

from bench.bench_new_multiscale import FUNCTIONS, WORKLOADS, data, load_snapshot, synchronize
from frame_analytics import functional


def warm(fn, cases, device: str, warmup: int):
    for i in range(warmup):
        x, y = cases[i % len(cases)]
        with torch.no_grad():
            result = fn(x, y, data_range=1.0, reduction="none")
        synchronize(device)
        float(result.sum().item())


def timed_batch(fn, cases, device: str, sample: int, calls: int):
    results = []
    synchronize(device)
    start_ns = perf_counter_ns()
    with torch.no_grad():
        for j in range(calls):
            x, y = cases[(sample + j) % len(cases)]
            results.append(fn(x, y, data_range=1.0, reduction="none"))
    synchronize(device)
    batch_ms = (perf_counter_ns() - start_ns) / 1e6
    # Consume every result outside the timing boundary. Holding only scalar
    # outputs does not retain image-sized intermediates between calls.
    checksum = float(torch.stack(results).sum().item())
    return batch_ms, batch_ms / calls, checksum


def run(args):
    torch.set_num_threads(args.threads)
    functional.set_compile_enabled(True)
    cases = data(args.device, WORKLOADS[args.workload])
    functions = {
        "baseline": load_snapshot(args.baseline_dir, args.metric, "baseline"),
        "candidate": load_snapshot(args.candidate_dir, args.metric, "candidate"),
    }
    for fn in functions.values():
        warm(fn, cases, args.device, args.warmup)
    rng = random.Random(20260923)
    rows = []
    for sample in range(args.samples):
        arms = list(functions)
        rng.shuffle(arms)
        for arm in arms:
            batch_ms, ms_per_call, checksum = timed_batch(
                functions[arm], cases, args.device, sample, args.calls)
            rows.append({"sample": sample, "arm": arm, "batch_ms": batch_ms,
                         "ms_per_call": ms_per_call, "checksum": checksum})
    stats = {arm: {"median_ms_per_call": statistics.median(
        r["ms_per_call"] for r in rows if r["arm"] == arm)}
        for arm in functions}
    print(json.dumps({"metric": args.metric, "device": args.device,
                      "workload": args.workload, "shape": WORKLOADS[args.workload],
                      "threads": args.threads, "compile_enabled": True,
                      "warmup": args.warmup, "samples": args.samples,
                      "calls_per_sample": args.calls, "rows": rows, "stats": stats}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--metric", choices=FUNCTIONS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--workload", choices=WORKLOADS, default="primary")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--calls", type=int, default=20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
