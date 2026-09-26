"""Temporary CUDA peak-allocation check for an isolated candidate."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from bench.bench_new_multiscale import FUNCTIONS, WORKLOADS, data, load_snapshot
from frame_analytics import functional


parser = argparse.ArgumentParser()
parser.add_argument("--baseline-dir", type=Path, required=True)
parser.add_argument("--candidate-dir", type=Path, required=True)
parser.add_argument("--metric", choices=FUNCTIONS, required=True)
parser.add_argument("--workload", choices=WORKLOADS, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
functional.set_compile_enabled(True)
cases = data("cuda", WORKLOADS[args.workload])
fns = {label: load_snapshot(path, args.metric, label) for label, path in
       (("baseline", args.baseline_dir), ("candidate", args.candidate_dir))}
for fn in fns.values():
    for i in range(5):
        x, y = cases[i % 4]
        with torch.no_grad():
            fn(x, y, data_range=1.0, reduction="none")
        torch.cuda.synchronize()
rows = {label: [] for label in fns}
for i in range(4):
    for label, fn in fns.items():
        x, y = cases[i % 4]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        with torch.no_grad():
            out = fn(x, y, data_range=1.0, reduction="none")
        torch.cuda.synchronize()
        rows[label].append(torch.cuda.max_memory_allocated() - start)
        del out
print(json.dumps({"metric": args.metric, "workload": args.workload,
                  "shape": WORKLOADS[args.workload], "bytes": rows,
                  "median_bytes": {label: statistics.median(v) for label, v in rows.items()}}))
