"""torch.profiler hotspot table for one metric / shape / mode.

    python -m bench.profile_perf_metrics fsim cuda 1x3x512x512 bwd [rows] [warm-shape]

``warm-shape`` first runs the metric at another shape, so compiled regions
are profiled in the state a multi-resolution process sees (dynamic shapes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.bench_perf_metrics import make_inputs, one_call, resolve  # noqa: E402


def main():
    metric, device, shape, mode = sys.argv[1:5]
    rows = int(sys.argv[5]) if len(sys.argv) > 5 else 25
    torch.set_num_threads(8)
    shape = tuple(int(v) for v in shape.split("x"))
    fn = resolve(metric)
    if len(sys.argv) > 6:
        wshape = tuple(int(v) for v in sys.argv[6].split("x"))
        wx, wy = make_inputs(wshape, device)
        for _ in range(2):
            one_call(fn, wx, wy, mode)
    x, y = make_inputs(shape, device)
    for _ in range(3):
        one_call(fn, x, y, mode)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    if device == "cuda":
        torch.cuda.synchronize()
    with profile(activities=acts) as prof:
        for _ in range(3):
            one_call(fn, x, y, mode)
        if device == "cuda":
            torch.cuda.synchronize()
    key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=rows, max_name_column_width=60))
    if device == "cuda":
        n = sum(e.count for e in prof.key_averages() if e.device_type.name == "CUDA")
        print(f"CUDA kernel launches per call ~ {n / 3:.0f}")


if __name__ == "__main__":
    main()
