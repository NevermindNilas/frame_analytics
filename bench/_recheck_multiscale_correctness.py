"""Temporary differential checks for isolated metric candidates."""

import argparse
import json
from pathlib import Path

import torch

from bench.bench_new_multiscale import FUNCTIONS, load_snapshot
from frame_analytics import functional


parser = argparse.ArgumentParser()
parser.add_argument("--baseline-dir", type=Path, required=True)
parser.add_argument("--candidate-dir", type=Path, required=True)
parser.add_argument("--metric", choices=FUNCTIONS, required=True)
parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
args = parser.parse_args()
torch.set_num_threads(4)
functional.set_compile_enabled(True)
before = load_snapshot(args.baseline_dir, args.metric, "baseline")
after = load_snapshot(args.candidate_dir, args.metric, "candidate")
gen = torch.Generator(device=args.device).manual_seed(6902026)
rows = []
for name, shape, dtype, mode in [
    ("rgb_f32", (1, 3, 192, 192), torch.float32, "noise"),
    ("gray_f32", (2, 1, 192, 256), torch.float32, "noise"),
    ("rgb_f64", (1, 3, 192, 192), torch.float64, "noise"),
    ("rgb_u8", (1, 3, 192, 192), torch.uint8, "noise"),
    ("identical", (1, 3, 192, 192), torch.float32, "identical"),
    ("constant", (1, 3, 192, 192), torch.float32, "constant"),
]:
    if dtype == torch.uint8:
        x = torch.randint(0, 256, shape, generator=gen, device=args.device,
                          dtype=dtype)
        y = (x.to(torch.int16) + 2).clamp(0, 255).to(dtype)
        kw = {"reduction": "none"}
    else:
        x = torch.rand(shape, generator=gen, device=args.device, dtype=dtype)
        y = (x + 0.04 * torch.randn(shape, generator=gen, device=args.device,
                                     dtype=dtype)).clamp(0, 1)
        kw = {"data_range": 1.0, "reduction": "none"}
    if mode == "identical":
        y = x
    elif mode == "constant":
        x = x * 0 + 0.5
        y = y * 0 + 0.52
    with torch.no_grad():
        a = before(x, y, **kw)
        b = after(x, y, **kw)
    diff = (a - b).abs().max().item()
    atol, rtol = (1e-10, 1e-8) if dtype == torch.float64 else (1e-6, 1e-5)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), (
        name, a, b, diff)
    rows.append({"case": name, "max_abs": diff})
print(json.dumps({"metric": args.metric, "device": args.device, "cases": rows}))
