"""Float32 error of pristine vs live code, measured against a float64 truth.

When a change reorders float32 arithmetic, "new differs from old by X" alone
cannot tell a rounding-level change from a real one. This prints each arm's
error against the pristine module run in float64 (eager); a rounding-level
change leaves both arms in the same error band.

    python -m bench.truth_perf_metrics --ref DIR --device cpu vifp:vifp fsim:fsim
"""
import argparse
import importlib
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.bench_perf_metrics import load_snapshot  # noqa: E402
from bench.check_perf_metrics import inputs  # noqa: E402
from frame_analytics import functional  # noqa: E402


def run(fn, x, y, dt, kw):
    xg = x.to(dt).requires_grad_(True)
    yg = y.to(dt).requires_grad_(True)
    out = fn(xg, yg, data_range=1.0, **kw)
    gx, gy = torch.autograd.grad(out.sum(), (xg, yg), allow_unused=True)
    z = torch.zeros_like(xg)
    return [t.detach().double() for t in (out, gx if gx is not None else z, gy if gy is not None else z)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shape", default="2x3x192x200")
    ap.add_argument("specs", nargs="+", help="module:function[:json-kwargs]")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    shape = tuple(int(v) for v in a.shape.split("x"))
    import json
    for spec in a.specs:
        parts = spec.split(":", 2)
        mod, fname = parts[0], parts[1]
        kw = json.loads(parts[2]) if len(parts) > 2 else {}
        old = getattr(load_snapshot(Path(a.ref), mod), fname)
        new = getattr(importlib.import_module(f"frame_analytics.{mod}"), fname)
        x, y = inputs(shape, a.device, torch.float32, 7)
        functional.set_compile_enabled(False)
        truth = run(old, x, y, torch.float64, kw)
        for comp in (False, True):
            functional.set_compile_enabled(comp)
            for name, fn in (("old", old), ("new", new)):
                o = run(fn, x, y, torch.float32, kw)
                e = [float((p - q).abs().max() / q.abs().max().clamp_min(1e-300)) for p, q in zip(o, truth)]
                print(f"{spec:28s} {name} {'compiled' if comp else 'eager   '} err vs f64: "
                      f"out {e[0]:.2e} gx {e[1]:.2e} gy {e[2]:.2e}", flush=True)


if __name__ == "__main__":
    main()
