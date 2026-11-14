"""Differential check: live metric modules vs a pristine snapshot directory.

Runs every case on both arms with identical seeded inputs and compares the
output (score, per-image vector or map) and the input gradients with respect
to *both* images. Prints one line per case with max abs / rel differences and
exits non-zero if any case exceeds ``--rtol``/``--atol`` (defaults are
float-rounding level for float32 work) or if NaN/Inf patterns disagree.

    python -m bench.check_perf_metrics --ref /path/to/pristine --device cuda
"""

from __future__ import annotations

import argparse
import math
import zlib
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.bench_perf_metrics import load_snapshot  # noqa: E402
from frame_analytics import functional  # noqa: E402
import importlib  # noqa: E402

RGB = [(2, 3, 192, 200), (1, 3, 177, 263), (1, 3, 540, 720)]
GRAY = [(3, 1, 193, 181)]

# (module, function, kwargs list, shapes, differentiable)
CASES = [
    ("adm", "adm_like", [{}, {"wavelet": "db2_like"}, {"reduction": "none", "scales": 3},
                         {"wavelet": "db2_like", "csf": (1.0, 2.0, 0.5), "reduction": "none"}],
     RGB + GRAY, True),
    ("ciede2000", "ciede2000", [{}, {"reduction": "none"}, {"return_map": True},
                                {"kL": 2.0, "kC": 1.5, "kH": 0.7}], RGB, True),
    ("dss", "dss", [{}, {"reduction": "none"}, {"luma": "bt709"}, {"luma": False}],
     RGB + GRAY, True),
    ("flip", "flip", [{}, {"reduction": "none"}, {"return_map": True},
                      {"ppd": 30.0, "exposure": 0.5}], RGB, True),
    ("fsim", "fsim", [{}, {"reduction": "none"}, {"return_map": True}, {"luma": "bt709"},
                      {"pc_mode": "chunked"}], RGB + GRAY, True),
    ("fsim", "fsimc", [{}, {"reduction": "none"}, {"return_map": True}], RGB, True),
    ("haarpsi", "haarpsi", [{}, {"reduction": "none"}, {"subsample": False},
                            {"luma": "bt601"}], RGB + GRAY, True),
    ("iwssim", "iw_ssim", [{}, {"reduction": "none"}, {"weight_mode": "ratio"},
                           {"weight_mode": "log"}, {"weight_mode": "uniform"},
                           {"parent": False}, {"luma": "bt601"}],
     [(2, 3, 192, 200), (1, 3, 181, 263), (1, 1, 256, 256)], True),
    ("mdsi", "mdsi", [{}, {"reduction": "none"}, {"return_map": True},
                      {"combination": "mult"}, {"gradient": "sobel", "downsample": False}],
     RGB + GRAY, True),
    ("ms_gmsd", "ms_gmsd", [{}, {"reduction": "none"}, {"downsample": False},
                            {"luma": "bt709"}], RGB + GRAY, True),
    ("nlpd", "nlpd", [{}, {"reduction": "none"}, {"scales": 3}, {"luma": "bt601"}],
     RGB + GRAY, True),
    ("psnrhvs", "psnr_hvs", [{}, {"reduction": "none"}], RGB + GRAY, True),
    ("psnrhvs", "psnr_hvs_m", [{}, {"reduction": "none"}], RGB + GRAY, True),
    ("psnrhvs", "mse_hvs_m", [{"reduction": "none"}], RGB, True),
    ("scielab", "scielab", [{}, {"reduction": "none"}, {"return_map": True},
                            {"ppd": 60.0}], RGB, True),
    ("srsim", "srsim", [{}, {"reduction": "none"}, {"downsample": False}],
     RGB + GRAY, True),
    ("srsim", "srsimc", [{}, {"reduction": "none"}], RGB, True),
    ("vifp", "vifp", [{}, {"reduction": "none"}, {"num_scales": 2}, {"luma": "bt601"}],
     RGB + GRAY, True),
    ("vsi", "vsi", [{}, {"reduction": "none"}], RGB + GRAY, True),
]


def inputs(shape, device, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    n, c, h, w = shape
    yy = torch.linspace(0, 1, h).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w).view(1, 1, 1, w)
    base = 0.5 + 0.3 * torch.sin(9 * xx + 4 * yy + torch.arange(c).view(1, c, 1, 1))
    x = (base + 0.2 * torch.rand(shape, generator=g)).clamp(0, 1)
    y = (x + 0.06 * torch.randn(shape, generator=g)).clamp(0, 1)
    if dtype == torch.uint8:
        return (x * 255).round().to(torch.uint8).to(device), \
            (y * 255).round().to(torch.uint8).to(device)
    return x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype)


def run(fn, x, y, kw, grad):
    if not grad:
        with torch.no_grad():
            return fn(x, y, **kw), None, None
    xg = x.detach().clone().requires_grad_(True)
    yg = y.detach().clone().requires_grad_(True)
    out = fn(xg, yg, **kw)
    # A fixed, non-uniform cotangent catches per-image / per-pixel mix-ups.
    cot = torch.linspace(0.5, 1.5, out.numel(), device=out.device,
                         dtype=out.dtype).view_as(out)
    gx, gy = torch.autograd.grad((out * cot).sum(), (xg, yg), allow_unused=True)
    return out.detach(), gx, gy


def diff(a, b):
    if a is None or b is None:
        return (0.0, 0.0, a is None and b is None)
    a64, b64 = a.double(), b.double()
    fa, fb = torch.isfinite(a64), torch.isfinite(b64)
    same_pattern = bool((fa == fb).all())
    if same_pattern and (~fa).any():
        # NaN/+inf/-inf must match position for position.
        same_pattern = bool((a64[~fa].nan_to_num(0) == b64[~fb].nan_to_num(0)).all())
    m = fa & fb
    if not m.any():
        return 0.0, 0.0, same_pattern
    d = (a64[m] - b64[m]).abs()
    scale = b64[m].abs().max().clamp_min(1e-30)
    return float(d.max()), float(d.max() / scale), same_pattern


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--modules", default=None)
    ap.add_argument("--rtol", type=float, default=2e-5)
    ap.add_argument("--grad-rtol", type=float, default=2e-4)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--dtypes", default="float32,float64,uint8")
    a = ap.parse_args(argv)
    if a.no_compile:
        functional.set_compile_enabled(False)
    warnings.simplefilter("ignore")
    ref = Path(a.ref)
    want = set(a.modules.split(",")) if a.modules else None
    dtypes = [getattr(torch, d) for d in a.dtypes.split(",")]
    worst = {"out": 0.0, "grad": 0.0}
    failures = 0
    n_cases = 0
    for module, fname, kws, shapes, differentiable in CASES:
        if want and module not in want:
            continue
        new = getattr(importlib.import_module(f"frame_analytics.{module}"), fname)
        old = getattr(load_snapshot(ref, module), fname)
        for shape in shapes:
            for dt in dtypes:
                if dt == torch.float64 and shape[-1] > 300:
                    continue  # keep float64 cases quick
                for kw in kws:
                    grad = differentiable and dt.is_floating_point
                    seed = zlib.crc32(repr((module, fname, shape, str(dt), sorted(kw.items()))).encode())
                    x, y = inputs(shape, a.device, dt, seed)
                    try:
                        ro = run(old, x, y, kw, grad)
                    except Exception as exc:
                        try:
                            run(new, x, y, kw, grad)
                            print(f"FAIL {module}.{fname} {shape} {dt} {kw}: ref raised "
                                  f"{type(exc).__name__} but new did not")
                            failures += 1
                        except Exception as exc2:
                            ok = type(exc2) is type(exc)
                            print(f"{'ok  ' if ok else 'FAIL'} {module}.{fname} {shape} {dt} {kw}: "
                                  f"both raise {type(exc).__name__}/{type(exc2).__name__}")
                            failures += 0 if ok else 1
                        continue
                    rn = run(new, x, y, kw, grad)
                    n_cases += 1
                    do, ro_rel, po = diff(rn[0], ro[0])
                    gx = diff(rn[1], ro[1])
                    gy = diff(rn[2], ro[2])
                    shape_ok = rn[0].shape == ro[0].shape and rn[0].dtype == ro[0].dtype
                    g_rel = max(gx[1], gy[1])
                    bad = (not shape_ok or not po or not gx[2] or not gy[2]
                           or ro_rel > a.rtol or g_rel > a.grad_rtol
                           or (math.isnan(ro_rel)))
                    worst["out"] = max(worst["out"], ro_rel)
                    worst["grad"] = max(worst["grad"], g_rel)
                    failures += bad
                    print(f"{'FAIL' if bad else 'ok  '} {module}.{fname} {tuple(shape)} "
                          f"{str(dt)[6:]} {kw}: out abs={do:.2e} rel={ro_rel:.2e} "
                          f"grad rel x={gx[1]:.2e} y={gy[1]:.2e}"
                          + ("" if shape_ok else f" SHAPE/DTYPE {rn[0].shape}/{rn[0].dtype} vs "
                             f"{ro[0].shape}/{ro[0].dtype}"), flush=True)
    print(f"cases={n_cases} failures={failures} worst out rel={worst['out']:.3e} "
          f"worst grad rel={worst['grad']:.3e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
