"""Warm latency / peak-memory harness for the 15 metric modules added in 0.7.0.

Usage (from the repository root)::

    python -m bench.bench_perf_metrics --device cuda --metrics fsim,vsi \
        --shapes 1x3x256x256,1x3x512x512 --modes fwd,bwd --out results.jsonl

``--ab REF`` loads a second copy of every touched module from ``REF`` (a
directory holding pristine ``<module>.py`` files, e.g. extracted with
``git show HEAD:frame_analytics/fsim.py``) and times both arms *interleaved*
call-by-call on the same inputs, so machine-load drift hits both equally.
Snapshot modules get their own ``_maybe_compile`` key namespace, so the two
arms never share a compiled callable.

Timing is per call, synchronised on CUDA; the report gives min and median
over repeats. ``bwd`` times forward + ``backward()`` of the scalar output
with respect to the distorted input. CUDA peak memory is the extra
``max_memory_allocated`` above the pre-call level for one call.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import statistics
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frame_analytics  # noqa: E402,F401
from frame_analytics import functional  # noqa: E402

# name -> (module, function, kwargs)
SPECS = {
    "adm": ("adm", "adm_like", {}),
    "adm_db2": ("adm", "adm_like", {"wavelet": "db2_like"}),
    "ciede2000": ("ciede2000", "ciede2000", {}),
    "dss": ("dss", "dss", {}),
    "flip": ("flip", "flip", {}),
    "fsim": ("fsim", "fsim", {}),
    "fsimc": ("fsim", "fsimc", {}),
    "haarpsi": ("haarpsi", "haarpsi", {}),
    "iwssim": ("iwssim", "iw_ssim", {}),
    "iwssim_ratio": ("iwssim", "iw_ssim", {"weight_mode": "ratio"}),
    "mdsi": ("mdsi", "mdsi", {}),
    "ms_gmsd": ("ms_gmsd", "ms_gmsd", {}),
    "nlpd": ("nlpd", "nlpd", {}),
    "psnr_hvs": ("psnrhvs", "psnr_hvs", {}),
    "psnr_hvs_m": ("psnrhvs", "psnr_hvs_m", {}),
    "scielab": ("scielab", "scielab", {}),
    "srsim": ("srsim", "srsim", {}),
    "srsimc": ("srsim", "srsimc", {}),
    "vifp": ("vifp", "vifp", {}),
    "vsi": ("vsi", "vsi", {}),
}

DEFAULT_SHAPES = "1x3x256x256,1x3x512x512,1x3x1080x1920,4x3x512x512"


def _shim_functional(prefix: str):
    """A stand-in for ``functional`` whose compile cache keys are namespaced."""
    shim = types.ModuleType(f"_shim_functional_{prefix}")
    shim.__dict__.update(functional.__dict__)

    def _maybe_compile(fn, key, dynamic=None):
        return functional._maybe_compile(fn, f"{prefix}::{key}", dynamic)

    shim._maybe_compile = _maybe_compile
    return shim


def load_snapshot(ref_dir: Path, module: str, prefix: str = "ref"):
    """Import ``ref_dir/<module>.py`` as a sibling of the live package module."""
    name = f"frame_analytics._{prefix}_{module}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ref_dir / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    shim = _shim_functional(prefix)
    for attr, val in list(vars(mod).items()):
        if val is functional:
            setattr(mod, attr, shim)
        elif val is functional._maybe_compile:
            setattr(mod, attr, shim._maybe_compile)
    return mod


def resolve(metric: str, ref_dir: Path | None = None):
    module, fn, kw = SPECS[metric]
    mod = (load_snapshot(ref_dir, module) if ref_dir is not None
           else importlib.import_module(f"frame_analytics.{module}"))
    f = getattr(mod, fn)
    return lambda x, y: f(x, y, data_range=1.0, **kw)


def make_inputs(shape, device, seed=20261114):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, c, h, w = shape
    # Smooth structure + texture so saliency / phase congruency are not
    # degenerate (pure noise is atypical for every one of these metrics).
    yy = torch.linspace(0, 1, h).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w).view(1, 1, 1, w)
    base = 0.5 + 0.3 * torch.sin(12 * xx + 5 * yy + torch.arange(c).view(1, c, 1, 1))
    x = (base + 0.15 * torch.rand(shape, generator=g)).clamp(0, 1)
    y = (x + 0.05 * torch.randn(shape, generator=g)).clamp(0, 1)
    return x.to(device), y.to(device)


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def one_call(fn, x, y, mode):
    if mode == "fwd":
        with torch.no_grad():
            return fn(x, y)
    yg = y.detach().requires_grad_(True)
    out = fn(x, yg)
    out.backward()
    return out


def timed(fn, x, y, mode, device):
    _sync(device)
    t0 = time.perf_counter_ns()
    one_call(fn, x, y, mode)
    _sync(device)
    return (time.perf_counter_ns() - t0) / 1e6


def peak_mem(fn, x, y, mode, device):
    if device != "cuda":
        return None
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    one_call(fn, x, y, mode)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def bench_config(arms, x, y, mode, device, min_reps, budget_s, warmup):
    for fn in arms.values():
        for _ in range(warmup):
            one_call(fn, x, y, mode)
    _sync(device)
    times = {k: [] for k in arms}
    t_start = time.perf_counter()
    reps = 0
    order = list(arms)
    while reps < min_reps or (time.perf_counter() - t_start < budget_s and reps < 200):
        for k in (order if reps % 2 == 0 else order[::-1]):
            times[k].append(timed(arms[k], x, y, mode, device))
        reps += 1
    out = {}
    for k, ts in times.items():
        out[k] = {"min_ms": min(ts), "med_ms": statistics.median(ts), "reps": len(ts),
                  "peak_mib": peak_mem(arms[k], x, y, mode, device)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metrics", default=",".join(SPECS))
    ap.add_argument("--shapes", default=DEFAULT_SHAPES)
    ap.add_argument("--modes", default="fwd,bwd")
    ap.add_argument("--ab", default=None, help="dir of pristine modules for interleaved A/B")
    ap.add_argument("--only-ref", action="store_true",
                    help="time only the --ab snapshot (baseline runs while editing)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-reps", type=int, default=5)
    ap.add_argument("--budget", type=float, default=1.5, help="seconds per config")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    if a.no_compile:
        functional.set_compile_enabled(False)
    shapes = [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")]
    ref = Path(a.ab) if a.ab else None
    sink = open(a.out, "a") if a.out else None
    for metric in a.metrics.split(","):
        arms = {} if a.only_ref else {"new": resolve(metric)}
        if ref is not None:
            arms["ref"] = resolve(metric, ref)
        for shape in shapes:
            x, y = make_inputs(shape, a.device)
            for mode in a.modes.split(","):
                try:
                    res = bench_config(arms, x, y, mode, a.device, a.min_reps,
                                       a.budget, a.warmup)
                except Exception as exc:  # report and continue
                    res = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                rec = {"metric": metric, "device": a.device, "shape": list(shape),
                       "mode": mode, "threads": a.threads,
                       "compile": not a.no_compile, "res": res}
                line = json.dumps(rec)
                if sink:
                    sink.write(line + "\n")
                    sink.flush()
                if "error" in res:
                    msg = res["error"]
                else:
                    msg = "  ".join(
                        f"{k}: {v['min_ms']:8.3f}/{v['med_ms']:8.3f}ms"
                        + (f" {v['peak_mib']:7.1f}MiB" if v["peak_mib"] is not None else "")
                        for k, v in res.items())
                    if "ref" in res and "new" in res:
                        msg += f"  new/ref(min)={res['new']['min_ms'] / res['ref']['min_ms']:.3f}"
                print(f"{metric:13s} {a.device:4s} {'x'.join(map(str, shape)):14s} {mode:3s} {msg}",
                      flush=True)
            del x, y
            if a.device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
