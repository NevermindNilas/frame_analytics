"""SSIMULACRA 2: speed, memory, and what the alternatives cost.

    python bench/bench_ssimulacra2.py
    python bench/bench_ssimulacra2.py --quick        # CUDA speed table only

Why this one spawns subprocesses
--------------------------------
Every other benchmark here times its cases in one process. This metric cannot
be measured that way and be honest about it: it is a six-scale pyramid, so one
call puts six shapes through the same compiled functions, and ``torch.compile``
specialises per shape. Six entries per input resolution against Dynamo's
recompile limit of 8 means the *second* resolution a process sees drops the
whole metric back to eager -- so a naive sweep would show 4.6 ms at 1080p and
then quietly report eager numbers for every row after it. Each case therefore
runs in a fresh interpreter, which is also what a real user gets: one
resolution per run.

There is no fast GPU baseline to compare against in Python. The
implementations in circulation are libjxl's C++ tool, the Rust port behind
``ssimulacra2_rs``, and CUDA/HIP kernels in vship -- none of them pip-installable
next to a torch tensor. The one PyPI package, ``ssimulacra2`` (a numpy/scipy
transcription), is timed here for scale, with the caveat printed alongside it:
it blurs with scipy's mirrored-border Gaussian instead of libjxl's zero-padded
recursive filter, so it is not computing quite the same number.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_analytics as fa

MIB = 2 ** 20

CUDA_SIZES = ((1, 512, 512), (1, 720, 1280), (1, 1080, 1920), (4, 1080, 1920),
              (1, 2160, 3840))
CPU_SIZES = ((1, 512, 512), (1, 720, 1280), (1, 1080, 1920))


# --------------------------------------------------------------------------- #
# the child process: one case, one number
# --------------------------------------------------------------------------- #


def _pair(n, h, w, device, seed=0):
    """A frame and a mildly damaged copy of it.

    Not two independent random tensors: SSIMULACRA 2 spends its time in the
    same kernels either way, but a pair that is *not* a distortion of the same
    image scores around -40 and makes every printed value meaningless.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    base = (128 + 60 * torch.sin(xx / 37.0) * torch.cos(yy / 53.0)
            + 40 * torch.sin((xx + yy) / 17.0))
    a = base.clamp(0, 255).expand(n, 3, h, w).contiguous()
    b = (a + 6.0 * torch.randn(a.shape, generator=g)).clamp(0, 255)
    return (a.to(torch.uint8).to(device), b.to(torch.uint8).to(device))


def _timeit(fn, *, iters, repeat, cuda):
    for _ in range(max(3, iters // 2)):
        fn()
    if cuda:
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeat):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if cuda:
            torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best


def _peak_mib(fn):
    fn()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - before) / MIB
    del out
    return peak


def run_case(kind, device, n, h, w, dtype):
    """One measurement, in this process. Prints a JSON line."""
    if kind == "eager":
        fa.set_compile_enabled(False)
    cuda = device == "cuda"
    wdt = {"fp32": torch.float32, "fp64": torch.float64}[dtype]
    a, b = _pair(n, h, w, device)
    fn = lambda: fa.ssimulacra2(a, b, data_range=255.0, dtype=wdt)  # noqa: E731

    if kind == "memory":
        return {"peak_mib": _peak_mib(fn)}

    if kind == "numpy":
        # the PyPI implementation, on the same pixels. Its public entry point
        # takes two file paths, so this drives the internals directly rather
        # than timing a PNG decode.
        from ssimulacra2 import ssimulacra2 as pkg

        arr_a = a[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        arr_b = b[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)

        def np_call():
            la = pkg.srgb_to_linear(arr_a.copy())
            lb = pkg.srgb_to_linear(arr_b.copy())
            m = pkg.Msssim()
            i1 = pkg.make_positive_xyb(pkg.linear_rgb_to_xyb(la))
            i2 = pkg.make_positive_xyb(pkg.linear_rgb_to_xyb(lb))
            for scale in range(pkg.kNumScales):
                if i1.shape[0] < 8 or i1.shape[1] < 8:
                    break
                mu1, mu2 = pkg.blur_image(i1), pkg.blur_image(i2)
                sd = pkg.MsssimScale()
                sd.avg_ssim = pkg.ssim_map(mu1, mu2, pkg.blur_image(i1 * i1),
                                           pkg.blur_image(i2 * i2),
                                           pkg.blur_image(i1 * i2))
                sd.avg_edgediff = pkg.edge_diff_map(i1, mu1, i2, mu2)
                m.scales.append(sd)
                if scale < pkg.kNumScales - 1:
                    la, lb = pkg.downsample(la, 2, 2), pkg.downsample(lb, 2, 2)
                    i1 = pkg.make_positive_xyb(pkg.linear_rgb_to_xyb(la))
                    i2 = pkg.make_positive_xyb(pkg.linear_rgb_to_xyb(lb))
            return m.score()

        return {"ms": _timeit(np_call, iters=1, repeat=2, cuda=False) * 1e3,
                "score": float(np_call())}

    iters = 10 if cuda else 3
    if n * h * w > 8_000_000:
        iters = max(2, iters // 3)
    return {"ms": _timeit(fn, iters=iters, repeat=3, cuda=cuda) * 1e3,
            "score": float(fn())}


# --------------------------------------------------------------------------- #
# the parent process: fan the cases out, print the tables
# --------------------------------------------------------------------------- #


def child(kind, device, n, h, w, dtype="fp32"):
    """Run one case in a fresh interpreter; None if it could not run."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--case", kind,
           "--device", device, "--n", str(n), "--h", str(h), "--w", str(w),
           "--dtype", dtype]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return None


def _row(label, cells, width=14):
    print(f"  {label:<22}" + "".join(f"{c:>{width}}" for c in cells))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.case:                      # child mode
        print(json.dumps(run_case(args.case, args.device, args.n, args.h,
                                  args.w, args.dtype)))
        return 0

    cuda = torch.cuda.is_available()
    print("device:", torch.cuda.get_device_name(0) if cuda else "cpu only")
    print("backend:", fa.backend_status())
    print("\nEach cell is a fresh interpreter -- see the module docstring for why.")

    if cuda:
        print("\n== CUDA, ms/call (uint8 in, sRGB) ==")
        _row("size", ["compiled", "eager", "speedup", "Mpx/s"])
        for n, h, w in CUDA_SIZES:
            c = child("compiled", "cuda", n, h, w)
            e = child("eager", "cuda", n, h, w) if not args.quick else None
            if c is None:
                _row(f"{n}x3x{h}x{w}", ["oom/err", "", "", ""])
                continue
            mpx = n * h * w / c["ms"] / 1e3
            _row(f"{n}x3x{h}x{w}", [
                f"{c['ms']:.2f}",
                f"{e['ms']:.2f}" if e else "-",
                f"{e['ms'] / c['ms']:.2f}x" if e else "-",
                f"{mpx:.0f}",
            ])

    if not args.quick:
        print("\n== CPU, ms/call ==")
        _row("size", ["compiled", "eager", "speedup"])
        for n, h, w in CPU_SIZES:
            c = child("compiled", "cpu", n, h, w)
            e = child("eager", "cpu", n, h, w)
            if c is None:
                continue
            _row(f"{n}x3x{h}x{w}", [
                f"{c['ms']:.1f}",
                f"{e['ms']:.1f}" if e else "-",
                f"{e['ms'] / c['ms']:.2f}x" if e else "-",
            ])

        print("\n== float64 (the accuracy mode), ms/call ==")
        _row("size", ["cuda fp32", "cuda fp64", "cost"])
        for n, h, w in ((1, 1080, 1920),) if cuda else ():
            f32 = child("compiled", "cuda", n, h, w, "fp32")
            f64 = child("compiled", "cuda", n, h, w, "fp64")
            if f32 and f64:
                _row(f"{n}x3x{h}x{w}", [f"{f32['ms']:.2f}", f"{f64['ms']:.2f}",
                                        f"{f64['ms'] / f32['ms']:.1f}x"])

        if cuda:
            print("\n== peak MiB above the input pair, CUDA forward ==")
            _row("size", ["peak MiB", "input MiB"])
            for n, h, w in CUDA_SIZES:
                m = child("memory", "cuda", n, h, w)
                if m is None:
                    continue
                _row(f"{n}x3x{h}x{w}", [f"{m['peak_mib']:.1f}",
                                        f"{2 * n * 3 * h * w / MIB:.1f}"])

        print("\n== the only other pip-installable implementation ==")
        for n, h, w in ((1, 512, 512), (1, 1080, 1920)):
            np_res = child("numpy", "cpu", n, h, w)
            ours = child("compiled", "cpu", n, h, w)
            cu = child("compiled", "cuda", n, h, w) if cuda else None
            if np_res is None:
                print("  skip `pip install ssimulacra2`: not installed")
                break
            _row(f"{n}x3x{h}x{w}", [
                f"numpy {np_res['ms']:.0f} ms",
                f"ours(cpu) {ours['ms']:.0f} ms" if ours else "-",
                f"ours(cuda) {cu['ms']:.1f} ms" if cu else "-",
                f"{np_res['ms'] / ours['ms']:.0f}x" if ours else "-",
            ], width=20)
            print(f"    scores: numpy {np_res['score']:.3f} vs ours "
                  f"{ours['score']:.3f} -- it blurs with a mirrored-border "
                  "scipy Gaussian,\n    not libjxl's zero-padded recursive "
                  "filter, so this gap is the metric, not the arithmetic.")

    print("\nNotes")
    print("  * one resolution per process is the fast path: six scales are six")
    print("    compiled shapes, and Dynamo's recompile limit is 8, so a process")
    print("    that scores a second resolution finishes it in eager.")
    print("  * `eager` is what you get without a working Inductor backend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
