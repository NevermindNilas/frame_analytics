"""Speed and memory benchmark for the training-oriented metrics.

MS-SSIM, GMSD and the pixel losses, against the libraries people actually
train with. Forward *and* forward+backward are timed separately, because for a
loss the backward pass is half the bill and the usual implementations pay for
it in saved activations.

    python bench/bench_training.py
    python bench/bench_training.py --sizes 512,1080p --batch 1,8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_analytics as fa

SIZES = {
    "256": (256, 256),
    "512": (512, 512),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
    "4k": (2160, 3840),
}


def timeit(fn, *, iters: int, repeat: int, cuda: bool) -> float:
    """Fastest seconds per call; see the note on ``bench.bench.timeit``."""
    for _ in range(max(3, iters)):
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


def peak_mem(fn, setup=None) -> float:
    """Peak CUDA allocation *attributable to fn*, in MiB.

    ``setup`` runs before the baseline is taken, and for a training step it
    releases ``.grad``. Left live from the warm-up, the gradient buffer sits in
    the baseline; the step then frees it and reallocates the same bytes, which
    nets to zero and reports the gradient as free. Every implementation pays
    for it, so it belongs in every row.
    """
    if setup is not None:
        setup()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 2 ** 20


def make_frames(h, w, n, c, seed=7):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 110 + 60 * np.sin(xx / 37.0) * np.cos(yy / 53.0) + 40 * np.sin((xx + yy) / 17.0)
    a = np.empty((n, c, h, w), np.uint8)
    b = np.empty((n, c, h, w), np.uint8)
    for i in range(n):
        for j in range(c):
            f = np.clip(base + 25 * rng.standard_normal((h, w)), 0, 255)
            a[i, j] = f.round()
            b[i, j] = np.clip(f + 6 * rng.standard_normal((h, w)), 0, 255).round()
    return a, b


def _row(name, cells, width=11):
    print(f"  {name:<26}" + "".join(f"{c:>{width}}" for c in cells))


def _fmt(v):
    if v is None:
        return "-"
    return f"{v * 1e3:.3f}"


# --------------------------------------------------------------------------- #


def ms_ssim_table(dev, sizes, batches, iters, repeat):
    print("\n" + "=" * 96)
    print(f"MS-SSIM, {dev}, uint8 in -- ms/call (forward, no grad)")
    print("=" * 96)

    try:
        import pytorch_msssim as pm
    except ImportError:
        pm = None
    try:
        import piq
    except ImportError:
        piq = None

    cols = [f"{s}x{n}" for s in sizes for n in batches]
    _row("", cols)

    rows = {"frame_analytics": [], "frame_analytics (torch path)": []}
    if pm:
        rows["pytorch-msssim"] = []
    if piq:
        rows["piq"] = []

    for s in sizes:
        h, w = SIZES[s]
        for n in batches:
            a, b = make_frames(h, w, n, 3)
            x = torch.from_numpy(a).to(dev)
            y = torch.from_numpy(b).to(dev)
            xf, yf = x.float(), y.float()
            cuda = dev == "cuda"
            with torch.no_grad():
                rows["frame_analytics"].append(
                    timeit(lambda: fa.ms_ssim(x, y, data_range=255.0),
                           iters=iters, repeat=repeat, cuda=cuda))
                rows["frame_analytics (torch path)"].append(
                    timeit(lambda: fa.ms_ssim(x, y, data_range=255.0,
                                              backend_hint="torch"),
                           iters=iters, repeat=repeat, cuda=cuda))
                if pm:
                    rows["pytorch-msssim"].append(
                        timeit(lambda: pm.ms_ssim(xf, yf, data_range=255.0),
                               iters=iters, repeat=repeat, cuda=cuda))
                if piq:
                    rows["piq"].append(
                        timeit(lambda: piq.multi_scale_ssim(xf, yf, data_range=255.0),
                               iters=iters, repeat=repeat, cuda=cuda))
            del x, y, xf, yf
            if cuda:
                torch.cuda.empty_cache()

    for k, v in rows.items():
        _row(k, [_fmt(t) for t in v])


def ms_ssim_train_table(dev, sizes, batches, iters, repeat):
    print("\n" + "=" * 96)
    print(f"MS-SSIM as a loss, {dev}, float32 -- ms/step (forward + backward)")
    print("=" * 96)
    try:
        import pytorch_msssim as pm
    except ImportError:
        pm = None

    cols = [f"{s}x{n}" for s in sizes for n in batches]
    _row("", cols)
    rows = {"frame_analytics": [], "frame_analytics (torch path)": []}
    mem = {"frame_analytics": [], "frame_analytics (torch path)": []}
    if pm:
        rows["pytorch-msssim"] = []
        mem["pytorch-msssim"] = []

    def step(fn, x):
        def run():
            if x.grad is not None:
                x.grad = None
            (1.0 - fn()).backward()
        return run

    for s in sizes:
        h, w = SIZES[s]
        for n in batches:
            a, b = make_frames(h, w, n, 3)
            x = (torch.from_numpy(a).to(dev).float() / 255.0).requires_grad_(True)
            y = torch.from_numpy(b).to(dev).float() / 255.0
            cuda = dev == "cuda"
            cases = {
                "frame_analytics": lambda: fa.ms_ssim(x, y, data_range=1.0),
                "frame_analytics (torch path)":
                    lambda: fa.ms_ssim(x, y, data_range=1.0, backend_hint="torch"),
            }
            if pm:
                cases["pytorch-msssim"] = lambda: pm.ms_ssim(x, y, data_range=1.0)
            def drop_grad():
                x.grad = None

            for k, fn in cases.items():
                run = step(fn, x)
                rows[k].append(timeit(run, iters=iters, repeat=repeat, cuda=cuda))
                if cuda:
                    mem[k].append(peak_mem(run, setup=drop_grad))
            del x, y
            if cuda:
                torch.cuda.empty_cache()

    for k, v in rows.items():
        _row(k, [_fmt(t) for t in v])
    if dev == "cuda":
        print("\n  peak CUDA memory per step, MiB (lower is better)")
        for k, v in mem.items():
            _row(k, [f"{m:.1f}" for m in v])


def gmsd_table(dev, sizes, batches, iters, repeat):
    print("\n" + "=" * 96)
    print(f"GMSD, {dev}, uint8 in -- ms/call")
    print("=" * 96)
    try:
        import piq
    except ImportError:
        piq = None

    cols = [f"{s}x{n}" for s in sizes for n in batches]
    _row("", cols)
    rows = {"frame_analytics": [], "frame_analytics (torch path)": []}
    if piq:
        rows["piq"] = []

    for s in sizes:
        h, w = SIZES[s]
        for n in batches:
            a, b = make_frames(h, w, n, 3)
            x = torch.from_numpy(a).to(dev)
            y = torch.from_numpy(b).to(dev)
            xf, yf = x.float() / 255.0, y.float() / 255.0
            cuda = dev == "cuda"
            with torch.no_grad():
                rows["frame_analytics"].append(
                    timeit(lambda: fa.gmsd(x, y, data_range=255.0),
                           iters=iters, repeat=repeat, cuda=cuda))
                rows["frame_analytics (torch path)"].append(
                    timeit(lambda: fa.gmsd(x, y, data_range=255.0,
                                           backend_hint="torch"),
                           iters=iters, repeat=repeat, cuda=cuda))
                if piq:
                    rows["piq"].append(
                        timeit(lambda: piq.gmsd(xf, yf, data_range=1.0),
                               iters=iters, repeat=repeat, cuda=cuda))
            del x, y, xf, yf
            if cuda:
                torch.cuda.empty_cache()

    for k, v in rows.items():
        _row(k, [_fmt(t) for t in v])


def pixel_table(dev, sizes, batches, iters, repeat):
    print("\n" + "=" * 96)
    print(f"Pixel losses, {dev}, uint8 in -- ms/call")
    print("=" * 96)
    cols = [f"{s}x{n}" for s in sizes for n in batches]
    _row("", cols)
    rows = {"fa.l1": [], "torch l1_loss": [], "fa.charbonnier": [],
            "torch charbonnier": [], "fa.huber": [], "torch huber_loss": []}

    for s in sizes:
        h, w = SIZES[s]
        for n in batches:
            a, b = make_frames(h, w, n, 3)
            x = torch.from_numpy(a).to(dev)
            y = torch.from_numpy(b).to(dev)
            cuda = dev == "cuda"
            eps2 = 1e-6

            def t_l1():
                return torch.nn.functional.l1_loss(x.float(), y.float())

            def t_charb():
                d = x.float() - y.float()
                return torch.sqrt(d * d + eps2).mean()

            def t_huber():
                return torch.nn.functional.huber_loss(x.float(), y.float())

            with torch.no_grad():
                for k, fn in [("fa.l1", lambda: fa.l1(x, y)),
                              ("torch l1_loss", t_l1),
                              ("fa.charbonnier", lambda: fa.charbonnier(x, y)),
                              ("torch charbonnier", t_charb),
                              ("fa.huber", lambda: fa.huber(x, y)),
                              ("torch huber_loss", t_huber)]:
                    rows[k].append(timeit(fn, iters=iters, repeat=repeat, cuda=cuda))
            del x, y
            if cuda:
                torch.cuda.empty_cache()

    for k, v in rows.items():
        _row(k, [_fmt(t) for t in v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,1080p")
    ap.add_argument("--batch", default="1,8")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    batches = [int(b) for b in args.batch.split(",")]
    print("backend:", fa.backend_status())

    devices = [] if args.cpu_only else (["cuda"] if torch.cuda.is_available() else [])
    devices.append("cpu")
    for dev in devices:
        it = args.iters if dev == "cuda" else max(3, args.iters // 5)
        rp = args.repeat if dev == "cuda" else 3
        ms_ssim_table(dev, sizes, batches, it, rp)
        gmsd_table(dev, sizes, batches, it, rp)
        pixel_table(dev, sizes, batches, it, rp)
        if dev == "cuda":
            ms_ssim_train_table(dev, sizes, batches, it, rp)


if __name__ == "__main__":
    main()
