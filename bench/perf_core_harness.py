"""A/B timing harness for the core metrics (perf-core pass, 2026-11-14).

Imports whichever ``frame_analytics`` is first on ``PYTHONPATH``, so the same
script times a saved baseline package and the working tree in separate
processes. Prints one JSON object per case.

    PYTHONPATH=<pkg root> python bench/perf_core_harness.py --device cuda \
        --metrics ssim,psnr --shapes 512,1080p --dtypes u8,f32 --modes fwd

Timing is min over ``--repeat`` blocks of N calls, device-synchronised around
each block (N picked so a block is ~``--block-ms``). Memory is the CUDA peak
above the inputs (and, for fwd+bwd, includes the gradient).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

import frame_analytics as fa

SHAPES = {
    "256": (1, 3, 256, 256),
    "512": (1, 3, 512, 512),
    "1080p": (1, 3, 1080, 1920),
    "1080px8": (8, 3, 1080, 1920),
    "4kx8": (8, 3, 2160, 3840),
    "4k": (1, 3, 2160, 3840),
}

FWD = {
    "mse": lambda a, b, L: fa.mse(a, b),
    "psnr": lambda a, b, L: fa.psnr(a, b, data_range=L),
    "ssim": lambda a, b, L: fa.ssim(a, b, data_range=L),
    "ssim_none": lambda a, b, L: fa.ssim(a, b, data_range=L, reduction="none"),
    "ssim_map": lambda a, b, L: fa.ssim(a, b, data_range=L, return_map=True),
    "ms_ssim": lambda a, b, L: fa.ms_ssim(a, b, data_range=L),
    "gmsd": lambda a, b, L: fa.gmsd(a, b, data_range=L),
    "l1": lambda a, b, L: fa.l1(a, b),
    "charbonnier": lambda a, b, L: fa.charbonnier(a, b),
    "huber": lambda a, b, L: fa.huber(a, b),
    "lpips": lambda a, b, L: fa.lpips(a, b, data_range=L),
    "ssimulacra2": lambda a, b, L: fa.ssimulacra2(a, b, data_range=L),
    "psnr_luma": lambda a, b, L: fa.psnr(a, b, data_range=L, luma="matlab", crop_border=4),
    "ssim_luma": lambda a, b, L: fa.ssim(a, b, data_range=L, luma="matlab", crop_border=4),
}

# loss forms: scalar to differentiate, prediction is the first argument
BWD = {
    "ssim": lambda a, b, L: 1 - fa.ssim(a, b, data_range=L),
    "ms_ssim": lambda a, b, L: 1 - fa.ms_ssim(a, b, data_range=L),
    "gms": lambda a, b, L: 1 - fa.gms(a, b, data_range=L),
    "mse": lambda a, b, L: fa.mse(a, b),
    "l1": lambda a, b, L: fa.l1(a, b),
    "charbonnier": lambda a, b, L: fa.charbonnier(a, b),
    "huber": lambda a, b, L: fa.huber(a, b),
    "lpips": lambda a, b, L: fa.lpips(a, b, data_range=L),
}


def make(shape, dtype, device, seed=7):
    g = torch.Generator().manual_seed(seed)
    n, c, h, w = shape
    yy = torch.arange(h, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, w)
    base = 110 + 60 * torch.sin(xx / 37.0) * torch.cos(yy / 53.0) + 40 * torch.sin((xx + yy) / 17.0)
    a = (base + 25 * torch.randn(n, c, h, w, generator=g)).clamp(0, 255)
    b = (a + 6 * torch.randn(n, c, h, w, generator=g)).clamp(0, 255)
    a, b = a.round(), b.round()
    if dtype == "u8":
        return a.to(torch.uint8).to(device), b.to(torch.uint8).to(device), 255.0
    dt = {"f32": torch.float32, "f16": torch.float16, "f64": torch.float64}[dtype]
    return (a / 255).to(dt).to(device), (b / 255).to(dt).to(device), 1.0


def sync(cuda):
    if cuda:
        torch.cuda.synchronize()


_spin_buf = None


def spin(cuda, ms):
    """Keep the GPU busy for ``ms`` so it is at boost clock when timing starts.

    An idle 3090 sits at ~210 MHz and takes a while to ramp; without this the
    first blocks of a case measure the ramp, not the kernel.
    """
    global _spin_buf
    if not cuda or ms <= 0:
        return
    if _spin_buf is None:
        _spin_buf = torch.randn(2048, 2048, device="cuda")
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < ms:
        for _ in range(4):
            _spin_buf @ _spin_buf
        torch.cuda.synchronize()


def timeit(fn, cuda, repeat, block_ms, warm=3, spin_ms=0.0):
    spin(cuda, spin_ms)
    for _ in range(warm):
        fn()
    sync(cuda)
    t0 = time.perf_counter()
    fn()
    sync(cuda)
    one = max(time.perf_counter() - t0, 1e-6)
    iters = max(1, min(2000, int(block_ms / 1e3 / one)))
    best = float("inf")
    for _ in range(repeat):
        sync(cuda)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync(cuda)
        best = min(best, (time.perf_counter() - t0) / iters)
    return best, iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metrics", default="mse,psnr,ssim,ms_ssim,gmsd,l1,charbonnier,huber")
    ap.add_argument("--shapes", default="512,1080p,1080px8,4kx8")
    ap.add_argument("--dtypes", default="u8,f32")
    ap.add_argument("--modes", default="fwd")
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--block-ms", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--spin-ms", type=float, default=300.0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cuda = args.device.startswith("cuda")

    for shape_key in args.shapes.split(","):
        for dt in args.dtypes.split(","):
            a, b, L = make(SHAPES[shape_key], dt, args.device)
            for mode in args.modes.split(","):
                for m in args.metrics.split(","):
                    table = FWD if mode == "fwd" else BWD
                    if m not in table:
                        continue
                    if mode == "bwd" and dt == "u8":
                        continue
                    rec = dict(tag=args.tag, device=args.device, shape=shape_key,
                               dtype=dt, mode=mode, metric=m)
                    try:
                        if mode == "fwd":
                            fn = lambda: table[m](a, b, L)
                            with torch.no_grad():
                                val = fn()
                                if cuda:
                                    torch.cuda.synchronize()
                                    torch.cuda.reset_peak_memory_stats()
                                    base = torch.cuda.memory_allocated()
                                    fn()
                                    torch.cuda.synchronize()
                                    rec["peak_mib"] = (torch.cuda.max_memory_allocated() - base) / 2**20
                                t, it = timeit(fn, cuda, args.repeat, args.block_ms, spin_ms=args.spin_ms)
                            rec["value"] = (val.double().mean().item() if val.numel() > 1
                                            else float(val))
                        else:
                            x = a.clone().requires_grad_(True)

                            def fn():
                                x.grad = None
                                loss = table[m](x, b, L)
                                loss.backward()
                                return loss
                            val = fn()
                            g0 = x.grad.double().abs().sum().item()
                            if cuda:
                                x.grad = None
                                torch.cuda.synchronize()
                                torch.cuda.reset_peak_memory_stats()
                                base = torch.cuda.memory_allocated()
                                fn()
                                torch.cuda.synchronize()
                                rec["peak_mib"] = (torch.cuda.max_memory_allocated() - base) / 2**20
                            t, it = timeit(fn, cuda, args.repeat, args.block_ms, spin_ms=args.spin_ms)
                            rec["value"] = float(val)
                            rec["grad_abs_sum"] = g0
                        rec["ms"] = t * 1e3
                        rec["iters"] = it
                    except Exception as exc:  # report, keep going
                        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    print(json.dumps(rec), flush=True)
            del a, b
            if cuda:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
