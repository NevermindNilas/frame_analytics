"""Head-to-head against rahul-goel/fused-ssim.

fused-ssim is the fastest published CUDA SSIM we are aware of -- a fully fused
forward+backward kernel out of the 3D Gaussian Splatting world.

Comparison is set up to be fair to it:
  * float32 input in [0,1] and C1=0.01^2, C2=0.03^2, which is what its API
    hardcodes (i.e. data_range=1.0)
  * padding="valid", so both implementations evaluate the same H-10 x W-10
    support as the paper
  * train=False, so it skips writing the three derivative maps it needs for
    backward. train=True is also timed, since that is its default.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_analytics as fa
from frame_analytics import reference as ref

from fused_ssim import fused_ssim

SIZES = [("512", 512, 512), ("1080p", 1080, 1920), ("4k", 2160, 3840)]


def timeit(fn, iters=30, repeat=5):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) / iters)
    return statistics.median(samples)


def make_pair(h, w, seed=11):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.clip(110 + 60 * np.sin(xx / 37.0) * np.cos(yy / 53.0)
                + 40 * np.sin((xx + yy) / 17.0), 0, 255).round()
    b = np.clip(a + 6 * rng.standard_normal((h, w)), 0, 255).round()
    return a, b


def accuracy():
    print("=" * 88)
    print("ACCURACY vs float64 ssim_index.m  (float32 input in [0,1], data_range=1.0)")
    print("=" * 88)
    for name, h, w in [("256", 256, 256), ("512", 512, 512), ("1080p", 1080, 1920)]:
        a, b = make_pair(h, w)
        a01, b01 = a / 255.0, b / 255.0
        gold = ref.ssim_reference(a01, b01, data_range=1.0)

        x = torch.as_tensor(a01, dtype=torch.float32, device="cuda")[None, None]
        y = torch.as_tensor(b01, dtype=torch.float32, device="cuda")[None, None]

        ours = float(fa.ssim(x, y, data_range=1.0))
        fs_valid = float(fused_ssim(x, y, padding="valid", train=False))
        fs_same = float(fused_ssim(x, y, padding="same", train=False))

        print(f"\n  {name} ({h}x{w})   reference = {gold:.12f}")
        for label, v in [("frame_analytics", ours),
                         ("fused-ssim padding=valid", fs_valid),
                         ("fused-ssim padding=same (its default)", fs_same)]:
            print(f"    {label:<38} {v:.12f}  |err|={abs(v-gold):.2e}")


def speed():
    print("\n" + "=" * 88)
    print("THROUGHPUT  (ms/call, float32 in [0,1], 1 channel)")
    print("=" * 88)
    # touch every code path once before the first timed row, so the first entry
    # is not paying for lazy CUDA-module load / compile
    wx = torch.rand(1, 1, 256, 256, device="cuda")
    wy = torch.rand(1, 1, 256, 256, device="cuda")
    for _ in range(5):
        fa.ssim(wx, wy, data_range=1.0)
        fused_ssim(wx, wy, padding="valid", train=False)
        fused_ssim(wx, wy, padding="valid", train=True)
    torch.cuda.synchronize()

    print(f"  {'size':<10} {'batch':>5} {'frame_analytics':>17} {'fused-ssim':>12} "
          f"{'fs train=True':>14} {'speedup':>9}")
    print("  " + "-" * 76)
    for name, h, w in SIZES:
        for n in (1, 8):
            a, b = make_pair(h, w)
            x = torch.as_tensor(a / 255.0, dtype=torch.float32, device="cuda")
            y = torch.as_tensor(b / 255.0, dtype=torch.float32, device="cuda")
            x = x[None, None].expand(n, 1, h, w).contiguous()
            y = y[None, None].expand(n, 1, h, w).contiguous()

            t_ours = timeit(lambda: fa.ssim(x, y, data_range=1.0))
            t_fs = timeit(lambda: fused_ssim(x, y, padding="valid", train=False))
            t_fst = timeit(lambda: fused_ssim(x, y, padding="valid", train=True))
            print(f"  {name:<10} {n:>5} {t_ours*1e3:>17.3f} {t_fs*1e3:>12.3f} "
                  f"{t_fst*1e3:>14.3f} {t_fs/t_ours:>8.2f}x")

    # uint8 is our real advantage for video: half the bandwidth, and fused-ssim
    # has no uint8 path at all (its API takes float tensors only).
    print("\n  uint8 input (fused-ssim requires float32, so this is ours only):")
    for name, h, w in SIZES:
        a, b = make_pair(h, w)
        xu = torch.as_tensor(a, dtype=torch.uint8, device="cuda")[None, None].contiguous()
        yu = torch.as_tensor(b, dtype=torch.uint8, device="cuda")[None, None].contiguous()
        t = timeit(lambda: fa.ssim(xu, yu, data_range=255.0))
        print(f"    {name:<10} {t*1e3:8.3f} ms")


def gradients():
    print("\n" + "=" * 88)
    print("BACKWARD  (ms/call, forward+backward, 1080p x1)")
    print("=" * 88)
    h, w = 1080, 1920
    a, b = make_pair(h, w)
    y = torch.as_tensor(b / 255.0, dtype=torch.float32, device="cuda")[None, None]
    # one persistent leaf, grad zeroed per iteration: allocating a fresh
    # requires_grad tensor inside the timed loop measures the allocator, not
    # the kernels, and it made this row swing by 40% between runs
    x = torch.as_tensor(a / 255.0, dtype=torch.float32,
                        device="cuda")[None, None].requires_grad_(True)

    def run_fs():
        x.grad = None
        (1.0 - fused_ssim(x, y, padding="valid", train=True)).backward()

    def run_ours():
        x.grad = None
        (1.0 - fa.ssim(x, y, data_range=1.0)).backward()

    print(f"  fused-ssim      {timeit(run_fs, iters=30, repeat=7)*1e3:8.3f} ms")
    print(f"  frame_analytics {timeit(run_ours, iters=30, repeat=7)*1e3:8.3f} ms")
    print("  (fused-ssim has a hand-written backward kernel; ours falls back to "
          "autograd\n   over the portable path, since the native kernels are "
          "forward-only)")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0))
    print("backend:", fa.backend_status())
    accuracy()
    speed()
    gradients()
