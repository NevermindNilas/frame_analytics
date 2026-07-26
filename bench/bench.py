"""Speed + accuracy benchmark against the libraries people actually use.

Every entry is timed the same way: warm up, then take the fastest of `repeat`
runs of `iters` calls each, synchronising the device around the whole block.
Accuracy is measured against the float64 reference in one pass so the table
shows what each implementation costs *and* what it gets wrong.

    python bench/bench.py                 # default sweep
    python bench/bench.py --quick
    python bench/bench.py --sizes 1080p,4k --batch 1,8
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
from bench import baselines as bl
from frame_analytics import reference as ref

SIZES = {
    "256": (256, 256),
    "512": (512, 512),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
    "4k": (2160, 3840),
}


# --------------------------------------------------------------------------- #


def timeit(fn, *, iters: int, repeat: int, cuda: bool) -> float:
    """Fastest seconds per call over ``repeat`` batches of ``iters``.

    Min, not median: the cheapest entries here are tens of microseconds, where a
    median is dominated by whatever else the machine did during the sample -- a
    scheduler preemption, a thread pool that had parked, a frequency step. Those
    add time and never subtract it, so the minimum is the estimator that
    converges on the thing being measured. It is also the difference between a
    reproducible table and one that moves 4x between runs.
    """
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


# --------------------------------------------------------------------------- #


def accuracy_table():
    print("\n" + "=" * 96)
    print("ACCURACY  -- absolute error vs float64 ssim_index.m on a 512x512 natural image")
    print("=" * 96)
    a, b = make_frames(512, 512, 1, 1)
    a2, b2 = a[0, 0].astype(np.float64), b[0, 0].astype(np.float64)
    gold = ref.ssim_reference(a2, b2, 255.0)
    gold_psnr = ref.psnr_reference(a2, b2, 255.0)
    print(f"  reference SSIM = {gold:.12f}   reference PSNR = {gold_psnr:.9f} dB\n")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ta = torch.from_numpy(a).to(dev)
    tb = torch.from_numpy(b).to(dev)
    taf = ta.float()
    tbf = tb.float()

    rows = []

    def add(name, val):
        rows.append((name, float(val), abs(float(val) - gold)))

    add("frame_analytics (native)", fa.ssim(ta, tb, data_range=255.0))
    add("frame_analytics (torch path)", fa.ssim(ta, tb, data_range=255.0, backend_hint="torch"))
    add("frame_analytics (cpu native)", fa.ssim(torch.from_numpy(a), torch.from_numpy(b), data_range=255.0))

    try:
        from skimage.metrics import structural_similarity
        add("scikit-image (paper settings)",
            structural_similarity(a2, b2, data_range=255.0, gaussian_weights=True,
                                  sigma=1.5, use_sample_covariance=False))
        add("scikit-image (library defaults)",
            structural_similarity(a2, b2, data_range=255.0))
    except Exception as exc:
        print("  skimage unavailable:", exc)

    try:
        add("OpenCV tutorial recipe", bl.opencv_ssim(a[0, 0], b[0, 0]))
    except Exception as exc:
        print("  opencv unavailable:", exc)

    try:
        from pytorch_msssim import ssim as pm_ssim
        add("pytorch-msssim", pm_ssim(taf, tbf, data_range=255.0))
    except Exception as exc:
        print("  pytorch_msssim unavailable:", exc)

    try:
        import piq
        add("piq (defaults)", piq.ssim(taf / 255.0, tbf / 255.0, data_range=1.0))
        add("piq (downsample=False)",
            piq.ssim(taf / 255.0, tbf / 255.0, data_range=1.0, downsample=False))
    except Exception as exc:
        print("  piq unavailable:", exc)

    try:
        from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
        add("torchmetrics", tm_ssim(taf, tbf, data_range=255.0))
    except Exception as exc:
        print("  torchmetrics unavailable:", exc)

    try:
        import kornia
        add("kornia", kornia.metrics.ssim(taf, tbf, 11, max_val=255.0).mean())
    except Exception as exc:
        print("  kornia unavailable:", exc)

    print(f"  {'implementation':<34} {'SSIM':>16} {'|err|':>12}")
    print("  " + "-" * 64)
    for name, val, err in rows:
        flag = "" if err < 5e-6 else ("  <-- off-spec" if err > 1e-3 else "  <-- loose")
        print(f"  {name:<34} {val:>16.12f} {err:>12.2e}{flag}")

    # MSE / PSNR
    print()
    prows = []
    prows.append(("frame_analytics", float(fa.psnr(ta, tb, data_range=255.0))))
    prows.append(("numpy float64", ref.psnr_reference(a2, b2, 255.0)))
    try:
        import cv2
        prows.append(("cv2.PSNR", cv2.PSNR(a[0, 0], b[0, 0])))
    except Exception:
        pass
    try:
        from skimage.metrics import peak_signal_noise_ratio
        prows.append(("scikit-image", peak_signal_noise_ratio(a2, b2, data_range=255.0)))
    except Exception:
        pass
    prows.append(("naive torch fp32", float(10 * torch.log10(
        torch.tensor(255.0 ** 2) / bl.naive_mse_torch(ta, tb).cpu()))))
    print(f"  {'PSNR implementation':<34} {'dB':>16} {'|err|':>12}")
    print("  " + "-" * 64)
    for name, val in prows:
        print(f"  {name:<34} {val:>16.9f} {abs(val - gold_psnr):>12.2e}")


# --------------------------------------------------------------------------- #


def bench_device(dev, sizes, batches, iters, repeat):
    cuda = dev == "cuda"
    for size_key in sizes:
        h, w = SIZES[size_key]
        for n in batches:
            a, b = make_frames(h, w, n, 1)
            ta = torch.from_numpy(a).to(dev)
            tb = torch.from_numpy(b).to(dev)
            taf, tbf = ta.float(), tb.float()
            mpix = n * h * w / 1e6

            print(f"\n--- {dev.upper()}  {size_key} ({h}x{w})  batch={n}  "
                  f"[{mpix:.2f} Mpix/call] ---")
            entries = []

            def add(name, fn, note=""):
                try:
                    t = timeit(fn, iters=iters, repeat=repeat, cuda=cuda)
                    entries.append((name, t, note))
                except Exception as exc:
                    entries.append((name, float("nan"), f"{type(exc).__name__}"))

            # ---- SSIM ----
            add("SSIM  frame_analytics native (uint8)",
                lambda: fa.ssim(ta, tb, data_range=255.0))
            add("SSIM  frame_analytics native (fp32)",
                lambda: fa.ssim(taf, tbf, data_range=255.0))
            add("SSIM  frame_analytics torch-path (uint8)",
                lambda: fa.ssim(ta, tb, data_range=255.0, backend_hint="torch"))
            add("SSIM  naive 5x2D-conv torch (fp32)",
                lambda: bl.naive_ssim_torch(taf, tbf))
            try:
                from pytorch_msssim import ssim as pm_ssim
                add("SSIM  pytorch-msssim (fp32)",
                    lambda: pm_ssim(taf, tbf, data_range=255.0))
            except Exception:
                pass
            try:
                import piq
                xa, xb = taf / 255.0, tbf / 255.0
                add("SSIM  piq (fp32, downsample=False)",
                    lambda: piq.ssim(xa, xb, data_range=1.0, downsample=False))
            except Exception:
                pass
            try:
                from torchmetrics.functional.image import structural_similarity_index_measure as tm
                add("SSIM  torchmetrics (fp32)", lambda: tm(taf, tbf, data_range=255.0))
            except Exception:
                pass
            try:
                import kornia
                add("SSIM  kornia (fp32)",
                    lambda: kornia.metrics.ssim(taf, tbf, 11, max_val=255.0).mean())
            except Exception:
                pass
            if not cuda:
                try:
                    from skimage.metrics import structural_similarity
                    a0 = a[0, 0].astype(np.float64)
                    b0 = b[0, 0].astype(np.float64)
                    add("SSIM  scikit-image (1 image, fp64)",
                        lambda: structural_similarity(a0, b0, data_range=255.0,
                                                      gaussian_weights=True, sigma=1.5,
                                                      use_sample_covariance=False),
                        note=f"x{n} images" if n > 1 else "")
                except Exception:
                    pass
                try:
                    a0 = a[0, 0]
                    b0 = b[0, 0]
                    add("SSIM  OpenCV recipe (1 image, fp32)",
                        lambda: bl.opencv_ssim(a0, b0),
                        note=f"x{n} images" if n > 1 else "")
                except Exception:
                    pass

            # ---- MSE / PSNR ----
            add("PSNR  frame_analytics native (uint8)",
                lambda: fa.psnr(ta, tb, data_range=255.0))
            add("PSNR  frame_analytics torch-path (uint8)",
                lambda: fa.psnr(ta, tb, data_range=255.0, dtype=torch.float32) if False
                else _psnr_torch_path(ta, tb))
            add("MSE   naive torch (fp32 tensors)",
                lambda: bl.naive_mse_torch(taf, tbf))
            add("MSE   naive torch (uint8 in)",
                lambda: bl.naive_mse_torch(ta, tb))
            if not cuda:
                a0, b0 = a[0, 0], b[0, 0]
                add("MSE   numpy float64 (1 image)", lambda: bl.numpy_mse(a0, b0),
                    note=f"x{n} images" if n > 1 else "")
                try:
                    import cv2
                    add("PSNR  cv2.PSNR (1 image)", lambda: cv2.PSNR(a0, b0),
                        note=f"x{n} images" if n > 1 else "")
                except Exception:
                    pass

            base = next((t for nm, t, _ in entries if nm.startswith("SSIM  frame_analytics native (uint8)")), None)
            print(f"  {'implementation':<44} {'ms/call':>10} {'Mpix/s':>11} {'speedup':>9}")
            print("  " + "-" * 78)
            for name, t, note in entries:
                if t != t:  # nan
                    print(f"  {name:<44} {'--':>10} {'--':>11} {'--':>9}  {note}")
                    continue
                sp = (base / t) if (base and t) else float("nan")
                rel = f"{1/sp:>8.2f}x" if sp == sp and sp else "     --"
                print(f"  {name:<44} {t*1e3:>10.3f} {mpix/t:>11.1f} {rel:>9}  {note}")


def _psnr_torch_path(ta, tb):
    from frame_analytics.functional import _mse_kernel_mean, _maybe_compile
    fn = _maybe_compile(_mse_kernel_mean, "mse:False")
    m = fn(ta, tb, torch.float32)
    return 10.0 * torch.log10(torch.as_tensor(255.0 ** 2, device=m.device, dtype=m.dtype) / m)


# --------------------------------------------------------------------------- #


def bench_streaming(size_key="1080p", frames=300):
    if not torch.cuda.is_available():
        return
    h, w = SIZES[size_key]
    print("\n" + "=" * 96)
    print(f"STREAMING  -- {frames} frames of {size_key}, host uint8 in, python float out")
    print("=" * 96)
    a, b = make_frames(h, w, 4, 3)
    host = [(torch.from_numpy(a[i % 4]), torch.from_numpy(b[i % 4])) for i in range(8)]

    sm = fa.StreamingMetrics((1, 3, h, w), device="cuda", dtype=torch.uint8, data_range=255.0)
    print(f"  cuda graph captured: {sm.graph_captured}")
    for i in range(16):
        sm.update(*host[i % 8])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(frames):
        sm.update(*host[i % 8])
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  StreamingMetrics : {dt/frames*1e3:8.3f} ms/frame   {frames/dt:8.1f} fps "
          f"(mse+psnr+ssim, 3 channels)")

    ref_a = torch.from_numpy(a[0]).cuda()[None]
    ref_b = torch.from_numpy(b[0]).cuda()[None]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(frames):
        fa.ssim(ref_a, ref_b, data_range=255.0)
        fa.psnr(ref_a, ref_b, data_range=255.0)
    torch.cuda.synchronize()
    dt2 = time.perf_counter() - t0
    print(f"  functional calls : {dt2/frames*1e3:8.3f} ms/frame   {frames/dt2:8.1f} fps "
          f"(already-resident tensors, no H2D)")


# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,1080p,4k")
    ap.add_argument("--batch", default="1,8")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument("--devices", default="cuda,cpu")
    ap.add_argument("--no-streaming", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.sizes, args.batch, args.iters, args.repeat = "512,1080p", "1", 10, 3

    print("device :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only")
    print("torch  :", torch.__version__)
    print("threads:", torch.get_num_threads())
    print("backend:", fa.backend_status())

    if not args.no_accuracy:
        accuracy_table()

    sizes = [s for s in args.sizes.split(",") if s in SIZES]
    batches = [int(b) for b in args.batch.split(",")]

    print("\n" + "=" * 96)
    print("THROUGHPUT")
    print("=" * 96)
    want = [d for d in args.devices.split(",") if d]
    for dev in want:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        bench_device(dev, sizes, batches, args.iters, args.repeat)

    if not args.no_streaming:
        bench_streaming()


if __name__ == "__main__":
    main()
