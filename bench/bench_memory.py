"""Memory benchmark: what each metric costs on top of the two input frames.

Speed is only half the bill. An evaluation metric that materialises a dozen
full-resolution intermediates decides how big a batch fits, and a loss that
saves its activations decides how big a *model* fits next to it.

Two instruments, because the two devices do not offer the same one:

* CUDA -- ``torch.cuda.max_memory_allocated`` around the call, minus what was
  already live. Exact, and attributable to the call alone.
* CPU -- the OS high-water mark, read once in a subprocess that does nothing
  else (see ``cpu_peak_isolated`` for why the alternatives lie). This counts
  pages the process had to obtain, so two temporaries that reuse the same
  freed pages count once: it answers "how much more memory does this need"
  rather than "how many bytes did it request".

The inputs themselves are excluded from every figure -- they are what you
already have, not what the metric costs you.

    python bench/bench_memory.py
    python bench/bench_memory.py --sizes 512,1080p,4k --batch 1,8
    python bench/bench_memory.py --devices cuda
    python bench/bench_memory.py --devices cpu --cpu-sizes 512,1080p
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any, Callable

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

MIB = 2 ** 20


# --------------------------------------------------------------------------- #
# measurement


def cuda_peak(fn, setup=None) -> float:
    """Peak CUDA allocation attributable to ``fn``, in MiB.

    Warm first: the first call through any path allocates workspaces, loads a
    module, and may compile -- none of which is the per-call cost of the
    metric, and all of which the caching allocator hands back to the second
    call for free.

    ``setup`` runs after the warm-up and *before* the baseline is taken. For a
    training step that means releasing ``.grad``, which the warm-up left live:
    measured against a baseline that already contains the gradient buffer, a
    backward pass that frees it and immediately reallocates the same bytes
    nets out to zero and the row reads as if the gradient were free.
    """
    fn()
    if setup is not None:
        setup()
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


def peak_rss_bytes() -> int:
    """The OS's own high-water mark for this process, monotonic since start."""
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss * 1024 if sys.platform != "darwin" else rss


def cpu_peak_isolated(fn, warm) -> float:
    """Peak CPU footprint of one call to ``fn``, in MiB.

    There is no CPU equivalent of ``max_memory_allocated``, and the two
    obvious substitutes both lie. Process RSS sampled around the call reports
    zero whenever the temporary is served out of a heap the process already
    grew. The profiler's own numbers are nested -- ``conv2d`` ->
    ``convolution`` -> ``_convolution`` -> ``mkldnn_convolution`` -> ``empty``
    each report the same buffer -- so summing them multiplies one allocation
    by its call depth, and summing the *self* figures instead drops every
    temporary that was freed inside the op that made it.

    What is left is the OS high-water mark, which is exact but monotonic: it
    can only be read once per process, because the first big call raises it
    for good. Hence ``--one``: the caller runs each candidate in its own
    interpreter, and this function is the last thing that happens in it.

    ``warm`` runs the same code path on a small input first, so that lazy
    work -- loading the extension, growing the allocator's arenas, MKLDNN's
    first-call setup -- is charged to the baseline instead of to the metric.
    """
    warm()
    gc.collect()
    base = peak_rss_bytes()
    out = fn()
    del out
    return max(0.0, (peak_rss_bytes() - base) / MIB)


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
# candidate sets


Cases = dict[str, Callable[[], Any]]


def ssim_cases(x8, y8, xf, yf) -> Cases:
    cases: Cases = {
        "frame_analytics (uint8 in)": lambda: fa.ssim(x8, y8, data_range=255.0),
        "frame_analytics (fp32 in)": lambda: fa.ssim(xf, yf, data_range=255.0),
        "frame_analytics (torch path)":
            lambda: fa.ssim(xf, yf, data_range=255.0, backend_hint="torch"),
    }
    try:
        from pytorch_msssim import ssim as pm_ssim
        cases["pytorch-msssim"] = lambda: pm_ssim(xf, yf, data_range=255.0)
    except ImportError:
        pass
    try:
        import piq
        cases["piq"] = lambda: piq.ssim(xf, yf, data_range=255.0, downsample=False)
    except ImportError:
        pass
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure as tm
        cases["torchmetrics"] = lambda: tm(xf, yf, data_range=255.0)
    except ImportError:
        pass
    try:
        import kornia
        cases["kornia"] = lambda: kornia.metrics.ssim(xf, yf, 11, max_val=255.0).mean()
    except ImportError:
        pass
    return cases


def ms_ssim_cases(x8, y8, xf, yf) -> Cases:
    cases: Cases = {
        "frame_analytics (uint8 in)": lambda: fa.ms_ssim(x8, y8, data_range=255.0),
        "frame_analytics (fp32 in)": lambda: fa.ms_ssim(xf, yf, data_range=255.0),
        "frame_analytics (torch path)":
            lambda: fa.ms_ssim(xf, yf, data_range=255.0, backend_hint="torch"),
    }
    try:
        import pytorch_msssim as pm
        cases["pytorch-msssim"] = lambda: pm.ms_ssim(xf, yf, data_range=255.0)
    except ImportError:
        pass
    try:
        import piq
        cases["piq"] = lambda: piq.multi_scale_ssim(xf, yf, data_range=255.0)
    except ImportError:
        pass
    return cases


def gmsd_cases(x8, y8, xf, yf) -> Cases:
    cases: Cases = {
        "frame_analytics (uint8 in)": lambda: fa.gmsd(x8, y8, data_range=255.0),
        "frame_analytics (fp32 in)": lambda: fa.gmsd(xf, yf, data_range=255.0),
        "frame_analytics (torch path)":
            lambda: fa.gmsd(xf, yf, data_range=255.0, backend_hint="torch"),
    }
    try:
        import piq
        cases["piq"] = lambda: piq.gmsd(xf / 255.0, yf / 255.0, data_range=1.0)
    except ImportError:
        pass
    return cases


def psnr_cases(x8, y8, xf, yf) -> Cases:
    cases: Cases = {
        "frame_analytics (uint8 in)": lambda: fa.psnr(x8, y8, data_range=255.0),
        "frame_analytics (fp32 in)": lambda: fa.psnr(xf, yf, data_range=255.0),
        "naive torch (fp32)":
            lambda: 10.0 * torch.log10(255.0 ** 2 / ((xf - yf) ** 2).mean()),
    }
    try:
        from torchmetrics.functional.image import peak_signal_noise_ratio as tm_psnr
        cases["torchmetrics"] = lambda: tm_psnr(xf, yf, data_range=255.0)
    except ImportError:
        pass
    try:
        import kornia
        cases["kornia"] = lambda: kornia.metrics.psnr(xf, yf, 255.0)
    except ImportError:
        pass
    return cases


def l1_cases(x8, y8, xf, yf) -> Cases:
    return {
        "frame_analytics (uint8 in)": lambda: fa.l1(x8, y8),
        "frame_analytics (fp32 in)": lambda: fa.l1(xf, yf),
        "torch F.l1_loss (fp32)": lambda: torch.nn.functional.l1_loss(xf, yf),
        "frame_analytics charbonnier": lambda: fa.charbonnier(x8, y8),
        "torch charbonnier (fp32)":
            lambda: torch.sqrt((xf - yf) ** 2 + 1e-6).mean(),
    }


METRICS = [
    ("SSIM", ssim_cases),
    ("MS-SSIM", ms_ssim_cases),
    ("GMSD", gmsd_cases),
    ("PSNR", psnr_cases),
    ("L1 / Charbonnier", l1_cases),
]


# --------------------------------------------------------------------------- #
# tables


def _cell(v, width=14):
    """A native kernel's whole footprint is the output scalar, so the column has
    to survive four orders of magnitude without printing it as 0.0."""
    if v is None or v != v:
        return f"{'--':>{width}}"
    if v >= 1.0:
        return f"{v:>{width}.1f}"
    return f"{v:>{width}.3f}"


def _print_table(title, rows, cols, unit_note, mpix):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"  {unit_note}")
    head = "".join(f"{c:>14}" for c in cols)
    print(f"\n  {'implementation':<32}{head}")
    print("  " + "-" * (32 + 14 * len(cols)))
    for name, vals in rows.items():
        cells = "".join(_cell(v) for v in vals)
        print(f"  {name:<32}{cells}")
    if mpix:
        print(f"\n  input frames excluded; both frames are "
              + ", ".join(f"{c} = {m:.1f} MiB uint8 / {4*m:.1f} MiB fp32"
                          for c, m in zip(cols, mpix)))


def forward_tables(dev, sizes, batches, channels):
    """CUDA only -- the CPU side goes through ``cpu_forward_tables``."""
    cols = [f"{s}x{n}" for s in sizes for n in batches]
    per_case = {name: {} for name, _ in METRICS}
    mpix = []

    for s in sizes:
        h, w = SIZES[s]
        for n in batches:
            a, b = make_frames(h, w, n, channels)
            x8 = torch.from_numpy(a).to(dev)
            y8 = torch.from_numpy(b).to(dev)
            xf, yf = x8.float(), y8.float()
            mpix.append(2 * n * channels * h * w / MIB)

            for metric, builder in METRICS:
                cases = builder(x8, y8, xf, yf)
                for name, fn in cases.items():
                    slot = per_case[metric].setdefault(name, [])
                    try:
                        with torch.no_grad():
                            slot.append(cuda_peak(fn))
                    except Exception as exc:
                        slot.append(float("nan"))
                        print(f"  ! {metric} {name} @ {s}x{n}: "
                              f"{type(exc).__name__}: {exc}")
                    torch.cuda.empty_cache()

            del x8, y8, xf, yf
            gc.collect()
            torch.cuda.empty_cache()

    unit = "peak CUDA allocation above the inputs, MiB per call (lower is better)"
    for metric, _ in METRICS:
        _print_table(f"{metric} -- forward, {dev}, {channels}-channel frames",
                     per_case[metric], cols, unit, mpix)


# --------------------------------------------------------------------------- #
# CPU: one subprocess per candidate, because the OS high-water mark is
# monotonic and can therefore only be read once per process.


# Big enough for MS-SSIM's five scales -- every implementation refuses a frame
# under ~176 px, and the warm-up has to take the same code path as the real call.
WARM_SIZE = (192, 192)


def case_names(metric, channels):
    """Which implementations exist for ``metric``, without measuring anything."""
    h, w = WARM_SIZE
    a, b = make_frames(h, w, 1, channels)
    x8, y8 = torch.from_numpy(a), torch.from_numpy(b)
    builder = dict(METRICS)[metric]
    return list(builder(x8, y8, x8.float(), y8.float()))


def run_one(spec: str) -> None:
    """``--one`` worker: measure a single candidate and print its peak."""
    metric, impl, size, batch, channels = spec.split("|")
    batch, channels = int(batch), int(channels)
    h, w = SIZES[size]
    builder = dict(METRICS)[metric]

    wa, wb = make_frames(*WARM_SIZE, 1, channels)
    wx, wy = torch.from_numpy(wa), torch.from_numpy(wb)
    warm = builder(wx, wy, wx.float(), wy.float())[impl]

    a, b = make_frames(h, w, batch, channels)
    x8, y8 = torch.from_numpy(a), torch.from_numpy(b)
    fn = builder(x8, y8, x8.float(), y8.float())[impl]

    with torch.no_grad():
        print(f"PEAK_MIB {cpu_peak_isolated(fn, warm):.4f}")


def cpu_forward_tables(sizes, batches, channels, only=()):
    import subprocess

    cols = [f"{s}x{n}" for s in sizes for n in batches]
    for metric, _ in METRICS:
        if only and metric not in only:
            continue
        rows = {}
        for name in case_names(metric, channels):
            vals = []
            for s in sizes:
                for n in batches:
                    spec = f"{metric}|{name}|{s}|{n}|{channels}"
                    proc = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--one", spec],
                        capture_output=True, text=True)
                    line = next((l for l in proc.stdout.splitlines()
                                 if l.startswith("PEAK_MIB")), None)
                    if line is None:
                        print(f"  ! {metric} {name} @ {s}x{n}: "
                              f"{proc.stderr.strip().splitlines()[-1:] or 'no output'}")
                        vals.append(float("nan"))
                    else:
                        vals.append(float(line.split()[1]))
            rows[name] = vals
        _print_table(f"{metric} -- forward, cpu, {channels}-channel frames",
                     rows, cols,
                     "peak process footprint above the inputs, MiB per call "
                     "(one process per row, see cpu_peak_isolated)", None)


def training_table(sizes, batches, channels):
    """Forward + backward, float32, the way a loss is actually used."""
    cols = [f"{s}x{n}" for s in sizes for n in batches]
    rows = {}

    def step(fn, x):
        def run():
            if x.grad is not None:
                x.grad = None
            (1.0 - fn()).backward()
        return run

    for s in sizes:
        h, w = SIZES[s]
        for n in batches:
            a, b = make_frames(h, w, n, channels)
            x = (torch.from_numpy(a).cuda().float() / 255.0).requires_grad_(True)
            y = torch.from_numpy(b).cuda().float() / 255.0

            cases: Cases = {
                "SSIM  frame_analytics": lambda: fa.ssim(x, y, data_range=1.0),
                "SSIM  frame_analytics (torch)":
                    lambda: fa.ssim(x, y, data_range=1.0, backend_hint="torch"),
                "MS-SSIM  frame_analytics": lambda: fa.ms_ssim(x, y, data_range=1.0),
                "MS-SSIM  frame_analytics (torch)":
                    lambda: fa.ms_ssim(x, y, data_range=1.0, backend_hint="torch"),
                "GMSD  frame_analytics": lambda: -fa.gmsd(x, y, data_range=1.0),
                "L1  frame_analytics": lambda: -fa.l1(x, y),
            }
            try:
                from pytorch_msssim import ssim as pm_ssim, ms_ssim as pm_ms
                cases["SSIM  pytorch-msssim"] = lambda: pm_ssim(x, y, data_range=1.0)
                cases["MS-SSIM  pytorch-msssim"] = lambda: pm_ms(x, y, data_range=1.0)
            except ImportError:
                pass
            try:
                import piq
                cases["SSIM  piq"] = lambda: piq.ssim(x, y, data_range=1.0,
                                                      downsample=False)
                cases["MS-SSIM  piq"] = lambda: piq.multi_scale_ssim(x, y, data_range=1.0)
                cases["GMSD  piq"] = lambda: -piq.gmsd(x, y, data_range=1.0)
            except ImportError:
                pass
            cases["L1  torch"] = lambda: -torch.nn.functional.l1_loss(x, y)

            def drop_grad():
                x.grad = None

            for name, fn in cases.items():
                slot = rows.setdefault(name, [])
                try:
                    slot.append(cuda_peak(step(fn, x), setup=drop_grad))
                except Exception as exc:
                    slot.append(float("nan"))
                    print(f"  ! {name} @ {s}x{n}: {type(exc).__name__}: {exc}")
                torch.cuda.empty_cache()

            del x, y
            gc.collect()
            torch.cuda.empty_cache()

    _print_table("Loss step -- forward + backward, cuda, float32 RGB",
                 rows, cols,
                 "peak CUDA allocation above the inputs, MiB per step "
                 "(includes the gradient w.r.t. the prediction, which every "
                 "implementation pays)",
                 None)


def streaming_note(size_key="1080p", channels=3):
    if not torch.cuda.is_available():
        return
    h, w = SIZES[size_key]
    a, b = make_frames(h, w, 1, channels)
    print("\n" + "=" * 100)
    print("StreamingMetrics -- steady-state footprint")
    print("=" * 100)
    gc.collect()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    sm = fa.StreamingMetrics((1, channels, h, w), device="cuda", dtype=torch.uint8,
                             data_range=255.0,
                             metrics=("psnr", "ssim", "ms_ssim", "gmsd"))
    ha, hb = torch.from_numpy(a), torch.from_numpy(b)
    for _ in range(8):
        sm.update(ha, hb)
    torch.cuda.synchronize()
    resident = (torch.cuda.memory_allocated() - before) / MIB

    torch.cuda.reset_peak_memory_stats()
    live = torch.cuda.memory_allocated()
    for _ in range(32):
        sm.update(ha, hb)
    torch.cuda.synchronize()
    per_frame = (torch.cuda.max_memory_allocated() - live) / MIB
    print(f"  graph captured        : {sm.graph_captured}")
    print(f"  resident (device frames + graph pool): {resident:.1f} MiB allocated, "
          f"{(torch.cuda.memory_reserved() - reserved_before) / MIB:.1f} MiB reserved")
    print(f"  additional per frame  : {per_frame:.1f} MiB "
          f"(4 metrics: psnr, ssim, ms_ssim, gmsd)")
    print("  note: only the first instance in a process pays for the graph's private "
          "pool;\n        a later one reuses it and measures ~6 MiB lower.")
    del sm, ha, hb
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,1080p,4k")
    ap.add_argument("--batch", default="1,8")
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--devices", default="cuda,cpu")
    ap.add_argument("--cpu-sizes", default="1080p",
                    help="sizes for the CPU tables, which cost one process per row")
    ap.add_argument("--metrics", default="",
                    help="comma-separated subset of " + ",".join(m for m, _ in METRICS))
    ap.add_argument("--no-training", action="store_true")
    ap.add_argument("--no-streaming", action="store_true")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.one:
        run_one(args.one)
        return

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip() in SIZES]
    batches = [int(b) for b in args.batch.split(",")]
    cpu_sizes = [s.strip() for s in args.cpu_sizes.split(",") if s.strip() in SIZES]
    wanted = tuple(m.strip() for m in args.metrics.split(",") if m.strip())

    print("device :", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "cpu only")
    print("torch  :", torch.__version__)
    print("backend:", fa.backend_status())

    for dev in [d for d in args.devices.split(",") if d]:
        if dev == "cuda":
            if torch.cuda.is_available():
                forward_tables(dev, sizes, batches, args.channels)
        else:
            cpu_forward_tables(cpu_sizes, batches, args.channels, only=wanted)

    if torch.cuda.is_available() and not args.no_training:
        training_table(sizes, batches, args.channels)
    if not args.no_streaming:
        streaming_note()


if __name__ == "__main__":
    main()
