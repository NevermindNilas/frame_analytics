"""VapourSynth throughput: how many frames a second the filter scores.

    python bench/bench_vapoursynth.py
    python bench/bench_vapoursynth.py --frames 400 --accelerator cpu

What is being measured is the *filter*, not the kernel: the numbers include
VapourSynth's frame request, the host->device copy of every plane, the Python
selector, and setting the frame properties. The kernel tables in
``bench/bench.py`` are the other half -- for SSIM at 1080p they say 0.085 ms,
and anything above that here is the plumbing.

The source clips are short and looped, so the decoder is not in the
measurement; a real script reads from a demuxer and that cost is its own.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vapoursynth as vs

from frame_analytics import vapoursynth as fa_vs

core = vs.core

SIZES = [("720p", 1280, 720), ("1080p", 1920, 1080), ("4K", 3840, 2160)]
CASES = [
    ("psnr", ["psnr"]),
    ("ssim", ["ssim"]),
    ("psnr+ssim", ["psnr", "ssim"]),
    ("ms_ssim", ["ms_ssim"]),
    ("gmsd", ["gmsd"]),
    ("psnr+ssim+ms_ssim+gmsd", ["psnr", "ssim", "ms_ssim", "gmsd"]),
]


SRC_FRAMES = 4          # distinct frames, looped; the cache serves the repeats


def _source(width, height, fmt, seed, distort, length=1024):
    """A short clip of real-looking noise, held in cache and looped.

    Looping is what keeps frame *generation* out of the measurement: after the
    first pass VapourSynth serves all four source frames from its cache, so
    what is left in the loop is the filter under test.
    """
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=fmt,
                               length=SRC_FRAMES)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = 128 + 60 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
    frames = []
    for _ in range(SRC_FRAMES):
        f = base + 6.0 * rng.standard_normal((height, width))
        if distort:
            f += 9.0 * rng.standard_normal((height, width))
        frames.append(np.clip(f, 0, 255).astype(np.uint8))

    def sel(n, f):
        out = f.copy()
        src = frames[n % SRC_FRAMES]
        for p in range(out.format.num_planes):
            sh = out.format.subsampling_h if p else 0
            sw = out.format.subsampling_w if p else 0
            np.asarray(out[p])[:] = src[::1 << sh, ::1 << sw]
        return out

    src = core.std.ModifyFrame(blank, blank, sel)
    return core.std.Loop(src, times=-(-length // SRC_FRAMES))


def _time(clip, frames):
    """Time a full pass over ``frames`` frames.

    Trimmed rather than broken out of: abandoning a ``frames()`` generator
    leaves the core's prefetch in flight, and a worker thread that calls back
    into Python during interpreter shutdown crashes the process.
    """
    clip = clip[:frames]
    t0 = time.perf_counter()
    n = 0
    for _ in clip.frames(close=True):
        n += 1
    return time.perf_counter() - t0, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--accelerator", default="auto", choices=fa_vs.ACCELERATORS,
                    help="auto (default), gpu, cuda or cpu")
    ap.add_argument("--device", default=None,
                    help="a specific torch device (cuda:1, mps); overrides --accelerator")
    ap.add_argument("--format", default="YUV420P8")
    ap.add_argument("--size", default=None, choices=[s[0] for s in SIZES],
                    help="one size only -- the honest way to read the "
                         "SSIMULACRA 2 row (see the note under the table)")
    args = ap.parse_args()

    sizes = [s for s in SIZES if args.size is None or s[0] == args.size]
    fmt = getattr(vs, args.format)
    device = fa_vs.resolve_device(args.accelerator, args.device)
    print(f"VapourSynth {core}\ndevice={device}  format={args.format}  "
          f"frames={args.frames}  threads={core.num_threads}\n")

    head = f"{'metrics':<24}" + "".join(f"{name:>18}" for name, _, _ in sizes)
    print(head)
    print("-" * len(head))

    for label, feats in CASES:
        row = f"{label:<24}"
        for _, w, h in sizes:
            ref = _source(w, h, fmt, 1, False)
            dist = _source(w, h, fmt, 2, True)
            scored = fa_vs.Metric(ref, dist, feats, device=device)
            _time(scored, 8)                       # warm up compile + allocator
            secs, n = _time(scored, args.frames)
            row += f"{n / secs:>13.1f} fps"
        print(row)

    print("\nSSIMULACRA 2 and LPIPS want RGB, so they get their own pass:")
    for label, feats in (("lpips", ["lpips"]), ("ssimulacra2", ["ssimulacra2"])):
        row = f"{label:<24}"
        for _, w, h in sizes:
            ref = _source(w, h, vs.RGB24, 1, False)
            dist = _source(w, h, vs.RGB24, 2, True)
            scored = fa_vs.Metric(ref, dist, feats, device=device)
            _time(scored, 4)
            secs, n = _time(scored, min(args.frames, 60))
            row += f"{n / secs:>13.1f} fps"
        print(row)

    if args.size is None and len(sizes) > 1:
        print("\nNote: SSIMULACRA 2 builds a six-scale pyramid, so one call puts six\n"
              "shapes through the same compiled functions -- a second resolution in\n"
              "the same process exhausts Dynamo's recompile limit and the rest of\n"
              "the row finishes in eager. Run --size <one> for the number a real\n"
              "single-resolution script gets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
