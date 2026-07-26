"""LPIPS: speed, memory and accuracy against the alternatives.

Unlike the rest of the benchmarks in this directory there is no fused kernel
here to be dramatically faster than anyone -- every implementation, ours
included, spends its time inside the same cuDNN convolutions. Read the tables
accordingly: the interesting columns are memory, and the accuracy row that
explains why our forward is *not* the fastest.

    python bench/bench_lpips.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_analytics as fa
from frame_analytics import reference as ref

MIB = 2 ** 20
SIZES = ((1, 256, 256), (1, 512, 512), (1, 1080, 1920), (8, 256, 256), (8, 512, 512))


def timeit(fn, *, iters: int, repeat: int, cuda: bool) -> float:
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


def peak_mib(fn) -> float:
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


def competitors(device):
    """Everything else that computes LPIPS-Alex, as ``name -> callable``."""
    out = {}
    try:
        import lpips as pkg
        net = pkg.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()
        out["lpips (reference pkg)"] = lambda a, b: net(a * 2 - 1, b * 2 - 1).mean()
    except Exception as exc:
        print(f"  skip lpips: {exc}")
    try:
        import torchmetrics.image.lpip as tml
        m = tml.LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True).to(device)
        out["torchmetrics"] = lambda a, b: m(a, b)
    except Exception as exc:
        print(f"  skip torchmetrics: {exc}")
    try:
        import piq
        # piq ships only the VGG trunk for LPIPS, so this row is a different
        # metric, not a slower way of computing the same one. Timed anyway --
        # it is what a piq user actually pays.
        p = piq.LPIPS(reduction="mean").to(device)
        out["piq (vgg trunk)"] = lambda a, b: p(a, b)
    except Exception as exc:
        print(f"  skip piq: {exc}")
    return out


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device; LPIPS is a conv workload and the CPU numbers "
              "are not interesting enough to print")
        return 0
    dev = torch.device("cuda")
    torch.manual_seed(0)
    print("device:", torch.cuda.get_device_name(0))
    print("backend:", fa.backend_status())
    others = competitors(dev)

    # ---- accuracy ------------------------------------------------------- #
    print("\n== accuracy: worst |err| vs the float64 reference forward ==")
    w64 = ref.load_lpips_weights()
    rng = np.random.default_rng(11)
    rows = {}
    for _ in range(3):
        a = rng.integers(0, 256, (1, 3, 64, 64)).astype(np.float64)
        b = np.clip(a + 30.0 * rng.standard_normal(a.shape), 0, 255).round()
        want = float(ref.lpips_reference(a, b, data_range=255.0, weights=w64)[0])
        ta = torch.as_tensor(a / 255.0, dtype=torch.float32, device=dev)
        tb = torch.as_tensor(b / 255.0, dtype=torch.float32, device=dev)
        cand = {
            "frame_analytics": lambda: fa.lpips(ta, tb, data_range=1.0),
            "frame_analytics (allow_tf32=True)":
                lambda: fa.lpips(ta, tb, data_range=1.0, allow_tf32=True),
        }
        for name, fn in others.items():
            cand[name] = lambda fn=fn: fn(ta, tb)
        for name, fn in cand.items():
            if name.startswith("piq"):
                continue                       # different trunk, different metric
            with torch.no_grad():
                err = abs(float(fn()) - want)
            rows[name] = max(rows.get(name, 0.0), err)
    for name, err in rows.items():
        print(f"  {name:<38} {err:.2e}")
    print("  The gap is TF32: cuDNN picks it by default on Ampere and later, and")
    print("  every implementation above except ours leaves that default alone.")

    # ---- speed ---------------------------------------------------------- #
    hdr = ["frame_analytics", "frame_analytics (tf32)"] + list(others)

    def call(name, a, b):
        if name == "frame_analytics":
            return fa.lpips(a, b, data_range=1.0)
        if name.endswith("(tf32)"):
            return fa.lpips(a, b, data_range=1.0, allow_tf32=True)
        return others[name](a, b)

    # Warm every (implementation, size) pair before the first timed row.
    # Otherwise that row absorbs lazy CUDA module load, cuDNN algorithm
    # selection and Dynamo compilation, and reads several ms high -- which is
    # exactly how the 256x256 row ends up looking slower than the 512x512 one.
    print("\nwarming up...", flush=True)
    for n, h, w in SIZES:
        a = torch.rand(n, 3, h, w, device=dev)
        b = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
        for name in hdr:
            try:
                with torch.no_grad():
                    call(name, a, b)
            except Exception:
                pass
        del a, b
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n== forward, ms/call (fp32 in, [0,1]) ==")
    print(f"  {'size':<16}" + "".join(f"{h[:22]:>24}" for h in hdr))
    for n, h, w in SIZES:
        a = torch.rand(n, 3, h, w, device=dev)
        b = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
        cells = []
        for name in hdr:
            fn = lambda name=name: call(name, a, b)
            try:
                with torch.no_grad():
                    t = timeit(fn, iters=10, repeat=3, cuda=True)
                cells.append(f"{t * 1e3:24.3f}")
            except Exception:
                cells.append(f"{'oom/err':>24}")
        print(f"  {f'{n}x3x{h}x{w}':<16}" + "".join(cells))
        del a, b
        gc.collect()
        torch.cuda.empty_cache()

    # ---- memory --------------------------------------------------------- #
    print("\n== peak MiB above the input pair, forward ==")
    print(f"  {'size':<16}" + "".join(f"{h[:22]:>24}" for h in hdr))
    for n, h, w in SIZES:
        a = torch.rand(n, 3, h, w, device=dev)
        b = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
        cells = []
        for name in hdr:
            fn = lambda name=name: call(name, a, b)
            try:
                with torch.no_grad():
                    cells.append(f"{peak_mib(fn):24.2f}")
            except Exception:
                cells.append(f"{'oom/err':>24}")
        print(f"  {f'{n}x3x{h}x{w}':<16}" + "".join(cells))
        del a, b
        gc.collect()
        torch.cuda.empty_cache()

    # ---- loss step ------------------------------------------------------ #
    print("\n== loss step (fwd+bwd), ms/call and peak MiB ==")
    for n, h, w in ((1, 256, 256), (8, 256, 256), (1, 1080, 1920)):
        a = torch.rand(n, 3, h, w, device=dev, requires_grad=True)
        b = torch.rand(n, 3, h, w, device=dev)

        def step(fn):
            def go():
                if a.grad is not None:
                    a.grad = None
                fn(a, b).backward()
            return go

        cells = []
        for name in hdr:
            f = lambda p, t, name=name: call(name, p, t)
            try:
                t = timeit(step(f), iters=5, repeat=3, cuda=True)
                m = peak_mib(step(f))
                cells.append(f"{t * 1e3:10.2f} /{m:8.1f}")
            except Exception:
                cells.append(f"{'oom/err':>20}")
        print(f"  {f'{n}x3x{h}x{w}':<16}" + "".join(f"{c:>22}" for c in cells))
        del a, b
        gc.collect()
        torch.cuda.empty_cache()

    print("\n  header order:", hdr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
