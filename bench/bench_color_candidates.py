"""Isolated baseline/current color-metric timing and differential checks.

The baseline source is the exact dirty-tree snapshot captured before edits.
Run one process per arm/device/metric to avoid compile/cache interactions.
"""

import argparse
import importlib.util
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_analytics import functional


BASELINE = Path(r"C:\Users\nilas\AppData\Local\Temp\frame_analytics_opt_color_baseline_20260923")
ROOT = Path(__file__).resolve().parents[1] / "frame_analytics"
CANDIDATES = Path(__file__).resolve().parent / "results" / "color_opt_recheck_20260923" / "candidates"


class NamespacedFunctional:
    """Keep baseline/current lazy-compile wrappers separate in one process."""

    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(functional, name)

    def _maybe_compile(self, fn, key, dynamic=None):
        return functional._maybe_compile(fn, f"color:{self.source}:{key}", dynamic)


def load(metric, source):
    if source in ("baseline", "current"):
        path = (BASELINE if source == "baseline" else ROOT) / f"{metric}.py"
    else:
        path = CANDIDATES / f"{source}.py"
    spec = importlib.util.spec_from_file_location(f"frame_analytics._color_{source}_{metric}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._F = NamespacedFunctional(source)
    return getattr(module, metric), module


def make_pairs(device, size, dtype=torch.float32):
    gen = torch.Generator(device=device).manual_seed(1107)
    pairs = []
    for _ in range(4):
        x = torch.rand(1, 3, size, size, device=device, dtype=dtype, generator=gen)
        y = (x + 0.05 * torch.randn(x.shape, device=device, dtype=dtype, generator=gen)).clamp(0, 1)
        pairs.append((x, y))
    return pairs


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def gpu_telemetry():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            check=True, text=True, capture_output=True, timeout=5)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def bench(args):
    functional.set_compile_enabled(args.compiled)
    torch.set_num_threads(4)
    fn, mod = load(args.metric, args.source)
    gpu_before = gpu_telemetry() if args.device == "cuda" else None
    pairs = make_pairs(args.device, args.size)
    kwargs = {"return_map": args.return_map}
    if args.metric in ("flip", "scielab"):
        kwargs["ppd"] = args.ppd if args.ppd else (67.0 if args.metric == "flip" else 30.0)
    with torch.no_grad():
        for i in range(args.warmup):
            fn(*pairs[i % len(pairs)], **kwargs)
            sync(args.device)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            start_allocated = torch.cuda.memory_allocated()
        samples = []
        checksum = 0.0
        for i in range(args.samples):
            sync(args.device)
            t = time.perf_counter()
            for j in range(args.batch_calls):
                x, y = pairs[(i + j) % len(pairs)]
                out = fn(x, y, **kwargs)
            sync(args.device)
            samples.append((time.perf_counter() - t) * 1000 / args.batch_calls)
            checksum += float(out.sum())
    result = {"metric": args.metric, "source": args.source, "device": args.device,
              "size": args.size, "dtype": "float32", "batch": 1,
              "ppd": kwargs.get("ppd"), "return_map": args.return_map,
              "compiled": args.compiled, "torch": torch.__version__,
              "threads": 4, "warmup": args.warmup, "samples": samples,
              "batch_calls_per_sample": args.batch_calls,
              "median_ms": statistics.median(samples), "checksum": checksum,
              "source_file": str(Path(mod.__file__).resolve()),
              "gpu_before": gpu_before,
              "gpu_after": gpu_telemetry() if args.device == "cuda" else None}
    if args.device == "cuda":
        result["peak_extra_allocated_bytes"] = torch.cuda.max_memory_allocated() - start_allocated
    import frame_analytics
    result["package_file"] = str(Path(frame_analytics.__file__).resolve())
    assert Path(frame_analytics.__file__).resolve().parent == ROOT.resolve()
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            fn(*pairs[0], **kwargs)
            sync(args.device)
        result["profile"] = prof.key_averages().table(
            sort_by="self_cuda_time_total" if args.device == "cuda" else "self_cpu_time_total",
            row_limit=12)
    print(json.dumps(result))


def compare(args):
    functional.set_compile_enabled(args.compiled)
    torch.set_num_threads(4)
    base, _ = load(args.metric, "baseline")
    curr, _ = load(args.metric, args.candidate)
    gpu_before = gpu_telemetry() if args.device == "cuda" else None
    pairs = make_pairs(args.device, args.size)
    kwargs = {"return_map": args.return_map}
    if args.metric in ("flip", "scielab"):
        kwargs["ppd"] = args.ppd if args.ppd else (67.0 if args.metric == "flip" else 30.0)
    rng = random.Random(2409)
    btimes, ctimes, checksums = [], [], [0.0, 0.0]
    with torch.no_grad():
        for fn in (base, curr):
            for i in range(args.warmup):
                fn(*pairs[i % len(pairs)], **kwargs)
                sync(args.device)
        for i in range(args.samples):
            arms = [(base, btimes, 0), (curr, ctimes, 1)]
            if rng.randrange(2):
                arms.reverse()
            for fn, samples, idx in arms:
                sync(args.device)
                t = time.perf_counter()
                for j in range(args.batch_calls):
                    out = fn(*pairs[(i + j) % len(pairs)], **kwargs)
                sync(args.device)
                samples.append((time.perf_counter() - t) * 1000 / args.batch_calls)
                checksums[idx] += float(out.sum())
    effects = [100.0 * (b - c) / b for b, c in zip(btimes, ctimes)]
    boot = []
    for _ in range(2000):
        draw = [effects[rng.randrange(len(effects))] for _ in effects]
        boot.append(statistics.median(draw))
    boot.sort()
    import frame_analytics
    assert Path(frame_analytics.__file__).resolve().parent == ROOT.resolve()
    print(json.dumps({"metric": args.metric, "device": args.device, "size": args.size,
                      "dtype": "float32", "batch": 1, "ppd": kwargs.get("ppd"),
                      "return_map": args.return_map, "compiled": args.compiled,
                      "warmup_per_arm": args.warmup, "samples_per_arm": args.samples,
                      "batch_calls_per_sample": args.batch_calls,
                      "baseline_ms": btimes, "candidate_ms": ctimes,
                      "baseline_median_ms": statistics.median(btimes),
                      "candidate_median_ms": statistics.median(ctimes),
                      "median_paired_improvement_pct": statistics.median(effects),
                      "bootstrap_ci95_pct": [boot[49], boot[1949]],
                      "checksums": checksums, "torch": torch.__version__,
                      "candidate_source": args.candidate,
                      "gpu_before": gpu_before,
                      "gpu_after": gpu_telemetry() if args.device == "cuda" else None,
                      "package_file": str(Path(frame_analytics.__file__).resolve())}))


def differential(args):
    functional.set_compile_enabled(False)
    base, _ = load(args.metric, "baseline")
    curr, _ = load(args.metric, args.candidate)
    gen = torch.Generator(device=args.device).manual_seed(1729)
    cases = []
    for dtype, size in ((torch.float32, 17), (torch.float32, 64),
                        (torch.float64, 17), (torch.uint8, 17)):
        if dtype == torch.uint8:
            x = torch.randint(0, 256, (2, 3, size, size), device=args.device, dtype=dtype, generator=gen)
            y = torch.randint(0, 256, (2, 3, size, size), device=args.device, dtype=dtype, generator=gen)
        else:
            x = torch.rand(2, 3, size, size, device=args.device, dtype=dtype, generator=gen)
            y = torch.rand(2, 3, size, size, device=args.device, dtype=dtype, generator=gen)
        for ret, red in ((False, "mean"), (False, "none"), (True, "mean")):
            kw = {"return_map": ret, "reduction": red}
            if args.metric in ("flip", "scielab"):
                kw["ppd"] = 30.0 if size == 17 else 67.0
            cases.append((x, y, kw))
    results = []
    with torch.no_grad():
        for i, (x, y, kw) in enumerate(cases):
            a, b = base(x, y, **kw), curr(x, y, **kw)
            sync(args.device)
            assert a.shape == b.shape and a.dtype == b.dtype, (i, a.shape, b.shape, a.dtype, b.dtype)
            assert torch.equal(torch.isnan(a), torch.isnan(b)), (i, "NaN location")
            diff = (a - b).abs()
            max_abs = float(diff.max())
            denom = a.abs().clamp_min(1e-7)
            max_rel = float((diff / denom).max())
            assert torch.allclose(a, b, atol=2e-5, rtol=2e-5, equal_nan=True), (i, max_abs, max_rel)
            results.append({"case": i, "dtype": str(x.dtype), "size": x.shape[-1],
                            "return_map": kw["return_map"], "reduction": kw["reduction"],
                            "max_abs": max_abs, "max_rel": max_rel})
    if args.metric == "flip":
        import os
        gen = torch.Generator(device=args.device).manual_seed(988)
        x = torch.rand(1, 3, 17, 17, device=args.device, generator=gen)
        y = torch.rand(1, 3, 17, 17, device=args.device, generator=gen)
        for separable in ("1", "0"):
            os.environ["FLIP_SEPARABLE"] = separable
            kwargs = {"ppd": 30.0, "exposure": 0.7, "eps": 1e-8,
                      "return_map": True}
            a = base(x, y, **kwargs)
            b = curr(x, y, **kwargs)
            assert torch.allclose(a, b, atol=2e-5, rtol=2e-5), separable
            results.append({"variant": f"separable={separable}, exposure=0.7, eps=1e-8",
                            "max_abs": float((a - b).abs().max())})
        os.environ.pop("FLIP_SEPARABLE", None)
        for fn in (base, curr):
            xa = x.detach().clone().requires_grad_()
            ya = y.detach().clone().requires_grad_()
            fn(xa, ya, ppd=30.0).backward()
            if fn is base:
                grads = (xa.grad.detach().clone(), ya.grad.detach().clone())
            else:
                assert torch.allclose(grads[0], xa.grad, atol=1e-5, rtol=1e-4)
                assert torch.allclose(grads[1], ya.grad, atol=1e-5, rtol=1e-4)
        results.append({"variant": "input_gradients", "max_abs": max(
            float((grads[0] - xa.grad).abs().max()),
            float((grads[1] - ya.grad).abs().max()))})
    print(json.dumps({"metric": args.metric, "device": args.device,
                      "candidate_source": args.candidate, "cases": results}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metric", choices=("ciede2000", "flip", "scielab"), required=True)
    p.add_argument("--source", choices=("baseline", "current", "ciede_stacked",
                                       "scielab_front", "scielab_lab"), default="current")
    p.add_argument("--candidate", choices=("current", "ciede_stacked",
                                          "scielab_front", "scielab_lab"), default="current")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--ppd", type=float)
    p.add_argument("--return-map", action="store_true")
    p.add_argument("--compiled", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--batch-calls", type=int, default=20,
                   help="serial varying calls timed between one pair of device synchronizations")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--diff", action="store_true")
    p.add_argument("--compare", action="store_true")
    a = p.parse_args()
    differential(a) if a.diff else compare(a) if a.compare else bench(a)
