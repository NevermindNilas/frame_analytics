"""Paired, warm serial per-frame timings against the dirty-tree snapshot.

Example: python bench/bench_transform_candidates.py --metric dss --device cpu
The four baseline modules live in %TEMP%/frame_analytics_transform_baseline_20260923.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import frame_analytics
import frame_analytics.functional as functional


BASELINE = Path(os.environ["TEMP"]) / "frame_analytics_transform_baseline_20260923"


def _module(name: str, baseline: bool, candidate_dir: Optional[Path] = None):
    if not baseline and candidate_dir is None:
        return importlib.import_module(f"frame_analytics.{name}")
    source_dir = BASELINE if baseline else candidate_dir
    variant = "baseline" if baseline else "candidate"
    spec = importlib.util.spec_from_file_location(
        f"frame_analytics._{variant}_{name}", source_dir / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _case(mod, metric: str, x, y):
    if metric == "dss":
        return lambda: mod.dss(x, y, reduction="none")
    if metric == "haarpsi":
        return lambda: mod.haarpsi(x, y, reduction="none")
    if metric == "adm_haar":
        return lambda: mod.adm_like(x, y, wavelet="haar", reduction="none")
    if metric == "adm_db2":
        return lambda: mod.adm_like(x, y, wavelet="db2_like", reduction="none")
    if metric == "psnr_hvs":
        return lambda: mod.psnr_hvs(x, y, reduction="none")
    if metric == "psnr_hvs_m":
        return lambda: mod.psnr_hvs_m(x, y, reduction="none")
    raise ValueError(metric)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", required=True, choices=(
        "dss", "haarpsi", "adm_haar", "adm_db2", "psnr_hvs", "psnr_hvs_m"))
    p.add_argument("--device", required=True, choices=("cpu", "cuda"))
    p.add_argument("--held-out", action="store_true")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--profile-only", action="store_true")
    p.add_argument("--profile-new", action="store_true")
    p.add_argument("--memory-only", action="store_true")
    p.add_argument("--compiled", action="store_true",
                   help="Use the package's default torch.compile epilogue path")
    p.add_argument("--candidate-dir", type=Path,
                   help="Load candidate modules from this isolated directory")
    args = p.parse_args()
    package_path = Path(frame_analytics.__file__).resolve()
    assert package_path.parent == Path(__file__).resolve().parents[1] / "frame_analytics", package_path
    if not args.compiled:
        functional.set_compile_enabled(False)  # stable eager comparison
    device = torch.device(args.device)
    name = "adm" if args.metric.startswith("adm_") else (
        "psnrhvs" if args.metric.startswith("psnr_hvs") else args.metric)
    base, new = _module(name, True), _module(name, False, args.candidate_dir)
    shape = (1, 3, 128, 160) if args.held_out else (2, 1, 256, 256)
    gen = torch.Generator(device=device).manual_seed(116 if args.held_out else 73)
    fixtures = []
    for _ in range(4):
        x = torch.rand(shape, generator=gen, device=device)
        y = (x + 0.05 * torch.randn(shape, generator=gen, device=device)).clamp(0, 1)
        fixtures.append((x, y))
    calls = {"base": [_case(base, args.metric, x, y) for x, y in fixtures],
             "new": [_case(new, args.metric, x, y) for x, y in fixtures]}
    if args.memory_only:
        if device.type != "cuda":
            raise ValueError("--memory-only requires --device cuda")
        peaks = {}
        with torch.inference_mode():
            for arm in ("base", "new"):
                for call in calls[arm]:
                    call()
                _sync(device)
                start_allocated = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                out = calls[arm][0]()
                _sync(device)
                peaks[arm] = torch.cuda.max_memory_allocated(device) - start_allocated
                del out
        print(json.dumps({"metric": args.metric, "device": args.device,
                          "package_path": str(package_path),
                          "incremental_peak_bytes": peaks}))
        return
    if args.profile_only:
        from torch.profiler import ProfilerActivity, profile

        arm = "new" if args.profile_new else "base"
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with torch.inference_mode():
            for _ in range(3):
                calls[arm][0]()
            _sync(device)
            with profile(activities=activities) as prof:
                for _ in range(3):
                    calls[arm][0]()
                _sync(device)
        sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=12))
        return
    rng = random.Random(9328)
    with torch.inference_mode():
        for j in range(args.warmup):
            for arm in ("base", "new"):
                calls[arm][j % len(fixtures)]()
        _sync(device)
        raw = {"base": [], "new": []}
        outputs = {}
        orders = []
        for _ in range(args.samples):
            order = ["base", "new"]
            rng.shuffle(order)
            orders.append(order)
            for arm in order:
                _sync(device)
                start = time.perf_counter_ns()
                for j in range(args.repeat):
                    outputs[arm] = calls[arm][j % len(fixtures)]()
                _sync(device)
                raw[arm].append((time.perf_counter_ns() - start) / (1e6 * args.repeat))
    a, b = outputs["base"], outputs["new"]
    close = torch.allclose(a, b, atol=1e-5, rtol=1e-5, equal_nan=True)
    dif = float((a - b).abs().max().item())
    paired = [b / a for a, b in zip(raw["base"], raw["new"])]
    boot = []
    for _ in range(10000):
        draw = [paired[rng.randrange(len(paired))] for _ in paired]
        boot.append(statistics.median(draw))
    boot.sort()
    print(json.dumps({"metric": args.metric, "device": args.device,
                      "shape": shape, "threads": torch.get_num_threads(),
                      "package_path": str(package_path),
                      "candidate_dir": str(args.candidate_dir) if args.candidate_dir else None,
                      "torch": torch.__version__, "compile": args.compiled,
                      "warmup": args.warmup, "samples": args.samples,
                      "repeat": args.repeat, "varying_fixtures": len(fixtures),
                      "order": orders, "ms": raw,
                      "median_ms": {k: statistics.median(v) for k, v in raw.items()},
                      "paired_ratio_median": statistics.median(paired),
                      "paired_ratio_ci95": [boot[250], boot[9749]],
                      "allclose": close, "max_abs_diff": dif}))


if __name__ == "__main__":
    main()
