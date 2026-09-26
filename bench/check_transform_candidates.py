"""Differential checks for transform metrics against the dirty-tree snapshot."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import frame_analytics.functional as functional


BASELINE = Path(os.environ["TEMP"]) / "frame_analytics_transform_baseline_20260923"


def _baseline(name):
    spec = importlib.util.spec_from_file_location(
        f"frame_analytics._baseline_{name}", BASELINE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _candidate(name, directory):
    if directory is None:
        return importlib.import_module(f"frame_analytics.{name}")
    spec = importlib.util.spec_from_file_location(
        f"frame_analytics._candidate_{name}", directory / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _pair(shape, dtype, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.rand(shape, generator=gen, device=device)
    y = (x + torch.randn(shape, generator=gen, device=device) * 0.04).clamp(0, 1)
    if dtype == torch.uint8:
        x = (x * 255).to(dtype)
        y = (y * 255).to(dtype)
    else:
        x, y = x.to(dtype), y.to(dtype)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path)
    args = parser.parse_args()
    functional.set_compile_enabled(False)
    cases = [
        ("dss", "dss", (2, 3, 65, 71), torch.float32, {}),
        ("dss", "dss", (1, 1, 32, 40), torch.float64, {"dct_size": 4, "percentile": 1.0}),
        ("dss", "dss", (1, 3, 64, 64), torch.uint8, {"luma": "bt709", "kernel_size": 5}),
        ("haarpsi", "haarpsi", (2, 3, 65, 71), torch.float32, {}),
        ("haarpsi", "haarpsi", (1, 1, 32, 40), torch.float64, {"subsample": False}),
        ("haarpsi", "haarpsi", (1, 3, 64, 64), torch.uint8, {"luma": "bt709"}),
        ("adm", "adm_like", (2, 3, 65, 71), torch.float32, {}),
        ("adm", "adm_like", (1, 1, 32, 40), torch.float64, {"scales": 2, "thresholds": 0.01}),
        ("adm", "adm_like", (1, 3, 65, 71), torch.uint8, {"wavelet": "db2_like"}),
        ("adm", "adm_like", (1, 1, 64, 64), torch.float32, {"wavelet": "db2_like", "csf": (1, 2, 3)}),
        ("adm", "adm_like", (1, 3, 48, 56), torch.float64, {"wavelet": "db2_like", "scales": 3}),
        ("psnrhvs", "mse_hvs", (2, 3, 65, 71), torch.float32, {}),
        ("psnrhvs", "mse_hvs_m", (1, 1, 32, 40), torch.float32, {}),
        ("psnrhvs", "psnr_hvs", (1, 3, 64, 64), torch.uint8, {"eps": 1e-8}),
        ("psnrhvs", "psnr_hvs_m", (1, 3, 64, 64), torch.uint8, {"eps": 1e-8}),
    ]
    modules = {name: (_baseline(name), _candidate(name, args.candidate_dir))
               for name in {case[0] for case in cases}}
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        for idx, (module, fn, shape, dtype, kwargs) in enumerate(cases):
            x, y = _pair(shape, dtype, device, 112 + idx)
            args = dict(kwargs, reduction="none")
            with torch.inference_mode():
                a = getattr(modules[module][0], fn)(x, y, **args)
                b = getattr(modules[module][1], fn)(x, y, **args)
            ok = a.dtype == b.dtype and a.shape == b.shape and torch.allclose(
                a, b, atol=1e-5, rtol=1e-5, equal_nan=True)
            max_abs = (a - b).abs().max().item()
            print(device, module, fn, idx, ok, f"max_abs={max_abs:.9g}")
            if not ok:
                raise AssertionError((device, module, fn, idx, a, b))
        # The transform is also used with autograd-enabled tensors. Compare
        # both input gradients for the changed db2 path on a non-boundary pair.
        x, y = _pair((1, 1, 32, 40), torch.float32, device, 901)
        gradients = []
        for mod in modules["adm"]:
            # Inference-mode checks above populate this module's weight caches
            # with inference tensors, which autograd cannot save for backward.
            # Clear them to compare the normal fresh-process training path.
            mod._db2_cache.clear()
            mod._csf_cache.clear()
            mod._like_weight_cache.clear()
            xa, ya = x.clone().requires_grad_(), y.clone().requires_grad_()
            score = mod.adm_like(xa, ya, wavelet="db2_like")
            gradients.append(torch.autograd.grad(score, (xa, ya)))
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
        print(device, "adm db2 gradients: OK")
    print("differential: OK")


if __name__ == "__main__":
    main()
