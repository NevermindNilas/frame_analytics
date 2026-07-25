"""pytest wrapper around the accuracy gate, plus API-surface checks.

`tests/validate.py` is the substantive check -- it compares every backend
against the float64 reference. This file makes it runnable under pytest and
adds the cheap contract tests that do not belong in a printed table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_analytics as fa
from frame_analytics import reference as ref

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
HINTS = ["torch", "auto"]


def _pair(h=97, w=131, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.clip(120 + 50 * np.sin(xx / 11.0) * np.cos(yy / 13.0), 0, 255).round()
    b = np.clip(a + 7 * rng.standard_normal((h, w)), 0, 255).round()
    return a, b


def test_validate_suite():
    """The full cross-backend accuracy gate."""
    import validate

    assert validate.main() == 0


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("hint", HINTS)
def test_ssim_matches_reference(dev, hint):
    a, b = _pair()
    want = ref.ssim_reference(a, b, 255.0)
    x = torch.as_tensor(a, dtype=torch.uint8, device=dev)
    y = torch.as_tensor(b, dtype=torch.uint8, device=dev)
    got = float(fa.ssim(x, y, data_range=255.0, backend_hint=hint))
    assert abs(got - want) < 5e-6


@pytest.mark.parametrize("dev", DEVICES)
def test_input_ranks_agree(dev):
    a, b = _pair()
    x = torch.as_tensor(a, dtype=torch.uint8, device=dev)
    y = torch.as_tensor(b, dtype=torch.uint8, device=dev)
    v2 = float(fa.ssim(x, y, data_range=255.0))
    v3 = float(fa.ssim(x[None], y[None], data_range=255.0))
    v4 = float(fa.ssim(x[None, None], y[None, None], data_range=255.0))
    assert v2 == v3 == v4


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dt", [torch.uint8, torch.float32, torch.float64])
def test_dtypes_agree(dev, dt):
    a, b = _pair()
    want = ref.ssim_reference(a, b, 255.0)
    x = torch.as_tensor(a, dtype=dt, device=dev)
    y = torch.as_tensor(b, dtype=dt, device=dev)
    assert abs(float(fa.ssim(x, y, data_range=255.0)) - want) < 5e-6


@pytest.mark.parametrize("dev", DEVICES)
def test_map_mean_equals_scalar(dev):
    a, b = _pair()
    x = torch.as_tensor(a, dtype=torch.uint8, device=dev)
    y = torch.as_tensor(b, dtype=torch.uint8, device=dev)
    m = fa.ssim(x, y, data_range=255.0, return_map=True)
    assert m.shape == (1, 1, a.shape[0] - 10, a.shape[1] - 10)
    assert abs(float(m.mean()) - float(fa.ssim(x, y, data_range=255.0))) < 5e-6


@pytest.mark.parametrize("dev", DEVICES)
def test_identity_and_zero(dev):
    x = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8, device=dev)
    assert abs(float(fa.ssim(x, x, data_range=255.0)) - 1.0) < 1e-6
    assert float(fa.mse(x, x)) == 0.0
    assert float(fa.psnr(x, x, data_range=255.0)) == float("inf")


@pytest.mark.parametrize("dev", DEVICES)
def test_reduction_none_shape(dev):
    x = torch.randint(0, 256, (5, 1, 64, 64), dtype=torch.uint8, device=dev)
    y = torch.randint(0, 256, (5, 1, 64, 64), dtype=torch.uint8, device=dev)
    for fn in (fa.ssim, fa.mse):
        assert fn(x, y, reduction="none").shape == (5,)
    assert fa.psnr(x, y, reduction="none").shape == (5,)
    assert abs(float(fa.ssim(x, y, reduction="none").mean())
               - float(fa.ssim(x, y))) < 1e-9


@pytest.mark.parametrize("dev", DEVICES)
def test_psnr_matches_mse(dev):
    import math

    x = torch.randint(0, 256, (2, 1, 96, 96), dtype=torch.uint8, device=dev)
    y = torch.randint(0, 256, (2, 1, 96, 96), dtype=torch.uint8, device=dev)
    m = float(fa.mse(x, y))
    p = float(fa.psnr(x, y, data_range=255.0))
    assert abs(p - 10 * math.log10(255.0 ** 2 / m)) < 1e-9


@pytest.mark.parametrize("dev", DEVICES)
def test_gradients_flow(dev):
    x = torch.rand(1, 1, 64, 64, device=dev, requires_grad=True)
    y = torch.rand(1, 1, 64, 64, device=dev)
    loss = fa.SSIM(data_range=1.0).loss(x, y)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_mismatched_inputs_raise():
    x = torch.zeros(1, 1, 32, 32)
    assert pytest.raises(ValueError, fa.ssim, x, torch.zeros(1, 1, 32, 33))
    assert pytest.raises(ValueError, fa.ssim, torch.zeros(1, 1, 8, 8),
                         torch.zeros(1, 1, 8, 8))          # smaller than window
    assert pytest.raises(ValueError, lambda: fa.mse(x, x, reduction="sum"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_streaming_matches_functional():
    a = torch.randint(0, 256, (1, 3, 240, 320), dtype=torch.uint8)
    b = (a.float() + 9).clamp(0, 255).to(torch.uint8)
    sm = fa.StreamingMetrics((1, 3, 240, 320), device="cuda", dtype=torch.uint8,
                             data_range=255.0)
    out = sm.update(a, b)
    ac, bc = a.cuda(), b.cuda()
    assert abs(out["ssim"] - float(fa.ssim(ac, bc, data_range=255.0))) < 1e-12
    assert abs(out["psnr"] - float(fa.psnr(ac, bc, data_range=255.0))) < 1e-12
    assert abs(out["mse"] - float(fa.mse(ac, bc))) < 1e-12


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("win", [3, 5, 7, 9, 11])
def test_window_sizes(dev, win):
    a, b = _pair()
    want = ref.ssim_reference(a, b, 255.0, win_size=win)
    x = torch.as_tensor(a, dtype=torch.uint8, device=dev)
    y = torch.as_tensor(b, dtype=torch.uint8, device=dev)
    assert abs(float(fa.ssim(x, y, data_range=255.0, win_size=win)) - want) < 5e-6
