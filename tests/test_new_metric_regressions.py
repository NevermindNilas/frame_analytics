"""Regression coverage for metric autograd, caches, and streaming integration."""

from __future__ import annotations

import importlib
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_analytics as fa
from frame_analytics import functional as functional

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
METRIC_MODULES = (
    "adm", "ciede2000", "dss", "flip", "fsim", "haarpsi", "iwssim",
    "mdsi", "ms_gmsd", "nlpd", "psnrhvs", "scielab", "srsim", "vifp", "vsi",
)


@pytest.fixture(autouse=True)
def eager_metrics(monkeypatch):
    # These bugs must also be fixed on installations without a compiler.
    monkeypatch.setattr(functional, "_COMPILE_ENABLED", False)


@pytest.fixture
def fresh_constants(monkeypatch):
    for name in METRIC_MODULES + ("functional",):
        module = importlib.import_module("frame_analytics." + name)
        for key, value in list(vars(module).items()):
            if isinstance(value, dict) and key.lower().endswith("_cache"):
                if key != "_compiled_cache":
                    monkeypatch.setattr(module, key, {})


def _pair(device, size=64, channels=3, dtype=torch.float32):
    rng = torch.Generator(device=device).manual_seed(19)
    shape = (1, channels, size, size)
    return (torch.rand(shape, generator=rng, device=device, dtype=dtype),
            torch.rand(shape, generator=rng, device=device, dtype=dtype))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric", ["DSS", "VIFP"])
def test_loss_backward_on_nonidentical_images(device, metric, fresh_constants):
    pred, target = _pair(device)
    pred.requires_grad_()
    loss = getattr(fa, metric)().loss(pred, target)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


@pytest.mark.parametrize("metric", ["DSS", "VIFP", "CIEDE2000", "SCIELAB", "FLIP"])
def test_loss_gradient_matches_finite_difference(metric):
    pred, target = _pair("cpu", dtype=torch.float64)
    pred.requires_grad_()
    criterion = getattr(fa, metric)(dtype=torch.float64)
    criterion.loss(pred, target).backward()
    index = pred.grad.abs().flatten().argmax().item()
    analytic = pred.grad.flatten()[index].item()
    plus, minus = pred.detach().clone(), pred.detach().clone()
    step = 1e-5
    plus.flatten()[index] += step
    minus.flatten()[index] -= step
    numeric = (criterion.loss(plus, target) - criterion.loss(minus, target)) / (2 * step)
    assert analytic == pytest.approx(numeric.item(), rel=1e-4, abs=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_vifp_identical_images_remain_connected_to_autograd(device):
    pred, _ = _pair(device)
    pred.requires_grad_()
    loss = fa.VIFP().loss(pred, pred.detach().clone())
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("channels,luma", [(1, None), (3, True)])
def test_ms_gmsd_minimum_size_backward(device, channels, luma):
    pred, target = _pair(device, size=48, channels=channels)
    pred.requires_grad_()
    loss = fa.MS_GMSD(luma=luma).loss(pred, target)
    assert loss.item() > 0.0
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric", ["CIEDE2000", "SCIELAB", "FLIP"])
@pytest.mark.parametrize("black_prediction", [False, True])
def test_color_loss_backward_with_black_image(device, metric, black_prediction):
    pred, target = _pair(device)
    if black_prediction:
        pred.zero_()
    else:
        target.zero_()
    pred.requires_grad_()
    loss = getattr(fa, metric)().loss(pred, target)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric", ["CIEDE2000", "SCIELAB", "FLIP", "MS_GMSD"])
def test_zero_loss_has_finite_zero_gradient(device, metric):
    pred, _ = _pair(device)
    pred.requires_grad_()
    loss = getattr(fa, metric)().loss(pred, pred.detach().clone())
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,kwargs", [
    ("adm_like", {}), ("adm_like", {"wavelet": "db2_like"}),
    ("dss", {}), ("haarpsi", {}), ("psnr_hvs", {}), ("psnr_hvs_m", {}),
    ("ciede2000", {}), ("scielab", {}), ("flip", {}),
    ("ms_gmsd", {}), ("vifp", {}), ("iw_ssim", {}), ("nlpd", {"scales": 4}),
    ("fsim", {}), ("srsim", {}), ("vsi", {}), ("mdsi", {}),
])
def test_inference_warmup_then_training(device, name, kwargs, fresh_constants):
    pred, target = _pair(device, size=256 if name == "iw_ssim" else 64)
    fn = getattr(fa, name)
    with torch.inference_mode():
        fn(pred, target, **kwargs)
    pred.requires_grad_()
    fn(pred, target, **kwargs).backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,kwargs", [
    ("scielab", {}), ("dss", {}), ("psnr_hvs", {}),
    ("psnr_hvs_m", {}), ("adm_like", {"wavelet": "db2_like"}), ("flip", {}),
])
def test_metric_precision_is_independent_of_autocast(device, name, kwargs):
    pred, target = _pair(device)
    fn = getattr(fa, name)
    expected = fn(pred, target, **kwargs)
    with torch.autocast(device, dtype=torch.bfloat16 if device == "cpu" else torch.float16):
        actual = fn(pred, target, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_fsim_directional_filter_retains_quadrature():
    fsim = importlib.import_module("frame_analytics.fsim")
    n = 64
    axis = torch.arange(n, dtype=torch.float64)
    image = torch.sin(2 * math.pi * axis / 8).expand(n, n)
    radial, spread = fsim._planes(n, n, torch.device("cpu"), torch.float64)
    response = torch.fft.ifft2(torch.fft.fft2(image) * radial[0] * spread[0])
    assert response.imag.square().mean().sqrt().item() == pytest.approx(
        0.3148969738, abs=1e-9)


def test_fsim_matches_author_pc2_fixture():
    # Zhang's FeatureSIM.m / phasecong2: 40x40 square vs 5x5 box blur.
    image = torch.zeros(1, 1, 64, 64, dtype=torch.float64)
    image[:, :, 12:52, 12:52] = 1.0
    blurred = F.avg_pool2d(image, 5, stride=1, padding=2)
    assert fa.fsim(image, blurred).item() == pytest.approx(0.7618871562, abs=2e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("metric", ["vifp", "iw_ssim"])
def test_streaming_capture_fallback_keeps_cuda_usable(metric):
    # A failed capture can invalidate the stream, so isolate the regression
    # from the parent test process's CUDA context.
    script = f"""
import torch, frame_analytics as fa
fa.set_compile_enabled(False)
torch.set_num_threads(2)
torch.manual_seed(19)
x = torch.randint(0, 256, (1,3,256,256), device='cuda', dtype=torch.uint8)
y = torch.randint(0, 256, x.shape, device='cuda', dtype=torch.uint8)
expected = float(getattr(fa, {metric!r})(x,y))
scorer = fa.StreamingMetrics(x.shape, device='cuda', metrics=({metric!r},))
for _ in range(2):
    actual = scorer.update(x,y)[{metric!r}]
    assert abs(actual - expected) < 1e-6, (actual, expected)
torch.ones(1,device='cuda').cpu()
"""
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
