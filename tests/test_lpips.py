"""LPIPS: agreement with the upstream package, with the float64 reference,
and the contract tests the rest of the library's metrics get.

LPIPS has no paper formula to be correct against -- it *is* its weights -- so
correctness is pinned two independent ways:

* against ``richzhang/PerceptualSimilarity`` itself, when installed. This is
  the number the literature reports and the one that matters.
* against ``reference.lpips_reference``, a float64 numpy transcription that
  shares no code with the torch path. This one runs everywhere, including CI
  with no ``lpips`` installed, and catches an algorithmic slip (a misplaced
  pool, a stride, an epsilon on the wrong side of a square root) that agreeing
  with nothing at all would hide.
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

# float32 ULP at a typical LPIPS value (~0.2) is ~1.5e-08, and the score is a
# sum of five means over feature maps, so a few ULP is the floor.
TOL32 = 5e-07
TOL64 = 1e-08


def _pair(n=2, h=72, w=88, blend=0.35, seed=5):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, 3, h, w, generator=g, dtype=torch.float64)
    b = torch.rand(n, 3, h, w, generator=g, dtype=torch.float64)
    return a, a * (1.0 - blend) + b * blend


@pytest.fixture(scope="module")
def w64():
    return ref.load_lpips_weights()


def test_weights_ship():
    p = fa.lpips_weights_path("alex")
    assert p.is_file()
    assert "alex" in fa.available_nets()
    # 9.4 MiB is the whole point of slicing the trunk out; if this ever jumps
    # to 233 MiB somebody has started shipping the classifier head.
    assert p.stat().st_size < 12 * 2**20


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("blend", [0.0, 0.2, 0.75])
def test_matches_float64_reference(dev, blend, w64):
    a, b = _pair(blend=blend)
    want = ref.lpips_reference(a.numpy(), b.numpy(), data_range=1.0, weights=w64)

    got32 = fa.lpips(a.float().to(dev), b.float().to(dev), data_range=1.0,
                     reduction="none").double().cpu().numpy()
    assert np.abs(got32 - want).max() < TOL32

    got64 = fa.lpips(a.to(dev), b.to(dev), data_range=1.0, reduction="none",
                     dtype=torch.float64).cpu().numpy()
    assert np.abs(got64 - want).max() < TOL64


def test_matches_upstream_package():
    """The number papers print. Skipped when `lpips` is not installed."""
    pkg = pytest.importorskip("lpips")
    net = pkg.LPIPS(net="alex", version="0.1", verbose=False).eval()

    worst = 0.0
    for blend in (0.0, 0.1, 0.5, 1.0):
        a, b = _pair(blend=blend)
        a, b = a.float(), b.float()
        with torch.no_grad():
            want = net(a * 2 - 1, b * 2 - 1).flatten()
        got = fa.lpips(a, b, data_range=1.0, reduction="none")
        worst = max(worst, float((want - got).abs().max()))
    assert worst < TOL32, f"worst disagreement with lpips package: {worst:g}"


def test_identical_images_score_zero():
    a, _ = _pair()
    assert float(fa.lpips(a, a.clone(), data_range=1.0)) == 0.0


def test_reduction_and_shapes():
    a, b = _pair(n=3)
    a, b = a.float(), b.float()
    per = fa.lpips(a, b, data_range=1.0, reduction="none")
    assert per.shape == (3,)
    assert torch.allclose(per.mean(), fa.lpips(a, b, data_range=1.0))

    m = fa.lpips(a, b, data_range=1.0, return_map=True)
    assert m.shape == (3, 1, a.shape[-2], a.shape[-1])
    # the map is the same quantity resampled, so its mean tracks the scalar
    assert torch.allclose(m.mean(dim=(1, 2, 3)), per, atol=1e-4)

    # (C,H,W) and (H,W) promote like everywhere else
    assert fa.lpips(a[0], b[0], data_range=1.0, reduction="none").shape == (1,)


def test_uint8_matches_float():
    a, b = _pair()
    au = (a * 255).round().clamp(0, 255).to(torch.uint8)
    bu = (b * 255).round().clamp(0, 255).to(torch.uint8)
    want = fa.lpips(au.float() / 255.0, bu.float() / 255.0, data_range=1.0)
    assert abs(float(fa.lpips(au, bu)) - float(want)) < TOL32


def test_grayscale_is_replicated():
    a, b = _pair()
    g0, g1 = a[:, :1].float(), b[:, :1].float()
    assert torch.allclose(fa.lpips(g0, g1, data_range=1.0),
                          fa.lpips(g0.expand(-1, 3, -1, -1),
                                   g1.expand(-1, 3, -1, -1), data_range=1.0))


def test_crop_border():
    a, b = _pair(h=80, w=80)
    a, b = a.float(), b.float()
    assert torch.allclose(fa.lpips(a, b, data_range=1.0, crop_border=4),
                          fa.lpips(a[..., 4:-4, 4:-4], b[..., 4:-4, 4:-4],
                                   data_range=1.0))


@pytest.mark.parametrize("dev", DEVICES)
def test_gradient_flows_to_both_inputs(dev):
    a, b = _pair(h=64, w=64)
    a = a.float().to(dev).requires_grad_(True)
    b = b.float().to(dev).requires_grad_(True)
    fa.lpips(a, b, data_range=1.0).backward()
    for g in (a.grad, b.grad):
        assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_gradient_matches_central_differences():
    """Differenced on the float64 path: an fp32 score has ~1e-8 of headroom,
    which is the same size as the perturbation's effect."""
    a, b = _pair(n=1, h=64, w=64)
    a = a.requires_grad_(True)
    fa.lpips(a, b, data_range=1.0, dtype=torch.float64).backward()
    grad = a.grad
    assert grad is not None

    h = 1e-5
    for idx in [(0, 0, 10, 12), (0, 2, 33, 41)]:
        plus, minus = a.detach().clone(), a.detach().clone()
        plus[idx] += h
        minus[idx] -= h
        fd = float(fa.lpips(plus, b, data_range=1.0, dtype=torch.float64)
                   - fa.lpips(minus, b, data_range=1.0, dtype=torch.float64)) / (2 * h)
        assert abs(float(grad[idx]) - fd) <= 1e-6 * max(abs(fd), 1e-9) + 1e-12


def test_module_holds_no_parameters():
    """A frozen ImageNet trunk must not reach the caller's optimiser or their
    checkpoint -- hence buffers with persistent=False, not parameters."""
    crit = fa.LPIPS(data_range=1.0)
    assert list(crit.parameters()) == []
    assert dict(crit.state_dict()) == {}

    a, b = _pair()
    a, b = a.float(), b.float()
    assert torch.allclose(crit(a, b), crit.loss(a, b))
    assert torch.allclose(crit(a, b), fa.lpips(a, b, data_range=1.0))


def test_trunk_is_shared_across_instances():
    """Two modules, one 9.4 MiB copy of the weights."""
    from frame_analytics import perceptual

    fa.LPIPS(), fa.LPIPS()
    m0 = perceptual._load_net("alex", torch.device("cpu"), torch.float32)
    m1 = perceptual._load_net("alex", torch.device("cpu"), torch.float32)
    assert m0 is m1


def test_rejects_bad_input():
    a, b = _pair()
    a, b = a.float(), b.float()
    with pytest.raises(ValueError):
        fa.lpips(a, b, data_range=1.0, reduction="sum")
    with pytest.raises(ValueError):
        fa.lpips(a[:, :2], b[:, :2], data_range=1.0)
    with pytest.raises(ValueError):
        fa.lpips(a[..., :16, :16], b[..., :16, :16], data_range=1.0)
    with pytest.raises(ValueError):
        fa.lpips(a, b[:1], data_range=1.0)
    with pytest.raises(ValueError):
        fa.lpips(a, b, data_range=1.0, net="vgg")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_streaming_metrics_captures_lpips():
    sm = fa.StreamingMetrics((1, 3, 128, 128), device="cuda", dtype=torch.uint8,
                             metrics=("psnr", "ssim", "lpips"))
    g = torch.Generator().manual_seed(11)
    a = torch.randint(0, 256, (1, 3, 128, 128), generator=g, dtype=torch.uint8)
    b = torch.randint(0, 256, (1, 3, 128, 128), generator=g, dtype=torch.uint8)
    out = sm.update(a, b)
    assert set(out) == {"psnr", "ssim", "lpips"}
    want = float(fa.lpips(a.cuda(), b.cuda()))
    assert abs(out["lpips"] - want) < 1e-4
