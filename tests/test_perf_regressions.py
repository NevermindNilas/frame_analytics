"""Regressions found by adversarial review of the 2026-11 performance pass.

Each test pins a case the fused native paths once got wrong: functorch /
forward-mode AD, CUDA grid limits, empty tensors, too-small GMS inputs, and
concurrent CPU calls from several Python threads.
"""

import sys
import threading
import warnings
from pathlib import Path

import pytest
import torch
import torch.autograd.forward_ad as fwAD

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_analytics.functional as F  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_LOSSES = {
    "mse": F.mse,
    "l1": F.l1,
    "charbonnier": F.charbonnier,
    "huber": F.huber,
    "gms": lambda a, b: F.gms(a, b, data_range=1.0),
    "ssim": lambda a, b: F.ssim(a, b, data_range=1.0),
}


@cuda
@pytest.mark.parametrize("name", list(_LOSSES))
def test_functorch_grad_matches_autograd(name):
    fn = _LOSSES[name]
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.rand(2, 3, 32, 32, generator=g).cuda()
    y = torch.rand(2, 3, 32, 32, generator=g).cuda()
    xr = x.clone().requires_grad_()
    fn(xr, y).backward()
    got = torch.func.grad(lambda a: fn(a, y))(x)
    torch.testing.assert_close(got, xr.grad, rtol=1e-4, atol=1e-7)
    per = torch.func.vmap(torch.func.grad(lambda a, b: fn(a[None], b[None])))(x, y)
    assert per.shape == x.shape and torch.isfinite(per).all()


@cuda
@pytest.mark.parametrize("name", list(_LOSSES))
def test_forward_ad_on_grad_requiring_input(name):
    fn = _LOSSES[name]
    x = torch.rand(1, 3, 32, 32, device="cuda", requires_grad=True)
    y = torch.rand(1, 3, 32, 32, device="cuda")
    with fwAD.dual_level():
        out = fn(fwAD.make_dual(x, torch.ones_like(x)), y)
        assert fwAD.unpack_dual(out).tangent is not None


@cuda
@pytest.mark.parametrize("fn", [F.mse, F.l1])
def test_pixel_grad_more_images_than_grid_y(fn):
    torch.manual_seed(0)
    x = torch.rand(70000, 1, 4, 4, device="cuda", requires_grad=True)
    y = torch.rand(70000, 1, 4, 4, device="cuda")
    v = fn(x, y)
    v.backward()
    ref = fn(x.detach().double(), y.double())
    assert abs(float(v.detach()) - float(ref)) < 1e-9
    per = fn(x, y, reduction="none")
    assert per.shape == (70000,)
    torch.testing.assert_close(per, fn(x.detach().double(), y.double(),
                                       reduction="none").to(per.dtype))


@cuda
def test_gms_grad_more_planes_than_grid_z():
    x = torch.rand(70000, 1, 8, 8, device="cuda", requires_grad=True)
    y = torch.rand(70000, 1, 8, 8, device="cuda")
    F.gms(x, y, data_range=1.0).backward()
    assert torch.isfinite(x.grad).all()


@cuda
@pytest.mark.parametrize("shape", [(0, 3, 8, 8), (2, 3, 0, 8)])
def test_pixel_grad_empty(shape):
    x = torch.rand(shape, device="cuda", requires_grad=True)
    y = torch.rand(shape, device="cuda")
    v = F.mse(x, y)
    v.backward()
    assert torch.isnan(v) and x.grad.shape == x.shape


@cuda
def test_gms_grad_too_small_raises_value_error():
    x = torch.rand(1, 1, 5, 5, device="cuda", requires_grad=True)
    y = torch.rand(1, 1, 5, 5, device="cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # no "native declined"
        with pytest.raises(ValueError, match="Prewitt"):
            F.gms(x, y, data_range=1.0, downsample=True)


@pytest.mark.parametrize("metric", ["mse", "ssim", "gmsd"])
def test_concurrent_cpu_calls_match_serial(metric):
    calls = {
        "mse": lambda a, b: F.mse(a, b, reduction="none"),
        "ssim": lambda a, b: F.ssim(a, b, data_range=1.0, reduction="none"),
        "gmsd": lambda a, b: F.gmsd(a, b, data_range=1.0, reduction="none"),
    }
    fn = calls[metric]
    g = torch.Generator().manual_seed(0)
    data = []
    for s in [(2, 3, 190, 177), (1, 3, 333, 640), (4, 1, 64, 300), (1, 1, 600, 23)]:
        x = torch.rand(s, generator=g)
        data.append((x, (x + 0.05 * torch.randn(s, generator=g)).clamp(0, 1)))
    serial = [fn(x, y) for x, y in data]
    errors = []

    def worker(k):
        try:
            for i in range(30):
                j = (k + i) % len(data)
                if not torch.equal(fn(*data[j]), serial[j]):
                    errors.append(j)
        except Exception as e:  # pragma: no cover - the failure being pinned
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# --------------------------------------------------------------------------- #
# metric modules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["flip", "scielab", "dss", "haarpsi"])
def test_plane_conv_metrics_under_torch_func(name):
    import importlib

    fn = getattr(importlib.import_module(f"frame_analytics.{name}"), name)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 3, 48, 48, generator=g, dtype=torch.float64)
    y = (x + 0.05 * torch.randn(x.shape, generator=g, dtype=torch.float64)).clamp(0, 1)
    got = torch.func.grad(lambda t: fn(x, t))(y)
    yy = y.clone().requires_grad_()
    (ref,) = torch.autograd.grad(fn(x, yy), yy)
    torch.testing.assert_close(got, ref)
    torch.func.vmap(torch.func.grad(lambda t: fn(x, t)))(torch.stack([y, y.flip(-1)]))


def test_plane_corr_transforms():
    from frame_analytics._metric_utils import _PlaneCorr

    torch.manual_seed(0)
    for groups, c, o in [(1, 1, 3), (3, 3, 3)]:
        w = torch.randn(o, 1, 5, 4, dtype=torch.float64)
        for pad in [(0, 0), (2, 1), (4, 3)]:
            x = torch.randn(2, c, 9, 8, dtype=torch.float64, requires_grad=True)

            def f(t):
                return _PlaneCorr.apply(t, w, pad, groups)

            assert torch.autograd.gradcheck(f, (x,))
            assert torch.autograd.gradgradcheck(f, (x,))
            assert torch.autograd.gradcheck(f, (x,), check_forward_ad=True,
                                            check_backward_ad=False)


def test_fsim_map_double_backward():
    from frame_analytics.fsim import fsim

    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 72, dtype=torch.float64)
    y = (x + 0.05 * torch.randn_like(x)).clamp(0, 1).requires_grad_(True)
    (g,) = torch.autograd.grad(fsim(x, y, return_map=True).mean(), y,
                               create_graph=True)
    (g * g).sum().backward()
    assert torch.isfinite(y.grad).all()


@cuda
def test_psnr_hvs_m_ignores_tf32():
    from frame_analytics.psnrhvs import psnr_hvs_m

    torch.manual_seed(0)
    x = torch.rand(1, 3, 256, 256, device="cuda")
    y = ((x + 0.01 * torch.randn_like(x)).clamp(0, 1) * 255).round() / 255
    ref = float(psnr_hvs_m(x.cpu().double(), y.cpu().double(), data_range=1.0))
    old = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        got = float(psnr_hvs_m(x, y, data_range=1.0))
    finally:
        torch.backends.cuda.matmul.fp32_precision = old
    assert abs(got - ref) < 1e-3
