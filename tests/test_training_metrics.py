"""Accuracy and contract tests for the training-oriented metrics.

MS-SSIM, GMSD/GMS, the three pixel losses, and the luma / border-crop
reporting conventions. Every value is checked against the float64 reference in
:mod:`frame_analytics.reference`, on both backends and both devices, because
"the fast path and the slow path agree" is the only claim that actually
matters here.
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

TOL = 5e-6


def _pair(h=200, w=232, seed=11, noise=7.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.clip(120 + 50 * np.sin(xx / 11.0) * np.cos(yy / 13.0)
                + 12 * rng.standard_normal((h, w)), 0, 255).round()
    b = np.clip(a + noise * rng.standard_normal((h, w)), 0, 255).round()
    return a, b


def _rgb_pair(h=140, w=176, seed=12, noise=9.0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (3, h, w)).astype(np.float64)
    b = np.clip(a + noise * rng.standard_normal((3, h, w)), 0, 255).round()
    return a, b


def _t(arr, dev, dt=torch.uint8):
    return torch.as_tensor(arr, dtype=dt, device=dev)


# --------------------------------------------------------------------------- #
# MS-SSIM
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("dt", [torch.uint8, torch.float32])
def test_ms_ssim_matches_reference(dev, hint, dt):
    a, b = _pair()
    want = ref.ms_ssim_reference(a, b, 255.0)
    got = float(fa.ms_ssim(_t(a, dev, dt)[None, None], _t(b, dev, dt)[None, None],
                           data_range=255.0, backend_hint=hint))
    assert abs(got - want) < TOL


@pytest.mark.parametrize("dev", DEVICES)
def test_ms_ssim_agrees_with_pytorch_msssim(dev):
    """An independently written implementation of the same convention."""
    pm = pytest.importorskip("pytorch_msssim")
    a, b = _pair(256, 256)
    x = _t(a, dev, torch.float64)[None, None]
    y = _t(b, dev, torch.float64)[None, None]
    theirs = float(pm.ms_ssim(x, y, data_range=255.0, win_size=11, win_sigma=1.5))
    ours = float(fa.ms_ssim(x.float(), y.float(), data_range=255.0))
    assert abs(ours - theirs) < TOL


@pytest.mark.parametrize("dev", DEVICES)
def test_ms_ssim_identity_and_batch(dev):
    a, b = _pair()
    x = _t(a, dev)[None, None]
    # five scales of float32 arithmetic: the identity is exact in real numbers
    # but not in the machine, so this is a rounding bound, not a tolerance
    assert abs(float(fa.ms_ssim(x, x, data_range=255.0)) - 1.0) < 1e-7

    r1 = ref.ms_ssim_reference(a, b, 255.0)
    a2, b2 = _pair(seed=13, noise=25.0)
    r2 = ref.ms_ssim_reference(a2, b2, 255.0)
    xa = _t(np.stack([a, a2])[:, None], dev)
    xb = _t(np.stack([b, b2])[:, None], dev)
    per = fa.ms_ssim(xa, xb, data_range=255.0, reduction="none").double().cpu()
    assert per.shape == (2,)
    assert abs(float(per[0]) - r1) < TOL
    assert abs(float(per[1]) - r2) < TOL
    assert abs(float(fa.ms_ssim(xa, xb, data_range=255.0)) - 0.5 * (r1 + r2)) < TOL


@pytest.mark.parametrize("dev", DEVICES)
def test_ms_ssim_channels_are_averaged_after_the_product(dev):
    """Each channel gets its own weighted product; the mean comes last.

    This is the pytorch-msssim convention. Averaging the *scales* across
    channels first would give a different (and less standard) number, so it is
    worth pinning.
    """
    a1, b1 = _pair(seed=21)
    a2, b2 = _pair(seed=22, noise=30.0)
    r = 0.5 * (ref.ms_ssim_reference(a1, b1, 255.0)
               + ref.ms_ssim_reference(a2, b2, 255.0))
    x = _t(np.stack([a1, a2])[None], dev)
    y = _t(np.stack([b1, b2])[None], dev)
    assert abs(float(fa.ms_ssim(x, y, data_range=255.0)) - r) < TOL


def test_ms_ssim_rejects_images_too_small_for_the_pyramid():
    # 5 scales of halving against an 11x11 window needs 11 * 2^4 = 176 px.
    x = torch.zeros(1, 1, 175, 300)
    with pytest.raises(ValueError, match="too small"):
        fa.ms_ssim(x, x, data_range=1.0)
    fa.ms_ssim(torch.zeros(1, 1, 176, 300), torch.zeros(1, 1, 176, 300),
               data_range=1.0)


@pytest.mark.parametrize("dev", DEVICES)
def test_ms_ssim_custom_weights(dev):
    a, b = _pair()
    w = (0.4, 0.6)
    want = ref.ms_ssim_reference(a, b, 255.0, weights=w)
    got = float(fa.ms_ssim(_t(a, dev)[None, None], _t(b, dev)[None, None],
                           data_range=255.0, weights=w))
    assert abs(got - want) < TOL


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape", [(1, 1, 192, 192), (2, 3, 180, 200)])
@pytest.mark.parametrize("wrt", [("x",), ("y",), ("x", "y")])
def test_ms_ssim_native_backward_matches_autograd(shape, wrt):
    torch.manual_seed(0)
    x0 = torch.rand(*shape, device="cuda")
    y0 = (x0 + 0.05 * torch.randn_like(x0)).clamp(0, 1)

    def grads(hint):
        x = x0.clone().requires_grad_("x" in wrt)
        y = y0.clone().requires_grad_("y" in wrt)
        (1.0 - fa.ms_ssim(x, y, data_range=1.0, backend_hint=hint)).backward()
        return x.grad, y.grad

    for r, n in zip(grads("torch"), grads("auto")):
        assert (r is None) == (n is None)
        if r is None:
            continue
        rel = (r - n).abs().max().item() / max(r.abs().max().item(), 1e-12)
        assert rel < 2e-4, f"relative gradient error {rel:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ms_ssim_backward_matches_finite_differences():
    """Independent check against the float64 forward, not against autograd."""
    rng = np.random.default_rng(2)
    h = w = 180
    xa = rng.random((h, w))
    ya = np.clip(xa + 0.05 * rng.standard_normal((h, w)), 0, 1)
    x = torch.as_tensor(xa, dtype=torch.float32,
                        device="cuda")[None, None].requires_grad_(True)
    y = torch.as_tensor(ya, dtype=torch.float32, device="cuda")[None, None]
    fa.ms_ssim(x, y, data_range=1.0).backward()
    g = x.grad[0, 0].double().cpu().numpy()

    eps = 1e-4
    for (i, j) in [(40, 60), (90, 90), (120, 33)]:
        xp = xa.copy(); xp[i, j] += eps
        xm = xa.copy(); xm[i, j] -= eps
        fd = (ref.ms_ssim_reference(xp, ya, 1.0)
              - ref.ms_ssim_reference(xm, ya, 1.0)) / (2 * eps)
        assert abs(fd - g[i, j]) / max(abs(fd), 1e-12) < 1e-3


# --------------------------------------------------------------------------- #
# GMSD / GMS
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("down", [True, False])
def test_gmsd_matches_reference(dev, hint, down):
    a, b = _pair()
    eps = (1e-6 * 255.0) ** 2
    want = ref.gmsd_reference(a, b, 255.0, downsample=down, eps=eps)
    got = float(fa.gmsd(_t(a, dev)[None, None], _t(b, dev)[None, None],
                        data_range=255.0, downsample=down, backend_hint=hint))
    assert abs(got - want) < TOL


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("hint", HINTS)
def test_gmsd_odd_sizes(dev, hint):
    """Odd H/W: the 2x downsample drops the last row/column, as avg_pool2d does."""
    a, b = _pair(h=157, w=203, seed=17)
    eps = (1e-6 * 255.0) ** 2
    want = ref.gmsd_reference(a, b, 255.0, eps=eps)
    got = float(fa.gmsd(_t(a, dev)[None, None], _t(b, dev)[None, None],
                        data_range=255.0, backend_hint=hint))
    assert abs(got - want) < TOL


@pytest.mark.parametrize("dev", DEVICES)
def test_gmsd_identity_is_zero_and_gms_is_one(dev):
    a, _ = _pair()
    x = _t(a, dev)[None, None]
    # the map is identically 1 in exact arithmetic; what is left is the
    # float32 spread of q around it, which is what the deviation then measures
    assert float(fa.gmsd(x, x, data_range=255.0)) < 1e-6
    assert abs(float(fa.gms(x, x, data_range=255.0)) - 1.0) < 1e-7


@pytest.mark.parametrize("dev", DEVICES)
def test_gmsd_map_reduces_to_the_scalars(dev):
    a, b = _pair()
    x, y = _t(a, dev)[None, None], _t(b, dev)[None, None]
    m = fa.gmsd(x, y, data_range=255.0, return_map=True)
    assert m.shape == (1, 1, a.shape[0] // 2 - 2, a.shape[1] // 2 - 2)
    assert abs(float(m.double().std(unbiased=False))
               - float(fa.gmsd(x, y, data_range=255.0))) < 1e-6
    assert abs(float(m.double().mean())
               - float(fa.gms(x, y, data_range=255.0))) < 1e-6


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("hint", HINTS)
def test_gmsd_batch_and_channels(dev, hint):
    a1, b1 = _pair(seed=31)
    a2, b2 = _pair(seed=32, noise=28.0)
    eps = (1e-6 * 255.0) ** 2
    r1 = ref.gmsd_reference(a1, b1, 255.0, eps=eps)
    r2 = ref.gmsd_reference(a2, b2, 255.0, eps=eps)
    xa = _t(np.stack([a1, a2])[:, None], dev)
    xb = _t(np.stack([b1, b2])[:, None], dev)
    per = fa.gmsd(xa, xb, data_range=255.0, reduction="none",
                  backend_hint=hint).double().cpu()
    assert per.shape == (2,)
    assert abs(float(per[0]) - r1) < TOL
    assert abs(float(per[1]) - r2) < TOL


@pytest.mark.parametrize("dev", DEVICES)
def test_gms_loss_gradient_is_finite_at_the_optimum(dev):
    """`1 - gms` must survive pred == target; the deviation's does not."""
    x = torch.rand(1, 1, 64, 64, device=dev, requires_grad=True)
    y = x.detach().clone()
    loss = fa.GMSD(data_range=1.0).loss(x, y)
    loss.backward()
    assert torch.isfinite(x.grad).all()


# --------------------------------------------------------------------------- #
# pixel losses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("dt", [torch.uint8, torch.float32, torch.float64])
def test_pixel_losses_match_reference(dev, hint, dt):
    a, b = _pair()
    x, y = _t(a, dev, dt)[None, None], _t(b, dev, dt)[None, None]
    checks = [
        (fa.l1(x, y, backend_hint=hint), ref.l1_reference(a, b), 1e-9),
        (fa.charbonnier(x, y, backend_hint=hint),
         ref.charbonnier_reference(a, b), 1e-6),
        (fa.huber(x, y, backend_hint=hint), ref.huber_reference(a, b), 1e-9),
        (fa.charbonnier(x, y, eps=3.0, backend_hint=hint),
         ref.charbonnier_reference(a, b, eps=3.0), 1e-6),
        (fa.huber(x, y, delta=4.0, backend_hint=hint),
         ref.huber_reference(a, b, delta=4.0), 1e-9),
    ]
    for got, want, tol in checks:
        assert abs(float(got) - want) < tol * max(1.0, abs(want))


@pytest.mark.parametrize("dev", DEVICES)
def test_huber_matches_torch(dev):
    a, b = _pair()
    x = _t(a, dev, torch.float32)[None, None]
    y = _t(b, dev, torch.float32)[None, None]
    for delta in (0.5, 1.0, 17.0):
        theirs = float(torch.nn.functional.huber_loss(x, y, delta=delta))
        assert abs(float(fa.huber(x, y, delta=delta)) - theirs) < 1e-4


@pytest.mark.parametrize("dev", DEVICES)
def test_l1_matches_torch(dev):
    a, b = _pair()
    x = _t(a, dev, torch.float32)[None, None]
    y = _t(b, dev, torch.float32)[None, None]
    assert abs(float(fa.l1(x, y))
               - float(torch.nn.functional.l1_loss(x, y))) < 1e-4


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("fn", [fa.l1, fa.charbonnier, fa.huber])
def test_pixel_losses_reduction_none(dev, fn):
    x = torch.randint(0, 256, (5, 3, 40, 40), dtype=torch.uint8, device=dev)
    y = torch.randint(0, 256, (5, 3, 40, 40), dtype=torch.uint8, device=dev)
    per = fn(x, y, reduction="none")
    assert per.shape == (5,)
    assert abs(float(per.mean()) - float(fn(x, y))) < 1e-9


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("fn", [fa.l1, fa.charbonnier, fa.huber])
def test_pixel_losses_are_differentiable(dev, fn):
    x = torch.rand(1, 3, 32, 32, device=dev, requires_grad=True)
    y = torch.rand(1, 3, 32, 32, device=dev)
    fn(x, y).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0


@pytest.mark.parametrize("dev", DEVICES)
def test_pixel_losses_zero_on_identity(dev):
    x = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8, device=dev)
    assert float(fa.l1(x, x)) == 0.0
    assert float(fa.huber(x, x)) == 0.0
    assert abs(float(fa.charbonnier(x, x, eps=0.25)) - 0.25) < 1e-9


def test_huber_rejects_bad_delta():
    x = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError):
        fa.huber(x, x, delta=0.0)


# --------------------------------------------------------------------------- #
# luma + crop_border
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("mode", ["bt601", "bt709", "matlab"])
@pytest.mark.parametrize("crop", [0, 4])
def test_luma_and_crop_match_reference(dev, mode, crop):
    a, b = _rgb_pair()
    ya = ref.rgb_to_luma_reference(a, mode)
    yb = ref.rgb_to_luma_reference(b, mode)
    if crop:
        ya = ya[crop:-crop, crop:-crop]
        yb = yb[crop:-crop, crop:-crop]

    x, y = _t(a, dev)[None], _t(b, dev)[None]
    kw = dict(luma=mode, crop_border=crop)
    assert abs(float(fa.ssim(x, y, data_range=255.0, **kw))
               - ref.ssim_reference(ya, yb, 255.0)) < TOL
    assert abs(float(fa.psnr(x, y, data_range=255.0, **kw))
               - ref.psnr_reference(ya, yb, 255.0)) < 1e-5
    assert abs(float(fa.mse(x, y, **kw)) - ref.mse_reference(ya, yb)) < 1e-4
    assert abs(float(fa.l1(x, y, **kw)) - ref.l1_reference(ya, yb)) < 1e-4


@pytest.mark.parametrize("dev", DEVICES)
def test_luma_true_is_bt601(dev):
    a, b = _rgb_pair()
    x, y = _t(a, dev)[None], _t(b, dev)[None]
    assert (float(fa.psnr(x, y, data_range=255.0, luma=True))
            == float(fa.psnr(x, y, data_range=255.0, luma="bt601")))


def test_luma_rejects_bad_input():
    x = torch.zeros(1, 3, 32, 32)
    with pytest.raises(ValueError):
        fa.psnr(x, x, luma="rec2020")
    with pytest.raises(TypeError):
        fa.psnr(x, x, luma=3)
    with pytest.raises(ValueError):
        fa.psnr(torch.zeros(1, 2, 32, 32), torch.zeros(1, 2, 32, 32), luma=True)


def test_crop_border_validation():
    x = torch.zeros(1, 1, 32, 32)
    with pytest.raises(ValueError):
        fa.mse(x, x, crop_border=16)
    with pytest.raises(ValueError):
        fa.mse(x, x, crop_border=-1)


@pytest.mark.parametrize("dev", DEVICES)
def test_crop_border_equals_manual_slice(dev):
    a, b = _pair()
    x, y = _t(a, dev)[None, None], _t(b, dev)[None, None]
    for cb in (1, 3, 8):
        manual = float(fa.ssim(x[..., cb:-cb, cb:-cb].contiguous(),
                               y[..., cb:-cb, cb:-cb].contiguous(),
                               data_range=255.0))
        assert abs(float(fa.ssim(x, y, data_range=255.0, crop_border=cb))
                   - manual) < 1e-12


# --------------------------------------------------------------------------- #
# module wrappers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dev", DEVICES)
def test_modules_agree_with_functions(dev):
    a, b = _pair()
    x, y = _t(a, dev)[None, None], _t(b, dev)[None, None]
    pairs = [
        (fa.MSSSIM(data_range=255.0)(x, y), fa.ms_ssim(x, y, data_range=255.0)),
        (fa.GMSD(data_range=255.0)(x, y), fa.gmsd(x, y, data_range=255.0)),
        (fa.L1()(x, y), fa.l1(x, y)),
        (fa.Charbonnier(eps=2.0)(x, y), fa.charbonnier(x, y, eps=2.0)),
        (fa.Huber(delta=3.0)(x, y), fa.huber(x, y, delta=3.0)),
    ]
    for got, want in pairs:
        assert float(got) == float(want)

    assert abs(float(fa.MSSSIM(data_range=255.0).loss(x, y))
               - (1.0 - float(fa.ms_ssim(x, y, data_range=255.0)))) < 1e-12
    assert abs(float(fa.GMSD(data_range=255.0).loss(x, y))
               - (1.0 - float(fa.gms(x, y, data_range=255.0)))) < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("offset", [0, 1, 2, 3])
@pytest.mark.parametrize("fn", [fa.mse, fa.l1, fa.huber])
def test_unaligned_uint8_views(fn, offset):
    """A contiguous uint8 *slice* need not be 4-byte aligned.

    The vectorised kernel loads `uchar4`, and a misaligned load does not fail
    softly -- it faults, and the fault poisons the CUDA context for the rest of
    the process, so no Python-level fallback can rescue it.
    """
    n = 64 * 33
    a = torch.randint(0, 256, (n + 4,), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, (n + 4,), dtype=torch.uint8, device="cuda")
    x = a[offset:offset + n].view(1, 1, 33, 64)
    y = b[offset:offset + n].view(1, 1, 33, 64)
    assert x.is_contiguous() and x.data_ptr() % 4 == offset % 4
    got = float(fn(x, y))
    want = float(fn(x.contiguous().clone(), y.contiguous().clone(),
                    backend_hint="torch") if fn is not fa.mse
                 else ((x.double() - y.double()) ** 2).mean())
    assert abs(got - want) < 1e-9 * max(1.0, abs(want))


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("fn,kw,want", [
    ("mse", {}, 1e-12),
    ("huber", {}, 5e-13),
    ("l1", {}, 1e-6),
    ("charbonnier", {"eps": 0.0}, 1e-6),
])
def test_float64_input_is_not_narrowed_to_float32(dev, fn, kw, want):
    """A float64 pair with a 1e-6 residual has no float32 answer.

    The reduction kernels load at working precision, which is float32 for
    everything but float64 input; getting that wrong is invisible on uint8 data
    and a 10x error here.
    """
    x = torch.full((1, 1, 64, 64), 100.0, dtype=torch.float64, device=dev)
    y = x + 1e-6
    got = float(getattr(fa, fn)(x, y, **kw))
    assert abs(got - want) <= 1e-3 * want


@pytest.mark.parametrize("dev", DEVICES)
def test_no_grad_still_reaches_the_native_kernel(dev):
    """`requires_grad` on a tensor nobody will differentiate is not a reason
    to decline the fast path -- and it is the normal evaluation shape."""
    from frame_analytics import backend

    x = torch.rand(1, 1, 64, 64, device=dev, requires_grad=True)
    y = torch.rand(1, 1, 64, 64, device=dev)
    assert not backend._usable(x, y)
    with torch.no_grad():
        assert backend._usable(x, y)
    with torch.inference_mode():
        assert backend._usable(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fn", ["ssim", "ms_ssim"])
def test_double_backward_refuses_loudly_and_has_a_way_out(fn):
    """A second derivative through the fused kernel must not return zeros."""
    torch.manual_seed(0)
    x = torch.rand(1, 1, 192, 192, device="cuda", requires_grad=True)
    y = (x.detach() + 0.05 * torch.randn_like(x)).clamp(0, 1)
    kw = dict(data_range=1.0)
    call = getattr(fa, fn)

    with pytest.raises(RuntimeError, match="twice differentiable"):
        g, = torch.autograd.grad(call(x, y, **kw), x, create_graph=True)
        torch.autograd.grad(g.sum(), x)

    # ...and the escape hatch named in that message actually works
    fa.set_compile_enabled(False)
    try:
        g, = torch.autograd.grad(call(x, y, backend_hint="torch", **kw), x,
                                 create_graph=True)
        g2, = torch.autograd.grad(g.sum(), x)
        assert torch.isfinite(g2).all() and float(g2.abs().max()) > 0
    finally:
        fa.set_compile_enabled(True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fn", ["ssim", "ms_ssim", "gmsd", "gms"])
def test_autocast_does_not_move_the_metric(dt, fn):
    """A measurement must not change because of the surrounding compute context."""
    torch.manual_seed(0)
    x = torch.rand(1, 3, 192, 192, device="cuda", requires_grad=True)
    y = (x.detach() + 0.05 * torch.randn_like(x)).clamp(0, 1)
    call = getattr(fa, fn)
    plain = float(call(x, y, data_range=1.0))
    with torch.autocast("cuda", dtype=dt):
        cast = float(call(x, y, data_range=1.0))
    assert abs(cast - plain) <= 1e-6 * max(abs(plain), 1e-3)


@pytest.mark.parametrize("dev", DEVICES)
def test_backend_hint_contract(dev):
    """`torch` must never touch the extension; `native` must never fall back."""
    a, b = _pair()
    x, y = _t(a, dev)[None, None], _t(b, dev)[None, None]
    for fn in (fa.ms_ssim, fa.gmsd, fa.l1):
        kw = {"data_range": 255.0} if fn is not fa.l1 else {}
        assert fn(x, y, backend_hint="torch", **kw) is not None
    if not fa.backend_status()["available"]:
        return
    # float64 has no native kernel, so "native" must raise rather than lie
    xd, yd = x.double(), y.double()
    with pytest.raises(RuntimeError):
        fa.ms_ssim(xd, yd, data_range=255.0, backend_hint="native")
    # neither does the GMS map, and that must be just as loud
    for fn in (fa.gms, fa.gmsd):
        with pytest.raises(RuntimeError, match="native GMSD"):
            fn(x, y, data_range=255.0, return_map=True, backend_hint="native")
