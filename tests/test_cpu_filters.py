"""Contract checks for the CPU separable-convolution layout optimization."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frame_analytics as fa
from frame_analytics import functional as impl


def batch_folded(packed, win, padding=0):
    n, c, h, w = packed.shape
    k = win.numel()
    with impl._no_autocast(packed):
        out = F.conv2d(packed.reshape(n * c, 1, h, w),
                       win.reshape(1, 1, 1, k), padding=(0, padding))
        out = F.conv2d(out, win.reshape(1, 1, k, 1), padding=(padding, 0))
    return out.reshape(n, c, h - k + 1 + 2 * padding, w - k + 1 + 2 * padding)


@pytest.mark.parametrize("shape", [(1, 1, 15, 17), (2, 5, 17, 23), (1, 12, 8, 9),
                                  (1, 15, 31, 37), (2, 12, 129, 137)])
@pytest.mark.parametrize("padding,k", [(0, 3), (5, 11)])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "channels_last"])
def test_filter_values_layout_and_no_mutation(shape, padding, k, layout):
    g = torch.Generator().manual_seed(16)
    x = torch.rand(shape, generator=g)
    if layout == "strided":
        x = x.transpose(-1, -2)
    elif layout == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last)
    win = impl.gaussian_window_1d(k, 1.5, device=x.device, dtype=x.dtype)
    before, before_win = x.clone(), win.clone()
    expected = batch_folded(x, win, padding)
    actual = impl._separable_conv(x, win, padding)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)
    assert actual.is_contiguous()
    assert actual.dtype == x.dtype
    assert torch.equal(x, before) and torch.equal(win, before_win)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(actual, impl._separable_conv(x, win, padding))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_filter_without_onednn(dtype):
    x = torch.rand(2, 5, 129, 137, dtype=dtype)
    win = impl.gaussian_window_1d(3, 1.5, device=x.device, dtype=dtype)
    with torch.backends.mkldnn.flags(enabled=False):
        assert torch.equal(impl._sep_filter(x, win), batch_folded(x, win))


@pytest.mark.parametrize("padding", [0, 2])
def test_filter_first_and_second_derivatives(padding):
    g = torch.Generator().manual_seed(281)
    x = torch.rand(2, 5, 129, 137, generator=g, requires_grad=True)
    win = torch.rand(5, generator=g, requires_grad=True)

    def derivatives(fn):
        out = fn(x, win, padding)
        first = torch.autograd.grad(out.square().mean(), (x, win), create_graph=True)
        second = torch.autograd.grad(sum(v.square().sum() for v in first), (x, win))
        return first + second

    for got, want in zip(derivatives(impl._separable_conv), derivatives(batch_folded)):
        error = (got - want).abs().max() / want.abs().max().clamp_min(1e-12)
        assert float(error) < 2e-4


@pytest.mark.parametrize("metric", ["ssim", "ms_ssim", "ssimulacra2"])
@pytest.mark.parametrize("channels", [1, 3])
def test_public_scores_and_both_input_gradients(monkeypatch, metric, channels):
    g = torch.Generator().manual_seed(525)
    x = torch.rand(2, channels, 141, 157, generator=g).requires_grad_()
    y = (x.detach() + 0.03 * torch.randn(x.shape, generator=g)).clamp(0, 1).requires_grad_()
    kwargs = {"reduction": "none"}
    if metric != "ssimulacra2":
        kwargs["backend_hint"] = "torch"
    if metric == "ms_ssim":
        kwargs["weights"] = (0.4, 0.6)
    previous = impl._COMPILE_ENABLED
    fa.set_compile_enabled(False)
    try:
        call = getattr(fa, metric)
        actual = call(x, y, **kwargs)
        grads = torch.autograd.grad(actual.sum(), (x, y))
        monkeypatch.setattr(impl, "_separable_conv", batch_folded)
        import importlib
        s2 = importlib.import_module("frame_analytics.ssimulacra2")
        monkeypatch.setattr(s2, "_separable_conv", batch_folded)
        expected = call(x, y, **kwargs)
        expected_grads = torch.autograd.grad(expected.sum(), (x, y))
        # Much tighter than the metric's float64-reference accuracy gate.
        torch.testing.assert_close(actual, expected, rtol=0, atol=5e-6)
        for got, want in zip(grads, expected_grads):
            # SSIMULACRA 2's zero fourth norms have undefined derivatives
            # (notably on replicated grayscale). Preserve the baseline mask.
            assert torch.equal(torch.isnan(got), torch.isnan(want))
            assert torch.equal(torch.isinf(got), torch.isinf(want))
            finite = torch.isfinite(want)
            if not finite.any():
                continue
            got, want = got[finite], want[finite]
            error = (got - want).abs().max() / want.abs().max().clamp_min(1e-12)
            assert float(error) < 2e-4
    finally:
        fa.set_compile_enabled(previous)


def test_public_ssim_map_and_double_backward(monkeypatch):
    x = torch.rand(1, 3, 129, 137, requires_grad=True)
    y = torch.rand_like(x, requires_grad=True)
    previous = impl._COMPILE_ENABLED
    fa.set_compile_enabled(False)
    try:
        def evaluate():
            out = fa.ssim(x, y, backend_hint="torch", return_map=True)
            first = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
            second = torch.autograd.grad(first.square().sum(), (x, y))
            return (out,) + second
        actual = evaluate()
        monkeypatch.setattr(impl, "_separable_conv", batch_folded)
        expected = evaluate()
        assert actual[0].shape == (1, 3, 119, 127)
        for got, want in zip(actual, expected):
            error = (got - want).abs().max() / want.abs().max().clamp_min(1e-12)
            assert float(error) < 2e-4
    finally:
        fa.set_compile_enabled(previous)
