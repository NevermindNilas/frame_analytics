"""SSIMULACRA 2 contract tests.

The substantive accuracy check lives in ``tests/validate.py``: the torch path
against a float64 transcription that runs libjxl's *recursive* Gaussian rather
than the equivalent FIR taps. What is here is the surface -- shapes, dtypes,
reductions, and the two properties that are easy to get quietly wrong
(asymmetry, and 100 on an identical pair).

Absolute agreement with libjxl was checked out of band against the reference
port's own test vector (``rust-av/ssimulacra2``, tank_source/tank_distorted):
that port scores 17.385, this package scores 17.507, and running libjxl's
recursion in float64 instead of float32 gives 17.50983 -- i.e. the gap is the
canonical implementation's float32 IIR drifting, not a disagreement about the
metric. The port's own test allows +-0.25 for exactly that reason.
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


def _rgb_pair(h=64, w=72, noise=6.0, seed=11):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 128 + 60 * np.sin(xx / 9.0) * np.cos(yy / 7.0)
    a = np.stack([np.clip(base + 30 * np.sin(xx / 23.0), 0, 255),
                  np.clip(base * 0.8 + 40, 0, 255),
                  np.clip(base * 1.1 - 10 + 20 * np.cos(yy / 17.0), 0, 255)]).round()
    b = np.clip(a + noise * rng.standard_normal(a.shape), 0, 255).round()
    return a, b


@pytest.mark.parametrize("dev", DEVICES)
def test_matches_reference(dev):
    a, b = _rgb_pair()
    want = ref.ssimulacra2_reference(a, b, 255.0)
    x = torch.as_tensor(a, dtype=torch.uint8, device=dev)
    y = torch.as_tensor(b, dtype=torch.uint8, device=dev)
    assert abs(float(fa.ssimulacra2(x, y, data_range=255.0)) - want) < 5e-3
    assert abs(float(fa.ssimulacra2(x, y, data_range=255.0,
                                    dtype=torch.float64)) - want) < 1e-8


@pytest.mark.parametrize("dev", DEVICES)
def test_identical_is_100(dev):
    a, _ = _rgb_pair()
    x = torch.as_tensor(a, dtype=torch.uint8, device=dev)
    assert float(fa.ssimulacra2(x, x, data_range=255.0)) == 100.0


@pytest.mark.parametrize("dev", DEVICES)
def test_asymmetric_in_its_arguments(dev):
    """Ringing and blurring are different maps, so the order is not free."""
    a, b = _rgb_pair(noise=12.0)
    x = torch.as_tensor(a, dtype=torch.uint8, device=dev)
    y = torch.as_tensor(b, dtype=torch.uint8, device=dev)
    assert float(fa.ssimulacra2(x, y, data_range=255.0)) != \
        float(fa.ssimulacra2(y, x, data_range=255.0))


@pytest.mark.parametrize("dev", DEVICES)
def test_monotone_in_distortion(dev):
    a, mild = _rgb_pair(noise=3.0, seed=5)
    _, harsh = _rgb_pair(noise=30.0, seed=5)
    x = torch.as_tensor(a, dtype=torch.uint8, device=dev)
    s_mild = float(fa.ssimulacra2(x, torch.as_tensor(mild, dtype=torch.uint8,
                                                     device=dev), data_range=255.0))
    s_harsh = float(fa.ssimulacra2(x, torch.as_tensor(harsh, dtype=torch.uint8,
                                                      device=dev), data_range=255.0))
    assert s_mild > s_harsh


@pytest.mark.parametrize("dev", DEVICES)
def test_batch_and_reduction(dev):
    a1, b1 = _rgb_pair(seed=1)
    a2, b2 = _rgb_pair(noise=20.0, seed=2)
    x = torch.as_tensor(np.stack([a1, a2]), dtype=torch.uint8, device=dev)
    y = torch.as_tensor(np.stack([b1, b2]), dtype=torch.uint8, device=dev)
    per = fa.ssimulacra2(x, y, data_range=255.0, reduction="none")
    assert per.shape == (2,)
    assert abs(float(fa.ssimulacra2(x, y, data_range=255.0))
               - float(per.mean())) < 1e-9
    # Each batch element is scored independently. Bit-identical in float64;
    # in float32 the CPU backend vectorises a batch of two differently from a
    # batch of one, which is worth ~1e-5 of a score point -- four orders below
    # what the metric resolves, but not zero.
    solo = float(fa.ssimulacra2(x[0], y[0], data_range=255.0))
    assert abs(solo - float(per[0])) < 1e-3
    solo64 = float(fa.ssimulacra2(x[0], y[0], data_range=255.0,
                                  dtype=torch.float64))
    per64 = fa.ssimulacra2(x, y, data_range=255.0, reduction="none",
                           dtype=torch.float64)
    assert solo64 == float(per64[0])


@pytest.mark.parametrize("dev", DEVICES)
def test_float_input_matches_uint8(dev):
    a, b = _rgb_pair()
    u8 = (torch.as_tensor(a, dtype=torch.uint8, device=dev),
          torch.as_tensor(b, dtype=torch.uint8, device=dev))
    f32 = (torch.as_tensor(a / 255.0, dtype=torch.float32, device=dev),
           torch.as_tensor(b / 255.0, dtype=torch.float32, device=dev))
    v_u8 = float(fa.ssimulacra2(*u8, data_range=255.0))
    v_f32 = float(fa.ssimulacra2(*f32))          # data_range defaults to 1.0
    # not bit-identical: the float tensors already carry a/255 rounded to
    # float32, which is a different input, worth ~1e-6 of a score point on its
    # own before the float32 pipeline adds its own ~2e-4
    assert abs(v_u8 - v_f32) < 2e-3


@pytest.mark.parametrize("dev", DEVICES)
def test_module_matches_function(dev):
    a, b = _rgb_pair()
    x = torch.as_tensor(a, dtype=torch.uint8, device=dev)
    y = torch.as_tensor(b, dtype=torch.uint8, device=dev)
    m = fa.SSIMULACRA2(data_range=255.0)
    assert float(m(x, y)) == float(fa.ssimulacra2(x, y, data_range=255.0))


@pytest.mark.parametrize("dev", DEVICES)
def test_grayscale_is_replicated(dev):
    a, b = _rgb_pair()
    g_a, g_b = a[0:1], b[0:1]
    x = torch.as_tensor(g_a, dtype=torch.uint8, device=dev)
    y = torch.as_tensor(g_b, dtype=torch.uint8, device=dev)
    want = ref.ssimulacra2_reference(np.repeat(g_a, 3, axis=0),
                                     np.repeat(g_b, 3, axis=0), 255.0)
    # float32 is looser here than on colour input: on gray, X = (m0 - m1)/2 is
    # a difference of two nearly equal cube roots, so the X channel's error
    # maps start from a cancelled quantity. float64 has no such trouble.
    assert abs(float(fa.ssimulacra2(x, y, data_range=255.0)) - want) < 3e-2
    assert abs(float(fa.ssimulacra2(x, y, data_range=255.0,
                                    dtype=torch.float64)) - want) < 1e-8


def test_rejects_bad_input():
    a, b = _rgb_pair()
    x = torch.as_tensor(a, dtype=torch.uint8)
    y = torch.as_tensor(b, dtype=torch.uint8)
    with pytest.raises(ValueError):
        fa.ssimulacra2(x, y, data_range=255.0, reduction="sum")
    with pytest.raises(ValueError):                 # 8x8 floor
        fa.ssimulacra2(x[:, :7], y[:, :7], data_range=255.0)
    with pytest.raises(ValueError):                 # 2-channel is not RGB
        fa.ssimulacra2(x[:2], y[:2], data_range=255.0)
    with pytest.raises(ValueError):                 # shape mismatch
        fa.ssimulacra2(x, y[:, :-1], data_range=255.0)


def test_blur_taps_are_the_recursive_gaussian():
    """The FIR the fast path convolves with is libjxl's recursion, exactly.

    Marching the IIR over a delta is the cheapest possible statement of the
    claim the whole implementation rests on.
    """
    from frame_analytics.ssimulacra2 import recursive_gaussian_taps

    taps = np.array(recursive_gaussian_taps(1.5))
    radius, n2, d1 = ref._s2_recursive_gaussian(1.5)
    assert len(taps) == 2 * radius + 1
    assert abs(taps.sum() - 1.0) < 1e-12

    delta = np.zeros(64)
    delta[32] = 1.0
    got = ref._s2_blur_axis(delta.reshape(1, 1, 64), 2, (radius, n2, d1)).ravel()
    assert np.abs(got[32 - radius:32 + radius + 1] - taps).max() < 1e-12
    # and nothing outside the support: the truncated cosines cancel exactly
    assert np.abs(np.r_[got[:32 - radius], got[32 + radius + 1:]]).max() < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_autocast_does_not_move_the_metric(dt):
    """A measurement must not change because of the surrounding compute context.

    The blur is shifted adds by python-float literals rather than a
    convolution, so nothing in it is autocast-eligible -- but that is a
    property of how it is written, and rewriting it as a conv (which the
    uncompiled path still does) would quietly put the metric in fp16 inside
    any modern training loop.
    """
    torch.manual_seed(0)
    x = torch.rand(1, 3, 192, 192, device="cuda")
    y = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    plain = float(fa.ssimulacra2(x, y, data_range=1.0))
    with torch.autocast("cuda", dtype=dt):
        cast = float(fa.ssimulacra2(x, y, data_range=1.0))
    assert abs(cast - plain) <= 1e-6 * max(abs(plain), 1e-3)


def test_crop_border():
    a, b = _rgb_pair(h=80, w=88)
    x = torch.as_tensor(a, dtype=torch.uint8)
    y = torch.as_tensor(b, dtype=torch.uint8)
    want = float(fa.ssimulacra2(x[:, 4:-4, 4:-4], y[:, 4:-4, 4:-4], data_range=255.0))
    got = float(fa.ssimulacra2(x, y, data_range=255.0, crop_border=4))
    assert abs(got - want) < 1e-9
