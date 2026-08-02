"""SSIMULACRA 2 -- Sneyers' perceptual metric, as libjxl computes it.

Higher is better.  100 is a pixel-identical pair, ~90 is "visually lossless",
~50 is "clearly visible artefacts", and the score is unbounded below (a badly
broken pair lands well under zero).

What it actually is
-------------------
Six scales, three XYB channels, three error maps, two norms -- 108 sub-scores
weighted by a fitted vector and pushed through a cubic and a power law.  The
three error maps per scale/channel are:

``SSIM'``
    SSIM with the luminance denominator dropped.  The usual
    ``2*mu1*mu2 / (mu1^2 + mu2^2)`` weights errors in the darks more heavily,
    which only makes sense for linear luma -- and these values are neither
    linear nor luma, they are gamma-ish X/Y/B channels.
``ringing``
    the distorted image has an edge where the original is smooth.
``blurring``
    the original has an edge where the distorted image is smooth.

The 4-norm sits next to the mean for every one of them, which is what makes
the metric sensitive to *localised* damage: a mean cannot tell a small ugly
block apart from a slight haze spread over the frame.

Conventions this follows exactly
--------------------------------
* input is **sRGB-encoded RGB**; it is linearised with the exact piecewise
  curve, and the pyramid is built by box-downsampling in *linear* RGB, with
  the XYB conversion redone at every scale.  Downsampling an odd dimension
  repeats the last row/column, as ``Downsample()`` does with its
  ``min(ox*fx+ix, xsize-1)`` clamp.
* the blur is jxl's ``FastGaussian`` at sigma 1.5.  That is the recursive
  truncated-cosine filter of Charalampidis 2016, whose impulse response is
  *finite*: radius ``round(3.2795*sigma + 0.2546) = 5``, and the taps at
  exactly +-5 are zero because every omega_k*radius is an odd multiple of
  pi/2.  So the recursion is an 11-tap symmetric FIR and is implemented here
  as one -- see :func:`recursive_gaussian_taps`.  Out-of-bounds input is
  **zero**, not mirrored and not renormalised (libjxl reads 0 past the edge),
  so the blurred planes genuinely darken towards the border.  That is part of
  the metric; the reference in :mod:`frame_analytics.reference` runs the
  actual IIR recursion instead, which is what pins the equivalence.
* float32 maps and float64 accumulators, which is upstream's split too: the
  three error maps are computed in float, and ``SSIMMap``/``EdgeDiffMap`` sum
  them into ``double``.  So ``dtype`` here selects the precision of the
  *pipeline* -- colour, blur, moments, maps -- while the norms and the weighted
  sum are float64 either way.  It matters because ``E[x^2] - E[x]^2`` in
  float32 is a cancellation trap that cannot be fixed by centring the way
  :func:`frame_analytics.functional.ssim` does: ``num_m`` and the 0.55/0.42
  offsets tie the formula to the absolute values.  Pass ``dtype=torch.float64``
  to see how much that costs (it is ~1e-4 of a score point on natural images).

Score of fewer than 6 scales
----------------------------
Images below 8 px on a side after some halving get fewer scales, and upstream
then walks the *same* weight vector from the start -- so a 5-scale score reads
weights meant for scales 0..4 of channel X into channels Y and B.  It is a
quirk rather than a design, but it is what the reference implementation
computes, so it is reproduced here rather than corrected.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .functional import _maybe_compile, _no_autocast, _prep, _work_dtype

__all__ = ["ssimulacra2", "recursive_gaussian_taps", "SSIMULACRA2_WEIGHTS"]

# libjxl tools/ssimulacra2.cc
_C2 = 0.0009
_NUM_SCALES = 6

# lib/jxl/cms/opsin_params.h
_M = (
    (0.30, 1.0 - 0.078 - 0.30, 0.078),
    (0.23, 1.0 - 0.078 - 0.23, 0.078),
    (0.24342268924547819, 0.20476744424496821,
     1.0 - 0.24342268924547819 - 0.20476744424496821),
)
_OPSIN_BIAS = 0.0037930732552754493
_OPSIN_BIAS_CBRT = _OPSIN_BIAS ** (1.0 / 3.0)

# Msssim::Score(), in its own order: for each channel, for each scale, for each
# norm, three consecutive weights (ssim, ringing, blurring).
SSIMULACRA2_WEIGHTS = (
    0.0, 0.0007376606707406586, 0.0,
    0.0, 0.0007793481682867309, 0.0,
    0.0, 0.0004371155730107379, 0.0,
    1.1041726426657346, 0.00066284834129271, 0.00015231632783718752,
    0.0, 0.0016406437456599754, 0.0,
    1.8422455520539298, 11.441172603757666, 0.0,
    0.0007989109436015163, 0.000176816438078653, 0.0,
    1.8787594979546387, 10.94906990605142, 0.0,
    0.0007289346991508072, 0.9677937080626833, 0.0,
    0.00014003424285435884, 0.9981766977854967, 0.00031949755934435053,
    0.0004550992113792063, 0.0, 0.0,
    0.0013648766163243398, 0.0, 0.0,
    0.0, 0.0, 0.0,
    7.466890328078848, 0.0, 17.445833984131262,
    0.0006235601634041466, 0.0, 0.0,
    6.683678146179332, 0.00037724407979611296, 1.027889937768264,
    225.20515300849274, 0.0, 0.0,
    19.213238186143016, 0.0011401524586618361, 0.001237755635509985,
    176.39317598450694, 0.0, 0.0,
    24.43300999870476, 0.28520802612117757, 0.0004485436923833408,
    0.0, 0.0, 0.0,
    34.77906344483772, 44.835625328877896, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0008680556573291698, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0005313191874358747, 0.0,
    0.00016533814161379112, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0004179171803251336, 0.0017290828234722833, 0.0,
    0.0020827005846636437, 0.0, 0.0,
    8.826982764996862, 23.19243343998926, 0.0,
    95.1080498811086, 0.9863978034400682, 0.9834382792465353,
    0.0012286405048278493, 171.2667255897307, 0.9807858872435379,
    0.0, 0.0, 0.0,
    0.0005130064588990679, 0.0, 0.00010854057858411537,
)
assert len(SSIMULACRA2_WEIGHTS) == 108


# --------------------------------------------------------------------------- #
# the blur
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def recursive_gaussian_taps(sigma: float = 1.5) -> Tuple[float, ...]:
    """The impulse response of ``jxl::CreateRecursiveGaussian(sigma)``.

    Cached: the taps are handed to the compiled blur as a tuple of python
    floats, i.e. as part of its guard set, and there is no reason to redo a
    3x3 solve and a dozen trig calls per call to get the same eleven numbers.

    Charalampidis 2016, "Recursive Implementation of the Gaussian Filter Using
    Truncated Cosine Functions": the kernel is a sum of three cosines truncated
    at radius ``N = round(3.2795*sigma + 0.2546)``, and libjxl evaluates it with
    a marginally-stable IIR recursion rather than as taps.  Both are the same
    filter; taps are what a convolution wants.

    Returned as ``2N+1`` values summing to 1, computed in double exactly as
    ``CreateRecursiveGaussian`` computes its coefficients.
    """
    radius = round(3.2795 * sigma + 0.2546)
    pi_div_2r = math.pi / (2.0 * radius)
    omega = (pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r)

    p = (+1.0 / math.tan(0.5 * omega[0]),          # (37)
         -1.0 / math.tan(0.5 * omega[1]),
         +1.0 / math.tan(0.5 * omega[2]))
    r = (+p[0] * p[0] / math.sin(omega[0]),        # (44)
         -p[1] * p[1] / math.sin(omega[1]),
         +p[2] * p[2] / math.sin(omega[2]))
    rho = [math.exp(-0.5 * sigma * sigma * w * w) / radius for w in omega]  # (50)

    d13 = p[0] * r[1] - r[0] * p[1]                # (52)
    d35 = p[1] * r[2] - r[1] * p[2]
    d51 = p[2] * r[0] - r[2] * p[0]
    zeta_15 = d35 / d13
    zeta_35 = d51 / d13

    a = [[p[0], p[1], p[2]], [r[0], r[1], r[2]], [zeta_15, zeta_35, 1.0]]  # (56)
    gamma = [1.0, radius * radius - sigma * sigma,                         # (55)
             zeta_15 * rho[0] + zeta_35 * rho[1] + rho[2]]
    beta = _solve3(a, gamma)                                               # (53)

    # (39): the weights are normalised, so the taps sum to 1 by construction.
    if abs(beta[0] * p[0] + beta[1] * p[1] + beta[2] * p[2] - 1.0) > 1e-12:
        raise RuntimeError("recursive Gaussian solve did not normalise")

    return tuple(
        sum(beta[k] * math.cos(omega[k] * t) for k in range(3))
        for t in range(-radius, radius + 1)
    )


def _solve3(a, b):
    """3x3 solve by Cramer's rule -- keeps this module numpy-free."""
    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    d = det(a)
    if d == 0.0:
        raise RuntimeError("singular matrix in recursive Gaussian solve")
    out = []
    for col in range(3):
        m = [[b[row] if c == col else a[row][c] for c in range(3)]
             for row in range(3)]
        out.append(det(m) / d)
    return out


_blur_cache: dict = {}
_weight_cache: dict = {}


def _blur_window(sigma: float, device, dtype) -> torch.Tensor:
    """Cached tap vector on the right device.

    Cached for the reason every other window in this package is: building it
    per call is a host-to-device copy, which syncs and is illegal inside a CUDA
    graph capture.
    """
    key = (sigma, str(device), dtype)
    w = _blur_cache.get(key)
    if w is None:
        w = torch.tensor(recursive_gaussian_taps(sigma),
                         dtype=torch.float64).to(device=device, dtype=dtype)
        _blur_cache[key] = w
    return w


def _weights_tensor(device) -> torch.Tensor:
    key = str(device)
    w = _weight_cache.get(key)
    if w is None:
        w = torch.tensor(SSIMULACRA2_WEIGHTS, dtype=torch.float64, device=device)
        _weight_cache[key] = w
    return w


def _blur_h(x: torch.Tensor, taps) -> torch.Tensor:
    """One zero-padded horizontal pass, written as shifted adds.

    ``taps`` arrives as a tuple of python floats, so ``torch.compile`` bakes
    them in as literals and emits a single kernel that reads each input once
    and accumulates 11 neighbours in registers. That is 2.6x faster than
    handing cuDNN an 11x1 convolution over a 1-channel batch, which is what
    this used to do -- and on CUDA the two agree bit for bit, because both
    accumulate the taps in the same order.
    """
    k = len(taps)
    w = x.shape[-1]
    xp = F.pad(x, (k // 2, k // 2, 0, 0))
    out = taps[0] * xp[..., 0:w]
    for i in range(1, k):
        out = out + taps[i] * xp[..., i:i + w]
    return out


def _blur_v(x: torch.Tensor, taps) -> torch.Tensor:
    """The vertical half of the same pass."""
    k = len(taps)
    h = x.shape[-2]
    xp = F.pad(x, (0, 0, k // 2, k // 2))
    out = taps[0] * xp[..., 0:h, :]
    for i in range(1, k):
        out = out + taps[i] * xp[..., i:i + h, :]
    return out


def _blur_conv(planes: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """The same filter as two 1-D convolutions -- the uncompiled path.

    Eager shifted adds would materialise a full-resolution temporary per tap,
    so when there is no Inductor to fuse them this is the better shape: the
    channel axis folds into the batch so both passes are plain single-channel
    convolutions, as in :func:`frame_analytics.functional._sep_filter`.
    """
    n, c, h, w = planes.shape
    rad = win.numel() // 2
    flat = planes.reshape(n * c, 1, h, w)
    with _no_autocast(flat):
        out = F.conv2d(flat, win.view(1, 1, 1, -1), padding=(0, rad))
        out = F.conv2d(out, win.view(1, 1, -1, 1), padding=(rad, 0))
    return out.reshape(n, c, h, w)


def _blur(planes: torch.Tensor, win: torch.Tensor, taps) -> torch.Tensor:
    """Separable zero-padded blur over every plane, output same size as input.

    Zero padding is not a convenience here, it is the definition: libjxl's
    scan reads 0 outside the image and does not renormalise, so the blurred
    border is genuinely attenuated and the metric was tuned with that in it.
    """
    fh = _maybe_compile(_blur_h, "ssimu2:blurh", dynamic=False)
    fv = _maybe_compile(_blur_v, "ssimu2:blurv", dynamic=False)
    if fh.active and fv.active:
        return fv(fh(planes, taps), taps)
    return _blur_conv(planes, win)


# --------------------------------------------------------------------------- #
# colour
# --------------------------------------------------------------------------- #


def _srgb_to_linear(v: torch.Tensor) -> torch.Tensor:
    """The exact piecewise sRGB EOTF, on data already scaled to [0, 1]."""
    return torch.where(v <= 0.04045, v / 12.92,
                       ((v + 0.055) / 1.055).clamp_min(0.0).pow(2.4))


def _xyb_positive(lin: torch.Tensor) -> torch.Tensor:
    """Linear sRGB ``(N,3,H,W)`` -> the 0..1-ish XYB of ``MakePositiveXYB``.

    ``ToXYB`` at libjxl's default intensity target of 255 nits leaves the
    opsin matrix unscaled, so this is the plain absorbance + cube root, and
    then the three affine adjustments that put X, Y and B-Y into roughly the
    same 0..1 range -- which is what lets one C2 serve all three channels.
    """
    r = lin[:, 0:1]
    g = lin[:, 1:2]
    b = lin[:, 2:3]
    m0 = (_M[0][0] * r + _M[0][1] * g + _M[0][2] * b + _OPSIN_BIAS).clamp_min(0.0)
    m1 = (_M[1][0] * r + _M[1][1] * g + _M[1][2] * b + _OPSIN_BIAS).clamp_min(0.0)
    m2 = (_M[2][0] * r + _M[2][1] * g + _M[2][2] * b + _OPSIN_BIAS).clamp_min(0.0)
    m0 = m0.pow(1.0 / 3.0) - _OPSIN_BIAS_CBRT
    m1 = m1.pow(1.0 / 3.0) - _OPSIN_BIAS_CBRT
    m2 = m2.pow(1.0 / 3.0) - _OPSIN_BIAS_CBRT

    x = 0.5 * (m0 - m1)
    y = 0.5 * (m0 + m1)
    return torch.cat([x * 14.0 + 0.42, y + 0.01, (m2 - y) + 0.55], dim=1)


def _downsample2(lin: torch.Tensor) -> torch.Tensor:
    """2x2 box downsample with libjxl's edge clamp.

    ``Downsample()`` reads ``min(ox*2+ix, xsize-1)``, i.e. an odd trailing
    row/column is averaged with a repeat of itself rather than dropped, so the
    output is ``ceil(size/2)`` and not ``floor``.  Replicate-padding first and
    then taking a clean 2x2 mean is the same arithmetic.
    """
    h, w = lin.shape[-2], lin.shape[-1]
    if h % 2 or w % 2:
        lin = F.pad(lin, (0, w % 2, 0, h % 2), mode="replicate")
    return F.avg_pool2d(lin, 2)


# --------------------------------------------------------------------------- #
# the two error maps
# --------------------------------------------------------------------------- #


def _norms(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(mean, 4-norm)`` per image and channel, both float64 ``(N,3)``."""
    one = v.mean(dim=-1, dtype=torch.float64)
    q = v * v
    four = (q * q).mean(dim=-1, dtype=torch.float64).pow(0.25)
    return one, four


def _scale_maps(x1, x2, mu1, mu2, s11, s22, s12):
    """The six per-channel norms of one scale, as one ``(N, 3, 6)`` tensor.

    Takes its planes **flat**, ``(N, 3, H*W)``. Two reasons, both about the
    compiled kernel rather than the arithmetic: a reduction over one axis
    tiles better than one over two (0.38 ms against 0.55 at 1080p), and a
    single trailing dimension is the shape a dynamic-shape kernel handles
    well, which is what lets one compiled entry serve all six scales -- see
    the call site.

    One region: every intermediate is frame-shaped, and the three maps share
    ``mu1``/``mu2``, so splitting them would re-read the same planes 3 times.
    """
    # Upstream writes this as 1 - num_m * num_s / den_s with
    # num_m = 1 - (mu1-mu2)^2. Factoring the subtraction into the numerator --
    #   1 - (1-q)*ns/ds  ==  ((ds - ns) + q*ns) / ds
    # -- is algebraically the same and better in two ways. d is small on a good
    # match, so the original spends its precision computing 1 minus something
    # very close to 1; and on an identical pair ds and ns are bitwise equal and
    # q is exactly 0, so the numerator is exactly 0 and d is exactly 0 whatever
    # the divide does with it. That matters because Inductor's Triton backend
    # emits an approximate fp32 divide (a/a comes back 1 +- 1e-7), and this
    # metric's 100 - 10*s^0.628 has infinite slope at s = 0 -- written the
    # upstream way, an identical pair scores 99.99 on CUDA.
    v11 = s11 - mu1 * mu1
    v22 = s22 - mu2 * mu2
    v12 = s12 - mu1 * mu2
    q = (mu1 - mu2) * (mu1 - mu2)
    num_s = (v12 + v12) + _C2
    den_s = (v11 + v22) + _C2
    d = (((den_s - num_s) + q * num_s) / den_s).clamp_min(0.0)

    # (1 + |dist - blur(dist)|) / (1 + |orig - blur(orig)|) - 1, with the same
    # subtraction folded in: above zero the distorted image has structure the
    # original does not (ringing, banding, blocking), below zero the original
    # had structure that did not survive.
    e1 = (x1 - mu1).abs()
    e2 = (x2 - mu2).abs()
    e = (e2 - e1) / (1.0 + e1)
    art = e.clamp_min(0.0)
    lost = (-e).clamp_min(0.0)

    ssim1, ssim4 = _norms(d)
    art1, art4 = _norms(art)
    lost1, lost4 = _norms(lost)
    return torch.stack([ssim1, ssim4, art1, art4, lost1, lost4], dim=-1)


def ssimulacra2(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    crop_border: int = 0,
) -> torch.Tensor:
    """SSIMULACRA 2 (Sneyers 2022/2023). Higher is better; 100 is identical.

    Parameters
    ----------
    x, y
        Original and distorted, same shape, ``(3,H,W)`` or ``(N,3,H,W)``, any
        dtype.  **sRGB-encoded RGB** -- the metric linearises them itself, so
        handing it linear-light data silently scores something else.  A
        1-channel input is replicated to three, which is what upstream does
        with a grayscale image.  Both must be at least 8x8.
    data_range
        Input scale: 255 for integers, 1.0 for floats unless given.
    reduction
        ``"mean"`` -> scalar over the batch. ``"none"`` -> ``(N,)``.
    dtype
        Compute dtype, float32 by default -- which is what upstream uses and
        therefore what its published scores carry.  float64 is available and
        moves a typical score by ~1e-4.
    crop_border
        Drop this many pixels from every edge first.  Not a SSIMULACRA 2
        convention; it is here for parity with the rest of the package.

    Notes
    -----
    The order of arguments matters, unlike SSIM: the ringing and blurring maps
    are asymmetric, so ``ssimulacra2(a, b) != ssimulacra2(b, a)``.  The first
    argument is the original.

    Differentiable, but a poor loss as-is: the final ``100 - 10*s^0.628`` has
    an infinite derivative as the pair converges, the same failure mode
    :func:`frame_analytics.functional.gmsd` has.

    >>> import frame_analytics as fa
    >>> fa.ssimulacra2(orig, dist, data_range=255.0)
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")

    x4, y4, L, _ = _prep(x, y, None, crop_border, data_range)
    wdt = _work_dtype(x4, dtype)
    if wdt not in (torch.float32, torch.float64):
        raise ValueError(f"SSIMULACRA 2 needs a floating compute dtype, got {wdt}")

    c = x4.shape[1]
    if c == 1:
        x4 = x4.expand(-1, 3, -1, -1)
        y4 = y4.expand(-1, 3, -1, -1)
    elif c != 3:
        raise ValueError(f"SSIMULACRA 2 needs 1- or 3-channel RGB, got {c} channels")
    if min(x4.shape[-2], x4.shape[-1]) < 8:
        raise ValueError(
            f"SSIMULACRA 2 needs at least 8x8, got {tuple(x4.shape[-2:])}")

    lin1 = _srgb_to_linear(x4.to(wdt) * (1.0 / L))
    lin2 = _srgb_to_linear(y4.to(wdt) * (1.0 / L))
    win = _blur_window(1.5, x4.device, wdt)
    taps = recursive_gaussian_taps(1.5)
    # dynamic=False, and it is the difference between a compiled epilogue that
    # pays for itself and one that costs more than it saves. A pyramid puts six
    # shapes through the same code object on every call, and Dynamo's
    # automatic-dynamic marking is keyed on the code object, not on the
    # call site: left at the default, scale 1 re-specialises what scale 0 just
    # compiled, and the dynamic-shape reduction kernel that results runs the
    # 1080p epilogue in 7.7 ms where the static one takes 0.56.
    #
    # The cost is one Dynamo cache entry per shape, six per resolution against
    # its limit of 8 -- so a process that scores two different resolutions
    # trips the limit and drops back to eager (still correct, ~2.4x slower).
    # Video scoring, which is one resolution for the length of the run, is the
    # case being optimised for; sweep several resolutions in one process and
    # you will see the warning and the eager numbers.
    xyb = _maybe_compile(_xyb_positive, "ssimu2:xyb", dynamic=False)
    maps = _maybe_compile(_scale_maps, "ssimu2:maps", dynamic=False)

    per_scale = []
    prev_h, prev_w = lin1.shape[-2], lin1.shape[-1]
    for scale in range(_NUM_SCALES):
        # Upstream tests the *previous* scale's size, because it shrinks its
        # buffers after the check. So a scale is computed iff the one above it
        # was at least 8x8, which lets the last one be smaller than that.
        if prev_h < 8 or prev_w < 8:
            break
        if scale:
            lin1 = _downsample2(lin1)
            lin2 = _downsample2(lin2)

        x1 = xyb(lin1)
        x2 = xyb(lin2)

        # mu1, mu2, s11, s22, s12 from one blur over 15 packed planes rather
        # than five blurs of three.
        f = _blur(torch.cat([x1, x2, x1 * x1, x2 * x2, x1 * x2], dim=1),
                  win, taps)
        # Viewed as (N, 5, 3, H*W) rather than split on the channel axis: the
        # five groups are then plain views whatever the batch size, where
        # `f.split(3, 1)` hands the epilogue strided planes it would have to
        # copy before it could flatten them.
        fv = f.view(f.shape[0], 5, 3, -1)
        flat = x1.shape[0], 3, -1
        per_scale.append(maps(x1.reshape(*flat), x2.reshape(*flat),
                              fv[:, 0], fv[:, 1], fv[:, 2], fv[:, 3], fv[:, 4]))
        prev_h, prev_w = lin1.shape[-2], lin1.shape[-1]

    # Msssim::Score()'s order: channel-major, then scale, then norm, then the
    # three maps. Flattened into one (N, 108) dot product rather than 108
    # scalar multiplies, each of which would be its own kernel launch.
    cols = []
    for ch in range(3):
        for sc in per_scale:
            for norm in range(2):
                cols.append(sc[:, ch, norm])          # SSIM'
                cols.append(sc[:, ch, 2 + norm])      # ringing
                cols.append(sc[:, ch, 4 + norm])      # blurring
    raw = torch.stack(cols, dim=-1).abs()
    w = _weights_tensor(x4.device)[: raw.shape[-1]]
    s = (raw * w).sum(dim=-1)

    s = s * 0.9562382616834844
    s = (2.326765642916932 * s - 0.020884521182843837 * s * s
         + 6.248496625763138e-05 * s * s * s)
    # s <= 0 means the weighted error landed at or below zero, i.e. a perfect
    # match; the power law is undefined there and upstream returns 100 flat.
    score = torch.where(s > 0.0,
                        100.0 - 10.0 * s.clamp_min(0.0).pow(0.6276336467831387),
                        torch.full_like(s, 100.0))
    return score if reduction == "none" else score.mean()
