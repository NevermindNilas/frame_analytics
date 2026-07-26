"""Ground-truth reference implementations.

Deliberately slow, deliberately obvious. These follow the source papers
literally and run in float64. Every fast kernel in this package is validated
against these.

References
----------
Wang, Bovik, Sheikh, Simoncelli, "Image Quality Assessment: From Error
Visibility to Structural Similarity", IEEE TIP 13(4), 2004.
Canonical implementation: ``ssim_index.m`` (Wang's original MATLAB release).

Wang, Simoncelli, Bovik, "Multiscale Structural Similarity for Image Quality
Assessment", Asilomar 2003. Canonical implementation: ``msssim.m``.

Xue, Zhang, Mou, Bovik, "Gradient Magnitude Similarity Deviation: A Highly
Efficient Perceptual Image Quality Index", IEEE TIP 23(2), 2014.

Zhang, Isola, Efros, Shechtman, Wang, "The Unreasonable Effectiveness of Deep
Features as a Perceptual Metric", CVPR 2018.  Canonical implementation:
``richzhang/PerceptualSimilarity``.  LPIPS is the one metric here whose
definition is a set of weights rather than a formula, so this transcription
pins the *forward*; agreement with the reference package itself is asserted
separately in ``tests/test_lpips.py``.

  window   = fspecial('gaussian', 11, 1.5)   (normalised to sum 1)
  C1       = (K1 * L)^2,  K1 = 0.01
  C2       = (K2 * L)^2,  K2 = 0.03
  mu_x     = filter2(window, x, 'valid')
  sigma_x2 = filter2(window, x.*x, 'valid') - mu_x.^2
  sigma_xy = filter2(window, x.*y, 'valid') - mu_x.*mu_y
  map      = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) /
             ((mu_x^2 + mu_y^2 + C1) * (sigma_x2 + sigma_y2 + C2))
  mssim    = mean(map)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "gaussian_window_2d",
    "mse_reference",
    "psnr_reference",
    "ssim_reference",
    "ms_ssim_reference",
    "gmsd_reference",
    "l1_reference",
    "charbonnier_reference",
    "huber_reference",
    "rgb_to_luma_reference",
    "lpips_reference",
    "load_lpips_weights",
    "LUMA_COEFFS",
    "MS_SSIM_WEIGHTS",
]

# Multi-scale weights from Wang et al. 2003, table 1 (beta_i = gamma_i).
MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

# (r, g, b, offset).  The offset is expressed as a fraction of the data range,
# so the whole conversion is scale-equivariant: it means the same thing for
# uint8 0..255 input and for float 0..1 input.
#
#   bt601   full-range luma, ITU-R BT.601 (the plain 0.299/0.587/0.114)
#   bt709   full-range luma, ITU-R BT.709 (HD primaries)
#   matlab  studio-range Y' of MATLAB's rgb2ycbcr, i.e. what BasicSR's
#           `test_y_channel=True` computes.  This is the convention behind
#           essentially every published Y-PSNR / Y-SSIM number in the
#           super-resolution literature.
LUMA_COEFFS = {
    "bt601": (0.299, 0.587, 0.114, 0.0),
    "bt709": (0.2126, 0.7152, 0.0722, 0.0),
    "matlab": (65.481 / 255.0, 128.553 / 255.0, 24.966 / 255.0, 16.0 / 255.0),
}


def gaussian_window_2d(win_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """``fspecial('gaussian', win_size, sigma)`` in float64, sum-normalised."""
    r = (win_size - 1) / 2.0
    coords = np.arange(win_size, dtype=np.float64) - r
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    h = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    # MATLAB's fspecial zeroes entries below eps*max before normalising.
    h[h < np.finfo(np.float64).eps * h.max()] = 0.0
    s = h.sum()
    if s != 0:
        h /= s
    return h


def _valid_correlate2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """'valid' correlation, the operation MATLAB's ``filter2`` performs."""
    kh, kw = kernel.shape
    h, w = img.shape
    oh, ow = h - kh + 1, w - kw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError(f"image {img.shape} smaller than window {kernel.shape}")
    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            k = kernel[i, j]
            if k != 0.0:
                out += k * img[i : i + oh, j : j + ow]
    return out


def mse_reference(x: np.ndarray, y: np.ndarray) -> float:
    d = x.astype(np.float64) - y.astype(np.float64)
    return float(np.mean(d * d))


def psnr_reference(x: np.ndarray, y: np.ndarray, data_range: float = 255.0) -> float:
    m = mse_reference(x, y)
    if m == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range * data_range) / m))


def _ssim_maps(x, y, data_range, win_size, sigma, K1, K2):
    """Return ``(ssim_map, cs_map)``.

    ``cs_map`` is the contrast-structure factor alone -- the second half of the
    SSIM product.  It is what multi-scale SSIM accumulates at every scale but
    the coarsest.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    w = gaussian_window_2d(win_size, sigma)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_x = _valid_correlate2d(x, w)
    mu_y = _valid_correlate2d(y, w)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_xx = _valid_correlate2d(x * x, w) - mu_xx
    sigma_yy = _valid_correlate2d(y * y, w) - mu_yy
    sigma_xy = _valid_correlate2d(x * y, w) - mu_xy

    cs_map = (2.0 * sigma_xy + C2) / (sigma_xx + sigma_yy + C2)
    l_map = (2.0 * mu_xy + C1) / (mu_xx + mu_yy + C1)
    return l_map * cs_map, cs_map


def ssim_reference(
    x: np.ndarray,
    y: np.ndarray,
    data_range: float = 255.0,
    win_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    return_map: bool = False,
):
    """Single-channel SSIM, exactly as ``ssim_index.m`` computes it."""
    ssim_map, _ = _ssim_maps(x, y, data_range, win_size, sigma, K1, K2)
    if return_map:
        return float(ssim_map.mean()), ssim_map
    return float(ssim_map.mean())


# --------------------------------------------------------------------------- #
# pixel losses
# --------------------------------------------------------------------------- #


def l1_reference(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x.astype(np.float64) - y.astype(np.float64))))


def charbonnier_reference(x: np.ndarray, y: np.ndarray, eps: float = 1e-3) -> float:
    """``mean sqrt(d^2 + eps^2)`` -- the LapSRN / Charbonnier penalty.

    Note the ``eps`` is squared inside the root.  Some implementations add a
    bare ``eps``; ours is the one where ``eps`` has the units of the residual,
    so the default is meaningful for 0..1 data as written.
    """
    d = x.astype(np.float64) - y.astype(np.float64)
    return float(np.mean(np.sqrt(d * d + float(eps) ** 2)))


def huber_reference(x: np.ndarray, y: np.ndarray, delta: float = 1.0) -> float:
    """Matches ``torch.nn.HuberLoss``: quadratic inside ``delta``, linear out."""
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    delta = float(delta)
    quad = 0.5 * d * d
    lin = delta * (d - 0.5 * delta)
    return float(np.mean(np.where(d <= delta, quad, lin)))


# --------------------------------------------------------------------------- #
# luma
# --------------------------------------------------------------------------- #


def rgb_to_luma_reference(img: np.ndarray, mode: str = "bt601",
                          data_range: float = 255.0) -> np.ndarray:
    """``(3, H, W)`` RGB -> ``(H, W)`` luma, in the same units as the input."""
    img = np.asarray(img, dtype=np.float64)
    if img.shape[0] != 3:
        raise ValueError(f"expected a 3-channel image, got {img.shape}")
    try:
        cr, cg, cb, off = LUMA_COEFFS[mode]
    except KeyError:
        raise ValueError(f"unknown luma mode {mode!r}; "
                         f"expected one of {sorted(LUMA_COEFFS)}") from None
    return cr * img[0] + cg * img[1] + cb * img[2] + off * float(data_range)


# --------------------------------------------------------------------------- #
# MS-SSIM
# --------------------------------------------------------------------------- #


def _avg_pool2(x: np.ndarray) -> np.ndarray:
    """Non-overlapping 2x2 mean; an odd trailing row/column is dropped.

    This is ``F.avg_pool2d(x, 2)``, which is what every PyTorch MS-SSIM does.
    Wang's ``msssim.m`` instead runs a 2x2 box filter with MATLAB's ``'same'``
    zero padding and then keeps the odd-indexed samples, which differs at the
    borders and by a half-pixel phase.  We follow the PyTorch convention.
    """
    h, w = x.shape[-2] // 2 * 2, x.shape[-1] // 2 * 2
    v = x[..., :h, :w]
    return v.reshape(*v.shape[:-2], h // 2, 2, w // 2, 2).mean(axis=(-3, -1))


def ms_ssim_reference(
    x: np.ndarray,
    y: np.ndarray,
    data_range: float = 255.0,
    win_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    weights: Sequence[float] = MS_SSIM_WEIGHTS,
) -> float:
    """Single-channel MS-SSIM (Wang et al. 2003).

    ``prod_j cs_j^{w_j}`` over every scale but the last, times the full SSIM of
    the coarsest scale raised to its own weight.  Negative factors are clamped
    to zero before exponentiation -- a fractional power of a negative number is
    not real, and every implementation in circulation clamps the same way.
    """
    weights = list(weights)
    cx = np.asarray(x, dtype=np.float64)
    cy = np.asarray(y, dtype=np.float64)
    out = 1.0
    for i, wi in enumerate(weights):
        if i:
            cx, cy = _avg_pool2(cx), _avg_pool2(cy)
        if min(cx.shape[-2:]) < win_size:
            raise ValueError(
                f"image too small for {len(weights)} scales: scale {i} is "
                f"{cx.shape[-2:]} against an {win_size}x{win_size} window"
            )
        s_map, cs_map = _ssim_maps(cx, cy, data_range, win_size, sigma, K1, K2)
        v = float(s_map.mean()) if i == len(weights) - 1 else float(cs_map.mean())
        out *= max(v, 0.0) ** wi
    return out


# --------------------------------------------------------------------------- #
# GMSD
# --------------------------------------------------------------------------- #

# Prewitt pair from the GMSD paper, scaled by 1/3 as in the released code.
_PREWITT_X = np.array([[1.0, 0.0, -1.0]] * 3, dtype=np.float64) / 3.0
_PREWITT_Y = _PREWITT_X.T.copy()


def gmsd_reference(
    x: np.ndarray,
    y: np.ndarray,
    data_range: float = 255.0,
    T: Optional[float] = None,
    downsample: bool = True,
    eps: float = 0.0,
) -> float:
    """Gradient magnitude similarity deviation (Xue et al. 2014). Lower = better.

    Two deliberate deviations from the released MATLAB, both documented in the
    package README:

    * the 2x downsample is a clean non-overlapping 2x2 mean (``avg_pool2d``)
      rather than MATLAB's ``conv2(...,'same')`` + odd-index subsample, which
      is half a pixel out of phase and zero-pads the border;
    * the Prewitt pair is applied with ``'valid'`` support rather than
      zero-padded ``'same'``.  Zero padding invents a full-amplitude edge all
      the way round the frame, and the score is a *standard deviation*, so
      those fake edges move the number.

    ``T`` defaults to the paper's 170 rescaled to the data range; it has the
    units of a squared gradient, hence the ``L^2`` scaling.

    ``eps`` is added inside the gradient-magnitude root. The paper has no such
    term, and at the default it shifts the score by ~1e-9 relative -- but it is
    what keeps ``d|g|/dg`` finite on a flat patch, so the fast paths always
    carry one and the reference has to be able to reproduce it exactly.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if T is None:
        T = 170.0 * (float(data_range) / 255.0) ** 2

    if downsample:
        x, y = _avg_pool2(x), _avg_pool2(y)

    gx1 = _valid_correlate2d(x, _PREWITT_X)
    gy1 = _valid_correlate2d(x, _PREWITT_Y)
    gx2 = _valid_correlate2d(y, _PREWITT_X)
    gy2 = _valid_correlate2d(y, _PREWITT_Y)
    g1 = np.sqrt(gx1 * gx1 + gy1 * gy1 + eps)
    g2 = np.sqrt(gx2 * gx2 + gy2 * gy2 + eps)

    q = (2.0 * g1 * g2 + T) / (g1 * g1 + g2 * g2 + T)
    return float(np.sqrt(((q - q.mean()) ** 2).mean()))


# --------------------------------------------------------------------------- #
# LPIPS (Zhang et al. 2018)
#
# Deliberately naive: an explicit tap-by-tap convolution and an explicit
# max-pool, both in float64. The point is not speed -- the gate runs it on
# 64x64 patches -- it is that this shares no code with the torch path, so a
# misplaced pool, a stride, or an epsilon on the wrong side of a square root
# shows up as a disagreement rather than as two implementations being wrong
# together.
# --------------------------------------------------------------------------- #

# The reference implementation's ScalingLayer: ImageNet mean/std expressed in
# [-1, 1] units.  Its normalize_tensor() adds eps to the *norm*, not to the
# sum of squares, which is a different function near zero.
_LPIPS_SHIFT = np.array([-0.030, -0.088, -0.188], dtype=np.float64).reshape(1, 3, 1, 1)
_LPIPS_SCALE = np.array([0.458, 0.448, 0.450], dtype=np.float64).reshape(1, 3, 1, 1)
_LPIPS_NORM_EPS = 1e-10


def load_lpips_weights(net: str = "alex") -> dict:
    """The shipped weight blob as float64 numpy arrays.

    Imports torch lazily: this module is otherwise numpy-only, and the blob is
    a torch archive because that is what both upstream projects publish.
    """
    import torch

    from .perceptual import lpips_weights_path

    path = lpips_weights_path(net)
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        blob = torch.load(path, map_location="cpu")
    return {
        "channels": list(blob["channels"]),
        "pool_before": list(blob["pool_before"]),
        "trunk": [
            {
                "weight": t["weight"].double().numpy(),
                "bias": t["bias"].double().numpy(),
                "stride": int(t["stride"]),
                "padding": int(t["padding"]),
            }
            for t in blob["trunk"]
        ],
        "lin": [w.double().numpy() for w in blob["lin"]],
    }


def _conv2d_reference(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                      stride: int, pad: int) -> np.ndarray:
    """(N,C,H,W) * (O,C,kh,kw) -> (N,O,oh,ow), summed tap by tap."""
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    kh, kw = w.shape[2], w.shape[3]
    oh = (x.shape[2] - kh) // stride + 1
    ow = (x.shape[3] - kw) // stride + 1
    if oh <= 0 or ow <= 0:
        raise ValueError(f"input {x.shape} too small for a {kh}x{kw} kernel")
    out = np.zeros((x.shape[0], w.shape[0], oh, ow), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            patch = x[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride]
            out += np.einsum("nchw,oc->nohw", patch, w[:, :, i, j])
    return out + b.reshape(1, -1, 1, 1)


def _maxpool2d_reference(x: np.ndarray, k: int = 3, stride: int = 2) -> np.ndarray:
    oh = (x.shape[2] - k) // stride + 1
    ow = (x.shape[3] - k) // stride + 1
    out = np.full((x.shape[0], x.shape[1], oh, ow), -np.inf, dtype=np.float64)
    for i in range(k):
        for j in range(k):
            np.maximum(out, x[:, :, i:i + stride * oh:stride,
                              j:j + stride * ow:stride], out=out)
    return out


def lpips_reference(x: np.ndarray, y: np.ndarray, data_range: float = 255.0,
                    weights: Optional[dict] = None) -> np.ndarray:
    """LPIPS per image, ``(N,)``. Lower is better; 0 is identical.

    ``x``/``y`` are ``(N,3,H,W)`` (or ``(3,H,W)``) in ``[0, data_range]``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim == 3:
        x, y = x[None], y[None]
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expected (N,3,H,W), got {x.shape}")
    if weights is None:
        weights = load_lpips_weights()

    def taps(img: np.ndarray) -> list:
        h = (img * (2.0 / float(data_range)) - 1.0 - _LPIPS_SHIFT) / _LPIPS_SCALE
        out = []
        for layer, pool in zip(weights["trunk"], weights["pool_before"]):
            if pool:
                h = _maxpool2d_reference(h)
            h = _conv2d_reference(h, layer["weight"], layer["bias"],
                                  layer["stride"], layer["padding"])
            h = np.maximum(h, 0.0)
            out.append(h)
        return out

    total = np.zeros(x.shape[0], dtype=np.float64)
    for f0, f1, lin in zip(taps(x), taps(y), weights["lin"]):
        n0 = f0 / (np.sqrt((f0 ** 2).sum(axis=1, keepdims=True)) + _LPIPS_NORM_EPS)
        n1 = f1 / (np.sqrt((f1 ** 2).sum(axis=1, keepdims=True)) + _LPIPS_NORM_EPS)
        d = (n0 - n1) ** 2
        total += np.einsum("nchw,oc->n", d, lin[:, :, 0, 0]) / (d.shape[2] * d.shape[3])
    return total
