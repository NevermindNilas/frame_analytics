"""HaarPSI (Reisenhofer et al. 2018).

Built on :mod:`frame_analytics.functional` helpers. Preserved spec points
from ``proposals/haarpsi.py``: NTSC RGB->YIQ, 2x2 box-mean subsample,
``2^-j`` box-difference taps (2 orientations x 3 scales; scale 3
weight-only), ``S(a,b,C)`` with ``C=30``, ``(S_I+S_Q)/2`` chroma term, and
the logistic ``l_a`` / inverse-logit-squared pooling with ``alpha=4.2``
and float64 accumulation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from . import functional as _fn

__all__ = ["haarpsi"]

_REF_C = 30.0
_REF_ALPHA = 4.2
_MIN_SIDE_SUB = 16
_MIN_SIDE_FULL = 8

_YIQ = ((0.299, 0.587, 0.114),
        (0.596, -0.274, -0.322),
        (0.211, -0.523, 0.312))


def _to_yiq(house: torch.Tensor):
    r, g, b = house[:, 0:1], house[:, 1:2], house[:, 2:3]
    m = _YIQ
    y = m[0][0] * r + m[0][1] * g + m[0][2] * b
    i = m[1][0] * r + m[1][1] * g + m[1][2] * b
    q = m[2][0] * r + m[2][1] * g + m[2][2] * b
    return y, i, q


_bank_cache: dict = {}


@torch.inference_mode(False)
def _box_bank(planes: int, device, dtype) -> torch.Tensor:
    key = ("box", planes, str(device), dtype)
    k = _bank_cache.get(key)
    if k is None:
        k = torch.full((planes, 1, 2, 2), 0.25, device=device, dtype=dtype)
        _bank_cache[key] = k
    return k


@torch.inference_mode(False)
def _box_same(device, dtype) -> torch.Tensor:
    key = ("boxsame", str(device), dtype)
    k = _bank_cache.get(key)
    if k is None:
        k = torch.full((1, 1, 2, 2), 0.25, device=device, dtype=dtype)
        _bank_cache[key] = k
    return k


@torch.inference_mode(False)
def _haar_row_bank(scale: int, device, dtype) -> torch.Tensor:
    key = ("hrow", scale, str(device), dtype)
    k = _bank_cache.get(key)
    if k is None:
        s = 2 ** scale
        diff = torch.cat([-torch.ones(s // 2), torch.ones(s // 2)]) * 2.0 ** (-scale)
        ones = torch.ones(s)
        k = torch.stack([ones, diff]).view(2, 1, 1, s).to(device=device, dtype=dtype)
        _bank_cache[key] = k
    return k


@torch.inference_mode(False)
def _haar_col_bank(scale: int, device, dtype) -> torch.Tensor:
    key = ("hcol", scale, str(device), dtype)
    k = _bank_cache.get(key)
    if k is None:
        s = 2 ** scale
        diff = torch.cat([-torch.ones(s // 2), torch.ones(s // 2)]) * 2.0 ** (-scale)
        ones = torch.ones(s)
        k = torch.stack([diff, ones]).view(2, 1, s, 1).to(device=device, dtype=dtype)
        _bank_cache[key] = k
    return k


def _conv_haar_scale(y: torch.Tensor, scale: int):
    s = 2 ** scale
    with _fn._no_autocast(y):
        rows = F.conv2d(y, _haar_row_bank(scale, y.device, y.dtype),
                        padding=(0, s // 2))[..., 1:]
        both = F.conv2d(rows, _haar_col_bank(scale, y.device, y.dtype),
                        padding=(s // 2, 0), groups=2)[..., 1:, :]
    return both[:, 0:1], both[:, 1:2]


def _conv_box_same(t: torch.Tensor) -> torch.Tensor:
    with _fn._no_autocast(t):
        return F.conv2d(t, _box_same(t.device, t.dtype), padding=1)[..., 1:, 1:]


def _subsample(t: torch.Tensor) -> torch.Tensor:
    if t.shape[-2] % 2 == 0 and t.shape[-1] % 2 == 0:
        return F.avg_pool2d(t, 2)
    p = F.pad(t, (0, 1, 0, 1))
    with _fn._no_autocast(p):
        return F.conv2d(p, _box_bank(t.shape[1], t.device, t.dtype),
                        stride=2, groups=t.shape[1])


def _sim(a: torch.Tensor, b: torch.Tensor, C: float) -> torch.Tensor:
    return (2.0 * a * b + C) / (a * a + b * b + C)


def _epilogue_gray(h1x, h1y, h2x, h2y, h3x, h3y,
                   v1x, v1y, v2x, v2y, v3x, v3y,
                   C: float, alpha: float) -> torch.Tensor:
    lsh = (_sim(h1x.abs(), h1y.abs(), C) + _sim(h2x.abs(), h2y.abs(), C)) / 2
    lsv = (_sim(v1x.abs(), v1y.abs(), C) + _sim(v2x.abs(), v2y.abs(), C)) / 2
    wh = torch.maximum(h3x.abs(), h3y.abs())
    wv = torch.maximum(v3x.abs(), v3y.abs())
    sh, sv = torch.sigmoid(alpha * lsh), torch.sigmoid(alpha * lsv)
    num = ((sh * wh).sum(dim=(1, 2, 3), dtype=torch.float64)
           + (sv * wv).sum(dim=(1, 2, 3), dtype=torch.float64))
    den = ((wh + wv).sum(dim=(1, 2, 3), dtype=torch.float64))
    fb = torch.cat([sh, sv], dim=1).mean(dim=(1, 2, 3),
                                         dtype=torch.float64)
    m = torch.where(den > 0, num / den.clamp_min(1e-300), fb)
    return ((torch.log(m / (1.0 - m)) / alpha) ** 2).clamp(0.0, 1.0)


def _epilogue_rgb(h1x, h1y, h2x, h2y, h3x, h3y,
                  v1x, v1y, v2x, v2y, v3x, v3y,
                  ix, iy, qx, qy, C: float, alpha: float) -> torch.Tensor:
    lsh = (_sim(h1x.abs(), h1y.abs(), C) + _sim(h2x.abs(), h2y.abs(), C)) / 2
    lsv = (_sim(v1x.abs(), v1y.abs(), C) + _sim(v2x.abs(), v2y.abs(), C)) / 2
    wh = torch.maximum(h3x.abs(), h3y.abs())
    wv = torch.maximum(v3x.abs(), v3y.abs())
    lsc = (_sim(ix.abs(), iy.abs(), C) + _sim(qx.abs(), qy.abs(), C)) / 2
    wc = (wh + wv) / 2
    sh = torch.sigmoid(alpha * lsh)
    sv = torch.sigmoid(alpha * lsv)
    sc = torch.sigmoid(alpha * lsc)
    num = ((sh * wh).sum(dim=(1, 2, 3), dtype=torch.float64)
           + (sv * wv).sum(dim=(1, 2, 3), dtype=torch.float64)
           + (sc * wc).sum(dim=(1, 2, 3), dtype=torch.float64))
    den = (wh + wv + wc).sum(dim=(1, 2, 3), dtype=torch.float64)
    fb = torch.cat([sh, sv, sc], dim=1).mean(dim=(1, 2, 3),
                                             dtype=torch.float64)
    m = torch.where(den > 0, num / den.clamp_min(1e-300), fb)
    return ((torch.log(m / (1.0 - m)) / alpha) ** 2).clamp(0.0, 1.0)


def haarpsi(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    subsample: bool = True,
    C: float = _REF_C,
    alpha: float = _REF_ALPHA,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
) -> torch.Tensor:
    """Haar wavelet-based perceptual similarity index (higher is better).

    ``luma``/``crop_border`` follow :mod:`frame_analytics.functional`:
    the crop is applied first; ``luma`` projects 3-channel input to luma
    (forcing the grayscale path) instead of the YIQ colour path.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    if not float(alpha) > 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if not float(C) > 0:
        raise ValueError(f"C must be > 0, got {C}")

    x4, y4, L, _ = _fn._prep(x, y, None, crop_border, data_range)
    if not L > 0:
        raise ValueError(f"data_range must be > 0, got {L}")
    if x4.shape[1] not in (1, 3):
        raise ValueError(f"expected 1 or 3 channels, got {x4.shape[1]}")
    wdt = _fn._work_dtype(x4, dtype)
    if luma not in (None, False) and x4.shape[1] == 3:
        x4, y4 = _fn._apply_luma(x4, y4, _fn._resolve_luma(luma), L, wdt)
    c = x4.shape[1]
    h0, w0 = x4.shape[-2], x4.shape[-1]
    floor = _MIN_SIDE_SUB if subsample else _MIN_SIDE_FULL
    if min(h0, w0) < floor:
        raise ValueError(
            f"image {(h0, w0)} is smaller than the HaarPSI floor {floor}x{floor} "
            f"(subsample={subsample})")
    rgb = c == 3

    with _fn._no_autocast(x4):
        house = torch.cat([x4, y4], dim=0).to(wdt) * (255.0 / L)
        if subsample:
            house = _subsample(house)
        n2 = house.shape[0]
        n = n2 // 2
        if rgb:
            Y, I, Q = _to_yiq(house)
        else:
            Y, I, Q = house, None, None

        (h1, v1) = _conv_haar_scale(Y, 1)
        (h2, v2) = _conv_haar_scale(Y, 2)
        (h3, v3) = _conv_haar_scale(Y, 3)
        h1x, v1x, h2x, v2x, h3x, v3x = h1[:n], v1[:n], h2[:n], v2[:n], h3[:n], v3[:n]
        h1y, v1y, h2y, v2y, h3y, v3y = h1[n:], v1[n:], h2[n:], v2[n:], h3[n:], v3[n:]

        if rgb:
            chroma = _conv_box_same(torch.cat([I, Q], dim=0))
            ix, iy, qx, qy = (chroma[0:n], chroma[n:2 * n],
                              chroma[2 * n:3 * n], chroma[3 * n:4 * n])
            epi = _fn._maybe_compile(_epilogue_rgb, "haarpsi:epi:rgb")
            per_image = epi(h1x, h1y, h2x, h2y, h3x, h3y,
                            v1x, v1y, v2x, v2y, v3x, v3y,
                            ix, iy, qx, qy, float(C), float(alpha))
        else:
            epi = _fn._maybe_compile(_epilogue_gray, "haarpsi:epi:gray")
            per_image = epi(h1x, h1y, h2x, h2y, h3x, h3y,
                            v1x, v1y, v2x, v2y, v3x, v3y,
                            float(C), float(alpha))

    return per_image if reduction == "none" else per_image.mean()
