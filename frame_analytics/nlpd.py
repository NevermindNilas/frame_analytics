"""Normalized Laplacian Pyramid Distance (NLPD) -- Hepburn/RGB lineage.

Final package module (torch-only). Shared input front end comes from
:mod:`frame_analytics.functional` (:func:`_prep`, :func:`_apply_luma`,
:func:`_work_dtype`, :func:`_no_autocast`, :func:`_maybe_compile`); no
local duplicates. ``torch.compile`` uses the shared eager-fallback wrapper;
:func:`set_compile_enabled` is a passthrough to ``functional``.

Hepburn / RGB lineage (Laparra et al. 2016 with Hepburn et al. 2020 NLDL
weights): 5x5 Burt & Adelson low-pass, bilinear upsample + band split,
per-level divisive normalization with the six fitted 3x3 ``P_j`` filters
and ``sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]``,
per-level RMSE pooled with the L0.6 norm across levels. Lower is better,
0 for identical.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F

from . import functional as _functional
from .functional import (
    _apply_luma,
    _maybe_compile,
    _no_autocast,
    _prep,
    _work_dtype,
)

__all__ = ["nlpd", "effective_depth", "set_compile_enabled"]

_LAPLACIAN_1D = (0.05, 0.25, 0.4, 0.25, 0.05)

_HEPBURN_SIGMAS = (0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782)

_HEPBURN_DN_3X3 = (
    ((0.0, 0.1011, 0.0), (0.1493, 0.0, 0.1460), (0.0, 0.1015, 0.0)),
    ((0.0, 0.0757, 0.0), (0.1986, 0.0, 0.1846), (0.0, 0.0837, 0.0)),
    ((0.0, 0.0477, 0.0), (0.2138, 0.0, 0.2243), (0.0, 0.0467, 0.0)),
    ((0.0, 0.0, 0.0), (0.2503, 0.0, 0.2616), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.2598, 0.0, 0.2552), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.2215, 0.0, 0.0717), (0.0, 0.0, 0.0)),
)

_MAX_LEVELS = len(_HEPBURN_SIGMAS)
_NORM_ORD = 0.6
_MIN_BAND = 8


def set_compile_enabled(flag: bool) -> None:
    """Passthrough to :func:`frame_analytics.functional.set_compile_enabled`."""
    _functional.set_compile_enabled(flag)


_lap_cache: dict = {}
_dn_cache: dict = {}


@torch.inference_mode(False)
def _lap_filter(device, dtype: torch.dtype, dims: int) -> torch.Tensor:
    key = (str(device), str(dtype), int(dims))
    w = _lap_cache.get(key)
    if w is None:
        # Outer product in float64, THEN cast: float32-first rounding puts
        # the center/corners 1 ULP off Hepburn's float32 literal table.
        k1 = torch.tensor(_LAPLACIAN_1D, dtype=torch.float64)
        k2 = (k1[:, None] * k1[None, :]).view(1, 1, 5, 5)
        w = (k2.to(device=device, dtype=dtype)
             .expand(int(dims), 1, 5, 5).contiguous())
        _lap_cache[key] = w
    return w


@torch.inference_mode(False)
def _dn_filter(level: int, device, dtype: torch.dtype, dims: int) -> torch.Tensor:
    key = (int(level), str(device), str(dtype), int(dims))
    w = _dn_cache.get(key)
    if w is None:
        w = torch.tensor(_HEPBURN_DN_3X3[int(level)], device=device,
                         dtype=dtype).view(1, 1, 3, 3)
        w = w.expand(int(dims), 1, 3, 3).contiguous()
        _dn_cache[key] = w
    return w


def _pad_calc(h: int, w: int, filt: int, stride: int):
    out_h = math.ceil(h / stride)
    out_w = math.ceil(w / stride)
    pad_h = max((out_h - 1) * stride + filt - h, 0)
    pad_w = max((out_w - 1) * stride + filt - w, 0)
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    return [pad_left, pad_w - pad_left, pad_top, pad_h - pad_top]


def _pad_symmetric(x: torch.Tensor, padding) -> torch.Tensor:
    if not any(padding):
        return x
    l, r, t, b = padding
    h, w = x.shape[-2], x.shape[-1]
    if l < w and r < w and t < h and b < h:
        return F.pad(x, padding, mode="reflect")
    return F.pad(x, padding, mode="replicate")


def _conv_dw(x: torch.Tensor, w: torch.Tensor, stride: int) -> torch.Tensor:
    """Depthwise ``groups=C`` convolution with minimal symmetric padding."""
    k = w.shape[-1]
    h, w_ = x.shape[-2], x.shape[-1]
    xp = _pad_symmetric(x, _pad_calc(h, w_, k, stride))
    with _no_autocast(xp):
        return F.conv2d(xp, w, stride=stride, groups=x.shape[1])


def effective_depth(h: int, w: int, scales: int = 6) -> int:
    """Feasible pyramid levels for a ``h`` x ``w`` image (ceil-halving)."""
    want = min(int(scales), _MAX_LEVELS)
    s, ch, cw = 0, int(h), int(w)
    while s < want and min(ch, cw) >= _MIN_BAND:
        s += 1
        ch, cw = (ch + 1) // 2, (cw + 1) // 2
    return max(s, 1)


def _l06_pool(stacked):
    """L0.6 across levels: ``(sum_j e_j^0.6)^(1/0.6)`` over dim 0."""
    return stacked.pow(_NORM_ORD).sum(dim=0).pow(1.0 / _NORM_ORD)


def _dn_sq_diff(zx, zy, denx, deny):
    """Fused DN divide + pair difference + square (per-level epilogue)."""
    d = zx / denx - zy / deny
    return d * d


def nlpd(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    scales: int = 6,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
) -> torch.Tensor:
    """Normalized Laplacian Pyramid Distance, Hepburn/RGB lineage.

    Lower is better; bitwise-identical inputs give exactly ``0``.
    Per-channel RGB (no luma projection unless ``luma=`` requests one),
    per-level RMSE over ``(C,H,W)``, L0.6 norm across levels.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    if int(scales) < 1:
        raise ValueError(f"scales must be >= 1, got {scales}")

    x4, y4, L, mode = _prep(x, y, luma, crop_border, data_range)
    if L <= 0:
        raise ValueError(f"data_range must be > 0, got {data_range}")
    wdt = _work_dtype(x4, dtype)
    x4, y4 = _apply_luma(x4, y4, mode, L, wdt)

    n, c, h, w = x4.shape
    s = effective_depth(h, w, scales)
    if s < min(int(scales), _MAX_LEVELS):
        warnings.warn(
            f"nlpd: {h}x{w} fits {s} level(s) with bands >= {_MIN_BAND}px; "
            f"clamping requested scales={scales} to {s}",
            RuntimeWarning, stacklevel=2)

    xf = (x4.to(wdt) / L).contiguous()
    yf = (y4.to(wdt) / L).contiguous()
    with torch.no_grad():
        same = ((xf.reshape(n, -1) - yf.reshape(n, -1)).abs().max(dim=1)
                .values == 0)
    cur = torch.cat([xf, yf], dim=0)
    lap = _lap_filter(cur.device, wdt, c)
    epi = _maybe_compile(_dn_sq_diff, "nlpd:dn_sq_diff")

    levels: list[torch.Tensor] = []
    for j in range(s):
        lo = _conv_dw(cur, lap, stride=2)
        up = F.interpolate(lo, size=cur.shape[-2:], mode="bilinear",
                           align_corners=True)
        band = cur - _conv_dw(up, lap, stride=1)
        amp = _conv_dw(band.abs(), _dn_filter(j, cur.device, wdt, c), stride=1)
        den = amp + float(_HEPBURN_SIGMAS[j])
        zx, zy = band[:n], band[n:]
        dx, dy = den[:n], den[n:]
        sq = epi(zx, zy, dx, dy)
        mse = sq.reshape(n, -1).mean(dim=1, dtype=torch.float64)
        levels.append(mse.sqrt())
        cur = lo

    stacked = torch.stack(levels)
    per_image = _l06_pool(stacked)
    per_image = torch.where(same.to(per_image.device),
                            torch.zeros_like(per_image), per_image)
    return per_image.mean() if reduction == "mean" else per_image


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.rand(2, 3, 64, 64)
    b = torch.clamp(a + 0.05 * torch.randn_like(a), 0, 1)
    print(f"identical={float(nlpd(a, a))} dist={float(nlpd(a, b)):.4f}")
    assert float(nlpd(a, a)) == 0.0
    assert float(nlpd(a, b)) > 0.0
    assert tuple(nlpd(a, b, reduction='none').shape) == (2,)
    print("nlpd self-test: OK")
