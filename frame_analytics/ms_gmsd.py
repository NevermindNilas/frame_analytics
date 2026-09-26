"""Multi-scale GMSD (MS-GMSD, Zhang et al. 2017).

Final package module (torch-only). Shared input front end comes from
:mod:`frame_analytics.functional` (:func:`_prep`, :func:`_apply_luma`,
:func:`_work_dtype`, :func:`_no_autocast`, :func:`_maybe_compile`); no
local duplicates. ``torch.compile`` uses the shared eager-fallback wrapper;
:func:`set_compile_enabled` is a passthrough to ``functional``.

Method: per-scale GMSD over a 2x2 avg-pool pyramid fused by normalized RMS::

    ms_gmsd = sqrt(sum_i w_i * gmsd_i^2 / sum_i w_i)

Lower is better; 0 is identical. Default M = 4, ``MS_GMSD_WEIGHTS``.
"""

from __future__ import annotations

from typing import Optional, Sequence

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

__all__ = ["ms_gmsd", "MS_GMSD_WEIGHTS", "set_compile_enabled"]

# Default fusion weights for M = 4 scales (sum == 1, so the /sum(w)
# normalisation is a no-op for them).
MS_GMSD_WEIGHTS = (0.096, 0.596, 0.289, 0.019)


def set_compile_enabled(flag: bool) -> None:
    """Passthrough to :func:`frame_analytics.functional.set_compile_enabled`."""
    _functional.set_compile_enabled(flag)


# --------------------------------------------------------------------------- #
# cached constants (no H2D copy inside a timed call)
# --------------------------------------------------------------------------- #

_prewitt_cache: dict = {}
_fuse_weight_cache: dict = {}


@torch.inference_mode(False)
def _prewitt_pair(device, dtype) -> torch.Tensor:
    """``(2, 1, 3, 3)``: horizontal + vertical Prewitt taps, /3."""
    key = (device, dtype)
    k = _prewitt_cache.get(key)
    if k is not None:
        return k
    row = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64) / 3.0
    kx = row.expand(3, 3).contiguous()
    ky = kx.t().contiguous()
    k = torch.stack([kx, ky]).unsqueeze(1).to(device=device, dtype=dtype)
    _prewitt_cache[key] = k
    return k


@torch.inference_mode(False)
def _fuse_weights(w: tuple, device) -> torch.Tensor:
    """Fusion weights as a cached ``(M,)`` float64 device tensor."""
    key = (w, device)
    t = _fuse_weight_cache.get(key)
    if t is None:
        t = torch.tensor(w, dtype=torch.float64, device=device)
        _fuse_weight_cache[key] = t
    return t


# --------------------------------------------------------------------------- #
# GMS machinery: one batch-folded Prewitt conv + a compiled epilogue
# --------------------------------------------------------------------------- #

def _cast_pool2(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``avg_pool2d`` over a dtype conversion, fused into a single kernel."""
    return F.avg_pool2d(x.to(dtype), 2)


def _gms_grads(xf: torch.Tensor, yf: torch.Tensor) -> torch.Tensor:
    """Stacked Prewitt responses of both images, ``(2*N*C, 2, H-2, W-2)``."""
    n, c, h, w = xf.shape
    if h < 3 or w < 3:
        raise ValueError(f"image {h}x{w} is smaller than the 3x3 Prewitt window")
    k = _prewitt_pair(xf.device, xf.dtype)
    both = torch.cat([xf, yf], dim=0).reshape(2 * n * c, 1, h, w)
    with _no_autocast(both):
        return F.conv2d(both, k)


def _gmsd_epilogue(g: torch.Tensor, n: int, T: float, eps: float) -> torch.Tensor:
    """Per-image GMS variance from stacked grads ``g`` (``(2*N*C, 2, h, w)``).

    Gradient-magnitude roots, GMS quotient and float64 variance reduction
    in one compiled unit. Map widened *before* the variance (``var`` runs
    Welford, no cancelling subtraction).
    """
    nc = g.shape[0] // 2
    ga, gb = g[:nc], g[nc:]
    g1 = torch.sqrt((ga * ga).sum(dim=1) + eps)
    g2 = torch.sqrt((gb * gb).sum(dim=1) + eps)
    q = (2.0 * g1 * g2 + T) / (g1 * g1 + g2 * g2 + T)
    f = q.reshape(n, -1).double()
    return f.var(dim=1, correction=0).clamp_min(0.0)


# --------------------------------------------------------------------------- #
# MS-GMSD
# --------------------------------------------------------------------------- #

def ms_gmsd(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    T: Optional[float] = None,
    eps: Optional[float] = None,
    downsample: bool = True,
    weights: Optional[Sequence[float]] = None,
    num_scales: Optional[int] = None,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
) -> torch.Tensor:
    """Multi-scale GMSD (Zhang et al. 2017). Lower is better; 0 is identical.

    Per-scale GMSD over a 2x2 avg-pool pyramid, fused by normalized RMS
    pooling: ``sqrt(sum_i w_i * gmsd_i^2 / sum_i w_i)``.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")

    w = tuple(float(v) for v in (weights if weights is not None else MS_GMSD_WEIGHTS))
    if not w:
        raise ValueError("weights must not be empty")
    if any(v < 0 for v in w):
        raise ValueError(f"fusion weights must be >= 0, got {w!r}")
    if num_scales is None:
        num_scales = len(w)
    elif num_scales != len(w):
        raise ValueError(
            f"num_scales={num_scales} disagrees with len(weights)={len(w)}")
    if num_scales < 1:
        raise ValueError(f"num_scales must be >= 1, got {num_scales}")
    wsum = sum(w)
    if wsum <= 0:
        raise ValueError("fusion weights must not all be zero")

    x4, y4, L, mode = _prep(x, y, luma, crop_border, data_range)
    wdt = _work_dtype(x4, dtype)
    x4, y4 = _apply_luma(x4, y4, mode, L, wdt)
    n = x4.shape[0]
    Tv = float(T) if T is not None else 170.0 * (L / 255.0) ** 2
    ev = float(eps) if eps is not None else (1e-6 * L) ** 2

    hh, ww = x4.shape[-2], x4.shape[-1]
    div = (2 if downsample else 1) * (1 << (num_scales - 1))
    if min(hh, ww) < 3 * div:
        raise ValueError(
            f"image {(hh, ww)} is too small for {num_scales} scales "
            f"(downsample={downsample}); needs at least {3 * div}px on a side")

    pool = _maybe_compile(_cast_pool2, "ms_gmsd:cast_pool2")
    epi = _maybe_compile(_gmsd_epilogue, "ms_gmsd:epi")

    if downsample:
        if x4.dtype != wdt:
            cx, cy = pool(x4, wdt), pool(y4, wdt)
        else:
            cx, cy = F.avg_pool2d(x4.to(wdt), 2), F.avg_pool2d(y4.to(wdt), 2)
    else:
        cx, cy = x4.to(wdt), y4.to(wdt)
    if cx.device.type == "cpu" and cx.shape[1] > 1:
        cx = cx.to(memory_format=torch.channels_last)
        cy = cy.to(memory_format=torch.channels_last)

    per_scale = []
    for i in range(num_scales):
        if i:
            cx, cy = F.avg_pool2d(cx, 2), F.avg_pool2d(cy, 2)
        g = _gms_grads(cx, cy)
        per_scale.append(epi(g, n, Tv, ev))
    stacked = torch.stack(per_scale)  # (M, N) float64
    # Fuse variances directly; sqrt(var) followed by squaring produces NaN
    # gradients when a scale contains just one sample (the 48px minimum).
    variance = (stacked * _fuse_weights(w, stacked.device)
                .view(-1, 1)).sum(dim=0).div(wsum)
    fused = _functional._safe_sqrt(variance)
    return fused if reduction == "none" else fused.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 3, 128, 128)
    y = (x + 0.05 * torch.randn(2, 3, 128, 128)).clamp(0, 1)
    from . import functional as _fa

    a = ms_gmsd(x, y, data_range=1.0, weights=(1, 0, 0, 0))
    b = _fa.gmsd(x, y, data_range=1.0, downsample=True)
    print(f"scale0==gmsd diff={float((a - b).abs()):.3e}")
    assert torch.allclose(a.double(), b.double(), atol=1e-9, rtol=1e-6)
    full = ms_gmsd(x, y, data_range=1.0)
    print(f"default 4-scale fused={float(full):.6f}")
    assert torch.isfinite(full) and float(full) >= 0.0
    per = ms_gmsd(x, y, data_range=1.0, reduction="none")
    assert per.shape == (2,)
    print("ms_gmsd self-test: OK")
