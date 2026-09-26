"""SR-SIM (Zhang & Li, ICIP 2012) and SR-SIMc.

Built on :mod:`frame_analytics.functional` helpers. Preserved spec points
from ``proposals/srsim.py``: Scharr ``/16`` gradient magnitudes, ``C1=0.40``
saliency similarity, ``C2=225`` gradient similarity (0..255 units),
``alpha=0.5`` (``sqrt``) weighting with ``w = max(s1, s2)`` saliency pooling,
float64 accumulators, compiled epilogue with eager fallback.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import functional as _fn

__all__ = ["srsim", "srsimc", "spectral_residual_saliency"]

_C1 = 0.40
_C2_AT_255 = 225.0
_T3_AT_255 = 200.0
_T4_AT_255 = 200.0
_CHROMA_WEIGHT = 0.03

_SR_SCALE = 0.25
_SR_AVG = 3
_SR_GAU_SIZE = 10
_SR_GAU_SIGMA = 3.8
_SR_LOG_EPS = 1e-8

_YIQ_I = (0.595716, -0.274453, -0.321263)
_YIQ_Q = (0.211456, -0.522591, 0.311135)


def _rgb_to_iq(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    ci, cq = _YIQ_I, _YIQ_Q
    return (ci[0] * r + ci[1] * g + ci[2] * b,
            cq[0] * r + cq[1] * g + cq[2] * b)


def _box_downsample(x: torch.Tensor, f: int) -> torch.Tensor:
    if f <= 1:
        return x
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = (f - h % f) % f, (f - w % f) % f
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return F.avg_pool2d(x, f)


_scharr_cache: dict = {}


@torch.inference_mode(False)
def _scharr_pair(device, dtype) -> torch.Tensor:
    key = (str(device), dtype)
    k = _scharr_cache.get(key)
    if k is None:
        hx = torch.tensor([[3.0, 0.0, -3.0],
                           [10.0, 0.0, -10.0],
                           [3.0, 0.0, -3.0]]) / 16.0
        k = torch.stack([hx, hx.t()]).unsqueeze(1).to(device=device,
                                                      dtype=dtype).contiguous()
        _scharr_cache[key] = k
    return k


def _scharr_magnitudes(y: torch.Tensor) -> torch.Tensor:
    k = _scharr_pair(y.device, y.dtype)
    with _fn._no_autocast(y):
        g = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), k)
    return torch.sqrt(g[:, 0:1] * g[:, 0:1] + g[:, 1:2] * g[:, 1:2])


_sr_gauss_cache: dict = {}


@torch.inference_mode(False)
def _sr_gauss(device, dtype) -> torch.Tensor:
    key = (str(device), dtype)
    w = _sr_gauss_cache.get(key)
    if w is None:
        import math
        r = (_SR_GAU_SIZE - 1) / 2.0
        coords = torch.arange(_SR_GAU_SIZE, dtype=torch.float64) - r
        g = torch.exp(-(coords * coords) / (2.0 * _SR_GAU_SIGMA * _SR_GAU_SIGMA))
        w = (g / g.sum()).to(device=device, dtype=dtype).contiguous()
        _sr_gauss_cache[key] = w
    return w


def _separable_blur(x: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    c, k = x.shape[1], win.numel()
    p = k // 2
    wx = win.view(1, 1, 1, k).expand(c, 1, 1, k)
    wy = win.view(1, 1, k, 1).expand(c, 1, k, 1)
    with _fn._no_autocast(x):
        x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="replicate"), wx.to(x.dtype),
                     groups=c)
        x = F.conv2d(F.pad(x, (0, 0, p, p), mode="replicate"), wy.to(x.dtype),
                     groups=c)
    return x


def _spectral_core(small: torch.Tensor) -> torch.Tensor:
    small = small.contiguous()
    spec = torch.fft.fft2(small)
    logamp = torch.log(torch.abs(spec) + _SR_LOG_EPS)
    phase = torch.angle(spec)
    p = _SR_AVG // 2
    avg = F.avg_pool2d(F.pad(logamp, (p, p, p, p), mode="replicate"),
                       _SR_AVG, stride=1)
    return torch.abs(torch.fft.ifft2(torch.polar(torch.exp(logamp - avg),
                                                 phase))) ** 2


def _saliency_batched(img: torch.Tensor) -> torch.Tensor:
    _, _, h, w = img.shape
    hs = max(1, int(round(h * _SR_SCALE)))
    ws = max(1, int(round(w * _SR_SCALE)))
    mode = "bicubic" if min(hs, ws) >= 4 and min(h, w) >= 4 else "bilinear"
    small = F.interpolate(img, size=(hs, ws), mode=mode, align_corners=False)
    sal = _separable_blur(_spectral_core(small), _sr_gauss(img.device, img.dtype))
    mn = sal.amin(dim=(1, 2, 3), keepdim=True)
    mx = sal.amax(dim=(1, 2, 3), keepdim=True)
    sal = (sal - mn) / (mx - mn).clamp_min(1e-12)
    if (hs, ws) != (h, w):
        sal = F.interpolate(sal, size=(h, w), mode=mode, align_corners=False)
    return sal


def spectral_residual_saliency(img: torch.Tensor) -> torch.Tensor:
    """Spectral-residual saliency of a single-channel ``(N,1,H,W)`` map."""
    t = _fn._as_nchw(img)
    if t.ndim != 4 or t.shape[1] != 1:
        raise ValueError(f"expected (N,1,H,W), got {tuple(img.shape)}")
    return _saliency_batched(t)


def _pool_luma(s1, s2, g1, g2, c1: float, c2: float):
    s_vs = (2.0 * s1 * s2 + c1) / (s1 * s1 + s2 * s2 + c1)
    s_g = (2.0 * g1 * g2 + c2) / (g1 * g1 + g2 * g2 + c2)
    w = torch.maximum(s1, s2)
    num = (s_vs * torch.sqrt(s_g) * w).sum(dim=(1, 2, 3), dtype=torch.float64)
    den = w.sum(dim=(1, 2, 3), dtype=torch.float64)
    return num, den


def _pool_chroma(s1, s2, g1, g2, ix1, iy2, qx1, qy2,
                 c1: float, c2: float, t3: float, t4: float, lam: float):
    s_vs = (2.0 * s1 * s2 + c1) / (s1 * s1 + s2 * s2 + c1)
    s_g = (2.0 * g1 * g2 + c2) / (g1 * g1 + g2 * g2 + c2)
    w = torch.maximum(s1, s2)
    s_c = ((2.0 * ix1 * iy2 + t3) / (ix1 * ix1 + iy2 * iy2 + t3)
           * (2.0 * qx1 * qy2 + t4) / (qx1 * qx1 + qy2 * qy2 + t4))
    num = (s_vs * torch.sqrt(s_g) * torch.pow(s_c.clamp_min(0.0), lam)
           * w).sum(dim=(1, 2, 3), dtype=torch.float64)
    den = w.sum(dim=(1, 2, 3), dtype=torch.float64)
    return num, den


def _per_image_scores(x4: torch.Tensor, y4: torch.Tensor, L: float,
                      wdt: torch.dtype, downsample: bool,
                      chroma_weight: Optional[float]) -> torch.Tensor:
    xf = x4.to(wdt)
    yf = y4.to(wdt)
    color = x4.shape[1] == 3

    if color:
        lum_x = _fn.rgb_to_luma(xf, "bt601", data_range=L, dtype=wdt)
        lum_y = _fn.rgb_to_luma(yf, "bt601", data_range=L, dtype=wdt)
    else:
        lum_x = xf[:, 0:1]
        lum_y = yf[:, 0:1]

    if downsample:
        f = max(1, int(round(min(x4.shape[-2], x4.shape[-1]) / 256.0)))
        lum_x = _box_downsample(lum_x, f)
        lum_y = _box_downsample(lum_y, f)

    both = torch.cat([lum_x, lum_y], dim=0)
    s12 = _saliency_batched(both)
    g12 = _scharr_magnitudes(both)
    s1, s2 = s12[:lum_x.shape[0]], s12[lum_x.shape[0]:]
    g1, g2 = g12[:lum_x.shape[0]], g12[lum_x.shape[0]:]

    k = (L / 255.0) ** 2
    if chroma_weight is not None and color:
        ix1, qx1 = _rgb_to_iq(xf)
        iy2, qy2 = _rgb_to_iq(yf)
        if downsample:
            ix1, qx1 = _box_downsample(ix1, f), _box_downsample(qx1, f)
            iy2, qy2 = _box_downsample(iy2, f), _box_downsample(qy2, f)
        if ix1.shape[-2:] != lum_x.shape[-2:]:
            sz = lum_x.shape[-2:]
            ix1 = F.interpolate(ix1, size=sz, mode="bilinear", align_corners=False)
            qx1 = F.interpolate(qx1, size=sz, mode="bilinear", align_corners=False)
            iy2 = F.interpolate(iy2, size=sz, mode="bilinear", align_corners=False)
            qy2 = F.interpolate(qy2, size=sz, mode="bilinear", align_corners=False)
        epi = _fn._maybe_compile(_pool_chroma, "srsim:epi:chroma")
        num, den = epi(s1, s2, g1, g2, ix1, iy2, qx1, qy2,
                       _C1, _C2_AT_255 * k, _T3_AT_255 * k, _T4_AT_255 * k,
                       float(chroma_weight))
    else:
        epi = _fn._maybe_compile(_pool_luma, "srsim:epi:luma")
        num, den = epi(s1, s2, g1, g2, _C1, _C2_AT_255 * k)

    return torch.where(den > 0, num / den, torch.ones_like(num))


def _front(x, y, data_range, luma, crop_border, dtype):
    x4, y4, L, _ = _fn._prep(x, y, None, crop_border, data_range)
    if not L > 0:
        raise ValueError(f"data_range must be > 0, got {data_range!r}")
    if x4.shape[1] not in (1, 3):
        raise ValueError(f"expected 1 or 3 channels, got {x4.shape[1]}")
    wdt = _fn._work_dtype(x4, dtype)
    if luma not in (None, False) and x4.shape[1] == 3:
        x4, y4 = _fn._apply_luma(x4, y4, _fn._resolve_luma(luma), L, wdt)
    return x4, y4, L, wdt


def srsim(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
    downsample: bool = True,
) -> torch.Tensor:
    """Spectral-residual similarity (Zhang & Li 2012), luma only."""
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    x4, y4, L, wdt = _front(x, y, data_range, luma, crop_border, dtype)
    scores = _per_image_scores(x4, y4, L, wdt, downsample, None)
    return scores.mean() if reduction == "mean" else scores


def srsimc(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
    downsample: bool = True,
    chroma_weight: float = _CHROMA_WEIGHT,
) -> torch.Tensor:
    """SR-SIMc: :func:`srsim` with a YIQ-chroma factor.

    Grayscale inputs fall back to plain SR-SIM. ``luma`` forces the
    grayscale path (then the chroma term is inactive).
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    if chroma_weight < 0:
        raise ValueError(f"chroma_weight must be >= 0, got {chroma_weight}")
    x4, y4, L, wdt = _front(x, y, data_range, luma, crop_border, dtype)
    scores = _per_image_scores(x4, y4, L, wdt, downsample, float(chroma_weight))
    return scores.mean() if reduction == "mean" else scores
