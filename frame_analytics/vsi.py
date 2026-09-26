"""VSI: visual saliency-induced index (Zhang, Shen & Li, TIP 2014).

Built on :mod:`frame_analytics.functional` helpers. Preserved spec points
from ``proposals/vsi.py``: full SDSP saliency (256x256, Lab, log-Gabor,
centre + warm-colour bias), LMN opponent colour, Scharr ``/16`` gradient
modulus on decimated L, ``C_VS=1.27`` / ``C_GM=386`` / ``C_CH=130``,
``alpha=0.40`` / ``lambda=0.020`` with sign-aware real chroma power, and
``max(VS1, VS2)`` saliency weighting with float64 pooling.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F

from . import functional as _fn

__all__ = ["vsi", "sdsp_saliency"]

_C_VS = 1.27
_C_GM = 386.0
_C_CH = 130.0
_ALPHA = 0.40
_LAMBDA = 0.020
_LAM_COS = math.cos(math.pi * _LAMBDA)
_SALIENCY_SIZE = 256
_OMEGA0 = 0.0210
_SIGMA_F = 1.34
_SIGMA_D = 145.0
_SIGMA_C = 0.001
_LMN = ((0.06, 0.63, 0.27), (0.30, 0.04, -0.35), (0.34, -0.60, 0.17))


def _srgb_to_lab(rgb01: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]

    def _lin(c):
        return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    rl, gl, bl = _lin(r), _lin(g), _lin(b)
    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    eps, kappa = 0.008856, 903.3

    def _f(t, ref):
        t = t / ref
        return torch.where(t > eps, t.clamp_min(0) ** (1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = _f(X, 0.9642), _f(Y, 1.0), _f(Z, 0.8251)
    return torch.cat([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=1)


def _lmn(t: torch.Tensor):
    r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    m = _LMN
    return (m[0][0] * r + m[0][1] * g + m[0][2] * b,
            m[1][0] * r + m[1][1] * g + m[1][2] * b,
            m[2][0] * r + m[2][1] * g + m[2][2] * b)


_lg_cache: dict = {}
_sd_cache: dict = {}


@torch.inference_mode(False)
def _log_gabor(device, dtype) -> torch.Tensor:
    key = (str(device), dtype)
    lg = _lg_cache.get(key)
    if lg is None:
        S = _SALIENCY_SIZE
        v = (torch.arange(S, device=device, dtype=dtype) - S // 2) / S
        u1, u2 = torch.meshgrid(v, v, indexing="xy")
        radius = torch.sqrt(u1 * u1 + u2 * u2)
        r = radius * (radius <= 0.25)
        r = torch.fft.ifftshift(r)
        r[0, 0] = 1.0
        lg = torch.exp(-(torch.log(r / _OMEGA0) ** 2) / (2.0 * _SIGMA_F ** 2))
        lg[0, 0] = 0.0
        _lg_cache[key] = lg
    return lg


@torch.inference_mode(False)
def _centre_bias(device, dtype) -> torch.Tensor:
    key = (str(device), dtype)
    sd = _sd_cache.get(key)
    if sd is None:
        S = _SALIENCY_SIZE
        yy = torch.arange(1, S + 1, device=device, dtype=dtype).view(S, 1)
        xx = torch.arange(1, S + 1, device=device, dtype=dtype).view(1, S)
        d2 = (yy - S / 2.0) ** 2 + (xx - S / 2.0) ** 2
        sd = torch.exp(-d2 / (_SIGMA_D ** 2)).view(1, 1, S, S)
        _sd_cache[key] = sd
    return sd


def _sdsp_batch(t255: torch.Tensor) -> torch.Tensor:
    M, _, H, W = t255.shape
    wdt = t255.dtype
    S = _SALIENCY_SIZE
    ds = F.interpolate(t255, size=(S, S), mode="bilinear", align_corners=False)
    lab = _srgb_to_lab(ds / 255.0)

    spec = torch.fft.fft2(lab.reshape(M * 3, S, S))
    resp = torch.fft.ifft2(spec * _log_gabor(t255.device, wdt)).real
    resp = resp.reshape(M, 3, S, S)
    sf = (resp * resp).sum(dim=1, keepdim=True).sqrt()

    A, B = lab[:, 1:2], lab[:, 2:3]
    nA = (A - A.amin(dim=(1, 2, 3), keepdim=True)) / (
        (A.amax(dim=(1, 2, 3), keepdim=True) - A.amin(dim=(1, 2, 3), keepdim=True))
        .clamp_min(1e-12)
    )
    nB = (B - B.amin(dim=(1, 2, 3), keepdim=True)) / (
        (B.amax(dim=(1, 2, 3), keepdim=True) - B.amin(dim=(1, 2, 3), keepdim=True))
        .clamp_min(1e-12)
    )
    sc = 1.0 - torch.exp(-(nA * nA + nB * nB) / (_SIGMA_C ** 2))

    vs = F.interpolate(sf * _centre_bias(t255.device, wdt) * sc,
                       size=(H, W), mode="bilinear", align_corners=False)
    eps = torch.finfo(wdt).eps
    vs = (vs - vs.amin(dim=(1, 2, 3), keepdim=True)) / (
        (vs.amax(dim=(1, 2, 3), keepdim=True) - vs.amin(dim=(1, 2, 3), keepdim=True)) + eps
    )
    return vs.clamp(0.0, 1.0)


def sdsp_saliency(
    x: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Full SDSP visual saliency in ``[0, 1]``, shape ``(N, 1, H, W)``."""
    t = _fn._as_nchw(x)
    if t.ndim != 4 or t.shape[1] not in (1, 3):
        raise ValueError(f"VSI needs a 1- or 3-channel image; got {tuple(x.shape)}")
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    L = float(data_range) if data_range is not None else _fn._infer_data_range(t)
    if not L > 0:
        raise ValueError(f"data_range must be > 0, got {data_range!r}")
    wdt = _fn._work_dtype(t, dtype)
    return _sdsp_batch(t.to(wdt) * (255.0 / L))


_scharr_cache: dict = {}


@torch.inference_mode(False)
def _scharr_pair(device, dtype) -> torch.Tensor:
    key = (str(device), dtype)
    k = _scharr_cache.get(key)
    if k is None:
        hx = torch.tensor([[3.0, 0.0, -3.0],
                           [10.0, 0.0, -10.0],
                           [3.0, 0.0, -3.0]]) / 16.0
        k = torch.stack([hx, hx.t()]).unsqueeze(1).to(device=device, dtype=dtype)
        _scharr_cache[key] = k
    return k


def _gradient_pair(l1: torch.Tensor, l2: torch.Tensor):
    k = _scharr_pair(l1.device, l1.dtype)
    with _fn._no_autocast(l1):
        g = F.conv2d(torch.cat([l1, l2], dim=0), k, padding=1)
    n = l1.shape[0]
    g1 = torch.sqrt(g[:n, 0:1] ** 2 + g[:n, 1:2] ** 2)
    g2 = torch.sqrt(g[n:, 0:1] ** 2 + g[n:, 1:2] ** 2)
    return g1, g2


def _vsi_epilogue(vs1, vs2, g1, g2, m1, m2, n1, n2):
    s_vs = (2.0 * vs1 * vs2 + _C_VS) / (vs1 * vs1 + vs2 * vs2 + _C_VS)
    s_g = (2.0 * g1 * g2 + _C_GM) / (g1 * g1 + g2 * g2 + _C_GM)
    s_m = (2.0 * m1 * m2 + _C_CH) / (m1 * m1 + m2 * m2 + _C_CH)
    s_n = (2.0 * n1 * n2 + _C_CH) / (n1 * n1 + n2 * n2 + _C_CH)
    prod = s_m * s_n
    chroma = torch.where(prod >= 0, prod.clamp_min(0).pow(_LAMBDA),
                         prod.abs().pow(_LAMBDA) * _LAM_COS)
    return s_vs * s_g.pow(_ALPHA) * chroma, torch.maximum(vs1, vs2)


def _box_decimate(t: torch.Tensor, f: int) -> torch.Tensor:
    return t if f <= 1 else F.avg_pool2d(t, f, stride=f)


def vsi(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
) -> torch.Tensor:
    """Visual saliency-induced index (Zhang et al. 2014). Higher is better.

    ``1.0`` for identical inputs. ``luma``/``crop_border`` follow
    :mod:`frame_analytics.functional`: the crop is applied first; ``luma``
    projects 3-channel input to luma (replicated back to RGB for the
    saliency path) instead of scoring colour.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    if x is y:
        n = x.shape[0] if x.ndim == 4 else 1
        per_image = torch.ones((n,), device=x.device, dtype=torch.float64)
        return per_image.mean() if reduction == "mean" else per_image
    x4, y4, L, _ = _fn._prep(x, y, None, crop_border, data_range)
    if not L > 0:
        raise ValueError(f"data_range must be > 0, got {data_range!r}")
    if x4.shape[1] not in (1, 3):
        raise ValueError(f"VSI needs a 1- or 3-channel image; got {tuple(x.shape)}")
    wdt = _fn._work_dtype(x4, dtype)
    if luma not in (None, False) and x4.shape[1] == 3:
        x4, y4 = _fn._apply_luma(x4, y4, _fn._resolve_luma(luma), L, wdt)
    if x4.shape[1] == 1:
        warnings.warn(
            "VSI is defined on RGB; the 1-channel input was repeated to 3 channels "
            "(PIQ-style)."
        )
        x4 = x4.repeat(1, 3, 1, 1)
        y4 = y4.repeat(1, 3, 1, 1)
    n, _, h, w = x4.shape
    scale = 255.0 / L

    both = torch.cat([x4.to(wdt) * scale, y4.to(wdt) * scale], dim=0)
    vs = _sdsp_batch(both)
    vs1, vs2 = vs.split(n)
    x255, y255 = both.split(n)

    L1, M1, N1 = _lmn(x255)
    L2, M2, N2 = _lmn(y255)

    f = max(1, int(round(min(h, w) / 256.0)))
    L1, L2 = _box_decimate(L1, f), _box_decimate(L2, f)
    M1, M2 = _box_decimate(M1, f), _box_decimate(M2, f)
    N1, N2 = _box_decimate(N1, f), _box_decimate(N2, f)
    vs1, vs2 = _box_decimate(vs1, f), _box_decimate(vs2, f)

    g1, g2 = _gradient_pair(L1, L2)

    epi = _fn._maybe_compile(_vsi_epilogue, "vsi:epilogue")
    local, wgt = epi(vs1, vs2, g1, g2, M1, M2, N1, N2)

    eps = torch.finfo(torch.float64).eps
    num = (local * wgt).sum(dim=(1, 2, 3), dtype=torch.float64)
    den = wgt.sum(dim=(1, 2, 3), dtype=torch.float64)
    per_image = (num + eps) / (den + eps)
    return per_image.mean() if reduction == "mean" else per_image
