"""Mean Deviation Similarity Index (MDSI, Nafchi et al. 2016).

Built on :mod:`frame_analytics.functional` helpers. Preserved spec points
from ``proposals/mdsi.py``: LHM Gaussian colour model, 3x3 Prewitt ``/3``
gradient similarity with the fused-image correction, joint H+M
chromaticity similarity, summation (``alpha``) or multiplicative
(``GS^gamma * CS^beta``) combination, and generalised deviation pooling
with complex fractional powers and float64 accumulation.

MDSI is a *distortion*: 0 for identical inputs, larger for worse quality.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F

from . import functional as _fn

__all__ = [
    "mdsi",
    "gcs_map",
    "MDSI",
    "MDSI_DEFAULTS",
]

MDSI_DEFAULTS = {
    "c1": 140.0,
    "c2": 55.0,
    "c3": 550.0,
    "alpha": 0.6,
    "rho": 1.0,
    "q": 0.25,
    "o": 0.25,
}

_LHM_ROWS = ((0.2989, 0.5870, 0.1140),
             (0.30, 0.04, -0.35),
             (0.34, -0.60, 0.17))

_lhm_weight_cache: dict = {}


@torch.inference_mode(False)
def _lhm_weight(scale: float, device, dtype) -> torch.Tensor:
    key = (float(scale), str(device), str(dtype))
    w = _lhm_weight_cache.get(key)
    if w is None:
        base = torch.tensor(_LHM_ROWS, dtype=torch.float64)
        w = (base * float(scale)).to(device=device, dtype=dtype)
        w = w.reshape(3, 3, 1, 1).contiguous()
        if len(_lhm_weight_cache) < 32:
            _lhm_weight_cache[key] = w
    return w


_gradient_kernel_cache: dict = {}


@torch.inference_mode(False)
def _gradient_kernels(op: str, device, dtype) -> torch.Tensor:
    key = (op, str(device), str(dtype))
    k = _gradient_kernel_cache.get(key)
    if k is not None:
        return k
    if op == "prewitt":
        hx = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]]
        ) / 3.0
    elif op == "sobel":
        hx = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
        ) / 4.0
    else:
        raise ValueError(f"gradient must be 'prewitt' or 'sobel', got {op!r}")
    k = torch.stack([hx, hx.transpose(-1, -2)]).to(device=device, dtype=dtype)
    _gradient_kernel_cache[key] = k
    return k


_cis_cache: dict = {}


def _cis(exp: float):
    key = float(exp)
    v = _cis_cache.get(key)
    if v is None:
        v = (math.cos(math.pi * key), math.sin(math.pi * key))
        _cis_cache[key] = v
    return v


def _front_lhm(x4: torch.Tensor, y4: torch.Tensor, L: float,
               wdt: torch.dtype, downsample: bool):
    """Batched RGB->LHM for already-cropped ``(N,3,H,W)`` inputs."""
    n = x4.shape[0]
    x4 = x4.to(wdt)
    y4 = y4.to(wdt)
    if downsample:
        k = max(1, round(min(x4.shape[-2:]) / 256))
        if k > 1:
            pad = [(k - 1) // 2, k // 2, (k - 1) // 2, k // 2]
            x4 = F.avg_pool2d(F.pad(x4, pad), k)
            y4 = F.avg_pool2d(F.pad(y4, pad), k)
    b = torch.cat([x4, y4], dim=0)
    if b.is_cuda:
        b = b.to(memory_format=torch.channels_last)
    with _fn._no_autocast(b):
        lhm = F.conv2d(b, _lhm_weight(255.0 / L, b.device, wdt))
    return lhm[:n], lhm[n:]


def _prepare_pair(x, y, data_range, luma, crop_border, dtype, downsample):
    x4, y4, L, _ = _fn._prep(x, y, None, crop_border, data_range)
    if not L > 0:
        raise ValueError(f"data_range must be > 0, got {data_range!r}")
    if x4.shape[1] == 1:
        warnings.warn(
            "MDSI is defined on RGB; the single channel is triplicated "
            "(chromaticity then measures nothing).",
            stacklevel=4,
        )
        x4 = x4.repeat(1, 3, 1, 1)
        y4 = y4.repeat(1, 3, 1, 1)
    elif x4.shape[1] != 3:
        raise ValueError(
            f"MDSI needs 1- or 3-channel input, got {x4.shape[1]} channels")
    wdt = _fn._work_dtype(x4, dtype)
    if luma not in (None, False):
        # Score on luma: project then triplicate back so the LHM /
        # chromaticity path runs on achromatic data.
        mode = _fn._resolve_luma(luma)
        lx, ly = _fn._apply_luma(x4, y4, mode, L, wdt)
        x4 = lx.repeat(1, 3, 1, 1)
        y4 = ly.repeat(1, 3, 1, 1)
    xl, yl = _front_lhm(x4, y4, L, wdt, downsample)
    return xl, yl, L, wdt


def _gradients(xl: torch.Tensor, yl: torch.Tensor,
               kernels: torch.Tensor):
    lum = torch.cat([xl[:, 0:1], yl[:, 0:1],
                     0.5 * (xl[:, 0:1] + yl[:, 0:1])], dim=0)
    with _fn._no_autocast(lum):
        g = F.conv2d(lum, kernels, padding=1)
    m = torch.sqrt((g * g).sum(dim=1, keepdim=True))
    n = xl.shape[0]
    return m[:n], m[n:2 * n], m[2 * n:]


def _similarity(a: torch.Tensor, b: torch.Tensor, c: float) -> torch.Tensor:
    return (2.0 * a * b + c) / (a * a + b * b + c)


def _gcs_tail(xl: torch.Tensor, yl: torch.Tensor, gs_hat: torch.Tensor,
              c3: float, alpha: float) -> torch.Tensor:
    wdt = gs_hat.dtype
    xh = xl[:, 1:2].to(torch.float64)
    yh = yl[:, 1:2].to(torch.float64)
    xm = xl[:, 2:3].to(torch.float64)
    ym = yl[:, 2:3].to(torch.float64)
    cs = ((2.0 * (xh * yh + xm * ym) + c3)
          / (xh * xh + yh * yh + xm * xm + ym * ym + c3)).to(wdt)
    return cs + alpha * (gs_hat - cs)


def _pow_real(base: torch.Tensor, exp: float) -> torch.Tensor:
    r = base.abs().pow(exp)
    cosq, sinq = _cis(exp)
    neg = base < 0
    return torch.stack([torch.where(neg, r * cosq, r),
                        torch.where(neg, r * sinq, torch.zeros_like(r))],
                       dim=-1)


def _cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
         a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]], dim=-1)


def _cpow_complex(z: torch.Tensor, exp: float) -> torch.Tensor:
    r = z.pow(2).sum(dim=-1).sqrt()
    phi = torch.atan2(z[..., 1], z[..., 0])
    rp = r.pow(exp)
    phip = (phi * exp).unsqueeze(-1)
    return torch.cat([rp.unsqueeze(-1) * torch.cos(phip),
                      rp.unsqueeze(-1) * torch.sin(phip)], dim=-1)


def _cabs(z: torch.Tensor) -> torch.Tensor:
    return z.pow(2).sum(dim=-1).sqrt()


def _pool_sum(g: torch.Tensor, q: float, rho: float, o: float) -> torch.Tensor:
    p = g.abs().pow(q)
    cosq, sinq = _cis(q)
    neg = g < 0
    p64 = p.to(torch.float64)
    re = torch.where(neg, p64 * cosq, p64)
    im = torch.where(neg, p64 * sinq, torch.zeros_like(p64))
    ns = p64.shape[1] * p64.shape[2] * p64.shape[3]
    mr = (re.sum(dim=(1, 2, 3)) / ns).view(-1, 1, 1, 1)
    mi = (im.sum(dim=(1, 2, 3)) / ns).view(-1, 1, 1, 1)
    dev = torch.sqrt((re - mr) ** 2 + (im - mi) ** 2)
    if rho == 1:
        s = dev.sum(dim=(1, 2, 3)) / ns
    else:
        s = dev.pow(rho).sum(dim=(1, 2, 3)) / ns
    return s.pow(o / rho)


def _pool_complex(zd: torch.Tensor, rho: float, o: float) -> torch.Tensor:
    mct = zd.mean(dim=(2, 3), keepdim=True)
    dev = _cabs(zd - mct)
    ns = zd.shape[1] * zd.shape[2] * zd.shape[3]
    if rho == 1:
        s = dev.sum(dim=(1, 2, 3)) / ns
    else:
        s = dev.pow(rho).sum(dim=(1, 2, 3)) / ns
    return s.pow(o / rho)


def _compute_gcs(x, y, data_range, c1, c2, c3, alpha, gradient, dtype,
                 luma, crop_border, downsample):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    for name, c in (("c1", c1), ("c2", c2), ("c3", c3)):
        if not c > 0:
            raise ValueError(f"{name} must be > 0, got {c!r}")
    xl, yl, _, wdt = _prepare_pair(x, y, data_range, luma, crop_border,
                                   dtype, downsample)
    kernels = _gradient_kernels(gradient, xl.device, wdt)
    gx, gy, gf = _gradients(xl, yl, kernels)
    gs_hat = (_similarity(gx, gy, c1)
              + _similarity(gx, gf, c2) - _similarity(gy, gf, c2))
    tail = _fn._maybe_compile(_gcs_tail, "mdsi:tail")
    return tail(xl, yl, gs_hat, float(c3), float(alpha)), xl, yl, wdt


def gcs_map(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    c1: float = MDSI_DEFAULTS["c1"],
    c2: float = MDSI_DEFAULTS["c2"],
    c3: float = MDSI_DEFAULTS["c3"],
    alpha: float = MDSI_DEFAULTS["alpha"],
    gradient: str = "prewitt",
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
    downsample: bool = True,
) -> torch.Tensor:
    """Combined gradient-chromaticity similarity map, ``(N,1,H',W')``."""
    g, _, _, _ = _compute_gcs(x, y, data_range, c1, c2, c3, alpha, gradient,
                              dtype, luma, crop_border, downsample)
    return g


def mdsi(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    reduction: str = "mean",
    c1: float = MDSI_DEFAULTS["c1"],
    c2: float = MDSI_DEFAULTS["c2"],
    c3: float = MDSI_DEFAULTS["c3"],
    alpha: float = MDSI_DEFAULTS["alpha"],
    rho: float = MDSI_DEFAULTS["rho"],
    q: float = MDSI_DEFAULTS["q"],
    o: float = MDSI_DEFAULTS["o"],
    combination: str = "sum",
    beta: float = 0.1,
    gamma: float = 0.2,
    gradient: str = "prewitt",
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
    downsample: bool = True,
    return_map: bool = False,
) -> torch.Tensor:
    """Mean Deviation Similarity Index -- a distortion, lower is better.

    ``mdsi(x, x)`` is exactly 0. ``luma``/``crop_border`` follow
    :mod:`frame_analytics.functional`.
    """
    if reduction not in ("mean", "none", "sum"):
        raise ValueError(
            f"reduction must be 'mean', 'none' or 'sum', got {reduction!r}")
    if combination not in ("sum", "mult"):
        raise ValueError(
            f"combination must be 'sum' or 'mult', got {combination!r}")
    if rho <= 0:
        raise ValueError(f"rho must be > 0, got {rho!r}")

    if combination == "sum":
        g, _, _, _ = _compute_gcs(x, y, data_range, c1, c2, c3, alpha,
                                  gradient, dtype, luma, crop_border,
                                  downsample)
        if return_map:
            return g
        pool = _fn._maybe_compile(_pool_sum, "mdsi:pool")
        per_image = pool(g, float(q), float(rho), float(o)).to(torch.float64)
    else:
        if return_map:
            raise ValueError("return_map is only defined for combination='sum'")
        xl, yl, _, _ = _prepare_pair(x, y, data_range, luma, crop_border,
                                     dtype, downsample)
        kernels = _gradient_kernels(gradient, xl.device, xl.dtype)
        gx, gy, gf = _gradients(xl, yl, kernels)
        gs_hat = (_similarity(gx, gy, float(c1))
                  + _similarity(gx, gf, float(c2))
                  - _similarity(gy, gf, float(c2)))
        tail = _fn._maybe_compile(_gcs_tail, "mdsi:tail")
        cs = tail(xl, yl, gs_hat, float(c3), 0.0)
        z = _cmul(_pow_real(gs_hat, float(gamma)), _pow_real(cs, float(beta)))
        z = _cpow_complex(z, float(q))
        per_image = _pool_complex(z.to(torch.float64),
                                  float(rho), float(o)).to(torch.float64)

    if reduction == "none":
        return per_image
    if reduction == "sum":
        return per_image.sum()
    return per_image.mean()


class MDSI(torch.nn.Module):
    """``torch.nn.Module`` wrapper around :func:`mdsi`."""

    def __init__(
        self,
        *,
        data_range: Optional[float] = None,
        reduction: str = "mean",
        c1: float = MDSI_DEFAULTS["c1"],
        c2: float = MDSI_DEFAULTS["c2"],
        c3: float = MDSI_DEFAULTS["c3"],
        alpha: float = MDSI_DEFAULTS["alpha"],
        rho: float = MDSI_DEFAULTS["rho"],
        q: float = MDSI_DEFAULTS["q"],
        o: float = MDSI_DEFAULTS["o"],
        combination: str = "sum",
        beta: float = 0.1,
        gamma: float = 0.2,
        gradient: str = "prewitt",
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.alpha, self.rho, self.q, self.o = alpha, rho, q, o
        self.combination = combination
        self.beta, self.gamma = beta, gamma
        self.gradient = gradient
        self.downsample = downsample

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return mdsi(
            x, y, data_range=self.data_range, reduction=self.reduction,
            c1=self.c1, c2=self.c2, c3=self.c3, alpha=self.alpha,
            rho=self.rho, q=self.q, o=self.o, combination=self.combination,
            beta=self.beta, gamma=self.gamma, gradient=self.gradient,
            downsample=self.downsample,
        )
