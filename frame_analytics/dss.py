"""DCT Subband Similarity (DSS), Balanov et al., ICIP 2015.

Final package module (torch-only). Shared input front end comes from
:mod:`frame_analytics.functional` (:func:`_prep`, :func:`_work_dtype`,
:func:`_no_autocast`, :func:`_maybe_compile`); no local duplicates.
``torch.compile`` uses the shared eager-fallback wrapper;
:func:`set_compile_enabled` is a passthrough to ``functional``.

Pipeline: BT.601 luma for 3-channel input (otherwise per-channel),
rescale to 0..255, crop to multiple of 8, orthonormal 8x8 block DCT-II
straight into subband layout, pointwise Gaussian-variance similarity
(``C=1000`` DC / ``C=300`` AC, Pearson term on DC), worst-5% pooling per
subband, Gaussian frequency weighting (sigma=1.55). Output in [0, 1].
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import functional as _functional
from .functional import (
    _maybe_compile,
    _no_autocast,
    _prep,
    _work_dtype,
)

__all__ = ["dss", "set_compile_enabled"]

_DCT_SIZE = 8
_DC_C = 1000.0
_AC_C = 300.0
_SIGMA_WEIGHT = 1.55
_KERNEL_SIZE = 3
_SIGMA_SIMILARITY = 1.5
_PERCENTILE = 0.05
_WEIGHT_SKIP = 1e-2


def set_compile_enabled(flag: bool) -> None:
    """Passthrough to :func:`frame_analytics.functional.set_compile_enabled`."""
    _functional.set_compile_enabled(flag)


_DCT_CACHE: dict = {}
_GAUSS_CACHE: dict = {}
_WEIGHT_CACHE: dict = {}


@torch.inference_mode(False)
def _dct_matrix(n: int, device, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal DCT-II matrix (matches MATLAB ``dct`` / PIQ); cached."""
    key = (n, str(device), dtype)
    m = _DCT_CACHE.get(key)
    if m is None:
        j = torch.arange(n, dtype=torch.float64)
        i = torch.arange(n, dtype=torch.float64).unsqueeze(1)
        m = torch.cos(math.pi * (2 * j + 1) * i / (2 * n))
        m[0] *= 1.0 / math.sqrt(2.0)
        m = (m * math.sqrt(2.0 / n)).to(device=device, dtype=dtype)
        _DCT_CACHE[key] = m
    return m


def _block_dct_subbands(x: torch.Tensor, y: torch.Tensor, n: int = 8
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block DCT of both images, straight into ``(N, n, n, Hb, Wb)`` subbands."""
    both = torch.cat([x, y], dim=0)
    nb = both.shape[0]
    h, w = both.shape[-2], both.shape[-1]
    hb, wb = h // n, w // n
    c = _dct_matrix(n, both.device, both.dtype)
    blk = both.reshape(nb, hb, n, wb, n).permute(0, 1, 3, 2, 4)
    blk = blk.contiguous().view(nb * hb * wb, n, n)
    with _no_autocast(blk):
        blk = c @ blk @ c.t()
    blk = blk.view(nb, hb, wb, n, n).permute(0, 3, 4, 1, 2).contiguous()
    return blk[: x.shape[0]], blk[x.shape[0]:]


@torch.inference_mode(False)
def _gaussian_kernel(k: int, sigma: float, device, dtype) -> torch.Tensor:
    key = (k, sigma, str(device), dtype)
    g = _GAUSS_CACHE.get(key)
    if g is None:
        r = (k - 1) / 2.0
        c = torch.arange(k, dtype=torch.float64) - r
        v = torch.exp(-(c * c) / (2.0 * sigma * sigma))
        v = v / v.sum()
        g = (v.unsqueeze(0) * v.unsqueeze(1)).to(device=device, dtype=dtype)
        _GAUSS_CACHE[key] = g
    return g


@torch.inference_mode(False)
def _freq_weights(d: int, sigma: float, device):
    """Normalised float64 Gaussian subband weights + kept ``(m, n)`` list."""
    key = (d, sigma, str(device))
    hit = _WEIGHT_CACHE.get(key)
    if hit is None:
        coords = torch.arange(1, d + 1, dtype=torch.float64)
        sq = (coords - 0.5) ** 2
        w = (-(sq.unsqueeze(0) + sq.unsqueeze(1)) / (2 * sigma ** 2)).exp()
        kept = [(m, n) for m in range(d) for n in range(d)
                if w[m, n].item() >= _WEIGHT_SKIP]
        wfull = torch.zeros_like(w)
        for m, n in kept:
            wfull[m, n] = w[m, n]
        wfull = wfull / wfull.sum().clamp_min(torch.finfo(torch.float64).tiny)
        hit = (wfull.to(device=device), kept)
        _WEIGHT_CACHE[key] = hit
    return hit


def _epi_left(vx, vy, c):
    s = _functional._safe_sqrt(vx * vy)
    return (2.0 * s + c) / (vx + vy + c), s


def _epi_right(vxy, s, c):
    return (vxy + c) / (s + c)


def dss(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
    dct_size: int = _DCT_SIZE,
    sigma_weight: float = _SIGMA_WEIGHT,
    kernel_size: int = _KERNEL_SIZE,
    sigma_similarity: float = _SIGMA_SIMILARITY,
    percentile: float = _PERCENTILE,
) -> torch.Tensor:
    """DCT Subband Similarity (Balanov et al. 2015), in [0, 1], higher is better.

    3-channel inputs default to the BT.601 luma plane (pass ``luma=False``
    with 1-channel slices, or use the shared ``luma`` convention to force a
    projection); other channel counts are scored per channel and averaged.
    """
    if reduction not in ("mean", "none", "sum"):
        raise ValueError(f"reduction must be 'mean', 'none' or 'sum', got {reduction!r}")
    if sigma_weight == 0 or sigma_similarity == 0:
        raise ValueError("Gaussian sigmas must be non-zero")
    if not 0 < percentile <= 1:
        raise ValueError(f"percentile must be in (0, 1], got {percentile}")
    if dct_size <= 0 or kernel_size <= 0:
        raise ValueError("dct_size and kernel_size must be positive")

    # Shared front end for shape/device/range/crop conventions. DSS keeps its
    # historical default (3-channel -> BT.601 luma) when the caller did not
    # ask for a projection explicitly.
    _luma_arg = luma if (luma is not None and luma is not False) else None
    x4, y4, L, mode = _prep(x, y, _luma_arg, crop_border, data_range)
    wdt = _work_dtype(x4, dtype)
    n_batch, n_chan = x4.shape[0], x4.shape[1]
    if min(x4.shape[-2], x4.shape[-1]) < dct_size:
        raise ValueError(
            f"image {tuple(x4.shape[-2:])} smaller than DCT size {dct_size}")

    cdt = wdt if wdt in (torch.float32, torch.float64) else torch.float32
    scale = 255.0 / L
    if mode is not None:
        # Explicit shared luma projection (bt601/bt709/matlab).
        from .functional import _apply_luma as _al

        xl, yl = _al(x4, y4, mode, L, cdt)
        # _apply_luma works in the input's own units; rescale to 0..255:
        xl = xl.to(cdt) * scale
        yl = yl.to(cdt) * scale
        # fold channels into batch (luma is single-channel here)
        xl = xl.reshape(n_batch * xl.shape[1], 1, xl.shape[-2], xl.shape[-1])
        yl = yl.reshape(n_batch * yl.shape[1], 1, yl.shape[-2], yl.shape[-1])
    elif n_chan == 3:
        w = torch.tensor((0.299, 0.587, 0.114),
                         device=x4.device, dtype=cdt).view(1, 3, 1, 1)
        xl = (x4.to(cdt) * scale * w).sum(dim=1, keepdim=True)
        yl = (y4.to(cdt) * scale * w).sum(dim=1, keepdim=True)
    else:
        xl = x4.to(cdt) * scale
        yl = y4.to(cdt) * scale
        xl = xl.reshape(n_batch * n_chan, 1, xl.shape[-2], xl.shape[-1])
        yl = yl.reshape(n_batch * n_chan, 1, yl.shape[-2], yl.shape[-1])

    h = (xl.shape[-2] // dct_size) * dct_size
    w_ = (xl.shape[-1] // dct_size) * dct_size
    xl, yl = xl[:, :, :h, :w_], yl[:, :, :h, :w_]
    if not xl.is_contiguous():
        xl = xl.contiguous()
    if not yl.is_contiguous():
        yl = yl.contiguous()

    d = dct_size
    dx_all, dy_all = _block_dct_subbands(xl, yl, d)
    np_ = dx_all.shape[0]
    hs, ws = dx_all.shape[-2], dx_all.shape[-1]

    wfull, kept = _freq_weights(d, sigma_weight, dx_all.device)
    flat_idx = [m * d + n for m, n in kept]
    na = len(kept)
    xf = dx_all.reshape(np_, d * d, hs, ws)[:, flat_idx].reshape(np_ * na, 1, hs, ws)
    yf = dy_all.reshape(np_, d * d, hs, ws)[:, flat_idx].reshape(np_ * na, 1, hs, ws)

    k = _gaussian_kernel(kernel_size, sigma_similarity,
                         dx_all.device, dx_all.dtype).view(1, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    with _no_autocast(xf):
        mu = F.conv2d(torch.cat([xf, yf], dim=0), k, padding=pad)
        sq = F.conv2d(torch.cat([xf * xf, yf * yf], dim=0), k, padding=pad)
    mu_x, mu_y = mu[: np_ * na], mu[np_ * na:]
    vx = (sq[: np_ * na] - mu_x * mu_x).clamp_min(0.0)
    vy = (sq[np_ * na:] - mu_y * mu_y).clamp_min(0.0)
    vx = vx.reshape(np_, na, -1)
    vy = vy.reshape(np_, na, -1)

    cvec = torch.tensor([_DC_C if (m == 0 and n == 0) else _AC_C
                         for m, n in kept],
                        device=dx_all.device, dtype=dx_all.dtype).view(1, na, 1)
    epi_left = _maybe_compile(_epi_left, "dss:epi_left")
    left, s = epi_left(vx, vy, cvec)

    npts = left.shape[-1]
    kpts = min(npts, max(1, round(percentile * npts)))
    sim = torch.topk(left, kpts, dim=-1, largest=False).values.mean(
        dim=-1, dtype=torch.float64)

    dc_idx = next((i for i, mn in enumerate(kept) if mn == (0, 0)), None)
    if dc_idx is not None:
        xdf = xf.reshape(np_, na, hs, ws)[:, dc_idx:dc_idx + 1].reshape(np_, 1, hs, ws)
        ydf = yf.reshape(np_, na, hs, ws)[:, dc_idx:dc_idx + 1].reshape(np_, 1, hs, ws)
        with _no_autocast(xf):
            vxy = (F.conv2d(xdf * ydf, k, padding=pad)
                   - (mu_x.reshape(np_, na, hs, ws)[:, dc_idx:dc_idx + 1]
                      * mu_y.reshape(np_, na, hs, ws)[:, dc_idx:dc_idx + 1]))
        vxy = vxy.reshape(np_, 1, -1)
        epi_right = _maybe_compile(_epi_right, "dss:epi_right")
        right = epi_right(vxy, s[:, dc_idx:dc_idx + 1],
                          torch.tensor(_DC_C, device=dx_all.device,
                                       dtype=dx_all.dtype))
        dc_factor = torch.topk(
            right, kpts, dim=-1, largest=False).values.mean(
                dim=-1, dtype=torch.float64).squeeze(-1)
        sim = torch.cat([sim[:, :dc_idx],
                         (sim[:, dc_idx] * dc_factor).unsqueeze(1),
                         sim[:, dc_idx + 1:]], dim=1)

    w64 = wfull[[m for m, _ in kept], [n for _, n in kept]].to(torch.float64)
    pooled = (sim * w64.unsqueeze(0)).sum(dim=-1)
    per_image = pooled.reshape(n_batch, pooled.shape[0] // n_batch).mean(dim=1)

    if reduction == "none":
        return per_image
    if reduction == "sum":
        return per_image.sum()
    return per_image.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.rand(2, 3, 64, 64)
    b = (a + 0.05 * torch.randn_like(a)).clamp(0, 1)
    ident = dss(a, a)
    dist = dss(a, b)
    per = dss(a, b, reduction="none")
    print(f"identical={ident.item():.6f} distorted={dist.item():.6f} per={per.tolist()}")
    assert abs(ident.item() - 1.0) < 1e-6, ident
    assert 0.0 <= dist.item() <= 1.0 and dist.item() < 1.0, dist
    assert per.shape == (2,)
    print("dss self-test: OK")
