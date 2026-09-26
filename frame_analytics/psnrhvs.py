"""PSNR-HVS and PSNR-HVS-M (Egiazarian et al.).

Torch-only NCHW implementation closing the ``vapoursynth`` id-1 gap at the
algorithm level.

References
----------
* K. Egiazarian, J. Astola, N. Ponomarenko, V. Lukin, F. Battisti, M. Carli,
  "New full-reference quality metrics based on HVS", VPQM-06 (PSNR-HVS:
  8x8 DCT + CSF weighting).
* N. Ponomarenko, F. Silvestri, K. Egiazarian, M. Carli, J. Astola, V. Lukin,
  "On between-coefficient contrast masking of DCT basis functions", VPQM-07
  (PSNR-HVS-M: adds between-coefficient masking).
* Reference implementation ``psnrhvsm.m`` (Ponomarenko, 2006); the ``CSF``
  and ``MASK`` tables below are transcribed from it verbatim.

Algorithm (per 8x8 non-overlapping block, DC excluded from masking)
-------------------------------------------------------------------
1. ``A_dct = dct2(A)``, ``B_dct = dct2(B)`` (orthonormal DCT-II, as MATLAB).
2. PSNR-HVS: ``MSE_JND = mean((|A-B| * CSF)^2)`` over all coefficients of
   all blocks, then ``PSNR = 10*log10(L^2 / MSE_JND)``.
3. PSNR-HVS-M: masking energy per block ``Emax = max(maskeff(A),
   maskeff(B))`` with ``maskeff = sqrt(sum_AC(dct^2 * MaskCof) * pop) / 32``
   where ``pop`` is the quadrant-to-whole variance ratio (edge correction);
   each non-DC error is reduced ``u -> max(0, u - Emax / MaskCof)`` before
   the CSF weighting in step 2.

Input conventions follow :mod:`frame_analytics.functional` (helpers reused):
``(H,W)`` / ``(C,H,W)`` / ``(N,C,H,W)``; channels scored as independent
planes and averaged per image; ``data_range`` defaults to 255 for integer
input and 1.0 for float. Border pixels that do not fill an 8x8 block are
dropped, as in ``psnrhvsm.m``. Compute in float32, final per-image reduction
in float64. Block cores run through :func:`functional._maybe_compile` with
its permanent eager fallback.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from . import functional as _F
from .functional import set_compile_enabled

__all__ = ["psnr_hvs", "psnr_hvs_m", "mse_hvs", "mse_hvs_m", "set_compile_enabled"]

_COMPUTE = torch.float32   # block math dtype
_ACCUM = torch.float64     # final per-image reduction dtype

# --------------------------------------------------------------------------- #
# tables from psnrhvsm.m (CSFCof, MaskCof), indexed [vertical][horizontal]
# --------------------------------------------------------------------------- #

CSF = (
    (1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887),
    (2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911),
    (1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555),
    (1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082),
    (1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222),
    (1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729),
    (0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803),
    (0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950),
)

MASK = (
    (0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874),
    (0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058),
    (0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888),
    (0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015),
    (0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866),
    (0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815),
    (0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803),
    (0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203),
)


class _Tables:
    """Per-device constants: ``C``/``CT`` DCT basis, ``W = CSF^2`` weight,
    ``MW`` masking energy weight (DC zeroed), ``IW = 1/MASK`` (DC zeroed)."""

    __slots__ = ("C", "CT", "W", "MW", "IW")


_TBL_CACHE: dict = {}


@torch.inference_mode(False)
def _tables(device) -> _Tables:
    key = str(device)
    t = _TBL_CACHE.get(key)
    if t is None:
        t = _Tables()
        n = torch.arange(8, dtype=_COMPUTE)
        k = n.view(-1, 1)
        # Orthonormal DCT-II basis (dct2 convention, as MATLAB).
        c = torch.cos(math.pi * (2.0 * n + 1.0) * k / 16.0) * 0.5
        c[0] *= 1.0 / math.sqrt(2.0)
        t.C = c.to(device=device)
        t.CT = t.C.T.contiguous()
        csf = torch.tensor(CSF, dtype=_COMPUTE, device=device)
        msk = torch.tensor(MASK, dtype=_COMPUTE, device=device)
        t.W = csf * csf
        t.MW = msk.clone()
        t.MW[0, 0] = 0.0
        t.IW = msk.reciprocal()
        t.IW[0, 0] = 0.0
        _TBL_CACHE[key] = t
    return t


def _dct_matrix(device) -> torch.Tensor:
    """Orthonormal 8x8 DCT-II basis (``dct2`` convention), cached, float32."""
    return _tables(device).C


def _prep(x, y, data_range):
    x4, y4 = _F._check_pair(x, y)
    L = float(data_range) if data_range is not None else _F._infer_data_range(x4)
    h, w = x4.shape[-2], x4.shape[-1]
    hc, wc = (h // 8) * 8, (w // 8) * 8
    if hc == 0 or wc == 0:
        raise ValueError(f"image {h}x{w} is smaller than one 8x8 block")
    return x4[..., :hc, :wc], y4[..., :hc, :wc], L


# --------------------------------------------------------------------------- #
# compiled cores: (B,8,8) float32 blocks in, (B,) float32 per-block MSE out
# --------------------------------------------------------------------------- #

def _hvs_core(db, C, CT, W):
    """CSF-weighted error energy of DCT *difference* blocks (DCT is linear)."""
    D = C @ db @ CT
    return ((D * D) * W).sum(dim=(1, 2)) * (1.0 / 64.0)


def _hvsm_core(xb, yb, C, CT, W, MW, IW):
    """Masked variant: between-coefficient masking before CSF weighting."""
    Xd = C @ xb @ CT
    Yd = C @ yb @ CT
    u = (Xd - Yd).abs()

    Qx = Xd * Xd
    Qy = Yd * Yd
    # AC energy (DC excluded) doubles as the masking sum's base and, via
    # Parseval, as the whole-block spatial variance -- no second pixel pass.
    xac = Qx.sum(dim=(1, 2)) - Qx[:, 0, 0]
    yac = Qy.sum(dim=(1, 2)) - Qy[:, 0, 0]
    mX = (Qx * MW).sum(dim=(1, 2))
    mY = (Qy * MW).sum(dim=(1, 2))
    wholeX = xac * (64.0 / 63.0)
    wholeY = yac * (64.0 / 63.0)

    qsumX = torch.zeros_like(wholeX)
    qsumY = torch.zeros_like(wholeY)
    for qx, qy in ((xb[:, :4, :4], yb[:, :4, :4]),
                   (xb[:, :4, 4:], yb[:, :4, 4:]),
                   (xb[:, 4:, :4], yb[:, 4:, :4]),
                   (xb[:, 4:, 4:], yb[:, 4:, 4:])):
        mx = qx.mean(dim=(1, 2), keepdim=True)
        my = qy.mean(dim=(1, 2), keepdim=True)
        qsumX = qsumX + ((qx - mx) ** 2).sum(dim=(1, 2)) * (16.0 / 15.0)
        qsumY = qsumY + ((qy - my) ** 2).sum(dim=(1, 2)) * (16.0 / 15.0)
    popX = torch.where(wholeX > 0, qsumX / wholeX, 0.0)
    popY = torch.where(wholeY > 0, qsumY / wholeY, 0.0)
    eX = torch.sqrt(mX * popX) * (1.0 / 32.0)
    eY = torch.sqrt(mY * popY) * (1.0 / 32.0)

    emax = torch.maximum(eX, eY)
    um = torch.relu(u - emax.view(-1, 1, 1) * IW)  # DC threshold is 0
    return ((um * um) * W).sum(dim=(1, 2)) * (1.0 / 64.0)


def _mse_jnd_per_image(x4, y4, masking: bool) -> torch.Tensor:
    """Perceptual MSE per batch element, ``(N,)`` float64 on the input device."""
    n = x4.shape[0]
    T = _tables(x4.device)
    # Per-device-type keys: a CPU-Inductor failure must not poison CUDA.
    dts = x4.device.type
    if masking:
        Xb = x4.to(_COMPUTE).unfold(2, 8, 8).unfold(3, 8, 8).reshape(-1, 8, 8)
        Yb = y4.to(_COMPUTE).unfold(2, 8, 8).unfold(3, 8, 8).reshape(-1, 8, 8)
        core = _F._maybe_compile(_hvsm_core, f"psnrhvs:hvsm:{dts}")
        with _F._no_autocast(Xb):
            pb = core(Xb, Yb, T.C, T.CT, T.W, T.MW, T.IW)
    else:
        Xf = x4.to(_COMPUTE)
        Yf = y4.to(_COMPUTE)
        # one DCT of the difference instead of two (linearity), one block copy
        Db = (Xf.unfold(2, 8, 8).unfold(3, 8, 8)
              - Yf.unfold(2, 8, 8).unfold(3, 8, 8)).reshape(-1, 8, 8)
        core = _F._maybe_compile(_hvs_core, f"psnrhvs:hvs:{dts}")
        with _F._no_autocast(Db):
            pb = core(Db, T.C, T.CT, T.W)
    return pb.to(_ACCUM).view(n, -1).mean(dim=1)


def _mse_jnd(x, y, masking: bool, data_range, reduction: str) -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    x4, y4, _ = _prep(x, y, data_range)
    per_image = _mse_jnd_per_image(x4, y4, masking)
    return per_image if reduction == "none" else per_image.mean()


def _psnr_from_jnd(x, y, masking: bool, data_range, reduction: str, eps: float) -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    x4, y4, L = _prep(x, y, data_range)
    m = _mse_jnd_per_image(x4, y4, masking)
    if eps:
        m = m.clamp_min(eps)
    bias = 10.0 * math.log10(L * L)
    out = bias - 10.0 * torch.log10(m)  # m == 0 -> +inf, as functional.psnr
    return out if reduction == "none" else out.mean()


def mse_hvs(x: torch.Tensor, y: torch.Tensor, *, data_range: Optional[float] = None,
            reduction: str = "mean") -> torch.Tensor:
    """CSF-weighted MSE (the ``MSE_JND`` inside PSNR-HVS), input units squared."""
    return _mse_jnd(x, y, masking=False, data_range=data_range, reduction=reduction)


def mse_hvs_m(x: torch.Tensor, y: torch.Tensor, *, data_range: Optional[float] = None,
              reduction: str = "mean") -> torch.Tensor:
    """Masked CSF-weighted MSE (the ``MSE_JND`` inside PSNR-HVS-M)."""
    return _mse_jnd(x, y, masking=True, data_range=data_range, reduction=reduction)


def psnr_hvs(x: torch.Tensor, y: torch.Tensor, *, data_range: Optional[float] = None,
             reduction: str = "mean", eps: float = 0.0) -> torch.Tensor:
    """PSNR-HVS in dB; identical inputs give ``+inf`` unless ``eps`` is set."""
    return _psnr_from_jnd(x, y, masking=False, data_range=data_range,
                          reduction=reduction, eps=eps)


def psnr_hvs_m(x: torch.Tensor, y: torch.Tensor, *, data_range: Optional[float] = None,
               reduction: str = "mean", eps: float = 0.0) -> torch.Tensor:
    """PSNR-HVS-M in dB; identical inputs give ``+inf`` unless ``eps`` is set."""
    return _psnr_from_jnd(x, y, masking=True, data_range=data_range,
                          reduction=reduction, eps=eps)
