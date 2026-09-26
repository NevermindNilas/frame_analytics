"""CIEDE2000 colour difference -- torch-only counterpart of the libvmaf id-4 gap.

sRGB -> XYZ -> Lab (D65) followed by the CIEDE2000 formula of Sharma et al.
("The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary
Test Data, and Mathematical Observations", 2005), with no ``colour-science``
dependency.

API mirrors :mod:`frame_analytics.functional`: inputs are ``(N,3,H,W)`` RGB
(or ``(3,H,W)``), ``data_range`` defaults to 255 for integer input and 1.0
for float, ``reduction="mean"`` gives a scalar and ``"none"`` gives ``(N,)``,
and ``return_map=True`` gives the per-pixel delta-E map ``(N,1,H,W)``.
Compute in the working dtype, final reduction in float64; elementwise regions
run through :func:`functional._maybe_compile` (``dynamic=True``) with its
permanent eager fallback.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from . import functional as _F
from .functional import set_compile_enabled

__all__ = [
    "ciede2000",
    "srgb_to_lab",
    "delta_e_00",
    "set_compile_enabled",
    "compile_status",
]

# IEC 61966-2-1 linear-sRGB -> XYZ (D65), rows X/Y/Z.
_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)

# D65 reference white, as reciprocals (multiply, never divide).
_INV_XN, _INV_YN, _INV_ZN = 1.0 / 0.95047, 1.0 / 1.0, 1.0 / 1.08883

# Hoisted once: 25**7 needs 10 digits, float32 only carries ~7 (6103515625
# rounds to 6103515648), so every ratio against it runs in float64.
_P25_7 = 6103515625.0

_D2R = math.pi / 180.0
_R2D = 180.0 / math.pi

# CIE f(t): delta = 6/29 throughout.
_DELTA3 = (6.0 / 29.0) ** 3
_DELTA2_3 = 3.0 * (6.0 / 29.0) ** 2
_DELTA_OFF = 4.0 / 29.0

_F64 = torch.float64


def compile_status() -> dict:
    """``{cache key: state}`` for this module's compiled regions."""
    out = {}
    for k, v in _F._compiled_cache.items():
        if "ciede2000" in k:
            out[k] = ("eager-fallback" if v._eager_only
                      else "compiled" if v._compiled is not None else "not-run")
    return out


_xyz_cache: dict = {}


@torch.inference_mode(False)
def _xyz_cols(device, dtype):
    """sRGB->XYZ matrix as three ``(1,3,1,1)`` columns, cached on-device."""
    key = (str(device), dtype)
    cols = _xyz_cache.get(key)
    if cols is None:
        m = _SRGB_TO_XYZ
        dev = torch.device(device)
        cols = tuple(
            torch.tensor([m[0][i], m[1][i], m[2][i]], dtype=torch.float64)
            .view(1, 3, 1, 1)
            .to(device=dev, dtype=dtype)
            for i in range(3)
        )
        _xyz_cache[key] = cols
    return cols


def _srgb_to_linear(v: torch.Tensor) -> torch.Tensor:
    """Exact piecewise inverse sRGB companding, on [0, 1] data."""
    return torch.where(
        v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055).clamp_min(0.0).pow(2.4)
    )


def _lab_f(t: torch.Tensor) -> torch.Tensor:
    """The CIE ``f`` nonlinearity."""
    return torch.where(
        t > _DELTA3, t.clamp_min(_DELTA3).pow(1.0 / 3.0), t / _DELTA2_3 + _DELTA_OFF
    )


def _convert_one(t, inv_l: float, cr, cg, cb):
    """NCHW sRGB (already in the working dtype) -> NCHW CIELAB, D65."""
    lin = _srgb_to_linear((t * inv_l).clamp(0.0, 1.0))
    xyz = cr * lin[:, 0:1] + cg * lin[:, 1:2] + cb * lin[:, 2:3]
    fx = _lab_f(xyz[:, 0:1] * _INV_XN)
    fy = _lab_f(xyz[:, 1:2] * _INV_YN)
    fz = _lab_f(xyz[:, 2:3] * _INV_ZN)
    return torch.cat([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=1)


def srgb_to_lab(
    rgb: torch.Tensor,
    data_range: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """sRGB ``(N,3,H,W)`` (or ``(3,H,W)``) -> CIELAB, same layout, D65."""
    t4 = _F._as_nchw(rgb)
    if t4.shape[1] != 3:
        raise ValueError(f"need 3-channel RGB, got {t4.shape[1]} channels")
    L = float(data_range) if data_range is not None else _F._infer_data_range(t4)
    wdt = _F._work_dtype(t4, dtype)
    conv = _F._maybe_compile(_convert_one, f"ciede2000:convert_one:{t4.device.type}",
                             dynamic=True)
    with _F._no_autocast(t4):
        return conv(t4.to(wdt), 1.0 / L, *_xyz_cols(t4.device, wdt))


# --------------------------------------------------------------------------- #
# vectorised CIEDE2000 (Sharma et al.). Plane-based: every argument is one
# broadcastable plane. Chroma/hue in the input precision; the T/SL/SC/SH/RT
# block (and G/Rc, whose 25**7 denominator is not float32-exact) in float64.
# --------------------------------------------------------------------------- #

def _de00_planes(L1, a1, b1, L2, a2, b2, kL: float, kC: float, kH: float):
    C1 = _F._safe_sqrt(a1 * a1 + b1 * b1)
    C2 = _F._safe_sqrt(a2 * a2 + b2 * b2)
    Cb7 = (C1 + C2).to(_F64) / 2.0
    Cb7 = Cb7**7
    G = 0.5 * (1.0 - _F._safe_sqrt(Cb7 / (Cb7 + _P25_7)))

    a1p = (1.0 + G.to(a1.dtype)) * a1
    a2p = (1.0 + G.to(a2.dtype)) * a2
    C1p = _F._safe_sqrt(a1p * a1p + b1 * b1)
    C2p = _F._safe_sqrt(a2p * a2p + b2 * b2)
    # Hue is defined as zero for achromatic colors. Mask its operands too:
    # atan2(0, 0) otherwise poisons backward even when its hue is unused.
    h1p = torch.atan2(torch.where(C1p > 0, b1, 0.0),
                      torch.where(C1p > 0, a1p, 1.0)) * _R2D
    h1p = h1p + 360.0 * (h1p < 0.0)
    h2p = torch.atan2(torch.where(C2p > 0, b2, 0.0),
                      torch.where(C2p > 0, a2p, 1.0)) * _R2D
    h2p = h2p + 360.0 * (h2p < 0.0)

    # Branchless hue wraps: dh into (-180, 180], hbar into [0, 360).
    zero_prod = (C1p * C2p) == 0.0
    dh = h2p - h1p
    dh = dh - 360.0 * (dh > 180.0) + 360.0 * (dh < -180.0)
    dhp = torch.where(zero_prod, 0.0, dh)
    dHp = 2.0 * _F._safe_sqrt(C1p * C2p) * torch.sin(dhp * (math.pi / 360.0))

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0
    hs = h1p + h2p
    over = (h1p - h2p).abs() > 180.0
    hbar = hs / 2.0 + 180.0 * over * (1.0 - 2.0 * (hs >= 360.0))
    hbarp = torch.where(zero_prod, hs, hbar)

    h64 = hbarp.to(_F64)
    Cb64 = Cbarp.to(_F64)
    Lb64 = Lbarp.to(_F64)
    T = (
        1.0
        - 0.17 * torch.cos((h64 - 30.0) * _D2R)
        + 0.24 * torch.cos(2.0 * h64 * _D2R)
        + 0.32 * torch.cos((3.0 * h64 + 6.0) * _D2R)
        - 0.20 * torch.cos((4.0 * h64 - 63.0) * _D2R)
    )
    dRo = 30.0 * torch.exp(-(((h64 - 275.0) / 25.0) ** 2))
    Cp7 = Cb64**7
    Rc = 2.0 * _F._safe_sqrt(Cp7 / (Cp7 + _P25_7))
    Ld = Lb64 - 50.0
    SL = 1.0 + (0.015 * Ld * Ld) / torch.sqrt(20.0 + Ld * Ld)
    SC = 1.0 + 0.045 * Cb64
    SH = 1.0 + 0.015 * Cb64 * T
    RT = -torch.sin(2.0 * dRo * _D2R) * Rc

    l_term = (L2 - L1).to(_F64) / (kL * SL)
    c_term = (C2p - C1p).to(_F64) / (kC * SC)
    h_term = dHp.to(_F64) / (kH * SH)
    return _F._safe_sqrt(
        (l_term * l_term + c_term * c_term + h_term * h_term
         + RT * c_term * h_term).clamp_min(0.0)
    )


def _de00_fn(device):
    dts = device.type if isinstance(device, torch.device) else str(device)
    return _F._maybe_compile(_de00_planes, f"ciede2000:de00:{dts}", dynamic=True)


def _rgb_pair_to_demap(x4, y4, inv_l: float, cr, cg, cb,
                       kL: float, kC: float, kH: float):
    """Single fused RGB->Lab->dE region for the image path."""
    lab1 = _convert_one(x4, inv_l, cr, cg, cb)
    lab2 = _convert_one(y4, inv_l, cr, cg, cb)
    return _de00_planes(
        lab1[:, 0:1], lab1[:, 1:2], lab1[:, 2:3],
        lab2[:, 0:1], lab2[:, 1:2], lab2[:, 2:3],
        kL, kC, kH,
    )


def _rgb_demap_fn(device):
    dts = device.type if isinstance(device, torch.device) else str(device)
    return _F._maybe_compile(_rgb_pair_to_demap, f"ciede2000:rgb_demap:{dts}",
                             dynamic=True)


def delta_e_00(
    lab1: torch.Tensor,
    lab2: torch.Tensor,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> torch.Tensor:
    """Elementwise CIEDE2000 over ``(..., 3)`` Lab tensors (Sharma et al.)."""
    if lab1.shape != lab2.shape:
        raise ValueError(f"shape mismatch: {tuple(lab1.shape)} vs {tuple(lab2.shape)}")
    if lab1.shape[-1] != 3:
        raise ValueError(f"need (..., 3) Lab, got {tuple(lab1.shape)}")
    cdt = torch.promote_types(lab1.dtype, lab2.dtype)
    if cdt not in (torch.float32, torch.float64):
        cdt = torch.float32
    f1, f2 = lab1.to(cdt), lab2.to(cdt)
    with _F._no_autocast(f1):
        de = _de00_fn(f1.device)(
            f1[..., 0], f1[..., 1], f1[..., 2],
            f2[..., 0], f2[..., 1], f2[..., 2],
            float(kL), float(kC), float(kH),
        )
    return de.to(cdt)


def ciede2000(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    reduction: str = "mean",
    return_map: bool = False,
    dtype: Optional[torch.dtype] = None,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> torch.Tensor:
    """Mean CIEDE2000 colour difference between two sRGB images.

    Lower is better; 0 is identical. A just-noticeable difference is ~1,
    and fully different primaries score in the dozens.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    x4, y4 = _F._check_pair(x, y)
    if x4.shape[1] != 3:
        raise ValueError(f"ciede2000 needs 3-channel RGB, got {x4.shape[1]} channels")
    L = float(data_range) if data_range is not None else _F._infer_data_range(x4)
    wdt = _F._work_dtype(x4, dtype)

    with _F._no_autocast(x4):
        dmap = _rgb_demap_fn(x4.device)(
            x4.to(wdt), y4.to(wdt), 1.0 / L, *_xyz_cols(x4.device, wdt),
            float(kL), float(kC), float(kH),
        )
    if return_map:
        return dmap.to(wdt)
    per_image = dmap.mean(dim=(1, 2, 3), dtype=torch.float64)
    return per_image if reduction == "none" else per_image.mean()


# --------------------------------------------------------------------------- #
# Sharma et al. (2005) supplementary test data (kept for reference; the
# self-test lives in proposals/ciede2000.py).
# --------------------------------------------------------------------------- #

_SHARMA_LAB1 = [
    [50.0000, 2.6772, -79.7751],
    [50.0000, 2.5000, 0.0000],
]
_SHARMA_LAB2 = [
    [50.0000, 0.0000, -82.7485],
    [50.0000, 0.0000, -2.5000],
]
_SHARMA_DE = [2.0425, 4.3065]
