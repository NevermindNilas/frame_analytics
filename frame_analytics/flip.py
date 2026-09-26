"""FLIP difference evaluator (Andersson et al. 2020).

Torch-only NCHW implementation. Input conventions follow
:mod:`frame_analytics.functional` (helpers reused).

Pipeline: sRGB EOTF decode -> linear RGB -> XYZ (D65) -> YCxCz (linearised
CIELAB) -> per-channel CSF spatial filtering -> back to linear RGB, clamped
to the unit cube -> CIELAB -> Hunt adjustment -> HyAB distance -> ``^qc``
(``qc=0.7``) -> piecewise remap via ``(pc, pt) = (0.4, 0.95)`` normalised by
``cmax``. Feature: normalised achromatic channel -> separable 1st/2nd
derivative Gaussian responses (``w = 0.082``), ``((1/sqrt2) * df)^qf``.
Final error: ``delta_c^(1 - delta_f)``; mean-pooled. Lower is better, 0 for
identical inputs (exactly), saturates at 1.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from . import functional as _F
from .functional import set_compile_enabled

__all__ = ["flip", "set_compile_enabled"]

#: FLIP paper / NVlabs torch-port constants (fitted, not tuned here).
_QC, _QF, _PC, _PT = 0.7, 0.5, 0.4, 0.95

#: D65 white point (NVlabs torch port) and its inverse for normalisation.
_WHITE = (0.950428545, 1.0, 1.088900371)
_INV_WHITE = (1.052156925, 1.0, 0.918357670)

#: CSF Gaussian parameters per channel: (a1, b1, a2, b2). A/RG are single
#: Gaussians (a2 == 0); BY is a sum of two (NVlabs torch port).
_CSF = {
    "A": (1.0, 0.0047, 0.0, 1e-5),
    "RG": (1.0, 0.0053, 0.0, 1e-5),
    "BY": (34.1, 0.04, 13.5, 0.025),
}
_CSF_MAX_B = 0.04  # max b over channels: shared kernel radius per ppd

#: Feature-detection edge width in degrees (2x std of the human edge filter).
_W_EDGE_DEG = 0.082


def _use_separable() -> bool:
    """Env kill-switch: ``FLIP_SEPARABLE=0`` forces full-2-D convolutions."""
    import os

    return os.environ.get("FLIP_SEPARABLE", "1") != "0"


def _maybe(fn, key: str, device_type: str):
    """Per-(stage, device-type) wrapper over :func:`functional._maybe_compile`."""
    return _F._maybe_compile(fn, f"flip:{key}:{device_type}")


def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    """Exact piecewise sRGB EOTF decode; input in [0, 1]."""
    lo = c / 12.92
    hi = ((c.clamp_min(0.0) + 0.055) / 1.055).pow(2.4)
    return torch.where(c <= 0.04045, lo, hi)


def _rgb_xyz_mats(device, dtype):
    """linRGB<->XYZ D65 matrices (NVlabs rational coefficients, float64-built)."""
    fwd = torch.tensor(
        [[10135552 / 24577794, 8788810 / 24577794, 4435075 / 24577794],
         [2613072 / 12288897, 8788810 / 12288897, 887015 / 12288897],
         [1425312 / 73733382, 8788810 / 73733382, 70074185 / 73733382]],
        device=device, dtype=torch.float64,
    )
    return fwd.to(dtype=dtype), torch.linalg.inv(fwd).to(dtype=dtype)


_mat_cache: dict = {}


@torch.inference_mode(False)
def _mats(device, dtype):
    """linRGB<->XYZ matrices + white points, cached per (device, dtype)."""
    key = (str(device), str(dtype))
    hit = _mat_cache.get(key)
    if hit is not None:
        return hit
    m, mi = _rgb_xyz_mats(device, dtype)
    out = (m, mi,
           torch.tensor(_INV_WHITE, device=device, dtype=dtype),
           torch.tensor(_WHITE, device=device, dtype=dtype))
    _mat_cache[key] = out
    return out


def _fwd_matrix_input(x01: torch.Tensor, mat, inv_w, exposure_gain: float):
    """Fused sRGB[0,1] -> YCxCz (decode, gain, XYZ, white-norm, lin-Lab)."""
    lin = _srgb_to_linear(x01)
    if exposure_gain != 1.0:
        lin = (lin * exposure_gain).clamp(0.0, 1.0)
    n, _, h, w = lin.shape
    xyz = (mat @ lin.reshape(n, 3, -1)).reshape(n, 3, h, w)
    x = xyz[:, 0:1] * inv_w[0]
    y = xyz[:, 1:2] * inv_w[1]
    z = xyz[:, 2:3] * inv_w[2]
    return torch.cat([116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)], dim=1)


def _back_matrix_input(o: torch.Tensor, inv_mat, white):
    """Fused YCxCz -> linear RGB clamped to the unit cube."""
    y = (o[:, 0:1] + 16.0) / 116.0
    cx = o[:, 1:2] / 500.0
    cz = o[:, 2:3] / 200.0
    xyz = torch.cat([y + cx, y, y - cz], dim=1)
    xyz = xyz * white.view(1, 3, 1, 1).to(xyz.dtype)
    n, _, h, w = xyz.shape
    lin = (inv_mat @ xyz.reshape(n, 3, -1)).reshape(n, 3, h, w)
    return lin.clamp(0.0, 1.0)


def _lab_hunt(lin: torch.Tensor, mat, inv_w):
    """Fused linear RGB -> XYZ -> CIELAB -> Hunt-adjusted Lab."""
    n, _, h, w = lin.shape
    xyz = (mat @ lin.reshape(n, 3, -1)).reshape(n, 3, h, w)
    t = xyz * inv_w.view(1, 3, 1, 1).to(xyz.dtype)
    delta = 6.0 / 29.0
    cube = delta ** 3
    f = torch.where(t > cube, t.clamp_min(cube).pow(1.0 / 3.0),
                    (t / (3.0 * delta * delta)) + (4.0 / 29.0))
    Lv = 116.0 * f[:, 1:2] - 16.0
    av = 500.0 * (f[:, 0:1] - f[:, 1:2])
    bv = 200.0 * (f[:, 1:2] - f[:, 2:3])
    hunt = 0.01 * Lv
    return torch.cat([Lv, hunt * av, hunt * bv], dim=1)


def _cmax_qc() -> float:
    """``HyAB(Hunt(green), Hunt(blue))^qc`` as a python float (import-time)."""
    with torch.no_grad():
        dev = torch.device("cpu")
        m, _ = _rgb_xyz_mats(dev, torch.float64)
        iw = torch.tensor(_INV_WHITE, dtype=torch.float64)
        g = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        b = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        g[0, 1, 0, 0] = 1.0
        b[0, 2, 0, 0] = 1.0
        d = _lab_hunt(g, m, iw) - _lab_hunt(b, m, iw)
        hyab = d[:, 0:1].abs() + torch.sqrt((d[:, 1:3] ** 2).sum(dim=1,
                                                                keepdim=True))
        return float(hyab.pow(_QC).item())


_CMAX_QC = _cmax_qc()

_kernel_cache: dict = {}


def _csf_radius(ppd: float) -> int:
    """Shared kernel radius: ``ceil(3*sqrt(max_b/(2*pi^2))*ppd)`` (reference)."""
    import math

    return int(math.ceil(3.0 * math.sqrt(_CSF_MAX_B / (2.0 * math.pi ** 2))
                         * ppd))


@torch.inference_mode(False)
def _csf_1d(ppd: float, channel: str, device, dtype):
    """``(kernels, radius)``: one normalized 1-D kernel per Gaussian term."""
    import math

    key = ("csf", round(ppd, 6), channel, str(device), str(dtype))
    hit = _kernel_cache.get(key)
    if hit is not None:
        return hit
    a1, b1, a2, b2 = _CSF[channel]
    r = _csf_radius(ppd)
    xs = torch.arange(-r, r + 1, device=device, dtype=torch.float64) / ppd
    terms, masses = [], []
    for a, b in ((a1, b1), (a2, b2)):
        if a == 0:
            continue
        u = math.sqrt(a * math.sqrt(math.pi / b)) * torch.exp(
            -(math.pi ** 2) * xs * xs / b)
        masses.append(float(u.sum()) ** 2)  # 2-D mass of this Gaussian term
        terms.append((u / u.sum()).to(dtype=dtype))
    out = (terms, masses, r)
    _kernel_cache[key] = out
    return out


def _conv_rows(p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r = k.numel() // 2
    with _F._no_autocast(p):
        p = F.pad(p, (r, r, 0, 0), mode="replicate")
        return F.conv2d(p, k.view(1, 1, 1, -1))


def _conv_cols(p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r = k.numel() // 2
    with _F._no_autocast(p):
        p = F.pad(p, (0, 0, r, r), mode="replicate")
        return F.conv2d(p, k.view(1, 1, -1, 1))


def _sep2(p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Separable 2-D pass (full 2-D outer product under FLIP_SEPARABLE=0)."""
    if _use_separable():
        return _conv_cols(_conv_rows(p, k), k)
    r = k.numel() // 2
    k2 = (k[:, None] * k[None, :]).unsqueeze(0).unsqueeze(0)
    with _F._no_autocast(p):
        p = F.pad(p, (r, r, r, r), mode="replicate")
        return F.conv2d(p, k2)


def _filter_channel(p: torch.Tensor, ppd: float, channel: str) -> torch.Tensor:
    """CSF-filter one stacked ``(2N,1,H,W)`` channel (BY sums two terms)."""
    terms, masses, _ = _csf_1d(ppd, channel, p.device, p.dtype)
    if len(terms) == 1:
        return _sep2(p, terms[0])
    acc = masses[0] * _sep2(p, terms[0])
    for m, k in zip(masses[1:], terms[1:]):
        acc = acc + m * _sep2(p, k)
    return acc / sum(masses)


@torch.inference_mode(False)
def _feature_1d(ppd: float, kind: str, device, dtype):
    """``(d, g, radius)`` 1-D feature kernels with reference normalisation."""
    import math

    key = ("feat", round(ppd, 6), kind, str(device), str(dtype))
    hit = _kernel_cache.get(key)
    if hit is not None:
        return hit
    sd = 0.5 * _W_EDGE_DEG * ppd
    r = int(math.ceil(3.0 * sd))
    xs = torch.arange(-r, r + 1, device=device, dtype=torch.float64)
    g = torch.exp(-(xs * xs) / (2.0 * sd * sd))
    if kind == "edge":
        d = -xs * g
        d = d / (-float(d[d < 0].sum()))
    else:
        s = (xs * xs / (sd * sd) - 1.0) * g
        pos, neg = s.clamp_min(0.0), s.clamp_max(0.0)
        d = pos / float(pos.sum()) + neg / (-float(neg.sum()))
    g = g / g.sum()
    out = (d.to(dtype=dtype), g.to(dtype=dtype), r)
    _kernel_cache[key] = out
    return out


def _feature_mag(yy: torch.Tensor, ppd: float, kind: str) -> torch.Tensor:
    """Gradient-style magnitude of edge/point responses, ``(2N,1,H,W)``."""
    d, g, _ = _feature_1d(ppd, kind, yy.device, yy.dtype)
    if _use_separable():
        fx = _conv_cols(_conv_rows(yy, d), g)
        fy = _conv_cols(_conv_rows(yy, g), d)
    else:
        r = d.numel() // 2
        k2 = (d[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
        with _F._no_autocast(yy):
            yp = F.pad(yy, (r, r, r, r), mode="replicate")
            fx = F.conv2d(yp, k2)
            fy = F.conv2d(yp, k2.transpose(-1, -2))
    return _F._safe_sqrt(fx * fx + fy * fy)


def _epilogue(p: torch.Tensor, cmax: float, pc: float, pt: float):
    """Remap ``HyAB^qc`` to [0, 1]: [0, pc*cmax] -> [0, pt], rest -> (pt, 1]."""
    pccmax = pc * cmax
    return torch.where(p < pccmax, (pt / pccmax) * p,
                       pt + ((p - pccmax) / (cmax - pccmax)) * (1.0 - pt))


def _finish(dc: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
    """``dc^(1-df)`` with zero-colour-error defined as 0, clamped to [0, 1]."""
    return _F._safe_positive_power(dc, 1.0 - df).clamp(0.0, 1.0)


def flip(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    pixels_per_degree: float = 67.0,
    ppd: Optional[float] = None,
    exposure: float = 0.0,
    reduction: str = "mean",
    return_map: bool = False,
    dtype: Optional[torch.dtype] = None,
    eps: float = 0.0,
) -> torch.Tensor:
    """FLIP difference (Andersson et al. 2020). Lower is better.

    ``0`` for identical inputs (exactly); large differences saturate at 1.
    ``ppd`` is an alias for ``pixels_per_degree`` (viewing condition, must
    be > 0; default 67.0).
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    if ppd is not None:
        pixels_per_degree = ppd
    if pixels_per_degree is None or float(pixels_per_degree) <= 0:
        raise ValueError(
            f"pixels_per_degree must be > 0, got {pixels_per_degree!r}")
    ppdv = float(pixels_per_degree)
    x4, y4 = _F._check_pair(x, y)
    if x4.shape[1] != 3:
        raise ValueError(f"flip needs 3-channel RGB, got {x4.shape[1]} channels")
    L = float(data_range) if data_range is not None else _F._infer_data_range(x4)
    if L <= 0:
        raise ValueError(f"data_range must be > 0, got {L!r}")
    wdt = _F._work_dtype(x4, dtype)
    gain = 2.0 ** float(exposure) if exposure else 1.0
    fwd = _maybe(_fwd_matrix_input, "fwd", x4.device.type)
    back = _maybe(_back_matrix_input, "back", x4.device.type)
    lht = _maybe(_lab_hunt, "lab", x4.device.type)
    epi = _maybe(_epilogue, "epi", x4.device.type)

    mat, inv_mat, iw, wht = _mats(x4.device, wdt)

    both = torch.cat([x4.to(wdt), y4.to(wdt)], dim=0) / L
    with _F._no_autocast(both):
        opp = fwd(both.clamp(0.0, 1.0), mat, iw, gain)      # (2N,3,H,W) YCxCz

    filt = torch.cat([_filter_channel(opp[:, 0:1], ppdv, "A"),
                      _filter_channel(opp[:, 1:2], ppdv, "RG"),
                      _filter_channel(opp[:, 2:3], ppdv, "BY")], dim=1)
    with _F._no_autocast(filt):
        lin = back(filt, inv_mat, wht)
        hunt = lht(lin, mat, iw)                            # Hunt-Lab

    n = x4.shape[0]
    h1, h2 = hunt[:n], hunt[n:]
    d = h1 - h2
    hyab = d[:, 0:1].abs() + _F._safe_sqrt(
        ((d[:, 1:2] ** 2) + (d[:, 2:3] ** 2)).clamp_min(eps))
    dc = epi(_F._safe_positive_power(hyab, _QC),
             _CMAX_QC, _PC, _PT)                            # (N,1,H,W)

    yy = (opp[:, 0:1] + 16.0) / 116.0
    if yy.is_cuda:
        edge = _feature_mag(yy, ppdv, "edge")
        point = _feature_mag(yy, ppdv, "point")
        e1, e2 = edge[:n], edge[n:]
        p1, p2 = point[:n], point[n:]
    else:
        e1 = _feature_mag(yy[:n], ppdv, "edge")
        e2 = _feature_mag(yy[n:], ppdv, "edge")
        p1 = _feature_mag(yy[:n], ppdv, "point")
        p2 = _feature_mag(yy[n:], ppdv, "point")
    df = torch.maximum((e1 - e2).abs(), (p2 - p1).abs())
    if eps:
        df = df.clamp_min(eps)
    df = _F._safe_positive_power(0.7071067811865476 * df, _QF)

    err = _finish(dc, df).to(torch.float64)

    if return_map:
        return err
    per_image = err.mean(dim=(1, 2, 3), dtype=torch.float64)
    return per_image if reduction == "none" else per_image.mean()
