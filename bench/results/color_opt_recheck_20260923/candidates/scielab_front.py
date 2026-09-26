"""S-CIELab spatial colour difference (Zhang & Wandell 1996).

Torch-only NCHW pipeline (no extensions, no SciPy)::

    sRGB (NCHW) -> linear RGB -> XYZ (D65) -> opponent (Poirson & Wandell)
      -> per-channel separable Gaussian CSF filtering (PPD-scaled)
      -> XYZ -> CIELab -> per-pixel CIE76 delta-E -> mean aggregation.

Opponent matrix, CSF spreads (including the 7.0-degree achromatic lobe), the
``gauss.m`` halfwidth parametrisation, and the D65 white / ``xyz2lab.m``
constants follow the 1996 MATLAB implementation (see proposals/scielab.py).
Input conventions and compile policy follow :mod:`frame_analytics.functional`
(helpers reused): NCHW RGB, ``data_range`` inference, ``reduction`` in
``{"mean", "none"}``, float32 work dtype unless the input is float64, float64
final reduction.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import functional as _F
from .functional import set_compile_enabled

__all__ = [
    "scielab",
    "set_compile_enabled",
    "SCIELAB_CSF",
    "SCIELAB_OPP",
    "PPD_DEFAULT",
    "WHITE_D65",
    "SRGB_TO_XYZ",
]

#: Default viewing resolution, in pixels per degree of visual angle.
PPD_DEFAULT = 30.0

#: D65 reference white used by the original ``xyz2lab.m``.
WHITE_D65 = (95.05, 100.0, 108.88)

#: sRGB (linear, D65) -> XYZ, IEC 61966-2-1, for 0..1 inputs to 0..1 outputs
#: (scaled by 100 at use time to match the opponent matrix / Lab convention).
SRGB_TO_XYZ = (
    (0.4123907993, 0.3575843390, 0.1804807884),
    (0.2126390059, 0.7151686788, 0.0721923154),
    (0.0193308187, 0.1191947798, 0.9505321524),
)

#: XYZ (0..100) -> Poirson & Wandell opponent space; ``cmatrix('xyz2opp', 2)``.
#: Rows are (BW achromatic, RG red-green, BY blue-yellow).
SCIELAB_OPP = (
    (0.2787336, 0.7218031, -0.1065520),
    (-0.4487736, 0.2898056, 0.0771569),
    (0.0859513, -0.5899859, 0.5011089),
)

#: Per-opponent-channel CSF as ``(spread_deg, weight)`` pairs (normalised set).
SCIELAB_CSF: Tuple[Tuple[Tuple[float, float], ...], ...] = (
    ((0.05, 1.00327), (0.225, 0.114416), (7.0, -0.117686)),
    ((0.0685, 0.616725), (0.826, 0.383275)),
    ((0.092, 0.567885), (0.6451, 0.432115)),
)

#: ``2*sqrt(2*ln2)``: FWHM-to-sigma divisor for the ``gauss.m`` parametrisation.
_FWHM_TO_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

_kernel_cache: dict = {}
_mat_cache: dict = {}


def _colour_mats(dtype, device):
    """``(sRGB->opponent 0..100, opp2xyz, white)`` as device tensors."""
    key = (str(dtype), str(device))
    hit = _mat_cache.get(key)
    if hit is not None:
        return hit
    s = torch.tensor(SRGB_TO_XYZ, dtype=torch.float64)
    o = torch.tensor(SCIELAB_OPP, dtype=torch.float64)
    oi = torch.linalg.inv(o)
    wn = torch.tensor(WHITE_D65, dtype=torch.float64)
    mats = (((o @ s) * 100.0).to(dtype=dtype, device=device),
            oi.to(dtype=dtype, device=device),
            wn.to(dtype=dtype, device=device))
    _mat_cache[key] = mats
    return mats


def _front_rgb_to_opp(stacked01: torch.Tensor,
                      rgb_to_opp: torch.Tensor) -> torch.Tensor:
    """Linearise (sRGB EOTF) and project to opponent space."""
    lo = stacked01 <= 0.04045
    lin = torch.where(lo, stacked01 / 12.92, ((stacked01 + 0.055) / 1.055).pow(2.4))
    return torch.einsum("ij,njhw->nihw", rgb_to_opp, lin)


def _gauss_halfwidth_kernel(halfwidth_px: float, width: int,
                            dtype, device) -> torch.Tensor:
    """1-D Gaussian in the ``gauss.m`` halfwidth parametrisation."""
    alpha = 2.0 * math.sqrt(math.log(2.0)) / (halfwidth_px - 1.0)
    r = (width - 1) / 2.0
    coords = torch.arange(width, dtype=torch.float64, device=device) - r
    g = torch.exp(-(alpha * coords) ** 2)
    return (g / g.sum()).to(dtype=dtype)


def _component_width(halfwidth_px: float, max_width: int) -> int:
    """Odd support for one Gaussian: ``6*sigma`` tails, capped by the window."""
    if halfwidth_px <= 1.0 + 1e-9:
        return 1
    sigma = (halfwidth_px - 1.0) / _FWHM_TO_SIGMA
    w = int(math.ceil(6.0 * sigma)) * 2 + 1
    return max(1, min(w, max_width))


def _csf_kernels(ppd: float, max_width: int, dtype, device):
    """Per-channel ``[(weight, kernel1d, pad)]`` lists (``pad == 0``: delta)."""
    key = (round(float(ppd), 6), max_width, str(dtype), str(device))
    hit = _kernel_cache.get(key)
    if hit is not None:
        return hit
    chans = []
    for pairs in SCIELAB_CSF:
        comps = []
        for _s, _w in pairs:
            w = _component_width(_s * ppd, max_width)
            if w <= 1:
                k = None
            else:
                k = _gauss_halfwidth_kernel(_s * ppd, w, dtype, device)
            comps.append((_w, k, (w - 1) // 2))
        chans.append(comps)
    _kernel_cache[key] = chans
    return chans


def _support_width(ppd: float, h: int, w: int, cap: int = 255) -> int:
    """Odd filter support covering one degree of visual angle."""
    width = int(math.ceil(ppd / 2.0)) * 2 - 1
    width = max(width, 1)
    width = min(width, cap)
    side = min(h, w)
    if width > side:
        width = side if side % 2 else side - 1
    return max(width, 1)


def _separable_crop(padded: torch.Tensor, k: torch.Tensor, trim: int,
                    h: int, w: int) -> torch.Tensor:
    """Horizontal + vertical valid pass, then center-crop ``trim`` px (a view)."""
    kh = k.view(1, 1, 1, -1)
    kv = k.view(1, 1, -1, 1)
    with _F._no_autocast(padded):
        out = F.conv2d(padded, kh)
        out = F.conv2d(out, kv)
    if trim:
        out = out.narrow(2, trim, h).narrow(3, trim, w)
    return out


def _filter_stacked(opp: torch.Tensor, ppd: float) -> torch.Tensor:
    """CSF-filter the stacked ``(2N,3,H,W)`` opponent pair, separably."""
    _, _, h, w = opp.shape
    max_width = _support_width(ppd, h, w)
    comps = _csf_kernels(ppd, max_width, opp.dtype, opp.device)
    outs = []
    for c in range(3):
        raw = opp[:, c:c + 1]
        chan = comps[c]
        P = max(p for _, _, p in chan)
        padded = F.pad(raw, (P, P, P, P), mode="reflect") if P else raw
        acc = None
        for weight, k, p in chan:
            if k is None:
                f = raw
            else:
                f = _separable_crop(padded, k, P - p, h, w)
            acc = weight * f if acc is None else acc + weight * f
        outs.append(acc if acc is not None else raw)
    return torch.cat(outs, dim=1)


def _labs_of_pair(f1: torch.Tensor, f2: torch.Tensor, Minv: torch.Tensor,
                  white: torch.Tensor):
    """Filtered opponent halves -> ``(Lab1, Lab2)``; ``xyz2lab.m`` formula."""
    xyz1 = torch.einsum("ij,njhw->nihw", Minv, f1)
    xyz2 = torch.einsum("ij,njhw->nihw", Minv, f2)
    return _to_lab(xyz1, white), _to_lab(xyz2, white)


def _to_lab(xyz: torch.Tensor, white: torch.Tensor) -> torch.Tensor:
    t = (xyz / white.view(1, 3, 1, 1)).clamp_min(0.0)
    thr = 0.008856
    f = torch.where(t > thr, t.pow(1.0 / 3.0), 7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    y = t[:, 1:2]
    L = torch.where(y > thr, 116.0 * fy - 16.0, 903.3 * y)
    return torch.cat([L, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=1)


def _delta_map(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    return ((lab1 - lab2).pow(2).sum(dim=1, keepdim=True)).sqrt()


def _epi_lab_map(f1, f2, Minv, white):
    lab1, lab2 = _labs_of_pair(f1, f2, Minv, white)
    return _delta_map(lab1, lab2)


def _epi_lab_per_image(f1, f2, Minv, white):
    lab1, lab2 = _labs_of_pair(f1, f2, Minv, white)
    return _delta_map(lab1, lab2).reshape(lab1.shape[0], -1).mean(
        dim=1, dtype=torch.float64)


def _epi_lab_mean(f1, f2, Minv, white):
    lab1, lab2 = _labs_of_pair(f1, f2, Minv, white)
    return _delta_map(lab1, lab2).mean(dtype=torch.float64)


def scielab(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    ppd: float = PPD_DEFAULT,
    pixels_per_degree: Optional[float] = None,
    reduction: str = "mean",
    return_map: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """S-CIELab colour difference (Zhang & Wandell 1996). Higher is worse.

    ``ppd`` (alias ``pixels_per_degree``) scales every CSF spread
    (``spread_px = spread_deg * ppd``).
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    if pixels_per_degree is not None:
        ppd = pixels_per_degree
    if not ppd or ppd <= 0:
        raise ValueError(f"ppd must be > 0, got {ppd}")
    x4, y4 = _F._check_pair(x, y)
    if x4.shape[1] != 3:
        raise ValueError(f"scielab needs RGB input, got {x4.shape[1]} channels")
    L = float(data_range) if data_range is not None else _F._infer_data_range(x4)
    wdt = _F._work_dtype(x4, dtype)
    rgb_to_opp, Minv, white = _colour_mats(wdt, x4.device)

    stacked = torch.cat([x4, y4], dim=0).to(wdt) / L
    front = _F._maybe_compile(_front_rgb_to_opp, "scielab:front")
    opp = front(stacked.clamp(0.0, 1.0), rgb_to_opp)
    filt = _filter_stacked(opp, float(ppd))
    n = x4.shape[0]
    f1, f2 = filt[:n], filt[n:]

    if return_map:
        epi = _F._maybe_compile(_epi_lab_map, "scielab:epi_map")
    elif reduction == "none":
        epi = _F._maybe_compile(_epi_lab_per_image, "scielab:epi_per")
    else:
        epi = _F._maybe_compile(_epi_lab_mean, "scielab:epi_mean")
    out = epi(f1, f2, Minv, white)
    return out if return_map else (out if reduction == "none" else out.mean())
