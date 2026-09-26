"""Feature Similarity Index (FSIM / FSIMc), Zhang et al. 2011.

Portable PyTorch implementation built on :mod:`frame_analytics.functional`
helpers (NCHW handling, ``data_range`` inference, luma / border-crop
conventions, work-dtype selection, autocast guard, compile-with-fallback).

Spec (preserved from ``proposals/fsim.py``):

* phase-congruency PC2 global ratio (``sum_o Energy_o / sum_o sumAn_o``),
  no per-orientation normalisation, no spread weighting;
* ``F = max(1, floor(min(H,W)/256 + 0.5))`` zero-padded box-mean downsample
  (MATLAB round-half-away) on Y (and I/Q for FSIMc);
* Scharr gradients (``[3,10,3]/16`` separable) on downsampled luma;
* ``T1 = 0.85`` verbatim, ``T2 = 160`` (0..255 units);
* FSIMc chroma term ``real((S_I * S_Q)^lambda)``, ``T3 = T4 = 200``,
  ``lambda = 0.03``;
* flat images (zero PC weight) pool to NaN per spec, never a fallback.

Higher is better, 1.0 is identical (non-flat).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from . import functional as _fn

__all__ = ["fsim", "fsimc"]

_T1 = 0.85
_T2 = 160.0
_T3 = 200.0
_T4 = 200.0
_LAMBDA = 0.03

_LP_CUTOFF = 0.45
_LP_SHARPNESS = 15
_NOISE_K = 2.0
_PC_NOISE_SCALE = 1.7

_NSCALE = 4
_NORIENT = 4
_MIN_WAVE = 6.0
_MULT = 2.0
_SIGMA_ONF = 0.55
_DTHETA_ON_SIGMA = 1.2
_PC_EPS = 1e-4

_YIQ = ((0.299, 0.587, 0.114),
        (0.596, -0.274, -0.322),
        (0.211, -0.523, 0.312))


def _to_yiq(x: torch.Tensor):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    m = _YIQ
    return (m[0][0] * r + m[0][1] * g + m[0][2] * b,
            m[1][0] * r + m[1][1] * g + m[1][2] * b,
            m[2][0] * r + m[2][1] * g + m[2][2] * b)


_scharr1d_cache: dict = {}


@torch.inference_mode(False)
def _scharr1d(device, dtype):
    key = (str(device), dtype)
    k = _scharr1d_cache.get(key)
    if k is None:
        diff = torch.tensor([1.0, 0.0, -1.0], dtype=dtype, device=device)
        smooth = torch.tensor([3.0, 10.0, 3.0], dtype=dtype, device=device) / 16.0
        k = (diff.view(1, 1, 1, 3), smooth.view(1, 1, 1, 3),
             diff.view(1, 1, 3, 1), smooth.view(1, 1, 3, 1))
        _scharr1d_cache[key] = k
    return k


def _gradient_magnitude_stacked(gray: torch.Tensor) -> torch.Tensor:
    hd, hs, vd, vs = _scharr1d(gray.device, gray.dtype)
    with _fn._no_autocast(gray):
        gx = F.conv2d(F.conv2d(gray, vs, padding=(1, 0)), hd, padding=(0, 1))
        gy = F.conv2d(F.conv2d(gray, hs, padding=(0, 1)), vd, padding=(1, 0))
    return torch.sqrt(gx * gx + gy * gy)


_planes_cache: dict = {}
_noise_cache: dict = {}


@torch.inference_mode(False)
def _planes(h: int, w: int, device, dtype):
    key = (h, w, str(device), dtype)
    p = _planes_cache.get(key)
    if p is None:
        fy = torch.fft.fftfreq(h, d=1.0).to(device=device, dtype=dtype)
        fx = torch.fft.fftfreq(w, d=1.0).to(device=device, dtype=dtype)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx * xx + yy * yy)
        angle = torch.atan2(-yy, xx)
        lp = 1.0 / (1.0 + (radius / _LP_CUTOFF) ** (2 * _LP_SHARPNESS))
        log_radius = torch.log(radius.clamp_min(1e-9))
        log_sigma = math.log(_SIGMA_ONF)
        radial = []
        for s in range(_NSCALE):
            fo = 1.0 / (_MIN_WAVE * (_MULT ** s))
            lg = torch.exp(-((log_radius - math.log(fo)) ** 2)
                           / (2.0 * log_sigma ** 2))
            lg[radius == 0] = 0.0
            radial.append(lg * lp)
        radial = torch.stack(radial)
        theta_sigma = math.pi / _NORIENT / _DTHETA_ON_SIGMA
        spreads = []
        for o in range(_NORIENT):
            orient = o * math.pi / _NORIENT
            delta = angle - orient
            d = torch.atan2(delta.sin(), delta.cos()).abs()
            spreads.append(torch.exp(-(d * d) / (2.0 * theta_sigma * theta_sigma)))
        spreads = torch.stack(spreads)
        p = (radial, spreads)
        _planes_cache[key] = p
    return p


@torch.inference_mode(False)
def _noise_scalars(h: int, w: int, device, dtype):
    key = (h, w, str(device), dtype)
    n = _noise_cache.get(key)
    if n is None:
        radial, spreads = _planes(h, w, device, dtype)
        em, s2, cc = [], [], []
        for o in range(_NORIENT):
            hbank = radial * spreads[o]
            em.append((hbank[0] * hbank[0]).sum())
            # FeatureSIM.m estimates cross-scale noise from the real spatial
            # impulse responses, not the complete frequency-domain energy.
            impulse = torch.fft.ifft2(hbank).real * math.sqrt(h * w)
            s2.append((impulse * impulse).sum())
            g = impulse.reshape(_NSCALE, -1)
            cc.append(((g.sum(dim=0) ** 2).sum() - (g * g).sum()) / 2.0)
        n = (torch.stack(em), torch.stack(s2), torch.stack(cc))
        _noise_cache[key] = n
    return n


def _noise_threshold(a0: torch.Tensor, em: torch.Tensor, s2: torch.Tensor,
                     cc: torch.Tensor, orient: int) -> torch.Tensor:
    m = a0.shape[0]
    samples = (a0 * a0).reshape(m, -1)
    # MATLAB's median averages the middle two values for an even-sized map.
    median_e2 = samples.quantile(0.5, dim=1)
    npow = (median_e2 / math.log(2.0)) / em[orient]
    e2 = npow * (2.0 * s2[orient] + 4.0 * cc[orient])
    tau = torch.sqrt(e2 / 2.0)
    emean = tau * math.sqrt(math.pi / 2.0)
    esigma = torch.sqrt((2.0 - math.pi / 2.0) * tau * tau)
    return ((emean + _NOISE_K * esigma) / _PC_NOISE_SCALE).view(m, 1, 1)


def _downsample_factor(h: int, w: int) -> int:
    return max(1, math.floor(min(h, w) / 256.0 + 0.5))


def _matlab_box_down(t: torch.Tensor, f: int) -> torch.Tensor:
    p_top = f - 1 - f // 2
    p_bot = f // 2
    y = F.avg_pool2d(F.pad(t, (p_top, p_bot, p_top, p_bot)), f, stride=1)
    return y[:, :, ::f, ::f]


_PC_BATCHED_MAX_STACKED = 2 * 512 * 512
_PC_CHUNK = 1


def _phase_congruency_stacked(gray: torch.Tensor,
                              pc_mode: str = "auto") -> torch.Tensor:
    if pc_mode not in ("auto", "batched", "chunked"):
        raise ValueError(f"pc_mode must be 'auto', 'batched' or 'chunked', "
                         f"got {pc_mode!r}")
    m, _, h, w = gray.shape
    batched = (pc_mode == "batched"
               or (pc_mode == "auto" and m * h * w <= _PC_BATCHED_MAX_STACKED))
    radial, spreads = _planes(h, w, gray.device, gray.dtype)
    em, s2, cc = _noise_scalars(h, w, gray.device, gray.dtype)
    cxty = (torch.complex64 if gray.dtype != torch.float64
            else torch.complex128)
    spec = torch.fft.fft2(gray.squeeze(1))

    energy_all = torch.zeros((m, h, w), device=gray.device, dtype=gray.dtype)
    an_all = torch.zeros((m, h, w), device=gray.device, dtype=gray.dtype)
    if batched:
        chunks = [(0, m)]
    else:
        chunks = [(i, min(i + _PC_CHUNK, m)) for i in range(0, m, _PC_CHUNK)]
    for (lo, hi) in chunks:
        sp = spec[lo:hi]
        for o in range(_NORIENT):
            bank_o = (radial * spreads[o]).to(cxty)
            r = torch.fft.ifft2(sp.unsqueeze(1) * bank_o.unsqueeze(0))
            e, od = r.real, r.imag
            sum_e = e[:, 0] + e[:, 1] + e[:, 2] + e[:, 3]
            sum_o = od[:, 0] + od[:, 1] + od[:, 2] + od[:, 3]
            xenergy = torch.sqrt(sum_e * sum_e + sum_o * sum_o) + _PC_EPS
            mean_e = sum_e / xenergy
            mean_o = sum_o / xenergy
            # Batch scale arithmetic on CUDA to cut kernel launches. The
            # smaller per-scale temporaries are faster for larger CPU batches.
            if gray.is_cuda:
                me, mo = mean_e.unsqueeze(1), mean_o.unsqueeze(1)
                energy_o = (e * me + od * mo - (e * mo - od * me).abs()).sum(dim=1)
                amplitudes = torch.sqrt(e * e + od * od)
                sum_an = amplitudes.sum(dim=1)
                a0 = amplitudes[:, 0]
            else:
                energy_o = torch.zeros_like(sum_e)
                sum_an = torch.zeros_like(sum_e)
                for s in range(_NSCALE):
                    es, os = e[:, s], od[:, s]
                    energy_o = energy_o + (es * mean_e + os * mean_o
                                           - (es * mean_o - os * mean_e).abs())
                    an_s = torch.sqrt(es * es + os * os)
                    sum_an = sum_an + an_s
                    if s == 0:
                        a0 = an_s
            thresh = _noise_threshold(a0, em, s2, cc, o)
            energy_all[lo:hi] = (energy_all[lo:hi]
                                 + (energy_o - thresh).clamp_min(0.0))
            an_all[lo:hi] = an_all[lo:hi] + sum_an
    return (torch.where(an_all > 0, energy_all / an_all,
                        torch.zeros((), device=gray.device,
                                    dtype=gray.dtype)).unsqueeze(1))


def _chroma_factor(si: torch.Tensor, sq: torch.Tensor) -> torch.Tensor:
    base = si * sq
    mag = torch.pow(base.abs(), _LAMBDA)
    return torch.where(base < 0, mag * math.cos(math.pi * _LAMBDA), mag)


def _pool(numer_sum, denom_sum):
    nan = torch.full_like(denom_sum, float("nan"))
    return torch.where(denom_sum > 0, numer_sum / denom_sum.clamp_min(1e-300),
                       nan)


def _epi_gray(pc1, pc2, g1, g2):
    spc = (2.0 * pc1 * pc2 + _T1) / (pc1 * pc1 + pc2 * pc2 + _T1)
    sg = (2.0 * g1 * g2 + _T2) / (g1 * g1 + g2 * g2 + _T2)
    sl = spc * sg
    pcm = torch.maximum(pc1, pc2)
    num = (sl * pcm).sum(dim=(1, 2, 3), dtype=torch.float64)
    den = pcm.sum(dim=(1, 2, 3), dtype=torch.float64)
    return _pool(num, den)


def _epi_chroma(pc1, pc2, g1, g2, i1, i2, q1, q2):
    spc = (2.0 * pc1 * pc2 + _T1) / (pc1 * pc1 + pc2 * pc2 + _T1)
    sg = (2.0 * g1 * g2 + _T2) / (g1 * g1 + g2 * g2 + _T2)
    si = (2.0 * i1 * i2 + _T3) / (i1 * i1 + i2 * i2 + _T3)
    sq = (2.0 * q1 * q2 + _T4) / (q1 * q1 + q2 * q2 + _T4)
    sl = spc * sg * _chroma_factor(si, sq)
    pcm = torch.maximum(pc1, pc2)
    num = (sl * pcm).sum(dim=(1, 2, 3), dtype=torch.float64)
    den = pcm.sum(dim=(1, 2, 3), dtype=torch.float64)
    return _pool(num, den)


def _sim_map_gray(pc1, pc2, g1, g2):
    spc = (2.0 * pc1 * pc2 + _T1) / (pc1 * pc1 + pc2 * pc2 + _T1)
    sg = (2.0 * g1 * g2 + _T2) / (g1 * g1 + g2 * g2 + _T2)
    return spc * sg


def fsim(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    chromatic: bool = False,
    reduction: str = "mean",
    return_map: bool = False,
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
    pc_mode: str = "auto",
) -> torch.Tensor:
    """Feature Similarity Index (higher is better, 1.0 is identical).

    ``chromatic=True`` selects FSIMc (Y for structure, I/Q chroma term;
    requires 3-channel input). ``luma``/``crop_border`` follow
    :mod:`frame_analytics.functional`: ``luma`` projects 3-channel input to
    luma first (forcing the grayscale path; mutually exclusive with
    ``chromatic``), ``crop_border`` drops that many pixels per edge first.
    Flat images pool to NaN per spec.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    x4, y4, L, _ = _fn._prep(x, y, None, crop_border, data_range)
    if not L > 0:
        raise ValueError(f"data_range must be > 0, got {data_range!r}")
    wdt = _fn._work_dtype(x4, dtype)
    if luma not in (None, False):
        if chromatic:
            raise ValueError("chromatic=True is mutually exclusive with luma=...")
        if x4.shape[1] == 3:
            x4, y4 = _fn._apply_luma(x4, y4, _fn._resolve_luma(luma), L, wdt)
    c = x4.shape[1]
    if c not in (1, 3):
        raise ValueError(f"expected 1 or 3 channels, got {c}")
    if chromatic and c != 3:
        raise ValueError("chromatic=True (FSIMc) needs a 3-channel input")

    scale = 255.0 / L
    if c == 3:
        xw, yw = x4.to(wdt) * scale, y4.to(wdt) * scale
        if chromatic:
            y1, i1, q1 = _to_yiq(xw)
            y2, i2, q2 = _to_yiq(yw)
        else:
            y1 = _fn.rgb_to_luma(xw, "bt601", data_range=255.0, dtype=wdt)
            y2 = _fn.rgb_to_luma(yw, "bt601", data_range=255.0, dtype=wdt)
            i1 = i2 = q1 = q2 = None
    else:
        y1, y2 = x4.to(wdt) * scale, y4.to(wdt) * scale
        i1 = i2 = q1 = q2 = None

    ds = _downsample_factor(y1.shape[-2], y1.shape[-1])
    if ds > 1:
        y1, y2 = _matlab_box_down(y1, ds), _matlab_box_down(y2, ds)
        if chromatic:
            i1, i2 = _matlab_box_down(i1, ds), _matlab_box_down(i2, ds)
            q1, q2 = _matlab_box_down(q1, ds), _matlab_box_down(q2, ds)

    stacked = torch.cat([y1, y2], dim=0)
    pc = _phase_congruency_stacked(stacked, pc_mode=pc_mode)
    gm = _gradient_magnitude_stacked(stacked)
    pc1, pc2 = pc.chunk(2, dim=0)
    gm1, gm2 = gm.chunk(2, dim=0)

    if return_map:
        s_map = _sim_map_gray(pc1, pc2, gm1, gm2)
        if chromatic:
            si = (2.0 * i1 * i2 + _T3) / (i1 * i1 + i2 * i2 + _T3)
            sq = (2.0 * q1 * q2 + _T4) / (q1 * q1 + q2 * q2 + _T4)
            s_map = s_map * _chroma_factor(si, sq)
        return s_map

    if chromatic:
        epi = _fn._maybe_compile(_epi_chroma, "fsim:epi:chroma")
        per_image = epi(pc1, pc2, gm1, gm2, i1, i2, q1, q2)
    else:
        epi = _fn._maybe_compile(_epi_gray, "fsim:epi:gray")
        per_image = epi(pc1, pc2, gm1, gm2)
    return per_image.mean() if reduction == "mean" else per_image


def fsimc(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    reduction: str = "mean",
    return_map: bool = False,
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
    pc_mode: str = "auto",
) -> torch.Tensor:
    """FSIMc (colour FSIM): ``fsim(..., chromatic=True)``."""
    if luma not in (None, False):
        raise ValueError("fsimc is a colour metric; pass luma=None")
    return fsim(x, y, data_range=data_range, chromatic=True,
                reduction=reduction, return_map=return_map, dtype=dtype,
                luma=None, crop_border=crop_border, pc_mode=pc_mode)
