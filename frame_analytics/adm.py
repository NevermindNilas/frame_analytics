"""adm_like -- fast box-pyramid detail-loss score (ADM-inspired, NOT VMAF ADM).

Single public metric: :func:`adm_like` (``wavelet="haar"`` fast default,
``wavelet="db2_like"`` tap-matched path, NOT VMAF-exact). No ``adm`` alias --
the bare name wrongly implied VMAF ADM compatibility.

Status vs VMAF ADM: exact VMAF ADM (``libvmaf/src/feature/float_adm.c``,
``integer_adm.c``) is a 4-level Db2 (4-tap) decimated DWT
(``dec_lo = [-0.1294, 0.2241, 0.8365, 0.4830]``, QMF highpass), 2N-idx-1
mirror extension, Watson-97 / Li-SW contrast-sensitivity weighting, gain/decouple
(``R = gamma * X``) pipeline with per-subband contrast masking, pooled with
model-specific weights. This module implements none of those: the Db2 basis is
replaced by a Haar/box residual (``level - nearest_upsample(avg_pool(level))``),
masking is one scalar floor per scale, and pooling is a plain weighted mean.
Call it ``adm_like``.

Db2 path (``wavelet="db2_like"``; tap-matched, NOT VMAF-exact): separably-applied
Db2 4-tap DWT whose taps match PyWavelets ``db2`` (L2-norm-1)::

    dec_lo = [-0.1294, 0.2241, 0.8365, 0.4830]
    dec_hi = [-0.4829, 0.8365, -0.2241, -0.1294]

one grouped horizontal pass (2 filters/channel) + one grouped vertical pass
(2 filters per h-channel) with stride 2 and reflect extension (torch
``reflect`` mirrors without repeating the edge sample; it is NOT libvmaf's
2N-idx-1 mirror), yielding LL/LH/HL/HH per scale; LL streams to the next
scale, LH/HL/HH are scored with the same one-sided restoration + masking
and pooled (orientation means stacked into one epilogue call, weighted by
the ``csf`` hook).

VMAF-compat limits (read before comparing against libvmaf numbers): even
``wavelet="db2_like"`` is NOT bit-compatible with VMAF ADM: no gain/decouple
pass (``R = gamma * X``), no calibrated Watson97/Li-SW CSF tables (``csf`` is
a neutral 3-vector hook, not the CSF), critically sampled (not undecimated)
transform, ``reflect`` boundary instead of libvmaf's 2N-idx-1 mirror, no
EGL/NEG mode, all channels scored equally (libvmaf: luma only), scalar
masking floors instead of contrast-masking with the CSF denominator. Also NOT
implemented: two-tier border handling (libvmaf ``bf=0.1`` edge tier), p-norm
pooling (``p=3``) with the additive-noise floor (``0.03125``), and the
denominator guards (``den==0 -> 1.0``, ``numden_limit``, AIM-clip).
Expect correlation, not equality.

Conventions (via :mod:`frame_analytics.functional` helpers):
shapes ``(H,W)`` / ``(C,H,W)`` / ``(N,C,H,W)``; integer inputs use their
implicit range unless ``data_range`` is given; ``reduction="mean"`` -> scalar,
``"none"`` -> per-image ``(N,)``; compute in float32 (or ``dtype``),
reductions in float64; result in ``0..1`` (1 = identical).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from . import functional as _F
from .functional import set_compile_enabled

__all__ = [
    "adm_like",
    "set_compile_enabled",
    "ADM_LIKE_WEIGHTS",
    "ADM_LIKE_THRESHOLD_REL",
    "DB2_DEC_LO",
    "DB2_DEC_HI",
]

#: Default per-scale pooling weights (equal; normalised internally anyway).
ADM_LIKE_WEIGHTS = (0.25, 0.25, 0.25, 0.25)

#: Default masking threshold, as a fraction of the data range, per scale.
ADM_LIKE_THRESHOLD_REL = 0.02

#: Db2 analysis filters (PyWavelets ``db2`` dec_lo/dec_hi, L2-norm-1).
#: QMF pair: hi[k] = (-1)^k * lo[3-k].
DB2_DEC_LO = (-0.12940952255126037, 0.2241438680420134,
              0.8365163037378079, 0.48296291314453416)
DB2_DEC_HI = (-0.48296291314453416, 0.8365163037378079,
              -0.2241438680420134, -0.12940952255126037)

_db2_cache: dict = {}


@torch.inference_mode(False)
def _db2_weights(channels: int, wdt: torch.dtype, device):
    """Stacked separable Db2 taps, cached per (channels, dtype, device).

    Returns ``(wh, wv)``: ``wh`` is ``(2C,1,1,4)`` for the horizontal pass
    (``[lo; hi]`` per channel, ``groups=C``); ``wv`` is ``(4C,1,4,1)`` for
    the vertical pass (``[lo; hi]`` per h-channel, ``groups=2C``), so one
    call each yields LL/LH/HL/HH grouped per input channel.
    """
    key = (channels, str(wdt), str(device))
    hit = _db2_cache.get(key)
    if hit is None:
        lo = torch.tensor(DB2_DEC_LO, dtype=wdt, device=device)
        hi = torch.tensor(DB2_DEC_HI, dtype=wdt, device=device)
        pair = torch.stack([lo, hi])                      # (2, 4)
        wh = pair.repeat(channels, 1).view(2 * channels, 1, 1, 4).contiguous()
        wv = pair.repeat(2 * channels, 1).view(4 * channels, 1, 4, 1).contiguous()
        hit = (wh, wv)
        _db2_cache[key] = hit
    return hit


def _db2_dwt2(x: torch.Tensor, wh: torch.Tensor, wv: torch.Tensor):
    """One level of separably-applied Db2 DWT with reflect extension.

    Tap-matched, NOT VMAF-exact. torch ``reflect`` mirrors without repeating
    the edge sample; it is NOT libvmaf's 2N-idx-1 mirror, so boundary
    coefficients differ from libvmaf.

    ``(N,C,H,W)`` -> ``(LL, LH, HL, HH)``, each ``(N,C,ceil(H/2),ceil(W/2))``.
    """
    n, c, h, w = x.shape
    xp = F.pad(x, (1, 1 + (w & 1), 1, 1 + (h & 1)), mode="reflect")
    with _F._no_autocast(xp):
        hz = F.conv2d(xp, wh, stride=(1, 2), groups=c)    # (N,2C,Hp,W2)
        qd = F.conv2d(hz, wv, stride=(2, 1), groups=2 * c)  # (N,4C,H2,W2)
    quad = qd.view(n, c, 4, qd.shape[-2], qd.shape[-1])
    return quad[:, :, 0], quad[:, :, 1], quad[:, :, 2], quad[:, :, 3]


def _resolve_thresholds(thresholds, scales: int, L: float):
    if thresholds is None:
        return [ADM_LIKE_THRESHOLD_REL * L] * scales
    if isinstance(thresholds, (int, float)):
        return [float(thresholds)] * scales
    t = [float(v) for v in thresholds]
    if len(t) != scales:
        raise ValueError(f"expected {scales} thresholds, got {len(t)}")
    return t


_like_weight_cache: dict = {}
_csf_cache: dict = {}


@torch.inference_mode(False)
def _resolve_csf(csf, device) -> torch.Tensor:
    """Neutral Watson97 hook: 3 orientation gains ``(LH, HL, HH)``.

    Cached normalised device tensor. ``None`` -> equal. Weighting hook only --
    NOT calibrated CSF tables.
    """
    base = (1.0, 1.0, 1.0) if csf is None else tuple(float(v) for v in csf)
    if len(base) != 3:
        raise ValueError(f"csf needs 3 orientation gains (LH, HL, HH), got {len(base)}")
    key = (base, str(device))
    t = _csf_cache.get(key)
    if t is None:
        t = torch.tensor(base, dtype=torch.float64, device=device)
        t = t / t.sum().clamp_min(1e-12)
        _csf_cache[key] = t
    return t


@torch.inference_mode(False)
def _resolve_weights(weights, scales: int, device) -> torch.Tensor:
    if weights is None:
        base = (list(ADM_LIKE_WEIGHTS[:scales]) if scales <= len(ADM_LIKE_WEIGHTS)
                else [1.0] * scales)
    elif isinstance(weights, (int, float)):
        base = [float(weights)] * scales
    else:
        base = [float(v) for v in weights]
        if len(base) != scales:
            raise ValueError(f"expected {scales} weights, got {len(base)}")
    key = (tuple(base), str(device))
    t = _like_weight_cache.get(key)
    if t is None:
        t = torch.tensor(base, dtype=torch.float64, device=device)
        t = t / t.sum().clamp_min(1e-12)
        _like_weight_cache[key] = t
    return t


def _restore_pool(det_o: torch.Tensor, det_d: torch.Tensor,
                  T: float, eps: float) -> torch.Tensor:
    """Fused per-scale epilogue: restoration ratio + spatial mean, ``(N,)``.

    One-sided detail restoration ``min(|d|,|o|)/|o|`` (lost detail penalised,
    added detail clipped to 1), masked spots (``|o| <= T``) score exactly 1,
    so identical inputs give exactly 1. Widened to float64 only at the mean.
    NOTE (CUDA/Inductor): Triton codegen of the reduction is accurate to
    ~1e-9, so identical inputs score 1 - 3.5e-9 rather than exactly 1.0 on
    the compiled path; the eager path is bit-exact.
    """
    ao = det_o.abs()
    ad = det_d.abs()
    denom = ao.clamp_min(eps)
    ratio = torch.where(ao > T, torch.minimum(ad, ao) / denom,
                        torch.ones_like(denom))
    return ratio.reshape(ratio.shape[0], -1).mean(dim=1, dtype=torch.float64)


def adm_like(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    scales: int = 4,
    thresholds=None,
    weights: Optional[Sequence[float]] = None,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    wavelet: str = "haar",
    csf: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Box-pyramid detail-loss score in ``0..1`` (1 = identical). Higher is better.

    ADM-inspired, explicitly NOT VMAF-compatible (see module docstring).
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    if not isinstance(scales, int) or scales < 1:
        raise ValueError(f"scales must be a positive int, got {scales!r}")
    if wavelet not in ("haar", "db2_like"):
        raise ValueError(f"wavelet must be 'haar' or 'db2_like', got {wavelet!r}")
    x4, y4 = _F._check_pair(x, y)
    L = float(data_range) if data_range is not None else _F._infer_data_range(x4)
    if L <= 0:
        raise ValueError(f"data_range must be > 0, got {L!r}")
    wdt = _F._work_dtype(x4, dtype)

    if min(x4.shape[-2], x4.shape[-1]) < (1 << scales):
        raise ValueError(
            f"image {tuple(x4.shape[-2:])} too small for {scales} scales; "
            f"needs at least {1 << scales}px on a side"
        )

    Ts = _resolve_thresholds(thresholds, scales, L)
    w = _resolve_weights(weights, scales, x4.device)
    eps = 1e-12 * L
    epi = _F._maybe_compile(_restore_pool, "adm_like:restore_pool")

    n = x4.shape[0]
    acc = torch.zeros(n, dtype=torch.float64, device=x4.device)
    if wavelet == "db2_like":
        # The same filters process both images. Keeping them in one batch
        # halves the number of reflect-pad and convolution dispatches.
        cur = torch.cat([x4.to(wdt), y4.to(wdt)], dim=0)
        wh, wv = _db2_weights(cur.shape[1], wdt, cur.device)
        g = _resolve_csf(csf, cur.device)
        for s in range(scales):
            cur, LH, HL, HH = _db2_dwt2(cur, wh, wv)
            LHo, LHd = LH.split(n, dim=0)
            HLo, HLd = HL.split(n, dim=0)
            HHo, HHd = HH.split(n, dim=0)
            det_o = torch.cat([LHo, HLo, HHo], dim=0)
            det_d = torch.cat([LHd, HLd, HHd], dim=0)
            sub = (epi(det_o, det_d, Ts[s], eps).view(3, n) * g.view(3, 1)).sum(dim=0)
            acc += sub * w[s]
    else:
        cur_o = x4.to(wdt)
        cur_d = y4.to(wdt)
        for s in range(scales):
            low_o = F.avg_pool2d(cur_o, 2)
            low_d = F.avg_pool2d(cur_d, 2)
            size = cur_o.shape[-2:]
            det_o = cur_o - F.interpolate(low_o, size=size, mode="nearest")
            det_d = cur_d - F.interpolate(low_d, size=size, mode="nearest")
            cur_o, cur_d = low_o, low_d
            # NOTE: w[s] stays a 0-dim device tensor -- float(w[s]) would force
            # a device-to-host sync per scale on CUDA.
            acc += epi(det_o, det_d, Ts[s], eps) * w[s]
    per_image = acc.clamp_(0.0, 1.0)
    return per_image.mean() if reduction == "mean" else per_image
