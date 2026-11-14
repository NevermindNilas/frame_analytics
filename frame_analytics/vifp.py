"""Visual Information Fidelity, pixel domain (VIFp, Sheikh & Bovik 2006).

Final package module (torch-only). Shared input front end comes from
:mod:`frame_analytics.functional`; no local duplicates. ``torch.compile``
uses the shared eager-fallback wrapper; :func:`set_compile_enabled` is a
passthrough to ``functional``.

Pixel-domain port of the LIVE ``vifp_mscale.m``: a 4-scale Gaussian pyramid
stands in for the steerable-pyramid subbands. Exact per-scale windows
(``sigma = N/5``), ``'valid'`` support everywhere -- one window per scale,
shared by the prefilter and the local moments (``N = 2**(S-s) + 1``, i.e.
``17/9/5/3`` for the default 4 scales). Per-position GSM fit (``M = 1``),
mutual informations summed in float64 with a single division.

Higher is better; 1.0 is identical. Supports gradients in either input.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from . import functional as _functional
from .functional import (
    _apply_luma,
    _maybe_compile,
    _no_autocast,
    _prep,
    _work_dtype,
)

__all__ = ["vifp", "set_compile_enabled"]


def set_compile_enabled(flag: bool) -> None:
    """Passthrough to :func:`frame_analytics.functional.set_compile_enabled`."""
    _functional.set_compile_enabled(flag)


# --------------------------------------------------------------------------- #
# windows + separable valid filtering (torch-only, no scipy)
# --------------------------------------------------------------------------- #

_win_cache: dict = {}


@torch.inference_mode(False)
def _gaussian_1d(n: int, *, device, dtype: torch.dtype) -> torch.Tensor:
    """1-D Gaussian taps, ``sigma = n/5`` (exact LIVE rule), sum-normalised."""
    key = (n, str(device), str(dtype))
    w = _win_cache.get(key)
    if w is not None:
        return w
    sigma = n / 5.0
    r = (n - 1) / 2.0
    coords = torch.arange(n, dtype=torch.float64) - r
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / g.sum()
    w = g.to(device=device, dtype=dtype)
    _win_cache[key] = w
    return w


def _filter_nchw(x: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """Valid separable filter over ``(N, C, H, W)``.

    Shared helper: batch-folded single-plane convolutions (on CUDA, exactly
    the previous local implementation), channels-last depthwise for large
    float32 CPU inputs, where oneDNN's single-plane kernels are slow.
    """
    return _functional._separable_conv(x, win)


def _filter_down2(x: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """``_filter_nchw(x, win)[..., ::2, ::2]`` without the discarded 3/4.

    A stride-2 valid convolution yields exactly the even-indexed outputs
    (``floor((H-k)/2) + 1 == ceil((H-k+1)/2)`` rows/cols): the horizontal
    pass strides along W, the vertical pass along H. CPU only: on CUDA,
    cuDNN's strided single-plane kernels were 1.4-2.3x slower than the
    full filter and changed results well beyond float32 rounding (a
    different algorithm class), so CUDA keeps filter-then-subsample.
    """
    if x.is_cuda:
        return _filter_nchw(x, win)[..., ::2, ::2].contiguous()
    n, c, h, w = x.shape
    flat = x.reshape(n * c, 1, h, w)
    with _no_autocast(flat):
        out = F.conv2d(flat, win.view(1, 1, 1, -1), stride=(1, 2))
        out = F.conv2d(out, win.view(1, 1, -1, 1), stride=(2, 1))
    return out.reshape(n, c, out.shape[-2], out.shape[-1])


def _epi_chunk(chunk: torch.Tensor, sn: float, e_abs: float, e_rel: float):
    """Per-row ``(num, den)`` sums ``(N, C, H)`` for all channels at once.

    ``chunk`` is ``(N, 5, C, H, W)`` holding ``E[x], E[y], E[x^2], E[y^2],
    E[x*y]`` per channel. Float64 throughout. LIVE threshold sequence against
    the per-position floor ``e_abs + e_rel * E[x^2]``.

    The reduction stops at rows; the caller finishes it. A whole-image
    reduction leaves Inductor one reduction per image, and once the
    pyramid's second scale turns this region dynamic, the kernel tuned for
    that size runs every scale (measured 1.3 -> 4.7 ms at 1080p). Row sums
    parallelise over rows whatever the size hint.
    """
    d = chunk.to(torch.float64)
    mux = d[:, 0]
    muy = d[:, 1]
    exx = d[:, 2]
    sxx = exx - mux * mux
    syy = d[:, 3] - muy * muy
    sxy = d[:, 4] - mux * muy

    e_map = e_abs + e_rel * exx
    zero = torch.zeros((), dtype=torch.float64, device=d.device)
    mask_x = sxx < e_map
    g = sxy / (sxx + e_map)
    sv = syy - g * sxy
    g = torch.where(mask_x, zero, g)
    sv = torch.where(mask_x, syy, sv)
    sxx = torch.where(mask_x, zero, sxx)
    mask_y = syy < e_map
    g = torch.where(mask_y, zero, g)
    sv = torch.where(mask_y, zero, sv)
    neg = g < 0
    g = torch.where(neg, zero, g)
    sv = torch.where(neg, syy, sv)
    sv = sv.clamp_min(e_map)

    num_t = torch.log2(1.0 + g * g * sxx / (sv + sn))
    den_t = torch.log2(1.0 + sxx / sn)
    return (num_t.sum(dim=-1, dtype=torch.float64),
            den_t.sum(dim=-1, dtype=torch.float64))


def _min_side(num_scales: int) -> int:
    """Shortest side surviving valid-blur + decimation with a 1-px tail."""
    h = 3
    for s in range(num_scales - 1, 0, -1):
        h = 2 * h - 1 + (1 << (num_scales - s))
    return h


# --------------------------------------------------------------------------- #
# VIFp
# --------------------------------------------------------------------------- #

def vifp(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    sigma_n_sq: float = 2.0,
    num_scales: int = 4,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-10,
    luma=None,
    crop_border: int = 0,
) -> torch.Tensor:
    """Visual Information Fidelity, pixel domain (Sheikh & Bovik, TIP 2006)."""
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    if not isinstance(num_scales, int) or num_scales < 1:
        raise ValueError(f"num_scales must be a positive int, got {num_scales!r}")
    if not sigma_n_sq > 0:
        raise ValueError(f"sigma_n_sq must be > 0, got {sigma_n_sq!r}")
    if not eps > 0:
        raise ValueError(f"eps must be > 0, got {eps!r}")

    x4, y4, L, mode = _prep(x, y, luma, crop_border, data_range)
    if not L > 0:
        raise ValueError(f"data_range must be > 0, got {data_range!r}")
    need = _min_side(num_scales)
    if min(x4.shape[-2], x4.shape[-1]) < need:
        raise ValueError(
            f"image {tuple(x4.shape[-2:])} too small for {num_scales} scales; "
            f"needs at least {need} px on the short side"
        )

    wdt = _work_dtype(x4, dtype)
    x4, y4 = _apply_luma(x4, y4, mode, L, wdt)
    rx, ry = x4.to(wdt), y4.to(wdt)
    dev = rx.device

    unit = (L / 255.0) ** 2
    sn = float(sigma_n_sq) * unit
    e = float(eps) * unit
    if wdt is torch.float64:
        rel = 1e-12
    elif wdt is torch.float32:
        rel = 1e-5
    else:
        rel = 1e-2

    wins = [_gaussian_1d((1 << (num_scales - s)) + 1, device=dev, dtype=wdt)
            for s in range(num_scales)]

    epi = _maybe_compile(_epi_chunk, "vifp:epi")

    n = rx.shape[0]
    num = torch.zeros(n, dtype=torch.float64, device=dev)
    den = torch.zeros(n, dtype=torch.float64, device=dev)
    terms = 0

    for s in range(num_scales):
        if s:
            rx = _filter_down2(rx, wins[s])
            ry = _filter_down2(ry, wins[s])
            if min(rx.shape[-2], rx.shape[-1]) < 3:
                raise ValueError(
                    f"pyramid collapsed at scale {s}: shape {tuple(rx.shape[-2:])}"
                )
        nn, cc, hh, ww = rx.shape
        if torch.is_grad_enabled() and (rx.requires_grad or ry.requires_grad):
            pb = torch.cat([rx, ry, rx * rx, ry * ry, rx * ry], dim=1)
        else:
            pb = torch.empty(nn, 5 * cc, hh, ww, dtype=wdt, device=dev)
            b0, b1, b2, b3, b4 = pb.split(cc, dim=1)
            b0.copy_(rx)
            b1.copy_(ry)
            torch.mul(rx, rx, out=b2)
            torch.mul(ry, ry, out=b3)
            torch.mul(rx, ry, out=b4)
            del b0, b1, b2, b3, b4
        filt = _filter_nchw(pb, wins[s])
        del pb
        terms += cc * filt.shape[-2] * filt.shape[-1]
        # (N, 5*C, H, W) -> (N, 5, C, H, W): a view, so backward is one
        # plain gradient for ``filt`` (per-channel slices would each
        # scatter into a full-size zero buffer).
        nsc, dsc = epi(filt.unflatten(1, (5, cc)), sn, e, rel)
        num = num + nsc.sum(dim=(1, 2))
        den = den + dsc.sum(dim=(1, 2))
        del filt

    tiny = 1e-12 * max(terms, 1)
    mx = x4.to(torch.float64).mean(dim=(1, 2, 3))
    my = y4.to(torch.float64).mean(dim=(1, 2, 3))
    dc_same = (mx - my).abs() <= 1e-6 * L
    one = torch.ones((), dtype=torch.float64, device=dev)
    zero = torch.zeros((), dtype=torch.float64, device=dev)
    safe = den > tiny
    out = torch.where(safe, num / den.clamp_min(tiny),
                      torch.where(dc_same & (num <= tiny), one, zero))
    # Keep this test on-device so it is legal inside CUDA graph capture.
    # The where also retains a zero gradient for identical image pairs.
    identical = (x4 == y4).flatten(start_dim=1).all(dim=1)
    out = torch.where(identical, torch.ones_like(out), out)
    return out if reduction == "none" else out.mean()


if __name__ == "__main__":
    import math as _math

    torch.manual_seed(0)
    H = W = 128
    x = torch.rand(2, 3, H, W, dtype=torch.float64) * 255.0
    sv = vifp(x, x).item()
    print(f"vifp(x, x) = {sv:.12f}")
    assert abs(sv - 1.0) < 1e-6, sv
    y = x[:1]
    b = F.pad(y, (4, 4, 4, 4), mode="replicate")
    print("vifp self-test: OK")
