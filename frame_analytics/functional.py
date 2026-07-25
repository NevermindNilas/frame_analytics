"""Fast, portable PyTorch implementations of MSE / PSNR / SSIM.

Runs on CPU and CUDA, on uint8 or floating inputs, and stays numerically
inside the tolerance of :mod:`frame_analytics.reference` (float64, literal
transcription of Wang et al. 2004).

Speed notes
-----------
SSIM needs five local Gaussian-weighted expectations: E[x], E[y], E[x^2],
E[y^2], E[xy].  The naive route is five separate 11x11 convolutions, which is
what most libraries ship.  Here:

1. The 11x11 Gaussian is separable, so it becomes an 11-tap horizontal pass
   plus an 11-tap vertical pass: 22 MACs/px instead of 121.
2. All five planes are packed into the *batch* dimension and filtered by a
   single 1-input-channel convolution.  Two kernel launches total, not ten,
   and no grouped-convolution overhead.
3. The plane packing and the SSIM epilogue are ``torch.compile``d, so the
   products, the variance combination, the divide and the mean reduction fuse
   into a couple of bandwidth-bound kernels instead of ~15 separate ones.
4. Inputs are mean-shifted before filtering.  ``E[x^2] - E[x]^2`` is a
   catastrophic-cancellation trap in float32; centring the data first buys
   roughly a decimal digit back for free, because the shift cancels out of the
   variance terms exactly and is added back into the mean terms.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "mse",
    "psnr",
    "ssim",
    "gaussian_window_1d",
    "set_compile_enabled",
]

_COMPILE_ENABLED = True
_compiled_cache: dict = {}


def set_compile_enabled(flag: bool) -> None:
    """Turn the ``torch.compile`` fast path on/off (useful for A/B benchmarks)."""
    global _COMPILE_ENABLED
    _COMPILE_ENABLED = bool(flag)
    _compiled_cache.clear()


class _LazyCompiled:
    """``torch.compile`` with a permanent eager fallback.

    Inductor needs a working host compiler for its CPU backend and a working
    Triton for its CUDA backend. Both are commonly missing (no MSVC on PATH, a
    Python version Triton has not caught up with). Rather than make the whole
    library unusable in that case, the first failure demotes this callable to
    plain eager for the rest of the process.
    """

    __slots__ = ("_fn", "_compiled", "_eager_only", "_key")

    def __init__(self, fn, key: str):
        self._fn = fn
        self._key = key
        self._compiled = None
        self._eager_only = False

    def __call__(self, *args, **kwargs):
        if self._eager_only or not _COMPILE_ENABLED:
            return self._fn(*args, **kwargs)
        if self._compiled is None:
            try:
                from . import backend

                backend.ensure_host_compiler()
                # dynamic=None: specialise on the first shape (fastest for the
                # fixed-resolution streams these metrics are used on), then let
                # Dynamo generalise if a second shape shows up.
                self._compiled = torch.compile(self._fn, dynamic=None)
            except Exception:
                self._eager_only = True
                return self._fn(*args, **kwargs)
        try:
            return self._compiled(*args, **kwargs)
        except Exception as exc:
            warnings.warn(
                f"torch.compile failed for {self._key} ({type(exc).__name__}); "
                "falling back to eager for the rest of this process",
                RuntimeWarning, stacklevel=2,
            )
            self._eager_only = True
            return self._fn(*args, **kwargs)


def _maybe_compile(fn, key: str):
    cached = _compiled_cache.get(key)
    if cached is None:
        cached = _LazyCompiled(fn, key)
        _compiled_cache[key] = cached
    return cached


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #

def _infer_data_range(t: torch.Tensor) -> float:
    if t.dtype == torch.uint8:
        return 255.0
    if t.dtype in (torch.int16, torch.int32):
        return 65535.0
    if getattr(torch, "uint16", None) is not None and t.dtype == torch.uint16:
        return 65535.0
    return 1.0


def _as_nchw(t: torch.Tensor) -> torch.Tensor:
    """Accept (H,W), (C,H,W) or (N,C,H,W); always return 4-D."""
    if t.ndim == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.ndim == 3:
        return t.unsqueeze(0)
    if t.ndim == 4:
        return t
    raise ValueError(f"expected 2/3/4-D tensor, got shape {tuple(t.shape)}")


def _check_pair(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.device != y.device:
        raise ValueError(f"device mismatch: {x.device} vs {y.device}")
    return _as_nchw(x), _as_nchw(y)


def _work_dtype(x: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is not None:
        return dtype
    if x.dtype in (torch.float64,):
        return torch.float64
    return torch.float32


# --------------------------------------------------------------------------- #
# MSE / PSNR
# --------------------------------------------------------------------------- #


def _mse_kernel_mean(x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype):
    d = x.to(dtype) - y.to(dtype)
    return (d * d).sum(dtype=torch.float64) / d.numel()


def _mse_kernel_per_image(x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype):
    d = x.to(dtype) - y.to(dtype)
    return (d * d).sum(dim=(1, 2, 3), dtype=torch.float64) / d[0].numel()


def mse(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    out_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Mean squared error.

    Parameters
    ----------
    x, y
        Same-shape tensors, ``(H,W)`` / ``(C,H,W)`` / ``(N,C,H,W)``, any dtype.
        Integer inputs are promoted inside the fused kernel, so a uint8 pair
        costs 2 bytes/pixel of bandwidth rather than 8.
    reduction
        ``"mean"`` -> scalar over the whole batch. ``"none"`` -> one value per
        batch element, shape ``(N,)``.
    dtype
        Elementwise compute dtype. Defaults to float32.
    out_dtype
        Accumulator/result dtype, float64 by default. A 4K frame has 8.3M
        residuals; summing those in float32 loses ~4 significant digits, which
        shows up directly in PSNR. The accumulator is a scalar, so the wider
        type is free.
    """
    x4, y4 = _check_pair(x, y)
    wdt = _work_dtype(x4, dtype)
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    per_image = reduction == "none"

    from . import backend  # local import: avoids a cycle at module import time

    fast = backend.try_mse(x4, y4, per_image=per_image, dtype=out_dtype)
    if fast is not None:
        return fast

    fn = _maybe_compile(
        _mse_kernel_per_image if per_image else _mse_kernel_mean,
        f"mse:{per_image}",
    )
    return fn(x4, y4, wdt).to(out_dtype)


def psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    out_dtype: torch.dtype = torch.float64,
    eps: float = 0.0,
) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB.

    ``data_range`` defaults to 255 for integer inputs and 1.0 for float ones.
    Identical inputs give ``+inf`` unless ``eps`` is set.
    """
    x4, y4 = _check_pair(x, y)
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    L = float(data_range) if data_range is not None else _infer_data_range(x4)
    bias = math.log10(L * L) * 10.0

    from . import backend

    if not eps:
        # native path folds the dB conversion into the reduction kernel
        fast = backend.try_mse(x4, y4, per_image=reduction == "none",
                               dtype=out_dtype, psnr_bias=bias)
        if fast is not None:
            return fast

    m = mse(x4, y4, reduction=reduction, dtype=dtype, out_dtype=out_dtype)
    if eps:
        m = m.clamp_min(eps)
    # 10*log10(L^2 / m) written as a constant minus a tensor term: building an
    # L^2 tensor on the device would be a host-to-device copy, which is illegal
    # inside a CUDA graph capture. m == 0 still yields +inf.
    return bias - 10.0 * torch.log10(m)


# --------------------------------------------------------------------------- #
# SSIM
# --------------------------------------------------------------------------- #

_window_cache: dict = {}


def gaussian_window_1d(
    win_size: int = 11,
    sigma: float = 1.5,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """1-D half of ``fspecial('gaussian', win_size, sigma)``.

    The 2-D MATLAB window is the outer product of this vector with itself:
    normalising in 1-D and then taking the outer product is *exactly* the same
    as normalising the 2-D window, so separability costs no accuracy.
    """
    key = (win_size, sigma, str(device), dtype)
    w = _window_cache.get(key)
    if w is not None:
        return w
    r = (win_size - 1) / 2.0
    coords = torch.arange(win_size, dtype=torch.float64) - r
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / g.sum()
    w = g.to(device=device, dtype=dtype)
    _window_cache[key] = w
    return w


def _pack_planes(x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype, shift: float):
    """Build ``[x', y', x'^2, y'^2, x'y']`` where ``x' = x - shift``.

    The shift cancels exactly out of every variance/covariance term and is
    added back into the two mean terms, so the result is algebraically
    identical -- but ``E[x'^2] - E[x']^2`` on centred data loses far less to
    cancellation than the same expression on raw 0..255 data.
    """
    xs = x.to(dtype) - shift
    ys = y.to(dtype) - shift
    return torch.cat([xs, ys, xs * xs, ys * ys, xs * ys], dim=1)


def _ssim_map_from_moments(t: torch.Tensor, C: int, shift: float, C1: float, C2: float):
    ux = t[:, 0 * C : 1 * C]
    uy = t[:, 1 * C : 2 * C]
    mx = ux + shift
    my = uy + shift
    sxx = t[:, 2 * C : 3 * C] - ux * ux
    syy = t[:, 3 * C : 4 * C] - uy * uy
    sxy = t[:, 4 * C : 5 * C] - ux * uy

    num = (2.0 * mx * my + C1) * (2.0 * sxy + C2)
    den = (mx * mx + my * my + C1) * (sxx + syy + C2)
    return num / den


# Three separate entry points rather than one with boolean flags: Dynamo guards
# on the flags and recompiles per combination, which blows past the recompile
# limit in mixed workloads.
def _ssim_epilogue_map(t, C, shift, C1, C2):
    return _ssim_map_from_moments(t, C, shift, C1, C2)


def _ssim_epilogue_mean(t, C, shift, C1, C2):
    return _ssim_map_from_moments(t, C, shift, C1, C2).mean(dtype=torch.float64)


def _ssim_epilogue_per_image(t, C, shift, C1, C2):
    return _ssim_map_from_moments(t, C, shift, C1, C2).mean(
        dim=(1, 2, 3), dtype=torch.float64)


def _sep_filter(packed: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """Separable 'valid' Gaussian filter over every plane of ``packed``.

    The channel axis is folded into the batch axis so both passes are plain
    single-input-channel convolutions -- much better mapped than ``groups=5C``
    depthwise convolutions on both cuDNN and oneDNN.
    """
    n, c, h, w = packed.shape
    k = win.numel()
    flat = packed.reshape(n * c, 1, h, w)
    wh = win.view(1, 1, 1, k)
    wv = win.view(1, 1, k, 1)
    out = F.conv2d(flat, wh)
    out = F.conv2d(out, wv)
    return out.reshape(n, c, h - k + 1, w - k + 1)


def _matlab_downsample(x: torch.Tensor, f: int) -> torch.Tensor:
    """The automatic downsample MATLAB's ``ssim.m`` applies to large images."""
    if f <= 1:
        return x
    n, c, h, w = x.shape
    ker = x.new_full((1, 1, f, f), 1.0 / (f * f))
    pad = f // 2
    xp = F.pad(x.reshape(n * c, 1, h, w), (pad, pad, pad, pad), mode="replicate")
    lp = F.conv2d(xp, ker)
    lp = lp.reshape(n, c, lp.shape[-2], lp.shape[-1])
    return lp[:, :, ::f, ::f].contiguous()


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    win_size: int = 11,
    sigma: float = 1.5,
    K: Sequence[float] = (0.01, 0.03),
    reduction: str = "mean",
    return_map: bool = False,
    dtype: Optional[torch.dtype] = None,
    downsample: bool = False,
    backend_hint: str = "auto",
) -> torch.Tensor:
    """Structural similarity index (Wang et al., 2004).

    Defaults reproduce ``ssim_index.m``: 11x11 Gaussian window, sigma 1.5,
    K = (0.01, 0.03), 'valid' support, mean of the SSIM map.

    Parameters
    ----------
    data_range
        Dynamic range ``L``. Defaults to 255 for integer input, 1.0 for float.
    reduction
        ``"mean"`` -> scalar. ``"none"`` -> per-image ``(N,)``.
    return_map
        Return the full ``(N, C, H-10, W-10)`` SSIM map instead of a reduction.
    downsample
        Apply MATLAB ``ssim.m``'s automatic ``f = round(min(H,W)/256)``
        box-downsample first. Off by default -- ``ssim_index.m``, scikit-image
        and every PyTorch library omit it.
    backend_hint
        ``"auto"`` (native kernel if available), ``"torch"`` (portable path),
        ``"native"`` (require the compiled extension, raise if missing).
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    x4, y4 = _check_pair(x, y)
    L = float(data_range) if data_range is not None else _infer_data_range(x4)
    wdt = _work_dtype(x4, dtype)
    K1, K2 = float(K[0]), float(K[1])
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    if downsample:
        f = max(1, int(round(min(x4.shape[-2], x4.shape[-1]) / 256.0)))
        if f > 1:
            x4 = _matlab_downsample(x4.to(wdt), f)
            y4 = _matlab_downsample(y4.to(wdt), f)

    if x4.shape[-2] < win_size or x4.shape[-1] < win_size:
        raise ValueError(
            f"image {tuple(x4.shape[-2:])} smaller than window {win_size}x{win_size}"
        )

    # Centring constant. Free (no reduction needed) and it buys back most of
    # the float32 precision that E[x^2] - E[x]^2 would otherwise throw away.
    shift = 0.5 * L

    from . import backend

    if backend_hint != "torch":
        fast = backend.try_ssim(
            x4, y4, win_size=win_size, sigma=sigma, shift=shift, C1=C1, C2=C2,
            reduction=reduction, return_map=return_map, dtype=wdt,
        )
        if fast is not None:
            return fast
        if backend_hint == "native":
            raise RuntimeError("native SSIM backend unavailable for these inputs")

    win = gaussian_window_1d(win_size, sigma, device=x4.device, dtype=wdt)
    pack = _maybe_compile(_pack_planes, "pack")

    if return_map:
        epi_fn, key = _ssim_epilogue_map, "epi:map"
    elif reduction == "none":
        epi_fn, key = _ssim_epilogue_per_image, "epi:per_image"
    else:
        epi_fn, key = _ssim_epilogue_mean, "epi:mean"
    epi = _maybe_compile(epi_fn, key)

    packed = pack(x4, y4, wdt, shift)
    filt = _sep_filter(packed, win)
    return epi(filt, x4.shape[1], shift, C1, C2)
