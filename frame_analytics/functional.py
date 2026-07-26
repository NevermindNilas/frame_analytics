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
    "ms_ssim",
    "gmsd",
    "gms",
    "l1",
    "charbonnier",
    "huber",
    "rgb_to_luma",
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
# luma + border crop
#
# Both are conventions, not algorithms, but they are the two that decide
# whether a number is comparable with a published one.  Nearly every
# super-resolution / restoration paper reports PSNR and SSIM on the *luma*
# plane of a border-cropped frame, so a library that cannot do that produces
# right-looking numbers that quietly disagree with the literature.
#
# Both are pure views/affine maps, so they are applied once, up front, and the
# native kernels take them as a stride + a set of channel weights rather than
# as materialised tensors.
# --------------------------------------------------------------------------- #

# (r, g, b, offset-as-a-fraction-of-the-data-range); see reference.LUMA_COEFFS.
LUMA_COEFFS = {
    "bt601": (0.299, 0.587, 0.114, 0.0),
    "bt709": (0.2126, 0.7152, 0.0722, 0.0),
    "matlab": (65.481 / 255.0, 128.553 / 255.0, 24.966 / 255.0, 16.0 / 255.0),
}


def _resolve_luma(luma) -> Optional[str]:
    if luma is None or luma is False:
        return None
    if luma is True:
        return "bt601"
    if isinstance(luma, str):
        if luma not in LUMA_COEFFS:
            raise ValueError(f"unknown luma mode {luma!r}; "
                             f"expected one of {sorted(LUMA_COEFFS)}")
        return luma
    raise TypeError(f"luma must be None, a bool or a string, got {type(luma).__name__}")


def rgb_to_luma(t: torch.Tensor, mode: str = "bt601", *,
                data_range: Optional[float] = None,
                dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """``(N,3,H,W)`` RGB -> ``(N,1,H,W)`` luma, in the input's own units.

    Written as three scalar-weighted slices rather than a weight tensor and a
    ``sum``: the scalars are baked into the kernel by ``torch.compile`` (and by
    the native path), whereas a weight tensor would have to be created on the
    device on every call -- an H2D copy, which also makes the call illegal
    inside a CUDA graph capture.
    """
    t4 = _as_nchw(t)
    if t4.shape[1] != 3:
        raise ValueError(f"luma needs a 3-channel image, got {t4.shape[1]} channels")
    m = _resolve_luma(mode)
    if m is None:
        raise ValueError("rgb_to_luma needs a luma convention, not None/False")
    cr, cg, cb, off = LUMA_COEFFS[m]
    L = float(data_range) if data_range is not None else _infer_data_range(t4)
    wdt = _work_dtype(t4, dtype)
    v = t4.to(wdt)
    out = cr * v[:, 0:1] + cg * v[:, 1:2] + cb * v[:, 2:3]
    if off:
        out = out + off * L
    return out


def _crop_border(t: torch.Tensor, b: int) -> torch.Tensor:
    if not b:
        return t
    if b < 0:
        raise ValueError(f"crop_border must be >= 0, got {b}")
    h, w = t.shape[-2], t.shape[-1]
    if h <= 2 * b or w <= 2 * b:
        raise ValueError(f"crop_border={b} removes the whole {h}x{w} image")
    return t[..., b:h - b, b:w - b]


def _prep(x: torch.Tensor, y: torch.Tensor, luma, crop_border: int,
          data_range: Optional[float]):
    """Shared front end: shape check, border crop, luma-mode resolution.

    Returns ``(x4, y4, L, luma_mode)``.  The crop is a view; the luma
    projection is deferred, because the native kernels fold it into their load
    stage and only the portable path has to materialise it.
    """
    x4, y4 = _check_pair(x, y)
    L = float(data_range) if data_range is not None else _infer_data_range(x4)
    mode = _resolve_luma(luma)
    x4 = _crop_border(x4, crop_border)
    y4 = _crop_border(y4, crop_border)
    return x4, y4, L, mode


def _apply_luma(x4, y4, mode, L, dtype):
    if mode is None:
        return x4, y4
    return (rgb_to_luma(x4, mode, data_range=L, dtype=dtype),
            rgb_to_luma(y4, mode, data_range=L, dtype=dtype))


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
    luma=None,
    crop_border: int = 0,
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
    luma
        ``"bt601"`` / ``"bt709"`` / ``"matlab"`` (or ``True`` for ``"bt601"``)
        to score the luma plane of a 3-channel input instead of RGB. See
        :func:`rgb_to_luma`.
    crop_border
        Drop this many pixels from every edge first, as the super-resolution
        literature does (usually by the upscale factor).
    """
    x4, y4, L, mode = _prep(x, y, luma, crop_border, None)
    wdt = _work_dtype(x4, dtype)
    x4, y4 = _apply_luma(x4, y4, mode, L, wdt)
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
    luma=None,
    crop_border: int = 0,
) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB.

    ``data_range`` defaults to 255 for integer inputs and 1.0 for float ones.
    Identical inputs give ``+inf`` unless ``eps`` is set.

    ``luma`` and ``crop_border`` are the super-resolution reporting
    conventions; ``psnr(..., luma="matlab", crop_border=scale)`` is the Y-PSNR
    those papers quote.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    x4, y4, L, mode = _prep(x, y, luma, crop_border, data_range)
    x4, y4 = _apply_luma(x4, y4, mode, L, _work_dtype(x4, dtype))
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


def _no_autocast(t: torch.Tensor):
    """Run a convolution at the tensor's own dtype, whatever autocast says.

    ``F.conv2d`` is autocast-eligible, so inside a ``torch.autocast`` region --
    which is where every modern training loop lives -- these filters would
    silently run in fp16/bf16 and shift the metric. A *measurement* changing
    because of the surrounding compute context is a bug, and a quiet one: the
    result still comes back as float64.
    """
    return torch.autocast(device_type=t.device.type, enabled=False)


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
    with _no_autocast(flat):
        out = F.conv2d(flat, wh)
        out = F.conv2d(out, wv)
    return out.reshape(n, c, h - k + 1, w - k + 1)


def _reject_double_backward() -> None:
    """Refuse a ``create_graph=True`` backward through a native kernel.

    The fused backward is a CUDA kernel, so it is not itself an autograd graph
    and cannot be differentiated a second time. Left alone, that is not an
    error but something worse: the surrounding elementwise chain keeps the
    graph connected, so a gradient penalty or an HVP through MS-SSIM comes back
    populated -- and identically zero. Better to stop here and name the way out.

    Autograd runs backward functions with grad mode set to ``create_graph``, so
    this test is exactly "was a second derivative asked for".
    """
    if torch.is_grad_enabled():
        raise RuntimeError(
            "frame_analytics' fused SSIM backward is not twice differentiable, "
            "so create_graph=True (gradient penalties, Hessian-vector products, "
            "torch.func.hessian, MAML-style inner loops) cannot use it. For a "
            "second derivative use the portable path in eager mode:\n"
            "    frame_analytics.set_compile_enabled(False)\n"
            "    ...(x, y, backend_hint='torch')\n"
            "Both are needed -- torch.compile's AOT autograd does not support "
            "double backward either."
        )


class _FusedSSIM(torch.autograd.Function):
    """Differentiable native SSIM.

    Forward is the same fused kernel the inference path uses. Backward runs a
    second fused pair: one tile pass recomputes the local moments and emits the
    three coefficient maps, one scatter pass convolves them back through the
    Gaussian and combines with x and y. Nothing full-resolution crosses between
    forward and backward -- only x and y are saved, and the moments are cheaper
    to recompute than to store and reload.
    """

    @staticmethod
    def forward(ctx, x, y, win, shift, C1, C2, want_map):
        from . import backend

        res = backend.ssim_fused(x, y, win, shift, C1, C2, want_map)
        ctx.save_for_backward(x, y, win)
        ctx.consts = (shift, C1, C2)
        ctx.want_map = want_map
        ctx.npix = (x.shape[-2] - win.numel() + 1) * (x.shape[-1] - win.numel() + 1)
        ctx.nplanes = x.shape[0] * x.shape[1]
        if want_map:
            return res[2]
        return res[1]                      # batch-mean scalar

    @staticmethod
    def backward(ctx, grad_out):
        from . import backend

        _reject_double_backward()
        x, y, win = ctx.saved_tensors
        shift, C1, C2 = ctx.consts
        need_dx = ctx.needs_input_grad[0]
        need_dy = ctx.needs_input_grad[1]
        if not (need_dx or need_dy):
            return (None,) * 7

        empty = torch.empty(0, device=x.device, dtype=torch.float32)
        if ctx.want_map:
            dmap, gscalar = grad_out.contiguous(), empty
        else:
            # Scalar mean: every map position gets the same upstream gradient.
            # It is passed as a 1-element device tensor rather than a python
            # float -- reading it on the host would sync the stream once per
            # training step, and materialising a constant HxW map to read back
            # would be pure bandwidth.
            dmap, gscalar = empty, grad_out.reshape(1)

        dx, dy = backend.ssim_backward(x, y, dmap, gscalar, win, shift, C1, C2,
                                       need_dx, need_dy)
        return (dx if need_dx else None, dy if need_dy else None,
                None, None, None, None, None)


def _matlab_downsample(x: torch.Tensor, f: int) -> torch.Tensor:
    """The automatic downsample MATLAB's ``ssim.m`` applies to large images."""
    if f <= 1:
        return x
    n, c, h, w = x.shape
    ker = x.new_full((1, 1, f, f), 1.0 / (f * f))
    pad = f // 2
    xp = F.pad(x.reshape(n * c, 1, h, w), (pad, pad, pad, pad), mode="replicate")
    with _no_autocast(xp):
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
    luma=None,
    crop_border: int = 0,
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
    luma, crop_border
        Reporting conventions; see :func:`mse`.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    x4, y4, L, mode = _prep(x, y, luma, crop_border, data_range)
    wdt = _work_dtype(x4, dtype)
    x4, y4 = _apply_luma(x4, y4, mode, L, wdt)
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

    # Differentiable native path: only worth taking when a gradient is actually
    # wanted, since it saves x and y and builds an autograd node.
    if (backend_hint != "torch"
            and (x4.requires_grad or y4.requires_grad)
            and torch.is_grad_enabled()
            and reduction == "mean"
            and not downsample
            and backend.ssim_autograd_available(x4, y4, win_size)):
        win = gaussian_window_1d(win_size, sigma, device=x4.device,
                                 dtype=torch.float32)
        return _FusedSSIM.apply(x4, y4, win, shift, C1, C2, return_map)

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


# --------------------------------------------------------------------------- #
# MS-SSIM
# --------------------------------------------------------------------------- #

# Wang et al. 2003, table 1. beta_i == gamma_i, so one weight per scale.
MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

_ms_weight_cache: dict = {}


def _ms_weights(w, device) -> torch.Tensor:
    """The scale weights as a cached ``(M, 1, 1)`` device tensor.

    Cached rather than rebuilt because building it is a host-to-device copy,
    which is both a per-call sync point and illegal inside a CUDA graph
    capture -- the same reason :func:`gaussian_window_1d` caches.
    """
    key = (w, str(device))
    t = _ms_weight_cache.get(key)
    if t is None:
        t = torch.tensor(w, dtype=torch.float64, device=device).view(-1, 1, 1)
        _ms_weight_cache[key] = t
    return t


def _cast_pool2(x, dtype):
    """``avg_pool2d`` over a dtype conversion, fused into a single kernel.

    Written as one compiled function rather than ``x.to(dtype)`` followed by a
    pool because the intermediate is a *full-resolution* float copy of the
    input: at 1080p RGB that is a 25 MB tensor written and immediately read
    back, for a result four times smaller.
    """
    return F.avg_pool2d(x.to(dtype), 2)


def _ssim_cs_epilogue(t: torch.Tensor, C: int, shift: float, C1: float, C2: float):
    """Per-plane means of the SSIM map *and* of its contrast-structure factor.

    MS-SSIM needs the ``cs`` term at every scale and the full SSIM only at the
    coarsest one, and both come out of the same five moments -- computing them
    in one epilogue means one pass over the filtered planes instead of two.
    """
    ux = t[:, 0 * C : 1 * C]
    uy = t[:, 1 * C : 2 * C]
    mx = ux + shift
    my = uy + shift
    sxx = t[:, 2 * C : 3 * C] - ux * ux
    syy = t[:, 3 * C : 4 * C] - uy * uy
    sxy = t[:, 4 * C : 5 * C] - ux * uy

    cs = (2.0 * sxy + C2) / (sxx + syy + C2)
    lum = (2.0 * mx * my + C1) / (mx * mx + my * my + C1)
    return ((lum * cs).mean(dim=(2, 3), dtype=torch.float64),
            cs.mean(dim=(2, 3), dtype=torch.float64))


class _FusedSSIMCS(torch.autograd.Function):
    """Differentiable native ``(ssim, cs)`` plane means -- MS-SSIM's primitive.

    Nothing full-resolution crosses between forward and backward: only x, y and
    the window are saved, and the local moments are recomputed in the backward
    pass. Recomputing them costs one more read of the two input planes;
    *storing* them would cost five full-resolution planes per scale, which for
    a five-scale MS-SSIM is what makes the usual implementation's activation
    memory an order of magnitude bigger than the images it is scoring.
    """

    @staticmethod
    def forward(ctx, x, y, win, shift, C1, C2):
        from . import backend

        s, cs = backend.ssim_cs_forward(x, y, win, shift, C1, C2)
        ctx.save_for_backward(x, y, win)
        ctx.consts = (shift, C1, C2)
        return s, cs

    @staticmethod
    def backward(ctx, g_s, g_cs):
        from . import backend

        _reject_double_backward()
        x, y, win = ctx.saved_tensors
        shift, C1, C2 = ctx.consts
        need_dx = ctx.needs_input_grad[0]
        need_dy = ctx.needs_input_grad[1]
        if not (need_dx or need_dy):
            return (None,) * 6
        dx, dy = backend.ssim_cs_backward(x, y, g_s, g_cs, win, shift, C1, C2,
                                          need_dx, need_dy)
        return (dx if need_dx else None, dy if need_dy else None,
                None, None, None, None)


def _ssim_cs_pair(x4, y4, win, wdt, shift, C1, C2, backend_hint):
    """``(ssim, cs)``, each ``(N, C)``, from whichever backend can serve it."""
    from . import backend

    if (backend_hint != "torch"
            and (x4.requires_grad or y4.requires_grad)
            and torch.is_grad_enabled()
            and backend.ssim_cs_autograd_available(x4, y4, win.numel())):
        n, c = x4.shape[0], x4.shape[1]
        s, cs = _FusedSSIMCS.apply(x4, y4, win, shift, C1, C2)
        return s.view(n, c), cs.view(n, c)

    if backend_hint != "torch":
        fast = backend.try_ssim_cs(x4, y4, win=win, shift=shift, C1=C1, C2=C2)
        if fast is not None:
            return fast
        if backend_hint == "native":
            raise RuntimeError("native SSIM/CS backend unavailable for these inputs")

    pack = _maybe_compile(_pack_planes, "pack")
    epi = _maybe_compile(_ssim_cs_epilogue, "epi:cs")
    filt = _sep_filter(pack(x4, y4, wdt, shift), win.to(wdt))
    return epi(filt, x4.shape[1], shift, C1, C2)


def ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    win_size: int = 11,
    sigma: float = 1.5,
    K: Sequence[float] = (0.01, 0.03),
    weights: Optional[Sequence[float]] = None,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    backend_hint: str = "auto",
    luma=None,
    crop_border: int = 0,
) -> torch.Tensor:
    """Multi-scale SSIM (Wang, Simoncelli, Bovik, Asilomar 2003).

    ``prod_j cs_j^{w_j}`` across the pyramid, with the coarsest scale
    contributing the full SSIM rather than just its contrast-structure term.
    Scales are separated by a non-overlapping 2x2 mean, the convention every
    PyTorch implementation uses (Wang's ``msssim.m`` uses a half-pixel-offset
    zero-padded box filter instead; see :mod:`frame_analytics.reference`).

    The image must survive ``len(weights) - 1`` halvings and still be at least
    ``win_size`` on a side, i.e. ``win_size * 2**(scales-1)``: 176 px for the
    default 5 scales and 11x11 window.

    Differentiable on every backend; this is the loss half of the
    "MS-SSIM + L1" objective from Zhao et al., *Loss Functions for Image
    Restoration with Neural Networks* (2016).

    One property worth knowing before using it as a loss: the per-scale factors
    are clamped at zero, so on genuinely anti-correlated content the value *and
    its gradient* are exactly zero. That is the standard formulation -- every
    implementation clamps, because a fractional power of a negative number is
    not real -- but it means MS-SSIM alone cannot pull a diverged model back.
    Mixing in an L1 term, as Zhao et al. do, removes the dead zone.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    w = tuple(float(v) for v in (weights if weights is not None else MS_SSIM_WEIGHTS))
    if not w:
        raise ValueError("weights must not be empty")

    x4, y4, L, mode = _prep(x, y, luma, crop_border, data_range)
    wdt = _work_dtype(x4, dtype)
    x4, y4 = _apply_luma(x4, y4, mode, L, wdt)
    K1, K2 = float(K[0]), float(K[1])
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    shift = 0.5 * L

    need = (win_size - 1) * (1 << (len(w) - 1)) + (1 << (len(w) - 1))
    if min(x4.shape[-2], x4.shape[-1]) < need:
        raise ValueError(
            f"image {tuple(x4.shape[-2:])} is too small for {len(w)} scales with a "
            f"{win_size}x{win_size} window; needs at least {need} px on a side"
        )

    # built at the working dtype, as `ssim` does: the native kernels narrow it
    # to float32 themselves, but the portable path would otherwise inherit a
    # float32-rounded window even when the caller asked for float64
    win = gaussian_window_1d(win_size, sigma, device=x4.device, dtype=wdt)
    cx, cy = x4, y4
    terms = []
    for i, wi in enumerate(w):
        if i:
            # Scale 0 is served straight from uint8 -- that is where most of
            # the bandwidth is -- so the conversion only happens here, folded
            # into the first pooling.
            if cx.dtype != wdt:
                pool = _maybe_compile(_cast_pool2, "cast_pool2")
                cx, cy = pool(cx, wdt), pool(cy, wdt)
            else:
                cx, cy = F.avg_pool2d(cx, 2), F.avg_pool2d(cy, 2)
        s, cs = _ssim_cs_pair(cx, cy, win, wdt, shift, C1, C2, backend_hint)
        terms.append(s if i == len(w) - 1 else cs)

    # One stacked power and one product, rather than a clamp/pow/multiply per
    # scale: these are (N, C) tensors, so every one of those was a kernel
    # launch costing far more than the arithmetic in it.
    #
    # The clamp is not optional. A fractional power of a negative number is not
    # real; every implementation in circulation clamps the same way, and it
    # only ever engages on anti-correlated content, where the factor is meant
    # to be ~0 anyway.
    stacked = torch.stack(terms).clamp_min(0.0)
    per_plane = stacked.pow(_ms_weights(w, x4.device)).prod(dim=0)

    per_image = per_plane.mean(dim=1)           # channels averaged after the product
    return per_image if reduction == "none" else per_image.mean()


# --------------------------------------------------------------------------- #
# GMSD
# --------------------------------------------------------------------------- #

_prewitt_cache: dict = {}


def _prewitt_pair(device, dtype) -> torch.Tensor:
    """``(2, 1, 3, 3)``: the horizontal and vertical Prewitt taps, /3."""
    key = (str(device), dtype)
    k = _prewitt_cache.get(key)
    if k is not None:
        return k
    row = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64) / 3.0
    kx = row.expand(3, 3).contiguous()
    ky = kx.t().contiguous()
    k = torch.stack([kx, ky]).unsqueeze(1).to(device=device, dtype=dtype)
    _prewitt_cache[key] = k
    return k


def _gms_map(x4, y4, wdt, T, eps, downsample):
    """The gradient-magnitude similarity map, shape ``(N, C, H', W')``."""
    xf = x4.to(wdt)
    yf = y4.to(wdt)
    if downsample:
        xf = F.avg_pool2d(xf, 2)
        yf = F.avg_pool2d(yf, 2)
    n, c, h, w = xf.shape
    if h < 3 or w < 3:
        raise ValueError(f"image {h}x{w} is smaller than the 3x3 Prewitt window")

    # x and y stacked into the batch axis: one convolution, not two.
    k = _prewitt_pair(xf.device, wdt)
    both = torch.cat([xf, yf], dim=0).reshape(2 * n * c, 1, h, w)
    with _no_autocast(both):
        g = F.conv2d(both, k)                   # (2NC, 2, H-2, W-2)
    # The two halves are sliced *outside* the compiled kernel: Dynamo guards on
    # a Python int argument and recompiles per value, so passing the split
    # point in would add a batch-size dimension to the guard set and walk into
    # the recompile limit on a mixed workload. The slices are views.
    nc = n * c
    q = _maybe_compile(_gms_sq, "gms:sq")(g[:nc], g[nc:], T, eps)
    return q.reshape(n, c, h - 2, w - 2)


def _gms_sq(ga, gb, T: float, eps: float):
    # eps inside the root keeps d|g|/dg finite where the gradient vanishes --
    # a flat patch is exactly the case that would otherwise emit NaN into a
    # training step.
    g1 = torch.sqrt((ga * ga).sum(dim=1) + eps)
    g2 = torch.sqrt((gb * gb).sum(dim=1) + eps)
    return (2.0 * g1 * g2 + T) / (g1 * g1 + g2 * g2 + T)


def _gms_stats(q):
    """``(mean, population variance)`` per image, float64.

    The map has to be widened before the variance is taken, not after. GMS
    values sit within ~1e-3 of 1.0 on a good match, so the variance can be
    smaller than the float32 rounding of ``q * q`` -- squaring first and
    widening second returns a variance of *exactly zero* for two nearly
    identical frames, which is precisely the regime the metric exists to
    resolve. ``var`` runs Welford, so there is no cancelling subtraction
    either.
    """
    f = q.reshape(q.shape[0], -1).double()
    return f.mean(dim=1), f.var(dim=1, correction=0)


def _gmsd_common(x, y, data_range, T, eps, downsample, dtype, luma, crop_border,
                 reduction, return_map, backend_hint):
    """``(map, None)`` or ``(None, (gms_mean, gmsd))``, per image."""
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    x4, y4, L, mode = _prep(x, y, luma, crop_border, data_range)
    wdt = _work_dtype(x4, dtype)
    x4, y4 = _apply_luma(x4, y4, mode, L, wdt)
    # T has the units of a squared gradient, hence the L^2 scaling of the
    # paper's 170 (which was quoted for 0..255 data).
    Tv = float(T) if T is not None else 170.0 * (L / 255.0) ** 2
    ev = float(eps) if eps is not None else (1e-6 * L) ** 2

    if backend_hint != "torch":
        from . import backend

        # Two cases the native kernel cannot serve: it never materialises the
        # map, and it works in float32 throughout, so a caller who asked for
        # float64 has to get the portable path. Both still have to *say* so
        # under backend_hint="native" rather than quietly hand back the
        # portable result -- that hint exists to make fallbacks loud.
        native_ok = not return_map and wdt != torch.float64
        fast = (backend.try_gmsd(x4, y4, T=Tv, eps=ev, downsample=downsample)
                if native_ok else None)
        if fast is not None:
            return None, fast
        if backend_hint == "native":
            raise RuntimeError(
                "native GMSD backend unavailable for these inputs"
                + (" (return_map=True has no native kernel)" if return_map else ""))

    q = _gms_map(x4, y4, wdt, Tv, ev, downsample)
    if return_map:
        return q, None
    mean, var = _gms_stats(q)
    return None, (mean, var.clamp_min(0.0).sqrt())


def gms(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    T: Optional[float] = None,
    eps: Optional[float] = None,
    downsample: bool = True,
    reduction: str = "mean",
    return_map: bool = False,
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
    backend_hint: str = "auto",
) -> torch.Tensor:
    """Mean gradient-magnitude similarity. Higher is better, 1.0 is identical.

    ``1 - gms`` is the well-conditioned training objective from the GMSD
    family: unlike :func:`gmsd` its gradient stays finite when the two images
    agree, which is exactly where a converging model spends its time.
    """
    q, stats = _gmsd_common(x, y, data_range, T, eps, downsample, dtype, luma,
                            crop_border, reduction, return_map, backend_hint)
    if q is not None:
        return q
    per_image = stats[0]
    return per_image if reduction == "none" else per_image.mean()


def gmsd(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    T: Optional[float] = None,
    eps: Optional[float] = None,
    downsample: bool = True,
    reduction: str = "mean",
    return_map: bool = False,
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
    backend_hint: str = "auto",
) -> torch.Tensor:
    """Gradient magnitude similarity deviation (Xue et al. 2014). Lower is better.

    The standard deviation -- not the mean -- of the gradient-magnitude
    similarity map: the paper's insight is that *how unevenly* a distortion
    damages local structure predicts subjective quality better than how much it
    damages it on average.

    Borders and downsampling follow this package's conventions rather than the
    released MATLAB's; see :func:`frame_analytics.reference.gmsd_reference` for
    exactly which two things differ and why.

    As a training objective prefer ``1 - gms(...)``: ``d sqrt(var)/d var`` is
    unbounded as the variance goes to zero, which is precisely what happens as
    a model converges.

    Resolution floor: the similarity map is float32, whose spacing near 1.0 is
    ~6e-8, so a deviation below ~1e-7 is measuring rounding rather than the
    image pair. ``gmsd(x, x)`` lands around 3e-08 rather than exactly 0 for
    that reason. Pass ``dtype=torch.float64`` (which takes the portable path)
    if you need to resolve that far down; a typical GMSD is ~0.03.
    """
    q, stats = _gmsd_common(x, y, data_range, T, eps, downsample, dtype, luma,
                            crop_border, reduction, return_map, backend_hint)
    if q is not None:
        return q
    per_image = stats[1]
    return per_image if reduction == "none" else per_image.mean()


# --------------------------------------------------------------------------- #
# pixel losses
#
# All three are one pass over both inputs with a scalar out, i.e. pure
# bandwidth: the only things that matter are reading uint8 as uint8, not
# materialising the residual, and accumulating wide enough that a 4K frame's
# 8.3M terms do not eat the mantissa.
# --------------------------------------------------------------------------- #


# These op codes are shared with the C++/CUDA kernels; keep them in step with
# the OP_* constants in csrc/.
_OP_MSE = 0
_OP_L1 = 1
_OP_CHARB = 2
_OP_HUBER = 3


# One top-level function per (penalty, reduction) rather than one with an `op`
# argument: Dynamo guards on a Python int parameter and recompiles the *same*
# code object once per value, which walks straight into the recompile limit in
# a mixed workload and then silently drops everything back to eager.
def _l1_mean(x, y, dtype, param):
    d = x.to(dtype) - y.to(dtype)
    return d.abs().sum(dtype=torch.float64) / d.numel()


def _l1_per_image(x, y, dtype, param):
    d = x.to(dtype) - y.to(dtype)
    return d.abs().sum(dim=(1, 2, 3), dtype=torch.float64) / d[0].numel()


def _charb_mean(x, y, dtype, param):
    d = x.to(dtype) - y.to(dtype)
    return torch.sqrt(d * d + param).sum(dtype=torch.float64) / d.numel()


def _charb_per_image(x, y, dtype, param):
    d = x.to(dtype) - y.to(dtype)
    return (torch.sqrt(d * d + param).sum(dim=(1, 2, 3), dtype=torch.float64)
            / d[0].numel())


def _huber_terms(d, delta):
    a = d.abs()
    return torch.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def _huber_mean(x, y, dtype, param):
    d = x.to(dtype) - y.to(dtype)
    return _huber_terms(d, param).sum(dtype=torch.float64) / d.numel()


def _huber_per_image(x, y, dtype, param):
    d = x.to(dtype) - y.to(dtype)
    return (_huber_terms(d, param).sum(dim=(1, 2, 3), dtype=torch.float64)
            / d[0].numel())


_PIXEL_OPS = {
    "l1": (_OP_L1, _l1_mean, _l1_per_image),
    "charbonnier": (_OP_CHARB, _charb_mean, _charb_per_image),
    "huber": (_OP_HUBER, _huber_mean, _huber_per_image),
}


def _pixel_loss(name, x, y, param, reduction, dtype, out_dtype, luma,
                crop_border, backend_hint):
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    x4, y4, L, mode = _prep(x, y, luma, crop_border, None)
    wdt = _work_dtype(x4, dtype)
    x4, y4 = _apply_luma(x4, y4, mode, L, wdt)
    op, fn_mean, fn_per_image = _PIXEL_OPS[name]
    per_image = reduction == "none"

    from . import backend

    if backend_hint != "torch":
        fast = backend.try_pixel(x4, y4, op=op, param=param,
                                 per_image=per_image, dtype=out_dtype)
        if fast is not None:
            return fast
        if backend_hint == "native":
            raise RuntimeError(f"native {name} backend unavailable for these inputs")

    fn = _maybe_compile(fn_per_image if per_image else fn_mean,
                        f"pixel:{name}:{per_image}")
    return fn(x4, y4, wdt, param).to(out_dtype)


def l1(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    out_dtype: torch.dtype = torch.float64,
    luma=None,
    crop_border: int = 0,
    backend_hint: str = "auto",
) -> torch.Tensor:
    """Mean absolute error, in the input's own units."""
    return _pixel_loss("l1", x, y, 0.0, reduction, dtype, out_dtype, luma,
                       crop_border, backend_hint)


def charbonnier(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-3,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    out_dtype: torch.dtype = torch.float64,
    luma=None,
    crop_border: int = 0,
    backend_hint: str = "auto",
) -> torch.Tensor:
    """``mean sqrt((x - y)^2 + eps^2)`` -- the Charbonnier / pseudo-Huber penalty.

    ``eps`` has the units of the residual (LapSRN's 1e-3, quoted for 0..1
    data), so it is the width of the quadratic region, not its square.
    """
    return _pixel_loss("charbonnier", x, y, float(eps) ** 2, reduction, dtype,
                       out_dtype, luma, crop_border, backend_hint)


def huber(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    delta: float = 1.0,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    out_dtype: torch.dtype = torch.float64,
    luma=None,
    crop_border: int = 0,
    backend_hint: str = "auto",
) -> torch.Tensor:
    """Huber loss, matching ``torch.nn.HuberLoss`` term for term."""
    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")
    return _pixel_loss("huber", x, y, float(delta), reduction, dtype, out_dtype,
                       luma, crop_border, backend_hint)
