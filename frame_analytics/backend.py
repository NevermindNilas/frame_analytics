"""Native backend: load the C-ABI kernel libraries and dispatch to them.

The extension is entirely optional -- if it will not load, every public
function silently keeps using the portable PyTorch path.  Set ``FA_NATIVE=0``
to skip it, ``FA_NATIVE=1`` to make failures loud.

Three tiers, in order:

1. **prebuilt** -- a shared library shipped inside the wheel.  It talks the C
   ABI in ``csrc/fa_abi.h``, so it is not tied to a Python version or a torch
   build and one binary per platform serves everybody.
2. **JIT** -- the same sources compiled on demand by the host's compiler and
   cached, for platforms no wheel was built for.
3. **portable** -- pure PyTorch, in ``functional.py``.  Never fails.

Everything below is the marshalling layer for tiers 1 and 2: shapes and dtypes
become ints, tensors become raw pointers, and every output buffer -- including
scratch -- is allocated here rather than inside the kernel.  That last point is
not tidiness.  ``StreamingMetrics`` captures a CUDA graph, ``cudaMalloc`` is
illegal during capture, and torch's caching allocator is capture-aware; letting
the C side allocate would break graph capture outright.
"""

from __future__ import annotations

import ctypes
import os
import warnings
from typing import Any, Optional

import torch

from ._cabi import KernelError, ensure_host_compiler, status_text  # noqa: F401

_SUPPORTED = (torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.float64)

_DTYPE_CODE = {
    torch.uint8: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.float64: 4,
}

_PSNR_NONE = -1.0e300
_OP_MSE = 0  # mirrors functional._OP_MSE; duplicated to avoid an import cycle


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


class _Native:
    """What ``load()`` hands back: the two libraries plus where they came from."""

    def __init__(self) -> None:
        self.cpu: Any = None
        self.cpu_origin: Any = None
        self.cpu_isa: Any = None
        self.cuda: Any = None
        self.cuda_origin: Any = None
        self.cuda_error: Optional[str] = None

    @property
    def has_cuda(self) -> bool:
        return self.cuda is not None


_ext: Optional[_Native] = None
_load_attempted = False
_load_error: Optional[str] = None


def load(force: bool = False) -> Optional[_Native]:
    """Load (building first if necessary) and return the native libraries."""
    global _ext, _load_attempted, _load_error
    if _ext is not None and not force:
        return _ext
    if _load_attempted and not force:
        return None
    _load_attempted = True

    mode = os.environ.get("FA_NATIVE", "auto").lower()
    loud = mode in ("1", "on", "true", "yes")
    if mode in ("0", "off", "false", "no"):
        _load_error = "disabled via FA_NATIVE"
        return None

    from . import _cabi

    native = _Native()
    try:
        native.cpu, native.cpu_origin, native.cpu_isa = _cabi.load_cpu()
        # the pool is built on first use, so the thread count has to land
        # before any kernel runs; there is no ATen to ask across a C boundary
        native.cpu.fa_cpu_set_num_threads(int(torch.get_num_threads()))
        _load_error = None
    except Exception as exc:  # pragma: no cover - environment dependent
        _ext = None
        _load_error = f"{type(exc).__name__}: {exc}"
        if loud:
            raise
        warnings.warn(f"frame_analytics native backend unavailable ({_load_error}); "
                      "using the portable PyTorch path", RuntimeWarning, stacklevel=2)
        return None

    # CUDA is a separate library and a separate failure: a machine with no
    # toolkit, or macOS, still gets the CPU kernels.
    if torch.cuda.is_available():
        try:
            native.cuda, native.cuda_origin = _cabi.load_cuda()
        except Exception as exc:  # pragma: no cover - environment dependent
            native.cuda = None
            native.cuda_error = f"{type(exc).__name__}: {exc}"
            if loud:
                raise
            warnings.warn(
                f"frame_analytics CUDA kernels unavailable ({native.cuda_error}); "
                "CUDA calls will use the portable PyTorch path", RuntimeWarning,
                stacklevel=2)

    _ext = native
    return _ext


def status() -> dict:
    ext = load()
    return {
        "available": ext is not None,
        "cuda": bool(ext is not None and ext.has_cuda),
        "error": _load_error,
        "cpu_source": None if ext is None else ext.cpu_origin,
        "cpu_isa": None if ext is None else ext.cpu_isa,
        "cuda_source": None if ext is None else ext.cuda_origin,
        "cuda_error": None if ext is None else ext.cuda_error,
    }


# --------------------------------------------------------------------------- #
# marshalling helpers
# --------------------------------------------------------------------------- #


def _f64(device, *shape) -> torch.Tensor:
    """Uninitialised float64 output/scratch on ``device``.

    Every buffer the kernels write, scratch included, is allocated here.  Doing
    it on the C side would mean cudaMalloc, which is illegal inside the CUDA
    graph capture that StreamingMetrics performs; torch's caching allocator is
    capture-aware, so this route stays legal.
    """
    return torch.empty(shape, device=device, dtype=torch.float64)


def _contig(t: torch.Tensor) -> torch.Tensor:
    """``t.contiguous()`` without paying the dispatcher when it is a no-op.

    The overwhelmingly common case is an already-contiguous frame, and
    ``is_contiguous()`` is a direct attribute read where ``contiguous()`` is a
    full ATen dispatch.
    """
    return t if t.is_contiguous() else t.contiguous()


def _ptr(t: Optional[torch.Tensor]):
    """A tensor's data pointer, as a plain int.

    ctypes accepts an int wherever the prototype says ``c_void_p``, and skipping
    the wrapper object matters more than it looks: the smallest calls here are
    ~10 us end to end and mostly Python, so three object constructions per call
    is a percent or two of PSNR on a sub-megapixel frame.
    """
    return None if t is None else t.data_ptr()


def _dev(t: torch.Tensor) -> int:
    i = t.device.index
    return torch.cuda.current_device() if i is None else int(i)


def _stream(t: torch.Tensor):
    return torch.cuda.current_stream(t.device).cuda_stream


def _check(code: int, ext: _Native) -> None:
    if code != 0:
        raise KernelError(status_text(code, ext.cuda))


_win_cache: dict = {}


def _win_f32(win: torch.Tensor) -> torch.Tensor:
    """The window as contiguous float32, without recasting it every call.

    ``gaussian_window_1d`` caches its windows for the lifetime of the process,
    so keying on identity is safe and keeps the (device-side) cast off the hot
    path -- and out of a CUDA graph capture.
    """
    if win.dtype == torch.float32:
        return win if win.is_contiguous() else win.contiguous()
    key = (id(win), str(win.device))
    hit = _win_cache.get(key)
    if hit is not None and hit[0] is win:
        return hit[1]
    w = win.to(torch.float32).contiguous()
    _win_cache[key] = (win, w)
    return w


_warned: set = set()


def _warn_once(exc: BaseException) -> None:
    """A native-kernel failure is silently survivable -- the caller falls back
    to the PyTorch path -- but it should never be *invisible*, or a genuine bug
    just looks like a mysterious slowdown."""
    key = f"{type(exc).__name__}: {exc}"
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(f"frame_analytics native kernel declined this call ({key}); "
                  "falling back to the PyTorch path", RuntimeWarning, stacklevel=3)


def _usable(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Only the CUDA SSIM paths have a native backward; everything else is
    inference-only and defers to autograd when a gradient is actually wanted.

    "Actually wanted" is the operative word: ``requires_grad`` alone is not
    enough. The ordinary evaluation shape is ``pred = model(x)`` outside a
    ``no_grad`` block and the scoring inside one, which leaves ``requires_grad``
    set on a tensor that will never be differentiated. Testing the flag on its
    own turned every such call away from the native kernel -- an 8x slowdown on
    the single most common way anybody uses this library.
    """
    grad_wanted = (torch.is_grad_enabled()
                   and (x.requires_grad or y.requires_grad))
    return (
        x.dtype in _SUPPORTED
        and x.dtype == y.dtype
        and not grad_wanted
    )


# Block counts are a property of the kernel, so Python has to ask for them --
# but the answer only depends on the shape, and asking is a CUDA runtime call.
# Caching it keeps the query out of a graph capture entirely, since the warm-up
# iteration has already populated the entry the captured one needs.
_ws_cache: dict = {}


def _pixel_workspace(ext: _Native, n_images: int, n_per_image: int, dev: int) -> int:
    key = ("pix", n_images, n_per_image, dev)
    hit = _ws_cache.get(key)
    if hit is None:
        out = ctypes.c_int(0)
        _check(ext.cuda.fa_cuda_pixel_workspace(n_images, n_per_image, dev,
                                                ctypes.byref(out)), ext)
        hit = int(out.value)
        _ws_cache[key] = hit
    return hit


def _ssim_workspace(ext: _Native, h: int, w: int, k: int) -> int:
    key = ("ssim", h, w, k)
    hit = _ws_cache.get(key)
    if hit is None:
        out = ctypes.c_int(0)
        _check(ext.cuda.fa_cuda_ssim_workspace(h, w, k, ctypes.byref(out)), ext)
        hit = int(out.value)
        _ws_cache[key] = hit
    return hit


def _gmsd_workspace(ext: _Native, h: int, w: int, downsample: bool) -> int:
    key = ("gmsd", h, w, bool(downsample))
    hit = _ws_cache.get(key)
    if hit is None:
        out = ctypes.c_int(0)
        _check(ext.cuda.fa_cuda_gmsd_workspace(h, w, int(downsample),
                                               ctypes.byref(out)), ext)
        hit = int(out.value)
        _ws_cache[key] = hit
    return hit


# --------------------------------------------------------------------------- #
# pixel losses
# --------------------------------------------------------------------------- #


def _pixel_cuda(ext: _Native, x: torch.Tensor, y: torch.Tensor, *, op: int,
                param: float, per_image: bool,
                psnr_bias: float) -> torch.Tensor:
    n = int(x.shape[0])
    per = x.numel() // n
    dev = _dev(x)
    bpi = _pixel_workspace(ext, n, per, dev)
    partial = _f64(x.device, n, bpi)

    # Exactly one output tensor is allocated: the finalize kernel skips every
    # null sink, and at 512x512 an extra empty() plus its wrapper was a
    # measurable slice of the whole call.
    want_psnr = psnr_bias != _PSNR_NONE
    out = _f64(x.device, n) if per_image else _f64(x.device)
    sinks: list = [None, None, None, None]
    sinks[(2 if want_psnr else 0) + (0 if per_image else 1)] = out

    _check(ext.cuda.fa_cuda_pixel_reduce(
        _ptr(x), _ptr(y), _DTYPE_CODE[x.dtype], n, per, op, float(param),
        float(psnr_bias), _ptr(partial), bpi,
        _ptr(sinks[0]), _ptr(sinks[1]), _ptr(sinks[2]), _ptr(sinks[3]),
        dev, _stream(x)), ext)
    return out


def _pixel_cpu(ext: _Native, x: torch.Tensor, y: torch.Tensor, *, op: int,
               param: float, per_image: bool,
               psnr_bias: float) -> torch.Tensor:
    n = int(x.shape[0])
    per = x.numel() // n
    out = torch.empty((n,) if per_image else (), dtype=torch.float64)
    _check(ext.cpu.fa_cpu_pixel_reduce(
        _ptr(x), _ptr(y), _DTYPE_CODE[x.dtype], n, per, op, float(param),
        float(psnr_bias), int(per_image), _ptr(out)), ext)
    return out


def try_mse(x: torch.Tensor, y: torch.Tensor, *, per_image: bool,
            dtype: torch.dtype,
            psnr_bias: Optional[float] = None) -> Optional[torch.Tensor]:
    """Squared-error reduction.

    ``psnr_bias`` is ``10*log10(L^2)``; when given, the dB conversion happens
    inside the same device kernel and the PSNR is returned instead of the MSE.
    That removes two kernel launches from every PSNR call, which is most of the
    cost at small frame sizes.
    """
    ext = load()
    if ext is None or not _usable(x, y):
        return None
    bias = psnr_bias if psnr_bias is not None else _PSNR_NONE
    try:
        x, y = _contig(x), _contig(y)
        if x.is_cuda:
            if not ext.has_cuda:
                return None
            return _pixel_cuda(ext, x, y, op=_OP_MSE, param=0.0,
                               per_image=per_image, psnr_bias=bias)
        return _pixel_cpu(ext, x, y, op=_OP_MSE, param=0.0,
                          per_image=per_image, psnr_bias=bias)
    except Exception as exc:
        _warn_once(exc)
        return None


def try_pixel(x: torch.Tensor, y: torch.Tensor, *, op: int, param: float,
              per_image: bool, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """L1 / Charbonnier / Huber reduction; ``None`` if no kernel applies."""
    ext = load()
    if ext is None or not _usable(x, y):
        return None
    try:
        x, y = _contig(x), _contig(y)
        if x.is_cuda:
            if not ext.has_cuda:
                return None
            out = _pixel_cuda(ext, x, y, op=op, param=param,
                              per_image=per_image, psnr_bias=_PSNR_NONE)
        else:
            out = _pixel_cpu(ext, x, y, op=op, param=param,
                             per_image=per_image, psnr_bias=_PSNR_NONE)
        return out.to(dtype)
    except Exception as exc:
        _warn_once(exc)
        return None


# --------------------------------------------------------------------------- #
# SSIM
# --------------------------------------------------------------------------- #


def ssim_autograd_available(x: torch.Tensor, y: torch.Tensor, win_size: int) -> bool:
    """Can the differentiable native SSIM path handle this call?"""
    ext = load()
    return (
        ext is not None
        and ext.has_cuda
        and x.is_cuda
        and x.dtype == torch.float32
        and y.dtype == torch.float32
        and x.shape == y.shape
        and x.dim() == 4
        and win_size in (3, 5, 7, 9, 11)
        and not torch.cuda.is_current_stream_capturing()
    )


def ssim_fused(x: torch.Tensor, y: torch.Tensor, win: torch.Tensor,
               shift: float, C1: float, C2: float, want_map: bool):
    """Forward half of the differentiable native SSIM.

    Returns ``[per_image, total]``, plus the map when asked for it.
    """
    ext = load()
    x, y = _contig(x), _contig(y)
    winf = _win_f32(win)
    n, c, h, w = x.shape
    k = int(winf.numel())
    hout, wout = h - k + 1, w - k + 1
    bpp = _ssim_workspace(ext, h, w, k)

    partial = _f64(x.device, n * c, bpp)
    per_image = _f64(x.device, n)
    total = _f64(x.device)
    smap = torch.empty((n, c, hout, wout), device=x.device,
                       dtype=torch.float32) if want_map else None

    _check(ext.cuda.fa_cuda_ssim(
        _ptr(x), _ptr(y), _DTYPE_CODE[x.dtype], n, c, h, w, _ptr(winf), k,
        float(shift), float(C1), float(C2), _ptr(smap), _ptr(partial), bpp,
        _ptr(per_image), _ptr(total), _dev(x), _stream(x)), ext)
    return [per_image, total, smap] if want_map else [per_image, total]


def ssim_backward(x, y, dL_dmap, grad_scalar, win, shift, C1, C2,
                  need_dx, need_dy):
    ext = load()
    x, y = _contig(x), _contig(y)
    winf = _win_f32(win)
    n, c, h, w = x.shape
    k = int(winf.numel())
    hout, wout = h - k + 1, w - k + 1
    planes = n * c

    has_map = dL_dmap is not None and dL_dmap.numel() > 0
    dmap = dL_dmap.to(torch.float32).contiguous() if has_map else None
    gscalar = None
    if not has_map:
        gscalar = grad_scalar.to(torch.float32).contiguous().reshape(1)

    naux = 4 if need_dy else 3
    aux = torch.empty((naux, planes, hout, wout), device=x.device,
                      dtype=torch.float32)
    dx = torch.empty_like(x) if need_dx else None
    dy = torch.empty_like(y) if need_dy else None

    _check(ext.cuda.fa_cuda_ssim_backward(
        _ptr(x), _ptr(y), _ptr(dmap), _ptr(gscalar), _ptr(winf), k,
        n, c, h, w, float(shift), float(C1), float(C2),
        _ptr(aux), _ptr(dx), _ptr(dy), _dev(x), _stream(x)), ext)
    return dx, dy


def try_ssim(x: torch.Tensor, y: torch.Tensor, *, win_size: int, sigma: float,
             shift: float, C1: float, C2: float, reduction: str,
             return_map: bool, dtype: torch.dtype) -> Optional[torch.Tensor]:
    ext = load()
    if ext is None or not _usable(x, y):
        return None
    # the native kernels are instantiated for odd windows up to 11 (the paper's
    # 11 is the default; anything else falls through to the PyTorch path)
    if win_size not in (3, 5, 7, 9, 11) or dtype == torch.float64:
        return None

    from .functional import gaussian_window_1d

    # Built directly on the target device and cached there. Materialising it on
    # the host and copying per call would add an H2D transfer to every
    # invocation -- and would make the call illegal inside a CUDA graph capture.
    win = _win_f32(gaussian_window_1d(win_size, sigma, device=x.device,
                                      dtype=torch.float32))
    n, c = int(x.shape[0]), int(x.shape[1])
    h, w = int(x.shape[-2]), int(x.shape[-1])
    hout, wout = h - win_size + 1, w - win_size + 1
    npix = hout * wout

    try:
        x, y = _contig(x), _contig(y)
        if x.is_cuda:
            if not ext.has_cuda:
                return None
            bpp = _ssim_workspace(ext, h, w, win_size)
            partial = _f64(x.device, n * c, bpp)
            smap = None
            per_image = total = None
            if return_map:
                smap = torch.empty((n, c, hout, wout), device=x.device,
                                   dtype=torch.float32)
            elif reduction == "none":
                per_image = _f64(x.device, n)
            else:
                total = _f64(x.device)
            _check(ext.cuda.fa_cuda_ssim(
                _ptr(x), _ptr(y), _DTYPE_CODE[x.dtype], n, c, h, w,
                _ptr(win), win_size, float(shift), float(C1), float(C2),
                _ptr(smap), _ptr(partial), bpp, _ptr(per_image), _ptr(total),
                _dev(x), _stream(x)), ext)
            if return_map:
                return smap.to(dtype)
            return per_image if reduction == "none" else total

        per_plane = torch.empty((n * c,), dtype=torch.float64)
        smap = torch.empty((n, c, hout, wout),
                           dtype=torch.float32) if return_map else None
        _check(ext.cpu.fa_cpu_ssim(
            _ptr(x), _ptr(y), _DTYPE_CODE[x.dtype], n, c, h, w,
            _ptr(win), win_size, float(shift), float(C1), float(C2),
            _ptr(smap), _ptr(per_plane)), ext)
    except Exception as exc:
        _warn_once(exc)
        return None

    if return_map:
        return smap.to(dtype)
    # scalar reductions stay in float64 to match the portable path
    per_image = per_plane.view(n, c).sum(dim=1) / (npix * c)
    if reduction == "none":
        return per_image
    return per_image.mean()


# --------------------------------------------------------------------------- #
# MS-SSIM's per-scale primitive
# --------------------------------------------------------------------------- #


def ssim_cs_autograd_available(x: torch.Tensor, y: torch.Tensor,
                               win_size: int) -> bool:
    """Can the differentiable native MS-SSIM primitive handle this call?"""
    ext = load()
    return (
        ext is not None
        and ext.has_cuda
        and x.is_cuda
        and x.dtype == torch.float32
        and y.dtype == torch.float32
        and x.shape == y.shape
        and x.dim() == 4
        and win_size in (3, 5, 7, 9, 11)
        and not torch.cuda.is_current_stream_capturing()
    )


def _ssim_cs_cuda(ext: _Native, x, y, winf, shift, C1, C2):
    n, c, h, w = x.shape
    k = int(winf.numel())
    planes = n * c
    bpp = _ssim_workspace(ext, h, w, k)
    partial = _f64(x.device, planes, bpp)
    partial_cs = _f64(x.device, planes, bpp)
    out_s = _f64(x.device, planes)
    out_cs = _f64(x.device, planes)
    _check(ext.cuda.fa_cuda_ssim_cs(
        _ptr(x), _ptr(y), _DTYPE_CODE[x.dtype], n, c, h, w, _ptr(winf), k,
        float(shift), float(C1), float(C2), _ptr(partial), _ptr(partial_cs),
        bpp, _ptr(out_s), _ptr(out_cs), _dev(x), _stream(x)), ext)
    return out_s, out_cs


def ssim_cs_forward(x, y, win, shift, C1, C2):
    """``(ssim, cs)`` plane means from the differentiable native path."""
    ext = load()
    return _ssim_cs_cuda(ext, x.contiguous(), y.contiguous(), _win_f32(win),
                         shift, C1, C2)


def ssim_cs_backward(x, y, gs, gcs, win, shift, C1, C2, need_dx, need_dy):
    ext = load()
    x, y = _contig(x), _contig(y)
    winf = _win_f32(win)
    n, c, h, w = x.shape
    k = int(winf.numel())
    hout, wout = h - k + 1, w - k + 1
    planes = n * c

    gsf = gs.to(torch.float32).contiguous().reshape(-1)
    gcsf = gcs.to(torch.float32).contiguous().reshape(-1)
    naux = 4 if need_dy else 3
    aux = torch.empty((naux, planes, hout, wout), device=x.device,
                      dtype=torch.float32)
    dx = torch.empty_like(x) if need_dx else None
    dy = torch.empty_like(y) if need_dy else None

    _check(ext.cuda.fa_cuda_ssim_cs_backward(
        _ptr(x), _ptr(y), _ptr(gsf), _ptr(gcsf), _ptr(winf), k,
        n, c, h, w, float(shift), float(C1), float(C2),
        _ptr(aux), _ptr(dx), _ptr(dy), _dev(x), _stream(x)), ext)
    return dx, dy


def try_ssim_cs(x: torch.Tensor, y: torch.Tensor, *, win: torch.Tensor,
                shift: float, C1: float, C2: float):
    """``(ssim, cs)`` per plane, shaped ``(N, C)``; ``None`` if unavailable."""
    ext = load()
    if ext is None or not _usable(x, y):
        return None
    k = int(win.numel())
    if x.dtype == torch.float64 or k not in (3, 5, 7, 9, 11):
        return None
    if x.shape[-2] < k or x.shape[-1] < k:
        return None
    n, c = int(x.shape[0]), int(x.shape[1])
    try:
        x, y = _contig(x), _contig(y)
        winf = _win_f32(win)
        if x.is_cuda:
            if not ext.has_cuda:
                return None
            s, cs = _ssim_cs_cuda(ext, x, y, winf, shift, C1, C2)
        else:
            h, w = int(x.shape[-2]), int(x.shape[-1])
            s = torch.empty((n * c,), dtype=torch.float64)
            cs = torch.empty((n * c,), dtype=torch.float64)
            _check(ext.cpu.fa_cpu_ssim_cs(
                _ptr(x), _ptr(y), _DTYPE_CODE[x.dtype], n, c, h, w,
                _ptr(winf), k, float(shift), float(C1), float(C2),
                _ptr(s), _ptr(cs)), ext)
    except Exception as exc:
        _warn_once(exc)
        return None
    return s.view(n, c), cs.view(n, c)


# --------------------------------------------------------------------------- #
# GMSD
# --------------------------------------------------------------------------- #


def try_gmsd(x: torch.Tensor, y: torch.Tensor, *, T: float, eps: float,
             downsample: bool):
    """``(gms_mean, gmsd)`` per image, float64; ``None`` if unavailable."""
    ext = load()
    if ext is None or not _usable(x, y):
        return None
    n, c = int(x.shape[0]), int(x.shape[1])
    h, w = int(x.shape[-2]), int(x.shape[-1])
    try:
        x, y = _contig(x), _contig(y)
        if x.is_cuda:
            if not ext.has_cuda:
                return None
            bpp = _gmsd_workspace(ext, h, w, downsample)
            psum = _f64(x.device, n * c, bpp)
            psumsq = _f64(x.device, n * c, bpp)
            mean = _f64(x.device, n)
            dev = _f64(x.device, n)
            _check(ext.cuda.fa_cuda_gmsd(
                _ptr(x), _ptr(y), _DTYPE_CODE[x.dtype], n, c, h, w,
                float(T), float(eps), int(downsample), _ptr(psum),
                _ptr(psumsq), bpp, _ptr(mean), _ptr(dev),
                _dev(x), _stream(x)), ext)
        else:
            mean = torch.empty((n,), dtype=torch.float64)
            dev = torch.empty((n,), dtype=torch.float64)
            _check(ext.cpu.fa_cpu_gmsd(
                _ptr(x), _ptr(y), _DTYPE_CODE[x.dtype], n, c, h, w,
                float(T), float(eps), int(downsample), _ptr(mean),
                _ptr(dev)), ext)
    except Exception as exc:
        _warn_once(exc)
        return None
    return mean, dev
