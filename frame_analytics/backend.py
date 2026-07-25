"""Native (C++/CUDA) backend: build, load, dispatch.

The extension is entirely optional -- if it will not build, every public
function silently keeps using the portable PyTorch path.  Set ``FA_NATIVE=0``
to skip it, ``FA_NATIVE=1`` to make build failures loud.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch

_CSRC = Path(__file__).resolve().parent / "csrc"
_ext = None
_load_attempted = False
_load_error: Optional[str] = None

_SUPPORTED = (torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.float64)


# --------------------------------------------------------------------------- #
# MSVC discovery (Windows only)
# --------------------------------------------------------------------------- #


_msvc_done = False


def ensure_host_compiler() -> None:
    """Idempotent wrapper -- also used by the ``torch.compile`` CPU backend,
    which needs ``cl.exe`` on PATH to probe for AVX support."""
    global _msvc_done
    if _msvc_done:
        return
    _msvc_done = True
    try:
        _ensure_msvc_env()
    except Exception:
        pass


def _ensure_msvc_env() -> None:
    """Import a Visual Studio build environment into ``os.environ``.

    ``torch.utils.cpp_extension`` shells out to ``cl.exe`` and ``nvcc`` and
    expects both on PATH.  On a machine where VS was installed but no developer
    prompt is active, we source ``vcvars64.bat`` ourselves.
    """
    if sys.platform != "win32":
        return
    from shutil import which

    if which("cl") is not None:
        return

    candidates = []
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.exists():
        try:
            out = subprocess.run(
                [str(vswhere), "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
            if out:
                candidates.append(Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")
        except Exception:
            pass
    for root in (r"C:\Program Files\Microsoft Visual Studio",
                 r"C:\Program Files (x86)\Microsoft Visual Studio"):
        p = Path(root)
        if p.exists():
            candidates.extend(sorted(p.glob("*/*/VC/Auxiliary/Build/vcvars64.bat"), reverse=True))

    for vc in candidates:
        if not vc.exists():
            continue
        try:
            res = subprocess.run(f'"{vc}" >nul 2>&1 && set', shell=True,
                                 capture_output=True, text=True, timeout=120)
        except Exception:
            continue
        if res.returncode != 0:
            continue
        for line in res.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
        if which("cl") is not None:
            return


# --------------------------------------------------------------------------- #
# build / load
# --------------------------------------------------------------------------- #


def _arch_flags() -> list:
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]


def load(force: bool = False):
    """Compile (once) and return the native module, or ``None``."""
    global _ext, _load_attempted, _load_error
    if _ext is not None and not force:
        return _ext
    if _load_attempted and not force:
        return None
    _load_attempted = True

    mode = os.environ.get("FA_NATIVE", "auto").lower()
    if mode in ("0", "off", "false", "no"):
        _load_error = "disabled via FA_NATIVE"
        return None

    try:
        ensure_host_compiler()
        import torch.utils.cpp_extension as _cppext
        from torch.utils.cpp_extension import load as cpp_load

        sources = [str(_CSRC / "fa_cpu.cpp")]
        with_cuda = torch.cuda.is_available()
        # /Zc:preprocessor: CUDA 13's CCCL headers refuse to build against
        # MSVC's traditional preprocessor.
        extra_cflags = (
            ["/O2", "/fp:fast", "/arch:AVX2", "/Zc:preprocessor", "/DWITH_CUDA"]
            if sys.platform == "win32"
            else ["-O3", "-ffast-math", "-march=native", "-fopenmp", "-DWITH_CUDA"]
        )
        if not with_cuda:
            extra_cflags = [f for f in extra_cflags if "WITH_CUDA" not in f]
        else:
            sources.append(str(_CSRC / "fa_cuda.cu"))

        # Deliberately *not* --use_fast_math: it swaps in approximate division,
        # and the final num/den is where SSIM's accuracy lives. The kernel is
        # bandwidth-bound anyway, so IEEE division costs nothing measurable.
        cuda_flags = ["-O3", "-DWITH_CUDA", "--expt-relaxed-constexpr",
                      "-prec-div=true", "-prec-sqrt=true",
                      "-ftz=false"] + _arch_flags()
        if sys.platform == "win32":
            cuda_flags += ["-Xcompiler", "/Zc:preprocessor",
                           "-Xcompiler", "/wd4819"]

        build_dir = Path(os.environ.get("FA_BUILD_DIR",
                         Path(_cppext.get_default_build_root()) / "frame_analytics"))
        build_dir.mkdir(parents=True, exist_ok=True)

        _ext = cpp_load(
            name="frame_analytics_native",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=cuda_flags if with_cuda else None,
            with_cuda=with_cuda,
            build_directory=str(build_dir),
            verbose=os.environ.get("FA_VERBOSE", "0") == "1",
        )
        _load_error = None
    except Exception as exc:  # pragma: no cover - environment dependent
        _ext = None
        _load_error = f"{type(exc).__name__}: {exc}"
        if mode in ("1", "on", "true", "yes"):
            raise
        warnings.warn(f"frame_analytics native backend unavailable ({_load_error}); "
                      "using the portable PyTorch path", RuntimeWarning, stacklevel=2)
    return _ext


def status() -> dict:
    ext = load()
    return {
        "available": ext is not None,
        "cuda": bool(ext is not None and getattr(ext, "has_cuda", False)),
        "error": _load_error,
    }


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


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


def ssim_autograd_available(x: torch.Tensor, y: torch.Tensor, win_size: int) -> bool:
    """Can the differentiable native SSIM path handle this call?"""
    ext = load()
    return (
        ext is not None
        and getattr(ext, "has_cuda", False)
        and x.is_cuda
        and x.dtype == torch.float32
        and y.dtype == torch.float32
        and x.shape == y.shape
        and x.dim() == 4
        and win_size in (3, 5, 7, 9, 11)
        and not torch.cuda.is_current_stream_capturing()
    )


def ssim_backward(x, y, dL_dmap, grad_scalar, win, shift, C1, C2,
                  need_dx, need_dy):
    ext = load()
    return ext.ssim_backward_cuda(x, y, dL_dmap, grad_scalar, win,
                                  shift, C1, C2, need_dx, need_dy)


_PSNR_NONE = -1.0e300


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
    n = x.shape[0]
    per = x[0].numel()
    want_psnr = psnr_bias is not None
    try:
        if x.is_cuda:
            if not getattr(ext, "has_cuda", False):
                return None
            # comes back already reduced, divided and (optionally) in dB
            res = ext.mse_partial_cuda(x, y, n,
                                       psnr_bias if want_psnr else _PSNR_NONE)
            base = 2 if want_psnr else 0
            return res[base + (0 if per_image else 1)]
        sums = ext.mse_sums_cpu(x, y, n)                # (N,) float64
    except Exception as exc:
        _warn_once(exc)
        return None
    m = sums / per if per_image else sums.sum() / (per * n)
    if want_psnr:
        return psnr_bias - 10.0 * torch.log10(m)
    return m


def try_pixel(x: torch.Tensor, y: torch.Tensor, *, op: int, param: float,
              per_image: bool, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """L1 / Charbonnier / Huber reduction; ``None`` if no kernel applies."""
    ext = load()
    if ext is None or not _usable(x, y) or not hasattr(ext, "pixel_sums_cpu"):
        return None
    n = x.shape[0]
    per = x[0].numel()
    try:
        if x.is_cuda:
            if not getattr(ext, "has_cuda", False):
                return None
            res = ext.pixel_partial_cuda(x, y, n, op, float(param), _PSNR_NONE)
            return res[0 if per_image else 1].to(dtype)
        sums = ext.pixel_sums_cpu(x, y, n, op, float(param))   # (N,) float64
    except Exception as exc:
        _warn_once(exc)
        return None
    m = sums / per if per_image else sums.sum() / (per * n)
    return m.to(dtype)


def try_ssim_cs(x: torch.Tensor, y: torch.Tensor, *, win: torch.Tensor,
                shift: float, C1: float, C2: float):
    """``(ssim, cs)`` per plane, shaped ``(N, C)``; ``None`` if unavailable."""
    ext = load()
    if ext is None or not _usable(x, y) or not hasattr(ext, "ssim_cs_cpu"):
        return None
    k = win.numel()
    if x.dtype == torch.float64 or k not in (3, 5, 7, 9, 11):
        return None
    if x.shape[-2] < k or x.shape[-1] < k:
        return None
    try:
        if x.is_cuda:
            if not getattr(ext, "has_cuda", False):
                return None
            res = ext.ssim_cs_cuda(x, y, win.to(x.device), shift, C1, C2)
        else:
            res = ext.ssim_cs_cpu(x, y, win.cpu(), shift, C1, C2)
    except Exception as exc:
        _warn_once(exc)
        return None
    n, c = x.shape[0], x.shape[1]
    return res[0].view(n, c), res[1].view(n, c)


def ssim_cs_autograd_available(x: torch.Tensor, y: torch.Tensor,
                               win_size: int) -> bool:
    """Can the differentiable native MS-SSIM primitive handle this call?"""
    ext = load()
    return (
        ext is not None
        and getattr(ext, "has_cuda", False)
        and hasattr(ext, "ssim_cs_backward_cuda")
        and x.is_cuda
        and x.dtype == torch.float32
        and y.dtype == torch.float32
        and x.shape == y.shape
        and x.dim() == 4
        and win_size in (3, 5, 7, 9, 11)
        and not torch.cuda.is_current_stream_capturing()
    )


def ssim_cs_backward(x, y, gs, gcs, win, shift, C1, C2, need_dx, need_dy):
    ext = load()
    return ext.ssim_cs_backward_cuda(x, y, gs, gcs, win, shift, C1, C2,
                                     need_dx, need_dy)


def try_gmsd(x: torch.Tensor, y: torch.Tensor, *, T: float, eps: float,
             downsample: bool):
    """``(gms_mean, gmsd)`` per image, float64; ``None`` if unavailable."""
    ext = load()
    if ext is None or not _usable(x, y) or not hasattr(ext, "gmsd_sums_cpu"):
        return None
    try:
        if x.is_cuda:
            if not getattr(ext, "has_cuda", False):
                return None
            res = ext.gmsd_cuda(x, y, float(T), float(eps), bool(downsample))
        else:
            res = ext.gmsd_sums_cpu(x, y, float(T), float(eps), bool(downsample))
    except Exception as exc:
        _warn_once(exc)
        return None
    return res[0], res[1]


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
    win = gaussian_window_1d(win_size, sigma, device=x.device, dtype=torch.float32)
    n, c = x.shape[0], x.shape[1]
    hout = x.shape[-2] - win_size + 1
    wout = x.shape[-1] - win_size + 1
    npix = hout * wout

    try:
        if x.is_cuda:
            if not getattr(ext, "has_cuda", False):
                return None
            # (N,) and () come back already reduced and divided
            res = ext.ssim_fused_cuda(x, y, win, shift, C1, C2, return_map)
            if return_map:
                return res[2].to(dtype)
            return res[0] if reduction == "none" else res[1]
        res = ext.ssim_sums_cpu(x, y, win, shift, C1, C2, return_map)
        per_plane = res[0]                              # (N*C,) float64
    except Exception as exc:
        _warn_once(exc)
        return None

    if return_map:
        return res[1].to(dtype)
    # scalar reductions stay in float64 to match the portable path
    per_image = per_plane.view(n, c).sum(dim=1) / (npix * c)
    if reduction == "none":
        return per_image
    return per_image.mean()
