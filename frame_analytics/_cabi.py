"""Locate, build and bind the C-ABI kernel libraries.

Three things live here and nothing else does:

* **finding a prebuilt binary.**  Wheels ship ``frame_analytics/lib/`` with the
  shared libraries already compiled.  Because the boundary in ``fa_abi.h``
  mentions neither libtorch nor Python, one such binary per platform is valid
  for every Python version and every torch version -- which is the only reason
  prebuilt kernels can be published on PyPI at all.
* **building one on demand.**  A source-only install (or a platform CI never
  built for) compiles the same sources with the host's ``cl``/``c++``/``nvcc``
  and caches the result.  This is the old JIT path, minus torch's extension
  machinery: there is no ABI to match any more, so it needs no torch headers.
* **binding.**  ``ctypes`` prototypes, so a wrong argument is a Python
  ``TypeError`` rather than a corrupted stack.

The CPU library is compiled twice, at the platform baseline and with AVX2, and
the baseline build -- which is safe to load anywhere -- is asked at runtime
which one the host can actually run.  Shipping a single ``/arch:AVX2`` binary
would fault on pre-Haswell hardware; shipping only a baseline one would give up
the vector width the reduction loops exist for.

CUDA lives in its own library, loaded only when torch reports a CUDA device.
On macOS it is never built, never shipped and never referenced.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

_PKG = Path(__file__).resolve().parent
_CSRC = _PKG / "csrc"
_LIBDIR = _PKG / "lib"

FA_ABI_VERSION = 1

# Mirrors the codes in fa_abi.h.
FA_OK = 0
_STATUS_TEXT = {
    0: "ok",
    1: "unsupported dtype",
    2: "unsupported shape",
    3: "unsupported window size",
    4: "unknown pixel op",
    5: "invalid argument",
    6: "internal error",
    7: "no CUDA device",
}


class KernelError(RuntimeError):
    """A kernel entry point returned a non-zero status."""


def status_text(code: int, cuda_lib=None) -> str:
    if code < 0:
        if cuda_lib is not None:
            try:
                s = cuda_lib.fa_cuda_error_string(code)
                if s:
                    return s.decode("utf-8", "replace")
            except Exception:
                pass
        return f"CUDA runtime error {-code}"
    return _STATUS_TEXT.get(code, f"status {code}")


# --------------------------------------------------------------------------- #
# host compiler discovery (Windows)
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

    Building anything here shells out to ``cl.exe`` and ``nvcc`` and expects
    both on PATH.  On a machine where VS was installed but no developer prompt
    is active, we source ``vcvars64.bat`` ourselves.
    """
    if sys.platform != "win32":
        return
    if shutil.which("cl") is not None:
        return

    candidates: List[Path] = []
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
        if shutil.which("cl") is not None:
            return


# --------------------------------------------------------------------------- #
# naming / paths
# --------------------------------------------------------------------------- #


def library_filename(stem: str) -> str:
    if sys.platform == "win32":
        return f"{stem}.dll"
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def _is_x86() -> bool:
    return platform.machine().lower() in (
        "x86_64", "amd64", "x64", "i386", "i686", "x86",
    )


def build_root() -> Path:
    env = os.environ.get("FA_BUILD_DIR")
    if env:
        return Path(env)
    try:
        from torch.utils.cpp_extension import get_default_build_root

        return Path(get_default_build_root()) / "frame_analytics_cabi"
    except Exception:
        return Path(tempfile.gettempdir()) / "frame_analytics_cabi"


def _verbose() -> bool:
    return os.environ.get("FA_VERBOSE", "0") == "1"


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #


def _cuda_home() -> Optional[Path]:
    for var in ("CUDA_HOME", "CUDA_PATH"):
        v = os.environ.get(var)
        if v and Path(v).exists():
            return Path(v)
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME:
            return Path(CUDA_HOME)
    except Exception:
        pass
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    return None


def _nvcc() -> Optional[str]:
    home = _cuda_home()
    if home is not None:
        cand = home / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")
        if cand.exists():
            return str(cand)
    return shutil.which("nvcc")


def cuda_arch_flags() -> List[str]:
    """Architectures to build for.

    A local build targets the installed GPU, which halves compile time and is
    all a JIT build could ever need.  ``FA_CUDA_ARCHS`` overrides it -- CI sets
    a list plus a trailing PTX target so one binary keeps working on hardware
    newer than the toolkit that built it.
    """
    env = os.environ.get("FA_CUDA_ARCHS")
    if env:
        flags: List[str] = []
        for spec in env.replace(",", " ").split():
            spec = spec.strip()
            if not spec:
                continue
            if spec.endswith("+PTX"):
                num = spec[:-4].replace(".", "")
                flags.append(f"-gencode=arch=compute_{num},code=compute_{num}")
            else:
                num = spec.replace(".", "")
                flags.append(f"-gencode=arch=compute_{num},code=sm_{num}")
        return flags
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]
    except Exception:
        pass
    return []


def cpu_compile_command(source: Path, out: Path, avx2: bool,
                        objdir: Path) -> List[str]:
    if sys.platform == "win32":
        # /MT, not /MD: the library allocates nothing that crosses the boundary
        # (every buffer is the caller's), so a private static CRT is safe -- and
        # it drops the VC++ redistributable from the list of things a user has
        # to have installed for a downloaded wheel to import.
        cmd = ["cl", "/nologo", "/LD", "/O2", "/fp:fast", "/EHsc", "/std:c++17",
               "/Zc:preprocessor", "/DNDEBUG", "/MT"]
        if avx2:
            cmd.append("/arch:AVX2")
        cmd += [str(source), f"/Fo{objdir}{os.sep}", f"/Fe{out}"]
        return cmd
    cxx = os.environ.get("CXX") or "c++"
    cmd = [cxx, "-O3", "-ffast-math", "-std=c++17", "-fPIC", "-pthread",
           "-fvisibility=hidden", "-DNDEBUG"]
    if avx2:
        cmd += ["-mavx2", "-mfma"]
    if sys.platform == "darwin":
        cmd.append("-dynamiclib")
    else:
        # A released Linux wheel has to run against whatever libstdc++ the
        # user's distro shipped, and this library exports no C++ types -- so
        # linking the C++ runtime in statically removes the only ABI question
        # left and leaves glibc as the sole dependency.
        cmd += ["-shared", "-static-libstdc++", "-static-libgcc"]
    cmd += [str(source), "-o", str(out)]
    return cmd


def cuda_compile_command(source: Path, out: Path,
                         archs: Optional[Sequence[str]] = None) -> List[str]:
    nvcc = _nvcc()
    if nvcc is None:
        raise FileNotFoundError("nvcc not found")
    # Deliberately *not* --use_fast_math: it swaps in approximate division, and
    # the final num/den is where SSIM's accuracy lives. The kernel is
    # bandwidth-bound anyway, so IEEE division costs nothing measurable.
    #
    # -cudart static is what makes the result self-contained: the CUDA runtime
    # is linked in and the driver is opened lazily, so the binary imports
    # nothing but the platform C runtime.
    cmd = [nvcc, "-O3", "-shared", "-cudart", "static", "-std=c++17",
           "--expt-relaxed-constexpr", "-prec-div=true", "-prec-sqrt=true",
           "-ftz=false", "-DNDEBUG"]
    cmd += list(archs if archs is not None else cuda_arch_flags())
    if sys.platform == "win32":
        # CUDA 13's CCCL headers refuse to build against MSVC's traditional
        # preprocessor; /wd4819 silences the codepage warning on non-UTF8 hosts.
        # /MT matches the statically linked cudart, which is built against the
        # static CRT; /MD here produces an LNK4098 and two runtimes in one DLL.
        cmd += ["-Xcompiler", "/Zc:preprocessor", "-Xcompiler", "/wd4819",
                "-Xcompiler", "/MT"]
    else:
        cmd += ["-Xcompiler", "-fPIC", "-Xcompiler", "-fvisibility=hidden",
                "-Xcompiler", "-static-libstdc++", "-Xcompiler", "-static-libgcc"]
    cmd += [str(source), "-o", str(out)]
    return cmd


def _source_stamp(sources: Sequence[Path], cmd: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(f"abi{FA_ABI_VERSION}\n".encode())
    for s in sources:
        h.update(s.read_bytes())
    # the command line is part of the identity: an AVX2 build and a baseline
    # build come from byte-identical sources
    h.update("\n".join(str(c) for c in cmd[:-1]).encode())
    return h.hexdigest()[:16]


def _run_build(cmd: Sequence[str], cwd: Path) -> None:
    if _verbose():
        print("frame_analytics: " + " ".join(str(c) for c in cmd), file=sys.stderr)
    res = subprocess.run([str(c) for c in cmd], cwd=str(cwd),
                         capture_output=not _verbose(), text=True)
    if res.returncode != 0:
        detail = ""
        if not _verbose():
            detail = "\n" + (res.stderr or res.stdout or "").strip()[-4000:]
        raise RuntimeError(
            f"build failed ({' '.join(str(c) for c in cmd[:1])} exited "
            f"{res.returncode}); set FA_VERBOSE=1 for the full output{detail}"
        )


def _build_cached(stem: str, sources: Sequence[Path],
                  make_cmd) -> Path:
    """Compile ``sources`` into ``build_root()/<stem>`` unless already current.

    The stamp file holds a hash of the sources *and* the command line, so an
    edited kernel or a changed flag rebuilds and an unchanged one does not.
    """
    root = build_root() / stem
    root.mkdir(parents=True, exist_ok=True)
    out = root / library_filename(stem)
    cmd = make_cmd(out, root)
    stamp_want = _source_stamp(sources, cmd)
    stamp_file = root / "stamp"
    if out.exists() and stamp_file.exists():
        try:
            if stamp_file.read_text().strip() == stamp_want:
                return out
        except OSError:
            pass

    # build to a unique name and rename, so two interpreters racing on the same
    # cache cannot hand each other a half-written library
    tmp = root / f".{os.getpid()}{library_filename(stem)}"
    tmp_cmd = make_cmd(tmp, root)
    try:
        _run_build(tmp_cmd, root)
        os.replace(tmp, out)
        stamp_file.write_text(stamp_want)
    finally:
        for junk in root.glob(f".{os.getpid()}*"):
            try:
                junk.unlink()
            except OSError:
                pass
    return out


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _dlopen(path: Path) -> ctypes.CDLL:
    if sys.platform == "win32":
        # the libraries import nothing but the platform C runtime, so the
        # default search path is enough; winmode=0 keeps it that way
        return ctypes.CDLL(str(path), winmode=0)
    return ctypes.CDLL(str(path))


def _prebuilt(stem: str) -> Optional[Path]:
    if os.environ.get("FA_FORCE_JIT", "0") == "1":
        return None
    p = _LIBDIR / library_filename(stem)
    return p if p.exists() else None


_P = ctypes.c_void_p
_I = ctypes.c_int
_L = ctypes.c_int64
_D = ctypes.c_double


def _bind_cpu(lib: ctypes.CDLL) -> ctypes.CDLL:
    sig = {
        "fa_cpu_abi_version": ([], _I),
        "fa_cpu_has_avx2": ([], _I),
        "fa_cpu_set_num_threads": ([_I], _I),
        "fa_cpu_num_threads": ([], _I),
        "fa_cpu_pixel_reduce": ([_P, _P, _I, _L, _L, _I, _D, _D, _I, _P], _I),
        "fa_cpu_ssim": ([_P, _P, _I, _I, _I, _I, _I, _P, _I, _D, _D, _D, _P, _P], _I),
        "fa_cpu_ssim_cs": ([_P, _P, _I, _I, _I, _I, _I, _P, _I, _D, _D, _D, _P, _P], _I),
        "fa_cpu_gmsd": ([_P, _P, _I, _I, _I, _I, _I, _D, _D, _I, _P, _P], _I),
    }
    for name, (args, res) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = res
    return lib


def _bind_cuda(lib: ctypes.CDLL) -> ctypes.CDLL:
    sig = {
        "fa_cuda_abi_version": ([], _I),
        "fa_cuda_error_string": ([_I], ctypes.c_char_p),
        "fa_cuda_device_count": ([_P], _I),
        "fa_cuda_pixel_workspace": ([_L, _L, _I, _P], _I),
        "fa_cuda_pixel_reduce":
            ([_P, _P, _I, _L, _L, _I, _D, _D, _P, _I, _P, _P, _P, _P, _I, _P], _I),
        "fa_cuda_ssim_workspace": ([_I, _I, _I, _P], _I),
        "fa_cuda_ssim":
            ([_P, _P, _I, _I, _I, _I, _I, _P, _I, _D, _D, _D, _P, _P, _I, _P, _P,
              _I, _P], _I),
        "fa_cuda_ssim_cs":
            ([_P, _P, _I, _I, _I, _I, _I, _P, _I, _D, _D, _D, _P, _P, _I, _P, _P,
              _I, _P], _I),
        "fa_cuda_ssim_backward":
            ([_P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _D, _D, _D, _P, _P, _P,
              _I, _P], _I),
        "fa_cuda_ssim_cs_backward":
            ([_P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _D, _D, _D, _P, _P, _P,
              _I, _P], _I),
        "fa_cuda_gmsd_workspace": ([_I, _I, _I, _P], _I),
        "fa_cuda_gmsd":
            ([_P, _P, _I, _I, _I, _I, _I, _D, _D, _I, _P, _P, _I, _P, _P, _I,
              _P], _I),
    }
    for name, (args, res) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = res
    return lib


def _check_abi(lib: ctypes.CDLL, getter: str, origin: Path) -> None:
    got = getattr(lib, getter)()
    if got != FA_ABI_VERSION:
        raise RuntimeError(
            f"{origin.name} reports ABI version {got}, this build expects "
            f"{FA_ABI_VERSION}; delete it and let it rebuild"
        )


def load_cpu():
    """Return ``(lib, origin, isa)`` for the CPU kernels."""
    baseline_path = _prebuilt("fa_cpu")
    origin = "prebuilt"
    if baseline_path is None:
        ensure_host_compiler()
        baseline_path = _build_cached(
            "fa_cpu", [_CSRC / "fa_cpu.cpp", _CSRC / "fa_abi.h"],
            lambda out, objdir: cpu_compile_command(_CSRC / "fa_cpu.cpp", out,
                                                    False, objdir),
        )
        origin = "jit"
    lib = _bind_cpu(_dlopen(baseline_path))
    _check_abi(lib, "fa_cpu_abi_version", baseline_path)
    isa = "baseline"

    if _is_x86() and lib.fa_cpu_has_avx2():
        try:
            avx_path = _prebuilt("fa_cpu_avx2")
            avx_origin = "prebuilt"
            if avx_path is None:
                ensure_host_compiler()
                avx_path = _build_cached(
                    "fa_cpu_avx2", [_CSRC / "fa_cpu.cpp", _CSRC / "fa_abi.h"],
                    lambda out, objdir: cpu_compile_command(
                        _CSRC / "fa_cpu.cpp", out, True, objdir),
                )
                avx_origin = "jit"
            avx = _bind_cpu(_dlopen(avx_path))
            _check_abi(avx, "fa_cpu_abi_version", avx_path)
            lib, origin, isa = avx, avx_origin, "avx2"
        except Exception:
            # the baseline build is already loaded and correct; a missing or
            # unbuildable AVX2 sibling costs speed, never results
            if _verbose():
                import traceback

                traceback.print_exc()
    return lib, origin, isa


def load_cuda():
    """Return ``(lib, origin)`` for the CUDA kernels."""
    if sys.platform == "darwin":
        raise RuntimeError("no CUDA on macOS")
    path = _prebuilt("fa_cuda")
    origin = "prebuilt"
    if path is None:
        ensure_host_compiler()
        path = _build_cached(
            "fa_cuda", [_CSRC / "fa_cuda.cu", _CSRC / "fa_abi.h"],
            lambda out, objdir: cuda_compile_command(_CSRC / "fa_cuda.cu", out),
        )
        origin = "jit"
    lib = _bind_cuda(_dlopen(path))
    _check_abi(lib, "fa_cuda_abi_version", path)
    return lib, origin
