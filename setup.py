"""Build hook.

By default this produces a pure-Python (`py3-none-any`) wheel: the CUDA and C++
sources ship *inside* the wheel and are compiled on first use by
`frame_analytics.backend`, which caches the result in the torch extensions
directory.

That is a deliberate choice, not a shortcut. A torch C++ extension is ABI-locked
to the exact torch build it was compiled against -- one prebuilt wheel per
{python version} x {torch version} x {CUDA version} x {platform} is a
combinatorial matrix that would still miss whatever the user actually has
installed. Compiling on demand means one wheel works everywhere, and because
every native kernel has a portable PyTorch fallback, it also works on machines
with no compiler at all.

Set FA_BUILD_EXT=1 to compile ahead of time instead and get a platform wheel:

    FA_BUILD_EXT=1 pip install .
"""

import os
import sys
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).parent
CSRC = ROOT / "frame_analytics" / "csrc"

ext_modules = []
cmdclass = {}

if os.environ.get("FA_BUILD_EXT", "").lower() in ("1", "on", "true", "yes"):
    # torch is only imported on this path, so the default build works in an
    # isolated environment that has never heard of it
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    with_cuda = torch.cuda.is_available() or bool(os.environ.get("FORCE_CUDA"))

    if sys.platform == "win32":
        # /Zc:preprocessor: CUDA 13's CCCL headers refuse to build against
        # MSVC's traditional preprocessor
        cxx_flags = ["/O2", "/fp:fast", "/arch:AVX2", "/Zc:preprocessor"]
    else:
        cxx_flags = ["-O3", "-ffast-math", "-march=native"]

    nvcc_flags = ["-O3", "--expt-relaxed-constexpr",
                  "-prec-div=true", "-prec-sqrt=true", "-ftz=false"]
    if sys.platform == "win32":
        nvcc_flags += ["-Xcompiler", "/Zc:preprocessor"]

    if with_cuda:
        cxx_flags.append("/DWITH_CUDA" if sys.platform == "win32" else "-DWITH_CUDA")
        nvcc_flags.append("-DWITH_CUDA")
        ext_modules = [CUDAExtension(
            "frame_analytics_native",
            [str(CSRC / "fa_cpu.cpp"), str(CSRC / "fa_cuda.cu")],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )]
    else:
        ext_modules = [CppExtension(
            "frame_analytics_native",
            [str(CSRC / "fa_cpu.cpp")],
            extra_compile_args=cxx_flags,
        )]
    cmdclass = {"build_ext": BuildExtension}

# everything else (name, version, deps, package data) lives in pyproject.toml
setup(ext_modules=ext_modules, cmdclass=cmdclass)
