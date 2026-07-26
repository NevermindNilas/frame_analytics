"""Build hook.

Two shapes of wheel come out of this file.

**Default** -- a pure-Python (`py3-none-any`) wheel. The kernel sources ship
inside it and are compiled on first use by `frame_analytics._cabi`, which
caches the result. A machine with no compiler falls back to the portable
PyTorch path and still gets the same numbers.

**`FA_BUILD_EXT=1`** -- a platform wheel (`py3-none-win_amd64` and friends)
with the kernels already compiled into `frame_analytics/lib/`. This is what CI
publishes.

    FA_BUILD_EXT=1 pip install .

The tag on that platform wheel is deliberately `py3-none-<platform>` and not
`cp313-cp313-<platform>`. The libraries talk the C ABI in `csrc/fa_abi.h`:
raw pointers, an int dtype code, a stream handle, int return codes. No
libtorch, no pybind11, no Python C API. A torch C++ extension would instead be
locked to one {python} x {torch} x {CUDA} combination, and a wheel filename has
no field in which to encode "torch 2.6" -- pip would select on Python and
platform alone and hand somebody an `undefined symbol` ImportError. That is
why flash-attn and xformers run their own index instead of publishing to PyPI,
and why this package does not have to.

Set `FA_CUDA_ARCHS` (e.g. `"7.5 8.0 8.6 9.0 9.0+PTX"`) to pick the CUDA
architectures; a `+PTX` entry keeps the binary working on GPUs newer than the
toolkit that built it. Omitting it targets the local GPU only, which is fine
for a local build and wrong for a released wheel. Set `FA_SKIP_CUDA=1` to build
a CPU-only platform wheel -- the only option on macOS.
"""

import importlib.util
import os
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).parent
CSRC = ROOT / "frame_analytics" / "csrc"
LIBDIR = ROOT / "frame_analytics" / "lib"


def _load_cabi():
    """Import `frame_analytics/_cabi.py` on its own.

    Importing it as `frame_analytics._cabi` would run the package `__init__`,
    which imports torch -- and the default build has to work in an isolated
    environment that has never heard of torch.
    """
    spec = importlib.util.spec_from_file_location(
        "_fa_cabi_build", ROOT / "frame_analytics" / "_cabi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "on", "true", "yes")


BUILD_EXT = _truthy("FA_BUILD_EXT")


def _build_libraries() -> None:
    cabi = _load_cabi()
    cabi.ensure_host_compiler()
    LIBDIR.mkdir(parents=True, exist_ok=True)

    targets = [("fa_cpu", False)]
    # The AVX2 build is a second copy of the same source, not a second source.
    # One /arch:AVX2 binary would fault on pre-Haswell hardware; one baseline
    # binary would give up the vector width the reduction loops exist for, so
    # both ship and the loader picks at runtime.
    #
    # Whether to build it is a question about the *target*, not the host: an
    # arm64 macOS runner cross-compiling a universal2 binary still owes the
    # x86_64 slice its AVX2 sibling.
    mac_archs = os.environ.get("FA_MACOS_ARCHS", "").split()
    if cabi._is_x86() or "x86_64" in mac_archs:
        targets.append(("fa_cpu_avx2", True))

    for stem, avx2 in targets:
        out = LIBDIR / cabi.library_filename(stem)
        cmd = cabi.cpu_compile_command(CSRC / "fa_cpu.cpp", out, avx2, LIBDIR)
        print(f"building {out.name}", flush=True)
        cabi._run_build(cmd, LIBDIR)

    if sys.platform == "darwin" or _truthy("FA_SKIP_CUDA"):
        return
    out = LIBDIR / cabi.library_filename("fa_cuda")
    archs = cabi.cuda_arch_flags()
    if not archs:
        raise SystemExit(
            "FA_BUILD_EXT=1 needs CUDA architectures: set FA_CUDA_ARCHS "
            "(e.g. '7.5 8.0 8.6 9.0 9.0+PTX') or FA_SKIP_CUDA=1"
        )
    cmd = cabi.cuda_compile_command(CSRC / "fa_cuda.cu", out, archs)
    print(f"building {out.name}", flush=True)
    cabi._run_build(cmd, LIBDIR)

    # cl leaves .obj/.lib/.exp next to the dll; nothing but the dll belongs in
    # the wheel
    for junk in list(LIBDIR.glob("*.obj")) + list(LIBDIR.glob("*.exp")) + \
            list(LIBDIR.glob("*.lib")):
        junk.unlink()


class build_py(_build_py):
    def run(self):
        if BUILD_EXT:
            _build_libraries()
        super().run()


cmdclass = {"build_py": build_py}
distclass = Distribution

if BUILD_EXT:
    class BinaryDistribution(Distribution):
        """There are no `ext_modules`, but there *are* binaries.

        Without this the wheel is still tagged correctly but every file lands
        under `.data/purelib/`, because setuptools decides purelib-vs-platlib
        from `has_ext_modules()` rather than from the wheel tag.
        """

        def has_ext_modules(self):
            return True

    distclass = BinaryDistribution

    try:
        from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError:  # setuptools < 70.1
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        """Platform-specific, but not Python-specific.

        `root_is_pure = False` alone would tag the wheel `cp313-cp313-<plat>`,
        which is exactly the over-constraint this whole design exists to avoid:
        the binaries have no Python ABI in them at all.
        """

        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            _python, _abi, plat = super().get_tag()
            # inside a manylinux container the honest answer is `linux_x86_64`,
            # which PyPI rejects; CI supplies the policy tag it built against
            return "py3", "none", os.environ.get("FA_PLAT_TAG", plat)

    cmdclass["bdist_wheel"] = bdist_wheel

# everything else (name, version, deps, package data) lives in pyproject.toml
setup(cmdclass=cmdclass, distclass=distclass)
