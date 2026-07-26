# Prebuilt kernels

Wheels ship the compiled C-ABI kernel libraries here:

| file | contents |
|---|---|
| `fa_cpu.{dll,so,dylib}` | CPU kernels, platform baseline ISA |
| `fa_cpu_avx2.{dll,so,dylib}` | the same kernels with AVX2 enabled (x86 only) |
| `fa_cuda.{dll,so}` | CUDA kernels (never built or shipped on macOS) |

Nothing in this directory is checked into git. `python setup.py bdist_wheel`
with `FA_BUILD_EXT=1`, or the release workflow, compiles into it; an install
from source with no binaries here compiles on first use into a cache directory
instead, and a machine with no compiler falls back to the portable PyTorch
path.

The libraries talk the C ABI in `../csrc/fa_abi.h` -- no libtorch, no pybind11,
no Python C API -- which is what makes one binary per platform valid for every
Python version and every torch version.
