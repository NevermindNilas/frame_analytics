"""Optional ahead-of-time build of the native extension.

    pip install -e .            # pure python, kernels JIT-compiled on first use
    python setup.py build_ext --inplace   # compile now instead

The package works without this: `frame_analytics.backend` JIT-compiles the same
sources on first call and falls back to the portable PyTorch path if no
compiler is available.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
CSRC = ROOT / "frame_analytics" / "csrc"

ext_modules = []
cmdclass = {}

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    with_cuda = torch.cuda.is_available() or bool(
        __import__("os").environ.get("FORCE_CUDA"))

    if with_cuda:
        ext_modules = [CUDAExtension(
            "frame_analytics_native",
            [str(CSRC / "fa_cpu.cpp"), str(CSRC / "fa_cuda.cu")],
            extra_compile_args={
                "cxx": ["/O2", "/fp:fast", "/arch:AVX2", "/Zc:preprocessor", "/DWITH_CUDA"],
                "nvcc": ["-O3", "-DWITH_CUDA", "--expt-relaxed-constexpr",
                         "-prec-div=true", "-prec-sqrt=true", "-ftz=false"],
            },
        )]
    else:
        ext_modules = [CppExtension(
            "frame_analytics_native",
            [str(CSRC / "fa_cpu.cpp")],
            extra_compile_args=["/O2", "/fp:fast", "/arch:AVX2", "/Zc:preprocessor"],
        )]
    cmdclass = {"build_ext": BuildExtension}
except ImportError:
    pass

setup(
    name="frame_analytics",
    version="0.1.0",
    description="Fast MSE / PSNR / SSIM for PyTorch on CPU and CUDA",
    packages=find_packages(include=["frame_analytics", "frame_analytics.*"]),
    package_data={"frame_analytics": ["csrc/*.cu", "csrc/*.cpp"]},
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "numpy"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
