"""frame_analytics -- fast MSE / PSNR / SSIM for PyTorch (CPU + CUDA).

>>> import torch, frame_analytics as fa
>>> a = torch.randint(0, 256, (8, 3, 1080, 1920), dtype=torch.uint8, device="cuda")
>>> b = (a.float() + torch.randn_like(a, dtype=torch.float32) * 4).clamp(0, 255).byte()
>>> fa.psnr(a, b), fa.ssim(a, b)

Three layers, all producing the same numbers:

``frame_analytics.reference``
    float64 transcription of Wang et al. 2004 / ``ssim_index.m``. Ground truth.
``frame_analytics.functional``
    Portable PyTorch. Separable window, five planes packed into one
    convolution, ``torch.compile``d epilogue.
``frame_analytics.backend``
    Optional C++/CUDA extension. Single fused kernel, zero intermediates.
    Used automatically when it builds; skipped silently when it does not.
"""

from .functional import (  # noqa: F401
    MS_SSIM_WEIGHTS,
    charbonnier,
    gaussian_window_1d,
    gms,
    gmsd,
    huber,
    l1,
    ms_ssim,
    mse,
    psnr,
    rgb_to_luma,
    set_compile_enabled,
    ssim,
)
from .modules import (  # noqa: F401
    GMSD,
    LPIPS,
    MSE,
    PSNR,
    SSIM,
    Charbonnier,
    Huber,
    L1,
    MSSSIM,
    StreamingMetrics,
)
from .perceptual import available_nets, lpips, lpips_weights_path  # noqa: F401

__version__ = "0.2.0"

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
    "lpips",
    "rgb_to_luma",
    "MSE",
    "PSNR",
    "SSIM",
    "MSSSIM",
    "GMSD",
    "L1",
    "Charbonnier",
    "Huber",
    "LPIPS",
    "StreamingMetrics",
    "available_nets",
    "lpips_weights_path",
    "gaussian_window_1d",
    "set_compile_enabled",
    "backend_status",
    "MS_SSIM_WEIGHTS",
]


def backend_status() -> dict:
    """Report whether the native extension loaded, and why not if it did not."""
    from . import backend

    return backend.status()
