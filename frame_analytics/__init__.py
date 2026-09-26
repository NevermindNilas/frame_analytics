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

``frame_analytics.vapoursynth``
    VapourSynth filter with ``VapourSynth-VMAF``'s ``vmaf.Metric`` API. Imported
    on request, since it needs VapourSynth:
    ``from frame_analytics import vapoursynth as fa_vs``.
"""

try:
    import torch  # noqa: F401
except ModuleNotFoundError as e:  # pragma: no cover - depends on environment
    raise ModuleNotFoundError(
        "frame_analytics requires PyTorch, which is intentionally not installed "
        "automatically so an existing (e.g. CUDA) build is never overwritten by "
        "the CPU wheel from PyPI. Install the build that fits your machine from "
        "https://pytorch.org/get-started/locally/, or opt in to the PyPI wheel "
        "with `pip install frame-analytics[torch]`."
    ) from e

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
    ADM_LIKE,
    CIEDE2000,
    DSS,
    FLIP,
    FSIM,
    GMSD,
    HAARPSI,
    IWSSIM,
    LPIPS,
    MDSI,
    MS_GMSD,
    MSE,
    NLPD,
    PSNR,
    PSNR_HVS,
    PSNR_HVS_M,
    SCIELAB,
    SRSIM,
    SSIM,
    SSIMULACRA2,
    VIFP,
    VSI,
    Charbonnier,
    Huber,
    L1,
    MSSSIM,
    StreamingMetrics,
)
from .perceptual import available_nets, lpips, lpips_weights_path  # noqa: F401
from .ssimulacra2 import ssimulacra2  # noqa: F401
from .adm import adm_like  # noqa: F401
from .ciede2000 import ciede2000, delta_e_00, srgb_to_lab  # noqa: F401
from .dss import dss  # noqa: F401
from .flip import flip  # noqa: F401
from .fsim import fsim, fsimc  # noqa: F401
from .haarpsi import haarpsi  # noqa: F401
from .iwssim import iw_ssim, iwssim_lite  # noqa: F401
from .mdsi import gcs_map, mdsi  # noqa: F401
from .ms_gmsd import MS_GMSD_WEIGHTS, ms_gmsd  # noqa: F401
from .nlpd import effective_depth, nlpd  # noqa: F401
from .psnrhvs import mse_hvs, mse_hvs_m, psnr_hvs, psnr_hvs_m  # noqa: F401
from .scielab import scielab  # noqa: F401
from .srsim import srsim, srsimc  # noqa: F401
from .vifp import vifp  # noqa: F401
from .vsi import sdsp_saliency, vsi  # noqa: F401

__version__ = "0.7.0"

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
    "ssimulacra2",
    "ms_gmsd",
    "MS_GMSD_WEIGHTS",
    "vifp",
    "iw_ssim",
    "iwssim_lite",
    "dss",
    "nlpd",
    "effective_depth",
    "fsim",
    "fsimc",
    "srsim",
    "srsimc",
    "vsi",
    "sdsp_saliency",
    "mdsi",
    "gcs_map",
    "haarpsi",
    "adm_like",
    "psnr_hvs",
    "psnr_hvs_m",
    "mse_hvs",
    "mse_hvs_m",
    "ciede2000",
    "srgb_to_lab",
    "delta_e_00",
    "flip",
    "scielab",
    "rgb_to_luma",
    "MSE",
    "PSNR",
    "SSIM",
    "MSSSIM",
    "GMSD",
    "MS_GMSD",
    "VIFP",
    "IWSSIM",
    "DSS",
    "NLPD",
    "FSIM",
    "SRSIM",
    "VSI",
    "MDSI",
    "HAARPSI",
    "ADM_LIKE",
    "PSNR_HVS",
    "PSNR_HVS_M",
    "CIEDE2000",
    "FLIP",
    "SCIELAB",
    "L1",
    "Charbonnier",
    "Huber",
    "LPIPS",
    "SSIMULACRA2",
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
