"""Ground-truth reference implementations.

Deliberately slow, deliberately obvious. These follow the source papers
literally and run in float64. Every fast kernel in this package is validated
against these.

References
----------
Wang, Bovik, Sheikh, Simoncelli, "Image Quality Assessment: From Error
Visibility to Structural Similarity", IEEE TIP 13(4), 2004.
Canonical implementation: ``ssim_index.m`` (Wang's original MATLAB release).

  window   = fspecial('gaussian', 11, 1.5)   (normalised to sum 1)
  C1       = (K1 * L)^2,  K1 = 0.01
  C2       = (K2 * L)^2,  K2 = 0.03
  mu_x     = filter2(window, x, 'valid')
  sigma_x2 = filter2(window, x.*x, 'valid') - mu_x.^2
  sigma_xy = filter2(window, x.*y, 'valid') - mu_x.*mu_y
  map      = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) /
             ((mu_x^2 + mu_y^2 + C1) * (sigma_x2 + sigma_y2 + C2))
  mssim    = mean(map)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gaussian_window_2d",
    "mse_reference",
    "psnr_reference",
    "ssim_reference",
]


def gaussian_window_2d(win_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """``fspecial('gaussian', win_size, sigma)`` in float64, sum-normalised."""
    r = (win_size - 1) / 2.0
    coords = np.arange(win_size, dtype=np.float64) - r
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    h = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    # MATLAB's fspecial zeroes entries below eps*max before normalising.
    h[h < np.finfo(np.float64).eps * h.max()] = 0.0
    s = h.sum()
    if s != 0:
        h /= s
    return h


def _valid_correlate2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """'valid' correlation, the operation MATLAB's ``filter2`` performs."""
    kh, kw = kernel.shape
    h, w = img.shape
    oh, ow = h - kh + 1, w - kw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError(f"image {img.shape} smaller than window {kernel.shape}")
    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            k = kernel[i, j]
            if k != 0.0:
                out += k * img[i : i + oh, j : j + ow]
    return out


def mse_reference(x: np.ndarray, y: np.ndarray) -> float:
    d = x.astype(np.float64) - y.astype(np.float64)
    return float(np.mean(d * d))


def psnr_reference(x: np.ndarray, y: np.ndarray, data_range: float = 255.0) -> float:
    m = mse_reference(x, y)
    if m == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range * data_range) / m))


def ssim_reference(
    x: np.ndarray,
    y: np.ndarray,
    data_range: float = 255.0,
    win_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    return_map: bool = False,
):
    """Single-channel SSIM, exactly as ``ssim_index.m`` computes it."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    w = gaussian_window_2d(win_size, sigma)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_x = _valid_correlate2d(x, w)
    mu_y = _valid_correlate2d(y, w)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_xx = _valid_correlate2d(x * x, w) - mu_xx
    sigma_yy = _valid_correlate2d(y * y, w) - mu_yy
    sigma_xy = _valid_correlate2d(x * y, w) - mu_xy

    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2)
    ssim_map = num / den

    if return_map:
        return float(ssim_map.mean()), ssim_map
    return float(ssim_map.mean())
