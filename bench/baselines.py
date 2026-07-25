"""Reference implementations to benchmark against.

``naive_ssim_torch`` is what most PyTorch SSIM libraries actually do -- a 2-D
11x11 depthwise convolution per moment, five of them, then a long elementwise
epilogue. It is here so the speedup number is attributable to specific choices
(separability, plane packing, fusion) rather than to a strawman.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gaussian2d(win_size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    r = (win_size - 1) / 2.0
    c = torch.arange(win_size, dtype=torch.float64) - r
    g = torch.exp(-(c * c) / (2 * sigma * sigma))
    g = g / g.sum()
    return torch.outer(g, g).to(device=device, dtype=dtype)


def naive_ssim_torch(x, y, data_range=255.0, win_size=11, sigma=1.5):
    """Five separate 2-D convolutions -- the common library formulation."""
    x = x.float()
    y = y.float()
    C = x.shape[1]
    w = gaussian2d(win_size, sigma, x.device, torch.float32).expand(C, 1, win_size, win_size)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, w, groups=C)
    mu_y = F.conv2d(y, w, groups=C)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sxx = F.conv2d(x * x, w, groups=C) - mu_xx
    syy = F.conv2d(y * y, w, groups=C) - mu_yy
    sxy = F.conv2d(x * y, w, groups=C) - mu_xy
    num = (2 * mu_xy + C1) * (2 * sxy + C2)
    den = (mu_xx + mu_yy + C1) * (sxx + syy + C2)
    return (num / den).mean()


def naive_mse_torch(x, y):
    d = x.float() - y.float()
    return (d * d).mean()


def numpy_mse(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def opencv_ssim(a, b, data_range=255.0):
    """The SSIM recipe from the OpenCV docs (Gaussian 11x11, sigma 1.5).

    OpenCV has no SSIM in the main API; this is the canonical snippet everyone
    copies out of the 'Video Input with OpenCV and similarity measurement'
    tutorial, kept faithful apart from using 'valid' borders so the support
    matches the paper.
    """
    import cv2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    I1 = a.astype(np.float32)
    I2 = b.astype(np.float32)
    I1_2, I2_2, I1_I2 = I1 * I1, I2 * I2, I1 * I2
    kw = dict(ksize=(11, 11), sigmaX=1.5, borderType=cv2.BORDER_REPLICATE)
    mu1 = cv2.GaussianBlur(I1, **kw)
    mu2 = cv2.GaussianBlur(I2, **kw)
    mu1_2, mu2_2, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = cv2.GaussianBlur(I1_2, **kw) - mu1_2
    s2 = cv2.GaussianBlur(I2_2, **kw) - mu2_2
    s12 = cv2.GaussianBlur(I1_I2, **kw) - mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * s12 + C2
    t3 = t1 * t2
    t1 = mu1_2 + mu2_2 + C1
    t2 = s1 + s2 + C2
    ssim_map = t3 / (t1 * t2)
    return float(ssim_map[5:-5, 5:-5].mean())
