"""Stateful wrappers.

The functional API is stateless and re-derives the Gaussian window, the
constants and (under ``torch.compile``) the guard checks on every call.  That
is irrelevant for a 4K frame and very relevant for a 256x256 patch in a
training loop, where launch overhead dominates.

``SSIM``/``MSE``/``PSNR`` cache what they can.  ``StreamingMetrics`` goes
further: for a fixed frame shape it captures the whole computation into a CUDA
graph, so scoring a frame is one graph replay -- a handful of microseconds of
CPU time regardless of how many kernels are involved.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .functional import (
    MS_SSIM_WEIGHTS,
    _infer_data_range,
    charbonnier,
    gms,
    gmsd,
    huber,
    l1,
    ms_ssim,
    mse,
    psnr,
    ssim,
)
from .perceptual import lpips
from .ssimulacra2 import ssimulacra2

__all__ = ["MSE", "PSNR", "SSIM", "MSSSIM", "GMSD", "L1", "Charbonnier",
           "Huber", "LPIPS", "SSIMULACRA2", "StreamingMetrics"]


class MSE(nn.Module):
    def __init__(self, reduction: str = "mean", dtype: Optional[torch.dtype] = None,
                 out_dtype: torch.dtype = torch.float64, luma=None,
                 crop_border: int = 0):
        super().__init__()
        self.reduction = reduction
        self.dtype_ = dtype
        self.out_dtype = out_dtype
        self.luma = luma
        self.crop_border = crop_border

    def forward(self, x, y):
        return mse(x, y, reduction=self.reduction, dtype=self.dtype_,
                   out_dtype=self.out_dtype, luma=self.luma,
                   crop_border=self.crop_border)


class PSNR(nn.Module):
    """``PSNR(luma="matlab", crop_border=scale)`` is the Y-PSNR of the
    super-resolution literature."""

    def __init__(self, data_range: Optional[float] = None, reduction: str = "mean",
                 dtype: Optional[torch.dtype] = None,
                 out_dtype: torch.dtype = torch.float64, eps: float = 0.0,
                 luma=None, crop_border: int = 0):
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction
        self.dtype_ = dtype
        self.out_dtype = out_dtype
        self.eps = eps
        self.luma = luma
        self.crop_border = crop_border

    def forward(self, x, y):
        return psnr(x, y, data_range=self.data_range, reduction=self.reduction,
                    dtype=self.dtype_, out_dtype=self.out_dtype, eps=self.eps,
                    luma=self.luma, crop_border=self.crop_border)


class SSIM(nn.Module):
    """SSIM as a module. Differentiable through the portable PyTorch path.

    Note the native kernels are inference-only; when ``x.requires_grad`` the
    dispatcher automatically falls back to the autograd-capable path.
    """

    def __init__(self, data_range: Optional[float] = None, win_size: int = 11,
                 sigma: float = 1.5, K: Sequence[float] = (0.01, 0.03),
                 reduction: str = "mean", return_map: bool = False,
                 dtype: Optional[torch.dtype] = None, downsample: bool = False,
                 luma=None, crop_border: int = 0):
        super().__init__()
        self.data_range = data_range
        self.win_size = win_size
        self.sigma = sigma
        self.K = tuple(K)
        self.reduction = reduction
        self.return_map = return_map
        self.dtype_ = dtype
        self.downsample = downsample
        self.luma = luma
        self.crop_border = crop_border

    def forward(self, x, y):
        return ssim(x, y, data_range=self.data_range, win_size=self.win_size,
                    sigma=self.sigma, K=self.K, reduction=self.reduction,
                    return_map=self.return_map, dtype=self.dtype_,
                    downsample=self.downsample, luma=self.luma,
                    crop_border=self.crop_border)

    def loss(self, x, y):
        """``1 - SSIM``, for use as a training objective."""
        return 1.0 - self.forward(x, y)


class MSSSIM(nn.Module):
    """Multi-scale SSIM. ``.loss()`` gives ``1 - MS-SSIM``.

    The usual restoration objective is a mix, e.g.
    ``0.84 * MSSSIM().loss(p, t) + 0.16 * L1()(p, t)`` (Zhao et al. 2016).
    """

    def __init__(self, data_range: Optional[float] = None, win_size: int = 11,
                 sigma: float = 1.5, K: Sequence[float] = (0.01, 0.03),
                 weights: Optional[Sequence[float]] = None,
                 reduction: str = "mean", dtype: Optional[torch.dtype] = None,
                 luma=None, crop_border: int = 0):
        super().__init__()
        self.data_range = data_range
        self.win_size = win_size
        self.sigma = sigma
        self.K = tuple(K)
        self.weights = tuple(weights) if weights is not None else MS_SSIM_WEIGHTS
        self.reduction = reduction
        self.dtype_ = dtype
        self.luma = luma
        self.crop_border = crop_border

    def forward(self, x, y):
        return ms_ssim(x, y, data_range=self.data_range, win_size=self.win_size,
                       sigma=self.sigma, K=self.K, weights=self.weights,
                       reduction=self.reduction, dtype=self.dtype_,
                       luma=self.luma, crop_border=self.crop_border)

    def loss(self, x, y):
        return 1.0 - self.forward(x, y)


class GMSD(nn.Module):
    """Gradient magnitude similarity. ``forward`` is the deviation (the metric).

    ``.loss()`` is ``1 - mean(GMS)`` rather than the deviation itself: the
    deviation's derivative blows up as the two images converge, which is
    exactly where a training run lives.
    """

    def __init__(self, data_range: Optional[float] = None,
                 T: Optional[float] = None, eps: Optional[float] = None,
                 downsample: bool = True, reduction: str = "mean",
                 dtype: Optional[torch.dtype] = None, luma=None,
                 crop_border: int = 0):
        super().__init__()
        self.data_range = data_range
        self.T = T
        self.eps = eps
        self.downsample = downsample
        self.reduction = reduction
        self.dtype_ = dtype
        self.luma = luma
        self.crop_border = crop_border

    def _kwargs(self):
        return dict(data_range=self.data_range, T=self.T, eps=self.eps,
                    downsample=self.downsample, reduction=self.reduction,
                    dtype=self.dtype_, luma=self.luma,
                    crop_border=self.crop_border)

    def forward(self, x, y):
        return gmsd(x, y, **self._kwargs())

    def loss(self, x, y):
        return 1.0 - gms(x, y, **self._kwargs())


class _PixelLoss(nn.Module):
    """Shared plumbing for the three pointwise losses.

    ``forward`` and ``loss`` are the same thing here -- unlike SSIM, these are
    already penalties -- but both names exist so the modules are drop-in for
    each other in a training script.
    """

    _fn = None

    def __init__(self, reduction: str = "mean",
                 dtype: Optional[torch.dtype] = None,
                 out_dtype: torch.dtype = torch.float64, luma=None,
                 crop_border: int = 0):
        super().__init__()
        self.reduction = reduction
        self.dtype_ = dtype
        self.out_dtype = out_dtype
        self.luma = luma
        self.crop_border = crop_border
        self.extra = {}

    def forward(self, x, y):
        return type(self)._fn(x, y, reduction=self.reduction, dtype=self.dtype_,
                              out_dtype=self.out_dtype, luma=self.luma,
                              crop_border=self.crop_border, **self.extra)

    def loss(self, x, y):
        return self.forward(x, y)


class L1(_PixelLoss):
    """Mean absolute error."""

    _fn = staticmethod(l1)


class Charbonnier(_PixelLoss):
    """Charbonnier penalty; ``eps`` is the width of the quadratic region."""

    _fn = staticmethod(charbonnier)

    def __init__(self, eps: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.extra = {"eps": float(eps)}


class Huber(_PixelLoss):
    """Huber loss, matching ``torch.nn.HuberLoss``."""

    _fn = staticmethod(huber)

    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.extra = {"delta": float(delta)}


class LPIPS(nn.Module):
    """Learned perceptual similarity (Zhang et al. 2018). Lower is better.

    ``forward`` and ``.loss()`` are the same thing -- LPIPS is already a
    penalty -- but both names exist so this is drop-in for the other modules.

    The usual reason to reach for it: PSNR, SSIM and MS-SSIM all reward blur,
    so a model trained on them alone converges to something smooth. A small
    LPIPS term is what removes that incentive.

    >>> perc = fa.LPIPS(data_range=1.0)
    >>> loss = fa.L1()(pred, target) + 0.1 * perc.loss(pred, target)

    The 2.5M trunk weights are buffers, not parameters, so ``.parameters()``
    is empty: this module cannot leak frozen ImageNet weights into the
    caller's optimiser or checkpoint. They are also shared process-wide per
    (trunk, device, dtype), so constructing several of these costs nothing
    beyond the first.
    """

    def __init__(self, net: str = "alex", data_range: Optional[float] = None,
                 reduction: str = "mean", return_map: bool = False,
                 allow_tf32: bool = False, dtype: Optional[torch.dtype] = None,
                 luma=None, crop_border: int = 0):
        super().__init__()
        self.net = net
        self.data_range = data_range
        self.reduction = reduction
        self.return_map = return_map
        self.allow_tf32 = allow_tf32
        self.dtype_ = dtype
        self.luma = luma
        self.crop_border = crop_border

    def forward(self, x, y):
        return lpips(x, y, net=self.net, data_range=self.data_range,
                     reduction=self.reduction, return_map=self.return_map,
                     allow_tf32=self.allow_tf32, dtype=self.dtype_,
                     luma=self.luma, crop_border=self.crop_border)

    def loss(self, x, y):
        return self.forward(x, y)


class SSIMULACRA2(nn.Module):
    """SSIMULACRA 2 (Sneyers 2022/2023). Higher is better; 100 is identical.

    ``forward(orig, dist)`` -- the argument order matters here, unlike SSIM:
    the metric distinguishes an added edge from a lost one, so it is not
    symmetric.

    Inputs are **sRGB-encoded** RGB; the metric linearises them itself. There
    is no ``.loss()``: the final ``100 - 10*s^0.628`` has an unbounded
    derivative as the pair converges, so it is a reporting metric rather than
    an objective.
    """

    def __init__(self, data_range: Optional[float] = None,
                 reduction: str = "mean", dtype: Optional[torch.dtype] = None,
                 crop_border: int = 0):
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction
        self.dtype_ = dtype
        self.crop_border = crop_border

    def forward(self, x, y):
        return ssimulacra2(x, y, data_range=self.data_range,
                           reduction=self.reduction, dtype=self.dtype_,
                           crop_border=self.crop_border)


class StreamingMetrics:
    """Fixed-shape video scorer with CUDA-graph replay and pinned staging.

    Intended for the "score every frame of a stream" workload, where per-call
    CPU overhead, not GPU work, is usually the bottleneck.

    >>> sm = StreamingMetrics((1, 3, 1080, 1920), device="cuda", dtype=torch.uint8)
    >>> for ref, dist in frames:            # numpy / cpu tensors
    ...     out = sm.update(ref, dist)      # {"mse":..., "psnr":..., "ssim":...}

    ``metrics`` may name any of ``mse``, ``psnr``, ``ssim``, ``ms_ssim``,
    ``gmsd``, ``gms``, ``l1``, ``charbonnier``, ``huber``, ``lpips``. They all
    capture into the same graph, so scoring a frame on ten metrics is still one
    replay.

    ``lpips`` is capturable because the trunk weights are loaded and moved to
    the device during the pre-capture warm-up, so the captured region contains
    convolutions and nothing else -- no allocation, no host->device copy.
    """

    _KNOWN = ("mse", "psnr", "ssim", "ms_ssim", "gmsd", "gms", "l1",
              "charbonnier", "huber", "lpips", "ssimulacra2")

    def __init__(self, shape, device="cuda", dtype=torch.uint8,
                 data_range: Optional[float] = None, metrics=("mse", "psnr", "ssim"),
                 use_cuda_graph: bool = True, pinned_staging: bool = True):
        self.device = torch.device(device)
        self.dtype = dtype
        self.shape = tuple(shape)
        self.metrics = tuple(metrics)
        unknown = [m for m in self.metrics if m not in self._KNOWN]
        if unknown:
            raise ValueError(f"unknown metric(s) {unknown}; "
                             f"expected any of {list(self._KNOWN)}")
        self.data_range = data_range

        self.ref = torch.empty(self.shape, dtype=dtype, device=self.device)
        self.dist = torch.empty(self.shape, dtype=dtype, device=self.device)
        self._L = float(data_range) if data_range is not None else _infer_data_range(self.ref)

        self._host = None
        if pinned_staging and self.device.type == "cuda":
            try:
                self._host = (
                    torch.empty(self.shape, dtype=dtype, pin_memory=True),
                    torch.empty(self.shape, dtype=dtype, pin_memory=True),
                )
            except RuntimeError:
                self._host = None

        self._graph = None
        self._static_out = None
        if use_cuda_graph and self.device.type == "cuda":
            self._try_capture()

    # -- internals ---------------------------------------------------------- #

    def _compute(self):
        out = {}
        if "mse" in self.metrics or "psnr" in self.metrics:
            m = mse(self.ref, self.dist)
            if "mse" in self.metrics:
                out["mse"] = m
            if "psnr" in self.metrics:
                # constant folded on the host: materialising L^2 as a device
                # tensor would be an H2D copy, which graph capture rejects
                out["psnr"] = math.log10(self._L ** 2) * 10.0 - 10.0 * torch.log10(m)
        if "ssim" in self.metrics:
            out["ssim"] = ssim(self.ref, self.dist, data_range=self._L)
        if "ms_ssim" in self.metrics:
            out["ms_ssim"] = ms_ssim(self.ref, self.dist, data_range=self._L)
        if "gmsd" in self.metrics:
            out["gmsd"] = gmsd(self.ref, self.dist, data_range=self._L)
        if "gms" in self.metrics:
            out["gms"] = gms(self.ref, self.dist, data_range=self._L)
        if "lpips" in self.metrics:
            out["lpips"] = lpips(self.ref, self.dist, data_range=self._L)
        if "ssimulacra2" in self.metrics:
            # ref is the original, dist the distorted -- the one metric here
            # that is not symmetric in its two arguments.
            out["ssimulacra2"] = ssimulacra2(self.ref, self.dist,
                                             data_range=self._L)
        for name, fn in (("l1", l1), ("charbonnier", charbonnier),
                         ("huber", huber)):
            if name in self.metrics:
                out[name] = fn(self.ref, self.dist)
        return out

    def _try_capture(self):
        try:
            self.ref.zero_()
            self.dist.zero_()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):          # warm up allocator + kernels
                    self._compute()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._static_out = self._compute()
            self._graph = g
        except Exception:
            self._graph = None
            self._static_out = None

    def _stage(self, buf: torch.Tensor, src, host_slot: int) -> None:
        if not torch.is_tensor(src):
            src = torch.from_numpy(src)
        src = src.reshape(self.shape) if src.shape != self.shape else src
        if src.is_cuda:
            buf.copy_(src, non_blocking=True)
            return
        if self._host is not None:
            h = self._host[host_slot]
            h.copy_(src)
            buf.copy_(h, non_blocking=True)
        else:
            buf.copy_(src)

    # -- API ---------------------------------------------------------------- #

    def update(self, ref, dist, *, as_float: bool = True):
        """Score one frame pair. Returns a dict of metric -> value."""
        self._stage(self.ref, ref, 0)
        self._stage(self.dist, dist, 1)
        if self._graph is not None:
            self._graph.replay()
            out = self._static_out
        else:
            out = self._compute()
        if as_float:
            return {k: float(v) for k, v in out.items()}
        return {k: v.clone() for k, v in out.items()}

    @property
    def graph_captured(self) -> bool:
        return self._graph is not None
