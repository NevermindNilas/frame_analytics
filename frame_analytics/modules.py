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

from .functional import _infer_data_range, mse, psnr, ssim

__all__ = ["MSE", "PSNR", "SSIM", "StreamingMetrics"]


class MSE(nn.Module):
    def __init__(self, reduction: str = "mean", dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.reduction = reduction
        self.dtype_ = dtype

    def forward(self, x, y):
        return mse(x, y, reduction=self.reduction, dtype=self.dtype_)


class PSNR(nn.Module):
    def __init__(self, data_range: Optional[float] = None, reduction: str = "mean",
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction
        self.dtype_ = dtype

    def forward(self, x, y):
        return psnr(x, y, data_range=self.data_range, reduction=self.reduction,
                    dtype=self.dtype_)


class SSIM(nn.Module):
    """SSIM as a module. Differentiable through the portable PyTorch path.

    Note the native kernels are inference-only; when ``x.requires_grad`` the
    dispatcher automatically falls back to the autograd-capable path.
    """

    def __init__(self, data_range: Optional[float] = None, win_size: int = 11,
                 sigma: float = 1.5, K: Sequence[float] = (0.01, 0.03),
                 reduction: str = "mean", return_map: bool = False,
                 dtype: Optional[torch.dtype] = None, downsample: bool = False):
        super().__init__()
        self.data_range = data_range
        self.win_size = win_size
        self.sigma = sigma
        self.K = tuple(K)
        self.reduction = reduction
        self.return_map = return_map
        self.dtype_ = dtype
        self.downsample = downsample

    def forward(self, x, y):
        return ssim(x, y, data_range=self.data_range, win_size=self.win_size,
                    sigma=self.sigma, K=self.K, reduction=self.reduction,
                    return_map=self.return_map, dtype=self.dtype_,
                    downsample=self.downsample)

    def loss(self, x, y):
        """``1 - SSIM``, for use as a training objective."""
        return 1.0 - self.forward(x, y)


class StreamingMetrics:
    """Fixed-shape video scorer with CUDA-graph replay and pinned staging.

    Intended for the "score every frame of a stream" workload, where per-call
    CPU overhead, not GPU work, is usually the bottleneck.

    >>> sm = StreamingMetrics((1, 3, 1080, 1920), device="cuda", dtype=torch.uint8)
    >>> for ref, dist in frames:            # numpy / cpu tensors
    ...     out = sm.update(ref, dist)      # {"mse":..., "psnr":..., "ssim":...}
    """

    def __init__(self, shape, device="cuda", dtype=torch.uint8,
                 data_range: Optional[float] = None, metrics=("mse", "psnr", "ssim"),
                 use_cuda_graph: bool = True, pinned_staging: bool = True):
        self.device = torch.device(device)
        self.dtype = dtype
        self.shape = tuple(shape)
        self.metrics = tuple(metrics)
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
