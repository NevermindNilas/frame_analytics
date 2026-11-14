"""Private helpers shared by the metric modules (not public API).

``plane_conv``: convolve independent single-channel planes with a small,
constant filter bank, with a backward pass that is fast and FP32.

Why: PyTorch's default ``torch.backends.cudnn.allow_tf32`` is True, and for
single-plane batches (``(B, 1, H, W)`` with a ``(O, 1, kh, kw)`` bank)
cuDNN's heuristics picked TF32 tensor-core kernels for the backward
(``dgrad``) pass -- measured on an RTX 3090 for a 1x21 filter on two 1080p
planes: gradient error vs float64 3.3e-4 instead of 2.4e-7, and 13x slower
than PyTorch's native FP32 depthwise kernels. On CPU, oneDNN's backward for
these shapes ran ~9x slower than its forward. Global TF32 flags are not
touched (they are process-wide and racy across threads).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["plane_conv"]


class _Null:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NULL = _Null()


def _no_autocast(t: torch.Tensor):
    """Disable autocast around a convolution; free when it is already off."""
    if torch.is_autocast_enabled(t.device.type):
        return torch.autocast(device_type=t.device.type, enabled=False)
    return _NULL


class _PlaneCorr(torch.autograd.Function):
    """Cross-correlation by a constant bank; ``dL/dx`` via a forward correlation.

    The input gradient of ``y = corr(pad_p(x), w)`` is the full correlation
    of the output gradient with the spatially flipped bank:
    ``corr(pad_{k-1-p}(g), flip(w)^T)``. That is a *forward* convolution
    (fast oneDNN / cuDNN forward kernels), written with differentiable ops so
    double backward still works. Two layouts: ``groups == 1`` with one input
    plane (``w`` ``(O, 1, kh, kw)``), or depthwise ``groups == C`` with one
    filter per channel (``w`` ``(C, 1, kh, kw)``).
    """

    @staticmethod
    def forward(ctx, x, weight, padding, groups):
        ctx.save_for_backward(weight)
        ctx.padding = padding
        ctx.groups = groups
        return F.conv2d(x, weight, padding=padding, groups=groups)

    @staticmethod
    def backward(ctx, g):
        (weight,) = ctx.saved_tensors
        kh, kw = weight.shape[-2], weight.shape[-1]
        ph, pw = ctx.padding
        flipped = weight.flip(-1, -2)
        if ctx.groups == 1:
            flipped = flipped.transpose(0, 1)                # (1, O, kh, kw)
        with _no_autocast(g):
            gx = F.conv2d(g, flipped, padding=(kh - 1 - ph, kw - 1 - pw),
                          groups=ctx.groups)
        return gx, None, None, None


def _pair(v):
    return (v, v) if isinstance(v, int) else (int(v[0]), int(v[1]))


def plane_conv(x: torch.Tensor, weight: torch.Tensor, *,
               padding=0, cuda_depthwise: bool = True) -> torch.Tensor:
    """``F.conv2d(x, weight, padding=padding)`` for ``x`` ``(B, 1, H, W)``
    and a constant ``weight`` ``(O, 1, kh, kw)``: ``(B, O, H', W')``.

    Inference (no autograd through ``x``) is exactly ``F.conv2d``. When
    autograd will differentiate through ``x``:

    * CUDA, float32/float64, ``B >= 2``: one depthwise convolution of
      ``(1, B, H, W)`` with ``groups=B`` -- PyTorch's native FP32 depthwise
      kernels in both directions, never TF32.
      (``cuda_depthwise=False`` keeps plain ``F.conv2d`` on CUDA, for call
      sites measured to get FP32 cuDNN kernels and to be slower otherwise.)
    * otherwise: :class:`_PlaneCorr` (forward-correlation backward).

    Autocast is disabled either way: the planes keep their dtype.
    """
    pad = _pair(padding)
    w = weight if weight.dtype == x.dtype else weight.to(x.dtype)
    with _no_autocast(x):
        if (not (x.requires_grad and torch.is_grad_enabled())
                or weight.requires_grad or (x.is_cuda and not cuda_depthwise)
                or pad[0] > w.shape[-2] - 1 or pad[1] > w.shape[-1] - 1):
            return F.conv2d(x, w, padding=pad)
        b = x.shape[0]
        if (x.is_cuda and b >= 2 and x.shape[1] == 1
                and x.dtype in (torch.float32, torch.float64)):
            o = w.shape[0]
            h, wd = x.shape[-2], x.shape[-1]
            bank = w.repeat(b, 1, 1, 1)                       # (B*O, 1, kh, kw)
            out = F.conv2d(x.reshape(1, b, h, wd), bank, padding=pad, groups=b)
            return out.view(b, o, out.shape[-2], out.shape[-1])
        return _PlaneCorr.apply(x, w, pad, 1)


#: A compiled (Inductor) reduction over whole images gets roughly one thread
#: block per reduced group on CUDA. With few groups and large images that
#: starves the GPU -- and after a pyramid's second scale turns the region
#: dynamic, the kernel tuned for that size hint runs every scale. Measured on
#: an RTX 3090: VIFP 1.3 -> 4.7 ms per 1080p scale, MS-GMSD 3.2 ms per scale,
#: ADM 1.1 ms per scale. Below these limits the single fused kernel wins on
#: launch count (splitting cost up to 16% at 256x256), so it is kept there.
SPLIT_MIN_ELEMS = 65536
SPLIT_MAX_GROUPS = 4


def starved_reduction(t: torch.Tensor, groups: int, elems_per_group: int,
                      min_elems: int = SPLIT_MIN_ELEMS) -> bool:
    """Whether a compiled whole-image reduction of ``t`` should be split.

    ``min_elems`` is per call site (cheap per-element work needs larger
    groups before splitting pays). With autograd recording, the fused
    region's compiled backward keeps it ahead longer, so the bar is 4x.
    """
    if not t.is_cuda or groups >= SPLIT_MAX_GROUPS:
        return False
    if t.requires_grad and torch.is_grad_enabled():
        min_elems *= 4
    return elems_per_group > min_elems


def depthwise_conv(x: torch.Tensor, weight: torch.Tensor, *,
                   padding=0) -> torch.Tensor:
    """``F.conv2d(x, weight, padding=padding, groups=C)`` for ``x``
    ``(B, C, H, W)`` and a constant per-channel bank ``weight`` ``(C, 1, kh, kw)``.

    CPU with autograd: :class:`_PlaneCorr` (oneDNN's depthwise backward ran
    ~2x its forward). Otherwise plain ``F.conv2d`` -- on CUDA, float
    depthwise convolutions already use PyTorch's native FP32 kernels.
    """
    pad = _pair(padding)
    c = x.shape[1]
    w = weight if weight.dtype == x.dtype else weight.to(x.dtype)
    with _no_autocast(x):
        if (x.is_cuda or not (x.requires_grad and torch.is_grad_enabled())
                or weight.requires_grad or c < 2 or w.shape[0] != c
                or pad[0] > w.shape[-2] - 1 or pad[1] > w.shape[-1] - 1):
            return F.conv2d(x, w, padding=pad, groups=c)
        return _PlaneCorr.apply(x, w, pad, c)
