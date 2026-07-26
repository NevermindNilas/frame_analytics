"""LPIPS -- the learned perceptual metric of Zhang et al. 2018.

Unlike everything else in this package there is no float64 transcription of a
paper to be correct against: LPIPS *is* its weights, so the definition of
correct is "agrees with ``richzhang/PerceptualSimilarity``".  The gate in
:mod:`frame_analytics.reference` is still a float64 transcription of the
*forward*, which catches an algorithmic slip; the tests in
``tests/test_lpips.py`` pin the result against the reference package itself
when it is installed.

Why it is here at all: PSNR and SSIM both reward blur, so a model trained on
them converges to something smooth.  LPIPS is the term that punishes that, and
it is the third number every super-resolution and restoration paper reports.

Weights
-------
``weights/lpips_alex_v0.1.pt`` ships inside the wheel: 9.4 MiB, float32, no
download and no ``torchvision`` dependency.  It holds the five conv layers
LPIPS taps out of AlexNet plus the five 1x1 calibration convs -- see
``tools/build_lpips_weights.py`` for how it is built and why the 233 MiB
torchvision checkpoint shrinks that far (the classifier head is ~94% of it and
LPIPS discards it).

The VGG trunk is deliberately *not* shipped: 56 MiB for the variant used as a
training loss rather than the one papers report.

Memory
------
The tail -- unit-normalise, square the difference, apply the channel weights,
reduce -- is where the frame-shaped temporaries live, so it is fused into one
``torch.compile``d region per layer rather than five separate ones.  AlexNet's
first conv has stride 4, so nothing downstream of it is full resolution; this
is much cheaper than the VGG trunk, where ``relu1_2`` alone is 506 MiB per
image at 1080p.
"""

from __future__ import annotations

import os
import pathlib
import threading
from typing import Optional, cast

import torch
import torch.nn.functional as F

from .functional import _maybe_compile, _prep, _work_dtype, rgb_to_luma

__all__ = ["lpips", "lpips_weights_path", "available_nets"]

_WEIGHTS_DIR = pathlib.Path(__file__).resolve().parent / "weights"
_FILES = {"alex": "lpips_alex_v0.1.pt"}

# ScalingLayer of the reference implementation: the ImageNet mean/std the
# trunk was trained under, expressed in [-1, 1] units.
_SHIFT = (-0.030, -0.088, -0.188)
_SCALE = (0.458, 0.448, 0.450)

# normalize_tensor()'s epsilon. It is added to the norm, not to its square,
# which matters: the reference is x / (||x|| + 1e-10), not x / sqrt(|x|^2+eps).
_NORM_EPS = 1e-10

_cache: dict = {}
_cache_lock = threading.Lock()


class _no_tf32:
    """Force full float32 convolutions for the duration of the block.

    ``torch.backends.cudnn.allow_tf32`` defaults to *True*, so on Ampere and
    later every convolution here would otherwise run with a 10-bit mantissa.
    Measured on an RTX 3090 that moves LPIPS by ~7e-05 against the float64
    reference -- the fourth decimal, which is the decimal papers print. The
    convolutions are a rounding error inside a training step anyway, so the
    default is accuracy and ``allow_tf32=True`` is there for whoever measures
    otherwise.

    Restoring rather than using ``torch.backends.cudnn.flags`` is deliberate:
    that helper's signature defaults ``enabled=False``, so calling it to set
    one field quietly turns cuDNN off entirely.
    """

    __slots__ = ("_prev",)

    def __init__(self, active: bool):
        self._prev = None if not active else torch.backends.cudnn.allow_tf32

    def __enter__(self):
        if self._prev is not None:
            torch.backends.cudnn.allow_tf32 = False
        return self

    def __exit__(self, *exc):
        if self._prev is not None:
            torch.backends.cudnn.allow_tf32 = self._prev
        return False


def available_nets() -> tuple:
    """Trunks with weights present in this install. Currently ``("alex",)``."""
    return tuple(sorted(n for n, f in _FILES.items() if (_WEIGHTS_DIR / f).is_file()))


def lpips_weights_path(net: str = "alex") -> pathlib.Path:
    """Locate the weight blob, or explain precisely what is missing.

    ``FA_LPIPS_WEIGHTS`` overrides the search with an explicit file, which is
    how you point at a trunk this package does not ship.
    """
    override = os.environ.get("FA_LPIPS_WEIGHTS")
    if override:
        p = pathlib.Path(override)
        if not p.is_file():
            raise FileNotFoundError(
                f"FA_LPIPS_WEIGHTS={override!r} does not point at a file")
        return p
    if net not in _FILES:
        raise ValueError(f"unknown LPIPS trunk {net!r}; this package ships "
                         f"{sorted(_FILES)}")
    p = _WEIGHTS_DIR / _FILES[net]
    if not p.is_file():
        raise FileNotFoundError(
            f"LPIPS weights missing: {p}\n"
            "They ship inside the wheel, so this is either a source checkout "
            "that has not run tools/build_lpips_weights.py, or an install that "
            "stripped package data. Point FA_LPIPS_WEIGHTS at a blob to "
            "override.")
    return p


class _LpipsNet(torch.nn.Module):
    """Trunk + calibration, as buffers rather than parameters.

    Buffers, deliberately: an ``fa.LPIPS()`` dropped into a training script
    must not put 2.5M frozen ImageNet weights into the caller's optimiser, and
    must not have them show up in the checkpoint of the model being trained.
    ``.to(device)`` still moves them, which is the only thing we need.
    """

    def __init__(self, blob: dict):
        super().__init__()
        self.channels = list(blob["channels"])
        self.pool_before = list(blob["pool_before"])
        self.strides = [int(t["stride"]) for t in blob["trunk"]]
        self.paddings = [int(t["padding"]) for t in blob["trunk"]]
        self.n_layers = len(self.channels)
        for i, t in enumerate(blob["trunk"]):
            self.register_buffer(f"w{i}", t["weight"], persistent=False)
            self.register_buffer(f"b{i}", t["bias"], persistent=False)
        for i, w in enumerate(blob["lin"]):
            self.register_buffer(f"lin{i}", w, persistent=False)
        self.register_buffer(
            "shift", torch.tensor(_SHIFT).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "scale", torch.tensor(_SCALE).view(1, 3, 1, 1), persistent=False)

    def buf(self, name: str) -> torch.Tensor:
        """Registered buffers by generated name, typed. ``.to()`` rebinds the
        attribute, so they have to be fetched rather than held in a list."""
        return cast(torch.Tensor, getattr(self, name))

    def features(self, x: torch.Tensor) -> list:
        """The five post-ReLU taps, on input already in [-1, 1]."""
        h = (x - self.buf("shift")) / self.buf("scale")
        out = []
        for i in range(self.n_layers):
            if self.pool_before[i]:
                h = F.max_pool2d(h, kernel_size=3, stride=2)
            h = F.relu(F.conv2d(h, self.buf(f"w{i}"), self.buf(f"b{i}"),
                                stride=self.strides[i], padding=self.paddings[i]))
            out.append(h)
        return out


def _lpips_layer_map(f0: torch.Tensor, f1: torch.Tensor, lin: torch.Tensor):
    """Unit-normalise, square the difference, apply channel weights.

    One fused region: the two normalised copies and the squared difference are
    all feature-map-shaped, and materialising them separately is most of what
    the textbook formulation costs.
    """
    n0 = f0 / (f0.pow(2).sum(dim=1, keepdim=True).sqrt() + _NORM_EPS)
    n1 = f1 / (f1.pow(2).sum(dim=1, keepdim=True).sqrt() + _NORM_EPS)
    d = (n0 - n1).pow(2)
    return F.conv2d(d, lin)


def _lpips_layer_mean(f0, f1, lin):
    return _lpips_layer_map(f0, f1, lin).mean(dim=(2, 3))


def _load_net(net: str, device: torch.device, dtype: torch.dtype) -> _LpipsNet:
    key = (net, str(device), dtype)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            return hit
        path = lpips_weights_path(net)
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
        except (TypeError, RuntimeError):
            # torch<2.6 rejects some of the plain-python entries under
            # weights_only; the file is our own package data either way.
            blob = torch.load(path, map_location="cpu")
        if int(blob.get("format_version", 0)) != 1:
            raise RuntimeError(
                f"{path} has format_version {blob.get('format_version')!r}, "
                "expected 1 -- stale package data next to newer sources")
        model = _LpipsNet(blob).to(device=device, dtype=dtype).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _cache[key] = model
        return model


def lpips(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    net: str = "alex",
    data_range: Optional[float] = None,
    reduction: str = "mean",
    return_map: bool = False,
    allow_tf32: bool = False,
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
) -> torch.Tensor:
    """Learned Perceptual Image Patch Similarity (Zhang et al. 2018).

    Lower is better; 0 is identical. Typical values run 0.0-0.7.

    Parameters
    ----------
    x, y
        Same-shape tensors, ``(H,W)`` / ``(C,H,W)`` / ``(N,C,H,W)``, any dtype.
        3-channel RGB is the calibrated case; 1-channel input is replicated to
        three, which is what the field does with grayscale but is not something
        the human judgements behind the weights ever covered.
    net
        Trunk. ``"alex"`` is the only one shipped, and is the variant reported
        in the literature.
    data_range
        Input scale, as everywhere else in this package: 255 for integers, 1.0
        for floats unless given. Inputs are mapped to the ``[-1, 1]`` the trunk
        expects.
    reduction
        ``"mean"`` -> scalar. ``"none"`` -> ``(N,)``.
    return_map
        Return the spatial map, ``(N,1,H,W)``, instead of a number: each
        layer's contribution bilinearly resampled to the input resolution and
        summed. This is the reference implementation's ``spatial=True``.
    allow_tf32
        Let cuDNN use TF32 convolutions. Off by default, unlike the rest of
        PyTorch: TF32's 10-bit mantissa costs ~7e-05 against the float64
        reference, which is the fourth decimal and therefore the one papers
        print. Turn it on only if you have measured that you need the speed.
    luma, crop_border
        As elsewhere. ``crop_border`` is the usual literature convention;
        ``luma`` collapses to one channel and therefore hits the grayscale
        path above, so it is accepted but not recommended for LPIPS.

    Notes
    -----
    Both inputs are differentiable, so this works directly as a perceptual
    loss. The trunk is frozen and registered as buffers, so it neither enters
    the caller's optimiser nor the caller's checkpoint.

    ``allow_tf32`` covers the forward only -- the backward's convolutions run
    under whatever cuDNN setting is live when ``.backward()`` is called, which
    is a deliberate asymmetry: the *reported number* is what has to survive to
    four decimals, and a loss gradient does not.

    >>> import frame_analytics as fa
    >>> fa.lpips(a, b, data_range=255.0)
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")

    x4, y4, L, mode = _prep(x, y, luma, crop_border, data_range)
    wdt = _work_dtype(x4, dtype)
    if wdt not in (torch.float32, torch.float64):
        raise ValueError(f"LPIPS needs a floating compute dtype, got {wdt}")

    if mode is not None:
        x4 = rgb_to_luma(x4, mode, data_range=L, dtype=wdt)
        y4 = rgb_to_luma(y4, mode, data_range=L, dtype=wdt)

    c = x4.shape[1]
    if c == 1:
        x4 = x4.expand(-1, 3, -1, -1)
        y4 = y4.expand(-1, 3, -1, -1)
    elif c != 3:
        raise ValueError(f"LPIPS needs 1- or 3-channel input, got {c} channels")

    h, w = x4.shape[-2], x4.shape[-1]
    if h < 32 or w < 32:
        raise ValueError(
            f"LPIPS needs at least 32x32 after cropping, got {h}x{w}: the "
            "trunk's stride-4 conv and two max-pools leave nothing behind")

    # [0, L] -> [-1, 1]. Written as one affine map so a float input with
    # data_range=1.0 reduces to the reference's exact `2 * x - 1`.
    xs = x4.to(wdt) * (2.0 / L) - 1.0
    ys = y4.to(wdt) * (2.0 / L) - 1.0

    net_mod = _load_net(net, x4.device, wdt)

    with _no_tf32(active=x4.is_cuda and not allow_tf32 and wdt == torch.float32):
        f0 = net_mod.features(xs)
        f1 = net_mod.features(ys)

        # Keyed per layer, not per function: the five taps have five different
        # channel counts, so one shared compiled callable would need five
        # specialisations and blow through Dynamo's default cache_size_limit of
        # 8 as soon as a second dtype or resolution showed up -- at which point
        # it gives up and runs eager for the rest of the process.
        if return_map:
            maps = [
                F.interpolate(
                    _maybe_compile(_lpips_layer_map, f"lpips_map{i}")(
                        f0[i], f1[i], net_mod.buf(f"lin{i}")),
                    size=(h, w), mode="bilinear", align_corners=False)
                for i in range(net_mod.n_layers)
            ]
            acc = maps[0]
            for m in maps[1:]:
                acc = acc + m
            return acc

        per_layer = [
            _maybe_compile(_lpips_layer_mean, f"lpips_mean{i}")(
                f0[i], f1[i], net_mod.buf(f"lin{i}"))
            for i in range(net_mod.n_layers)
        ]
    total = per_layer[0]
    for v in per_layer[1:]:
        total = total + v
    per_image = total.squeeze(1)              # (N,1) -> (N,)
    return per_image if reduction == "none" else per_image.mean()
