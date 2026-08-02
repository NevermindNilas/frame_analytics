"""VapourSynth bindings -- the ``vmaf.Metric`` API, backed by these kernels.

    >>> import vapoursynth as vs
    >>> from frame_analytics import vapoursynth as fa_vs
    >>> dist = fa_vs.Metric(ref, dist, [0, 2, 3])     # psnr, ssim, ms_ssim
    >>> dist.set_output()

The signature, the feature numbering and the frame-property names follow
`VapourSynth-VMAF <https://github.com/HomeOfVapourSynthEvolution/VapourSynth-VMAF>`_
so a script written against that plugin keeps working:

====  =============  ==============================================
id    feature        frame properties
====  =============  ==============================================
0     psnr           ``psnr_y``, ``psnr_cb``, ``psnr_cr``
1     psnr_hvs       *not implemented -- libvmaf only*
2     ssim           ``float_ssim``
3     ms_ssim        ``float_ms_ssim``
4     ciede2000      *not implemented -- libvmaf only*
5     gmsd           ``gmsd``
6     gms            ``gms``
7     mse            ``mse_y``, ``mse_cb``, ``mse_cr``
8     l1             ``l1_y``, ``l1_cb``, ``l1_cr``
9     charbonnier    ``charbonnier_y``, ...
10    huber          ``huber_y``, ...
11    lpips          ``lpips``          *(RGB clips only)*
12    ssimulacra2    ``ssimulacra2``    *(RGB clips only)*
====  =============  ==============================================

Ids 1 and 4 are the two libvmaf features with no counterpart here; asking for
either raises rather than silently returning nothing.  Everything from 5 up is
an extension, and every feature can also be named as a string --
``Metric(ref, dist, ["psnr", "ssim"])`` -- which is the readable form.

Differences from the plugin, all of them widenings:

* **Formats.**  GRAY, YUV and RGB; 8-16 bit integer and 32-bit float.  The
  plugin is YUV-integer only.
* **Planes.**  The per-plane metrics run on each plane at its own resolution,
  as libvmaf does.  The single-value metrics (SSIM, MS-SSIM, GMSD, GMS) run on
  the luma plane of a GRAY/YUV clip -- again matching libvmaf, whose
  ``float_ssim`` is luma-only -- and on all three channels of an RGB clip.
* **LPIPS and SSIMULACRA 2** need real RGB, so they reject YUV input rather
  than guess a matrix.  Convert first::

      rgb = clip.resize.Bicubic(format=vs.RGBS, matrix_in_s="709")

  SSIMULACRA 2 additionally wants **sRGB-encoded** values, which is what a
  normal ``matrix_in_s`` conversion of normal video gives you.
* **Logging** is available on ``Metric`` itself (the plugin only logs from
  ``vmaf.VMAF``), in the same four formats.

The work happens on the GPU when there is one; ``accelerator="cpu"`` forces
otherwise, and ``accelerator="gpu"`` refuses to run at all without one, which
is what you want in a script whose timings assume a card.  ``device=`` takes a
torch device for the cases a word cannot express (``"cuda:1"``, ``"mps"``) and
wins over ``accelerator`` when both are given.  Frames are converted through
the buffer protocol, so a plane costs one host->device copy and nothing else.

Threads
-------
VapourSynth calls the selector from ``core.num_threads`` workers at once, which
is free parallelism on CUDA -- the copies and launches from different threads
overlap.  On ``accelerator="cpu"`` it collides with torch's own intra-op pool:
24 workers each asking for 24 threads is 576 threads fighting over 24 cores.
Either ``torch.set_num_threads(1)`` and let VapourSynth do the parallelism, or
``core.num_threads = 2`` and let torch do it; the first is usually faster for
frame-sized work.
"""

from __future__ import annotations

import atexit
import json
import math
import re
import threading
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import vapoursynth as vs
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "frame_analytics.vapoursynth needs the VapourSynth Python module "
        "(pip install vapoursynth, plus the VapourSynth runtime on Linux)"
    ) from exc

from . import functional as _F
from .perceptual import lpips as _lpips
from .ssimulacra2 import ssimulacra2 as _ssimulacra2

__all__ = [
    "ACCELERATORS",
    "Metric",
    "PSNR",
    "SSIM",
    "MSSSIM",
    "GMSD",
    "GMS",
    "MSE",
    "L1",
    "Charbonnier",
    "Huber",
    "LPIPS",
    "SSIMULACRA2",
    "available_features",
    "resolve_device",
    "pooled_scores",
]

core = vs.core

# ``np.asarray`` on a frame's memoryview gives a read-only array, and
# ``torch.from_numpy`` warns about exactly that on every call.  We never write
# through the tensor -- the copy to the compute device, or the dtype cast, is
# what materialises it -- so the warning carries no information here.  Scoped to
# this module by name so nobody else's from_numpy goes quiet.
warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable",
    category=UserWarning,
    module=re.escape(__name__),
)


# --------------------------------------------------------------------------- #
# feature table
# --------------------------------------------------------------------------- #

class _Feature:
    """One entry of the ``feature`` argument.

    ``kind`` decides what gets fed to ``fn``:

    ``per_plane``
        every plane on its own, at its own resolution; the property name gets
        a plane suffix.
    ``image``
        luma of a GRAY/YUV clip, all three channels of an RGB clip.
    ``rgb``
        all three channels, RGB clips only.
    """

    __slots__ = ("id", "name", "kind", "prop", "fn", "data_range", "backend_hint")

    def __init__(self, id_, name, kind, prop, fn, data_range=True, backend_hint=False):
        self.id = id_
        self.name = name
        self.kind = kind
        self.prop = prop
        self.fn = fn
        self.data_range = data_range      # does fn take data_range=?
        self.backend_hint = backend_hint  # does fn take backend_hint=?


_FEATURE_LIST = (
    _Feature(0, "psnr", "per_plane", "psnr", _F.psnr),
    _Feature(2, "ssim", "image", "float_ssim", _F.ssim, backend_hint=True),
    _Feature(3, "ms_ssim", "image", "float_ms_ssim", _F.ms_ssim, backend_hint=True),
    _Feature(5, "gmsd", "image", "gmsd", _F.gmsd, backend_hint=True),
    _Feature(6, "gms", "image", "gms", _F.gms, backend_hint=True),
    _Feature(7, "mse", "per_plane", "mse", _F.mse, data_range=False),
    _Feature(8, "l1", "per_plane", "l1", _F.l1, data_range=False, backend_hint=True),
    _Feature(9, "charbonnier", "per_plane", "charbonnier", _F.charbonnier,
             data_range=False, backend_hint=True),
    _Feature(10, "huber", "per_plane", "huber", _F.huber,
             data_range=False, backend_hint=True),
    _Feature(11, "lpips", "rgb", "lpips", _lpips),
    _Feature(12, "ssimulacra2", "rgb", "ssimulacra2", _ssimulacra2),
)

_BY_NAME = {f.name: f for f in _FEATURE_LIST}
_BY_ID = {f.id: f for f in _FEATURE_LIST}

# libvmaf's, and only libvmaf's
_LIBVMAF_ONLY = {1: "psnr_hvs", 4: "ciede2000"}

_SUFFIX = {
    vs.GRAY: ("_y",),
    vs.YUV: ("_y", "_cb", "_cr"),
    vs.RGB: ("_r", "_g", "_b"),
}


def available_features() -> Dict[str, int]:
    """``{feature name: id}`` for everything this module can compute."""
    return {f.name: f.id for f in _FEATURE_LIST}


def _resolve_features(feature) -> List[_Feature]:
    if feature is None:
        raise ValueError(
            "feature is required; pass an id or a name, e.g. 0 / \"psnr\" / "
            f"[0, 2, 3]. Available: {sorted(_BY_NAME)}"
        )
    # np.integer alongside int: a numpy scalar is not a Python int, and one
    # arriving on its own would otherwise be rejected while the same value
    # inside a list is accepted.
    if isinstance(feature, (str, int, np.integer)) and not isinstance(feature, bool):
        feature = [feature]
    if not isinstance(feature, Iterable):
        raise TypeError(f"feature must be an int, a str or a sequence of them, "
                        f"got {type(feature).__name__}")

    out: List[_Feature] = []
    for item in feature:
        if isinstance(item, str):
            key = item.strip().lower().replace("-", "_")
            if key not in _BY_NAME:
                raise ValueError(f"unknown feature {item!r}; "
                                 f"expected one of {sorted(_BY_NAME)}")
            f = _BY_NAME[key]
        elif isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            i = int(item)
            if i in _LIBVMAF_ONLY:
                raise ValueError(
                    f"feature {i} ({_LIBVMAF_ONLY[i]}) is a libvmaf feature with "
                    "no frame_analytics implementation; use vmaf.Metric for it"
                )
            if i not in _BY_ID:
                raise ValueError(f"unknown feature id {i}; expected one of "
                                 f"{sorted(_BY_ID)} (see available_features())")
            f = _BY_ID[i]
        else:
            raise TypeError(f"feature entries must be int or str, got "
                            f"{type(item).__name__}")
        if f not in out:
            out.append(f)
    if not out:
        raise ValueError("feature is empty; nothing to compute")
    return out


# --------------------------------------------------------------------------- #
# frame -> tensor
# --------------------------------------------------------------------------- #

def _plane_to_tensor(frame, plane: int, device: torch.device) -> torch.Tensor:
    """One plane of a VS frame as a ``(1, 1, H, W)`` tensor on ``device``.

    8-bit stays ``uint8`` all the way into the kernel -- that is the path the
    native backends are fastest on.  10/12/16-bit arrives as ``uint16``, which
    torch can hold but barely operate on, so it becomes float32 on the far side
    of the copy (on the GPU, where the widening is free).
    """
    arr = np.asarray(frame[plane])
    if arr.dtype == np.uint16 and getattr(torch, "uint16", None) is None:
        arr = arr.astype(np.float32)          # torch < 2.3 has no uint16 dtype
    t = torch.from_numpy(arr)
    if not t.is_contiguous():                 # padded stride
        t = t.contiguous()
    t = t.to(device)
    if t.dtype == getattr(torch, "uint16", None):
        t = t.to(torch.float32)
    return t.unsqueeze_(0).unsqueeze_(0)


def _data_range(fmt) -> float:
    if fmt.sample_type == vs.FLOAT:
        return 1.0
    return float((1 << fmt.bits_per_sample) - 1)


# --------------------------------------------------------------------------- #
# the scorer
# --------------------------------------------------------------------------- #

class _Runner:
    """Everything about a ``Metric`` call that does not change per frame."""

    def __init__(self, feats, fmt, device, dtype, data_range, crop_border,
                 backend_hint, options):
        self.feats = feats
        self.device = device
        self.n_planes = fmt.num_planes
        self.suffix = _SUFFIX[fmt.color_family]
        self.is_rgb = fmt.color_family == vs.RGB
        self.data_range = float(data_range) if data_range is not None else _data_range(fmt)

        base = {"dtype": dtype, "crop_border": int(crop_border)}
        self.kwargs = {}
        for f in feats:
            kw = dict(base)
            if f.data_range:
                kw["data_range"] = self.data_range
            if f.backend_hint:
                kw["backend_hint"] = backend_hint
            kw.update((options or {}).get(f.name, {}))
            self.kwargs[f.name] = kw

    @property
    def prop_names(self) -> List[str]:
        names = []
        for f in self.feats:
            if f.kind == "per_plane":
                names.extend(f.prop + s for s in self.suffix[:self.n_planes])
            else:
                names.append(f.prop)
        return names

    def scores(self, fref, fdist) -> Dict[str, float]:
        planes: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        def plane(p):
            if p not in planes:
                planes[p] = (_plane_to_tensor(fref, p, self.device),
                             _plane_to_tensor(fdist, p, self.device))
            return planes[p]

        image = None
        out: Dict[str, float] = {}
        with torch.inference_mode():
            for f in self.feats:
                kw = self.kwargs[f.name]
                if f.kind == "per_plane":
                    for p in range(self.n_planes):
                        x, y = plane(p)
                        out[f.prop + self.suffix[p]] = float(f.fn(x, y, **kw))
                else:
                    if image is None:
                        if self.is_rgb:
                            image = (torch.cat([plane(p)[0] for p in range(3)], 1),
                                     torch.cat([plane(p)[1] for p in range(3)], 1))
                        else:
                            image = plane(0)
                    out[f.prop] = float(f.fn(image[0], image[1], **kw))
        return out


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #

_LOG_FORMATS = {0: "xml", 1: "json", 2: "csv", 3: "sub"}


def _pool(values: Sequence[float]) -> Dict[str, float]:
    """min / max / mean / harmonic mean, the four libvmaf pools.

    The harmonic mean is libvmaf's shifted one, ``1 / mean(1/(v+1)) - 1``,
    which stays finite for a score of zero.
    """
    n = len(values)
    if not n:
        nan = float("nan")
        return {"min": nan, "max": nan, "mean": nan, "harmonic_mean": nan}
    hm = float("nan")
    if all(v > -1.0 for v in values):
        s = math.fsum(1.0 / (v + 1.0) for v in values)
        hm = (n / s) - 1.0 if s > 0.0 else float("inf")
    return {
        "min": min(values),
        "max": max(values),
        "mean": math.fsum(values) / n,
        "harmonic_mean": hm,
    }


def _fmt(v: float) -> str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.6f}"


class _Logger:
    """Collects per-frame scores and writes them once the clip has been read.

    The plugin writes its log when the filter is freed.  There is no such hook
    here, so the log goes out as soon as every frame has reported, and an
    ``atexit`` handler covers the run that is cut short -- a partial log rather
    than none.
    """

    def __init__(self, path: str, fmt: int, props: Sequence[str], clip):
        if fmt not in _LOG_FORMATS:
            raise ValueError(f"log_format must be one of {_LOG_FORMATS}, got {fmt}")
        self.path = str(path)
        self.fmt = _LOG_FORMATS[fmt]
        self.props = list(props)
        self.num_frames = clip.num_frames
        self.rows: Dict[int, Dict[str, float]] = {}
        self.lock = threading.Lock()
        self.written = False
        atexit.register(self.flush)

    def add(self, n: int, scores: Dict[str, float]) -> None:
        with self.lock:
            self.rows[n] = scores
            done = len(self.rows) >= self.num_frames
        if done:
            self.flush()

    def flush(self) -> None:
        with self.lock:
            if self.written or not self.rows:
                return
            self.written = True
            rows = [(n, self.rows[n]) for n in sorted(self.rows)]
        text = getattr(self, "_as_" + self.fmt)(rows)
        with open(self.path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)

    def _pooled(self, rows):
        return {p: _pool([r[p] for _, r in rows if p in r]) for p in self.props}

    def _as_xml(self, rows) -> str:
        from . import __version__

        out = ['<?xml version="1.0" encoding="utf-8"?>',
               f'<frame_analytics version="{__version__}">',
               "  <frames>"]
        for n, r in rows:
            attrs = " ".join(f'{p}="{_fmt(r[p])}"' for p in self.props if p in r)
            out.append(f'    <frame frameNum="{n}" {attrs}/>')
        out.append("  </frames>")
        out.append("  <pooled_metrics>")
        for p, s in self._pooled(rows).items():
            stats = " ".join(f'{k}="{_fmt(v)}"' for k, v in s.items())
            out.append(f'    <metric name="{p}" {stats}/>')
        out.append("  </pooled_metrics>")
        out.append("</frame_analytics>")
        return "\n".join(out) + "\n"

    def _as_json(self, rows) -> str:
        from . import __version__

        doc = {
            "version": __version__,
            "frames": [{"frameNum": n, "metrics": r} for n, r in rows],
            "pooled_metrics": self._pooled(rows),
        }
        # inf is what an identical frame pair scores on PSNR; JSON has no
        # literal for it, so it goes out as a string rather than as the
        # `Infinity` token, which is not valid JSON.
        def clean(o):
            if isinstance(o, float) and not math.isfinite(o):
                return _fmt(o)
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, list):
                return [clean(v) for v in o]
            return o

        return json.dumps(clean(doc), indent=2) + "\n"

    def _as_csv(self, rows) -> str:
        out = ["frameNum," + ",".join(self.props)]
        for n, r in rows:
            out.append(str(n) + "," + ",".join(_fmt(r[p]) if p in r else ""
                                               for p in self.props))
        return "\n".join(out) + "\n"

    def _as_sub(self, rows) -> str:
        """MicroDVD, which is what libvmaf's format 3 actually writes.

        ``{start}{end}`` are frame numbers, not timestamps, so the file needs
        no frame rate and cannot disagree with the clip's -- and the fields
        within a line are pipe-separated, MicroDVD's own line break.
        """
        out = []
        for n, r in rows:
            body = "".join(f"{p}: {_fmt(r[p])}|" for p in self.props if p in r)
            out.append(f"{{{n}}}{{{n + 1}}}frame: {n}|{body}")
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def _check_clips(reference, distorted, feats):
    for name, clip in (("reference", reference), ("distorted", distorted)):
        if not isinstance(clip, vs.VideoNode):
            raise vs.Error(f"Metric: {name} must be a clip, got "
                           f"{type(clip).__name__}")
        if clip.format is None or clip.width == 0 or clip.height == 0:
            raise vs.Error(f"Metric: {name} must have a constant format and "
                           "constant dimensions")
        if clip.format.color_family not in _SUFFIX:
            raise vs.Error(f"Metric: {name} must be GRAY, YUV or RGB, got "
                           f"{clip.format.color_family.name}")
        if clip.format.sample_type == vs.INTEGER and clip.format.bits_per_sample > 16:
            raise vs.Error(f"Metric: {name} has {clip.format.bits_per_sample}-bit "
                           "integer samples; 8-16 bit integer or 32-bit float only")

    if reference.format.id != distorted.format.id:
        raise vs.Error("Metric: reference and distorted must have the same format "
                       f"({reference.format.name} vs {distorted.format.name})")
    if (reference.width, reference.height) != (distorted.width, distorted.height):
        raise vs.Error("Metric: reference and distorted must have the same "
                       f"dimensions ({reference.width}x{reference.height} vs "
                       f"{distorted.width}x{distorted.height})")
    if reference.num_frames != distorted.num_frames:
        raise vs.Error("Metric: reference and distorted must have the same number "
                       f"of frames ({reference.num_frames} vs "
                       f"{distorted.num_frames})")

    rgb_only = [f.name for f in feats if f.kind == "rgb"]
    if rgb_only and reference.format.color_family != vs.RGB:
        raise vs.Error(
            f"Metric: {', '.join(rgb_only)} needs RGB input, got "
            f"{reference.format.name}. Convert first, e.g. "
            'clip.resize.Bicubic(format=vs.RGBS, matrix_in_s="709")'
        )


#: What ``accelerator`` accepts.  ``"gpu"`` is the word a VapourSynth script
#: reaches for; the torch spelling works too, and ``device`` is still there for
#: the cases a word cannot express, like a second card.
ACCELERATORS = ("auto", "gpu", "cuda", "cpu")


def resolve_device(accelerator="auto", device=None) -> torch.device:
    """Turn ``accelerator``/``device`` into one torch device.

    ``device`` wins when both are given: it is the more specific of the two,
    and the only way to name ``cuda:1`` or a backend this list has never heard
    of.  Asking for ``"gpu"`` without one present is an error rather than a
    silent fall back to the CPU -- a run that quietly went 40x slower is worse
    than one that stopped.
    """
    if device is not None:
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise vs.Error("Metric: device='cuda' but torch reports no CUDA device")
        return dev

    acc = str(accelerator).strip().lower()
    if acc == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if acc == "cpu":
        return torch.device("cpu")
    if acc in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise vs.Error(
                f"Metric: accelerator={accelerator!r} but torch reports no CUDA "
                "device. Use accelerator=\"cpu\", or accelerator=\"auto\" to take "
                "whichever is present; other torch backends are reachable as "
                'device="mps" and so on.'
            )
        return torch.device("cuda")
    raise vs.Error(f"Metric: accelerator must be one of {list(ACCELERATORS)}, "
                   f"got {accelerator!r}")


# --------------------------------------------------------------------------- #
# the filter
# --------------------------------------------------------------------------- #

def Metric(
    reference: vs.VideoNode,
    distorted: vs.VideoNode,
    feature: Union[int, str, Sequence[Union[int, str]], None] = None,
    *,
    accelerator: str = "auto",
    device: Union[str, torch.device, None] = None,
    dtype: Optional[torch.dtype] = None,
    data_range: Optional[float] = None,
    crop_border: int = 0,
    backend_hint: str = "auto",
    options: Optional[Dict[str, Dict[str, Any]]] = None,
    log_path: Optional[str] = None,
    log_format: int = 0,
) -> vs.VideoNode:
    """Score every frame pair and attach the scores as frame properties.

    Returns ``distorted`` with the properties added -- the same clip the
    plugin's ``vmaf.Metric`` returns, so ``.set_output()`` on the result gives
    you the distorted video and the numbers together.

    Parameters
    ----------
    reference, distorted
        Same format, same dimensions, same length.  GRAY/YUV/RGB, 8-16 bit
        integer or 32-bit float.  Argument order matters for SSIMULACRA 2,
        which is not symmetric.
    feature
        Ids or names, one or a list; see the table at the top of the module and
        :func:`available_features`.
    accelerator
        ``"auto"`` (GPU when there is one, else CPU), ``"gpu"``/``"cuda"``, or
        ``"cpu"``.  ``"gpu"`` raises when no CUDA device is present rather than
        quietly running 40x slower on the CPU.
    device
        A specific torch device, for what a word cannot say -- ``"cuda:1"``,
        ``"mps"``.  Overrides ``accelerator`` when both are given.
    dtype
        Compute dtype.  Default float32 (float64 for float64 input).
    data_range
        Peak value.  Default ``(1 << bits) - 1`` for integer clips and 1.0 for
        float, i.e. full-range.  Pass 219*2**(bits-8) to treat limited-range
        luma as such.
    crop_border
        Drop this many pixels from every edge first -- the convention the
        super-resolution literature reports.
    backend_hint
        ``"auto"``, ``"native"`` or ``"torch"``, for the metrics with a C-ABI
        kernel behind them.
    options
        Per-feature keyword overrides, e.g.
        ``{"ssim": {"win_size": 7}, "gmsd": {"downsample": False}}``.
    log_path, log_format
        Write every frame's scores to a file when the clip has been read
        through.  Formats: 0 XML, 1 JSON, 2 CSV, 3 subtitle (MicroDVD, as
        libvmaf writes it) -- the plugin's numbering.  Frames that are never
        requested never appear.

    Notes
    -----
    A PSNR of ``inf`` is what an identical frame pair scores.  Pass
    ``options={"psnr": {"eps": 1e-10}}`` for a large finite number instead.
    """
    feats = _resolve_features(feature)
    _check_clips(reference, distorted, feats)
    dev = resolve_device(accelerator, device)

    runner = _Runner(feats, distorted.format, dev, dtype, data_range,
                     crop_border, backend_hint, options)
    logger = _Logger(log_path, log_format, runner.prop_names, distorted) \
        if log_path is not None else None

    def _selector(n: int, f: Sequence[vs.VideoFrame]) -> vs.VideoFrame:
        scores = runner.scores(f[0], f[1])
        out = f[1].copy()
        for name, value in scores.items():
            out.props[name] = value
        if logger is not None:
            logger.add(n, scores)
        return out

    return core.std.ModifyFrame(clip=distorted, clips=[reference, distorted],
                                selector=_selector)


def _wrapper(name):
    def filt(reference, distorted, **kwargs):
        return Metric(reference, distorted, name, **kwargs)

    filt.__name__ = name
    filt.__qualname__ = name
    filt.__doc__ = (f"``Metric(reference, distorted, {name!r})``. "
                    "Takes every keyword :func:`Metric` takes.")
    return filt


#: One-metric spellings of :func:`Metric`, for scripts that want one number.
PSNR = _wrapper("psnr")
SSIM = _wrapper("ssim")
MSSSIM = _wrapper("ms_ssim")
GMSD = _wrapper("gmsd")
GMS = _wrapper("gms")
MSE = _wrapper("mse")
L1 = _wrapper("l1")
Charbonnier = _wrapper("charbonnier")
Huber = _wrapper("huber")
LPIPS = _wrapper("lpips")
SSIMULACRA2 = _wrapper("ssimulacra2")


def pooled_scores(clip: vs.VideoNode,
                  props: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """Read ``clip`` through and pool every float property it carries.

    The batch counterpart to :func:`Metric`: score a whole encode and get one
    number per metric, without a log file or an output target.

    >>> scored = fa_vs.Metric(ref, dist, ["psnr", "ssim"])
    >>> pooled_scores(scored)["float_ssim"]["mean"]

    ``props`` restricts the properties collected; by default it takes every
    float-valued property that is not one of VapourSynth's own ``_``-prefixed
    ones.
    """
    keep = set(props) if props is not None else None
    acc: Dict[str, List[float]] = {}
    for frame in clip.frames(close=True):
        for name, value in frame.props.items():
            if keep is not None:
                if name not in keep:
                    continue
            elif name.startswith("_"):
                continue
            if isinstance(value, float):
                acc.setdefault(name, []).append(value)
    return {name: _pool(values) for name, values in acc.items()}
