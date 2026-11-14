"""IW-SSIM (Wang & Li 2011) -- final package module, full GSM Eq.28 default.

Torch-only. Shared input front end comes from
:mod:`frame_analytics.functional` (:func:`_prep`, :func:`_apply_luma`,
:func:`_work_dtype`, :func:`_no_autocast`, :func:`_maybe_compile`,
:func:`gaussian_window_1d`); no local duplicates. ``torch.compile`` uses
the shared eager-fallback wrapper; :func:`set_compile_enabled` is a
passthrough to ``functional``.

``weight_mode="gsm"`` (default) is the full information-content weight
(block-GSM Eq.28, ref+distort ``g``/``vv`` with guards, log2 map, parent
gating, binomial-5 Laplacian pyramid, trim-4 alignment, beta product
pooling). ``"ratio"``/``"log"``/``"uniform"`` keep the lite scalar weights
as a fast fallback.

Color: every channel scored independently and product-pooled, then averaged
(no RGB->luma; project to luma first for perceptual use).
"""

from __future__ import annotations

import types
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from . import functional as _functional
from ._metric_utils import starved_reduction
from .functional import (
    _apply_luma,
    _maybe_compile,
    _no_autocast,
    _prep,
    _work_dtype,
    gaussian_window_1d,
)

__all__ = [
    "iw_ssim",
    "iwssim_lite",
    "IWSSIM_LITE_SCALE_WEIGHTS",
    "IWSSIM_LITE_NUM_SCALES",
    "IW_SSIM_NUM_SCALES",
    "set_compile_enabled",
]

IWSSIM_LITE_SCALE_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
IWSSIM_LITE_NUM_SCALES = 5
IW_SSIM_NUM_SCALES = IWSSIM_LITE_NUM_SCALES


def set_compile_enabled(flag: bool) -> None:
    """Passthrough to :func:`frame_analytics.functional.set_compile_enabled`."""
    _functional.set_compile_enabled(flag)


_scale_weight_cache: dict = {}


@torch.inference_mode(False)
def _scale_weights_tensor(w, device) -> torch.Tensor:
    key = (tuple(w), device)
    t = _scale_weight_cache.get(key)
    if t is None:
        t = torch.tensor(w, dtype=torch.float64, device=device).view(-1, 1, 1)
        _scale_weight_cache[key] = t
    return t


def _pack_planes(x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype, shift: float):
    xs = x.to(dtype) - shift
    ys = y.to(dtype) - shift
    return torch.cat([xs, ys, xs * xs, ys * ys, xs * ys], dim=1)


def _separable_conv(packed: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """Valid separable filter; channels-last depthwise on CPU f32 >= 128x128."""
    n, c, h, w = packed.shape
    k = win.numel()
    if (packed.device.type == "cpu" and packed.dtype == torch.float32 and c > 1
            and h * w >= 128 * 128
            and torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled):
        wh = win.view(1, 1, 1, k).expand(c, 1, 1, k).contiguous()
        wv = win.view(1, 1, k, 1).expand(c, 1, k, 1).contiguous()
        with _no_autocast(packed):
            out = F.conv2d(packed.contiguous(memory_format=torch.channels_last),
                           wh, padding=0, groups=c)
            out = F.conv2d(out, wv, padding=0, groups=c)
        return out.contiguous()
    flat = packed.reshape(n * c, 1, h, w)
    wh = win.view(1, 1, 1, k)
    wv = win.view(1, 1, k, 1)
    with _no_autocast(flat):
        out = F.conv2d(flat, wh)
        out = F.conv2d(out, wv)
    return out.reshape(n, c, h - k + 1, w - k + 1)


def _cast_pool2(x, dtype):
    """avg_pool2d over a dtype conversion, fused into one kernel."""
    return F.avg_pool2d(x.to(dtype), 2)


def _iw_moments(t: torch.Tensor, C: int, shift: float, C1: float, C2: float):
    ux = t[:, 0 * C:1 * C]
    uy = t[:, 1 * C:2 * C]
    mx = ux + shift
    my = uy + shift
    sxx = t[:, 2 * C:3 * C] - ux * ux
    syy = t[:, 3 * C:4 * C] - uy * uy
    sxy = t[:, 4 * C:5 * C] - ux * uy
    cs = (2.0 * sxy + C2) / (sxx + syy + C2)
    lum = (2.0 * mx * my + C1) / (mx * mx + my * my + C1)
    return sxx, lum * cs, cs


def _iw_terms_ratio(t, C, shift, C1, C2, Cw, _unused):
    sxx, ssim, cs = _iw_moments(t, C, shift, C1, C2)
    sxx_c = sxx.clamp_min(0.0)
    w = sxx_c / (sxx_c + Cw)
    return w * ssim, w * cs, w, ssim, cs


def _iw_terms_log(t, C, shift, C1, C2, _unused, sigma_nsq):
    sxx, ssim, cs = _iw_moments(t, C, shift, C1, C2)
    sxx_c = sxx.clamp_min(0.0)
    w = torch.log2(1.0 + sxx_c / sigma_nsq)
    return w * ssim, w * cs, w, ssim, cs


def _iw_terms_plain(t, C, shift, C1, C2, _u1, _u2):
    _, ssim, cs = _iw_moments(t, C, shift, C1, C2)
    return ssim, cs


def _iw_epi_planes(terms_fn, t, C, shift, C1, C2, Cw, sigma_nsq):
    """Per-(image, channel) float64 sums of ``terms_fn``'s maps, ``(N, C)``."""
    return tuple(m.sum(dim=(2, 3), dtype=torch.float64)
                 for m in terms_fn(t, C, shift, C1, C2, Cw, sigma_nsq))


def _iw_epi_rows(terms_fn, t, C, shift, C1, C2, Cw, sigma_nsq):
    """The same sums stopped at rows, ``(N, C, H)`` (few large planes; see
    ``_metric_utils.starved_reduction``); the caller finishes them."""
    return tuple(m.sum(dim=3, dtype=torch.float64)
                 for m in terms_fn(t, C, shift, C1, C2, Cw, sigma_nsq))


_TERMS = {"ratio": _iw_terms_ratio, "log": _iw_terms_log,
          "uniform": _iw_terms_plain}


def _iw_epilogue(mode: str, t: torch.Tensor, C: int, shift: float, C1: float,
                 C2: float, Cw: float, sigma_nsq: float):
    """``(N, C)`` float64 plane sums for weight mode ``mode``.

    ``uniform``: ``(ssim, cs)``; ``ratio``/``log``: ``(w*ssim, w*cs, w, ssim,
    cs)``. Fused into one compiled reduction unless that would starve the GPU.
    """
    terms_fn = _TERMS[mode]
    n, h, w = t.shape[0], t.shape[-2], t.shape[-1]
    if starved_reduction(t, n * C, h * w):
        rows = _maybe_compile(_iw_epi_rows, "iwssim:epi:rows")(
            terms_fn, t, C, shift, C1, C2, Cw, sigma_nsq)
        return tuple(r.sum(dim=-1) for r in rows)
    return _maybe_compile(_iw_epi_planes, "iwssim:epi:planes")(
        terms_fn, t, C, shift, C1, C2, Cw, sigma_nsq)


def _default_weights(num_scales: int):
    base = IWSSIM_LITE_SCALE_WEIGHTS
    if num_scales == len(base):
        return tuple(float(v) for v in base)
    if num_scales == 1:
        return (1.0,)
    s = base[:num_scales]
    tot = sum(s)
    return tuple(float(v) / tot for v in s)


def _center_crop(t: torch.Tensor, hn: int, wn: int) -> torch.Tensor:
    h, w = t.shape[-2], t.shape[-1]
    y0, x0 = (h - hn) // 2, (w - wn) // 2
    return t[..., y0:y0 + hn, x0:x0 + wn]


def _gsm_enlarge(x: torch.Tensor) -> torch.Tensor:
    t1 = F.interpolate(x, size=(int(4 * x.shape[-2] - 3), int(4 * x.shape[-1] - 3)),
                       mode="bilinear", align_corners=False)
    t2 = torch.zeros(x.shape[0], 1, 4 * x.shape[-2] - 1, 4 * x.shape[-1] - 1,
                     device=x.device, dtype=x.dtype).repeat(1, x.shape[1], 1, 1)
    t2[:, :, 1:-1, 1:-1] = t1
    t2[:, :, 0, :] = 2 * t2[:, :, 1, :] - t2[:, :, 2, :]
    t2[:, :, -1, :] = 2 * t2[:, :, -2, :] - t2[:, :, -3, :]
    t2[:, :, :, 0] = 2 * t2[:, :, :, 1] - t2[:, :, :, 2]
    t2[:, :, :, -1] = 2 * t2[:, :, :, -2] - t2[:, :, :, -3]
    return t2[:, :, ::2, ::2]


_binom5_cache: dict = {}


@torch.inference_mode(False)
def _binom5_kernel(device, dtype) -> torch.Tensor:
    """Separable 5-tap binomial ``[1,4,6,4,1]/16 x sqrt(2)``."""
    key = (device, dtype)
    k = _binom5_cache.get(key)
    if k is not None:
        return k
    import math
    b = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float64) / 16.0
    k = (b * math.sqrt(2.0)).to(device=device, dtype=dtype)
    _binom5_cache[key] = k
    return k


def _lap_step(x: torch.Tensor, k: torch.Tensor):
    """One Burt & Adelson level: ``(coarse_lowpass, detail)``."""
    c = x.shape[1]
    kh = k.view(1, 1, 1, 5)
    kv = k.view(1, 1, 5, 1)
    with _no_autocast(x):
        v = F.pad(x, [2, 2, 0, 0], mode="reflect")
        v = F.conv2d(v.reshape(-1, 1, v.shape[-2], v.shape[-1]), kh)
        v = v.reshape(x.shape[0], c, v.shape[-2], v.shape[-1])[:, :, :, ::2]
        v = F.pad(v, [0, 0, 2, 2], mode="reflect")
        v = F.conv2d(v.reshape(-1, 1, v.shape[-2], v.shape[-1]), kv)
        lo = v.reshape(x.shape[0], c, v.shape[-2], v.shape[-1])[:, :, ::2, :]
        up = torch.zeros(x.shape[0] * c, 1, lo.shape[-2] * 2, lo.shape[-1] * 2,
                         device=x.device, dtype=x.dtype)
        up[:, :, ::2, ::2] = lo.reshape(x.shape[0] * c, 1, lo.shape[-2], lo.shape[-1])
        up = F.conv2d(F.pad(up, [2, 2, 0, 0], mode="reflect"), kh)
        up = F.conv2d(F.pad(up, [0, 0, 2, 2], mode="reflect"), kv)
        uh, uw = up.shape[-2], up.shape[-1]
        hi = x - up.reshape(x.shape[0], c, uh, uw)[:, :, :x.shape[-2], :x.shape[-1]]
    return lo.contiguous(), hi.contiguous()


def _align_trim(ssim: torch.Tensor, cs: torch.Tensor, wmap: torch.Tensor,
                win_size: int, blk_size: int):
    """Centre-trim the info map onto the ``valid`` SSIM grid."""
    sh, sw = ssim.shape[-2], ssim.shape[-1]
    wh, ww = wmap.shape[-2], wmap.shape[-1]
    gap_h, gap_w = wh - sh, ww - sw
    want = win_size - blk_size
    if gap_h != want or gap_w != want:
        raise RuntimeError(
            f"map alignment gap {(gap_h, gap_w)} != win-blk {(want, want)} "
            f"(win={win_size} blk={blk_size})")
    trim = want // 2
    if (win_size, blk_size) == (11, 3):
        assert trim == 4, f"trim-4 sentinel moved: {trim}"
    if trim:
        wmap = wmap[..., trim:trim + sh, trim:trim + sw]
    return ssim, cs, wmap


#: Columns per partial Gram product. ``Yt @ Yt^T`` with ``K = H*W`` columns
#: and a 10x10 result leaves cuBLAS one output tile per matrix -- one thread
#: block streaming the whole image (~3.7 ms per 512x512 plane on an RTX
#: 3090). Splitting K into chunks, as a strided batch, and summing the
#: partial Grams spreads the reduction over the device.
_GRAM_CHUNK = 4096


def _gram_planar(Yt: torch.Tensor) -> torch.Tensor:
    """``Yt @ Yt^T`` for planar ``(..., N, K)`` data, accumulated in K-chunks."""
    lead, nn, k = Yt.shape[:-2], Yt.shape[-2], Yt.shape[-1]
    if k < 2 * _GRAM_CHUNK:
        return torch.matmul(Yt, Yt.transpose(-2, -1))
    s = k // _GRAM_CHUNK
    flat = Yt.reshape(-1, nn, k)
    parts = []
    for b in range(flat.shape[0]):
        # (s, N, CHUNK) with batch stride CHUNK: a strided-batched GEMM over
        # the original storage, no copy.
        m = flat[b, :, :s * _GRAM_CHUNK].unflatten(-1, (s, _GRAM_CHUNK)).transpose(0, 1)
        parts.append(torch.bmm(m, m.transpose(-2, -1)).sum(dim=0))
    out = torch.stack(parts).reshape(*lead, nn, nn)
    if k > s * _GRAM_CHUNK:
        rest = Yt[..., s * _GRAM_CHUNK:]
        out = out + torch.matmul(rest, rest.transpose(-2, -1))
    return out


def _gsm_iw_epilogue(Yt, Ci, vv, g, lam, sn: float, eps: float):
    """Quadratic form ``ss = y^T C^-1 y / N`` plus the Eq.28 log2 sum.

    Batch-merged shapes (``B = n*c``): ``Yt`` is planar ``(B, N, h, w)``,
    ``Ci`` is ``(B, N, N)``, ``vv``/``g`` are ``(B, h, w)`` and ``lam`` is
    ``(B, N, 1, 1)``. Merging n and c keeps Dynamo from specialising (and
    recompiling) on n == 1 / c == 1 separately.

    Eager runs keep the matmul formulation (``(Ci^T Yt) * Yt``). Under
    ``torch.compile`` the NxN quadratic form is written out term by term so
    Inductor emits a single pointwise pass over ``Yt`` instead of
    materialising the ``(B, N, h, w)`` product. Never run the unrolled form
    eagerly: it would be ~N^2 tiny kernels.
    """
    N = Yt.shape[1]
    if torch.compiler.is_compiling():
        # Fully unrolled, so the region is purely pointwise: with symbolic
        # sizes Inductor will not fuse a reduction over dim 1 and would
        # materialise a (B, N, h, w) buffer (measured 4x slower, +260 MiB at
        # 1080p). Same per-term operation order as the eager branch.
        ss = None
        for a in range(N):
            inner = None
            for b in range(N):
                t = Ci[:, b, a, None, None] * Yt[:, b]
                inner = t if inner is None else inner + t
            t = inner * Yt[:, a] / N
            ss = t if ss is None else ss + t
        base = (vv + (1.0 + g * g) * sn) * ss
        noise = sn * vv
        iw = None
        for k in range(N):
            t = torch.log2(1.0 + (base * lam[:, k] + noise) / (sn * sn))
            iw = t if iw is None else iw + t
    else:
        YC = torch.matmul(Ci.transpose(-2, -1), Yt.flatten(-2)).view_as(Yt)
        ss = (YC * Yt / N).sum(dim=1, keepdim=True)
        vv1 = vv.unsqueeze(1)
        g1 = g.unsqueeze(1)
        iw = torch.sum(torch.log2(1.0 + ((vv1 + (1.0 + g1 * g1) * sn) * ss * lam
                                         + sn * vv1) / (sn * sn)), dim=1)
    iw = torch.where(iw < eps, torch.zeros_like(iw), iw)
    return iw.clamp_min(0.0)


def _gsm_info_map(x: torch.Tensor, y: torch.Tensor, x_parent=None,
                  blk_size: int = 3, sigma_nsq: float = 0.4) -> torch.Tensor:
    """Full GSM Eq.28 information-content map (x = reference convention)."""
    n, c = x.shape[0], x.shape[1]
    dt = x.dtype
    dev = x.device
    eps = torch.finfo(dt).eps
    k = torch.full((c, 1, blk_size, blk_size), 1.0 / (blk_size * blk_size),
                   device=dev, dtype=dt)
    pu, pd = blk_size // 2, blk_size - blk_size // 2
    xp = F.pad(x, [pu, pd, pu, pd])
    yp = F.pad(y, [pu, pd, pu, pd])
    with _no_autocast(xp):
        mu_x = F.conv2d(xp, k, groups=c)
        mu_y = F.conv2d(yp, k, groups=c)
        sxx = F.conv2d(F.pad(x * x, [pu, pd, pu, pd]), k, groups=c) - mu_x * mu_x
        syy = F.conv2d(F.pad(y * y, [pu, pd, pu, pd]), k, groups=c) - mu_y * mu_y
        sxy = F.conv2d(F.pad(x * y, [pu, pd, pu, pd]), k, groups=c) - mu_x * mu_y
    sxx = sxx.clamp_min(0.0)
    syy = syy.clamp_min(0.0)

    no_x = sxx < eps
    no_y = syy < eps
    g = sxy / (sxx + eps)
    vv = syy - g * sxy
    g = torch.where(no_x | no_y, torch.zeros_like(g), g)
    vv = torch.where(no_x, syy, vv)
    vv = torch.where(no_y, torch.zeros_like(vv), vv)
    vv = vv.clamp_min(0.0)

    Ly, Lx = (blk_size - 1) // 2, (blk_size - 1) // 2
    nblv, nblh = x.shape[-2] - blk_size + 1, x.shape[-1] - blk_size + 1
    nexp = nblv * nblh
    N = blk_size * blk_size
    up = None
    if x_parent is not None:
        up = _gsm_enlarge(x_parent)[:, :, :x.shape[-2], :x.shape[-1]]
        N = N + 1
    # Neighbourhood vectors, planar: plane m is the image shifted by
    # (ny, nx) and cropped to the valid block grid. The shift never wraps
    # (|ny| <= Ly and the crop starts Ly in), so each plane is a plain slice
    # of ``x`` -- the same values a circular shift + crop produces. Planar
    # (N, K) storage keeps every copy and reduction below coalesced. Image
    # and channel are merged into one batch dim B from here on.
    B = n * c
    xb = x.flatten(0, 1)
    cols = [xb[:, Ly - ny:Ly - ny + nblv, Lx - nx:Lx - nx + nblh]
            for ny in range(-Ly, Ly + 1) for nx in range(-Lx, Lx + 1)]
    if up is not None:
        cols.append(up[:, :, Ly:Ly + nblv, Lx:Lx + nblh].flatten(0, 1))
    Yt = torch.stack(cols, dim=1)                      # (B, N, nblv, nblh)
    del cols

    Cu = _gram_planar(Yt.view(B, N, nexp)) / max(nexp, 1)
    eig_values, eig_vectors = torch.linalg.eigh(Cu)
    sum_eig = eig_values.sum(dim=-1).view(B, 1, 1)
    nz = torch.diag_embed(eig_values * (eig_values > 0))
    sum_nz = nz.sum(dim=(-2, -1), keepdim=True)
    L = nz * sum_eig / (sum_nz + (sum_nz == 0))
    Cu = torch.matmul(torch.matmul(eig_vectors, L), eig_vectors.transpose(-2, -1))
    Cu_inv = torch.linalg.pinv(Cu)
    lam = torch.diagonal(L, offset=0, dim1=-2, dim2=-1).reshape(B, N, 1, 1)
    g = _own(g[:, :, Ly:Ly + nblv, Lx:Lx + nblh].reshape(B, nblv, nblh))
    vv = _own(vv[:, :, Ly:Ly + nblv, Lx:Lx + nblh].reshape(B, nblv, nblh))
    grad = torch.is_grad_enabled() and (Yt.requires_grad or vv.requires_grad
                                        or g.requires_grad)
    iw = _gsm_epilogue_for(N, Yt, grad)(_own(Yt), _own(Cu_inv), vv, g, _own(lam),
             float(sigma_nsq), float(eps))
    return iw.view(n, c, nblv, nblh)


#: Largest neighbourhood (``blk_size**2`` + parent) given the unrolled,
#: compiled epilogue. Its trace is N^2 terms: N = 10 (the default 3x3 block
#: with parent) compiles in seconds, N = 26 (5x5) took ~100 s and N = 50 (7x7)
#: ~300 s for a gain the eager matmul form mostly matches at those sizes.
_GSM_UNROLL_MAX_N = 10
_gsm_epi_variants: dict = {}


def _gsm_epilogue_for(N: int, Yt: torch.Tensor, grad: bool):
    """The epilogue callable for this (N, grad mode, dtype, device).

    Each combination compiles its *own copy* of the function. Dynamo keys its
    cache -- and its recompile limit of 8 -- on the code object, so a single
    compiled function shared by N = 9 / 10, train / eval, float32 / float64
    and CPU / CUDA (each a guard failure) ran out of entries on an ordinary
    mixed workload and silently dropped every later variant to eager. A copy
    is left with only the B == 1 and square-vs-not specialisations: at most
    four entries.
    """
    if N > _GSM_UNROLL_MAX_N:
        return _gsm_iw_epilogue           # eager: the matmul formulation
    key = (N, grad, Yt.dtype, Yt.device.type)
    epi = _gsm_epi_variants.get(key)
    if epi is None:
        src = _gsm_iw_epilogue
        name = f"_gsm_iw_epilogue_n{N}_{'g' if grad else 'i'}_{str(Yt.dtype)[6:]}_{Yt.device.type}"
        code = src.__code__.replace(co_name=name)
        if hasattr(code, "co_qualname"):
            code = code.replace(co_qualname=name)
        clone = types.FunctionType(code, src.__globals__, name,
                                   src.__defaults__, src.__closure__)
        # dynamic=True plus non-view, batch-merged inputs: one graph whatever
        # the image size or batch.
        epi = _maybe_compile(clone, "iwssim:gsm_iw:" + name, dynamic=True)
        _gsm_epi_variants[key] = epi
    return epi


def _own(t: torch.Tensor) -> torch.Tensor:
    """``t`` as a non-view tensor (Dynamo guards view inputs via ``_base``)."""
    return t if t._base is None else t.clone()


def iw_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: Optional[float] = None,
    win_size: int = 11,
    sigma: float = 1.5,
    K: Sequence[float] = (0.01, 0.03),
    num_scales: int = IWSSIM_LITE_NUM_SCALES,
    scale_weights: Optional[Sequence[float]] = None,
    weight_mode: str = "gsm",
    weight_C: Optional[float] = None,
    sigma_nsq: float = 0.4,
    blk_size: int = 3,
    parent: bool = True,
    uniform_weights: bool = False,
    reduction: str = "mean",
    dtype: Optional[torch.dtype] = None,
    luma=None,
    crop_border: int = 0,
) -> torch.Tensor:
    """Information-weighted SSIM (Wang & Li 2011), full GSM Eq.28 default."""
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    if uniform_weights:
        weight_mode = "uniform"
    if weight_mode not in ("ratio", "log", "uniform", "gsm"):
        raise ValueError(
            f"weight_mode must be 'ratio'/'log'/'uniform'/'gsm', got {weight_mode!r}")
    if num_scales < 1:
        raise ValueError(f"num_scales must be >= 1, got {num_scales}")
    if blk_size < 1 or blk_size % 2 != 1:
        raise ValueError(f"blk_size must be odd >= 1, got {blk_size}")
    if scale_weights is None:
        w = _default_weights(num_scales)
    else:
        w = tuple(float(v) for v in scale_weights)
    if len(w) != num_scales:
        raise ValueError(f"num_scales={num_scales} disagrees with len(scale_weights)={len(w)}")
    if not w:
        raise ValueError("scale_weights must not be empty")

    x4, y4, L, mode = _prep(x, y, luma, crop_border, data_range)
    wdt = _work_dtype(x4, dtype)
    x4, y4 = _apply_luma(x4, y4, mode, L, wdt)
    K1, K2 = float(K[0]), float(K[1])
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    Cw = float(weight_C) if weight_C is not None else C2
    if Cw <= 0:
        raise ValueError(f"weight_C must be > 0, got {Cw}")
    snsq = float(sigma_nsq) * (L / 255.0) ** 2
    if snsq <= 0:
        raise ValueError(f"sigma_nsq must be > 0, got {sigma_nsq}")
    shift = 0.5 * L

    need = (win_size - 1) * (1 << (num_scales - 1)) + (1 << (num_scales - 1))
    if min(x4.shape[-2], x4.shape[-1]) < need:
        raise ValueError(
            f"image {tuple(x4.shape[-2:])} too small for {num_scales} scales with a "
            f"{win_size}x{win_size} window; needs at least {need} px on a side"
        )

    win = gaussian_window_1d(win_size, sigma, device=x4.device, dtype=wdt)
    pack = _maybe_compile(_pack_planes, "iwssim:pack")
    pool = _maybe_compile(_cast_pool2, "iwssim:cast_pool2")

    terms = []
    if weight_mode == "gsm":
        k5 = _binom5_kernel(x4.device, wdt)
        rx, ry = x4.to(wdt), y4.to(wdt)
        lo_x, dx_old = _lap_step(rx, k5)
        lo_y, dy_old = _lap_step(ry, k5)
        cx, cy = lo_x, lo_y
        for i in range(num_scales):
            if i < num_scales - 2:
                lo_x, dx = _lap_step(cx, k5)
                lo_y, dy = _lap_step(cy, k5)
                cx, cy = lo_x, lo_y
            else:
                dx, dy = cx, cy
            filt = _separable_conv(pack(dx_old, dy_old, wdt, shift), win.to(wdt))
            C = dx_old.shape[1]
            ux = filt[:, 0 * C:1 * C]
            uy = filt[:, 1 * C:2 * C]
            mx = ux + shift
            my = uy + shift
            sxx = filt[:, 2 * C:3 * C] - ux * ux
            syy = filt[:, 3 * C:4 * C] - uy * uy
            sxy = filt[:, 4 * C:5 * C] - ux * uy
            cs = (2.0 * sxy + C2) / (sxx + syy + C2)
            lum = (2.0 * mx * my + C1) / (mx * mx + my * my + C1)
            ssim = lum * cs
            if parent and i < num_scales - 2:
                wmap = _gsm_info_map(dx_old, dy_old, dx,
                                     blk_size=blk_size, sigma_nsq=snsq)
                _, _, w_c = _align_trim(ssim, cs, wmap.to(ssim.dtype),
                                        win_size, blk_size)
                hn, wn = w_c.shape[-2], w_c.shape[-1]
                tgt = _center_crop(cs, hn, wn)
            elif i == num_scales - 1:
                tgt = ssim
                w_c = torch.ones_like(ssim)
                hn, wn = tgt.shape[-2], tgt.shape[-1]
            else:
                wmap = _gsm_info_map(dx_old, dy_old, None,
                                     blk_size=blk_size, sigma_nsq=snsq)
                _, _, w_c = _align_trim(ssim, cs, wmap.to(ssim.dtype),
                                        win_size, blk_size)
                hn, wn = w_c.shape[-2], w_c.shape[-1]
                tgt = _center_crop(cs, hn, wn)
            num = (w_c * tgt).sum(dim=(2, 3), dtype=torch.float64)
            den = w_c.sum(dim=(2, 3), dtype=torch.float64)
            plain = tgt.sum(dim=(2, 3), dtype=torch.float64) / (hn * wn)
            score = torch.where(den > 0, num / den.clamp_min(1e-300), plain)
            terms.append(score)
            dx_old, dy_old = dx, dy
    else:
        cx, cy = x4, y4
        for i in range(num_scales):
            if i:
                if cx.dtype != wdt:
                    cx, cy = pool(cx, wdt), pool(cy, wdt)
                else:
                    cx, cy = F.avg_pool2d(cx, 2), F.avg_pool2d(cy, 2)
            filt = _separable_conv(pack(cx, cy, wdt, shift), win.to(wdt))
            npx = filt.shape[-2] * filt.shape[-1]
            if weight_mode == "uniform":
                s_sum, cs_sum = _iw_epilogue(weight_mode, filt, cx.shape[1], shift,
                                             C1, C2, Cw, snsq)
                score = (s_sum if i == num_scales - 1 else cs_sum) / npx
            else:
                ws_s, ws_c, w_sum, s_sum, cs_sum = _iw_epilogue(
                    weight_mode, filt, cx.shape[1], shift, C1, C2, Cw, snsq)
                plain = (s_sum if i == num_scales - 1 else cs_sum) / npx
                wtd = (ws_s if i == num_scales - 1 else ws_c) / w_sum.clamp_min(1e-300)
                score = torch.where(w_sum > 0, wtd, plain)
            terms.append(score)

    if len(terms) == 1:
        per_image = terms[0].mean(dim=1)
        return per_image if reduction == "none" else per_image.mean()

    stacked = torch.stack(terms).clamp_min(0.0)
    per_plane = stacked.pow(_scale_weights_tensor(w, x4.device)).prod(dim=0)
    per_image = per_plane.mean(dim=1)
    return per_image if reduction == "none" else per_image.mean()


def iwssim_lite(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible alias of :func:`iw_ssim` (first-draft name)."""
    return iw_ssim(*args, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.rand(2, 3, 256, 256)
    b = (a + 0.05 * torch.randn_like(a)).clamp(0, 1)
    v = iw_ssim(a, b, data_range=1.0)
    print(f"iw_ssim={float(v):.6f}")
    assert torch.isfinite(v) and 0.0 < float(v) <= 1.0
    per = iw_ssim(a, b, data_range=1.0, reduction="none")
    assert per.shape == (2,)
    print("iw_ssim self-test: OK")
