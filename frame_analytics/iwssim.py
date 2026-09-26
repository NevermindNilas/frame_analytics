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

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from . import functional as _functional
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


def _iw_epi_ratio(t: torch.Tensor, C: int, shift: float, C1: float, C2: float,
                  Cw: float, _unused: float):
    ux = t[:, 0 * C:1 * C]
    uy = t[:, 1 * C:2 * C]
    mx = ux + shift
    my = uy + shift
    sxx = t[:, 2 * C:3 * C] - ux * ux
    syy = t[:, 3 * C:4 * C] - uy * uy
    sxy = t[:, 4 * C:5 * C] - ux * uy
    cs = (2.0 * sxy + C2) / (sxx + syy + C2)
    lum = (2.0 * mx * my + C1) / (mx * mx + my * my + C1)
    ssim = lum * cs
    sxx_c = sxx.clamp_min(0.0)
    w = sxx_c / (sxx_c + Cw)
    return (
        (w * ssim).sum(dim=(2, 3), dtype=torch.float64),
        (w * cs).sum(dim=(2, 3), dtype=torch.float64),
        w.sum(dim=(2, 3), dtype=torch.float64),
        ssim.sum(dim=(2, 3), dtype=torch.float64),
        cs.sum(dim=(2, 3), dtype=torch.float64),
    )


def _iw_epi_log(t: torch.Tensor, C: int, shift: float, C1: float, C2: float,
                _unused: float, sigma_nsq: float):
    ux = t[:, 0 * C:1 * C]
    uy = t[:, 1 * C:2 * C]
    mx = ux + shift
    my = uy + shift
    sxx = t[:, 2 * C:3 * C] - ux * ux
    syy = t[:, 3 * C:4 * C] - uy * uy
    sxy = t[:, 4 * C:5 * C] - ux * uy
    cs = (2.0 * sxy + C2) / (sxx + syy + C2)
    lum = (2.0 * mx * my + C1) / (mx * mx + my * my + C1)
    ssim = lum * cs
    sxx_c = sxx.clamp_min(0.0)
    w = torch.log2(1.0 + sxx_c / sigma_nsq)
    return (
        (w * ssim).sum(dim=(2, 3), dtype=torch.float64),
        (w * cs).sum(dim=(2, 3), dtype=torch.float64),
        w.sum(dim=(2, 3), dtype=torch.float64),
        ssim.sum(dim=(2, 3), dtype=torch.float64),
        cs.sum(dim=(2, 3), dtype=torch.float64),
    )


def _iw_epi_plain(t: torch.Tensor, C: int, shift: float, C1: float, C2: float,
                  _u1: float, _u2: float):
    ux = t[:, 0 * C:1 * C]
    uy = t[:, 1 * C:2 * C]
    mx = ux + shift
    my = uy + shift
    sxx = t[:, 2 * C:3 * C] - ux * ux
    syy = t[:, 3 * C:4 * C] - uy * uy
    sxy = t[:, 4 * C:5 * C] - ux * uy
    cs = (2.0 * sxy + C2) / (sxx + syy + C2)
    lum = (2.0 * mx * my + C1) / (mx * mx + my * my + C1)
    ssim = lum * cs
    return (
        ssim.sum(dim=(2, 3), dtype=torch.float64),
        cs.sum(dim=(2, 3), dtype=torch.float64),
        None, None, None,
    )


_EPI = {"ratio": (_iw_epi_ratio, "iwssim:epi:ratio"),
        "log": (_iw_epi_log, "iwssim:epi:log"),
        "uniform": (_iw_epi_plain, "iwssim:epi:plain")}


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


def _gsm_shift(x: torch.Tensor, shift) -> torch.Tensor:
    ny, nx = shift
    h, w = x.shape[-2], x.shape[-1]
    yy = (-ny) % h if h else 0
    xx = (-nx) % w if w else 0
    v = torch.cat((x[..., yy:, :], x[..., :yy, :]), dim=-2) if yy else x
    v = torch.cat((v[..., xx:], v[..., :xx]), dim=-1) if xx else v
    return v


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
    Y = torch.zeros(n, c, nexp, N, device=dev, dtype=dt)
    m = -1
    for ny in range(-Ly, Ly + 1):
        for nx in range(-Lx, Lx + 1):
            m += 1
            foo = _gsm_shift(x, [ny, nx])[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
            Y[..., m] = foo.flatten(start_dim=-2, end_dim=-1)
    if up is not None:
        m += 1
        foo = up[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
        Y[..., m] = foo.flatten(start_dim=-2, end_dim=-1)

    Cu = torch.matmul(Y.transpose(-2, -1), Y) / max(nexp, 1)
    eig_values, eig_vectors = torch.linalg.eigh(Cu)
    sum_eig = eig_values.sum(dim=-1).view(n, c, 1, 1)
    nz = torch.diag_embed(eig_values * (eig_values > 0))
    sum_nz = nz.sum(dim=(-2, -1), keepdim=True)
    L = nz * sum_eig / (sum_nz + (sum_nz == 0))
    Cu = torch.matmul(torch.matmul(eig_vectors, L), eig_vectors.transpose(-2, -1))
    Cu_inv = torch.linalg.pinv(Cu)
    ss = torch.matmul(Y, Cu_inv) * Y / N
    ss = ss.sum(dim=-1, keepdim=True).view(n, c, nblv, nblh)
    g = g[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
    vv = vv[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
    lam = torch.diagonal(L, offset=0, dim1=-2, dim2=-1).unsqueeze(2).unsqueeze(3)
    sn = float(sigma_nsq)
    iw = torch.sum(torch.log2(1.0 + ((vv.unsqueeze(-1) + (1.0 + g.unsqueeze(-1)
            * g.unsqueeze(-1)) * sn) * ss.unsqueeze(-1) * lam
            + sn * vv.unsqueeze(-1)) / (sn * sn)), dim=-1)
    iw = torch.where(iw < eps, torch.zeros_like(iw), iw)
    return iw.clamp_min(0.0)


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
        epi_fn, epi_key = _EPI[weight_mode]
        epi = _maybe_compile(epi_fn, epi_key)
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
                s_sum, cs_sum, _, _, _ = epi(filt, cx.shape[1], shift, C1, C2, Cw, snsq)
                score = (s_sum if i == num_scales - 1 else cs_sum) / npx
            else:
                ws_s, ws_c, w_sum, s_sum, cs_sum = epi(
                    filt, cx.shape[1], shift, C1, C2, Cw, snsq)
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
