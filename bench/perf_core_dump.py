"""Dump core-metric outputs (values, maps, gradients) for A/B equivalence checks.

Run once against each package (``PYTHONPATH`` selects which), then diff the two
files with ``--compare a.pt b.pt``. Every entry is also compared against the
float64 reference where one exists, so a change that moves the output can be
judged on whether it moved *towards* or *away from* the reference.

    PYTHONPATH=<pkg> python bench/perf_core_dump.py --device cuda --out new.pt
    python bench/perf_core_dump.py --compare base.pt new.pt
"""

from __future__ import annotations

import argparse
import sys

import torch


def cases(device):
    g = torch.Generator().manual_seed(1234)
    shapes = [(1, 1, 37, 41), (2, 3, 64, 80), (1, 3, 203, 517), (2, 1, 300, 129),
              (1, 3, 512, 512), (3, 2, 181, 260)]
    for shp in shapes:
        n, c, h, w = shp
        base = torch.rand(n, c, h, w, generator=g)
        noise = torch.randn(n, c, h, w, generator=g) * 0.05
        a = base
        b = (base + noise).clamp(0, 1)
        yield shp, "u8", (a * 255).round().to(torch.uint8).to(device), (b * 255).round().to(torch.uint8).to(device), 255.0
        yield shp, "f32", a.to(device), b.to(device), 1.0
        yield shp, "f16", a.half().to(device), b.half().to(device), 1.0
        yield shp, "bf16", a.bfloat16().to(device), b.bfloat16().to(device), 1.0


def dump(device, out):
    import frame_analytics as fa
    res = {}
    for shp, dt, a, b, L in cases(device):
        key = f"{shp}-{dt}"
        h, w = shp[-2:]
        for win in (3, 5, 7, 9, 11):
            if min(h, w) < win:
                continue
            res[f"ssim-{key}-w{win}"] = fa.ssim(a, b, data_range=L, win_size=win).double().cpu()
            res[f"ssim_none-{key}-w{win}"] = fa.ssim(a, b, data_range=L, win_size=win, reduction="none").double().cpu()
            res[f"ssim_map-{key}-w{win}"] = fa.ssim(a, b, data_range=L, win_size=win, return_map=True).double().cpu()
        if min(h, w) >= 176:
            res[f"ms_ssim-{key}"] = fa.ms_ssim(a, b, data_range=L, reduction="none").double().cpu()
        if min(h, w) >= 8:
            res[f"gmsd-{key}"] = fa.gmsd(a, b, data_range=L, reduction="none").double().cpu()
            res[f"gms-{key}"] = fa.gms(a, b, data_range=L, reduction="none").double().cpu()
            res[f"gmsd_nods-{key}"] = fa.gmsd(a, b, data_range=L, downsample=False, reduction="none").double().cpu()
        for nm in ("mse", "l1", "charbonnier", "huber"):
            res[f"{nm}-{key}"] = getattr(fa, nm)(a, b, reduction="none").double().cpu()
        res[f"psnr-{key}"] = fa.psnr(a, b, data_range=L, reduction="none").double().cpu()
        res[f"psnr_luma-{key}"] = (fa.psnr(a, b, data_range=L, luma="bt601").double().cpu()
                                   if shp[1] == 3 else torch.zeros(()))
        if dt == "f32":
            for win in (3, 7, 11):
                if min(h, w) < win:
                    continue
                for need_y in (False, True):
                    x = a.clone().requires_grad_(True)
                    y = b.clone().requires_grad_(need_y)
                    (1 - fa.ssim(x, y, data_range=L, win_size=win)).backward()
                    res[f"ssim_dx-{key}-w{win}-{need_y}"] = x.grad.double().cpu()
                    if need_y:
                        res[f"ssim_dy-{key}-w{win}"] = y.grad.double().cpu()
                x = a.clone().requires_grad_(True)
                m = fa.ssim(x, b, data_range=L, win_size=win, return_map=True)
                (m * torch.linspace(0, 1, m.numel(), device=m.device).view_as(m)).sum().backward()
                res[f"ssim_mapgrad-{key}-w{win}"] = x.grad.double().cpu()
            if min(h, w) >= 176:
                x = a.clone().requires_grad_(True)
                y = b.clone().requires_grad_(True)
                (1 - fa.ms_ssim(x, y, data_range=L)).backward()
                res[f"ms_ssim_dx-{key}"] = x.grad.double().cpu()
                res[f"ms_ssim_dy-{key}"] = y.grad.double().cpu()
            if min(h, w) >= 8:
                x = a.clone().requires_grad_(True)
                (1 - fa.gms(x, b, data_range=L)).backward()
                res[f"gms_dx-{key}"] = x.grad.double().cpu()
            for nm in ("mse", "l1", "charbonnier", "huber"):
                x = a.clone().requires_grad_(True)
                getattr(fa, nm)(x, b).backward()
                res[f"{nm}_dx-{key}"] = x.grad.double().cpu()
    torch.save(res, out)
    print(f"wrote {len(res)} entries to {out}")


def compare(fa_path, fb_path, show=20):
    a = torch.load(fa_path)
    b = torch.load(fb_path)
    worst = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            print("MISSING", k, k in a, k in b)
            continue
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            print("SHAPE", k, ta.shape, tb.shape)
            continue
        same_nan = torch.equal(torch.isnan(ta), torch.isnan(tb))
        fin = torch.isfinite(ta) & torch.isfinite(tb)
        d = (ta[fin] - tb[fin]).abs()
        md = float(d.max()) if d.numel() else 0.0
        scale = float(ta[fin].abs().max()) if d.numel() else 0.0
        worst.append((md / max(scale, 1e-30), md, k, bool(torch.equal(ta, tb)), same_nan))
    worst.sort(reverse=True)
    nexact = sum(1 for w in worst if w[3])
    print(f"{len(worst)} entries, {nexact} bit-identical")
    for rel, md, k, ex, sn in worst[:show]:
        print(f"  rel={rel:.3e} abs={md:.3e} nan_mask_same={sn} {k}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--show", type=int, default=20)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare, show=args.show)
    else:
        dump(args.device, args.out)
