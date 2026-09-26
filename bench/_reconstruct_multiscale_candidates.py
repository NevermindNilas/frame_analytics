"""Temporary, exact reconstruction of the four previously rejected patches."""

from pathlib import Path
from tempfile import gettempdir


SOURCE = Path(__file__).resolve().parents[1] / "frame_analytics"
DEST = Path(gettempdir()) / "fa_multiscale_candidates_20260923"
DEST.mkdir(exist_ok=True)


def change(text: str, before: str, after: str) -> str:
    assert text.count(before) == 1, before[:60]
    return text.replace(before, after, 1)


def write(name: str, patches: list[tuple[str, str]]) -> None:
    text = (SOURCE / name).read_text(encoding="utf-8")
    for before, after in patches:
        text = change(text, before, after)
    (DEST / name).write_text(text, encoding="utf-8")


write("vifp.py", [(
    """        for ch in range(cc):
            chunk = torch.stack(
                (filt[:, ch], filt[:, cc + ch], filt[:, 2 * cc + ch],
                 filt[:, 3 * cc + ch], filt[:, 4 * cc + ch]), dim=1)
            nsc, dsc = epi(chunk, sn, e, rel)
            del chunk
            num = num + nsc
            den = den + dsc
""",
    """        # Batch the independent color planes into the epilogue. This keeps
        # its per-plane float64 reductions but avoids C compiled calls and
        # C separate five-plane stacks at each pyramid scale.
        chunk = (filt.reshape(nn, 5, cc, *filt.shape[-2:])
                 .permute(0, 2, 1, 3, 4)
                 .reshape(nn * cc, 5, *filt.shape[-2:]))
        nsc, dsc = epi(chunk, sn, e, rel)
        num = num + nsc.reshape(nn, cc).sum(dim=1)
        den = den + dsc.reshape(nn, cc).sum(dim=1)
        del chunk
""",
)])

write("nlpd.py", [(
    """    cur = torch.cat([xf, yf], dim=0)
    lap = _lap_filter(cur.device, wdt, c)
""",
    """    cur = torch.cat([xf, yf], dim=0)
    if cur.device.type == "cpu" and c > 1:
        cur = cur.to(memory_format=torch.channels_last)
    lap = _lap_filter(cur.device, wdt, c)
""",
)])

write("iwssim.py", [(
    """        lo_x, dx_old = _lap_step(rx, k5)
        lo_y, dy_old = _lap_step(ry, k5)
        cx, cy = lo_x, lo_y
        for i in range(num_scales):
            if i < num_scales - 2:
                lo_x, dx = _lap_step(cx, k5)
                lo_y, dy = _lap_step(cy, k5)
                cx, cy = lo_x, lo_y
            else:
                dx, dy = cx, cy
""",
    """        n = rx.shape[0]
        cur, detail = _lap_step(torch.cat([rx, ry], dim=0), k5)
        dx_old, dy_old = detail[:n], detail[n:]
        for i in range(num_scales):
            if i < num_scales - 2:
                cur, detail = _lap_step(cur, k5)
                dx, dy = detail[:n], detail[n:]
            else:
                dx, dy = cur[:n], cur[n:]
""",
)])

write("ms_gmsd.py", [(
    """def _gms_grads(xf: torch.Tensor, yf: torch.Tensor) -> torch.Tensor:
    \"\"\"Stacked Prewitt responses of both images, ``(2*N*C, 2, H-2, W-2)``.\"\"\"
    n, c, h, w = xf.shape
    if h < 3 or w < 3:
        raise ValueError(f\"image {h}x{w} is smaller than the 3x3 Prewitt window\")
    k = _prewitt_pair(xf.device, xf.dtype)
    both = torch.cat([xf, yf], dim=0).reshape(2 * n * c, 1, h, w)
    with _no_autocast(both):
        return F.conv2d(both, k)
""",
    """def _gms_grads(both: torch.Tensor) -> torch.Tensor:
    \"\"\"Stacked Prewitt responses of both images, ``(2*N*C, 2, H-2, W-2)``.\"\"\"
    n, c, h, w = both.shape
    if h < 3 or w < 3:
        raise ValueError(f\"image {h}x{w} is smaller than the 3x3 Prewitt window\")
    k = _prewitt_pair(both.device, both.dtype)
    flat = both.reshape(n * c, 1, h, w)
    with _no_autocast(flat):
        return F.conv2d(flat, k)
""",
), (
    """    if downsample:
        if x4.dtype != wdt:
            cx, cy = pool(x4, wdt), pool(y4, wdt)
        else:
            cx, cy = F.avg_pool2d(x4.to(wdt), 2), F.avg_pool2d(y4.to(wdt), 2)
    else:
        cx, cy = x4.to(wdt), y4.to(wdt)
    if cx.device.type == \"cpu\" and cx.shape[1] > 1:
        cx = cx.to(memory_format=torch.channels_last)
        cy = cy.to(memory_format=torch.channels_last)
""",
    """    both = torch.cat([x4, y4], dim=0)
    if downsample:
        both = pool(both, wdt) if both.dtype != wdt else F.avg_pool2d(both.to(wdt), 2)
    else:
        both = both.to(wdt)
    if both.device.type == \"cpu\" and both.shape[1] > 1:
        both = both.to(memory_format=torch.channels_last)
""",
), (
    """    per_scale = []
    for i in range(num_scales):
        if i:
            cx, cy = F.avg_pool2d(cx, 2), F.avg_pool2d(cy, 2)
        g = _gms_grads(cx, cy)
""",
    """    per_scale = []
    for i in range(num_scales):
        if i:
            both = F.avg_pool2d(both, 2)
        g = _gms_grads(both)
""",
)])

REFINED = Path(gettempdir()) / "fa_multiscale_refined_20260923"
REFINED.mkdir(exist_ok=True)
refined = (DEST / "ms_gmsd.py").read_text(encoding="utf-8")
refined = change(refined,
    """    both = torch.cat([x4, y4], dim=0)
    if downsample:
        both = pool(both, wdt) if both.dtype != wdt else F.avg_pool2d(both.to(wdt), 2)
    else:
        both = both.to(wdt)
""",
    """    if downsample:
        if x4.dtype != wdt:
            cx, cy = pool(x4, wdt), pool(y4, wdt)
        else:
            cx, cy = F.avg_pool2d(x4.to(wdt), 2), F.avg_pool2d(y4.to(wdt), 2)
        both = torch.cat([cx, cy], dim=0)
        del cx, cy
    else:
        both = torch.cat([x4.to(wdt), y4.to(wdt)], dim=0)
""")
(REFINED / "ms_gmsd.py").write_text(refined, encoding="utf-8")

print(DEST)
print(REFINED)
