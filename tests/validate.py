"""Accuracy gate.

Every backend is checked against `frame_analytics.reference` (float64,
literal ssim_index.m) and against scikit-image's independently written
implementation. Nothing ships unless it is inside the tolerances printed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_analytics as fa
from frame_analytics import reference as ref

# tolerance on the *scalar* SSIM/PSNR value. Both are means over >=10^5
# samples, so this is far tighter than the ~1e-3 spread the paper's own
# subjective-score fits operate at.
TOL_SSIM = 5e-6
TOL_MAP = 5e-5
TOL_PSNR = 1e-6

RNG = np.random.default_rng(0xC0FFEE)


def make_pair(h, w, kind="natural", noise=8.0, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "natural":
        # low-frequency base + texture: behaves like a photograph for the
        # local-variance terms, unlike uniform noise
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        base = (
            110
            + 60 * np.sin(xx / 37.0) * np.cos(yy / 53.0)
            + 40 * np.sin((xx + yy) / 17.0)
            + 25 * rng.standard_normal((h, w))
        )
        base = np.clip(base, 0, 255)
    elif kind == "uniform":
        base = rng.integers(0, 256, (h, w)).astype(np.float64)
    elif kind == "flat":
        base = np.full((h, w), 128.0)
    else:
        raise ValueError(kind)
    dist = np.clip(base + noise * rng.standard_normal((h, w)), 0, 255)
    return base.round(), dist.round()


def report(name, got, want, tol):
    err = abs(got - want)
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} got={got:.12f} "
          f"ref={want:.12f} |err|={err:.3e} tol={tol:.0e}")
    return ok


def main():
    print("backend:", fa.backend_status())
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    ok = True

    cases = [
        (64, 64, "uniform", 20.0),
        (127, 233, "natural", 8.0),
        (256, 256, "natural", 3.0),
        (512, 384, "natural", 25.0),
        (720, 1280, "natural", 12.0),
        (33, 41, "flat", 5.0),
    ]

    for h, w, kind, noise in cases:
        a, b = make_pair(h, w, kind, noise, seed=h * w)
        ssim_ref, map_ref = ref.ssim_reference(a, b, 255.0, return_map=True)
        mse_ref = ref.mse_reference(a, b)
        psnr_ref = ref.psnr_reference(a, b, 255.0)

        # scikit-image, configured to match the paper exactly
        try:
            from skimage.metrics import structural_similarity
            sk = structural_similarity(a, b, data_range=255.0, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False)
            ok &= report(f"[{h}x{w} {kind}] skimage vs reference", sk, ssim_ref, TOL_SSIM)
        except Exception as exc:
            print("  skip skimage:", exc)

        for dev in devices:
            au8 = torch.as_tensor(a, dtype=torch.uint8, device=dev)
            bu8 = torch.as_tensor(b, dtype=torch.uint8, device=dev)
            af = torch.as_tensor(a, dtype=torch.float32, device=dev)
            bf = torch.as_tensor(b, dtype=torch.float32, device=dev)

            for hint in ("torch", "auto"):
                tag = f"[{h}x{w} {kind}] {dev}/{hint}"
                v = float(fa.ssim(au8, bu8, data_range=255.0, backend_hint=hint))
                ok &= report(f"{tag} ssim uint8", v, ssim_ref, TOL_SSIM)
                v = float(fa.ssim(af, bf, data_range=255.0, backend_hint=hint))
                ok &= report(f"{tag} ssim fp32", v, ssim_ref, TOL_SSIM)

                m = fa.ssim(au8, bu8, data_range=255.0, return_map=True,
                            backend_hint=hint)
                merr = float((m.squeeze().double().cpu() -
                              torch.from_numpy(map_ref)).abs().max())
                mok = merr <= TOL_MAP
                ok &= mok
                print(f"  {'PASS' if mok else 'FAIL'}  {tag} ssim map "
                      f"max|err|={merr:.3e} tol={TOL_MAP:.0e}")

            ok &= report(f"[{h}x{w} {kind}] {dev} mse uint8",
                         float(fa.mse(au8, bu8)), mse_ref, 1e-9)
            ok &= report(f"[{h}x{w} {kind}] {dev} psnr uint8",
                         float(fa.psnr(au8, bu8)), psnr_ref, TOL_PSNR)

    # batch / multichannel consistency
    print("\n  -- batch & channel handling --")
    a1, b1 = make_pair(200, 300, "natural", 6.0, seed=1)
    a2, b2 = make_pair(200, 300, "natural", 30.0, seed=2)
    r1 = ref.ssim_reference(a1, b1)
    r2 = ref.ssim_reference(a2, b2)
    for dev in devices:
        xa = torch.as_tensor(np.stack([a1, a2])[:, None], dtype=torch.uint8, device=dev)
        xb = torch.as_tensor(np.stack([b1, b2])[:, None], dtype=torch.uint8, device=dev)
        per = fa.ssim(xa, xb, data_range=255.0, reduction="none").double().cpu().numpy()
        ok &= report(f"{dev} batch[0]", float(per[0]), r1, TOL_SSIM)
        ok &= report(f"{dev} batch[1]", float(per[1]), r2, TOL_SSIM)
        ok &= report(f"{dev} batch mean", float(fa.ssim(xa, xb, data_range=255.0)),
                     0.5 * (r1 + r2), TOL_SSIM)
        # 2-channel image = mean of the two per-channel SSIMs
        ca = torch.as_tensor(np.stack([a1, a2])[None], dtype=torch.uint8, device=dev)
        cb = torch.as_tensor(np.stack([b1, b2])[None], dtype=torch.uint8, device=dev)
        ok &= report(f"{dev} 2-channel mean", float(fa.ssim(ca, cb, data_range=255.0)),
                     0.5 * (r1 + r2), TOL_SSIM)

    # identity
    print("\n  -- degenerate cases --")
    for dev in devices:
        z = torch.randint(0, 256, (1, 1, 128, 128), dtype=torch.uint8, device=dev)
        ok &= report(f"{dev} ssim(x,x)", float(fa.ssim(z, z, data_range=255.0)), 1.0, 1e-6)
        ok &= report(f"{dev} mse(x,x)", float(fa.mse(z, z)), 0.0, 0.0)
        p = float(fa.psnr(z, z))
        good = p == float("inf")
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {dev} psnr(x,x) = {p}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** FAILURES PRESENT ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
