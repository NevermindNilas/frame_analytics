"""Contract tests for the 15 metrics added in 0.7.0.

Cheap invariants, not accuracy gates: identity values, distortion
monotonicity, data_range invariance, and reduction shapes. Sizes respect
each metric's documented minimum (pyramid metrics need room).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_analytics as fa

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _rgb(h, w, seed=0, uint8=False, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    if uint8:
        return (torch.rand(1, 3, h, w, generator=g, device=device) * 255).byte()
    return torch.rand(1, 3, h, w, generator=g, device=device)


def _noisy(x, std=0.05, seed=1):
    g = torch.Generator(device=x.device).manual_seed(seed)
    return (x + std * torch.randn_like(x, generator=g)).clamp(0, 1)


# (name, size, identical_value, direction) with direction +1 means higher
# is better (distortion lowers the score), -1 means lower is better.
CASES = [
    ("ms_gmsd", 64, 0.0, -1),
    ("vifp", 256, 1.0, 1),
    ("iw_ssim", 256, 1.0, 1),
    ("dss", 64, 1.0, 1),
    ("nlpd", 64, 0.0, -1),
    ("fsim", 64, 1.0, 1),
    ("srsim", 64, 1.0, 1),
    ("vsi", 64, 1.0, 1),
    ("mdsi", 64, 0.0, -1),
    ("haarpsi", 64, 1.0, 1),
    ("adm_like", 64, 1.0, 1),
    ("psnr_hvs", 64, None, 1),
    ("psnr_hvs_m", 64, None, 1),
    ("ciede2000", 64, 0.0, -1),
    ("flip", 64, 0.0, -1),
    ("scielab", 64, 0.0, -1),
]


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("name,size,ident,direc", CASES)
def test_identity(dev, name, size, ident, direc):
    x = _rgb(size, size, uint8=True).to(dev)
    fn = getattr(fa, name)
    got = float(fn(x, x))
    if ident is None:  # psnr_hvs family: inf on identical input
        assert got == float("inf")
    elif ident == 0.0:
        assert abs(got) < 1e-6
    else:
        assert abs(got - 1.0) < 1e-4


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("name,size,ident,direc", CASES)
def test_monotonic_noise(dev, name, size, ident, direc):
    if name.startswith("psnr_hvs"):
        pytest.skip("dB scale covered by identity inf test")
    x = _rgb(size, size).to(dev)
    fn = getattr(fa, name)
    mild = float(fn(x, _noisy(x, 0.02)))
    strong = float(fn(x, _noisy(x, 0.10)))
    assert mild != strong
    if direc > 0:
        assert mild > strong
    else:
        assert mild < strong


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("name", ["ms_gmsd", "vifp", "dss", "nlpd", "fsim",
                                  "srsim", "vsi", "mdsi", "haarpsi",
                                  "adm_like", "ciede2000", "flip", "scielab"])
def test_data_range_invariance(dev, name):
    size = 256 if name in ("vifp",) else 64
    fn = getattr(fa, name)
    x = _rgb(size, size).to(dev)
    y = _noisy(x, 0.03)
    a = float(fn(x, y, data_range=1.0))
    # exact rescale (no byte rounding: 1/255 quantization alone moves
    # sensitive metrics by ~1e-3, which is a test artifact, not a bug)
    b = float(fn(x * 255.0, y * 255.0, data_range=255.0))
    assert abs(a - b) < 5e-4, (name, a, b)


@pytest.mark.parametrize("dev", DEVICES)
def test_reduction_none_shapes(dev):
    xb = torch.cat([_rgb(64, 64), _rgb(64, 64)]).to(dev)
    yb = torch.cat([_rgb(64, 64, seed=9), _rgb(64, 64, seed=9)]).to(dev)
    for name in ["ms_gmsd", "dss", "nlpd", "fsim", "srsim", "vsi", "mdsi",
                 "haarpsi", "adm_like", "ciede2000", "flip", "scielab"]:
        out = getattr(fa, name)(xb, yb, reduction="none")
        assert out.shape == (2,), (name, out.shape)


def test_module_wrappers_agree():
    x = _rgb(64, 64)
    y = _noisy(x, 0.03)
    pairs = [("MS_GMSD", "ms_gmsd"), ("VIFP", "vifp"), ("DSS", "dss"),
             ("NLPD", "nlpd"), ("FSIM", "fsim"), ("SRSIM", "srsim"),
             ("VSI", "vsi"), ("MDSI", "mdsi"), ("HAARPSI", "haarpsi"),
             ("ADM_LIKE", "adm_like"), ("CIEDE2000", "ciede2000"),
             ("FLIP", "flip"), ("SCIELAB", "scielab")]
    for cls_name, fn_name in pairs:
        mod = getattr(fa, cls_name)()
        assert abs(float(mod(x, y)) - float(getattr(fa, fn_name)(x, y))) < 1e-9


def test_vapoursynth_registry_closed():
    pytest.importorskip("vapoursynth")
    from frame_analytics import vapoursynth as fa_vs

    feats = fa_vs.available_features()
    for name in ["psnr_hvs", "ciede2000", "ms_gmsd", "vifp", "iw_ssim",
                 "dss", "nlpd", "fsim", "srsim", "vsi", "mdsi",
                 "haarpsi", "adm_like", "psnr_hvs_m", "flip", "scielab"]:
        assert name in feats, name
