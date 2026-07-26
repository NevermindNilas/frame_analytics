"""Build ``frame_analytics/weights/lpips_alex_v0.1.pt`` from the upstream sources.

Run this once, on a machine that has ``torchvision`` and ``lpips`` installed;
the output is committed and ships inside the wheel.  Nothing at runtime --
neither the library nor CI -- needs either package.

    python tools/build_lpips_weights.py

What it extracts, and why the result is 9.4 MiB rather than 233:

* The trunk is ``torchvision.models.alexnet().features[:12]`` -- the five conv
  layers LPIPS taps, and nothing else.  The 233 MiB ``alexnet-owt-*.pth`` that
  torchvision downloads is ~94% classifier head, which LPIPS discards.  Only
  the ``features`` half is kept, and only up to ``relu5``; the trailing
  ``MaxPool2d`` is dropped as well since no tap follows it.
* The five 1x1 calibration convs come from ``lpips/weights/v0.1/alex.pth``
  (6 KB) and are stored verbatim.

Everything stays float32.  Halving to float16 would save 4.7 MiB and perturb
the metric in roughly the fourth decimal, which is the decimal papers print.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import torch

OUT = pathlib.Path(__file__).resolve().parent.parent / "frame_analytics" / "weights"
FORMAT_VERSION = 1

# (conv index inside torchvision's `features`, max-pool before this conv?)
_ALEX_TAPS = ((0, False), (3, True), (6, True), (8, False), (10, False))


def main() -> int:
    try:
        import torchvision
    except ImportError:
        print("needs torchvision (pip install torchvision)", file=sys.stderr)
        return 1
    try:
        import lpips as _lpips_pkg
    except ImportError:
        print("needs lpips (pip install lpips)", file=sys.stderr)
        return 1

    feats = torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1).features

    trunk = []
    for idx, _pool in _ALEX_TAPS:
        conv = feats[idx]
        assert isinstance(conv, torch.nn.Conv2d), f"features[{idx}] is not a conv"
        trunk.append({
            "weight": conv.weight.detach().clone(),
            "bias": conv.bias.detach().clone(),
            "stride": conv.stride[0],
            "padding": conv.padding[0],
        })

    cal_path = pathlib.Path(_lpips_pkg.__file__).parent / "weights" / "v0.1" / "alex.pth"
    cal_sd = torch.load(cal_path, map_location="cpu")
    lin = [cal_sd[f"lin{i}.model.1.weight"].detach().clone() for i in range(5)]

    chns = [w.shape[1] for w in lin]
    assert chns == [64, 192, 384, 256, 256], chns
    assert [t["weight"].shape[0] for t in trunk] == chns

    blob = {
        "format_version": FORMAT_VERSION,
        "net": "alex",
        "lpips_version": "0.1",
        "channels": chns,
        "pool_before": [p for _i, p in _ALEX_TAPS],
        "trunk": trunk,
        "lin": lin,
        "source": {
            "trunk": "torchvision AlexNet_Weights.IMAGENET1K_V1, features[:12]",
            "calibration": "richzhang/PerceptualSimilarity lpips/weights/v0.1/alex.pth",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "lpips_alex_v0.1.pt"
    torch.save(blob, dest, _use_new_zipfile_serialization=True)

    n = sum(t["weight"].numel() + t["bias"].numel() for t in trunk)
    n += sum(w.numel() for w in lin)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f"{dest}\n  {n:,} params, {dest.stat().st_size / 2**20:.2f} MiB")
    print(f"  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
