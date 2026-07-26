# Bundled weights

`lpips_alex_v0.1.pt` — 9.4 MiB, float32, everything `fa.lpips()` needs. Built by
`tools/build_lpips_weights.py`; checked in rather than generated at install
time, because generating it needs `torchvision` and `lpips` and no CI runner
has either.

Two upstream pieces, neither modified beyond being sliced out and repacked:

| part | source | license |
|---|---|---|
| conv trunk, `features[:12]` | torchvision `AlexNet_Weights.IMAGENET1K_V1` | BSD-3-Clause, `LICENSE.torchvision` |
| five 1×1 calibration convs | `richzhang/PerceptualSimilarity`, `lpips/weights/v0.1/alex.pth` | BSD-2-Clause, `LICENSE.PerceptualSimilarity` |

Both licenses permit binary redistribution with the notice attached, which is
what the two `LICENSE.*` files in this directory are. They ship in the wheel and
the sdist. `frame_analytics` itself is Apache-2.0; these weights are not, and
the terms above travel with them.

## Why it is 9.4 MiB and not 233

The checkpoint torchvision downloads for AlexNet is 233 MiB, ~94% of which is
the fully-connected classifier head. LPIPS taps five post-ReLU activations out
of the conv trunk and never touches the head, so only `features[:12]` is kept —
2,469,696 parameters. The trailing `MaxPool2d` goes too, since no tap follows
it. The calibration weights are the other 1,152 numbers.

Stored float32. float16 would halve it and move the metric in roughly the
fourth decimal, which is the decimal papers print.

## What is deliberately absent

The VGG16 trunk (56 MiB). It is the variant used as a *training* loss; the one
reported as a *metric* — and the one the LPIPS authors recommend — is AlexNet.
Point `FA_LPIPS_WEIGHTS` at a blob of the same layout to supply your own.
