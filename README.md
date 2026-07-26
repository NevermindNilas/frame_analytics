# frame_analytics

Image and video quality metrics for PyTorch — CPU and CUDA — at float64-reference
accuracy. Fused kernels for forward and backward, so the same code works as an
evaluation metric and as a training loss.

```bash
pip install frame-analytics
```

```python
import torch, frame_analytics as fa

a = torch.randint(0, 256, (8, 3, 1080, 1920), dtype=torch.uint8, device="cuda")
b = torch.randint(0, 256, (8, 3, 1080, 1920), dtype=torch.uint8, device="cuda")

fa.mse(a, b)
fa.psnr(a, b, data_range=255.0)
fa.ssim(a, b, data_range=255.0)      # Wang et al. 2004, exactly
fa.ms_ssim(a, b, data_range=255.0)   # Wang et al. 2003
fa.gmsd(a, b, data_range=255.0)      # Xue et al. 2014
fa.l1(a, b); fa.charbonnier(a, b); fa.huber(a, b)

fa.ssim(a, b, reduction="none")      # per-image, (8,)
fa.ssim(a, b, return_map=True)       # (8, 3, 1070, 1910)

# the convention the super-resolution literature reports
fa.psnr(a, b, data_range=255.0, luma="matlab", crop_border=4)
```

Accepts `(H,W)` / `(C,H,W)` / `(N,C,H,W)`, uint8 through float64. uint8 stays
uint8 into the kernel — 2 bytes/pixel of bandwidth instead of 8.

## Speed

Speedup over each library, across 512²→4K and batch 1→8, RTX 3090 / 16-thread CPU:

| | SSIM | MS-SSIM | GMSD | PSNR / MSE |
|---|---|---|---|---|
| pytorch-msssim | 16–22× | 7.5–14× | — | — |
| kornia | 18–32× | — | — | 2.0–8.5× |
| torchmetrics | 24–28× | — | — | 4.9–14× |
| piq | 25–30× | 6.7–20× | 40–50× | 12–37× |
| scikit-image (CPU) | 21–29× | — | — | 170–920× |
| OpenCV recipe (CPU) | 6.0–7.6× | — | — | 0.7–7.3× (`cv2.PSNR`) |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim) (CUDA) | 1.2–1.9× | — | — | — |
| torch built-ins (L1 / Huber) | — | — | — | 2.4–100× |

Selected absolute numbers, RTX 3090, ms/call (lower is better):

| CUDA, uint8 in | 512² | 1080p | 1080p ×8 | 4K ×8 |
|---|---:|---:|---:|---:|
| SSIM | 0.024 | 0.085 | 0.651 | 2.642 |
| PSNR | 0.020 | 0.028 | 0.045 | 0.151 |
| MS-SSIM | 0.331 | 0.512 | 3.088 | — |
| GMSD | 0.023 | 0.055 | 0.313 | — |
| L1 | 0.021 | 0.026 | 0.115 | — |

SSIM sustains 25.6 Gpixel/s (~11 700 fps at 1080p). PSNR at 4K ×8 hits 880 GB/s
— 94% of the card's theoretical bandwidth, the ceiling for anything that must
read both frames.

| CPU, uint8 in, ms/call | 512² | 1080p | 1080p ×8 | 4K ×8 |
|---|---:|---:|---:|---:|
| **PSNR**, 1 channel | **0.008** | **0.015** | **0.102** | **1.41** |
| `cv2.PSNR`, per frame | 0.006 | 0.049 | 0.740 | 4.24 |
| kornia | 0.046 | 0.114 | 3.29 | 14.4 |
| torchmetrics | 0.081 | 0.195 | 5.52 | 22.7 |
| scikit-image | 1.43 | 11.7 | 93.5 | 381 |
| | | | | |
| **L1**, RGB | **0.011** | **0.039** | **1.03** | — |
| `F.l1_loss` | 0.193 | 3.05 | 30.1 | — |

Streaming 1080p RGB via CUDA graphs, host uint8 in, python float out: **931 fps**
(2 941 fps for resident tensors).

Full per-size CPU and CUDA tables: `python bench/bench.py`.

### Where it doesn't win

- `cv2.PSNR` on one sub-megapixel single-channel frame: 0.006 ms against our
  0.008. Not the kernel — that runs the same 262 144 pixels in 3.8 µs, less than
  `cv2.PSNR` takes for the whole call — but the ~4 µs of Python in front of it,
  which is a fixed cost and so only visible when there is nothing else to pay
  for. One frame bigger, or one channel wider, and it inverts: 3.3× at 1080p,
  7.3× at 1080p ×8.
- **Without the compiled extension** the portable PyTorch fallback is only ~1.1×
  faster than `pytorch-msssim` and uses *more* memory. The speed claims are
  claims about the kernels; the fallback exists to be correct, not to win.

## Memory

Peak allocation *above the two input frames*, RTX 3090, 3-channel, MiB per
call. Nothing here is the frames themselves — those you already have:

| CUDA, forward | 1080p | 1080p ×8 | 4K ×8 |
|---|---:|---:|---:|
| **SSIM** | **0.02** | **0.19** | **0.75** |
| pytorch-msssim | 288 | 2253 | 9048 |
| piq | 336 | 2633 | 10568 |
| kornia | 360 | 2850 | 11397 |
| torchmetrics | 550 | 4388 | 17509 |
| | | | |
| **MS-SSIM** | **15.5** | **120** | **476** |
| pytorch-msssim | 288 | 2253 | 9048 |
| piq | 336 | 2633 | 10568 |
| | | | |
| **GMSD** | **0.02** | **0.19** | **0.75** |
| piq | 72 | 760 | 3040 |
| | | | |
| **PSNR / MSE / L1 / Charbonnier** | **0.007** | **0.007** | **0.007** |
| torchmetrics, kornia, `F.l1_loss` | 48 | 380 | 1520 |

A fused kernel has nowhere to put an intermediate. What is left for SSIM and
GMSD is the per-block partial sums — sized by the tile grid, so 4K ×8 costs
0.75 MiB where the alternatives cost 9–17 GiB — and for the pixel metrics, the
output scalar. The libraries above are not doing anything wrong; five or six
frame-shaped temporaries is what the textbook formulation asks for, and 2253
MiB at 1080p ×8 is almost exactly 6× the 380 MiB input.

MS-SSIM is the one metric that must materialise something: the four
downsampled copies of both frames. That geometric series sums to a third of a
frame each, and nothing full-resolution survives — 120 MiB at 1080p ×8, where
the frame pair itself is 380.

The CPU story is the same one. There is no `max_memory_allocated` for host
memory, so these are peak process footprints, each measured in its own
interpreter — how much more memory the OS had to hand over, which is the
number that decides whether the box swaps:

| CPU, forward | 1080p | 1080p ×8 |
|---|---:|---:|
| **SSIM** | **2.0** | **2.4** |
| pytorch-msssim | 433 | 2796 |
| piq | 446 | 3256 |
| kornia | 457 | 3305 |
| torchmetrics | 915 | 7026 |
| | | |
| **MS-SSIM**, uint8 in | **8.9** | **10.6** |
| **MS-SSIM**, fp32 in | 16.8 | 127 |
| pytorch-msssim | 462 | 2898 |
| piq | 471 | 3256 |
| | | |
| **GMSD** | **0.11** | **0.69** |
| piq | 84 | 793 |
| | | |
| **PSNR / MSE / L1** | **<0.01** | **<0.01** |
| kornia (PSNR) | 3.0 | 190 |
| torchmetrics, `F.l1_loss` | 27–28 | 380 |

The pixel metrics reduce two buffers to a scalar and allocate nothing at all.
SSIM's 2 MiB does not move when the pixel count goes up 8×, so it is fixed
working set rather than anything per-pixel. MS-SSIM is cheaper from uint8 than
from float because the cast is fused into the first pooling step: fed uint8 it
never materialises a full-resolution float copy of either frame, and only the
pyramid survives.

| CUDA, loss step (fwd + bwd, fp32) | 1080p | 1080p ×8 | 4K ×8 |
|---|---:|---:|---:|
| **SSIM** | **94** | **752** | **3022** |
| pytorch-msssim | 480 | 3763 | 15098 |
| piq | 456 | 3579 | 14350 |
| | | | |
| **MS-SSIM** | **118** | **942** | **3782** |
| pytorch-msssim | 480 | 3763 | 15098 |
| piq | 456 | 3579 | 14350 |
| | | | |
| **GMSD** | 78 | **616** | **2465** |
| piq | 72 | 760 | 3040 |
| | | | |
| **L1** | **24** | **190** | **760** |
| `F.l1_loss` | 96 | 760 | 3040 |

Every row includes the gradient with respect to the prediction — 24 MiB at
1080p, 760 at 4K ×8 — which is unavoidable and which every implementation
pays. For `fa.l1` it is the *entire* figure: the backward writes the gradient
and allocates nothing else. SSIM costs three more planes on top of it and
MS-SSIM four, at every size — the fused backward recomputes the local moments
instead of saving them, while the autograd implementations keep the forward's
activations, which is where the 15 GiB comes from. GMSD on a single 1080p
frame is the one row that loses, by 6 MiB.

`StreamingMetrics` at 1080p RGB holds **18 MiB** — the two device frames plus
the CUDA graph's private pool — and allocates **nothing** per frame, whether
it is scoring one metric or nine.

Without the compiled extension the portable path allocates like the rest of
the field, and slightly worse than the best of it: 2830 MiB for SSIM at
1080p ×8, and 4.5–4.7 GiB as a loss step depending on what `torch.compile`
managed to fuse.

Full tables: `python bench/bench_memory.py`.

## Accuracy

Defaults reproduce `ssim_index.m`: 11×11 Gaussian, σ=1.5, K=(0.01, 0.03),
`valid` support. Every kernel is gated against a float64 transcription of the
paper (`python tests/validate.py`).

| implementation | SSIM abs. error |
|---|---:|
| **frame_analytics** (CUDA / CPU / portable) | **2.4e-09 / 3.1e-09 / 6.9e-09** |
| pytorch-msssim | 1.0e-07 |
| fused-ssim (`padding="valid"`) | 3.3e-06 |
| kornia | 2.1e-05 |
| torchmetrics | 2.2e-05 |
| scikit-image (defaults) | 1.1e-03 |
| fused-ssim (`padding="same"`, its default) | 3.0e-03 |
| piq (defaults) | 8.8e-03 |

The bottom entries aren't bugs — different windows, MATLAB-style downsampling,
zero padding. Defensible defaults, different metric.

MSE and PSNR match float64 numpy exactly; the accumulator is float64 even when
the elementwise work is float32 (a 4K frame has 8.3M residuals, and summing
those in float32 loses ~4 significant digits straight into the dB figure).

Worst abs. error vs float64 reference, over CPU and CUDA, native and portable:

| metric | error |
|---|---:|
| MS-SSIM | 3.7e-08 |
| GMSD | 2.8e-09 |
| L1, Huber (uint8) | 8.9e-16 — exact, integer accumulator |
| Charbonnier | 1.3e-07 |

GMSD's similarity map is float32, so deviations below ~1e-7 measure rounding,
not the images (`gmsd(x, x)` ≈ 3e-08, not 0). A typical GMSD is ~0.03.

## Training

Every metric is differentiable, and the SSIM family has a fused CUDA backward
rather than autograd over the portable path — which is where the memory goes:

```python
crit_ms = fa.MSSSIM(data_range=1.0)
crit_l1 = fa.L1()
loss = 0.84 * crit_ms.loss(pred, target) + 0.16 * crit_l1(pred, target)
loss.backward()
```

That mix is Zhao et al., *Loss Functions for Image Restoration with Neural
Networks* (2016) — the reason MS-SSIM is here at all.

MS-SSIM loss step, 1080p ×8, float32 RGB: **12.7 ms / 942 MiB** against
pytorch-msssim's 81.3 ms / 3763 MiB — **6.4× faster on 4.0× less memory**. The
fused backward recomputes the local moments instead of storing them, so nothing
full-resolution survives the forward pass at any of the five scales. Both
figures include the 190 MiB gradient neither implementation can avoid; see
[Memory](#memory).

Gradients are verified two independent ways — against autograd over the portable
path, and against central differences on the float64 reference forward — both to
~1e-6 relative. The scalar-reduction backward issues no device→host sync, so it
does not stall the training pipeline.

Caveats:

- `GMSD.loss()` returns `1 − mean(GMS)`, not the deviation. `d√var/dvar` is
  unbounded as the variance goes to zero, which is exactly where a converging
  model lives; the mean is the well-behaved objective from the same map.
  `fa.gmsd()` still gives you the published metric.
- MS-SSIM's per-scale factors are clamped at zero, so on anti-correlated content
  the value **and its gradient** are exactly zero. Standard formulation, every
  implementation clamps — but MS-SSIM alone cannot pull a diverged model back.
  The L1 term above removes the dead zone.
- The native backward is CUDA + float32 only; CPU, float64, uint8 and
  `downsample=True` fall back to autograd over the portable path.
- The fused backward is a kernel, not a graph, so `create_graph=True` (gradient
  penalties, HVPs, `torch.func.hessian`) raises rather than silently returning
  zeros. For a second derivative use `backend_hint="torch"` with
  `fa.set_compile_enabled(False)`.

## Reporting conventions

`luma=` and `crop_border=` are on every metric. Nearly every super-resolution
and restoration paper reports PSNR/SSIM on the luma plane of a border-cropped
frame, so a library without them produces right-looking numbers that quietly
disagree with the literature.

```python
fa.psnr(a, b, data_range=255.0, luma="matlab", crop_border=scale)   # Y-PSNR
fa.ssim(a, b, data_range=255.0, luma="matlab", crop_border=scale)   # Y-SSIM
```

`"matlab"` is the studio-range Y′ of MATLAB's `rgb2ycbcr`, i.e. what BasicSR's
`test_y_channel=True` computes. `"bt601"` and `"bt709"` are the full-range
definitions. The crop is a view, so it costs nothing.

## Install

```bash
pip install frame-analytics
```

torch ≥ 2.0, numpy. The kernels arrive **precompiled**, one wheel per platform
and no version matrix — the same wheel works on every Python and every torch
build.

That is not the usual arrangement, because the usual arrangement cannot be
published. A torch C++ extension is ABI-locked to the exact {python} × {torch} ×
{CUDA} it was compiled against, and a wheel filename has nowhere to record
"torch 2.6": pip would match on Python and platform alone and hand you an
`undefined symbol` ImportError. So the kernels sit behind a plain C ABI instead
— raw pointers, an int dtype code, a stream handle, int return codes — and the
binaries link neither libtorch nor libpython. On Windows they import exactly one
library, `KERNEL32.dll`; on Linux, glibc. An NVIDIA driver at runtime is the
only external requirement, and only for the CUDA half.

Install on a platform with no wheel and the sources — which ship too — compile
on first call (~1 min, cached thereafter). No compiler at all and every native
kernel still has a portable PyTorch fallback returning the same numbers.
`fa.backend_status()` reports which tier is live and where its binary came from;
`backend_hint="torch"` / `"native"` forces either. On Windows the MSVC build
environment is located automatically.

To compile from source at install time instead:

```bash
FA_BUILD_EXT=1 FA_CUDA_ARCHS="8.6 9.0+PTX" pip install frame-analytics
```

From a checkout:

```bash
pip install -e .
python tests/validate.py           # accuracy gate
python bench/bench.py              # speed + accuracy tables
python bench/bench_training.py     # MS-SSIM / GMSD / pixel losses
python bench/bench_memory.py       # peak memory, forward and loss step
python bench/bench_fused_ssim.py   # head-to-head vs fused-ssim
```

## API

Every metric takes `reduction` (`"mean"` → scalar, `"none"` → per-image `(N,)`),
`dtype`, `luma`, `crop_border` and — where there is a kernel — `backend_hint`.

```python
mse        (x, y, *, reduction="mean", dtype=None, out_dtype=torch.float64,
            luma=None, crop_border=0)
psnr       (x, y, *, data_range=None, eps=0.0, ...)
ssim       (x, y, *, data_range=None, win_size=11, sigma=1.5, K=(0.01, 0.03),
            return_map=False, downsample=False, backend_hint="auto", ...)
ms_ssim    (x, y, *, data_range=None, win_size=11, sigma=1.5, K=(0.01, 0.03),
            weights=MS_SSIM_WEIGHTS, backend_hint="auto", ...)
gmsd       (x, y, *, data_range=None, T=None, eps=None, downsample=True,
            return_map=False, backend_hint="auto", ...)
gms        (x, y, ...)                      # mean of the same map
l1         (x, y, ...)
charbonnier(x, y, *, eps=1e-3, ...)         # mean sqrt(d^2 + eps^2)
huber      (x, y, *, delta=1.0, ...)        # matches torch.nn.HuberLoss
rgb_to_luma(t, mode="bt601", *, data_range=None, dtype=None)
```

`data_range` defaults to 255 for integer input, 1.0 for float. `downsample=True`
on `ssim` applies MATLAB `ssim.m`'s automatic box-downsample (off by default, as
in `ssim_index.m` and every PyTorch library); on `gmsd` it is the paper's 2×
prefilter and is *on* by default.

Module forms `MSE`, `PSNR`, `SSIM`, `MSSSIM`, `GMSD`, `L1`, `Charbonnier`,
`Huber` cache what they can and expose `.loss()`:

```python
crit = fa.SSIM(data_range=1.0)
loss = crit.loss(pred, target)     # 1 - SSIM, fused backward on CUDA float32
loss.backward()

sm = fa.StreamingMetrics((1, 3, 1080, 1920), device="cuda", dtype=torch.uint8,
                         metrics=("psnr", "ssim", "ms_ssim", "gmsd"))
for ref, dist in frames:
    out = sm.update(ref, dist)     # one graph replay, all four metrics
```

`StreamingMetrics` accepts any of `mse`, `psnr`, `ssim`, `ms_ssim`, `gmsd`,
`gms`, `l1`, `charbonnier`, `huber`; they all capture into the same CUDA graph,
so scoring a frame on nine metrics is still one replay.

## License

Apache 2.0.

Wang, Bovik, Sheikh, Simoncelli. *Image Quality Assessment: From Error
Visibility to Structural Similarity.* IEEE TIP 13(4), 2004.
