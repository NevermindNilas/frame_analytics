# frame_analytics

Image and video quality metrics for PyTorch — CPU and CUDA — at
float64-reference accuracy. Fused kernels for the forward pass and the
gradient, so the same code works as an evaluation metric and as a training
loss.

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

## Faster than every library measured

Speedup over each library, across 512²→4K and batch 1→8, RTX 3090 / 16-thread CPU:

| | SSIM | MS-SSIM | GMSD | PSNR / MSE |
|---|---|---|---|---|
| pytorch-msssim | 16–22× | 7.5–14× | — | — |
| kornia | 18–32× | — | — | 2.0–8.5× |
| torchmetrics | 24–28× | — | — | 4.9–14× |
| piq | 25–30× | 6.7–20× | 40–50× | 12–37× |
| scikit-image (CPU) | 21–29× | — | — | 26–113× |
| OpenCV recipe (CPU) | 6.0–7.6× | — | — | see below |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim) (CUDA) | 1.2–1.9× | — | — | — |
| torch built-ins (L1 / Huber) | — | — | — | 2.6–22× |

…while being the most accurate of all of them (below).

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

MSE and PSNR match float64 numpy exactly. The accumulator is float64 even when
the elementwise work is float32: a 4K frame has 8.3M residuals, and summing
those in float32 loses ~4 significant digits straight into the dB figure.

The training metrics are gated the same way, against float64 transcriptions of
their own papers, on every backend and device:

| metric | worst abs. error vs float64 reference |
|---|---:|
| MS-SSIM | 3.7e-08 |
| GMSD | 2.8e-09 |
| L1, Huber (uint8) | 8.9e-16 — exact, integer accumulator |
| Charbonnier | 1.3e-07 |

(Worst case over CPU and CUDA, native and portable, four image kinds.)

The MS-SSIM reference agrees with `piq` to 3e-15 and with `pytorch-msssim` to
2e-07 on power-of-two sizes, so the convention here is the community one rather
than a private variant. (At other sizes those two pad before pooling where this
package crops, which is a real difference, not an error on either side.)

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

`GMSD.loss()` returns `1 − mean(GMS)`, not the deviation. The deviation is the
published *metric*, but `d√var/dvar` is unbounded as the variance goes to zero,
which is exactly where a converging model lives; the mean is the well-behaved
objective from the same map. `fa.gmsd()` still gives you the metric.

Two things to know before differentiating:

- MS-SSIM's per-scale factors are clamped at zero, so on anti-correlated
  content the value **and its gradient** are exactly zero. That is the standard
  formulation — a fractional power of a negative number is not real, and every
  implementation clamps — but MS-SSIM alone cannot pull a diverged model back.
  The L1 term in the mix above removes the dead zone.
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
`test_y_channel=True` computes — the convention behind the published numbers.
`"bt601"` and `"bt709"` are the full-range definitions. The crop is a view, so
it costs nothing; the luma projection does materialise one single-channel plane
(the metric then runs on a third of the data, so it comes out ahead anyway).

## Benchmarks

RTX 3090, Ryzen 16 threads, torch 2.13+cu132. ms/call, lower is better.

**SSIM, CUDA** (uint8 in)

| | 512² | 1080p | 1080p ×8 | 4K | 4K ×8 |
|---|---:|---:|---:|---:|---:|
| **frame_analytics** | **0.024** | **0.085** | **0.651** | **0.324** | **2.642** |
| pytorch-msssim | 0.539 | 1.450 | 10.42 | 5.322 | 41.42 |
| kornia | 0.774 | 1.817 | 12.38 | 6.368 | 48.14 |
| torchmetrics | 0.576 | 2.275 | 17.69 | 9.031 | 71.02 |
| piq | 0.726 | 2.438 | 17.12 | 8.890 | 67.38 |

25.6 Gpixel/s — ~11 700 fps at 1080p, ~3 100 at 4K.

**PSNR, CUDA** (uint8 in; the others are float32-only)

| | 512² | 1080p | 1080p ×8 | 4K | 4K ×8 |
|---|---:|---:|---:|---:|---:|
| **frame_analytics** | **0.020** | **0.028** | **0.045** | **0.031** | **0.151** |
| kornia | 0.064 | 0.057 | 0.343 | 0.169 | 1.284 |
| torchmetrics | 0.098 | 0.147 | 0.574 | 0.327 | 2.063 |
| piq | 0.305 | 0.340 | 1.371 | 0.796 | 5.592 |

4K ×8 runs at **880 GB/s, 94% of the card's theoretical bandwidth** — the
ceiling for anything that must read both frames.

**MS-SSIM, CUDA** (uint8 in, RGB)

| | 512² | 512² ×8 | 1080p | 1080p ×8 |
|---|---:|---:|---:|---:|
| **frame_analytics** | **0.331** | **0.498** | **0.512** | **3.088** |
| frame_analytics (portable path) | 1.151 | 3.471 | 3.777 | 26.885 |
| pytorch-msssim | 2.481 | 6.988 | 6.959 | 44.513 |
| piq | 2.231 | 8.352 | 8.602 | 62.210 |

**GMSD, CUDA** (uint8 in, RGB)

| | 512² | 512² ×8 | 1080p | 1080p ×8 |
|---|---:|---:|---:|---:|
| **frame_analytics** | **0.023** | **0.058** | **0.055** | **0.313** |
| frame_analytics (portable path) | 0.281 | 0.578 | 0.580 | 3.856 |
| piq | 0.914 | 2.447 | 2.413 | 15.631 |

**Pixel losses, CUDA** (uint8 in; torch has to widen to float32 first)

| | 512² | 512² ×8 | 1080p | 1080p ×8 |
|---|---:|---:|---:|---:|
| **fa.l1** | **0.021** | **0.025** | **0.026** | **0.115** |
| `F.l1_loss` | 0.055 | 0.272 | 0.271 | 2.022 |
| **fa.charbonnier** | **0.018** | **0.027** | **0.027** | **0.132** |
| torch Charbonnier | 0.066 | 0.394 | 0.392 | 2.924 |
| **fa.huber** | **0.020** | **0.025** | **0.041** | **0.120** |
| `F.huber_loss` | 0.047 | 0.212 | 0.214 | 1.518 |

1080p ×8 L1 moves 99.6 MB in 0.115 ms — **866 GB/s, 93% of peak**. All four
pixel losses land within 15% of each other, i.e. all four are at the bandwidth
ceiling and the penalty function is free.

**CPU**

| SSIM | 512² | 1080p | 1080p ×4 | | PSNR | 1080p | 1080p ×8 | 4K ×8 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| **frame_analytics** | **1.10** | **6.16** | **21.85** | | **frame_analytics** | **0.14** | **0.52** | **1.96** |
| pytorch-msssim | 3.29 | 34.31 | 165.4 | | kornia | 0.19 | 4.02 | 17.0 |
| kornia | 3.87 | 34.19 | 173.0 | | torchmetrics | 0.29 | 6.17 | 27.7 |
| OpenCV recipe | 6.59 | 46.97 | — | | piq | 1.08 | 14.3 | 66.3 |
| scikit-image | 22.79 | 180.7 | — | | scikit-image | 7.32 | 58.8 | 227 |

**MS-SSIM / GMSD / pixel losses, CPU** (uint8 in, RGB)

| | 512² | 512² ×8 | 1080p | 1080p ×8 |
|---|---:|---:|---:|---:|
| **MS-SSIM** | **5.58** | **28.87** | **33.54** | **225.6** |
| pytorch-msssim | 14.64 | 134.3 | 120.4 | 1482.7 |
| piq | 15.44 | 124.2 | 117.5 | 1297.9 |
| | | | | |
| **GMSD** | **0.13** | **0.60** | **0.61** | **4.73** |
| piq | 2.21 | 12.15 | 10.27 | 125.7 |
| | | | | |
| **fa.charbonnier** | **0.07** | **0.22** | **0.34** | **1.63** |
| torch Charbonnier | 0.24 | 3.55 | 2.85 | 47.35 |
| **fa.huber** | **0.06** | **0.37** | **0.28** | **1.88** |
| `F.huber_loss` | 0.14 | 2.29 | 2.27 | 23.26 |
| **fa.l1** | 0.21 | **0.08** | **0.07** | **1.05** |
| `F.l1_loss` | **0.17** | 2.86 | 2.96 | 29.03 |

**Forward + backward** — SSIM as a training loss, `1 - ssim(x, y)` then `.backward()`

| | 512² | 512² ×8 | 1080p | 1080p ×8 | 4K | 4K ×8 |
|---|---:|---:|---:|---:|---:|---:|
| **frame_analytics** | **0.371** | **0.327** | **0.354** | **2.106** | **1.102** | **8.663** |
| fused-ssim | 0.444 | 0.535 | 0.469 | 2.630 | 1.410 | 10.741 |

**MS-SSIM as a loss**, `1 - ms_ssim(x, y)` then `.backward()`, float32 RGB.
Time in ms/step, memory is peak CUDA allocation for the step:

| | 512² | 512² ×8 | 1080p | 1080p ×8 |
|---|---:|---:|---:|---:|
| **frame_analytics** | **1.091** | **2.136** | **2.045** | **12.668** |
| frame_analytics (portable path) | 3.317 | 13.163 | 13.259 | 102.005 |
| pytorch-msssim | 5.281 | 12.885 | 13.008 | 81.337 |
| | | | | |
| **frame_analytics**, MiB | **11.7** | **94.0** | **93.9** | **752.0** |
| frame_analytics (portable path), MiB | 73.6 | 586.3 | 588.5 | 4701.8 |
| pytorch-msssim, MiB | 55.9 | 454.6 | 455.2 | 3573.5 |

**6.4× faster on 4.8× less memory** at 1080p ×8. The memory is the interesting
half: the fused backward recomputes the local moments instead of storing them,
so nothing full-resolution survives the forward pass at any of the five scales.

Gradients are verified two independent ways: against autograd over the portable
path, and against central differences on the float64 reference forward — both
to ~1e-6 relative. The scalar-reduction backward issues no device→host sync, so
it does not stall the training pipeline; there is a test asserting that.

**Streaming**, 1080p RGB, host uint8 in, python float out, via CUDA graphs:
**931 fps** (1.07 ms/frame including host→device); 2 941 fps for resident tensors.

### Where it doesn't win

`cv2.PSNR` on CPU. It is ~2–3× faster on a single small frame, roughly even at
4K, and we edge it only on large batches.

`F.l1_loss` on one 512² CPU frame — 0.17 ms against our 0.21. Below about a
megapixel the thread pool's wake-up costs more than the arithmetic; from 512²×8
upward we are 34× ahead. Everything else in the tables above, on both devices,
we lead.

**Without the compiled extension** the portable PyTorch path is only ~1.1×
faster than `pytorch-msssim` on MS-SSIM and uses *more* memory than it
(≈354 MB vs 288 MB at 1080p forward). The speed and memory claims above are
claims about the kernels; the fallback exists to be correct, not to win.

The native backward is CUDA + float32 only; CPU, float64, uint8 and
`downsample=True` fall back to autograd over the portable path — correct,
slower. It is also not twice differentiable: `create_graph=True` raises with a
message pointing at the portable path rather than returning silent zeros.

GMSD's similarity map is float32, so deviations below ~1e-7 are measuring
rounding, not the images (`gmsd(x, x)` ≈ 3e-08, not 0). `dtype=torch.float64`
resolves further, on the portable path. A typical GMSD is ~0.03.

## How

The usual SSIM formulation runs five 11×11 convolutions for the five local
expectations (E[x], E[y], E[x²], E[y²], E[xy]) and materialises a dozen
full-resolution intermediates. It is bandwidth-bound on its own temporaries.

- **Separable window** — 22 MACs/px instead of 121, and exactly equal, since
  normalising in 1-D then taking the outer product *is* the 2-D window.
- **Four blurred planes, not five.** SSIM never wants σ<sub>xx</sub> and
  σ<sub>yy</sub> apart, only their sum, so one plane can stand in for two. Which
  one matters: the algebraically obvious E[(x+y)²] − 2E[xy] carries ~4× the
  magnitude of the variance it has to produce and *loses* a factor of 2 in map
  accuracy. Blurring the difference keeps every intermediate the size of the
  answer — σ<sub>xx</sub>+σ<sub>yy</sub> = E[(x−y)²] + 2σ<sub>xy</sub> −
  (μ<sub>x</sub>−μ<sub>y</sub>)², and σ<sub>xy</sub> is already in hand. On
  matched frames x−y is near zero, so it comes out *more* accurate than the
  five-plane form, not less, while taking 20% off the ring buffer that this
  kernel is actually short of. Worth 10–14%.
- **One fused CUDA kernel, zero intermediates.** A block owns a 32×64 output
  tile and streams input through shared memory: rows staged, turned into four
  horizontal partial sums, pushed into a ring buffer; once 11 rows are resident
  the vertical tap runs and SSIM is folded into a block accumulator. DRAM
  traffic is the two input planes plus ~30% halo. Nothing else is written.
- **Compile-time window size.** Templating on the tap count makes the ring wrap
  a compare-and-subtract. GPUs have no integer division — at runtime tap count
  this kernel paid 11 emulated modulos per pixel. Worth 1.7× at 1080p, 6× at 512².
- **Register-blocked vertical tap.** Adjacent output rows share 10 of 11 ring
  rows, so one thread owning two rows reads 60 shared words per two pixels
  instead of 110. Shared memory is the scarce resource (an LDS instruction
  retires 32 lanes/cycle/SM against 128 for FMA). Another ~1.25×.
- **Device-side reductions**, including PSNR's dB conversion — at 1080p the
  metric is ~20 µs of GPU work, so extra launches were a double-digit share.
- **Mean-shift before filtering.** `E[x²]−E[x]²` is a cancellation trap in
  float32. Subtracting a constant cancels exactly out of every variance term and
  is added back into the means — algebraically identical, materially more
  accurate, free.
- **Backward reuses the forward's machinery.** With `a = g·∂S/∂μx`,
  `b = g·∂S/∂σxx`, `c = g·∂S/∂σxy`, the gradient collapses to three more
  Gaussian passes: `dL/dx = 2x'·(w⊛b) + y'·(w⊛c) + w⊛[a − 2b·μx' − c·μy']`.
  The window is symmetric, so that scatter is the *same* tile kernel with its
  origin moved back one halo. And `∂S/∂σyy = ∂S/∂σxx`, so `w⊛b` and `w⊛c` are
  shared between both input gradients.
- **CPU: same structure, own thread pool.** `at::parallel_for` is a compile-time
  alias for an OpenMP region; a JIT-loaded extension isn't built with `/openmp`,
  so the pragmas vanish and it silently runs single-threaded — that was 19×.
  Workers spin briefly before parking, because a plain condition-variable
  handoff costs ~45 µs, more than the entire metric under a megapixel.

And for the metrics added on top of it:

- **MS-SSIM reuses one pass per scale.** Every scale needs the contrast-structure
  term and the coarsest also needs the full SSIM, and both fall out of the same
  four moments — so the tile kernel emits both from registers rather than being
  run twice over the same two planes. The backward generalises the same way: the
  weighted product hands back one gradient per plane for the SSIM mean and one
  for the cs mean, and both fold into the same three coefficient maps. Nothing
  full-resolution is stored between forward and backward at any scale, which is
  where the 5× memory difference comes from — the moments are cheaper to
  recompute than to keep.
- **Accumulate in float, reduce in double.** GA102 retires float64 adds at 1/64
  the float rate. A `double +=` per pixel is invisible in a bandwidth-bound
  kernel until there are two of them, at which point MS-SSIM's tile kernel ran
  3× slower than the single-scale one for arithmetic reasons alone. Each thread
  sums at most eight values in [0,1] before widening, so the float accumulator
  costs ~1e-7 relative and the cross-block sum is still float64.
- **One reciprocal where the algebra allows it.** `-prec-div=true` makes a
  divide a full Newton sequence. SSIM is `A1·A2/(B1·B2)` and cs is `A2/B2`, so
  both come off a single `1/(B1·B2)`.
- **GMSD in one kernel.** Optional 2× box downsample, the Prewitt pair, the
  similarity map and *both* of its moments, with nothing full-resolution
  written. The naive route materialises six intermediates — four directional
  gradients and two magnitudes — before it can even start the map.
- **The variance is taken in float64, before the squaring.** GMS values sit
  within ~1e-3 of 1, so `E[q²] − E[q]²` at float32 returns exactly zero for two
  nearly identical frames: the signal is smaller than the rounding of `q·q`.
  That is the one place in this library where widening early is not optional.
- **Pixel losses are one templated kernel.** MSE, L1, Charbonnier and Huber
  differ only in a compile-time penalty; uint8 MSE and L1 take an exact 64-bit
  integer accumulator, and Huber is written `0.5·min(a,δ)² + δ(a − min(a,δ))` so
  the auto-vectoriser sees a min instead of a select — worth 3.3× on CPU.

## Install

```bash
pip install frame-analytics
```

torch ≥ 2.0, numpy. One universal `py3-none-any` wheel, no version matrix: a
torch C++ extension is ABI-locked to the exact torch build it was compiled
against, so prebuilt binaries would mean a wheel per {python} × {torch} × {CUDA}
× {platform} and would still miss whatever you actually have installed. The
kernel sources ship inside the wheel and compile on first call (~1 min, cached
in the torch extensions directory thereafter).

No compiler? It still works — every native kernel has a portable PyTorch
fallback that returns the same numbers. `fa.backend_status()` reports which is
live; `backend_hint="torch"` / `"native"` forces either. On Windows the MSVC
build environment is located automatically, no developer prompt needed.

To compile at install time instead, and get a platform wheel:

```bash
FA_BUILD_EXT=1 pip install frame-analytics
```

From a checkout:

```bash
pip install -e .
python tests/validate.py           # accuracy gate
python bench/bench.py              # speed + accuracy tables
python bench/bench_training.py     # MS-SSIM / GMSD / pixel losses
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
