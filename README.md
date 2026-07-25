# frame_analytics

MSE, PSNR and SSIM for PyTorch — CPU and CUDA — at float64-reference accuracy.
Fused kernels for the forward pass and the gradient, so the same code works as
an evaluation metric and as a training loss.

```python
import torch, frame_analytics as fa

a = torch.randint(0, 256, (8, 3, 1080, 1920), dtype=torch.uint8, device="cuda")
b = torch.randint(0, 256, (8, 3, 1080, 1920), dtype=torch.uint8, device="cuda")

fa.mse(a, b)
fa.psnr(a, b, data_range=255.0)
fa.ssim(a, b, data_range=255.0)      # Wang et al. 2004, exactly
fa.ssim(a, b, reduction="none")      # per-image, (8,)
fa.ssim(a, b, return_map=True)       # (8, 3, 1070, 1910)
```

Accepts `(H,W)` / `(C,H,W)` / `(N,C,H,W)`, uint8 through float64. uint8 stays
uint8 into the kernel — 2 bytes/pixel of bandwidth instead of 8.

## Faster than every library measured

Speedup over each library, across 512²→4K and batch 1→8, RTX 3090 / 16-thread CPU:

| | SSIM | PSNR / MSE |
|---|---|---|
| pytorch-msssim | 16–22× | — |
| kornia | 18–32× | 2.0–8.5× |
| torchmetrics | 24–28× | 4.9–14× |
| piq | 25–30× | 12–37× |
| scikit-image (CPU) | 21–29× | 26–113× |
| OpenCV recipe (CPU) | 6.0–7.6× | see below |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim) (CUDA) | 1.2–1.9× | — |

…while being the most accurate of all of them (below).

## Accuracy

Defaults reproduce `ssim_index.m`: 11×11 Gaussian, σ=1.5, K=(0.01, 0.03),
`valid` support. Every kernel is gated against a float64 transcription of the
paper (`python tests/validate.py`).

| implementation | SSIM abs. error |
|---|---:|
| **frame_analytics** (CUDA / CPU / portable) | **3.2e-09 / 3.1e-09 / 6.9e-09** |
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

**CPU**

| SSIM | 512² | 1080p | 1080p ×4 | | PSNR | 1080p | 1080p ×8 | 4K ×8 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| **frame_analytics** | **1.10** | **6.16** | **21.85** | | **frame_analytics** | **0.14** | **0.52** | **1.96** |
| pytorch-msssim | 3.29 | 34.31 | 165.4 | | kornia | 0.19 | 4.02 | 17.0 |
| kornia | 3.87 | 34.19 | 173.0 | | torchmetrics | 0.29 | 6.17 | 27.7 |
| OpenCV recipe | 6.59 | 46.97 | — | | piq | 1.08 | 14.3 | 66.3 |
| scikit-image | 22.79 | 180.7 | — | | scikit-image | 7.32 | 58.8 | 227 |

**Forward + backward** — SSIM as a training loss, `1 - ssim(x, y)` then `.backward()`

| | 512² | 512² ×8 | 1080p | 1080p ×8 | 4K | 4K ×8 |
|---|---:|---:|---:|---:|---:|---:|
| **frame_analytics** | **0.371** | **0.327** | **0.354** | **2.106** | **1.102** | **8.663** |
| fused-ssim | 0.444 | 0.535 | 0.469 | 2.630 | 1.410 | 10.741 |

Gradients are verified two independent ways: against autograd over the portable
path, and against central differences on the float64 reference forward — both
to ~1e-6 relative. The scalar-reduction backward issues no device→host sync, so
it does not stall the training pipeline; there is a test asserting that.

**Streaming**, 1080p RGB, host uint8 in, python float out, via CUDA graphs:
**931 fps** (1.07 ms/frame including host→device); 2 941 fps for resident tensors.

### Where it doesn't win

`cv2.PSNR` on CPU. It is ~2–3× faster on a single small frame, roughly even at
4K, and we edge it only on large batches. Everything else in the tables above,
on both devices, we lead.

The native backward is CUDA + float32 only; CPU, float64, uint8 and
`downsample=True` fall back to autograd over the portable path — correct, slower.

## How

The usual SSIM formulation runs five 11×11 convolutions for the five local
expectations (E[x], E[y], E[x²], E[y²], E[xy]) and materialises a dozen
full-resolution intermediates. It is bandwidth-bound on its own temporaries.

- **Separable window** — 22 MACs/px instead of 121, and exactly equal, since
  normalising in 1-D then taking the outer product *is* the 2-D window.
- **One fused CUDA kernel, zero intermediates.** A block owns a 32×64 output
  tile and streams input through shared memory: rows staged, turned into five
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
python bench/bench_fused_ssim.py   # head-to-head vs fused-ssim
```

## API

```python
mse (x, y, *, reduction="mean", dtype=None, out_dtype=torch.float64)
psnr(x, y, *, data_range=None, reduction="mean", dtype=None,
     out_dtype=torch.float64, eps=0.0)
ssim(x, y, *, data_range=None, win_size=11, sigma=1.5, K=(0.01, 0.03),
     reduction="mean", return_map=False, dtype=None, downsample=False,
     backend_hint="auto")
```

`data_range` defaults to 255 for integer input, 1.0 for float. `downsample=True`
applies MATLAB `ssim.m`'s automatic box-downsample (off by default, as in
`ssim_index.m` and every PyTorch library).

```python
crit = fa.SSIM(data_range=1.0)
loss = crit.loss(pred, target)     # 1 - SSIM, fused backward on CUDA float32
loss.backward()

sm = fa.StreamingMetrics((1, 3, 1080, 1920), device="cuda", dtype=torch.uint8)
for ref, dist in frames:
    out = sm.update(ref, dist)     # {"mse":…, "psnr":…, "ssim":…}
```

## License

Apache 2.0.

Wang, Bovik, Sheikh, Simoncelli. *Image Quality Assessment: From Error
Visibility to Structural Similarity.* IEEE TIP 13(4), 2004.
