# frame_analytics

Fast MSE, PSNR and SSIM for PyTorch — CPU and CUDA — at float64-reference
accuracy. Fused CUDA kernels for both the forward pass and the gradient, so it
works as an evaluation metric and as a training loss.

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

## Accuracy

Defaults reproduce `ssim_index.m`: 11×11 Gaussian, σ=1.5, K=(0.01, 0.03),
`valid` support. Every kernel is gated against a float64 transcription of the
paper (`python tests/validate.py`).

| implementation | abs. error vs float64 reference |
|---|---:|
| **frame_analytics** (CUDA / CPU / portable) | **3.2e-09 / 3.1e-09 / 6.9e-09** |
| fused-ssim (`padding="valid"`) | 3.3e-06 |
| pytorch-msssim | 1.0e-07 |
| kornia | 2.1e-05 |
| torchmetrics | 2.2e-05 |
| scikit-image (defaults) | 1.1e-03 |
| piq (defaults) | 8.8e-03 |
| fused-ssim (`padding="same"`, its default) | 3.0e-03 |

The bottom three aren't bugs — different windows and MATLAB-style downsampling
or zero padding. Defensible defaults, different metric. MSE/PSNR match float64
numpy exactly; the accumulator is float64 even when elementwise work is float32.

## Speed

RTX 3090, Ryzen 16 threads, torch 2.13+cu132. ms/call, lower is better.

**SSIM, CUDA** (uint8 in)

| | 512² | 1080p | 1080p ×8 | 4K | 4K ×8 |
|---|---:|---:|---:|---:|---:|
| **frame_analytics** | **0.024** | **0.085** | **0.651** | **0.324** | **2.642** |
| pytorch-msssim | 0.539 | 1.450 | 10.42 | 5.322 | 41.42 |
| kornia | 0.774 | 1.817 | 12.38 | 6.368 | 48.14 |
| torchmetrics | 0.576 | 2.275 | 17.69 | 9.031 | 71.02 |
| piq | 0.726 | 2.438 | 17.12 | 8.890 | 67.38 |

**16–32× faster**, 25.6 Gpixel/s — ~11 700 fps at 1080p, ~3 100 at 4K.

**vs [fused-ssim](https://github.com/rahul-goel/fused-ssim)** (the fastest published CUDA SSIM; float32, `padding="valid"`, `train=False`)

| | 512² | 1080p | 1080p ×8 | 4K | 4K ×8 |
|---|---:|---:|---:|---:|---:|
| **frame_analytics** | **0.057** | **0.098** | **0.706** | **0.336** | **2.788** |
| fused-ssim | 0.110 | 0.128 | 0.940 | 0.456 | 3.858 |
| speedup | 1.92× | 1.30× | 1.33× | 1.36× | 1.38× |

**SSIM, CPU**

| | 512² | 1080p | 1080p ×4 |
|---|---:|---:|---:|
| **frame_analytics** | **1.10** | **6.16** | **21.85** |
| pytorch-msssim | 3.29 | 34.31 | 165.4 |
| kornia | 3.87 | 34.19 | 173.0 |
| OpenCV tutorial recipe | 6.59 | 46.97 | — |
| scikit-image | 22.79 | 180.7 | — |

**PSNR, CUDA** — 0.019 ms at 1080p; 4K ×8 in 0.151 ms = **880 GB/s, 94% of the
card's theoretical bandwidth**, the ceiling for anything that must read both
frames. Naive torch on the same input: 2.76 ms (18× slower).

**Streaming**, 1080p RGB, host uint8 in, python float out, via CUDA graphs:
**931 fps** (1.07 ms/frame including host→device); 2 941 fps for resident tensors.

**Forward + backward** — SSIM as a training loss, `1 - ssim(x, y)` with `.backward()`

| | 512² | 512² ×8 | 1080p | 1080p ×8 | 4K | 4K ×8 |
|---|---:|---:|---:|---:|---:|---:|
| **frame_analytics** | **0.371** | **0.327** | **0.354** | **2.106** | **1.102** | **8.663** |
| fused-ssim | 0.444 | 0.535 | 0.469 | 2.630 | 1.410 | 10.741 |
| speedup | 1.20× | 1.64× | 1.32× | 1.25× | 1.28× | 1.24× |

Gradients are verified two independent ways: against autograd over the portable
path (~1e-6 relative) and against central differences on the float64 reference
forward (~1e-6 relative). The scalar-reduction backward performs no
device→host sync, so it does not stall the training pipeline — there is a test
asserting that.

The native backward is CUDA + float32 only. Anything else (CPU, float64, uint8,
`downsample=True`) falls back to autograd over the portable path, which is
correct but slower.

### Where it loses

**Small CPU PSNR.** `cv2.PSNR` on one cache-resident 1080p frame is ~2× faster;
below ~4 MB our thread-pool wakeup costs more than the arithmetic. From 4 MB up
we lead.

## How

SSIM needs five local Gaussian expectations (E[x], E[y], E[x²], E[y²], E[xy]).
The usual formulation runs five 11×11 convolutions and materialises a dozen
full-resolution intermediates; it is bandwidth-bound on its own temporaries.

- **Separable window** — 22 MACs/px instead of 121, and exactly equal, since
  normalising in 1-D then taking the outer product *is* the 2-D window.
- **One fused CUDA kernel, zero intermediates.** A block owns a 32×64 output
  tile and streams input through shared memory: rows staged, immediately turned
  into five horizontal partial sums, pushed into a ring buffer; once 11 rows are
  resident the vertical tap runs and SSIM is folded into a block accumulator.
  DRAM traffic is the two input planes plus ~30% halo. Nothing else is written.
- **Compile-time window size.** Templating on the tap count makes the ring wrap
  a compare-and-subtract. GPUs have no integer division — at runtime tap count
  this kernel paid 11 emulated modulos per pixel. Worth 1.7× at 1080p, 6× at 512².
- **Register-blocked vertical tap.** Adjacent output rows share 10 of 11 ring
  rows, so one thread owning two rows reads 60 shared words per two pixels
  instead of 110. Shared memory is the scarce resource here (an LDS instruction
  retires 32 lanes/cycle/SM against 128 for FMA). Worth another ~1.25×.
- **Device-side reductions**, including PSNR's dB conversion — at 1080p the
  metric is ~20 µs of GPU work, so extra launches were a double-digit share.
- **Mean-shift before filtering.** `E[x²]−E[x]²` is a cancellation trap in
  float32. Subtracting a constant cancels exactly out of every variance term and
  is added back into the means — algebraically identical, materially more
  accurate, and free.
- **CPU: same structure, own thread pool.** `at::parallel_for` is a compile-time
  alias for an OpenMP region; a JIT-loaded extension isn't built with `/openmp`,
  so the pragmas vanish and it silently runs single-threaded. That was 19×.
- **Backward reuses the forward's machinery.** With `a = g·∂S/∂μx`,
  `b = g·∂S/∂σxx`, `c = g·∂S/∂σxy`, the gradient collapses to three more
  Gaussian passes:
  `dL/dx = 2x'·(w⊛b) + y'·(w⊛c) + w⊛[a − 2b·μx' − c·μy']`.
  Since the window is symmetric, that scatter is the *same* tile kernel with its
  origin moved back by one halo — not a second algorithm. And `∂S/∂σyy = ∂S/∂σxx`,
  so `w⊛b` and `w⊛c` are shared between both input gradients. The moments are
  recomputed rather than saved: cheaper than storing and reloading five
  full-resolution planes.

## Install

```bash
pip install -e .                      # kernels JIT-compile on first use
python setup.py build_ext --inplace   # or ahead of time
```

torch ≥ 2.0, numpy. A C++/nvcc toolchain unlocks the native path; on Windows the
build environment is located automatically. Without one, everything still works
— the portable PyTorch path returns the same numbers. `fa.backend_status()`
reports which is live; `backend_hint="torch"` / `"native"` forces either.

```bash
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

As a training loss:

```python
crit = fa.SSIM(data_range=1.0)
loss = crit.loss(pred, target)      # 1 - SSIM, fused backward on CUDA float32
loss.backward()
```

Module forms `MSE`, `PSNR`, `SSIM` cache window and constants.
`StreamingMetrics` captures a CUDA graph for a fixed frame shape:

```python
sm = fa.StreamingMetrics((1, 3, 1080, 1920), device="cuda", dtype=torch.uint8)
for ref, dist in frames:
    out = sm.update(ref, dist)     # {"mse":…, "psnr":…, "ssim":…}
```

## License

Apache 2.0.

Wang, Bovik, Sheikh, Simoncelli. *Image Quality Assessment: From Error
Visibility to Structural Similarity.* IEEE TIP 13(4), 2004.
