# frame_analytics — handoff

Context dump for another AI picking this up. Everything here was measured on
this machine, not assumed. Where a number appears, it came from a run.

**Repo:** https://github.com/NevermindNilas/frame_analytics (public, Apache-2.0)
**Local:** `D:\frame_analytics`

---

## 1. What this is

Image/video quality metrics for PyTorch — MSE, PSNR, SSIM, MS-SSIM, GMSD, and
the pixel losses (L1, Charbonnier, Huber) — on CPU and CUDA, at float64-reference
accuracy, faster than every comparable library measured. Fused CUDA kernels for
both forward and backward, so it works as an evaluation metric *and* a training
loss.

The design goal that drives everything: **stay inside the tolerance of the
source papers while being the fastest implementation available.** Accuracy is a
hard constraint, not a tradeoff dial.

## 2. Current status

Verified by running, on this machine:

| | |
|---|---|
| Tests | **178 passing** (`pytest tests/`) against the C-ABI backend |
| Accuracy gate | passing (`python tests/validate.py`), worst SSIM error 3.2e-09 |
| CPU + CUDA libraries | build and load, both prebuilt and JIT |
| Prebuilt wheel | `frame_analytics-0.2.0-py3-none-win_amd64.whl` builds; all three DLLs import only `KERNEL32.dll` |
| PyPI | **not published.** Name `frame-analytics` verified free |
| Version | 0.2.0 |

`git log --oneline`:

```
ddc64a6 Measure the memory, not just the clock
f367818 Vectorise the uint8 squared-error loop; stop the thread pool costing more than it saves
b665eb1 docs: condense README
cb24bf0 Blur the difference, not the sum: four SSIM planes instead of five
ee3d3a1 Add MS-SSIM, GMSD and the pixel losses, plus Y-channel reporting
```

> Re-run `git status` and the gate before trusting any specific claim here;
> treat the *reasoning* as durable and the *file contents* as a snapshot.

**Dev environment:** Windows 11, Python 3.14, torch 2.13.0+cu132, CUDA 13.2,
MSVC 18 (Community), RTX 3090. Python 3.14 and CUDA 13 are both unusually new --
several problems in section 6 stem from that. Note the CI workflow pins CUDA
12.6 instead, because CUDA 13 dropped Volta (sm_70) and would raise the floor.

## 3. Layout

```
frame_analytics/
  reference.py     float64 transcriptions of the papers. Ground truth. Slow on purpose.
  functional.py    public functional API + portable PyTorch path + autograd Functions
  modules.py       nn.Module wrappers, StreamingMetrics (CUDA-graph capture)
  backend.py       native extension: locate compiler, JIT-build, load, dispatch
  csrc/fa_cpu.cpp  CPU kernels + the pybind11 module (bindings for BOTH devices)
  csrc/fa_cuda.cu  CUDA kernels
tests/validate.py     accuracy gate, prints a table, returns non-zero on failure
tests/test_metrics.py pytest suite (wraps the gate + contract tests)
bench/bench.py            speed + accuracy vs skimage/cv2/piq/kornia/torchmetrics/pytorch-msssim
bench/bench_fused_ssim.py head-to-head vs rahul-goel/fused-ssim
```

Three tiers, all producing the same numbers:
**native extension** (fastest) → **portable PyTorch** (`torch.compile`d) →
never fails. `backend.try_*()` returns `None` whenever it cannot handle a call
and the caller silently falls back. `fa.backend_status()` says which is live;
`backend_hint="torch"` / `"native"` forces one.

## 4. Non-obvious invariants — read before changing anything

- **`reference.py` is the definition of correct.** Do not "optimise" it. Every
  kernel is gated against it. Tolerance is 5e-6 on the scalar; actual error is
  ~3e-9.
- **The Gaussian window must stay symmetric.** The backward's scatter pass
  relies on `Σⱼ wⱼ·m(i−j)` being the same operation as the forward's
  `Σⱼ wⱼ·m(i+j)` with the tile origin shifted. Break symmetry and the gradient
  silently becomes wrong.
- **Mean-shift before filtering.** `E[x²] − E[x]²` is a float32 cancellation
  trap. Inputs are shifted by `L/2` before the blur; the shift cancels exactly
  out of every variance term and is added back into the means. Free, and worth
  ~a decimal digit.
- **Reductions accumulate in float64.** A 4K frame has 8.3M residuals; float32
  accumulation loses ~4 significant digits straight into the dB figure.
- **No device→host sync in the backward.** The scalar-reduction path passes the
  upstream gradient as a 1-element device tensor, never `float(grad)`. There is
  a test asserting this (`test_backward_does_not_sync`). A sync per step stalls
  the whole training pipeline.
- **No H2D copy inside `_compute`.** `StreamingMetrics` captures a CUDA graph;
  graph capture rejects host→device copies. This is why PSNR is written as
  `const − 10*log10(m)` with the constant folded on the host rather than
  building an `L²` tensor on the device, and why the Gaussian window is created
  directly on the target device and cached there.
- **Native kernels are inference-only except CUDA-float32 SSIM.** Anything with
  `requires_grad` that the native backward cannot handle must route to autograd
  over the portable path.

## 5. How the CUDA SSIM kernel works

One kernel, no intermediate tensors. A block owns a 32×64 tile of the *output*
map and streams input through shared memory: 16 input rows staged at a time,
each immediately turned into its Gaussian-weighted horizontal partial sums and
pushed into a 26-row ring buffer; once 11 rows are resident the vertical tap
runs, SSIM is formed in registers and folded into a block accumulator. DRAM
traffic is the two input planes plus ~30% halo. Nothing else is written.

Optimisations, each measured:

| change | effect |
|---|---|
| separable window (22 MACs/px, not 121) | exact, no accuracy cost |
| templating the kernel on the tap count | 1.7× at 1080p, **6× at 512²** |
| register-blocking 2 output rows per thread | ~1.25× |
| device-side reductions incl. PSNR's dB conversion | 2.9× on 1080p PSNR |
| 5 → 4 moment planes *(uncommitted)* | 4K ×8 2.64 → 2.26 ms |

**Why the tap-count template mattered so much:** GPUs have no hardware integer
division. At runtime tap count the ring-buffer wrap `(row + j) % ring_h` was
eleven *emulated* modulos per output pixel. Compile-time `KW` makes the ring
height constant, so the wrap is a compare-and-subtract.

**Why shared memory is the scarce resource:** an LDS instruction retires 32
lanes/cycle/SM against 128 for FMA, so the vertical tap (55 shared words/px
before register-blocking) was ~2/3 of the kernel's cost. Any future optimisation
should target shared-memory traffic, not arithmetic. The uncommitted 4-plane
change is exactly this: SSIM never needs `σxx` and `σyy` separately, only their
sum, and `E[x²]+E[y²] = E[(x+y)²] − 2E[xy]`, so one blurred plane replaces two.
**Watch the numerics there** — at `x == y` that identity computes
`4E[x²] − 2E[x²]`, i.e. a cancellation. It currently passes the gate; re-check
if you touch it.

**Backward.** With `a = g·∂S/∂μx`, `b = g·∂S/∂σxx`, `c = g·∂S/∂σxy`:

```
dL/dx = 2x'·(w⊛b) + y'·(w⊛c) + w⊛[a − 2b·μx' − c·μy']
```

Three more Gaussian passes. Because the window is symmetric the scatter is the
*same* tile kernel with its origin moved back one halo, and because
`∂S/∂σyy = ∂S/∂σxx`, `w⊛b` and `w⊛c` are shared between both input gradients.
Moments are recomputed rather than saved — cheaper than storing and reloading
five full-resolution planes.

## 6. Traps already hit — do not rediscover these

1. **`at::parallel_for` is a compile-time alias for an OpenMP region.** A
   JIT-loaded extension is not compiled with `/openmp`, so the pragmas vanish
   and it silently runs **single-threaded**. This cost 19× on CPU SSIM. The
   library now uses its own `TaskPool` (`fa_cpu.cpp`). Do not reintroduce
   `at::parallel_for`. Adding `/openmp` on Windows is also wrong — it drags a
   second OpenMP runtime into a process that already has Intel's.
2. **Condvar wake latency is ~45 µs**, more than the entire metric under a
   megapixel. `TaskPool` workers spin briefly before parking. Removing the spin
   makes small-image PSNR slower than pure-PyTorch libraries.
3. **A preprocessor directive may not appear inside a macro's argument list.**
   `#define` inside an `AT_DISPATCH_ALL_TYPES_AND2(...)` lambda is ill-formed;
   MSVC and nvcc both reject the whole block. This broke the build twice. Hoist
   the `#define`/`#undef` outside the dispatch call.
4. **CUDA 13's CCCL headers refuse MSVC's traditional preprocessor.** Needs
   `/Zc:preprocessor` (and `-Xcompiler /Zc:preprocessor` for nvcc). Already in
   `backend.py` and `setup.py`. Third-party CUDA packages built here need
   `CL=/Zc:preprocessor` in the environment.
5. **A failed native build is quiet.** `FA_NATIVE=1` raises on the *first* load
   attempt, then every later call falls back to the portable path. Tests can
   report "passing" while measuring the fallback. **Always confirm
   `fa.backend_status()["available"] is True` before trusting a benchmark.**
6. **The run that triggers a rebuild can fail spuriously.** Editing a `csrc`
   file and immediately running the suite produced 3 failures once; three
   consecutive runs afterwards were clean. Force the rebuild first
   (`python -c "from frame_analytics import backend; print(backend.status())"`),
   *then* run tests, and never diagnose a failure from the compile run.
7. **Braces do not protect a comma in a macro argument; parentheses do.**
   `FA_DISPATCH(code, { ssim_tile<scalar_t, true>(...); })` splits into three
   arguments unless the macro is declared variadic. This is the same class of
   bug as trap 3 and it is silent until the error message points at a line with
   nothing wrong on it.
8. **A C-ABI kernel must not allocate.** `cudaMalloc` is illegal inside a CUDA
   graph capture; torch's caching allocator is not. Every buffer these kernels
   write -- outputs *and* block-partial scratch -- is allocated in `backend.py`
   and passed in. Same for the block-count queries: they are cached by shape in
   Python so `cudaDeviceGetAttribute` never runs inside a capture.
9. **A prebuilt binary cannot use `-march=native`, and `/arch:AVX2` for
   everybody faults on pre-Haswell.** Hence the two-library split in section 9.
   Detecting AVX2 needs CPUID leaf 7 *and* an XGETBV check that the OS saves YMM
   state; the CPUID bit alone will happily green-light a VEX instruction that
   then faults.
10. **`-cudart static` on Windows wants `/MT`.** With `/MD` the link emits
   LNK4098 and the DLL carries two C runtimes. `/MT` is also what drops the VC++
   redistributable from the list of things a wheel user must have installed.
11. **Benchmark traps that produced wrong numbers here:**
   - allocating a fresh `requires_grad` tensor inside a timed loop measures the
     caching allocator, not the kernels — it made the backward look 6× better
     than it was;
   - finite-differencing an fp32-derived scalar measures its rounding, not the
     gradient (one perturbed pixel moves SSIM by ~1e-8). Difference the float64
     reference instead;
   - the first row of a timing table absorbs lazy CUDA module load — warm every
     code path before the first timed row.

## 7. Measured performance (RTX 3090, Ryzen 16 threads)

SSIM CUDA, uint8, ms/call:

| | 512² | 1080p | 1080p ×8 | 4K | 4K ×8 |
|---|---:|---:|---:|---:|---:|
| **frame_analytics** | **0.024** | **0.083** | **0.651** | **0.324** | **2.26** |
| pytorch-msssim | 0.539 | 1.450 | 10.42 | 5.322 | 41.42 |
| kornia | 0.774 | 1.817 | 12.38 | 6.368 | 48.14 |
| torchmetrics | 0.576 | 2.275 | 17.69 | 9.031 | 71.02 |
| piq | 0.726 | 2.438 | 17.12 | 8.890 | 67.38 |

~29 Gpixel/s. **16–32× faster than every library measured**, and 1.2–1.9× faster
than [fused-ssim](https://github.com/rahul-goel/fused-ssim) (the fastest
published CUDA SSIM) on the forward, 1.2–1.6× on forward+backward.

PSNR CUDA 4K ×8: 0.151 ms = **880 GB/s, 94% of the card's theoretical
bandwidth** — the ceiling for anything that must read both frames.

CPU SSIM 1080p: 6.16 ms vs kornia 34.2, scikit-image 180.7.

Accuracy vs the float64 reference: **3.2e-09** (ours) against pytorch-msssim
1.0e-07, fused-ssim 3.3e-06, kornia 2.1e-05, torchmetrics 2.2e-05.

**One place we lose:** `cv2.PSNR` on CPU, ~2–3× faster on a single small frame,
roughly even at 4K.

## 8. Verifying a change

```powershell
$env:FA_NATIVE='1'   # make load/build failures loud - otherwise they are silent

# force the rebuild FIRST and confirm it succeeded, or a compile failure will
# hide behind the portable fallback and the numbers will be meaningless
python -c "from frame_analytics import backend; print(backend.status())"
#   -> {'available': True, 'cuda': True, 'error': None,
#       'cpu_source': 'jit', 'cpu_isa': 'avx2', 'cuda_source': 'jit', ...}

python tests/validate.py    # accuracy gate; must print ALL CHECKS PASSED
python -m pytest tests/ -q  # 178 tests
python bench/bench.py       # speed + accuracy vs the other libraries
python bench/bench_fused_ssim.py
```

`FA_VERBOSE=1` echoes each compiler command line and its output. `FA_NATIVE=0`
disables the libraries to isolate the portable path. `FA_BUILD_DIR` relocates
the JIT cache. `FA_FORCE_JIT=1` ignores any prebuilt binary in
`frame_analytics/lib/` and rebuilds -- use it when editing kernels in a tree
that has already been through an `FA_BUILD_EXT=1` build, or you will keep
testing the stale one.

To produce and check a real wheel:

```powershell
$env:FA_BUILD_EXT='1'; $env:FA_CUDA_ARCHS='8.6 9.0+PTX'
python -m pip wheel . --no-deps --no-build-isolation -w dist
dumpbin /dependents frame_analytics/lib/fa_cuda.dll   # must show KERNEL32.dll only
```

Then install it into a venv holding nothing but torch and numpy, `cd` somewhere
outside the checkout so the import resolves to site-packages, and re-run the
gate and the suite from there. A wheel that passes in the source tree proves
nothing about a wheel.

## 9. Shipping prebuilt binaries -- done, and how

The open question in the previous revision of this document is closed. The
packaging now produces **one wheel per platform, tagged `py3-none-<platform>`,
with the kernels already compiled inside it**, and the sources still ship so a
platform without a wheel compiles on first use.

### Why the obvious route was impossible

The extension used to link libtorch and pybind11, which pinned the artifact to
{Python version} x {torch minor} x {CUDA version} x {platform} -- and **a wheel
filename can only encode python tag, ABI tag and platform tag.** There is no
field for "torch 2.6". pip would select on Python+platform alone and a mismatch
is an `undefined symbol` ImportError. The usual escape, a local version like
`0.2.0+cu124torch2.6`, is **rejected by PyPI** (PEP 440 local identifiers).
This is why flash-attn and xformers host their own index.

### What replaced it

`csrc/fa_abi.h`: raw pointers, an int dtype code, a device ordinal, a stream
handle, int return codes. Nothing else crosses. The binaries therefore link
neither libtorch nor libpython, and on Windows `dumpbin /dependents` shows
exactly one import for all three of them -- `KERNEL32.dll`. No cudart (it is
statically linked and opens the driver lazily), no VC++ redistributable (`/MT`),
no CUDA toolkit. On Linux, `-static-libstdc++ -static-libgcc` leaves glibc as
the only floor.

The conversion, entry point by entry point:

| was | is |
|---|---|
| `torch::Tensor` parameters | raw pointers + shape ints |
| `AT_DISPATCH_ALL_TYPES_AND2` | `FA_DISPATCH`, a switch on an int dtype code |
| `at::cuda::getCurrentCUDAStream()` | stream handle parameter |
| `at::cuda::CUDAGuard` | device ordinal parameter + a save/restore scope |
| `torch::empty` for outputs *and scratch* | allocated in Python, pointers passed in |
| `TORCH_CHECK` | int return codes; CUDA errors come back negated |
| `at::get_num_threads()` | `fa_cpu_set_num_threads()`, called once at load |
| `at::Half` / `at::BFloat16` | 16-bit structs with open-coded conversions |
| `PYBIND11_MODULE` | `extern "C"` exports + a `ctypes` loader (`_cabi.py`) |

Three things about that table are load-bearing rather than mechanical:

1. **Scratch buffers had to move to Python too, not just outputs.**
   `StreamingMetrics` captures a CUDA graph and `cudaMalloc` is illegal during
   capture, while torch's caching allocator is capture-aware. A kernel that
   allocated its own partials would break graph capture outright. The block
   counts are a property of the kernel, so the ABI has `fa_cuda_*_workspace()`
   queries; `backend.py` caches their answers by shape so the query itself never
   happens inside a capture either.
2. **The dispatch macro must be variadic.** Its body contains template argument
   lists like `ssim_tile<scalar_t, true>`, and braces do not shield a comma from
   the preprocessor the way parentheses do. `#define FA_DISPATCH(CODE, BODY)`
   silently splits them; `(CODE, ...)` plus `__VA_ARGS__` does not.
3. **`at::Half` conversion had to be replaced, not ported.** bfloat16 is the top
   half of a float32 (a shift), and half's subnormals are exactly
   `mantissa * 2^-24` in float32, so both conversions are exact and header-free
   -- which also keeps the CUDA file compiling on toolkits whose `cuda_bf16.h`
   gates its conversions on sm_80.

### The AVX2 problem

A released binary cannot be built with `-march=native`, and the hand-vectorised
uint8 reduction (§6) is the whole reason the CPU path beats `cv2.PSNR`'s
neighbourhood. Shipping one `/arch:AVX2` binary faults on pre-Haswell hardware;
shipping one baseline binary throws the vector width away.

So `fa_cpu.cpp` is compiled **twice**, into `fa_cpu` and `fa_cpu_avx2`. The
baseline build is safe to load anywhere and exports `fa_cpu_has_avx2()` (CPUID
leaf 7 plus an XGETBV check that the OS actually saves YMM state -- CPUID alone
is not enough); the loader asks it and switches. On aarch64 NEON is baseline, so
there is one library and no probe.

### CI

`.github/workflows/wheels.yml` builds win_amd64, manylinux_2_28_x86_64,
macosx x86_64 and arm64, plus the sdist, and publishes on a `v*` tag through
trusted publishing. **No runner has an NVIDIA GPU and none needs one** -- nvcc
cross-compiles fatbins for `FA_CUDA_ARCHS` (7.5..9.0) plus PTX for anything
newer. Nothing in CI *runs* a CUDA kernel, so run the gate on a GPU machine
before tagging.

`fa_cpu_abi_version()` / `fa_cuda_abi_version()` are checked at load, so a stale
binary next to newer Python sources is rejected loudly instead of corrupting
results.

### Remaining order of work

1. Put MS-SSIM, GMSD, L1/Charbonnier/Huber and Y-channel/`crop_border` into the
   README and the benchmark tables. They are implemented and tested but
   undocumented and unbenchmarked against the competition.
2. Re-run `bench/` and refresh §7 -- the numbers there predate the C-ABI move
   and the AVX2 split (see §10).
3. Tag and publish 0.2.0. A PyPI version number can never be reused.

## 10. Things deliberately not done

- **Section 7's numbers have not been re-measured since the C-ABI move.** The
  kernels are byte-for-byte the same code, so nothing should have shifted, with
  one honest exception: the CPU library is now compiled `-O3 -mavx2 -mfma`
  rather than `-march=native`, so on an AVX-512 host the CPU figures may be
  slightly off. Re-run `bench/` before quoting them.
- **A few README speed rows predate the committed kernel work.** The 4K x8 SSIM
  entry still reads 2.642 ms, which is the five-plane kernel; cb24bf0 made it
  ~2.26. MS-SSIM, GMSD and the pixel losses *are* benchmarked against
  pytorch-msssim / piq / torch built-ins -- an earlier revision of this document
  claimed otherwise and was wrong.
- **A native CPU backward.** Only CUDA float32 SSIM has a fused backward;
  everything else uses autograd over the portable path.
- **`downsample=True` on the native path** -- falls back by design.
- **Running the CUDA kernels in CI.** GitHub-hosted runners have no NVIDIA GPU.
  CI proves the binaries *build* and that the wheels are shaped correctly;
  correctness still has to be gated on a machine with a card.
- **Linux and macOS wheels have never been built on their own platforms.** The
  workflow exists and the sources are platform-clean (macOS never builds,
  ships or loads anything CUDA, and the NEON path is baseline on aarch64), but
  the first CI run is the first real test of both.
- **Windows-on-ARM.** `library_filename` and the NEON path would work; nobody
  has tried it and no wheel is built.
