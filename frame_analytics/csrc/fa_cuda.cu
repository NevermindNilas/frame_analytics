// Fused CUDA kernels for frame_analytics, behind the C ABI in fa_abi.h.
//
// The kernels are unchanged from the torch-extension version; what changed is
// the boundary. Nothing here includes torch, so the resulting binary links
// only against a statically linked CUDA runtime (which dlopens the driver) and
// the C runtime -- one .dll/.so per platform, valid for every Python and every
// torch build. Consequently:
//
//   * shapes and dtypes arrive as ints, not as torch::Tensor
//   * every output buffer, including scratch, is allocated by the caller.
//     That is not just tidiness: StreamingMetrics captures a CUDA graph, and
//     cudaMalloc is illegal during capture while torch's caching allocator is
//     capture-aware, so the allocation has to happen on the Python side
//   * the stream is a parameter, and the device ordinal replaces CUDAGuard
//   * TORCH_CHECK becomes a status code; a CUDA runtime error comes back
//     negated
//
// SSIM strategy -- one kernel, no intermediate tensors.
//
// The naive pipeline materialises five full-resolution planes (x, y, x^2, y^2,
// xy), filters each horizontally, filters each vertically, then runs a long
// elementwise epilogue: roughly 38 full-plane trips through DRAM.  SSIM is
// entirely bandwidth-bound, so that is where all the time goes.
//
// Here a block owns a 32 x 64 tile of the *output* map and streams the input
// through shared memory:
//
//   * 16 input rows are staged at a time (32 + 10 wide, for the horizontal halo)
//   * each staged row is immediately turned into its four Gaussian-weighted
//     horizontal partial sums and pushed into a 26-row ring buffer
//   * as soon as 11 rows are resident, the vertical tap runs, the SSIM value is
//     formed in registers and folded straight into a block accumulator
//
// Net DRAM traffic: the two input planes, read once each plus ~30% halo
// overlap. Nothing else is written except one partial sum per block.
//
// Shared memory: 16*42*2*4 (staging) + 26*4*32*4 (ring) = 18.2 KiB, so several
// blocks stay resident per SM.

#include "fa_abi.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <type_traits>

namespace fa {

constexpr int MAX_WIN = FA_MAX_WIN;
// Moment planes carried through the separable blur: x, y, x*y, (x-y)^2.
//
// The obvious set is five -- x, y, x^2, y^2, x*y -- but SSIM never wants
// sigma_xx and sigma_yy apart, only their sum, so one blurred plane can stand
// in for two.  The ring buffer is what this kernel is short of (see the
// vertical tap below), and dropping a plane takes 20% off both its footprint
// and its traffic, worth 10-14% end to end.
//
// Which substitute plane matters.  E[(x+y)^2] - 2*E[x*y] is the algebraically
// obvious one and it is the *worse* one: E[(x+y)^2] runs ~4x the magnitude of
// the variance it has to produce, so its rounding error arrives pre-amplified
// and the SSIM map lost a factor of ~2 in accuracy.  Blurring the difference
// instead keeps every intermediate the size of the answer,
//
//     sigma_xx + sigma_yy = E[(x-y)^2] + 2*sigma_xy - (mu_x - mu_y)^2
//
// and sigma_xy is already in hand.  On matched frames -- the case that
// actually gets measured -- x-y is near zero and this is more accurate than
// the five-plane form it replaces, not less: worst SSIM map error over the
// validate.py corpus goes 1.3e-05 -> 5.3e-06.
constexpr int NPLANE = 4;

// --------------------------------------------------------------------------
// narrow float types
//
// at::Half and at::BFloat16 came from headers this file no longer includes.
// half rides cuda_fp16's conversion; bfloat16 is just the top half of a
// float32, so it needs no header at all -- which also keeps this file
// compiling on toolkits whose cuda_bf16.h gates its conversions on sm_80.
// --------------------------------------------------------------------------

struct half_t { unsigned short bits; };
struct bf16_t { unsigned short bits; };

// --------------------------------------------------------------------------
// typed loads
// --------------------------------------------------------------------------

// `get` is the working-precision load every tiled kernel uses; `getd` is the
// widening one, for the reductions that promise float64 accuracy. They differ
// only for float64 input, where `get` would otherwise silently narrow the
// caller's data to float before the wide accumulator ever saw it.
template <typename T> struct Ld {
  __device__ static inline float get(const T* p, long i) { return static_cast<float>(p[i]); }
  __device__ static inline double getd(const T* p, long i) { return static_cast<double>(p[i]); }
};
template <> struct Ld<half_t> {
  __device__ static inline float get(const half_t* p, long i) {
    return __half2float(*reinterpret_cast<const __half*>(p + i));
  }
  __device__ static inline double getd(const half_t* p, long i) { return static_cast<double>(get(p, i)); }
};
template <> struct Ld<bf16_t> {
  __device__ static inline float get(const bf16_t* p, long i) {
    return __int_as_float(static_cast<int>(static_cast<unsigned int>(p[i].bits) << 16));
  }
  __device__ static inline double getd(const bf16_t* p, long i) { return static_cast<double>(get(p, i)); }
};

// --------------------------------------------------------------------------
// block reduction
// --------------------------------------------------------------------------

__device__ inline double block_reduce_sum(double v, double* smem) {
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int nthreads = blockDim.x * blockDim.y;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  double out = 0.0;
  if (tid == 0) {
    const int nwarps = (nthreads + 31) >> 5;
    for (int i = 0; i < nwarps; ++i) out += smem[i];
  }
  return out;
}

// Two independent sums in one pass. `smem` must hold 64 doubles: the two
// halves cannot share a buffer, because thread 0 is still reading the first
// while the other warps would already be writing the second.
__device__ inline void block_reduce_sum2(double a, double b, double* smem,
                                         double* out_a, double* out_b) {
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int nthreads = blockDim.x * blockDim.y;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    a += __shfl_down_sync(0xffffffffu, a, off);
    b += __shfl_down_sync(0xffffffffu, b, off);
  }
  if (lane == 0) { smem[warp] = a; smem[32 + warp] = b; }
  __syncthreads();
  if (tid == 0) {
    const int nwarps = (nthreads + 31) >> 5;
    double sa = 0.0, sb = 0.0;
    for (int i = 0; i < nwarps; ++i) { sa += smem[i]; sb += smem[32 + i]; }
    *out_a = sa;
    *out_b = sb;
  }
}

// Knuth's TwoSum: hi + lo carries the running sum to ~48 bits in float32
// arithmetic. A per-element float64 add costs a 1/64-rate instruction on
// GA102 -- one per output pixel was ~10% of the SSIM kernel, and enough to
// hold float MSE a third over its DRAM floor -- while a plain float sum over
// hundreds of terms would round at 2^-24 of the running total.
//
// The _rn intrinsics are load-bearing: they are never contracted into an FMA,
// so `hi + a*b` at a call site cannot be fused into a single rounding, which
// would silently break the error-free transformation. Nothing here is
// compiled with fast-math, so the compiler may not reassociate it either.
__device__ __forceinline__ void two_sum_acc(float& hi, float& lo, float v) {
  const float s = __fadd_rn(hi, v);
  const float bp = __fsub_rn(s, hi);
  lo = __fadd_rn(lo, __fadd_rn(__fsub_rn(hi, __fsub_rn(s, bp)), __fsub_rn(v, bp)));
  hi = s;
}

// --------------------------------------------------------------------------
// Final reduction
//
// Folding this into a kernel rather than doing partial.sum(1) / n on the torch
// side matters more than it looks: for a 1080p frame the metric itself is ~20
// us of GPU work, so three extra launches plus their Python dispatch is a
// double-digit percentage of the call.
//
// Block b < nrows reduces row b (one image). Block nrows reduces everything
// (the batch total), so both outputs come from a single launch.
// --------------------------------------------------------------------------

// psnr_bias == PSNR_NONE means "this is not an MSE reduction, skip the dB
// conversion". Otherwise it is 10*log10(L^2) and the kernel emits PSNR
// alongside MSE, which saves the caller two more launches for the log.
constexpr double PSNR_NONE = FA_PSNR_NONE;

// Counts are passed as counts, and the mean is a division. `sum * (1.0/n)` is
// a rounding of a rounding: it costs the last ulp, which is enough to make
// `gmsd(x, x)` come out at 3e-08 instead of 0 and to knock an exact integer
// uint8 L1 sum off by one ulp. There is one of these per output value, so the
// divide is free.
__global__ void finalize_kernel(const double* __restrict__ partial,
                                int nrows, int ncols,
                                double n_per_image, double n_total,
                                double psnr_bias,
                                double* __restrict__ out_per_image,
                                double* __restrict__ out_total,
                                double* __restrict__ out_psnr_per_image,
                                double* __restrict__ out_psnr_total) {
  __shared__ double warp_sums[32];
  const int row = blockIdx.x;
  const bool is_total = (row == nrows);
  const long begin = is_total ? 0 : static_cast<long>(row) * ncols;
  const long end = is_total ? static_cast<long>(nrows) * ncols : begin + ncols;

  double acc = 0.0;
  for (long i = begin + threadIdx.x; i < end; i += blockDim.x) acc += partial[i];
  const double s = block_reduce_sum(acc, warp_sums);
  if (threadIdx.x == 0) {
    const double v = s / (is_total ? n_total : n_per_image);
    if (is_total) { if (out_total) *out_total = v; }
    else if (out_per_image) out_per_image[row] = v;
    if (psnr_bias != PSNR_NONE) {
      // v == 0 gives log10(0) == -inf, hence +inf dB, which is correct
      const double db = psnr_bias - 10.0 * log10(v);
      if (is_total) { if (out_psnr_total) *out_psnr_total = db; }
      else if (out_psnr_per_image) out_psnr_per_image[row] = db;
    }
  }
}

// Per-*plane* (image x channel) means of two partial arrays at once. MS-SSIM
// needs its scales combined per channel before the channels are averaged, so
// unlike `finalize` this one must not fold C away.
__global__ void finalize_planes2_kernel(const double* __restrict__ pa,
                                        const double* __restrict__ pb,
                                        int ncols, double npix,
                                        double* __restrict__ oa,
                                        double* __restrict__ ob) {
  __shared__ double smem[64];
  const int plane = blockIdx.x;
  const long base = static_cast<long>(plane) * ncols;
  double a = 0.0, b = 0.0;
  for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
    a += pa[base + i];
    b += pb[base + i];
  }
  double sa = 0.0, sb = 0.0;
  block_reduce_sum2(a, b, smem, &sa, &sb);
  if (threadIdx.x == 0) {
    oa[plane] = sa / npix;
    ob[plane] = sb / npix;
  }
}

// --------------------------------------------------------------------------
// Pixel losses: MSE, L1, Charbonnier, Huber
//
// All four are the same kernel -- one grid-stride pass over both inputs, a
// per-element penalty, a block reduction -- so they share one template rather
// than four near-copies. The penalty is the only thing that differs, and it is
// a compile-time branch.
//
// MSE and L1 on uint8 have an *exact* integer form, which is why they get a
// separate accumulator type: a 4K frame is 8.3M terms, and float32 would lose
// four digits of the sum that PSNR then takes a logarithm of.
// --------------------------------------------------------------------------

// Huber written branchlessly. `0.5*min(a,d)^2 + d*max(a-d,0)` is the same
// function as the piecewise definition, but it is a couple of min/max ops
// instead of a select, which both nvcc and the CPU vectoriser prefer.
template <int OP, typename A>
__device__ inline A pix_term(A d, A param) {
  if (OP == FA_OP_MSE) return d * d;
  if (OP == FA_OP_L1) return fabs(d);
  if (OP == FA_OP_CHARB) return sqrt(d * d + param);
  const A a = fabs(d);
  const A lo = a < param ? a : param;
  return A(0.5) * lo * lo + param * (a - lo);
}

// The accumulator's width and the *term*'s width are two separate decisions,
// and conflating them is what made MSE slow.
//
// Accumulator: every narrow-input op now sums its float terms in float-float
// (TwoSum, see two_sum_acc) and widens once per thread. MSE needs it because
// its sum goes on to be a logarithm; L1/Charbonnier/Huber used a plain float
// accumulator, which at 4K x8 (~1200 terms per thread) drifted ~6e-6 relative
// from a float64 sum -- TwoSum costs a few FP32 ops on a DRAM-bound kernel and
// removes that. The cross-thread sum is float64 either way.
//
// Term: float64 only when the *input* is float64, where narrowing would throw
// away the caller's data before anything wide ever saw it. An fp16 or fp32
// residual squared in double buys no accuracy the inputs carry, and GA102
// retires float64 at 1/64 the float rate -- MSE was spending three fp64 ops per
// element (sub, mul, add) on a kernel that should be bound on DRAM, and came
// out ~3x over its bandwidth floor with fp16 input no faster than fp32.
// `param` arrives as a double and is narrowed only for the float term: a
// float64 input's Charbonnier eps^2 (1e-6 is not a float) used to be rounded
// to float32 before the float64 term ever saw it.
template <typename T, int OP>
__global__ void pixel_kernel(const T* __restrict__ a, const T* __restrict__ b,
                             double* __restrict__ partial, long n_per_image,
                             double param, int blocks_per_image) {
  __shared__ double warp_sums[32];
  constexpr bool WIDE_TERM = std::is_same<T, double>::value;
  const int img = blockIdx.y;
  const long base = static_cast<long>(img) * n_per_image;

  double acc = 0.0;
  // Narrow input: the float32 term rides a float-float (TwoSum) accumulator
  // instead of a float64 one -- same ~1e-14 relative sum, none of the
  // 1/64-rate FP64 adds.
  float thi = 0.0f, tlo = 0.0f;
  const long stride = static_cast<long>(blockDim.x) * blocks_per_image;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_per_image; i += stride) {
    if (WIDE_TERM) {
      const double d = Ld<T>::getd(a, base + i) - Ld<T>::getd(b, base + i);
      acc += pix_term<OP, double>(d, param);
    } else {
      const float d = Ld<T>::get(a, base + i) - Ld<T>::get(b, base + i);
      two_sum_acc(thi, tlo, pix_term<OP, float>(d, static_cast<float>(param)));
    }
  }
  const double mine = WIDE_TERM ? acc
                                : static_cast<double>(thi) + static_cast<double>(tlo);
  const double s = block_reduce_sum(mine, warp_sums);
  if (threadIdx.x == 0) partial[img * blocks_per_image + blockIdx.x] = s;
}

// uint8 specialisation: 4-wide vectorised loads. MSE and L1 additionally
// accumulate in 64-bit integers, so their sums are bit-exact.
template <int OP>
__global__ void pixel_kernel_u8(const uchar4* __restrict__ a, const uchar4* __restrict__ b,
                                double* __restrict__ partial, long n_vec, long tail_start,
                                const unsigned char* __restrict__ a_raw,
                                const unsigned char* __restrict__ b_raw,
                                long n_per_image, float param, int blocks_per_image) {
  __shared__ double warp_sums[32];
  constexpr bool EXACT = (OP == FA_OP_MSE || OP == FA_OP_L1);
  const int img = blockIdx.y;
  const long vec_base = static_cast<long>(img) * n_vec;

  long long iacc = 0;
  float fhi = 0.0f, flo = 0.0f;   // Charbonnier / Huber: TwoSum, as above
  const long stride = static_cast<long>(blockDim.x) * blocks_per_image;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
    const uchar4 va = a[vec_base + i];
    const uchar4 vb = b[vec_base + i];
    const int d0 = (int)va.x - (int)vb.x;
    const int d1 = (int)va.y - (int)vb.y;
    const int d2 = (int)va.z - (int)vb.z;
    const int d3 = (int)va.w - (int)vb.w;
    if (EXACT) {
      if (OP == FA_OP_MSE) {
        iacc += (long long)(d0 * d0) + (long long)(d1 * d1) +
                (long long)(d2 * d2) + (long long)(d3 * d3);
      } else {
        iacc += (long long)(abs(d0) + abs(d1) + abs(d2) + abs(d3));
      }
    } else {
      two_sum_acc(fhi, flo, pix_term<OP, float>((float)d0, param));
      two_sum_acc(fhi, flo, pix_term<OP, float>((float)d1, param));
      two_sum_acc(fhi, flo, pix_term<OP, float>((float)d2, param));
      two_sum_acc(fhi, flo, pix_term<OP, float>((float)d3, param));
    }
  }
  // ragged tail of this image
  const long img_base = static_cast<long>(img) * n_per_image;
  for (long i = tail_start + static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_per_image; i += stride) {
    const int d = (int)a_raw[img_base + i] - (int)b_raw[img_base + i];
    if (EXACT) {
      iacc += (OP == FA_OP_MSE) ? (long long)(d * d) : (long long)abs(d);
    } else {
      two_sum_acc(fhi, flo, pix_term<OP, float>((float)d, param));
    }
  }
  const double s = block_reduce_sum(
      EXACT ? static_cast<double>(iacc)
            : static_cast<double>(fhi) + static_cast<double>(flo),
      warp_sums);
  if (threadIdx.x == 0) partial[img * blocks_per_image + blockIdx.x] = s;
}

// --------------------------------------------------------------------------
// Pixel-loss backward
//
// d/dx of mean(f(x - y)) is g * f'(x - y) / n: one read of each input and one
// write per requested gradient. The autograd route over the compiled forward
// did the same arithmetic in ~1.8x the time at 4K x8 (7.6 ms against a 4.3 ms
// DRAM floor for forward + backward together).
//
// The upstream gradient is read off the device -- one float64 per image for
// reduction="none", one for "mean" -- so a training step never syncs. It is
// divided by the count in float64 and then rounded to the working precision,
// which is what autograd does with the float64 sum's gradient before handing
// it to the float32 elementwise chain; the derivative is then formed in that
// precision, as the chain does.
// --------------------------------------------------------------------------

template <int OP, typename A>
__device__ inline A pix_grad(A d, A param) {
  if (OP == FA_OP_MSE) return A(2) * d;
  if (OP == FA_OP_L1) return d > A(0) ? A(1) : (d < A(0) ? A(-1) : A(0));
  if (OP == FA_OP_CHARB) return d / sqrt(d * d + param);
  // Huber: d inside the quadratic zone (|d| <= delta, as torch.where picks
  // it at the boundary), delta * sign(d) outside
  return fabs(d) <= param ? d : (d > A(0) ? param : -param);
}

template <typename T> struct St {
  __device__ static inline void put(T* p, long i, float v) { p[i] = static_cast<T>(v); }
  __device__ static inline void putd(T* p, long i, double v) { p[i] = static_cast<T>(v); }
};
template <> struct St<half_t> {
  __device__ static inline void put(half_t* p, long i, float v) {
    *reinterpret_cast<__half*>(p + i) = __float2half_rn(v);
  }
  __device__ static inline void putd(half_t* p, long i, double v) { put(p, i, static_cast<float>(v)); }
};
template <> struct St<bf16_t> {
  // round-to-nearest-even on the top 16 bits, NaN kept quiet -- the same
  // conversion torch's float -> bfloat16 cast performs
  __device__ static inline void put(bf16_t* p, long i, float v) {
    unsigned int u = __float_as_uint(v);
    unsigned short out;
    if ((u & 0x7fffffffu) > 0x7f800000u) {
      out = static_cast<unsigned short>((u >> 16) | 0x40u);
    } else {
      u += 0x7fffu + ((u >> 16) & 1u);
      out = static_cast<unsigned short>(u >> 16);
    }
    p[i].bits = out;
  }
  __device__ static inline void putd(bf16_t* p, long i, double v) { put(p, i, static_cast<float>(v)); }
};

template <typename T, int OP>
__global__ void pixel_bwd_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                 const double* __restrict__ g, int per_image,
                                 long n_per_image, long total, double count,
                                 double param, T* __restrict__ da,
                                 T* __restrict__ db) {
  constexpr bool WIDE = std::is_same<T, double>::value;
  const long stride = static_cast<long>(blockDim.x) * gridDim.x;
  // "mean": one scale for every element, hoisted out of the loop
  const double g0 = per_image ? 0.0 : g[0] / count;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const double gd = per_image ? g[i / n_per_image] / count : g0;
    if (WIDE) {
      const double d = Ld<T>::getd(a, i) - Ld<T>::getd(b, i);
      const double v = gd * pix_grad<OP, double>(d, param);
      if (da) St<T>::putd(da, i, v);
      if (db) St<T>::putd(db, i, -v);
    } else {
      const float d = Ld<T>::get(a, i) - Ld<T>::get(b, i);
      const float v = static_cast<float>(gd) *
                      pix_grad<OP, float>(d, static_cast<float>(param));
      if (da) St<T>::put(da, i, v);
      if (db) St<T>::put(db, i, -v);
    }
  }
}

// --------------------------------------------------------------------------
// Fused SSIM
//
// Column-sliding tiles. A block of SCOL threads owns SCOL consecutive output
// columns and a strip of `tile_h` output rows; each thread owns one column and
// walks down it. Per input row:
//
//   * the block stages the row (SCOL + KW-1 wide) into shared memory
//   * each thread forms its column's KW-tap horizontal sums of the four moment
//     planes and drops them into a KW-deep ring held in *registers*
//   * once KW rows are in the ring, the vertical tap runs straight out of
//     those registers and the SSIM epilogue consumes the result
//
// The previous design (a 32x64 tile, the vertical ring in shared memory, the
// taps re-read from shared memory) was bound on LDS: ~70 shared-memory reads
// per output pixel against a pipe that retires 32 lanes/cycle/SM to the FMA
// pipe's 128. Here the only shared-memory reads left are the 2*KW of the
// horizontal tap, so the kernel becomes FMA-bound instead, at roughly a third
// of the shared-memory traffic.
//
// Register indexing has to be static for the ring to stay in registers, so the
// row loop is unrolled by KW: inside one group of KW rows, step `s` always
// writes slot `s`, and output row `r-(KW-1)` reads slots (s+1+j) % KW -- all
// compile-time constants. Rows are staged a group at a time.
//
// Every FMA chain is the same chain, in the same order, as the 0.7.0 kernel's
// -- horizontal j = 0..KW-1, vertical j = 0..KW-1, each starting from 0 -- so
// the four moments, and therefore the SSIM map and the backward's coefficient
// maps, are bit-identical to it. Only how the per-pixel values are *summed*
// changed (see TwoSum below).
// --------------------------------------------------------------------------

constexpr int SCOL = 128;  // output columns per block == threads per block

// Per-block strip height. Chosen so `tile_h + KW - 1` input rows are a whole
// number of KW-row groups (no idle tail steps), aiming at ~64 rows -- tall
// enough that the KW-1 halo rows are a small overhead -- and stepping down
// for small frames, where the grid would otherwise be too sparse to fill the
// GPU: a 512x512 plane is only four 128-column strips wide.
//
// Must depend on (Hout, Wout, KW) only: fa_cuda_ssim_workspace() has to agree
// with the launch, and it is not told N or C.
static int ssim_tile_h(int Hout, int Wout, int KW) {
  const int cols = (Wout + SCOL - 1) / SCOL;
  int target = 64;
  while (target > 8 && static_cast<long>(cols) * ((Hout + target - 1) / target) < 128)
    target >>= 1;
  const int groups = (target + KW - 1 + KW - 1) / KW;   // ceil((target+KW-1)/KW)
  return groups * KW - (KW - 1);
}

// What the tile kernel does once it has the four local moments.
constexpr int MODE_REDUCE = 0;   // fold SSIM into a block sum
constexpr int MODE_MAP = 1;      // ... and also write the SSIM map
constexpr int MODE_PREP = 2;     // write the backward's three coefficient maps
constexpr int MODE_CS = 3;       // block sums of SSIM *and* of its cs factor
constexpr int MODE_PREP_CS = 4;  // MODE_PREP with separate per-plane S/CS grads

// KW (the window size) is a template parameter, not a runtime value: the ring
// is indexed by it, the tap loops unroll on it, and all KW taps live in
// registers.
template <typename T, int OUT_MODE, int KW>
__global__ __launch_bounds__(SCOL, 6) void ssim_fused_kernel(
    const T* __restrict__ xp, const T* __restrict__ yp,
    double* __restrict__ partial,      // [planes][blocks_per_plane]
    float* __restrict__ map_out,       // optional [planes][Hout][Wout]
    const float* __restrict__ dL_dmap, // MODE_PREP: upstream grad, may be null
    const float* __restrict__ grad_scalar,  // MODE_PREP: used when dL_dmap null
    float grad_norm,                   // MODE_PREP*: 1/(npix*planes), or 1/npix
    float* __restrict__ out_b,         // MODE_PREP: g * dS/dsigma_xx
    float* __restrict__ out_c,         // MODE_PREP: g * dS/dsigma_xy
    float* __restrict__ out_tx,        // MODE_PREP: g*dS/dmu_x - 2b*mu_x' - c*mu_y'
    float* __restrict__ out_ty,        // MODE_PREP: same with x/y swapped, may be null
    int H, int W, int Hout, int Wout,
    float shift, float C1, float C2,
    const float* __restrict__ win,     // KW taps, sum 1
    int blocks_per_plane,
    double* __restrict__ partial_cs,   // MODE_CS: [planes][blocks_per_plane]
    const float* __restrict__ gs_plane,   // MODE_PREP_CS: dL/d(ssim_p), per plane
    const float* __restrict__ gcs_plane,  // MODE_PREP_CS: dL/d(cs_p), per plane
    int tile_h) {

  constexpr int HALO_K = KW - 1;
  constexpr int STAGE_WK = SCOL + HALO_K;

  __shared__ float s_x[KW][STAGE_WK];
  __shared__ float s_y[KW][STAGE_WK];
  __shared__ double s_warp[64];

  const int plane = blockIdx.z;
  const long plane_off = static_cast<long>(plane) * H * W;
  const int tile_x = blockIdx.x * SCOL;
  const int tile_y = blockIdx.y * tile_h;
  const int tx = threadIdx.x;

  float w[KW];
  #pragma unroll
  for (int j = 0; j < KW; ++j) w[j] = __ldg(win + j);

  const int out_rows = min(tile_h, Hout - tile_y);
  const int total_in_rows = out_rows + HALO_K;
  // Columns past Wout still stage and filter (they keep every thread at the
  // barriers) but never reach the epilogue.
  const bool col_ok = tile_x + tx < Wout;
  const bool col_in = tile_x + tx < W;               // staged column in-bounds
  const bool halo_t = tx < HALO_K;                   // also stages a halo column
  const bool halo_in = tile_x + SCOL + tx < W;
  const long map_col = static_cast<long>(plane) * Hout * Wout +
                       static_cast<long>(tile_y) * Wout + tile_x + tx;

  // The upstream-gradient scalars are loop-invariant; read them once.
  float g_uniform = 0.0f, gS = 0.0f, gCS = 0.0f;
  if (OUT_MODE == MODE_PREP && !dL_dmap) g_uniform = grad_scalar[0] * grad_norm;
  if (OUT_MODE == MODE_PREP_CS) {
    gS = gs_plane[plane] * grad_norm;
    gCS = gcs_plane[plane] * grad_norm;
  }

  float ring[KW][NPLANE];
  float acc_hi = 0.0f, acc_lo = 0.0f;      // SSIM (MODE_REDUCE/MAP/CS)
  float cs_hi = 0.0f, cs_lo = 0.0f;        // cs   (MODE_CS)

  for (int r0 = 0; r0 < total_in_rows; r0 += KW) {
    // ---- stage the next KW input rows ----------------------------------- //
    // Each thread stages its own column and, for the first KW-1 threads, one
    // halo column past the tile. Written out rather than as a strided loop:
    // the generic loop compiled to several peeled variants with a branch per
    // element, which was a visible share of the per-row instruction count.
    __syncthreads();   // the previous group is done reading s_x / s_y
    #pragma unroll
    for (int rr = 0; rr < KW; ++rr) {
      const int gy = tile_y + r0 + rr;
      const bool row_ok = (r0 + rr < total_in_rows) && gy < H;
      const long row_off = plane_off + static_cast<long>(gy) * W + tile_x;
      float xv = 0.0f, yv = 0.0f;
      if (row_ok && col_in) {
        xv = Ld<T>::get(xp, row_off + tx) - shift;
        yv = Ld<T>::get(yp, row_off + tx) - shift;
      }
      s_x[rr][tx] = xv;
      s_y[rr][tx] = yv;
      if (halo_t) {
        float hx = 0.0f, hy = 0.0f;
        if (row_ok && halo_in) {
          hx = Ld<T>::get(xp, row_off + SCOL + tx) - shift;
          hy = Ld<T>::get(yp, row_off + SCOL + tx) - shift;
        }
        s_x[rr][SCOL + tx] = hx;
        s_y[rr][SCOL + tx] = hy;
      }
    }
    __syncthreads();

    #pragma unroll
    for (int s = 0; s < KW; ++s) {
      const int r = r0 + s;
      if (r >= total_in_rows) break;   // block-uniform

      // ---- horizontal tap -> ring slot s --------------------------------- //
      {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        #pragma unroll
        for (int j = 0; j < KW; ++j) {
          const float xv = s_x[s][tx + j];
          const float yv = s_y[s][tx + j];
          const float dv = xv - yv;
          a0 = fmaf(w[j], xv, a0);
          a1 = fmaf(w[j], yv, a1);
          a2 = fmaf(w[j], xv * yv, a2);
          a3 = fmaf(w[j], dv * dv, a3);
        }
        ring[s][0] = a0;
        ring[s][1] = a1;
        ring[s][2] = a2;
        ring[s][3] = a3;
      }
      if (r < HALO_K) continue;        // block-uniform

      // ---- vertical tap out of the register ring + epilogue ------------- //
      // Output row o = r - (KW-1) reads input rows o..o+KW-1, which sit in
      // slots (s+1+j) % KW: r0 is a multiple of KW and -(KW-1) == 1 (mod KW).
      float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
      #pragma unroll
      for (int j = 0; j < KW; ++j) {
        const int slot = (s + 1 + j) % KW;
        b0 = fmaf(w[j], ring[slot][0], b0);
        b1 = fmaf(w[j], ring[slot][1], b1);
        b2 = fmaf(w[j], ring[slot][2], b2);
        b3 = fmaf(w[j], ring[slot][3], b3);
      }
      if (!col_ok) continue;

      const float ux = b0;                   // mean of the shifted data
      const float uy = b1;
      const float mx = ux + shift;           // true local mean
      const float my = uy + shift;
      const float sxy = b2 - ux * uy;
      const float du = ux - uy;
      const float sxx_syy = b3 + 2.0f * sxy - du * du;

      const float A1 = 2.0f * mx * my + C1;
      const float A2 = 2.0f * sxy + C2;
      const float B1 = mx * mx + my * my + C1;   // >= C1 > 0
      const float B2 = sxx_syy + C2;             // >= C2 > 0
      const long idx = map_col + static_cast<long>(r - HALO_K) * Wout;

      if (OUT_MODE == MODE_PREP_CS) {
        // Same three coefficient maps as MODE_PREP, but the upstream
        // gradient arrives as two per-plane scalars -- one for the plane's
        // mean SSIM, one for its mean cs -- because that is what MS-SSIM's
        // weighted product hands back. cs does not depend on the local
        // means at all, so only the SSIM half contributes to t_x / t_y.
        const float iB1 = 1.0f / B1;
        const float iB2 = 1.0f / B2;
        const float S = A1 * A2 * iB1 * iB2;
        const float CS = A2 * iB2;

        const float bb = -(gS * S + gCS * CS) * iB2;      // d/dsigma_xx
        const float cc = 2.0f * (gS * A1 * iB1 + gCS) * iB2;   // d/dsigma_xy
        const float d_ux = 2.0f * A2 * iB1 * iB2 * (my - mx * A1 * iB1);
        out_b[idx] = bb;
        out_c[idx] = cc;
        out_tx[idx] = gS * d_ux - 2.0f * bb * ux - cc * uy;
        if (out_ty) {
          const float d_uy = 2.0f * A2 * iB1 * iB2 * (mx - my * A1 * iB1);
          out_ty[idx] = gS * d_uy - 2.0f * bb * uy - cc * ux;
        }
      } else if (OUT_MODE == MODE_CS) {
        // One reciprocal, not two: SSIM is A1*A2/(B1*B2) and cs is A2/B2,
        // so both come off 1/(B1*B2) with a multiply. With -prec-div=true
        // a divide is a full Newton sequence, and there are two of them
        // per output pixel in the obvious formulation.
        const float inv = 1.0f / (B1 * B2);
        two_sum_acc(cs_hi, cs_lo, A2 * B1 * inv);
        two_sum_acc(acc_hi, acc_lo, A1 * A2 * inv);
      } else if (OUT_MODE == MODE_PREP) {
        const float iB1 = 1.0f / B1;
        const float iB2 = 1.0f / B2;
        const float S = A1 * A2 * iB1 * iB2;
        // Partials of S w.r.t. the independent moments (mu_x, sigma_xx,
        // sigma_xy). dS/dsigma_yy is identical to dS/dsigma_xx, which is
        // why one 'b' map serves both input gradients.
        const float d_sxy = 2.0f * A1 * iB1 * iB2;
        const float d_sxx = -S * iB2;
        const float d_ux = 2.0f * A2 * iB1 * iB2 * (my - mx * A1 * iB1);
        // The scalar-reduction case reads the upstream gradient straight
        // off the device. Pulling it to the host to pass as a launch
        // parameter would put a full sync in every training step.
        const float g = dL_dmap ? dL_dmap[idx] : g_uniform;
        const float bb = g * d_sxx;
        const float cc = g * d_sxy;
        out_b[idx] = bb;
        out_c[idx] = cc;
        // shifted means here: x(i)-mu_x is shift-invariant, and keeping the
        // magnitudes small limits cancellation when the scatter pass forms
        // 2*x'*P1 + ... + P3
        out_tx[idx] = g * d_ux - 2.0f * bb * ux - cc * uy;
        if (out_ty) {
          const float d_uy = 2.0f * A2 * iB1 * iB2 * (mx - my * A1 * iB1);
          out_ty[idx] = g * d_uy - 2.0f * bb * uy - cc * ux;
        }
      } else {
        const float v = (A1 * A2) / (B1 * B2);
        if (OUT_MODE == MODE_MAP) map_out[idx] = v;
        two_sum_acc(acc_hi, acc_lo, v);
      }
    }
  }

  const long po = static_cast<long>(plane) * blocks_per_plane +
                  blockIdx.y * gridDim.x + blockIdx.x;
  if (OUT_MODE == MODE_CS) {
    double s = 0.0, scs = 0.0;
    block_reduce_sum2(static_cast<double>(acc_hi) + static_cast<double>(acc_lo),
                      static_cast<double>(cs_hi) + static_cast<double>(cs_lo),
                      s_warp, &s, &scs);
    if (tx == 0) {
      partial[po] = s;
      partial_cs[po] = scs;
    }
  } else if (OUT_MODE != MODE_PREP && OUT_MODE != MODE_PREP_CS) {
    const double s = block_reduce_sum(
        static_cast<double>(acc_hi) + static_cast<double>(acc_lo), s_warp);
    if (tx == 0) partial[po] = s;
  }
}

// --------------------------------------------------------------------------
// Backward scatter
//
// dL/dx(i) = 2*x'(i)*(w conv b)(i) + y'(i)*(w conv c)(i) + (w conv t_x)(i)
//
// so the second half of the backward is three (or four) more separable
// Gaussian passes, from the valid HoutxWout grid back onto the full HxW grid
// with zeros outside.
//
// Because the Gaussian window is symmetric, sum_j w_j * m(i-j) is the same
// operation as the forward's sum_j w_j * m(i+j) once the staged tile origin is
// moved back by KW-1. So this kernel is the forward's column-sliding machinery
// with a shifted origin, NP planes instead of NPLANE, and no products -- not a
// separate algorithm. Same tap order as before, so dx/dy are bit-identical to
// the 0.7.0 kernel's.
// --------------------------------------------------------------------------

template <int KW, bool NEED_DY>
__global__ __launch_bounds__(SCOL, 4) void ssim_bwd_scatter_kernel(
    const float* __restrict__ xp, const float* __restrict__ yp,
    const float* __restrict__ mb, const float* __restrict__ mc,
    const float* __restrict__ mtx, const float* __restrict__ mty,
    float* __restrict__ dx, float* __restrict__ dy,
    const float* __restrict__ win,
    int H, int W, int Hout, int Wout, float shift, int tile_h) {

  constexpr int NP = NEED_DY ? 4 : 3;
  constexpr int HALO_K = KW - 1;
  constexpr int STAGE_WK = SCOL + HALO_K;

  __shared__ float s_stage[KW][NP][STAGE_WK];

  const int plane = blockIdx.z;
  const long map_off = static_cast<long>(plane) * Hout * Wout;
  const long img_off = static_cast<long>(plane) * H * W;
  const int tile_x = blockIdx.x * SCOL;
  const int tile_y = blockIdx.y * tile_h;
  const int tx = threadIdx.x;

  float w[KW];
  #pragma unroll
  for (int j = 0; j < KW; ++j) w[j] = __ldg(win + j);

  const int out_rows = min(tile_h, H - tile_y);
  const int total_in_rows = out_rows + HALO_K;
  const bool col_ok = tile_x + tx < W;
  // staged map columns for this thread: mx0 always, mx1 (>= 0) for the halo
  const int mx0 = tile_x + tx - HALO_K;
  const int mx1 = tile_x + SCOL + tx - HALO_K;
  const bool halo_t = tx < HALO_K;
  const long img_col = img_off + static_cast<long>(tile_y) * W + tile_x + tx;

  const float* __restrict__ src[4] = {mb, mc, mtx, mty};
  float ring[KW][NP];

  for (int r0 = 0; r0 < total_in_rows; r0 += KW) {
    // ---- stage map rows, origin shifted back by HALO_K, zero outside ---- //
    __syncthreads();
    #pragma unroll
    for (int rr = 0; rr < KW; ++rr) {
      const int my_ = tile_y + r0 + rr - HALO_K;
      const bool row_ok = (r0 + rr < total_in_rows) && my_ >= 0 && my_ < Hout;
      const long row_off = map_off + static_cast<long>(my_) * Wout + tile_x - HALO_K;
      const bool in0 = row_ok && mx0 >= 0 && mx0 < Wout;
      #pragma unroll
      for (int p = 0; p < NP; ++p)
        s_stage[rr][p][tx] = in0 ? src[p][row_off + tx] : 0.0f;
      if (halo_t) {
        const bool in1 = row_ok && mx1 < Wout;
        #pragma unroll
        for (int p = 0; p < NP; ++p)
          s_stage[rr][p][SCOL + tx] = in1 ? src[p][row_off + SCOL + tx] : 0.0f;
      }
    }
    __syncthreads();

    #pragma unroll
    for (int s = 0; s < KW; ++s) {
      const int r = r0 + s;
      if (r >= total_in_rows) break;   // block-uniform
      {
        float a[NP];
        #pragma unroll
        for (int p = 0; p < NP; ++p) a[p] = 0.f;
        #pragma unroll
        for (int j = 0; j < KW; ++j) {
          #pragma unroll
          for (int p = 0; p < NP; ++p) a[p] = fmaf(w[j], s_stage[s][p][tx + j], a[p]);
        }
        #pragma unroll
        for (int p = 0; p < NP; ++p) ring[s][p] = a[p];
      }
      if (r < HALO_K) continue;        // block-uniform

      float acc[NP];
      #pragma unroll
      for (int p = 0; p < NP; ++p) acc[p] = 0.f;
      #pragma unroll
      for (int j = 0; j < KW; ++j) {
        const int slot = (s + 1 + j) % KW;
        #pragma unroll
        for (int p = 0; p < NP; ++p) acc[p] = fmaf(w[j], ring[slot][p], acc[p]);
      }
      if (!col_ok) continue;

      const long o = img_col + static_cast<long>(r - HALO_K) * W;
      const float xv = xp[o] - shift;
      const float yv = yp[o] - shift;
      const float P1 = acc[0];
      const float P2 = acc[1];
      if (dx) dx[o] = 2.0f * xv * P1 + yv * P2 + acc[2];
      if (NEED_DY && dy) dy[o] = 2.0f * yv * P1 + xv * P2 + acc[3];
    }
  }
}

// --------------------------------------------------------------------------
// host-side plumbing
// --------------------------------------------------------------------------

// Every CUDA runtime error comes back to Python negated, so a positive return
// is always one of this library's own status codes.
#define FA_CUDA_TRY(EXPR)                                                      \
  do {                                                                         \
    const cudaError_t e_ = (EXPR);                                             \
    if (e_ != cudaSuccess) return -static_cast<int>(e_);                       \
  } while (0)

#define FA_LAUNCH_CHECK()                                                      \
  do {                                                                         \
    const cudaError_t e_ = cudaGetLastError();                                 \
    if (e_ != cudaSuccess) return -static_cast<int>(e_);                       \
  } while (0)

// Replaces at::cuda::CUDAGuard. Launching on a stream that belongs to a device
// other than the current one is an error, and the caller's notion of "current"
// is torch's, not ours.
struct DeviceScope {
  int prev = -1;
  bool changed = false;
  cudaError_t status = cudaSuccess;
  explicit DeviceScope(int dev) {
    status = cudaGetDevice(&prev);
    if (status == cudaSuccess && dev >= 0 && dev != prev) {
      status = cudaSetDevice(dev);
      changed = (status == cudaSuccess);
    }
  }
  ~DeviceScope() {
    if (changed) cudaSetDevice(prev);
  }
};

#define FA_DEVICE_SCOPE(DEV)                                                   \
  DeviceScope guard_(DEV);                                                     \
  if (guard_.status != cudaSuccess) return -static_cast<int>(guard_.status)

// cudaGetDeviceProperties costs tens of microseconds -- the same order as the
// kernel it is sizing -- so the one field that is actually needed is cached.
static int sm_count(int device) {
  constexpr int MAX_DEV = 64;
  static int cache[MAX_DEV] = {0};
  if (device < 0 || device >= MAX_DEV) {
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount,
                           device < 0 ? 0 : device);
    return v > 0 ? v : 1;
  }
  if (cache[device] == 0) {
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, device) !=
            cudaSuccess || v <= 0) {
      v = 1;
    }
    cache[device] = v;
  }
  return cache[device];
}

// Stands in for AT_DISPATCH_ALL_TYPES_AND2; variadic because the bodies below
// contain template argument lists, and only `...` keeps their commas intact.
#define FA_DISPATCH(CODE, ...)                                                 \
  switch (CODE) {                                                              \
    case FA_U8:   { using scalar_t = unsigned char; __VA_ARGS__ } break;       \
    case FA_F16:  { using scalar_t = half_t;        __VA_ARGS__ } break;       \
    case FA_BF16: { using scalar_t = bf16_t;        __VA_ARGS__ } break;       \
    case FA_F32:  { using scalar_t = float;         __VA_ARGS__ } break;       \
    case FA_F64:  { using scalar_t = double;        __VA_ARGS__ } break;       \
    default: return FA_E_DTYPE;                                                \
  }

static inline int finalize(const double* partial, int nrows, int ncols,
                           double n_per_image, double n_total,
                           double psnr_bias, double* out_per_image,
                           double* out_total, double* out_psnr_pi,
                           double* out_psnr_tot, cudaStream_t stream) {
  finalize_kernel<<<nrows + 1, 256, 0, stream>>>(
      partial, nrows, ncols, n_per_image, n_total, psnr_bias, out_per_image,
      out_total, out_psnr_pi, out_psnr_tot);
  FA_LAUNCH_CHECK();
  return FA_OK;
}

// The second half of every SSIM-family backward: three (or four) coefficient
// maps convolved back through the Gaussian and combined with x and y. Shared
// verbatim between the single-scale and the multi-scale entry points -- only
// how the maps were *produced* differs.
static int ssim_bwd_scatter(const float* x, const float* y, const float* ab,
                            const float* ac, const float* atx, const float* aty,
                            const float* win, int H, int W, int Hout, int Wout,
                            int K, int planes, double shift, float* dx,
                            float* dy, cudaStream_t stream) {
  // the scatter's output grid is the full HxW frame, not the valid map
  const int th = ssim_tile_h(H, W, K);
  const int gx = (W + SCOL - 1) / SCOL;
  const int gy = (H + th - 1) / th;
  dim3 block(SCOL);
  dim3 grid(gx, gy, planes);
#define FA_SCAT(KW, DY)                                                        \
  ssim_bwd_scatter_kernel<KW, DY><<<grid, block, 0, stream>>>(                 \
      x, y, ab, ac, atx, aty, dx, dy, win, H, W, Hout, Wout, (float)shift, th)
#define FA_SCAT_KW(DY)                                                         \
  switch (K) {                                                                 \
    case 11: FA_SCAT(11, DY); break;                                           \
    case 9:  FA_SCAT(9, DY);  break;                                           \
    case 7:  FA_SCAT(7, DY);  break;                                           \
    case 5:  FA_SCAT(5, DY);  break;                                           \
    case 3:  FA_SCAT(3, DY);  break;                                           \
    default: return FA_E_WINDOW;                                               \
  }
  if (dy) { FA_SCAT_KW(true); } else { FA_SCAT_KW(false); }
#undef FA_SCAT_KW
#undef FA_SCAT
  FA_LAUNCH_CHECK();
  return FA_OK;
}

// --------------------------------------------------------------------------
// GMSD
//
// Optional 2x box downsample, a Prewitt pair, the gradient-magnitude
// similarity map, and its first two moments -- all in one kernel, so the only
// DRAM traffic is reading the two input planes.
//
// The naive route materialises six full-resolution intermediates (four
// directional gradients and two magnitudes) before it can even start the
// similarity map. That is what every existing implementation does, and it is
// six times the bandwidth of the thing it is computing.
//
// The map is the input to a *standard deviation*, so both moments are
// accumulated in the same pass: sum(q) and sum(q^2) ride the same block
// reduction, and the variance is formed once per image at the end.
//
// Same column-sliding layout as the SSIM kernel: one thread per output
// column, walking down a strip, with the three rows the Prewitt pair needs
// kept in a register ring (row loop unrolled by 3 so the ring indices are
// static). The 0.7.0 kernel gave each thread 4 pixels of a 32x32 tile and
// summed them in float64 -- three FP64 ops per pixel plus a 5-level double
// shuffle reduction per 4 pixels, on a GPU that retires FP64 at 1/64 rate.
// That, not DRAM, was what it was waiting on. Here each thread sums a whole
// strip in float-float (TwoSum, with q^2 split exactly by an FMA), and goes
// to float64 once per thread.
//
// The Prewitt arithmetic and its order are unchanged, so the q map is
// bit-identical to the 0.7.0 kernel's.
// --------------------------------------------------------------------------

constexpr int GCOL = 128;   // output columns per block == threads per block

// Strip height: as ssim_tile_h, with 3-row groups and a 2-row halo. Depends
// on the output grid only, so fa_cuda_gmsd_workspace() agrees with the launch.
static int gmsd_tile_h(int Hout, int Wout) {
  const int cols = (Wout + GCOL - 1) / GCOL;
  // The halo is only two rows here, so short strips are cheap and a small
  // frame is better served by many of them: below ~256 strips per plane the
  // grid cannot hide this kernel's latency (a 512x512 frame downsampled is
  // two strips wide).
  int target = 64;
  while (target > 4 && static_cast<long>(cols) * ((Hout + target - 1) / target) < 256)
    target >>= 1;
  const int groups = (target + 2 + 2) / 3;   // ceil((target + 2) / 3)
  return groups * 3 - 2;
}

template <typename T, bool DOWN>
__device__ __forceinline__ float gmsd_load(const T* __restrict__ p, long o, int W) {
  if (DOWN) {
    return 0.25f * (Ld<T>::get(p, o) + Ld<T>::get(p, o + 1) +
                    Ld<T>::get(p, o + W) + Ld<T>::get(p, o + W + 1));
  }
  return Ld<T>::get(p, o);
}

template <typename T, bool DOWN>
__global__ __launch_bounds__(GCOL) void gmsd_kernel(
    const T* __restrict__ xp, const T* __restrict__ yp,
    double* __restrict__ psum, double* __restrict__ psumsq,
    int H, int W, int Hd, int Wd, int Hout, int Wout,
    float Tc, float eps, int blocks_per_plane, int tile_h) {

  constexpr int SW = GCOL + 2;
  __shared__ float s_x[3][SW];
  __shared__ float s_y[3][SW];
  __shared__ double s_warp[64];

  const int plane = blockIdx.z;
  const long plane_off = static_cast<long>(plane) * H * W;
  const int tile_x = blockIdx.x * GCOL;
  const int tile_y = blockIdx.y * tile_h;
  const int tx = threadIdx.x;

  const int out_rows = min(tile_h, Hout - tile_y);
  const int total_in_rows = out_rows + 2;
  const bool col_ok = tile_x + tx < Wout;
  const bool col_in = tile_x + tx < Wd;
  const bool halo_t = tx < 2;
  const bool halo_in = tile_x + GCOL + tx < Wd;
  // Input column of this thread's staged sample on the (possibly
  // downsampled) grid: 2*gx in the full-resolution frame when DOWN.
  const int in_col = DOWN ? 2 * (tile_x + tx) : tile_x + tx;
  const int in_col_h = DOWN ? 2 * (tile_x + GCOL + tx) : tile_x + GCOL + tx;

  float rx[3][3], ry[3][3];       // [ring slot][column offset 0..2]
  float s_hi = 0.f, s_lo = 0.f;   // sum(q)
  float q_hi = 0.f, q_lo = 0.f;   // sum(q^2)
  constexpr float THIRD = 1.0f / 3.0f;

  for (int r0 = 0; r0 < total_in_rows; r0 += 3) {
    __syncthreads();
    #pragma unroll
    for (int rr = 0; rr < 3; ++rr) {
      const int gy = tile_y + r0 + rr;
      const bool row_ok = (r0 + rr < total_in_rows) && gy < Hd;
      const long row_off = plane_off + static_cast<long>(DOWN ? 2 * gy : gy) * W;
      float xv = 0.0f, yv = 0.0f;
      if (row_ok && col_in) {
        xv = gmsd_load<T, DOWN>(xp, row_off + in_col, W);
        yv = gmsd_load<T, DOWN>(yp, row_off + in_col, W);
      }
      s_x[rr][tx] = xv;
      s_y[rr][tx] = yv;
      if (halo_t) {
        float hx = 0.0f, hy = 0.0f;
        if (row_ok && halo_in) {
          hx = gmsd_load<T, DOWN>(xp, row_off + in_col_h, W);
          hy = gmsd_load<T, DOWN>(yp, row_off + in_col_h, W);
        }
        s_x[rr][GCOL + tx] = hx;
        s_y[rr][GCOL + tx] = hy;
      }
    }
    __syncthreads();

    #pragma unroll
    for (int s = 0; s < 3; ++s) {
      const int r = r0 + s;
      if (r >= total_in_rows) break;   // block-uniform
      #pragma unroll
      for (int c = 0; c < 3; ++c) {
        rx[s][c] = s_x[s][tx + c];
        ry[s][c] = s_y[s][tx + c];
      }
      if (r < 2 || !col_ok) continue;

      // output row r-2 reads input rows r-2, r-1, r: slots s+1, s+2, s (mod 3)
      const int a = (s + 1) % 3, b = (s + 2) % 3, c2 = s;
      float g[2];
      #pragma unroll
      for (int which = 0; which < 2; ++which) {
        const float(*v)[3] = which ? ry : rx;
        const float dx = ((v[a][0] - v[a][2]) +
                          (v[b][0] - v[b][2]) +
                          (v[c2][0] - v[c2][2])) * THIRD;
        const float dy = ((v[a][0] - v[c2][0]) +
                          (v[a][1] - v[c2][1]) +
                          (v[a][2] - v[c2][2])) * THIRD;
        g[which] = sqrtf(fmaf(dx, dx, fmaf(dy, dy, eps)));
      }
      const float q = (2.0f * g[0] * g[1] + Tc) /
                      (g[0] * g[0] + g[1] * g[1] + Tc);
      two_sum_acc(s_hi, s_lo, q);
      // q*q split exactly into p + e, so the second moment carries the full
      // product the old float64 accumulator did
      const float p = __fmul_rn(q, q);
      two_sum_acc(q_hi, q_lo, p);
      q_lo += fmaf(q, q, -p);
    }
  }

  double s = 0.0, ssq = 0.0;
  block_reduce_sum2(static_cast<double>(s_hi) + static_cast<double>(s_lo),
                    static_cast<double>(q_hi) + static_cast<double>(q_lo),
                    s_warp, &s, &ssq);
  if (tx == 0) {
    const long o = static_cast<long>(plane) * blocks_per_plane +
                   blockIdx.y * gridDim.x + blockIdx.x;
    psum[o] = s;
    psumsq[o] = ssq;
  }
}

// --------------------------------------------------------------------------
// GMS backward: d mean(q) / d x (and / d y)
//
// q = (2 g1 g2 + T) / (g1^2 + g2^2 + T),  g = sqrt(px^2 + py^2 + eps),
// (px, py) the Prewitt pair of the (optionally 2x2-averaged) frame. So
//
//   dq/dg1 = 2 (g2 - g1) (g2 (g1 + g2) + T) / B^2,
//   dg1/dpx = px / g1,  dg1/dpy = py / g1
//
// and dL/ds is the *transposed* Prewitt correlation of the two coefficient
// maps (Gx, Gy) = (dL/dpx, dL/dpy), then the 2x2 average's adjoint (a quarter
// to each of the four source pixels; the dropped odd row/column gets zero).
//
// One kernel per 32x32 tile of the downsampled grid: stage that tile with a
// 2-pixel halo on every side, form (Gx, Gy) at every map position within 2
// above/left of the tile, then gather. Nothing full-resolution is written but
// the gradients themselves -- the portable path materialised the four
// directional gradients, both magnitudes and the map, and kept them for
// autograd. The upstream gradient is one float64 per image, read on device.
// --------------------------------------------------------------------------

constexpr int GB_T = 32;   // downsampled-grid tile edge
constexpr int GB_BY = 8;   // threads in y

template <bool DOWN, bool NEED_DY>
__global__ __launch_bounds__(GB_T * GB_BY) void gms_bwd_kernel(
    const float* __restrict__ xp, const float* __restrict__ yp,
    const double* __restrict__ g, double npix, int C,
    int H, int W, int Hd, int Wd, int Hout, int Wout, float Tc, float eps,
    float* __restrict__ dx, float* __restrict__ dy) {
  constexpr int SS = GB_T + 4;   // staged edge: 2-pixel halo each side
  constexpr int SG = GB_T + 2;   // coefficient-map edge: 2 above/left
  __shared__ float s_x[SS][SS];
  __shared__ float s_y[SS][SS];
  __shared__ float g_x1[SG][SG], g_y1[SG][SG];
  __shared__ float g_x2[NEED_DY ? SG : 1][NEED_DY ? SG : 1];
  __shared__ float g_y2[NEED_DY ? SG : 1][NEED_DY ? SG : 1];

  const int plane = blockIdx.z;
  const long plane_off = static_cast<long>(plane) * H * W;
  const int ty0 = blockIdx.y * GB_T;
  const int tx0 = blockIdx.x * GB_T;
  const int tid = threadIdx.y * GB_T + threadIdx.x;
  constexpr int NT = GB_T * GB_BY;
  const float gq = static_cast<float>(g[plane / C] / npix);
  constexpr float THIRD = 1.0f / 3.0f;

  // ---- stage rows/cols [t0-2, t0+GB_T+2) of the downsampled grid --------- //
  for (int idx = tid; idx < SS * SS; idx += NT) {
    const int r = idx / SS, c = idx - (idx / SS) * SS;
    const int gy = ty0 - 2 + r, gx = tx0 - 2 + c;
    float xv = 0.0f, yv = 0.0f;
    if (gy >= 0 && gy < Hd && gx >= 0 && gx < Wd) {
      if (DOWN) {
        const long o = plane_off + static_cast<long>(2 * gy) * W + 2 * gx;
        xv = 0.25f * (xp[o] + xp[o + 1] + xp[o + W] + xp[o + W + 1]);
        yv = 0.25f * (yp[o] + yp[o + 1] + yp[o + W] + yp[o + W + 1]);
      } else {
        const long o = plane_off + static_cast<long>(gy) * W + gx;
        xv = xp[o];
        yv = yp[o];
      }
    }
    s_x[r][c] = xv;
    s_y[r][c] = yv;
  }
  __syncthreads();

  // ---- coefficient maps at map positions [t0-2, t0+GB_T) ----------------- //
  // The forward's arithmetic for px, py, g and q, in the forward's order.
  for (int idx = tid; idx < SG * SG; idx += NT) {
    const int r = idx / SG, c = idx - (idx / SG) * SG;
    const int mr = ty0 - 2 + r, mc = tx0 - 2 + c;
    float gx1 = 0.f, gy1 = 0.f, gx2 = 0.f, gy2 = 0.f;
    if (mr >= 0 && mr < Hout && mc >= 0 && mc < Wout) {
      float px[2], py[2], gm[2];
      #pragma unroll
      for (int which = 0; which < 2; ++which) {
        const float(*v)[SS] = which ? s_y : s_x;
        px[which] = ((v[r][c] - v[r][c + 2]) + (v[r + 1][c] - v[r + 1][c + 2]) +
                     (v[r + 2][c] - v[r + 2][c + 2])) * THIRD;
        py[which] = ((v[r][c] - v[r + 2][c]) + (v[r][c + 1] - v[r + 2][c + 1]) +
                     (v[r][c + 2] - v[r + 2][c + 2])) * THIRD;
        gm[which] = sqrtf(fmaf(px[which], px[which], fmaf(py[which], py[which], eps)));
      }
      // dq/dg1 = 2 (g2 B - A g1) / B^2, and the numerator factors exactly:
      //   g2 B - A g1 = (g2 - g1) (g2 (g1 + g2) + T)
      // Written that way there is no cancellation on a good match (g1 ~ g2),
      // where the textbook form subtracts two nearly equal products.
      const float B = gm[0] * gm[0] + gm[1] * gm[1] + Tc;
      const float iB2 = 1.0f / (B * B);
      const float dg = gm[1] - gm[0];
      const float sg = gm[0] + gm[1];
      const float w1 = gq * (2.0f * dg * (gm[1] * sg + Tc) * iB2) / gm[0];
      gx1 = w1 * px[0];
      gy1 = w1 * py[0];
      if (NEED_DY) {
        const float w2 = gq * (-2.0f * dg * (gm[0] * sg + Tc) * iB2) / gm[1];
        gx2 = w2 * px[1];
        gy2 = w2 * py[1];
      }
    }
    g_x1[r][c] = gx1;
    g_y1[r][c] = gy1;
    if (NEED_DY) {
      g_x2[r][c] = gx2;
      g_y2[r][c] = gy2;
    }
  }
  __syncthreads();

  // ---- transposed Prewitt + the downsample's adjoint -------------------- //
  // ds(i, j) = sum_a [Gx(i-a, j) - Gx(i-a, j-2)] / 3
  //          + sum_b [Gy(i, j-b) - Gy(i-2, j-b)] / 3
  // with map position (i-a, j-b) at local (i-t0+2-a, j-t0+2-b).
  for (int lr = threadIdx.y; lr < GB_T; lr += GB_BY) {
    const int i = ty0 + lr, j = tx0 + threadIdx.x;
    if (i >= Hd || j >= Wd) continue;
    const int r = lr + 2, c = threadIdx.x + 2;
    float d1 = ((g_x1[r][c] - g_x1[r][c - 2]) + (g_x1[r - 1][c] - g_x1[r - 1][c - 2]) +
                (g_x1[r - 2][c] - g_x1[r - 2][c - 2])) * THIRD +
               ((g_y1[r][c] - g_y1[r - 2][c]) + (g_y1[r][c - 1] - g_y1[r - 2][c - 1]) +
                (g_y1[r][c - 2] - g_y1[r - 2][c - 2])) * THIRD;
    float d2 = 0.f;
    if (NEED_DY) {
      d2 = ((g_x2[r][c] - g_x2[r][c - 2]) + (g_x2[r - 1][c] - g_x2[r - 1][c - 2]) +
            (g_x2[r - 2][c] - g_x2[r - 2][c - 2])) * THIRD +
           ((g_y2[r][c] - g_y2[r - 2][c]) + (g_y2[r][c - 1] - g_y2[r - 2][c - 1]) +
            (g_y2[r][c - 2] - g_y2[r - 2][c - 2])) * THIRD;
    }
    if (DOWN) {
      // a quarter to each source pixel; the last downsampled row/column also
      // owns the full-resolution row/column floor-mode pooling dropped
      const int r_end = (i == Hd - 1) ? H : 2 * i + 2;
      const int c_end = (j == Wd - 1) ? W : 2 * j + 2;
      for (int fr = 2 * i; fr < r_end; ++fr) {
        for (int fc = 2 * j; fc < c_end; ++fc) {
          const bool in = fr < 2 * i + 2 && fc < 2 * j + 2;
          const long o = plane_off + static_cast<long>(fr) * W + fc;
          if (dx) dx[o] = in ? 0.25f * d1 : 0.0f;
          if (NEED_DY && dy) dy[o] = in ? 0.25f * d2 : 0.0f;
        }
      }
    } else {
      const long o = plane_off + static_cast<long>(i) * W + j;
      if (dx) dx[o] = d1;
      if (NEED_DY && dy) dy[o] = d2;
    }
  }
}

// Block b reduces image b: mean and population standard deviation of its map.
__global__ void gmsd_finalize_kernel(const double* __restrict__ psum,
                                     const double* __restrict__ psumsq,
                                     int ncols, double npix,
                                     double* __restrict__ out_mean,
                                     double* __restrict__ out_std) {
  __shared__ double smem[64];
  const int img = blockIdx.x;
  const long base = static_cast<long>(img) * ncols;
  double a = 0.0, b = 0.0;
  for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
    a += psum[base + i];
    b += psumsq[base + i];
  }
  double sa = 0.0, sb = 0.0;
  block_reduce_sum2(a, b, smem, &sa, &sb);
  if (threadIdx.x == 0) {
    const double m = sa / npix;
    const double var = sb / npix - m * m;
    out_mean[img] = m;
    out_std[img] = sqrt(var > 0.0 ? var : 0.0);
  }
}

// --------------------------------------------------------------------------
// MS-SSIM pyramid helpers
//
// avg_pool2d(2) of x and y in one launch, widening to float on the way. The
// summation order -- ((a + b) + c) + d, then / 4 -- is ATen's (and Inductor's
// avg_pool2d lowering's), so the pyramid is bit-identical to the one the
// per-scale Python loop builds with F.avg_pool2d / the compiled cast-pool.
// Floor mode: an odd trailing row/column is dropped, as there.
// --------------------------------------------------------------------------

template <typename T>
__global__ void pool2_pair_kernel(const T* __restrict__ xs, const T* __restrict__ ys,
                                  float* __restrict__ xd, float* __restrict__ yd,
                                  int H, int W, int Ho, int Wo, long total) {
  const long stride = static_cast<long>(blockDim.x) * gridDim.x;
  const long per_plane = static_cast<long>(Ho) * Wo;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const long plane = i / per_plane;
    const long rem = i - plane * per_plane;
    const int oy = static_cast<int>(rem / Wo);
    const int ox = static_cast<int>(rem - static_cast<long>(oy) * Wo);
    const long o = plane * H * W + static_cast<long>(2 * oy) * W + 2 * ox;
    xd[i] = (((Ld<T>::get(xs, o) + Ld<T>::get(xs, o + 1)) + Ld<T>::get(xs, o + W)) +
             Ld<T>::get(xs, o + W + 1)) / 4.0f;
    yd[i] = (((Ld<T>::get(ys, o) + Ld<T>::get(ys, o + 1)) + Ld<T>::get(ys, o + W)) +
             Ld<T>::get(ys, o + W + 1)) / 4.0f;
  }
}

// The weighted product over scales and the two means, in one tiny launch
// instead of the stack / clamp / pow / prod / mean / mean chain of (N, C)-sized
// torch ops, each of which was its own kernel launch and dispatch.
//
//   per_plane = prod_i max(t_i, 0) ^ w_i,  t_i = cs_i (i < M-1), ssim_{M-1}
//   per_image = mean over channels,  total = mean over images
//
// Same operations as the Python chain, in the same order (sequential product
// from scale 0, sum over channels then divide). `t < 0 ? 0 : t` rather than
// fmax so a NaN propagates exactly as clamp_min does.
//
// One block. The float64 pows go first, one per thread: a pow is a long
// chain of 1/64-rate FP64 instructions on GA102, and fifteen of them in one
// thread's serial loop took ~80 us -- more than the whole 512x512 pyramid.
// They are written back into cs_all (which only this kernel reads after the
// per-scale passes; its last scale's slot is otherwise unused).
__global__ void ms_combine_kernel(const double* __restrict__ s_all,
                                  double* __restrict__ cs_all,
                                  const double* __restrict__ weights, int M,
                                  int N, int C, double* __restrict__ per_image,
                                  double* __restrict__ total) {
  __shared__ double s_img[256];
  const int planes = N * C;
  const long nterm = static_cast<long>(M) * planes;
  for (long k = threadIdx.x; k < nterm; k += blockDim.x) {
    const int i = static_cast<int>(k / planes);
    double t = (i == M - 1) ? s_all[k] : cs_all[k];
    t = t < 0.0 ? 0.0 : t;
    cs_all[k] = pow(t, weights[i]);
  }
  __syncthreads();
  double run = 0.0;   // this thread's share of the sum over images
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    double acc = 0.0;
    for (int c = 0; c < C; ++c) {
      const int p = n * C + c;
      double v = 1.0;
      for (int i = 0; i < M; ++i) v *= cs_all[static_cast<long>(i) * planes + p];
      acc += v;
    }
    const double m = acc / C;
    if (per_image) per_image[n] = m;
    run += m;
  }
  if (!total) return;
  // N is a batch, so a shared-memory fold in thread order is plenty, and it
  // keeps the summation order fixed.
  s_img[threadIdx.x] = run;
  __syncthreads();
  if (threadIdx.x == 0) {
    double t = 0.0;
    for (int i = 0; i < static_cast<int>(blockDim.x); ++i) t += s_img[i];
    *total = t / N;
  }
}

// Pyramid geometry, shared by the workspace query and the launch.
//
// Level i is only ever read to build level i+1, so the workspace ping-pongs:
// odd levels at offset 0, even levels after the first one. Peak footprint is
// levels 1 + 2 -- what the per-scale Python loop held at its peak too --
// rather than the whole pyramid.
struct MsLevel { int H, W, bpp; long off; };   // off: float offset into pyr ws

static int ms_levels(int H, int W, int K, int M, int planes, MsLevel* lv,
                     long* pyr_floats, int* max_bpp) {
  int mb = 0;
  long level1 = 0, level2 = 0;
  for (int i = 0; i < M; ++i) {
    const int h = i ? lv[i - 1].H / 2 : H;
    const int w = i ? lv[i - 1].W / 2 : W;
    const int Hout = h - K + 1, Wout = w - K + 1;
    if (Hout <= 0 || Wout <= 0) return FA_E_SHAPE;
    const int th = ssim_tile_h(Hout, Wout, K);
    const int bpp = ((Wout + SCOL - 1) / SCOL) * ((Hout + th - 1) / th);
    const long sz = 2L * planes * h * w;   // x then y of this level
    if (i == 1) level1 = sz;
    if (i == 2) level2 = sz;
    lv[i] = MsLevel{h, w, bpp, (i % 2 == 1) ? 0L : level1};
    mb = std::max(mb, bpp);
  }
  *pyr_floats = level1 + level2;
  *max_bpp = mb;
  return FA_OK;
}

constexpr int MS_MAX_SCALES = 16;

}  // namespace fa

// --------------------------------------------------------------------------
// C ABI
// --------------------------------------------------------------------------

using namespace fa;

extern "C" {

FA_API int fa_cuda_abi_version(void) { return FA_ABI_VERSION; }

FA_API const char* fa_cuda_error_string(int code) {
  if (code < 0) return cudaGetErrorString(static_cast<cudaError_t>(-code));
  switch (code) {
    case FA_OK: return "ok";
    case FA_E_DTYPE: return "unsupported dtype";
    case FA_E_SHAPE: return "unsupported shape";
    case FA_E_WINDOW: return "unsupported window size";
    case FA_E_OP: return "unknown pixel op";
    case FA_E_ARG: return "invalid argument";
    case FA_E_NOCUDA: return "no CUDA device";
    default: return "internal error";
  }
}

FA_API int fa_cuda_device_count(int* out_count) {
  int n = 0;
  const cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess) {
    if (out_count) *out_count = 0;
    return -static_cast<int>(e);
  }
  if (out_count) *out_count = n;
  return FA_OK;
}

FA_API int fa_cuda_pixel_workspace(int64_t n_images, int64_t n_per_image,
                                   int device, int* out_bpi) {
  if (!out_bpi) return FA_E_ARG;
  if (n_images <= 0 || n_per_image <= 0) return FA_E_SHAPE;
  const int threads = 256;
  int bpi = static_cast<int>(std::min<int64_t>(
      (n_per_image + threads - 1) / threads,
      std::max<int64_t>(1, sm_count(device) * 8 / n_images + 1)));
  if (bpi < 1) bpi = 1;
  *out_bpi = bpi;
  return FA_OK;
}

FA_API int fa_cuda_pixel_reduce(const void* a, const void* b, int dtype,
                                int64_t n_images, int64_t n_per_image,
                                int op, double param, double psnr_bias,
                                double* partial, int bpi,
                                double* out_per_image, double* out_total,
                                double* out_psnr_per_image,
                                double* out_psnr_total,
                                int device, void* stream) {
  if (!a || !b || !partial) return FA_E_ARG;
  if (op < 0 || op > 3) return FA_E_OP;
  if (n_images <= 0 || n_per_image <= 0 || bpi <= 0) return FA_E_SHAPE;
  FA_DEVICE_SCOPE(device);
  cudaStream_t st = static_cast<cudaStream_t>(stream);

  const int threads = 256;
  dim3 grid(bpi, static_cast<unsigned>(n_images));
  const float p = static_cast<float>(param);

#define FA_PIX_U8(OP)                                                          \
  pixel_kernel_u8<OP><<<grid, threads, 0, st>>>(                               \
      reinterpret_cast<const uchar4*>(a), reinterpret_cast<const uchar4*>(b),  \
      partial, n_vec, tail, static_cast<const unsigned char*>(a),              \
      static_cast<const unsigned char*>(b), n_per_image, p, bpi)
#define FA_PIX(SC, OP)                                                         \
  pixel_kernel<SC, OP><<<grid, threads, 0, st>>>(                              \
      static_cast<const SC*>(a), static_cast<const SC*>(b), partial,           \
      n_per_image, param, bpi)
#define FA_PIX_DISPATCH(LAUNCH)                                                \
  switch (op) {                                                                \
    case FA_OP_MSE:   LAUNCH(FA_OP_MSE);   break;                              \
    case FA_OP_L1:    LAUNCH(FA_OP_L1);    break;                              \
    case FA_OP_CHARB: LAUNCH(FA_OP_CHARB); break;                              \
    default:          LAUNCH(FA_OP_HUBER); break;                              \
  }

  bool done = false;
  // The vectorised path needs each image's base to be 4-aligned, which means
  // both the stride between images *and* the pointer itself.
  //
  // A contiguous tensor does not give you the second one: a slice of a bigger
  // tensor is contiguous with a non-zero storage offset, so `x[:, :, 1:, :]`
  // on an odd-width uint8 frame hands us a pointer at offset 1. A `uchar4`
  // load from it faults, and a misaligned-address fault poisons the CUDA
  // context -- the try/except fallback in backend.py cannot rescue the
  // process, every later CUDA call in the interpreter dies too.
  if (dtype == FA_U8 && (n_per_image % 4 == 0 || n_images == 1)) {
    const uintptr_t pa = reinterpret_cast<uintptr_t>(a);
    const uintptr_t pb = reinterpret_cast<uintptr_t>(b);
    if (pa % 4 == 0 && pb % 4 == 0) {
      const long n_vec = static_cast<long>(n_per_image / 4);
      const long tail = n_vec * 4;
      FA_PIX_DISPATCH(FA_PIX_U8);
      FA_LAUNCH_CHECK();
      done = true;
    }
  }

// Hoisted out of the dispatch macro below: a preprocessor directive may not
// appear inside a macro's argument list.
#define FA_PIX_SC(OP) FA_PIX(scalar_t, OP)

  if (!done) {
    FA_DISPATCH(dtype, {
      FA_PIX_DISPATCH(FA_PIX_SC);
    })
    FA_LAUNCH_CHECK();
  }

#undef FA_PIX_SC
#undef FA_PIX_DISPATCH
#undef FA_PIX
#undef FA_PIX_U8

  return finalize(partial, static_cast<int>(n_images), bpi,
                  static_cast<double>(n_per_image),
                  static_cast<double>(n_images) * static_cast<double>(n_per_image),
                  psnr_bias, out_per_image, out_total, out_psnr_per_image,
                  out_psnr_total, st);
}

// Gradient of a pixel-loss reduction (ABI 2). g: one float64 (per_image == 0)
// or n_images of them, on the device. da / db may each be null.
FA_API int fa_cuda_pixel_backward(const void* a, const void* b, int dtype,
                                  int64_t n_images, int64_t n_per_image,
                                  int op, double param, const double* g,
                                  int per_image, void* da, void* db,
                                  int device, void* stream) {
  if (!a || !b || !g) return FA_E_ARG;
  if (!da && !db) return FA_E_ARG;
  if (op < 0 || op > 3) return FA_E_OP;
  if (n_images <= 0 || n_per_image <= 0) return FA_E_SHAPE;
  if (dtype == FA_U8) return FA_E_DTYPE;   // integers have no gradient
  FA_DEVICE_SCOPE(device);
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const long total = static_cast<long>(n_images) * n_per_image;
  const double count = per_image ? static_cast<double>(n_per_image)
                                 : static_cast<double>(total);
  const int threads = 256;
  const int blocks = static_cast<int>(
      std::min<long>((total + threads - 1) / threads, static_cast<long>(sm_count(device)) * 16));
#define FA_PBWD(SC, OP)                                                        \
  pixel_bwd_kernel<SC, OP><<<blocks, threads, 0, st>>>(                        \
      static_cast<const SC*>(a), static_cast<const SC*>(b), g, per_image,      \
      n_per_image, total, count, param, static_cast<SC*>(da),                  \
      static_cast<SC*>(db))
#define FA_PBWD_OPS(SC)                                                        \
  switch (op) {                                                                \
    case FA_OP_MSE:   FA_PBWD(SC, FA_OP_MSE);   break;                         \
    case FA_OP_L1:    FA_PBWD(SC, FA_OP_L1);    break;                         \
    case FA_OP_CHARB: FA_PBWD(SC, FA_OP_CHARB); break;                         \
    default:          FA_PBWD(SC, FA_OP_HUBER); break;                         \
  }
  switch (dtype) {
    case FA_F16:  FA_PBWD_OPS(half_t); break;
    case FA_BF16: FA_PBWD_OPS(bf16_t); break;
    case FA_F32:  FA_PBWD_OPS(float);  break;
    case FA_F64:  FA_PBWD_OPS(double); break;
    default: return FA_E_DTYPE;
  }
#undef FA_PBWD_OPS
#undef FA_PBWD
  FA_LAUNCH_CHECK();
  return FA_OK;
}

FA_API int fa_cuda_ssim_workspace(int H, int W, int K, int* out_bpp) {
  if (!out_bpp) return FA_E_ARG;
  if (K < 3 || K > MAX_WIN) return FA_E_WINDOW;
  const int Hout = H - K + 1, Wout = W - K + 1;
  if (Hout <= 0 || Wout <= 0) return FA_E_SHAPE;
  const int th = ssim_tile_h(Hout, Wout, K);
  *out_bpp = ((Wout + SCOL - 1) / SCOL) * ((Hout + th - 1) / th);
  return FA_OK;
}

FA_API int fa_cuda_ssim(const void* x, const void* y, int dtype,
                        int N, int C, int H, int W,
                        const float* win, int K,
                        double shift, double C1, double C2,
                        float* map_out,
                        double* partial, int bpp,
                        double* out_per_image, double* out_total,
                        int device, void* stream) {
  if (!x || !y || !win || !partial) return FA_E_ARG;
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return FA_E_SHAPE;
  if (K < 3 || K > MAX_WIN) return FA_E_WINDOW;
  const int Hout = H - K + 1, Wout = W - K + 1;
  if (Hout <= 0 || Wout <= 0) return FA_E_SHAPE;
  FA_DEVICE_SCOPE(device);
  cudaStream_t st = static_cast<cudaStream_t>(stream);

  const int planes = N * C;
  const int th = ssim_tile_h(Hout, Wout, K);
  const int gx = (Wout + SCOL - 1) / SCOL;
  const int gy = (Hout + th - 1) / th;
  if (bpp != gx * gy) return FA_E_ARG;

  dim3 block(SCOL);
  dim3 grid(gx, gy, planes);

#define FA_LAUNCH(SC, MODE, KW)                                                \
  ssim_fused_kernel<SC, MODE, KW><<<grid, block, 0, st>>>(                     \
      static_cast<const SC*>(x), static_cast<const SC*>(y), partial,           \
      map_out, nullptr, nullptr, 0.f, nullptr, nullptr, nullptr, nullptr,      \
      H, W, Hout, Wout, (float)shift, (float)C1, (float)C2,                    \
      win, bpp, nullptr, nullptr, nullptr, th)

#define FA_DISPATCH_KW(SC, MODE)                                               \
  switch (K) {                                                                 \
    case 11: FA_LAUNCH(SC, MODE, 11); break;                                   \
    case 9:  FA_LAUNCH(SC, MODE, 9);  break;                                   \
    case 7:  FA_LAUNCH(SC, MODE, 7);  break;                                   \
    case 5:  FA_LAUNCH(SC, MODE, 5);  break;                                   \
    case 3:  FA_LAUNCH(SC, MODE, 3);  break;                                   \
    default: return FA_E_WINDOW;                                               \
  }

  FA_DISPATCH(dtype, {
    if (map_out) { FA_DISPATCH_KW(scalar_t, MODE_MAP); }
    else         { FA_DISPATCH_KW(scalar_t, MODE_REDUCE); }
  })
  FA_LAUNCH_CHECK();

#undef FA_DISPATCH_KW
#undef FA_LAUNCH

  // Each image owns C consecutive plane-rows of `partial`, so one image's
  // contribution is a contiguous C*bpp span -- exactly what finalize() reduces.
  const double npix = static_cast<double>(Hout) * Wout;
  return finalize(partial, N, C * bpp, npix * C,
                  npix * C * static_cast<double>(N), PSNR_NONE, out_per_image,
                  out_total, nullptr, nullptr, st);
}

// --------------------------------------------------------------------------
// MS-SSIM's per-scale primitive
//
// One pass produces the plane means of *both* the SSIM map and its
// contrast-structure factor. MS-SSIM wants cs at every scale and the full SSIM
// only at the coarsest, and both fall out of the same four moments, so running
// the tile kernel twice would be pure waste: the second run would re-read both
// input planes to recompute numbers it already had in registers.
// --------------------------------------------------------------------------

// The launch half of fa_cuda_ssim_cs, shared with the pyramid entry point.
// Arguments are already validated and the device already selected.
static int ssim_cs_launch(const void* x, const void* y, int dtype, int N, int C,
                          int H, int W, const float* win, int K, double shift,
                          double C1, double C2, double* partial,
                          double* partial_cs, int bpp, double* out_s,
                          double* out_cs, cudaStream_t st) {
  const int Hout = H - K + 1, Wout = W - K + 1;
  const int planes = N * C;
  const int th = ssim_tile_h(Hout, Wout, K);
  const int gx = (Wout + SCOL - 1) / SCOL;
  const int gy = (Hout + th - 1) / th;
  if (bpp != gx * gy) return FA_E_ARG;

  dim3 block(SCOL);
  dim3 grid(gx, gy, planes);

#define FA_CS(SC, KW)                                                          \
  ssim_fused_kernel<SC, MODE_CS, KW><<<grid, block, 0, st>>>(                  \
      static_cast<const SC*>(x), static_cast<const SC*>(y), partial,           \
      nullptr, nullptr, nullptr, 0.f, nullptr, nullptr, nullptr, nullptr,      \
      H, W, Hout, Wout, (float)shift, (float)C1, (float)C2,                    \
      win, bpp, partial_cs, nullptr, nullptr, th)

  FA_DISPATCH(dtype, {
    switch (K) {
      case 11: FA_CS(scalar_t, 11); break;
      case 9:  FA_CS(scalar_t, 9);  break;
      case 7:  FA_CS(scalar_t, 7);  break;
      case 5:  FA_CS(scalar_t, 5);  break;
      case 3:  FA_CS(scalar_t, 3);  break;
      default: return FA_E_WINDOW;
    }
  })
  FA_LAUNCH_CHECK();
#undef FA_CS

  finalize_planes2_kernel<<<planes, 256, 0, st>>>(
      partial, partial_cs, bpp, static_cast<double>(Hout) * Wout, out_s, out_cs);
  FA_LAUNCH_CHECK();
  return FA_OK;
}

FA_API int fa_cuda_ssim_cs(const void* x, const void* y, int dtype,
                           int N, int C, int H, int W,
                           const float* win, int K,
                           double shift, double C1, double C2,
                           double* partial, double* partial_cs, int bpp,
                           double* out_s, double* out_cs,
                           int device, void* stream) {
  if (!x || !y || !win || !partial || !partial_cs || !out_s || !out_cs)
    return FA_E_ARG;
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return FA_E_SHAPE;
  if (K < 3 || K > MAX_WIN) return FA_E_WINDOW;
  const int Hout = H - K + 1, Wout = W - K + 1;
  if (Hout <= 0 || Wout <= 0) return FA_E_SHAPE;
  FA_DEVICE_SCOPE(device);
  return ssim_cs_launch(x, y, dtype, N, C, H, W, win, K, shift, C1, C2, partial,
                        partial_cs, bpp, out_s, out_cs,
                        static_cast<cudaStream_t>(stream));
}

// --------------------------------------------------------------------------
// MS-SSIM, whole pyramid in one call
//
// Per scale the Python loop paid two pool launches, the tile kernel, a
// finalize and ~30 us of dispatch -- at 512x512 the pyramid's host side cost
// more than its kernels. Here: one pooled (x, y) pair per scale into a
// caller-provided float workspace, the same MODE_CS tile kernel and finalize
// as fa_cuda_ssim_cs (reusing one partial buffer, since the scales run in
// stream order), and one combine kernel at the end.
// --------------------------------------------------------------------------

FA_API int fa_cuda_ms_ssim_workspace(int H, int W, int K, int C_planes,
                                     int n_scales, int64_t* out_pyr_floats,
                                     int* out_max_bpp) {
  if (!out_pyr_floats || !out_max_bpp) return FA_E_ARG;
  if (K < 3 || K > MAX_WIN) return FA_E_WINDOW;
  if (n_scales < 1 || n_scales > MS_MAX_SCALES || C_planes <= 0) return FA_E_ARG;
  MsLevel lv[MS_MAX_SCALES];
  long pf = 0;
  int mb = 0;
  const int rc = ms_levels(H, W, K, n_scales, C_planes, lv, &pf, &mb);
  if (rc != FA_OK) return rc;
  *out_pyr_floats = pf;
  *out_max_bpp = mb;
  return FA_OK;
}

FA_API int fa_cuda_ms_ssim(const void* x, const void* y, int dtype,
                           int N, int C, int H, int W,
                           const float* win, int K,
                           double shift, double C1, double C2,
                           int n_scales, const double* weights,
                           float* pyr, double* partial, double* partial_cs,
                           int max_bpp, double* s_all, double* cs_all,
                           double* out_per_image, double* out_total,
                           int device, void* stream) {
  if (!x || !y || !win || !weights || !partial || !partial_cs || !s_all ||
      !cs_all)
    return FA_E_ARG;
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return FA_E_SHAPE;
  if (K < 3 || K > MAX_WIN) return FA_E_WINDOW;
  if (n_scales < 1 || n_scales > MS_MAX_SCALES) return FA_E_ARG;
  const int planes = N * C;
  MsLevel lv[MS_MAX_SCALES];
  long pf = 0;
  int mb = 0;
  const int rc = ms_levels(H, W, K, n_scales, planes, lv, &pf, &mb);
  if (rc != FA_OK) return rc;
  if (mb != max_bpp || (pf > 0 && !pyr)) return FA_E_ARG;
  FA_DEVICE_SCOPE(device);
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const int nsm = sm_count(device);

  const void* cx = x;
  const void* cy = y;
  int cdt = dtype;
  for (int i = 0; i < n_scales; ++i) {
    if (i) {
      float* xd = pyr + lv[i].off;
      float* yd = xd + static_cast<long>(planes) * lv[i].H * lv[i].W;
      const long total = static_cast<long>(planes) * lv[i].H * lv[i].W;
      const int blocks = static_cast<int>(
          std::min<long>((total + 255) / 256, static_cast<long>(nsm) * 16));
      FA_DISPATCH(cdt, {
        pool2_pair_kernel<scalar_t><<<blocks, 256, 0, st>>>(
            static_cast<const scalar_t*>(cx), static_cast<const scalar_t*>(cy),
            xd, yd, lv[i - 1].H, lv[i - 1].W, lv[i].H, lv[i].W, total);
      })
      FA_LAUNCH_CHECK();
      cx = xd;
      cy = yd;
      cdt = FA_F32;
    }
    const int r = ssim_cs_launch(cx, cy, cdt, N, C, lv[i].H, lv[i].W, win, K,
                                 shift, C1, C2, partial, partial_cs, lv[i].bpp,
                                 s_all + static_cast<long>(i) * planes,
                                 cs_all + static_cast<long>(i) * planes, st);
    if (r != FA_OK) return r;
  }
  if (out_per_image || out_total) {
    ms_combine_kernel<<<1, 256, 0, st>>>(s_all, cs_all, weights, n_scales, N, C,
                                         out_per_image, out_total);
    FA_LAUNCH_CHECK();
  }
  return FA_OK;
}

// --------------------------------------------------------------------------
// Backward
//
// Two passes. The first recomputes the local moments (the same tile kernel as
// the forward, in MODE_PREP) and emits three coefficient maps; the second
// scatters them back through the Gaussian and combines with x and y.
//
// dL_dmap may be null, in which case every map position gets the same upstream
// gradient read off `grad_scalar` -- the mean-reduction case, where
// materialising a constant HoutxWout tensor just to read it back would be pure
// bandwidth, and where pulling the scalar to the host would sync the stream
// once per training step.
// --------------------------------------------------------------------------

FA_API int fa_cuda_ssim_backward(const float* x, const float* y,
                                 const float* dL_dmap,
                                 const float* grad_scalar,
                                 const float* win, int K,
                                 int N, int C, int H, int W,
                                 double shift, double C1, double C2,
                                 float* aux, float* dx, float* dy,
                                 int device, void* stream) {
  if (!x || !y || !win || !aux) return FA_E_ARG;
  if (!dx && !dy) return FA_E_ARG;
  if (!dL_dmap && !grad_scalar) return FA_E_ARG;
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return FA_E_SHAPE;
  if (K < 3 || K > MAX_WIN) return FA_E_WINDOW;
  const int Hout = H - K + 1, Wout = W - K + 1;
  if (Hout <= 0 || Wout <= 0) return FA_E_SHAPE;
  FA_DEVICE_SCOPE(device);
  cudaStream_t st = static_cast<cudaStream_t>(stream);

  const int planes = N * C;
  const float grad_norm =
      dL_dmap ? 0.0f
              : 1.0f / static_cast<float>(static_cast<double>(Hout) * Wout * planes);

  // b, c, t_x [, t_y] laid out as one allocation of contiguous planes
  const long aux_stride = static_cast<long>(planes) * Hout * Wout;
  float* ab = aux;
  float* ac = ab + aux_stride;
  float* atx = ac + aux_stride;
  float* aty = dy ? atx + aux_stride : nullptr;

  {
    const int th = ssim_tile_h(Hout, Wout, K);
    const int gx = (Wout + SCOL - 1) / SCOL;
    const int gy = (Hout + th - 1) / th;
    dim3 block(SCOL);
    dim3 grid(gx, gy, planes);
#define FA_PREP(KW)                                                            \
  ssim_fused_kernel<float, MODE_PREP, KW><<<grid, block, 0, st>>>(             \
      x, y, nullptr, nullptr, dL_dmap, grad_scalar, grad_norm, ab, ac, atx,    \
      aty, H, W, Hout, Wout, (float)shift, (float)C1, (float)C2, win, 0,       \
      nullptr, nullptr, nullptr, th)
    switch (K) {
      case 11: FA_PREP(11); break;
      case 9:  FA_PREP(9);  break;
      case 7:  FA_PREP(7);  break;
      case 5:  FA_PREP(5);  break;
      case 3:  FA_PREP(3);  break;
      default: return FA_E_WINDOW;
    }
#undef FA_PREP
    FA_LAUNCH_CHECK();
  }

  return ssim_bwd_scatter(x, y, ab, ac, atx, aty, win, H, W, Hout, Wout, K,
                          planes, shift, dx, dy, st);
}

// --------------------------------------------------------------------------
// MS-SSIM backward
//
// Identical machinery, different upstream: the weighted product over scales
// differentiates to one gradient for each plane's mean SSIM and one for its
// mean cs, both scalars living on the device. Passing them as pointers rather
// than as launch parameters is what keeps a training step free of
// device->host syncs.
// --------------------------------------------------------------------------

FA_API int fa_cuda_ssim_cs_backward(const float* x, const float* y,
                                    const float* gs, const float* gcs,
                                    const float* win, int K,
                                    int N, int C, int H, int W,
                                    double shift, double C1, double C2,
                                    float* aux, float* dx, float* dy,
                                    int device, void* stream) {
  if (!x || !y || !gs || !gcs || !win || !aux) return FA_E_ARG;
  if (!dx && !dy) return FA_E_ARG;
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return FA_E_SHAPE;
  if (K < 3 || K > MAX_WIN) return FA_E_WINDOW;
  const int Hout = H - K + 1, Wout = W - K + 1;
  if (Hout <= 0 || Wout <= 0) return FA_E_SHAPE;
  FA_DEVICE_SCOPE(device);
  cudaStream_t st = static_cast<cudaStream_t>(stream);

  const int planes = N * C;
  const long aux_stride = static_cast<long>(planes) * Hout * Wout;
  float* ab = aux;
  float* ac = ab + aux_stride;
  float* atx = ac + aux_stride;
  float* aty = dy ? atx + aux_stride : nullptr;
  const float grad_norm =
      1.0f / static_cast<float>(static_cast<double>(Hout) * Wout);

  {
    const int th = ssim_tile_h(Hout, Wout, K);
    const int gx = (Wout + SCOL - 1) / SCOL;
    const int gy = (Hout + th - 1) / th;
    dim3 block(SCOL);
    dim3 grid(gx, gy, planes);
#define FA_PREP_CS(KW)                                                         \
  ssim_fused_kernel<float, MODE_PREP_CS, KW><<<grid, block, 0, st>>>(          \
      x, y, nullptr, nullptr, nullptr, nullptr, grad_norm, ab, ac, atx, aty,   \
      H, W, Hout, Wout, (float)shift, (float)C1, (float)C2, win, 0, nullptr,   \
      gs, gcs, th)
    switch (K) {
      case 11: FA_PREP_CS(11); break;
      case 9:  FA_PREP_CS(9);  break;
      case 7:  FA_PREP_CS(7);  break;
      case 5:  FA_PREP_CS(5);  break;
      case 3:  FA_PREP_CS(3);  break;
      default: return FA_E_WINDOW;
    }
#undef FA_PREP_CS
    FA_LAUNCH_CHECK();
  }

  return ssim_bwd_scatter(x, y, ab, ac, atx, aty, win, H, W, Hout, Wout, K,
                          planes, shift, dx, dy, st);
}

// d(per-image mean GMS)/d(x, y) for float32 input (ABI 2). g: N float64
// upstream gradients on the device. dx / dy may each be null (not both).
FA_API int fa_cuda_gms_backward(const float* x, const float* y,
                                const double* g, int N, int C, int H, int W,
                                double Tc, double eps, int downsample,
                                float* dx, float* dy, int device, void* stream) {
  if (!x || !y || !g) return FA_E_ARG;
  if (!dx && !dy) return FA_E_ARG;
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return FA_E_SHAPE;
  const int Hd = downsample ? H / 2 : H;
  const int Wd = downsample ? W / 2 : W;
  const int Hout = Hd - 2, Wout = Wd - 2;
  if (Hout <= 0 || Wout <= 0) return FA_E_SHAPE;
  FA_DEVICE_SCOPE(device);
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const double npix = static_cast<double>(Hout) * Wout * C;
  dim3 block(GB_T, GB_BY);
  dim3 grid((Wd + GB_T - 1) / GB_T, (Hd + GB_T - 1) / GB_T, N * C);
#define FA_GMSB(DOWN, DY)                                                      \
  gms_bwd_kernel<DOWN, DY><<<grid, block, 0, st>>>(                            \
      x, y, g, npix, C, H, W, Hd, Wd, Hout, Wout, (float)Tc, (float)eps, dx, dy)
  if (downsample) {
    if (dy) { FA_GMSB(true, true); } else { FA_GMSB(true, false); }
  } else {
    if (dy) { FA_GMSB(false, true); } else { FA_GMSB(false, false); }
  }
#undef FA_GMSB
  FA_LAUNCH_CHECK();
  return FA_OK;
}

FA_API int fa_cuda_gmsd_workspace(int H, int W, int downsample, int* out_bpp) {
  if (!out_bpp) return FA_E_ARG;
  const int Hd = downsample ? H / 2 : H;
  const int Wd = downsample ? W / 2 : W;
  const int Hout = Hd - 2, Wout = Wd - 2;
  if (Hout <= 0 || Wout <= 0) return FA_E_SHAPE;
  const int th = gmsd_tile_h(Hout, Wout);
  *out_bpp = ((Wout + GCOL - 1) / GCOL) * ((Hout + th - 1) / th);
  return FA_OK;
}

FA_API int fa_cuda_gmsd(const void* x, const void* y, int dtype,
                        int N, int C, int H, int W,
                        double Tc, double eps, int downsample,
                        double* psum, double* psumsq, int bpp,
                        double* out_mean, double* out_std,
                        int device, void* stream) {
  if (!x || !y || !psum || !psumsq || !out_mean || !out_std) return FA_E_ARG;
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return FA_E_SHAPE;
  const int Hd = downsample ? H / 2 : H;
  const int Wd = downsample ? W / 2 : W;
  const int Hout = Hd - 2, Wout = Wd - 2;
  if (Hout <= 0 || Wout <= 0) return FA_E_SHAPE;
  FA_DEVICE_SCOPE(device);
  cudaStream_t st = static_cast<cudaStream_t>(stream);

  const int planes = N * C;
  const int th = gmsd_tile_h(Hout, Wout);
  const int gx = (Wout + GCOL - 1) / GCOL;
  const int gy = (Hout + th - 1) / th;
  if (bpp != gx * gy) return FA_E_ARG;

  dim3 block(GCOL);
  dim3 grid(gx, gy, planes);
  FA_DISPATCH(dtype, {
    if (downsample) {
      gmsd_kernel<scalar_t, true><<<grid, block, 0, st>>>(
          static_cast<const scalar_t*>(x), static_cast<const scalar_t*>(y),
          psum, psumsq, H, W, Hd, Wd, Hout, Wout, (float)Tc, (float)eps, bpp,
          th);
    } else {
      gmsd_kernel<scalar_t, false><<<grid, block, 0, st>>>(
          static_cast<const scalar_t*>(x), static_cast<const scalar_t*>(y),
          psum, psumsq, H, W, Hd, Wd, Hout, Wout, (float)Tc, (float)eps, bpp,
          th);
    }
  })
  FA_LAUNCH_CHECK();

  gmsd_finalize_kernel<<<N, 256, 0, st>>>(
      psum, psumsq, C * bpp, static_cast<double>(Hout) * Wout * C, out_mean,
      out_std);
  FA_LAUNCH_CHECK();
  return FA_OK;
}

}  // extern "C"
