// Fused CUDA kernels for MSE and SSIM.
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

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace fa {

constexpr int TILE_W = 32;          // output columns per block
constexpr int TILE_H = 64;          // output rows per block
constexpr int BLOCK_Y = 8;          // threads in y
constexpr int ROW_BLOCK = 2;        // output rows each thread owns (see below)
constexpr int ROWS_PER_STEP = BLOCK_Y * ROW_BLOCK;   // input rows staged per step
constexpr int MAX_WIN = 11;
constexpr int HALO = MAX_WIN - 1;             // 10
constexpr int STAGE_W = TILE_W + HALO;        // 42
constexpr int RING_H = ROWS_PER_STEP + HALO;  // 26
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
template <> struct Ld<at::Half> {
  __device__ static inline float get(const at::Half* p, long i) { return __half2float(*reinterpret_cast<const __half*>(p + i)); }
  __device__ static inline double getd(const at::Half* p, long i) { return static_cast<double>(get(p, i)); }
};
template <> struct Ld<at::BFloat16> {
  __device__ static inline float get(const at::BFloat16* p, long i) { return static_cast<float>(p[i]); }
  __device__ static inline double getd(const at::BFloat16* p, long i) { return static_cast<double>(get(p, i)); }
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
constexpr double PSNR_NONE = -1.0e300;

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
    if (is_total) *out_total = v;
    else out_per_image[row] = v;
    if (psnr_bias != PSNR_NONE) {
      // v == 0 gives log10(0) == -inf, hence +inf dB, which is correct
      const double db = psnr_bias - 10.0 * log10(v);
      if (is_total) *out_psnr_total = db;
      else out_psnr_per_image[row] = db;
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

static inline std::vector<torch::Tensor> finalize(torch::Tensor partial,
                                                  int nrows, int ncols,
                                                  double n_per_image,
                                                  double n_total,
                                                  double psnr_bias) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto opts = partial.options();
  auto per_image = torch::empty({nrows}, opts);
  auto total = torch::empty({}, opts);
  const bool want_psnr = (psnr_bias != PSNR_NONE);
  auto psnr_pi = torch::empty({want_psnr ? nrows : 0}, opts);
  auto psnr_tot = torch::empty({want_psnr ? 1 : 0}, opts);

  finalize_kernel<<<nrows + 1, 256, 0, stream>>>(
      partial.data_ptr<double>(), nrows, ncols, n_per_image, n_total,
      psnr_bias, per_image.data_ptr<double>(), total.data_ptr<double>(),
      want_psnr ? psnr_pi.data_ptr<double>() : nullptr,
      want_psnr ? psnr_tot.data_ptr<double>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (want_psnr) return {per_image, total, psnr_pi, psnr_tot.squeeze()};
  return {per_image, total};
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

constexpr int OP_MSE = 0;
constexpr int OP_L1 = 1;
constexpr int OP_CHARB = 2;      // param = eps^2
constexpr int OP_HUBER = 3;      // param = delta

// Huber written branchlessly. `0.5*min(a,d)^2 + d*max(a-d,0)` is the same
// function as the piecewise definition, but it is a couple of min/max ops
// instead of a select, which both nvcc and the CPU vectoriser prefer.
template <int OP, typename A>
__device__ inline A pix_term(A d, A param) {
  if (OP == OP_MSE) return d * d;
  if (OP == OP_L1) return fabs(d);
  if (OP == OP_CHARB) return sqrt(d * d + param);
  const A a = fabs(d);
  const A lo = a < param ? a : param;
  return A(0.5) * lo * lo + param * (a - lo);
}

// MSE and any float64 input accumulate in double; the rest ride a float
// accumulator that is widened once per thread at the end.
//
// This is not a precision shortcut, it is the difference between running and
// crawling: GA102 retires float64 adds at 1/64 the float rate, so a
// per-element `double +=` turned a kernel that should saturate DRAM into one
// bound on the FP64 pipe. Each thread sums only a few hundred terms before the
// widening, so the float accumulator's error stays around 1e-7 relative --
// three orders under the tolerance these metrics are gated at -- and the
// cross-thread sum is still exact-ish float64.
template <typename T, int OP>
__global__ void pixel_kernel(const T* __restrict__ a, const T* __restrict__ b,
                             double* __restrict__ partial, long n_per_image,
                             float param, int blocks_per_image) {
  __shared__ double warp_sums[32];
  constexpr bool WIDE = (OP == OP_MSE || std::is_same<T, double>::value);
  const int img = blockIdx.y;
  const long base = static_cast<long>(img) * n_per_image;

  double acc = 0.0;
  float facc = 0.0f;
  const long stride = static_cast<long>(blockDim.x) * blocks_per_image;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_per_image; i += stride) {
    if (WIDE) {
      const double d = Ld<T>::getd(a, base + i) - Ld<T>::getd(b, base + i);
      acc += pix_term<OP, double>(d, static_cast<double>(param));
    } else {
      const float d = Ld<T>::get(a, base + i) - Ld<T>::get(b, base + i);
      facc += pix_term<OP, float>(d, param);
    }
  }
  const double s = block_reduce_sum(WIDE ? acc : static_cast<double>(facc),
                                    warp_sums);
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
  constexpr bool EXACT = (OP == OP_MSE || OP == OP_L1);
  const int img = blockIdx.y;
  const long vec_base = static_cast<long>(img) * n_vec;

  long long iacc = 0;
  float facc = 0.0f;
  const long stride = static_cast<long>(blockDim.x) * blocks_per_image;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
    const uchar4 va = a[vec_base + i];
    const uchar4 vb = b[vec_base + i];
    const int d0 = (int)va.x - (int)vb.x;
    const int d1 = (int)va.y - (int)vb.y;
    const int d2 = (int)va.z - (int)vb.z;
    const int d3 = (int)va.w - (int)vb.w;
    if (EXACT) {
      if (OP == OP_MSE) {
        iacc += (long long)(d0 * d0) + (long long)(d1 * d1) +
                (long long)(d2 * d2) + (long long)(d3 * d3);
      } else {
        iacc += (long long)(abs(d0) + abs(d1) + abs(d2) + abs(d3));
      }
    } else {
      facc += pix_term<OP, float>((float)d0, param) +
              pix_term<OP, float>((float)d1, param) +
              pix_term<OP, float>((float)d2, param) +
              pix_term<OP, float>((float)d3, param);
    }
  }
  // ragged tail of this image
  const long img_base = static_cast<long>(img) * n_per_image;
  for (long i = tail_start + static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_per_image; i += stride) {
    const int d = (int)a_raw[img_base + i] - (int)b_raw[img_base + i];
    if (EXACT) {
      iacc += (OP == OP_MSE) ? (long long)(d * d) : (long long)abs(d);
    } else {
      facc += pix_term<OP, float>((float)d, param);
    }
  }
  const double s = block_reduce_sum(
      EXACT ? static_cast<double>(iacc) : static_cast<double>(facc), warp_sums);
  if (threadIdx.x == 0) partial[img * blocks_per_image + blockIdx.x] = s;
}

// --------------------------------------------------------------------------
// Fused SSIM
// --------------------------------------------------------------------------

// KW (the window size) is a template parameter, not a runtime value. That buys
// three things that together dominate this kernel's cost:
//   * the ring height becomes a compile-time constant, so the wrap-around is a
//     compare-and-subtract instead of an integer modulo. The GPU has no
//     hardware integer division; at runtime-KW this kernel paid 11 emulated
//     modulos per output pixel, which was the single largest cost in it.
//   * the tap loops unroll with no `j < win_size` predicate in the body
//   * the compiler can keep all 11 taps in registers
//
// Each thread owns ROW_BLOCK *consecutive* output rows rather than one. The
// vertical tap for output row o reads ring rows o..o+KW-1, so two adjacent
// output rows overlap in KW-1 of them: one thread covering both reads
// KW+ROW_BLOCK-1 rows instead of 2*KW. At KW=11 that is 60 shared-memory words
// per two pixels instead of 110. Shared-memory traffic, not arithmetic, is
// what this kernel is short of -- an LDS instruction retires 32 lanes/cycle/SM
// against 128 for FMA, so the vertical tap alone was ~2/3 of the cost.
//
// What the tile kernel does once it has the four local moments.
constexpr int MODE_REDUCE = 0;   // fold SSIM into a block sum
constexpr int MODE_MAP = 1;      // ... and also write the SSIM map
constexpr int MODE_PREP = 2;     // write the backward's three coefficient maps
constexpr int MODE_CS = 3;       // block sums of SSIM *and* of its cs factor
constexpr int MODE_PREP_CS = 4;  // MODE_PREP with separate per-plane S/CS grads

template <typename T, int OUT_MODE, int KW>
__global__ __launch_bounds__(TILE_W * BLOCK_Y) void ssim_fused_kernel(
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
    const float* __restrict__ gcs_plane) {  // MODE_PREP_CS: dL/d(cs_p), per plane

  constexpr int HALO_K = KW - 1;
  constexpr int STAGE_WK = TILE_W + HALO_K;
  constexpr int RING_HK = ROWS_PER_STEP + HALO_K;

  __shared__ float s_x[ROWS_PER_STEP][STAGE_WK];
  __shared__ float s_y[ROWS_PER_STEP][STAGE_WK];
  __shared__ float s_ring[RING_HK][NPLANE][TILE_W];
  __shared__ double s_warp[64];
  __shared__ float s_win[KW];

  const int plane = blockIdx.z;
  const long plane_off = static_cast<long>(plane) * H * W;
  const int tile_x = blockIdx.x * TILE_W;
  const int tile_y = blockIdx.y * TILE_H;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * TILE_W + tx;
  constexpr int NTHREADS = TILE_W * BLOCK_Y;

  if (tid < KW) s_win[tid] = win[tid];
  __syncthreads();

  const int out_cols = min(TILE_W, Wout - tile_x);
  const int out_rows = min(TILE_H, Hout - tile_y);
  const int total_in_rows = out_rows + HALO_K;

  double acc = 0.0;
  // MODE_CS keeps its two running sums in float. A thread owns at most
  // TILE_H/ROW_BLOCK/BLOCK_Y * ROW_BLOCK == 8 output pixels per block, and the
  // summands are in [0, 1], so the float sum is good to ~1e-7 relative -- while
  // a float64 add on GA102 retires at 1/64 the float rate, and *two* of them
  // per pixel made this kernel three times slower than the single-scale one.
  // The cross-block sum is still float64.
  float acc_cs = 0.0f;
  float acc_s = 0.0f;
  int produced = 0;

  for (int r0 = 0; r0 < total_in_rows; r0 += ROWS_PER_STEP) {
    const int nrows = min(ROWS_PER_STEP, total_in_rows - r0);

    // ---- stage nrows input rows (STAGE_WK wide) ------------------------- //
    const int need = nrows * STAGE_WK;
    for (int idx = tid; idx < need; idx += NTHREADS) {
      const int rr = idx / STAGE_WK;
      const int cc = idx - rr * STAGE_WK;
      const int gy = tile_y + r0 + rr;
      const int gx = tile_x + cc;
      float xv = 0.0f, yv = 0.0f;
      if (gy < H && gx < W) {
        const long o = plane_off + static_cast<long>(gy) * W + gx;
        xv = Ld<T>::get(xp, o) - shift;
        yv = Ld<T>::get(yp, o) - shift;
      }
      s_x[rr][cc] = xv;
      s_y[rr][cc] = yv;
    }
    __syncthreads();

    // ---- horizontal tap -> ring ---------------------------------------- //
    if (tx < out_cols) {
      #pragma unroll
      for (int k = 0; k < ROW_BLOCK; ++k) {
        const int rr = ty * ROW_BLOCK + k;
        if (rr >= nrows) break;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        #pragma unroll
        for (int j = 0; j < KW; ++j) {
          const float w = s_win[j];
          const float xv = s_x[rr][tx + j];
          const float yv = s_y[rr][tx + j];
          const float dv = xv - yv;
          a0 = fmaf(w, xv, a0);
          a1 = fmaf(w, yv, a1);
          a2 = fmaf(w, xv * yv, a2);
          a3 = fmaf(w, dv * dv, a3);
        }
        int slot = (r0 + rr) % RING_HK;
        s_ring[slot][0][tx] = a0;
        s_ring[slot][1][tx] = a1;
        s_ring[slot][2][tx] = a2;
        s_ring[slot][3][tx] = a3;
      }
    }
    __syncthreads();

    // ---- vertical tap + SSIM, for every output row now fully resident --- //
    const int ready = r0 + nrows - HALO_K;           // exclusive count
    const int lo = produced;
    const int hi = min(ready, out_rows);
    const int cnt = hi - lo;
    if (cnt > 0) {
      const int orow0 = lo + ty * ROW_BLOCK;
      if (tx < out_cols && orow0 < hi) {
        float b[ROW_BLOCK][NPLANE];
        #pragma unroll
        for (int k = 0; k < ROW_BLOCK; ++k)
          #pragma unroll
          for (int p = 0; p < NPLANE; ++p) b[k][p] = 0.f;

        const int base = orow0 % RING_HK;
        // one sweep over KW+ROW_BLOCK-1 ring rows feeds every owned output row
        #pragma unroll
        for (int t = 0; t < KW + ROW_BLOCK - 1; ++t) {
          int slot = base + t;
          if (slot >= RING_HK) slot -= RING_HK;
          const float r0v = s_ring[slot][0][tx];
          const float r1v = s_ring[slot][1][tx];
          const float r2v = s_ring[slot][2][tx];
          const float r3v = s_ring[slot][3][tx];
          #pragma unroll
          for (int k = 0; k < ROW_BLOCK; ++k) {
            const int j = t - k;
            if (j >= 0 && j < KW) {
              const float w = s_win[j];
              b[k][0] = fmaf(w, r0v, b[k][0]);
              b[k][1] = fmaf(w, r1v, b[k][1]);
              b[k][2] = fmaf(w, r2v, b[k][2]);
              b[k][3] = fmaf(w, r3v, b[k][3]);
            }
          }
        }

        #pragma unroll
        for (int k = 0; k < ROW_BLOCK; ++k) {
          const int orow = orow0 + k;
          if (orow >= hi) break;
          const float ux = b[k][0];              // mean of the shifted data
          const float uy = b[k][1];
          const float mx = ux + shift;           // true local mean
          const float my = uy + shift;
          const float sxy = b[k][2] - ux * uy;
          const float du = ux - uy;
          const float sxx_syy = b[k][3] + 2.0f * sxy - du * du;

          const float A1 = 2.0f * mx * my + C1;
          const float A2 = 2.0f * sxy + C2;
          const float B1 = mx * mx + my * my + C1;   // >= C1 > 0
          const float B2 = sxx_syy + C2;             // >= C2 > 0
          const long idx = static_cast<long>(plane) * Hout * Wout +
                           static_cast<long>(tile_y + orow) * Wout + tile_x + tx;

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
            const float gS = gs_plane[plane] * grad_norm;
            const float gCS = gcs_plane[plane] * grad_norm;

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
            acc_cs += A2 * B1 * inv;
            acc_s += A1 * A2 * inv;
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
            const float g = dL_dmap ? dL_dmap[idx] : grad_scalar[0] * grad_norm;
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
            acc += static_cast<double>(v);
          }
        }
      }
      produced = hi;
    }
    __syncthreads();
  }

  if (OUT_MODE == MODE_CS) {
    double s = 0.0, scs = 0.0;
    block_reduce_sum2(static_cast<double>(acc_s), static_cast<double>(acc_cs),
                      s_warp, &s, &scs);
    if (tid == 0) {
      const long o = static_cast<long>(plane) * blocks_per_plane +
                     blockIdx.y * gridDim.x + blockIdx.x;
      partial[o] = s;
      partial_cs[o] = scs;
    }
  } else if (OUT_MODE != MODE_PREP && OUT_MODE != MODE_PREP_CS) {
    const double s = block_reduce_sum(acc, s_warp);
    if (tid == 0) {
      partial[static_cast<long>(plane) * blocks_per_plane +
              blockIdx.y * gridDim.x + blockIdx.x] = s;
    }
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
// moved back by KW-1. So this kernel is the forward's tile machinery with a
// shifted origin, NP planes instead of NPLANE, and no products -- not a separate
// algorithm.
// --------------------------------------------------------------------------

template <int KW, bool NEED_DY>
__global__ __launch_bounds__(TILE_W * BLOCK_Y) void ssim_bwd_scatter_kernel(
    const float* __restrict__ xp, const float* __restrict__ yp,
    const float* __restrict__ mb, const float* __restrict__ mc,
    const float* __restrict__ mtx, const float* __restrict__ mty,
    float* __restrict__ dx, float* __restrict__ dy,
    const float* __restrict__ win,
    int H, int W, int Hout, int Wout, float shift) {

  constexpr int NP = NEED_DY ? 4 : 3;
  constexpr int HALO_K = KW - 1;
  constexpr int STAGE_WK = TILE_W + HALO_K;
  constexpr int RING_HK = ROWS_PER_STEP + HALO_K;

  __shared__ float s_stage[ROWS_PER_STEP][NP][STAGE_WK];
  __shared__ float s_ring[RING_HK][NP][TILE_W];
  __shared__ float s_win[KW];

  const int plane = blockIdx.z;
  const long map_off = static_cast<long>(plane) * Hout * Wout;
  const long img_off = static_cast<long>(plane) * H * W;
  const int tile_x = blockIdx.x * TILE_W;
  const int tile_y = blockIdx.y * TILE_H;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * TILE_W + tx;
  constexpr int NTHREADS = TILE_W * BLOCK_Y;

  if (tid < KW) s_win[tid] = win[tid];
  __syncthreads();

  const int out_cols = min(TILE_W, W - tile_x);
  const int out_rows = min(TILE_H, H - tile_y);
  const int total_in_rows = out_rows + HALO_K;

  const float* __restrict__ src[4] = {mb, mc, mtx, mty};
  int produced = 0;

  for (int r0 = 0; r0 < total_in_rows; r0 += ROWS_PER_STEP) {
    const int nrows = min(ROWS_PER_STEP, total_in_rows - r0);

    // ---- stage map rows, origin shifted back by HALO_K, zero outside ---- //
    const int need = nrows * STAGE_WK;
    for (int idx = tid; idx < need; idx += NTHREADS) {
      const int rr = idx / STAGE_WK;
      const int cc = idx - rr * STAGE_WK;
      const int my_ = tile_y + r0 + rr - HALO_K;
      const int mx_ = tile_x + cc - HALO_K;
      const bool inside = (my_ >= 0 && my_ < Hout && mx_ >= 0 && mx_ < Wout);
      const long o = map_off + static_cast<long>(my_) * Wout + mx_;
      #pragma unroll
      for (int p = 0; p < NP; ++p) {
        s_stage[rr][p][cc] = inside ? src[p][o] : 0.0f;
      }
    }
    __syncthreads();

    // ---- horizontal tap -> ring ---------------------------------------- //
    if (tx < out_cols) {
      #pragma unroll
      for (int k = 0; k < ROW_BLOCK; ++k) {
        const int rr = ty * ROW_BLOCK + k;
        if (rr >= nrows) break;
        float a[NP];
        #pragma unroll
        for (int p = 0; p < NP; ++p) a[p] = 0.f;
        #pragma unroll
        for (int j = 0; j < KW; ++j) {
          const float w = s_win[j];
          #pragma unroll
          for (int p = 0; p < NP; ++p) a[p] = fmaf(w, s_stage[rr][p][tx + j], a[p]);
        }
        const int slot = (r0 + rr) % RING_HK;
        #pragma unroll
        for (int p = 0; p < NP; ++p) s_ring[slot][p][tx] = a[p];
      }
    }
    __syncthreads();

    // ---- vertical tap + combine with x, y ------------------------------- //
    const int ready = r0 + nrows - HALO_K;
    const int lo = produced;
    const int hi = min(ready, out_rows);
    if (hi > lo) {
      const int orow0 = lo + ty * ROW_BLOCK;
      if (tx < out_cols && orow0 < hi) {
        float acc[ROW_BLOCK][NP];
        #pragma unroll
        for (int k = 0; k < ROW_BLOCK; ++k)
          #pragma unroll
          for (int p = 0; p < NP; ++p) acc[k][p] = 0.f;

        const int base = orow0 % RING_HK;
        #pragma unroll
        for (int t = 0; t < KW + ROW_BLOCK - 1; ++t) {
          int slot = base + t;
          if (slot >= RING_HK) slot -= RING_HK;
          float r[NP];
          #pragma unroll
          for (int p = 0; p < NP; ++p) r[p] = s_ring[slot][p][tx];
          #pragma unroll
          for (int k = 0; k < ROW_BLOCK; ++k) {
            const int j = t - k;
            if (j >= 0 && j < KW) {
              const float w = s_win[j];
              #pragma unroll
              for (int p = 0; p < NP; ++p) acc[k][p] = fmaf(w, r[p], acc[k][p]);
            }
          }
        }

        #pragma unroll
        for (int k = 0; k < ROW_BLOCK; ++k) {
          const int orow = orow0 + k;
          if (orow >= hi) break;
          const long o = img_off + static_cast<long>(tile_y + orow) * W + tile_x + tx;
          const float xv = xp[o] - shift;
          const float yv = yp[o] - shift;
          const float P1 = acc[k][0];
          const float P2 = acc[k][1];
          if (dx) dx[o] = 2.0f * xv * P1 + yv * P2 + acc[k][2];
          if (NEED_DY && dy) dy[o] = 2.0f * yv * P1 + xv * P2 + acc[k][3];
        }
      }
      produced = hi;
    }
    __syncthreads();
  }
}

// --------------------------------------------------------------------------
// launchers
// --------------------------------------------------------------------------

static inline int sm_count() {
  return at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
}

std::vector<torch::Tensor> pixel_partial(torch::Tensor a, torch::Tensor b,
                                         int64_t n_images, int64_t op,
                                         double param, double psnr_bias) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
  TORCH_CHECK(op >= 0 && op <= 3, "unknown pixel op ", op);
  a = a.contiguous();
  b = b.contiguous();
  const at::cuda::CUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const long total = a.numel();
  TORCH_CHECK(n_images > 0 && total % n_images == 0, "numel not divisible by batch");
  const long n_per_image = total / n_images;

  const int threads = 256;
  int bpi = (int)std::min<long>((n_per_image + threads - 1) / threads,
                                std::max(1, sm_count() * 8 / (int)n_images + 1));
  bpi = std::max(1, bpi);

  auto partial = torch::empty({n_images, bpi},
                              a.options().dtype(torch::kFloat64));
  dim3 grid(bpi, (unsigned)n_images);
  const float p = (float)param;

#define FA_PIX_U8(OP)                                                          \
  pixel_kernel_u8<OP><<<grid, threads, 0, stream>>>(                           \
      reinterpret_cast<const uchar4*>(a.data_ptr<uint8_t>()),                  \
      reinterpret_cast<const uchar4*>(b.data_ptr<uint8_t>()),                  \
      partial.data_ptr<double>(), n_vec, tail,                                 \
      a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(), n_per_image, p, bpi)
#define FA_PIX(SC, OP)                                                         \
  pixel_kernel<SC, OP><<<grid, threads, 0, stream>>>(                          \
      a.data_ptr<SC>(), b.data_ptr<SC>(), partial.data_ptr<double>(),          \
      n_per_image, p, bpi)
#define FA_PIX_DISPATCH(LAUNCH)                                                \
  switch ((int)op) {                                                           \
    case OP_MSE:   LAUNCH(OP_MSE);   break;                                    \
    case OP_L1:    LAUNCH(OP_L1);    break;                                    \
    case OP_CHARB: LAUNCH(OP_CHARB); break;                                    \
    default:       LAUNCH(OP_HUBER); break;                                    \
  }

  bool done = false;
  // The vectorised path needs each image's base to be 4-aligned, which means
  // both the stride between images *and* the pointer itself.
  //
  // `contiguous()` does not give you the second one: a slice of a bigger
  // tensor is contiguous with a non-zero storage offset, so `x[:, :, 1:, :]`
  // on an odd-width uint8 frame hands us a pointer at offset 1. A `uchar4`
  // load from it faults, and a misaligned-address fault poisons the CUDA
  // context -- the try/except fallback in backend.py cannot rescue the
  // process, every later CUDA call in the interpreter dies too.
  if (a.scalar_type() == torch::kUInt8 && (n_per_image % 4 == 0 || n_images == 1)) {
    const auto pa = reinterpret_cast<uintptr_t>(a.data_ptr<uint8_t>());
    const auto pb = reinterpret_cast<uintptr_t>(b.data_ptr<uint8_t>());
    if (pa % 4 == 0 && pb % 4 == 0) {
      const long n_vec = n_per_image / 4;
      const long tail = n_vec * 4;
      FA_PIX_DISPATCH(FA_PIX_U8);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      done = true;
    }
  }

// Hoisted out of the AT_DISPATCH call below: a preprocessor directive may not
// appear inside a macro's argument list, and AT_DISPATCH is a macro.
#define FA_PIX_SC(OP) FA_PIX(scalar_t, OP)

  if (!done) {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                               a.scalar_type(), "pixel_partial", [&] {
      FA_PIX_DISPATCH(FA_PIX_SC);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

#undef FA_PIX_SC
#undef FA_PIX_DISPATCH
#undef FA_PIX
#undef FA_PIX_U8

  return finalize(partial, (int)n_images, bpi,
                  static_cast<double>(n_per_image),
                  static_cast<double>(total), psnr_bias);
}

std::vector<torch::Tensor> mse_partial(torch::Tensor a, torch::Tensor b,
                                       int64_t n_images, double psnr_bias) {
  return pixel_partial(a, b, n_images, OP_MSE, 0.0, psnr_bias);
}

std::vector<torch::Tensor> ssim_fused(torch::Tensor x, torch::Tensor y,
                                      torch::Tensor win, double shift,
                                      double C1, double C2, bool want_map) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.dim() == 4 && x.sizes() == y.sizes(), "expected matching NCHW");
  TORCH_CHECK(win.dim() == 1 && win.numel() <= MAX_WIN, "window too large");
  x = x.contiguous();
  y = y.contiguous();
  auto winf = win.to(torch::kFloat32).contiguous().to(x.device());

  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int N = (int)x.size(0), C = (int)x.size(1);
  const int H = (int)x.size(2), W = (int)x.size(3);
  const int K = (int)winf.numel();
  const int Hout = H - K + 1, Wout = W - K + 1;
  TORCH_CHECK(Hout > 0 && Wout > 0, "image smaller than window");

  const int planes = N * C;
  const int gx = (Wout + TILE_W - 1) / TILE_W;
  const int gy = (Hout + TILE_H - 1) / TILE_H;
  const int bpp = gx * gy;

  auto partial = torch::empty({planes, bpp}, x.options().dtype(torch::kFloat64));
  torch::Tensor map;
  float* map_ptr = nullptr;
  if (want_map) {
    map = torch::empty({N, C, Hout, Wout}, x.options().dtype(torch::kFloat32));
    map_ptr = map.data_ptr<float>();
  }

  dim3 block(TILE_W, BLOCK_Y);
  dim3 grid(gx, gy, planes);

#define FA_LAUNCH(SC, MODE, KW)                                                \
  ssim_fused_kernel<SC, MODE, KW><<<grid, block, 0, stream>>>(                 \
      x.data_ptr<SC>(), y.data_ptr<SC>(), partial.data_ptr<double>(),          \
      map_ptr, nullptr, nullptr, 0.f, nullptr, nullptr, nullptr, nullptr,      \
      H, W, Hout, Wout, (float)shift, (float)C1, (float)C2,                    \
      winf.data_ptr<float>(), bpp, nullptr, nullptr, nullptr)

#define FA_DISPATCH_KW(SC, MODE)                                               \
  switch (K) {                                                                 \
    case 11: FA_LAUNCH(SC, MODE, 11); break;                                   \
    case 9:  FA_LAUNCH(SC, MODE, 9);  break;                                   \
    case 7:  FA_LAUNCH(SC, MODE, 7);  break;                                   \
    case 5:  FA_LAUNCH(SC, MODE, 5);  break;                                   \
    case 3:  FA_LAUNCH(SC, MODE, 3);  break;                                   \
    default: TORCH_CHECK(false, "unsupported window size ", K);                \
  }

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             x.scalar_type(), "ssim_fused", [&] {
    if (want_map) { FA_DISPATCH_KW(scalar_t, MODE_MAP); }
    else          { FA_DISPATCH_KW(scalar_t, MODE_REDUCE); }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#undef FA_DISPATCH_KW
#undef FA_LAUNCH

  // Each image owns C consecutive plane-rows of `partial`, so one image's
  // contribution is a contiguous C*bpp span -- exactly what finalize() reduces.
  const double npix = static_cast<double>(Hout) * Wout;
  auto red = finalize(partial, N, C * bpp, npix * C,
                      npix * C * static_cast<double>(N), PSNR_NONE);
  if (want_map) return {red[0], red[1], map};
  return red;
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

std::vector<torch::Tensor> ssim_cs(torch::Tensor x, torch::Tensor y,
                                   torch::Tensor win, double shift,
                                   double C1, double C2) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.dim() == 4 && x.sizes() == y.sizes(), "expected matching NCHW");
  TORCH_CHECK(win.dim() == 1 && win.numel() <= MAX_WIN, "window too large");
  x = x.contiguous();
  y = y.contiguous();
  auto winf = win.to(torch::kFloat32).contiguous().to(x.device());

  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int N = (int)x.size(0), C = (int)x.size(1);
  const int H = (int)x.size(2), W = (int)x.size(3);
  const int K = (int)winf.numel();
  const int Hout = H - K + 1, Wout = W - K + 1;
  TORCH_CHECK(Hout > 0 && Wout > 0, "image smaller than window");

  const int planes = N * C;
  const int gx = (Wout + TILE_W - 1) / TILE_W;
  const int gy = (Hout + TILE_H - 1) / TILE_H;
  const int bpp = gx * gy;

  auto opts64 = x.options().dtype(torch::kFloat64);
  auto partial = torch::empty({planes, bpp}, opts64);
  auto partial_cs = torch::empty({planes, bpp}, opts64);

  dim3 block(TILE_W, BLOCK_Y);
  dim3 grid(gx, gy, planes);

#define FA_CS(SC, KW)                                                          \
  ssim_fused_kernel<SC, MODE_CS, KW><<<grid, block, 0, stream>>>(              \
      x.data_ptr<SC>(), y.data_ptr<SC>(), partial.data_ptr<double>(),          \
      nullptr, nullptr, nullptr, 0.f, nullptr, nullptr, nullptr, nullptr,      \
      H, W, Hout, Wout, (float)shift, (float)C1, (float)C2,                    \
      winf.data_ptr<float>(), bpp, partial_cs.data_ptr<double>(),              \
      nullptr, nullptr)

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             x.scalar_type(), "ssim_cs", [&] {
    switch (K) {
      case 11: FA_CS(scalar_t, 11); break;
      case 9:  FA_CS(scalar_t, 9);  break;
      case 7:  FA_CS(scalar_t, 7);  break;
      case 5:  FA_CS(scalar_t, 5);  break;
      case 3:  FA_CS(scalar_t, 3);  break;
      default: TORCH_CHECK(false, "unsupported window size ", K);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#undef FA_CS

  auto s_out = torch::empty({planes}, opts64);
  auto cs_out = torch::empty({planes}, opts64);
  finalize_planes2_kernel<<<planes, 256, 0, stream>>>(
      partial.data_ptr<double>(), partial_cs.data_ptr<double>(), bpp,
      static_cast<double>(Hout) * Wout,
      s_out.data_ptr<double>(), cs_out.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {s_out, cs_out};
}

// --------------------------------------------------------------------------
// Backward
//
// Two passes. The first recomputes the local moments (the same tile kernel as
// the forward, in MODE_PREP) and emits three coefficient maps; the second
// scatters them back through the Gaussian and combines with x and y.
//
// dL_dmap may be undefined, in which case every map position gets
// `uniform_grad` -- the mean-reduction case, where materialising a constant
// HoutxWout tensor just to read it back would be pure bandwidth.
// --------------------------------------------------------------------------

// The second half of every SSIM-family backward: three (or four) coefficient
// maps convolved back through the Gaussian and combined with x and y. Shared
// verbatim between the single-scale and the multi-scale entry points -- only
// how the maps were *produced* differs.
static std::vector<torch::Tensor> ssim_bwd_scatter(
    const torch::Tensor& x, const torch::Tensor& y, const float* ab,
    const float* ac, const float* atx, const float* aty,
    const torch::Tensor& winf, int H, int W, int Hout, int Wout, int K,
    int planes, double shift, bool need_dx, bool need_dy,
    cudaStream_t stream) {
  torch::Tensor dx, dy;
  float* dx_ptr = nullptr;
  float* dy_ptr = nullptr;
  if (need_dx) { dx = torch::empty_like(x); dx_ptr = dx.data_ptr<float>(); }
  if (need_dy) { dy = torch::empty_like(y); dy_ptr = dy.data_ptr<float>(); }

  const int gx = (W + TILE_W - 1) / TILE_W;
  const int gy = (H + TILE_H - 1) / TILE_H;
  dim3 block(TILE_W, BLOCK_Y);
  dim3 grid(gx, gy, planes);
#define FA_SCAT(KW, DY)                                                        \
  ssim_bwd_scatter_kernel<KW, DY><<<grid, block, 0, stream>>>(                 \
      x.data_ptr<float>(), y.data_ptr<float>(), ab, ac, atx, aty, dx_ptr,      \
      dy_ptr, winf.data_ptr<float>(), H, W, Hout, Wout, (float)shift)
#define FA_SCAT_KW(DY)                                                         \
  switch (K) {                                                                 \
    case 11: FA_SCAT(11, DY); break;                                           \
    case 9:  FA_SCAT(9, DY);  break;                                           \
    case 7:  FA_SCAT(7, DY);  break;                                           \
    case 5:  FA_SCAT(5, DY);  break;                                           \
    case 3:  FA_SCAT(3, DY);  break;                                           \
    default: TORCH_CHECK(false, "unsupported window size ", K);                \
  }
  if (need_dy) { FA_SCAT_KW(true); } else { FA_SCAT_KW(false); }
#undef FA_SCAT_KW
#undef FA_SCAT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dy};
}

std::vector<torch::Tensor> ssim_backward(torch::Tensor x, torch::Tensor y,
                                         torch::Tensor dL_dmap,
                                         torch::Tensor grad_scalar,
                                         torch::Tensor win,
                                         double shift, double C1, double C2,
                                         bool need_dx, bool need_dy) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.dim() == 4 && x.sizes() == y.sizes(), "expected matching NCHW");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32,
              "backward requires float32 inputs, got ", x.scalar_type());
  TORCH_CHECK(need_dx || need_dy, "backward called with nothing to compute");
  x = x.contiguous();
  y = y.contiguous();
  auto winf = win.to(torch::kFloat32).contiguous().to(x.device());

  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int N = (int)x.size(0), C = (int)x.size(1);
  const int H = (int)x.size(2), W = (int)x.size(3);
  const int K = (int)winf.numel();
  const int Hout = H - K + 1, Wout = W - K + 1;
  TORCH_CHECK(Hout > 0 && Wout > 0, "image smaller than window");
  const int planes = N * C;

  const bool has_map = dL_dmap.defined() && dL_dmap.numel() > 0;
  const float* dmap_ptr = nullptr;
  const float* gscalar_ptr = nullptr;
  float grad_norm = 0.0f;
  if (has_map) {
    dL_dmap = dL_dmap.to(torch::kFloat32).contiguous();
    TORCH_CHECK(dL_dmap.numel() == (long)planes * Hout * Wout,
                "dL_dmap has the wrong number of elements");
    dmap_ptr = dL_dmap.data_ptr<float>();
  } else {
    TORCH_CHECK(grad_scalar.defined() && grad_scalar.numel() == 1,
                "need either dL_dmap or a 1-element grad_scalar");
    grad_scalar = grad_scalar.to(torch::kFloat32).contiguous();
    gscalar_ptr = grad_scalar.data_ptr<float>();
    grad_norm = 1.0f / (float)((double)Hout * Wout * planes);
  }

  // b, c, t_x [, t_y] laid out as one allocation of contiguous planes
  const int naux = need_dy ? 4 : 3;
  auto aux = torch::empty({naux, planes, Hout, Wout}, x.options());
  const long aux_stride = (long)planes * Hout * Wout;
  float* ab = aux.data_ptr<float>();
  float* ac = ab + aux_stride;
  float* atx = ac + aux_stride;
  float* aty = need_dy ? atx + aux_stride : nullptr;

  {
    const int gx = (Wout + TILE_W - 1) / TILE_W;
    const int gy = (Hout + TILE_H - 1) / TILE_H;
    dim3 block(TILE_W, BLOCK_Y);
    dim3 grid(gx, gy, planes);
#define FA_PREP(KW)                                                            \
  ssim_fused_kernel<float, MODE_PREP, KW><<<grid, block, 0, stream>>>(         \
      x.data_ptr<float>(), y.data_ptr<float>(), nullptr, nullptr, dmap_ptr,    \
      gscalar_ptr, grad_norm, ab, ac, atx, aty, H, W, Hout, Wout,              \
      (float)shift, (float)C1, (float)C2, winf.data_ptr<float>(), 0,           \
      nullptr, nullptr, nullptr)
    switch (K) {
      case 11: FA_PREP(11); break;
      case 9:  FA_PREP(9);  break;
      case 7:  FA_PREP(7);  break;
      case 5:  FA_PREP(5);  break;
      case 3:  FA_PREP(3);  break;
      default: TORCH_CHECK(false, "unsupported window size ", K);
    }
#undef FA_PREP
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return ssim_bwd_scatter(x, y, ab, ac, atx, aty, winf, H, W, Hout, Wout, K,
                          planes, shift, need_dx, need_dy, stream);
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

std::vector<torch::Tensor> ssim_cs_backward(torch::Tensor x, torch::Tensor y,
                                            torch::Tensor gs, torch::Tensor gcs,
                                            torch::Tensor win,
                                            double shift, double C1, double C2,
                                            bool need_dx, bool need_dy) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.dim() == 4 && x.sizes() == y.sizes(), "expected matching NCHW");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32,
              "backward requires float32 inputs, got ", x.scalar_type());
  TORCH_CHECK(need_dx || need_dy, "backward called with nothing to compute");
  x = x.contiguous();
  y = y.contiguous();
  auto winf = win.to(torch::kFloat32).contiguous().to(x.device());

  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int N = (int)x.size(0), C = (int)x.size(1);
  const int H = (int)x.size(2), W = (int)x.size(3);
  const int K = (int)winf.numel();
  const int Hout = H - K + 1, Wout = W - K + 1;
  TORCH_CHECK(Hout > 0 && Wout > 0, "image smaller than window");
  const int planes = N * C;

  gs = gs.to(torch::kFloat32).contiguous();
  gcs = gcs.to(torch::kFloat32).contiguous();
  TORCH_CHECK(gs.numel() == planes && gcs.numel() == planes,
              "expected one SSIM and one cs gradient per plane");

  const int naux = need_dy ? 4 : 3;
  auto aux = torch::empty({naux, planes, Hout, Wout}, x.options());
  const long aux_stride = (long)planes * Hout * Wout;
  float* ab = aux.data_ptr<float>();
  float* ac = ab + aux_stride;
  float* atx = ac + aux_stride;
  float* aty = need_dy ? atx + aux_stride : nullptr;
  const float grad_norm = 1.0f / (float)((double)Hout * Wout);

  {
    const int gx = (Wout + TILE_W - 1) / TILE_W;
    const int gy = (Hout + TILE_H - 1) / TILE_H;
    dim3 block(TILE_W, BLOCK_Y);
    dim3 grid(gx, gy, planes);
#define FA_PREP_CS(KW)                                                         \
  ssim_fused_kernel<float, MODE_PREP_CS, KW><<<grid, block, 0, stream>>>(      \
      x.data_ptr<float>(), y.data_ptr<float>(), nullptr, nullptr, nullptr,     \
      nullptr, grad_norm, ab, ac, atx, aty, H, W, Hout, Wout,                  \
      (float)shift, (float)C1, (float)C2, winf.data_ptr<float>(), 0,           \
      nullptr, gs.data_ptr<float>(), gcs.data_ptr<float>())
    switch (K) {
      case 11: FA_PREP_CS(11); break;
      case 9:  FA_PREP_CS(9);  break;
      case 7:  FA_PREP_CS(7);  break;
      case 5:  FA_PREP_CS(5);  break;
      case 3:  FA_PREP_CS(3);  break;
      default: TORCH_CHECK(false, "unsupported window size ", K);
    }
#undef FA_PREP_CS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return ssim_bwd_scatter(x, y, ab, ac, atx, aty, winf, H, W, Hout, Wout, K,
                          planes, shift, need_dx, need_dy, stream);
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
// --------------------------------------------------------------------------

constexpr int GTILE = 32;
constexpr int GBLOCK_Y = 8;
constexpr int GSTAGE = GTILE + 2;

template <typename T, bool DOWN>
__global__ __launch_bounds__(GTILE * GBLOCK_Y) void gmsd_kernel(
    const T* __restrict__ xp, const T* __restrict__ yp,
    double* __restrict__ psum, double* __restrict__ psumsq,
    int H, int W, int Hd, int Wd, int Hout, int Wout,
    float Tc, float eps, int blocks_per_plane) {

  __shared__ float s_x[GSTAGE][GSTAGE];
  __shared__ float s_y[GSTAGE][GSTAGE];
  __shared__ double s_warp[64];

  const int plane = blockIdx.z;
  const long plane_off = static_cast<long>(plane) * H * W;
  const int tile_x = blockIdx.x * GTILE;
  const int tile_y = blockIdx.y * GTILE;
  const int tid = threadIdx.y * GTILE + threadIdx.x;
  constexpr int NTHREADS = GTILE * GBLOCK_Y;

  for (int idx = tid; idx < GSTAGE * GSTAGE; idx += NTHREADS) {
    const int r = idx / GSTAGE;
    const int c = idx - r * GSTAGE;
    const int gy = tile_y + r;
    const int gx = tile_x + c;
    float xv = 0.0f, yv = 0.0f;
    if (gy < Hd && gx < Wd) {
      if (DOWN) {
        const long o = plane_off + static_cast<long>(2 * gy) * W + 2 * gx;
        xv = 0.25f * (Ld<T>::get(xp, o) + Ld<T>::get(xp, o + 1) +
                      Ld<T>::get(xp, o + W) + Ld<T>::get(xp, o + W + 1));
        yv = 0.25f * (Ld<T>::get(yp, o) + Ld<T>::get(yp, o + 1) +
                      Ld<T>::get(yp, o + W) + Ld<T>::get(yp, o + W + 1));
      } else {
        const long o = plane_off + static_cast<long>(gy) * W + gx;
        xv = Ld<T>::get(xp, o);
        yv = Ld<T>::get(yp, o);
      }
    }
    s_x[r][c] = xv;
    s_y[r][c] = yv;
  }
  __syncthreads();

  constexpr float THIRD = 1.0f / 3.0f;
  double acc = 0.0, acc_sq = 0.0;
  const int oc = tile_x + threadIdx.x;
  if (oc < Wout) {
    const int c = threadIdx.x;
    for (int r = threadIdx.y; r < GTILE; r += GBLOCK_Y) {
      if (tile_y + r >= Hout) break;
      float g[2];
      #pragma unroll
      for (int which = 0; which < 2; ++which) {
        const float(*s)[GSTAGE] = which ? s_y : s_x;
        const float dx = ((s[r][c] - s[r][c + 2]) +
                          (s[r + 1][c] - s[r + 1][c + 2]) +
                          (s[r + 2][c] - s[r + 2][c + 2])) * THIRD;
        const float dy = ((s[r][c] - s[r + 2][c]) +
                          (s[r][c + 1] - s[r + 2][c + 1]) +
                          (s[r][c + 2] - s[r + 2][c + 2])) * THIRD;
        g[which] = sqrtf(fmaf(dx, dx, fmaf(dy, dy, eps)));
      }
      const float q = (2.0f * g[0] * g[1] + Tc) /
                      (g[0] * g[0] + g[1] * g[1] + Tc);
      acc += static_cast<double>(q);
      acc_sq += static_cast<double>(q) * static_cast<double>(q);
    }
  }

  double s = 0.0, ssq = 0.0;
  block_reduce_sum2(acc, acc_sq, s_warp, &s, &ssq);
  if (tid == 0) {
    const long o = static_cast<long>(plane) * blocks_per_plane +
                   blockIdx.y * gridDim.x + blockIdx.x;
    psum[o] = s;
    psumsq[o] = ssq;
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

std::vector<torch::Tensor> gmsd_fused(torch::Tensor x, torch::Tensor y,
                                      double Tc, double eps, bool downsample) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.dim() == 4 && x.sizes() == y.sizes(), "expected matching NCHW");
  x = x.contiguous();
  y = y.contiguous();
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int N = (int)x.size(0), C = (int)x.size(1);
  const int H = (int)x.size(2), W = (int)x.size(3);
  const int Hd = downsample ? H / 2 : H;
  const int Wd = downsample ? W / 2 : W;
  const int Hout = Hd - 2, Wout = Wd - 2;
  TORCH_CHECK(Hout > 0 && Wout > 0, "image too small for a 3x3 gradient");

  const int planes = N * C;
  const int gx = (Wout + GTILE - 1) / GTILE;
  const int gy = (Hout + GTILE - 1) / GTILE;
  const int bpp = gx * gy;

  auto opts64 = x.options().dtype(torch::kFloat64);
  auto psum = torch::empty({planes, bpp}, opts64);
  auto psumsq = torch::empty({planes, bpp}, opts64);

  dim3 block(GTILE, GBLOCK_Y);
  dim3 grid(gx, gy, planes);
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             x.scalar_type(), "gmsd_fused", [&] {
    if (downsample) {
      gmsd_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
          psum.data_ptr<double>(), psumsq.data_ptr<double>(),
          H, W, Hd, Wd, Hout, Wout, (float)Tc, (float)eps, bpp);
    } else {
      gmsd_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
          x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
          psum.data_ptr<double>(), psumsq.data_ptr<double>(),
          H, W, Hd, Wd, Hout, Wout, (float)Tc, (float)eps, bpp);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto mean = torch::empty({N}, opts64);
  auto dev = torch::empty({N}, opts64);
  gmsd_finalize_kernel<<<N, 256, 0, stream>>>(
      psum.data_ptr<double>(), psumsq.data_ptr<double>(), C * bpp,
      static_cast<double>(Hout) * Wout * C,
      mean.data_ptr<double>(), dev.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {mean, dev};
}

}  // namespace fa
