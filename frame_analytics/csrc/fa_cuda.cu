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
//   * 8 input rows are staged at a time (32 + 10 wide, for the horizontal halo)
//   * each staged row is immediately turned into its five Gaussian-weighted
//     horizontal partial sums and pushed into an 18-row ring buffer
//   * as soon as 11 rows are resident, the vertical tap runs, the SSIM value is
//     formed in registers and folded straight into a block accumulator
//
// Net DRAM traffic: the two input planes, read once each plus ~30% halo
// overlap. Nothing else is written except one partial sum per block.
//
// Shared memory: 8*42*2*4 (staging) + 18*5*32*4 (ring) = 14.2 KiB, so several
// blocks stay resident per SM.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
constexpr int NPLANE = 5;

// --------------------------------------------------------------------------
// typed loads
// --------------------------------------------------------------------------

template <typename T> struct Ld {
  __device__ static inline float get(const T* p, long i) { return static_cast<float>(p[i]); }
};
template <> struct Ld<at::Half> {
  __device__ static inline float get(const at::Half* p, long i) { return __half2float(*reinterpret_cast<const __half*>(p + i)); }
};
template <> struct Ld<at::BFloat16> {
  __device__ static inline float get(const at::BFloat16* p, long i) { return static_cast<float>(p[i]); }
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

__global__ void finalize_kernel(const double* __restrict__ partial,
                                int nrows, int ncols,
                                double inv_per_image, double inv_total,
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
    const double v = s * (is_total ? inv_total : inv_per_image);
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

static inline std::vector<torch::Tensor> finalize(torch::Tensor partial,
                                                  int nrows, int ncols,
                                                  double inv_per_image,
                                                  double inv_total,
                                                  double psnr_bias) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto opts = partial.options();
  auto per_image = torch::empty({nrows}, opts);
  auto total = torch::empty({}, opts);
  const bool want_psnr = (psnr_bias != PSNR_NONE);
  auto psnr_pi = torch::empty({want_psnr ? nrows : 0}, opts);
  auto psnr_tot = torch::empty({want_psnr ? 1 : 0}, opts);

  finalize_kernel<<<nrows + 1, 256, 0, stream>>>(
      partial.data_ptr<double>(), nrows, ncols, inv_per_image, inv_total,
      psnr_bias, per_image.data_ptr<double>(), total.data_ptr<double>(),
      want_psnr ? psnr_pi.data_ptr<double>() : nullptr,
      want_psnr ? psnr_tot.data_ptr<double>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (want_psnr) return {per_image, total, psnr_pi, psnr_tot.squeeze()};
  return {per_image, total};
}

// --------------------------------------------------------------------------
// MSE  (grid-stride, integer-exact for uint8, double accumulate otherwise)
// --------------------------------------------------------------------------

template <typename T>
__global__ void mse_kernel(const T* __restrict__ a, const T* __restrict__ b,
                           double* __restrict__ partial, long n_per_image,
                           int n_images, int blocks_per_image) {
  __shared__ double warp_sums[32];
  const int img = blockIdx.y;
  const long base = static_cast<long>(img) * n_per_image;

  double acc = 0.0;
  const long stride = static_cast<long>(blockDim.x) * blocks_per_image;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_per_image; i += stride) {
    const float d = Ld<T>::get(a, base + i) - Ld<T>::get(b, base + i);
    acc += static_cast<double>(d) * static_cast<double>(d);
  }
  const double s = block_reduce_sum(acc, warp_sums);
  if (threadIdx.x == 0) partial[img * blocks_per_image + blockIdx.x] = s;
  (void)n_images;
}

// uint8 specialisation: 4-wide vectorised loads, exact 64-bit integer accumulate.
__global__ void mse_kernel_u8(const uchar4* __restrict__ a, const uchar4* __restrict__ b,
                              double* __restrict__ partial, long n_vec, long tail_start,
                              const unsigned char* __restrict__ a_raw,
                              const unsigned char* __restrict__ b_raw, long n_total,
                              long n_per_image, int blocks_per_image) {
  __shared__ double warp_sums[32];
  const int img = blockIdx.y;
  const long vec_base = static_cast<long>(img) * n_vec;

  long long acc = 0;
  const long stride = static_cast<long>(blockDim.x) * blocks_per_image;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
    const uchar4 va = a[vec_base + i];
    const uchar4 vb = b[vec_base + i];
    const int d0 = (int)va.x - (int)vb.x;
    const int d1 = (int)va.y - (int)vb.y;
    const int d2 = (int)va.z - (int)vb.z;
    const int d3 = (int)va.w - (int)vb.w;
    acc += (long long)(d0 * d0) + (long long)(d1 * d1) + (long long)(d2 * d2) + (long long)(d3 * d3);
  }
  // ragged tail of this image
  const long img_base = static_cast<long>(img) * n_per_image;
  for (long i = tail_start + static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_per_image; i += stride) {
    const int d = (int)a_raw[img_base + i] - (int)b_raw[img_base + i];
    acc += (long long)(d * d);
  }
  const double s = block_reduce_sum(static_cast<double>(acc), warp_sums);
  if (threadIdx.x == 0) partial[img * blocks_per_image + blockIdx.x] = s;
  (void)n_total;
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
// What the tile kernel does once it has the five local moments.
constexpr int MODE_REDUCE = 0;   // fold SSIM into a block sum
constexpr int MODE_MAP = 1;      // ... and also write the SSIM map
constexpr int MODE_PREP = 2;     // write the backward's three coefficient maps

template <typename T, int OUT_MODE, int KW>
__global__ __launch_bounds__(TILE_W * BLOCK_Y) void ssim_fused_kernel(
    const T* __restrict__ xp, const T* __restrict__ yp,
    double* __restrict__ partial,      // [planes][blocks_per_plane]
    float* __restrict__ map_out,       // optional [planes][Hout][Wout]
    const float* __restrict__ dL_dmap, // MODE_PREP: upstream grad, may be null
    const float* __restrict__ grad_scalar,  // MODE_PREP: used when dL_dmap null
    float grad_norm,                   // MODE_PREP: 1/(npix*planes)
    float* __restrict__ out_b,         // MODE_PREP: g * dS/dsigma_xx
    float* __restrict__ out_c,         // MODE_PREP: g * dS/dsigma_xy
    float* __restrict__ out_tx,        // MODE_PREP: g*dS/dmu_x - 2b*mu_x' - c*mu_y'
    float* __restrict__ out_ty,        // MODE_PREP: same with x/y swapped, may be null
    int H, int W, int Hout, int Wout,
    float shift, float C1, float C2,
    const float* __restrict__ win,     // KW taps, sum 1
    int blocks_per_plane) {

  constexpr int HALO_K = KW - 1;
  constexpr int STAGE_WK = TILE_W + HALO_K;
  constexpr int RING_HK = ROWS_PER_STEP + HALO_K;

  __shared__ float s_x[ROWS_PER_STEP][STAGE_WK];
  __shared__ float s_y[ROWS_PER_STEP][STAGE_WK];
  __shared__ float s_ring[RING_HK][NPLANE][TILE_W];
  __shared__ double s_warp[32];
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
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f;
        #pragma unroll
        for (int j = 0; j < KW; ++j) {
          const float w = s_win[j];
          const float xv = s_x[rr][tx + j];
          const float yv = s_y[rr][tx + j];
          a0 = fmaf(w, xv, a0);
          a1 = fmaf(w, yv, a1);
          a2 = fmaf(w, xv * xv, a2);
          a3 = fmaf(w, yv * yv, a3);
          a4 = fmaf(w, xv * yv, a4);
        }
        int slot = (r0 + rr) % RING_HK;
        s_ring[slot][0][tx] = a0;
        s_ring[slot][1][tx] = a1;
        s_ring[slot][2][tx] = a2;
        s_ring[slot][3][tx] = a3;
        s_ring[slot][4][tx] = a4;
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
          const float r4v = s_ring[slot][4][tx];
          #pragma unroll
          for (int k = 0; k < ROW_BLOCK; ++k) {
            const int j = t - k;
            if (j >= 0 && j < KW) {
              const float w = s_win[j];
              b[k][0] = fmaf(w, r0v, b[k][0]);
              b[k][1] = fmaf(w, r1v, b[k][1]);
              b[k][2] = fmaf(w, r2v, b[k][2]);
              b[k][3] = fmaf(w, r3v, b[k][3]);
              b[k][4] = fmaf(w, r4v, b[k][4]);
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
          const float sxx = b[k][2] - ux * ux;
          const float syy = b[k][3] - uy * uy;
          const float sxy = b[k][4] - ux * uy;

          const float A1 = 2.0f * mx * my + C1;
          const float A2 = 2.0f * sxy + C2;
          const float B1 = mx * mx + my * my + C1;   // >= C1 > 0
          const float B2 = sxx + syy + C2;           // >= C2 > 0
          const long idx = static_cast<long>(plane) * Hout * Wout +
                           static_cast<long>(tile_y + orow) * Wout + tile_x + tx;

          if (OUT_MODE == MODE_PREP) {
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

  if (OUT_MODE != MODE_PREP) {
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
// shifted origin, NP planes instead of 5, and no products -- not a separate
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

std::vector<torch::Tensor> mse_partial(torch::Tensor a, torch::Tensor b,
                                       int64_t n_images, double psnr_bias) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
  a = a.contiguous();
  b = b.contiguous();
  const at::cuda::CUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const long total = a.numel();
  TORCH_CHECK(total % n_images == 0, "numel not divisible by batch");
  const long n_per_image = total / n_images;

  const int threads = 256;
  int bpi = (int)std::min<long>((n_per_image + threads - 1) / threads,
                                std::max(1, sm_count() * 8 / (int)n_images + 1));
  bpi = std::max(1, bpi);

  auto partial = torch::empty({n_images, bpi},
                              a.options().dtype(torch::kFloat64));
  dim3 grid(bpi, (unsigned)n_images);

  bool done = false;
  if (a.scalar_type() == torch::kUInt8) {
    const long n_vec = n_per_image / 4;
    const long tail = n_vec * 4;
    // vectorised loads need each image's base to be 4-aligned
    if (n_per_image % 4 == 0 || n_images == 1) {
      mse_kernel_u8<<<grid, threads, 0, stream>>>(
          reinterpret_cast<const uchar4*>(a.data_ptr<uint8_t>()),
          reinterpret_cast<const uchar4*>(b.data_ptr<uint8_t>()),
          partial.data_ptr<double>(), n_vec, tail,
          a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(), total, n_per_image, bpi);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      done = true;
    }
  }

  if (!done) {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                               a.scalar_type(), "mse_partial", [&] {
      mse_kernel<scalar_t><<<grid, threads, 0, stream>>>(
          a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
          partial.data_ptr<double>(), n_per_image, (int)n_images, bpi);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return finalize(partial, (int)n_images, bpi,
                  1.0 / static_cast<double>(n_per_image),
                  1.0 / static_cast<double>(total), psnr_bias);
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
      winf.data_ptr<float>(), bpp)

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
  auto red = finalize(partial, N, C * bpp, 1.0 / (npix * C),
                      1.0 / (npix * C * static_cast<double>(N)), PSNR_NONE);
  if (want_map) return {red[0], red[1], map};
  return red;
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
      (float)shift, (float)C1, (float)C2, winf.data_ptr<float>(), 0)
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

  torch::Tensor dx, dy;
  float* dx_ptr = nullptr;
  float* dy_ptr = nullptr;
  if (need_dx) { dx = torch::empty_like(x); dx_ptr = dx.data_ptr<float>(); }
  if (need_dy) { dy = torch::empty_like(y); dy_ptr = dy.data_ptr<float>(); }

  {
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
  }

  return {dx, dy};
}

}  // namespace fa
