// CPU kernels for frame_analytics, behind the C ABI in fa_abi.h.
//
// Same structural trick as the CUDA path: the 11x11 Gaussian is separable and
// all five expectation planes (x, y, x^2, y^2, xy) are produced in one sweep,
// so nothing full-resolution is ever written to memory.  Work is tiled to
// (row-block x column-block) so each thread's ring buffer -- 11 rows x 5 planes
// x 256 columns = 56 KiB -- stays resident in L2.
//
// Inner loops are written j-outer / c-inner over contiguous float arrays so
// MSVC and GCC both auto-vectorise them to AVX2 FMA chains.
//
// This file is compiled *twice* into two separate shared libraries: once at the
// platform baseline and once with AVX2 enabled.  The loader asks the baseline
// build (which is safe to load anywhere) whether the host has AVX2 and, if so,
// uses the sibling instead.  Compiling one binary with /arch:AVX2 and shipping
// it to everybody would fault on pre-Haswell hardware; compiling only a
// baseline one would give up the vector width these loops exist for.

#include "fa_abi.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

// The reduction loops below are hand-vectorised for the uint8 squared-error
// case only.  Everything else in this file auto-vectorises; that one does not,
// because the int64 widening multiply defeats both MSVC's and GCC's
// vectorisers, and it is the single hottest loop in the library.
#if defined(__AVX2__)
#include <immintrin.h>
#define FA_AVX2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#include <arm_neon.h>
#define FA_NEON 1
#endif

#if defined(_MSC_VER)
#define FA_RESTRICT __restrict
#else
#define FA_RESTRICT __restrict__
#endif

namespace fa {
namespace cpu {

// -------------------------------------------------------------------------
// Narrow float types
//
// at::Half and at::BFloat16 came from the torch headers this file no longer
// includes, so the two conversions are open-coded.  Both are exact: bfloat16
// is the top half of a float32 and half's mantissa fits, so neither rounds.
// -------------------------------------------------------------------------

struct half_t {
  uint16_t bits;
  operator float() const {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exp = (bits >> 10) & 0x1Fu;
    const uint32_t man = bits & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
      // subnormal (or zero): value is man * 2^-24, which float32 holds exactly
      float f = static_cast<float>(man) * (1.0f / 16777216.0f);
      std::memcpy(&u, &f, 4);
      u |= sign;
    } else if (exp == 31) {
      u = sign | 0x7F800000u | (man << 13);
    } else {
      u = sign | ((exp + (127u - 15u)) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &u, 4);
    return out;
  }
};

struct bf16_t {
  uint16_t bits;
  operator float() const {
    const uint32_t u = static_cast<uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &u, 4);
    return out;
  }
};

// Stands in for AT_DISPATCH_ALL_TYPES_AND2. The five codes are exactly the
// dtypes backend.py lets through.
//
// Variadic on purpose: the body contains template argument lists such as
// `ssim_tile<scalar_t, true>`, and braces do not shield a comma from the
// preprocessor the way parentheses do. Only `...`/`__VA_ARGS__` puts the
// pieces back together.
#define FA_DISPATCH(CODE, ...)                                                 \
  switch (CODE) {                                                              \
    case FA_U8:   { using scalar_t = uint8_t; __VA_ARGS__ } break;             \
    case FA_F16:  { using scalar_t = half_t;  __VA_ARGS__ } break;             \
    case FA_BF16: { using scalar_t = bf16_t;  __VA_ARGS__ } break;             \
    case FA_F32:  { using scalar_t = float;   __VA_ARGS__ } break;             \
    case FA_F64:  { using scalar_t = double;  __VA_ARGS__ } break;             \
    default: return FA_E_DTYPE;                                                \
  }

constexpr int COL_TILE = 256;
constexpr int ROW_TILE = 64;
constexpr int MAX_WIN = FA_MAX_WIN;

// -------------------------------------------------------------------------
// Task pool
//
// ``at::parallel_for`` is a compile-time alias for an OpenMP parallel region.
// A JIT-loaded extension is not compiled with /openmp (and adding it on
// Windows would drag a second OpenMP runtime into a process that already has
// Intel's), so those pragmas are dropped and every at::parallel_for here would
// silently run single-threaded. This pool sidesteps the whole question: plain
// std::thread, work-stealing over a task index, calling thread participates.
// -------------------------------------------------------------------------

// Workers spin briefly on the generation counter before sleeping on the
// condition variable, and the caller does the same while waiting for them to
// finish. A pure condvar handoff costs ~45us of wall clock per call, which is
// more than the entire metric for anything under about a megapixel -- it made
// small-image PSNR slower than the pure-PyTorch libraries. The spin is bounded,
// so an idle pool still parks itself.
constexpr int SPIN_ROUNDS = 8192;

static inline void cpu_relax() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#else
  std::this_thread::yield();
#endif
}

// Every one of these is written by one thread and read by all the others on
// the launch path, so two of them sharing a line turns a single store into a
// storm of invalidations across the whole pool.  `cursor_` is the worst
// offender: with 16 threads racing `fetch_add` on it, anything living on the
// same line is effectively unreadable for the duration of the call.
#define FA_PAD alignas(64)

class TaskPool {
 public:
  explicit TaskPool(int nworkers) {
    workers_.reserve(nworkers);
    for (int i = 0; i < nworkers; ++i)
      workers_.emplace_back([this, i] { worker(i); });
  }

  int size() const { return static_cast<int>(workers_.size()) + 1; }

  void run(int64_t ntasks, const std::function<void(int64_t)>& fn) {
    if (ntasks <= 0) return;
    const int W = static_cast<int>(workers_.size());
    if (W == 0 || ntasks == 1) {
      for (int64_t i = 0; i < ntasks; ++i) fn(i);
      return;
    }
    // The calling thread takes a share too, so `ntasks - 1` helpers is the
    // most that can ever be useful.  Waking all 16 for three tasks cost more
    // in cursor contention than the tasks did in work.
    const int want = static_cast<int>(std::min<int64_t>(ntasks - 1, W));

    fn_ = &fn;
    total_.store(ntasks, std::memory_order_relaxed);
    cursor_.store(0, std::memory_order_relaxed);
    target_.store(want, std::memory_order_relaxed);
    active_.store(want, std::memory_order_relaxed);
    // seq_cst, not release: this store and the `parked_` load below must not
    // be reordered against each other.  Under acquire/release alone both this
    // thread and a worker on its way into the condition variable can read a
    // stale value of the other's flag, and the wake-up is lost.
    generation_.fetch_add(1, std::memory_order_seq_cst);

    if (parked_.load(std::memory_order_seq_cst) > 0) {
      std::lock_guard<std::mutex> lk(mu_);
      cv_work_.notify_all();
    }

    drain();  // the calling thread is a worker too

    for (int i = 0; i < SPIN_ROUNDS; ++i) {
      if (active_.load(std::memory_order_acquire) == 0) return;
      cpu_relax();
    }
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [this] { return active_.load(std::memory_order_acquire) == 0; });
  }

 private:
  void drain() {
    const int64_t n = total_.load(std::memory_order_relaxed);
    for (;;) {
      const int64_t i = cursor_.fetch_add(1, std::memory_order_relaxed);
      if (i >= n) return;
      (*fn_)(i);
    }
  }

  void worker(int id) {
    uint64_t seen = 0;
    for (;;) {
      bool woke = false;
      for (int i = 0; i < SPIN_ROUNDS; ++i) {
        if (stop_.load(std::memory_order_acquire)) return;
        if (generation_.load(std::memory_order_acquire) != seen) { woke = true; break; }
        cpu_relax();
      }
      if (!woke) {
        std::unique_lock<std::mutex> lk(mu_);
        parked_.fetch_add(1, std::memory_order_seq_cst);
        // predicate is re-checked under the lock, so a generation bump that
        // lands between the spin ending and the lock being taken is not lost
        cv_work_.wait(lk, [&] {
          return stop_.load(std::memory_order_acquire) ||
                 generation_.load(std::memory_order_seq_cst) != seen;
        });
        parked_.fetch_sub(1, std::memory_order_seq_cst);
        if (stop_.load(std::memory_order_acquire)) return;
      }
      seen = generation_.load(std::memory_order_acquire);
      // no share this round -- back to spinning without touching `cursor_` or
      // `active_`, both of which the participating threads are hammering
      if (id >= target_.load(std::memory_order_acquire)) continue;
      drain();
      // Only the thread that finishes last takes the mutex.  Having every
      // worker take it to decrement turned each call into a 15-way convoy,
      // which was most of the pool's launch cost.
      if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lk(mu_);
        cv_done_.notify_all();
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_work_, cv_done_;
  const std::function<void(int64_t)>* fn_ = nullptr;
  FA_PAD std::atomic<int64_t> cursor_{0};
  FA_PAD std::atomic<int64_t> total_{0};
  FA_PAD std::atomic<int> active_{0};
  FA_PAD std::atomic<int> target_{0};
  FA_PAD std::atomic<int> parked_{0};
  FA_PAD std::atomic<uint64_t> generation_{0};
  FA_PAD std::atomic<bool> stop_{false};
};

// The thread count used to arrive as at::get_num_threads(); across a C boundary
// there is no ATen to ask, so the loader sets it from torch.get_num_threads()
// before the first call.  Only the first reader wins -- the pool is built once.
static std::atomic<int> g_nthreads{0};
static std::once_flag g_pool_once;
static TaskPool* g_pool = nullptr;

// Deliberately leaked: joining worker threads during static destruction races
// with the interpreter tearing down around us.
static TaskPool& pool() {
  std::call_once(g_pool_once, [] {
    int n = g_nthreads.load(std::memory_order_relaxed);
    if (n <= 0) n = static_cast<int>(std::thread::hardware_concurrency());
    if (n <= 0) n = 1;
    g_pool = new TaskPool(std::max(0, n - 1));
  });
  return *g_pool;
}

// -------------------------------------------------------------------------
// Pixel losses: MSE, L1, Charbonnier, Huber
//
// One templated sweep, as on the CUDA side. MSE and L1 over uint8 take an
// exact integer accumulator; the other two are float terms folded into a
// double sum, which is still far wider than the float32 that every other
// library reduces in.
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// uint8 squared error, vectorised by hand
//
// The scalar form of this loop -- `acc += (int64_t)d * (int64_t)d` -- is the
// one loop in the file that neither MSVC nor GCC will vectorise: the widening
// multiply into a 64-bit accumulator has no lane-preserving form, so both give
// up and emit scalar code.  It measured 5.1 Gelem/s per thread, which is how
// `cv2.PSNR` (39-45 Gelem/s on one core, from exactly this instruction
// sequence) came to beat a 16-thread version of this function outright.
//
// The trick is that the *pairwise* widening multiply does exist: `madd_epi16`
// on x86, `vmull_u8` + `vpadalq_u16` on NEON.  Squares of a uint8 difference
// fit in 16 bits, so both inputs stay narrow and only the accumulator has to
// widen.  A 32-bit accumulator then has to be flushed to 64 bits before it can
// overflow, which is what the blocking is for.
//
// The result is bit-identical to the scalar loop: every operation is exact
// integer arithmetic, and integer addition is associative, so the
// reassociation the vector form performs changes nothing.
// -------------------------------------------------------------------------

#if defined(FA_AVX2)
// Each `madd_epi16` lane holds the sum of two squares, at most 2*255^2 =
// 130050, and there are two of them per iteration.  2^31 / (2*130050) = 8256
// iterations of headroom; 4096 keeps a comfortable margin and costs one flush
// per 128 KiB.
constexpr int64_t FA_U8_BLOCK = 4096 * 32;

static int64_t sq_diff_u8_simd(const uint8_t* a, const uint8_t* b, int64_t n) {
  const __m256i zero = _mm256_setzero_si256();
  int64_t total = 0;
  for (int64_t i = 0; i < n; i += FA_U8_BLOCK) {
    const int64_t stop = std::min(i + FA_U8_BLOCK, n);
    __m256i acc = zero;
    int64_t j = i;
    for (; j + 32 <= stop; j += 32) {
      const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
      const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
      // |a - b| on unsigned bytes, without ever forming a signed difference:
      // saturating subtract each way, then or.  One of the two is always zero.
      const __m256i d = _mm256_or_si256(_mm256_subs_epu8(va, vb),
                                        _mm256_subs_epu8(vb, va));
      const __m256i lo = _mm256_unpacklo_epi8(d, zero);
      const __m256i hi = _mm256_unpackhi_epi8(d, zero);
      acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lo, lo));
      acc = _mm256_add_epi32(acc, _mm256_madd_epi16(hi, hi));
    }
    // widen all eight i32 lanes to i64 and fold; the lanes are sums of squares,
    // so a zero-extend is the right widening
    const __m256i wide = _mm256_add_epi64(_mm256_unpacklo_epi32(acc, zero),
                                          _mm256_unpackhi_epi32(acc, zero));
    alignas(32) int64_t buf[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(buf), wide);
    total += buf[0] + buf[1] + buf[2] + buf[3];
    for (; j < stop; ++j) {
      const int d = static_cast<int>(a[j]) - static_cast<int>(b[j]);
      total += static_cast<int64_t>(d) * static_cast<int64_t>(d);
    }
  }
  return total;
}
#define FA_HAVE_U8_SIMD 1

#elif defined(FA_NEON)
// `vpadalq_u16` folds two u16 squares into each u32 lane per source register,
// so a lane grows by at most 4*255^2 = 260100 per iteration.  2^32 / 260100 =
// 16513; 4096 iterations again leaves a wide margin.
constexpr int64_t FA_U8_BLOCK = 4096 * 16;

static int64_t sq_diff_u8_simd(const uint8_t* a, const uint8_t* b, int64_t n) {
  int64_t total = 0;
  for (int64_t i = 0; i < n; i += FA_U8_BLOCK) {
    const int64_t stop = std::min(i + FA_U8_BLOCK, n);
    uint32x4_t acc = vdupq_n_u32(0);
    int64_t j = i;
    for (; j + 16 <= stop; j += 16) {
      const uint8x16_t d = vabdq_u8(vld1q_u8(a + j), vld1q_u8(b + j));
      const uint8x8_t dl = vget_low_u8(d);
      const uint8x8_t dh = vget_high_u8(d);
      acc = vpadalq_u16(acc, vmull_u8(dl, dl));
      acc = vpadalq_u16(acc, vmull_u8(dh, dh));
    }
    total += static_cast<int64_t>(vgetq_lane_u32(acc, 0)) +
             static_cast<int64_t>(vgetq_lane_u32(acc, 1)) +
             static_cast<int64_t>(vgetq_lane_u32(acc, 2)) +
             static_cast<int64_t>(vgetq_lane_u32(acc, 3));
    for (; j < stop; ++j) {
      const int d = static_cast<int>(a[j]) - static_cast<int>(b[j]);
      total += static_cast<int64_t>(d) * static_cast<int64_t>(d);
    }
  }
  return total;
}
#define FA_HAVE_U8_SIMD 1
#endif

// The penalty is evaluated in float and only the *sum* is double. That is not
// a shortcut: a float square root is an 8-wide AVX2 instruction against 4-wide
// for the double one, and it retires in roughly half the cycles, so the
// all-double loop ran the Charbonnier penalty at a quarter of the throughput
// for digits that a subsequent sum over millions of terms averages away
// anyway. MSE is the exception -- rounding the square before accumulating it
// is exactly the error the wide accumulator exists to avoid -- and so is a
// float64 input tensor, where the caller has asked for the precision.
template <int OP, typename A>
static inline A pix_term_cpu(A d, A p) {
  if (OP == FA_OP_MSE) return d * d;
  if (OP == FA_OP_L1) return std::fabs(d);
  if (OP == FA_OP_CHARB) return std::sqrt(d * d + p);
  // Huber as `0.5*min(a,delta)^2 + delta*(a - min(a,delta))`. Identical to the
  // piecewise definition, but it is one min and one subtract rather than a
  // select over two different expressions, so it survives auto-vectorisation.
  const A a = std::fabs(d);
  const A lo = a < p ? a : p;
  return A(0.5) * lo * lo + p * (a - lo);
}

template <typename T, int OP>
static double pix_sum_range(const T* a, const T* b, int64_t begin, int64_t end,
                            double p) {
  constexpr bool WIDE = (OP == FA_OP_MSE || std::is_same<T, double>::value);
  double acc = 0.0;
  if (WIDE) {
    for (int64_t i = begin; i < end; ++i) {
      const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
      acc += pix_term_cpu<OP, double>(d, p);
    }
  } else {
    const float pf = static_cast<float>(p);
    for (int64_t i = begin; i < end; ++i) {
      const float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
      acc += pix_term_cpu<OP, float>(d, pf);
    }
  }
  return acc;
}

template <int OP>
static double pix_sum_u8(const uint8_t* a, const uint8_t* b, int64_t begin,
                         int64_t end, double p) {
#if defined(FA_HAVE_U8_SIMD)
  // L1 is deliberately left to the compiler: the abs-and-widen form does
  // vectorise, and the auto-vectorised loop measures faster than a
  // hand-written `sad_epu8` chain.
  if (OP == FA_OP_MSE) {
    return static_cast<double>(sq_diff_u8_simd(a + begin, b + begin, end - begin));
  }
#endif
  if (OP == FA_OP_MSE || OP == FA_OP_L1) {
    int64_t acc = 0;
    for (int64_t i = begin; i < end; ++i) {
      const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
      acc += (OP == FA_OP_MSE) ? static_cast<int64_t>(d) * static_cast<int64_t>(d)
                               : static_cast<int64_t>(d < 0 ? -d : d);
    }
    return static_cast<double>(acc);
  }
  const float pf = static_cast<float>(p);
  double acc = 0.0;
  for (int64_t i = begin; i < end; ++i) {
    const float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
    acc += pix_term_cpu<OP, float>(d, pf);
  }
  return acc;
}

// Smallest slice worth handing to a second thread.  64 Ki elements is only
// about a microsecond of vectorised integer work, so this floor is only
// defensible while the pool's launch cost stays in the same range; raising it
// to 1 Mi -- on the theory that the faster kernel wanted coarser slices --
// measured 2.5x *worse* at 512x512 and no better anywhere else.
constexpr int64_t MIN_PARALLEL_CHUNK = 1 << 16;

// Per-image sums into `sums` (n_images doubles).
static int pixel_sums(const void* a, const void* b, int dtype,
                      int64_t n_images, int64_t per, int op, double param,
                      double* sums) {
  if (op < 0 || op > 3) return FA_E_OP;
  if (n_images <= 0 || per <= 0) return FA_E_SHAPE;

  // one chunk per image per thread-sized slice, so the whole batch is one
  // parallel region rather than N of them
  const int64_t chunk = std::max<int64_t>(MIN_PARALLEL_CHUNK,
                                          (per + pool().size() - 1) / pool().size());
  const int64_t per_img_chunks = (per + chunk - 1) / chunk;
  const int64_t ntasks = per_img_chunks * n_images;
  // the common shapes are a handful of tasks, and a heap allocation per call
  // is measurable against a reduction that now takes tens of microseconds
  double stack_partial[64];
  std::vector<double> heap_partial;
  double* partial = stack_partial;
  if (ntasks > static_cast<int64_t>(sizeof(stack_partial) / sizeof(double))) {
    heap_partial.resize(static_cast<size_t>(ntasks));
    partial = heap_partial.data();
  }

// Defined outside the dispatch switch below rather than inside it: a
// preprocessor directive may not appear within a macro's argument list, and
// FA_DISPATCH is a macro, so a #define nested in its body argument is
// ill-formed.  (This bit both MSVC and nvcc under AT_DISPATCH before.)
#define FA_CASE(OP)                                                            \
  case OP:                                                                     \
    v = is_u8 ? pix_sum_u8<OP>(au, bu, s, e, param)                            \
              : pix_sum_range<scalar_t, OP>(ap, bp, s, e, param);              \
    break;

  FA_DISPATCH(dtype, {
    const scalar_t* ap = static_cast<const scalar_t*>(a);
    const scalar_t* bp = static_cast<const scalar_t*>(b);
    const bool is_u8 = (dtype == FA_U8);
    const uint8_t* au = reinterpret_cast<const uint8_t*>(a);
    const uint8_t* bu = reinterpret_cast<const uint8_t*>(b);
    pool().run(ntasks, [&](int64_t t) {
      const int64_t img = t / per_img_chunks;
      const int64_t k = t - img * per_img_chunks;
      const int64_t s = img * per + k * chunk;
      const int64_t e = std::min(s + chunk, img * per + per);
      double v = 0.0;
      switch (op) {
        FA_CASE(FA_OP_MSE)
        FA_CASE(FA_OP_L1)
        FA_CASE(FA_OP_CHARB)
        default: FA_CASE(FA_OP_HUBER)
      }
      partial[t] = v;
    });
  })

#undef FA_CASE

  for (int64_t i = 0; i < n_images; ++i) {
    double acc = 0.0;
    const int64_t base = i * per_img_chunks;
    for (int64_t k = 0; k < per_img_chunks; ++k) acc += partial[base + k];
    sums[i] = acc;
  }
  return FA_OK;
}

// -------------------------------------------------------------------------
// SSIM
// -------------------------------------------------------------------------

struct Task {
  int plane, row0, rows, col0, cols;
};

template <typename T, bool WANT_CS>
static void ssim_tile(const T* xp, const T* yp, int H, int W, int Hout, int Wout,
                      const float* win, int K, float shift, float C1, float C2,
                      const Task& t, float* map_out, double* out_sum,
                      double* out_cs_sum) {
  const int cols = t.cols;
  const int span = cols + K - 1;   // input columns needed
  const int64_t plane_off = static_cast<int64_t>(t.plane) * H * W;

  // per-thread scratch
  std::vector<float> raw(5 * span);
  std::vector<float> ring(static_cast<size_t>(K) * 5 * cols);
  std::vector<float> acc(5 * cols);

  auto horiz_row = [&](int in_row, float* FA_RESTRICT dst) {
    const T* FA_RESTRICT xrow = xp + plane_off + static_cast<int64_t>(in_row) * W + t.col0;
    const T* FA_RESTRICT yrow = yp + plane_off + static_cast<int64_t>(in_row) * W + t.col0;
    float* FA_RESTRICT rx = raw.data();
    float* FA_RESTRICT ry = rx + span;
    float* FA_RESTRICT rxx = ry + span;
    float* FA_RESTRICT ryy = rxx + span;
    float* FA_RESTRICT rxy = ryy + span;
    for (int c = 0; c < span; ++c) {
      const float xv = static_cast<float>(xrow[c]) - shift;
      const float yv = static_cast<float>(yrow[c]) - shift;
      rx[c] = xv;
      ry[c] = yv;
      rxx[c] = xv * xv;
      ryy[c] = yv * yv;
      rxy[c] = xv * yv;
    }
    // j == 0 initialises instead of a separate memset pass
    for (int p = 0; p < 5; ++p) {
      const float* FA_RESTRICT src = rx + static_cast<size_t>(p) * span;
      float* FA_RESTRICT d = dst + static_cast<size_t>(p) * cols;
      const float w0 = win[0];
      for (int c = 0; c < cols; ++c) d[c] = w0 * src[c];
      for (int j = 1; j < K; ++j) {
        const float w = win[j];
        const float* FA_RESTRICT s = src + j;
        for (int c = 0; c < cols; ++c) d[c] += w * s[c];
      }
    }
  };

  // Prime the ring with the first K-1 input rows of this tile. The slot index
  // must be keyed on the *global* row number, because the read side below is
  // too -- keying it on the tile-local index only agrees when row0 % K == 0.
  for (int r = 0; r < K - 1; ++r) {
    const int row = t.row0 + r;
    horiz_row(row, ring.data() + static_cast<size_t>(row % K) * 5 * cols);
  }

  double sum = 0.0;
  double sum_cs = 0.0;
  for (int o = 0; o < t.rows; ++o) {
    const int newest = t.row0 + o + K - 1;
    horiz_row(newest, ring.data() + static_cast<size_t>(newest % K) * 5 * cols);

    for (int j = 0; j < K; ++j) {
      const float w = win[j];
      const float* FA_RESTRICT s =
          ring.data() + static_cast<size_t>((t.row0 + o + j) % K) * 5 * cols;
      float* FA_RESTRICT ap = acc.data();
      const int n5 = 5 * cols;
      if (j == 0) {
        for (int c = 0; c < n5; ++c) ap[c] = w * s[c];
      } else {
        for (int c = 0; c < n5; ++c) ap[c] += w * s[c];
      }
    }

    const float* FA_RESTRICT b0 = acc.data();
    const float* FA_RESTRICT b1 = b0 + cols;
    const float* FA_RESTRICT b2 = b1 + cols;
    const float* FA_RESTRICT b3 = b2 + cols;
    const float* FA_RESTRICT b4 = b3 + cols;
    float* FA_RESTRICT mrow = map_out
        ? map_out + static_cast<int64_t>(t.plane) * Hout * Wout +
              static_cast<int64_t>(t.row0 + o) * Wout + t.col0
        : nullptr;
    double rowsum = 0.0;
    double rowsum_cs = 0.0;
    for (int c = 0; c < cols; ++c) {
      const float mx = b0[c] + shift;
      const float my = b1[c] + shift;
      const float sxx = b2[c] - b0[c] * b0[c];
      const float syy = b3[c] - b1[c] * b1[c];
      const float sxy = b4[c] - b0[c] * b1[c];
      if (WANT_CS) {
        const float cs = (2.0f * sxy + C2) / (sxx + syy + C2);
        const float lum = (2.0f * mx * my + C1) / (mx * mx + my * my + C1);
        rowsum += static_cast<double>(lum * cs);
        rowsum_cs += static_cast<double>(cs);
      } else {
        const float num = (2.0f * mx * my + C1) * (2.0f * sxy + C2);
        const float den = (mx * mx + my * my + C1) * (sxx + syy + C2);
        const float v = num / den;
        if (mrow) mrow[c] = v;
        rowsum += static_cast<double>(v);
      }
    }
    sum += rowsum;
    sum_cs += rowsum_cs;
  }
  *out_sum = sum;
  if (WANT_CS) *out_cs_sum = sum_cs;
}

static int ssim_impl(const void* x, const void* y, int dtype, int N, int C,
                     int H, int W, const float* win, int K, double shift,
                     double C1, double C2, float* map_out, bool want_cs,
                     double* out_a, double* out_b) {
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return FA_E_SHAPE;
  if (K < 1 || K > MAX_WIN) return FA_E_WINDOW;
  const int Hout = H - K + 1, Wout = W - K + 1;
  if (Hout <= 0 || Wout <= 0) return FA_E_SHAPE;
  const int planes = N * C;

  std::vector<Task> tasks;
  for (int p = 0; p < planes; ++p)
    for (int r = 0; r < Hout; r += ROW_TILE)
      for (int c = 0; c < Wout; c += COL_TILE)
        tasks.push_back(Task{p, r, std::min(ROW_TILE, Hout - r), c,
                             std::min(COL_TILE, Wout - c)});

  std::vector<double> sums(tasks.size(), 0.0);
  std::vector<double> sums_cs(want_cs ? tasks.size() : 0, 0.0);

  FA_DISPATCH(dtype, {
    const scalar_t* xp = static_cast<const scalar_t*>(x);
    const scalar_t* yp = static_cast<const scalar_t*>(y);
    double* csp = want_cs ? sums_cs.data() : nullptr;
    pool().run(static_cast<int64_t>(tasks.size()), [&](int64_t i) {
      if (csp) {
        ssim_tile<scalar_t, true>(xp, yp, H, W, Hout, Wout, win, K, (float)shift,
                                  (float)C1, (float)C2, tasks[i], nullptr,
                                  &sums[i], &csp[i]);
      } else {
        ssim_tile<scalar_t, false>(xp, yp, H, W, Hout, Wout, win, K, (float)shift,
                                   (float)C1, (float)C2, tasks[i], map_out,
                                   &sums[i], nullptr);
      }
    });
  })

  // fold task sums back to one value per plane, in a fixed order
  for (int p = 0; p < planes; ++p) out_a[p] = 0.0;
  for (size_t i = 0; i < tasks.size(); ++i) out_a[tasks[i].plane] += sums[i];

  if (want_cs) {
    for (int p = 0; p < planes; ++p) out_b[p] = 0.0;
    for (size_t i = 0; i < tasks.size(); ++i) out_b[tasks[i].plane] += sums_cs[i];
    // divide, do not multiply by a precomputed reciprocal
    const double npix = static_cast<double>(Hout) * Wout;
    for (int p = 0; p < planes; ++p) {
      out_a[p] /= npix;
      out_b[p] /= npix;
    }
  }
  return FA_OK;
}

// -------------------------------------------------------------------------
// GMSD
//
// Same shape as the CUDA kernel: optional 2x box downsample, a Prewitt pair,
// the similarity map and its first two moments, with nothing full-resolution
// written out. Three downsampled rows are enough state to produce one output
// row, so the whole thing runs out of a 3-row ring in L1.
// -------------------------------------------------------------------------

template <typename T, bool DOWN>
static void gmsd_tile(const T* xp, const T* yp, int H, int W, int Hout,
                      int Wout, float Tc, float eps, const Task& t,
                      double* out_sum, double* out_sumsq) {
  const int cols = t.cols;
  const int span = cols + 2;
  const int64_t plane_off = static_cast<int64_t>(t.plane) * H * W;

  std::vector<float> bx(3 * span), by(3 * span);

  auto load_row = [&](int dr) {
    float* FA_RESTRICT dx = bx.data() + static_cast<size_t>(dr % 3) * span;
    float* FA_RESTRICT dy = by.data() + static_cast<size_t>(dr % 3) * span;
    if (DOWN) {
      const T* FA_RESTRICT xr = xp + plane_off + static_cast<int64_t>(2 * dr) * W + 2 * t.col0;
      const T* FA_RESTRICT yr = yp + plane_off + static_cast<int64_t>(2 * dr) * W + 2 * t.col0;
      for (int c = 0; c < span; ++c) {
        dx[c] = 0.25f * (static_cast<float>(xr[2 * c]) + static_cast<float>(xr[2 * c + 1]) +
                         static_cast<float>(xr[2 * c + W]) + static_cast<float>(xr[2 * c + W + 1]));
        dy[c] = 0.25f * (static_cast<float>(yr[2 * c]) + static_cast<float>(yr[2 * c + 1]) +
                         static_cast<float>(yr[2 * c + W]) + static_cast<float>(yr[2 * c + W + 1]));
      }
    } else {
      const T* FA_RESTRICT xr = xp + plane_off + static_cast<int64_t>(dr) * W + t.col0;
      const T* FA_RESTRICT yr = yp + plane_off + static_cast<int64_t>(dr) * W + t.col0;
      for (int c = 0; c < span; ++c) {
        dx[c] = static_cast<float>(xr[c]);
        dy[c] = static_cast<float>(yr[c]);
      }
    }
  };

  load_row(t.row0);
  load_row(t.row0 + 1);

  constexpr float THIRD = 1.0f / 3.0f;
  double sum = 0.0, sumsq = 0.0;
  for (int o = 0; o < t.rows; ++o) {
    load_row(t.row0 + o + 2);
    const float* FA_RESTRICT x0 = bx.data() + static_cast<size_t>((t.row0 + o) % 3) * span;
    const float* FA_RESTRICT x1 = bx.data() + static_cast<size_t>((t.row0 + o + 1) % 3) * span;
    const float* FA_RESTRICT x2 = bx.data() + static_cast<size_t>((t.row0 + o + 2) % 3) * span;
    const float* FA_RESTRICT y0 = by.data() + static_cast<size_t>((t.row0 + o) % 3) * span;
    const float* FA_RESTRICT y1 = by.data() + static_cast<size_t>((t.row0 + o + 1) % 3) * span;
    const float* FA_RESTRICT y2 = by.data() + static_cast<size_t>((t.row0 + o + 2) % 3) * span;

    for (int c = 0; c < cols; ++c) {
      const float ax = ((x0[c] - x0[c + 2]) + (x1[c] - x1[c + 2]) + (x2[c] - x2[c + 2])) * THIRD;
      const float ay = ((x0[c] - x2[c]) + (x0[c + 1] - x2[c + 1]) + (x0[c + 2] - x2[c + 2])) * THIRD;
      const float bx_ = ((y0[c] - y0[c + 2]) + (y1[c] - y1[c + 2]) + (y2[c] - y2[c + 2])) * THIRD;
      const float by_ = ((y0[c] - y2[c]) + (y0[c + 1] - y2[c + 1]) + (y0[c + 2] - y2[c + 2])) * THIRD;
      const float g1 = std::sqrt(ax * ax + ay * ay + eps);
      const float g2 = std::sqrt(bx_ * bx_ + by_ * by_ + eps);
      const float q = (2.0f * g1 * g2 + Tc) / (g1 * g1 + g2 * g2 + Tc);
      sum += static_cast<double>(q);
      sumsq += static_cast<double>(q) * static_cast<double>(q);
    }
  }
  *out_sum = sum;
  *out_sumsq = sumsq;
  (void)Hout;
  (void)Wout;
}

}  // namespace cpu
}  // namespace fa

// --------------------------------------------------------------------------
// C ABI
// --------------------------------------------------------------------------

using namespace fa::cpu;

extern "C" {

FA_API int fa_cpu_abi_version(void) { return FA_ABI_VERSION; }

FA_API int fa_cpu_has_avx2(void) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#if defined(_MSC_VER)
  int regs[4];
  __cpuid(regs, 0);
  if (regs[0] < 7) return 0;
  __cpuid(regs, 1);
  const bool osxsave = (regs[2] & (1 << 27)) != 0;
  const bool avx = (regs[2] & (1 << 28)) != 0;
  if (!osxsave || !avx) return 0;
  // XMM and YMM state must actually be saved by the OS, or a VEX-encoded
  // instruction faults even though CPUID advertises the feature.
  const unsigned long long xcr0 = _xgetbv(0);
  if ((xcr0 & 0x6) != 0x6) return 0;
  __cpuidex(regs, 7, 0);
  return (regs[1] & (1 << 5)) != 0 ? 1 : 0;
#else
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") ? 1 : 0;
#endif
#else
  return 0;
#endif
}

FA_API int fa_cpu_set_num_threads(int n) {
  if (n > 0) g_nthreads.store(n, std::memory_order_relaxed);
  return pool().size();
}

FA_API int fa_cpu_num_threads(void) { return pool().size(); }

// psnr_bias == FA_PSNR_NONE means "skip the dB conversion", matching the CUDA
// side's sentinel.
//
// Sums, divided, and optionally converted to dB -- the whole reduction, before
// returning to Python. Doing those last three steps as torch ops on scalar
// tensors cost ~10us of dispatch per call, which at 512x512 was half the total
// runtime of `psnr()`. The CUDA path has always folded them into its finalize
// kernel; this is the CPU counterpart.
FA_API int fa_cpu_pixel_reduce(const void* a, const void* b, int dtype,
                               int64_t n_images, int64_t n_per_image,
                               int op, double param, double psnr_bias,
                               int per_image, double* out) {
  if (!a || !b || !out) return FA_E_ARG;
  if (n_images <= 0 || n_per_image <= 0) return FA_E_SHAPE;

  double stack_sums[64];
  std::vector<double> heap_sums;
  double* sums = stack_sums;
  if (n_images > static_cast<int64_t>(sizeof(stack_sums) / sizeof(double))) {
    heap_sums.resize(static_cast<size_t>(n_images));
    sums = heap_sums.data();
  }

  const int rc = pixel_sums(a, b, dtype, n_images, n_per_image, op, param, sums);
  if (rc != FA_OK) return rc;

  const bool want_psnr = (psnr_bias != FA_PSNR_NONE);
  // divide, do not multiply by a precomputed reciprocal: the last ulp is the
  // difference between an exact integer uint8 sum and one that is off by one
  const double n_each = static_cast<double>(n_per_image);
  const double n_total = static_cast<double>(n_images) * n_each;

  if (!per_image) {
    double acc = 0.0;
    for (int64_t i = 0; i < n_images; ++i) acc += sums[i];
    const double mean = acc / n_total;
    // mean == 0 gives log10(0) == -inf, hence +inf dB, which is correct
    out[0] = want_psnr ? psnr_bias - 10.0 * std::log10(mean) : mean;
    return FA_OK;
  }
  for (int64_t i = 0; i < n_images; ++i) {
    const double m = sums[i] / n_each;
    out[i] = want_psnr ? psnr_bias - 10.0 * std::log10(m) : m;
  }
  return FA_OK;
}

FA_API int fa_cpu_ssim(const void* x, const void* y, int dtype,
                       int N, int C, int H, int W,
                       const float* win, int K,
                       double shift, double C1, double C2,
                       float* map_out, double* out_per_plane) {
  if (!x || !y || !win || !out_per_plane) return FA_E_ARG;
  return ssim_impl(x, y, dtype, N, C, H, W, win, K, shift, C1, C2, map_out,
                   false, out_per_plane, nullptr);
}

// Plane means of the SSIM map and of its contrast-structure factor -- the
// per-scale primitive MS-SSIM is built out of.
FA_API int fa_cpu_ssim_cs(const void* x, const void* y, int dtype,
                          int N, int C, int H, int W,
                          const float* win, int K,
                          double shift, double C1, double C2,
                          double* out_ssim, double* out_cs) {
  if (!x || !y || !win || !out_ssim || !out_cs) return FA_E_ARG;
  return ssim_impl(x, y, dtype, N, C, H, W, win, K, shift, C1, C2, nullptr,
                   true, out_ssim, out_cs);
}

FA_API int fa_cpu_gmsd(const void* x, const void* y, int dtype,
                       int N, int C, int H, int W,
                       double Tc, double eps, int downsample,
                       double* out_mean, double* out_std) {
  if (!x || !y || !out_mean || !out_std) return FA_E_ARG;
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return FA_E_SHAPE;

  const int Hd = downsample ? H / 2 : H;
  const int Wd = downsample ? W / 2 : W;
  const int Hout = Hd - 2, Wout = Wd - 2;
  if (Hout <= 0 || Wout <= 0) return FA_E_SHAPE;
  const int planes = N * C;

  std::vector<Task> tasks;
  for (int p = 0; p < planes; ++p)
    for (int r = 0; r < Hout; r += ROW_TILE)
      for (int c = 0; c < Wout; c += COL_TILE)
        tasks.push_back(Task{p, r, std::min(ROW_TILE, Hout - r), c,
                             std::min(COL_TILE, Wout - c)});

  std::vector<double> sums(tasks.size(), 0.0), sumsq(tasks.size(), 0.0);
  FA_DISPATCH(dtype, {
    const scalar_t* xp = static_cast<const scalar_t*>(x);
    const scalar_t* yp = static_cast<const scalar_t*>(y);
    pool().run(static_cast<int64_t>(tasks.size()), [&](int64_t i) {
      if (downsample) {
        gmsd_tile<scalar_t, true>(xp, yp, H, W, Hout, Wout, (float)Tc,
                                  (float)eps, tasks[i], &sums[i], &sumsq[i]);
      } else {
        gmsd_tile<scalar_t, false>(xp, yp, H, W, Hout, Wout, (float)Tc,
                                   (float)eps, tasks[i], &sums[i], &sumsq[i]);
      }
    });
  })

  for (int i = 0; i < N; ++i) { out_mean[i] = 0.0; out_std[i] = 0.0; }
  for (size_t i = 0; i < tasks.size(); ++i) {
    const int img = tasks[i].plane / C;
    out_mean[img] += sums[i];
    out_std[img] += sumsq[i];
  }
  // divide, do not multiply by a precomputed reciprocal: N * (1/N) is not
  // exactly 1, and that last ulp is the difference between gmsd(x, x) == 0
  // and gmsd(x, x) == 3e-08
  const double npix = static_cast<double>(Hout) * Wout * C;
  for (int i = 0; i < N; ++i) {
    const double m = out_mean[i] / npix;
    const double var = out_std[i] / npix - m * m;
    out_mean[i] = m;
    out_std[i] = std::sqrt(var > 0.0 ? var : 0.0);
  }
  return FA_OK;
}

}  // extern "C"
