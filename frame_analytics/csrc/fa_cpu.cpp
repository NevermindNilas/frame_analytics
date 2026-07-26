// CPU kernels + Python bindings for frame_analytics.
//
// Same structural trick as the CUDA path: the 11x11 Gaussian is separable and
// all five expectation planes (x, y, x^2, y^2, xy) are produced in one sweep,
// so nothing full-resolution is ever written to memory.  Work is tiled to
// (row-block x column-block) so each thread's ring buffer -- 11 rows x 5 planes
// x 256 columns = 56 KiB -- stays resident in L2.
//
// Inner loops are written j-outer / c-inner over contiguous float arrays so
// MSVC and GCC both auto-vectorise them to AVX2/AVX-512 FMA chains.

#include <torch/extension.h>
#include <ATen/Parallel.h>
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

#ifdef WITH_CUDA
namespace fa {
std::vector<torch::Tensor> mse_partial(torch::Tensor a, torch::Tensor b,
                                       int64_t n_images, double psnr_bias);
std::vector<torch::Tensor> ssim_fused(torch::Tensor x, torch::Tensor y,
                                      torch::Tensor win, double shift,
                                      double C1, double C2, bool want_map);
std::vector<torch::Tensor> ssim_backward(torch::Tensor x, torch::Tensor y,
                                         torch::Tensor dL_dmap,
                                         torch::Tensor grad_scalar,
                                         torch::Tensor win,
                                         double shift, double C1, double C2,
                                         bool need_dx, bool need_dy);
std::vector<torch::Tensor> pixel_partial(torch::Tensor a, torch::Tensor b,
                                         int64_t n_images, int64_t op,
                                         double param, double psnr_bias);
std::vector<torch::Tensor> ssim_cs(torch::Tensor x, torch::Tensor y,
                                   torch::Tensor win, double shift,
                                   double C1, double C2);
std::vector<torch::Tensor> ssim_cs_backward(torch::Tensor x, torch::Tensor y,
                                            torch::Tensor gs, torch::Tensor gcs,
                                            torch::Tensor win, double shift,
                                            double C1, double C2,
                                            bool need_dx, bool need_dy);
std::vector<torch::Tensor> gmsd_fused(torch::Tensor x, torch::Tensor y,
                                      double Tc, double eps, bool downsample);
}
#endif

namespace fa {
namespace cpu {

constexpr int COL_TILE = 256;
constexpr int ROW_TILE = 64;
constexpr int MAX_WIN = 11;

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

// Deliberately leaked: joining worker threads during static destruction races
// with the interpreter tearing down around us.
static TaskPool& pool() {
  static TaskPool* p = new TaskPool(std::max(0, static_cast<int>(at::get_num_threads()) - 1));
  return *p;
}

// -------------------------------------------------------------------------
// Pixel losses: MSE, L1, Charbonnier, Huber
//
// One templated sweep, as on the CUDA side. MSE and L1 over uint8 take an
// exact integer accumulator; the other two are float terms folded into a
// double sum, which is still far wider than the float32 that every other
// library reduces in.
// -------------------------------------------------------------------------

constexpr int OP_MSE = 0;
constexpr int OP_L1 = 1;
constexpr int OP_CHARB = 2;      // param = eps^2
constexpr int OP_HUBER = 3;      // param = delta

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
  if (OP == OP_MSE) return d * d;
  if (OP == OP_L1) return std::fabs(d);
  if (OP == OP_CHARB) return std::sqrt(d * d + p);
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
  constexpr bool WIDE = (OP == OP_MSE || std::is_same<T, double>::value);
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
  if (OP == OP_MSE) {
    return static_cast<double>(sq_diff_u8_simd(a + begin, b + begin, end - begin));
  }
#endif
  if (OP == OP_MSE || OP == OP_L1) {
    int64_t acc = 0;
    for (int64_t i = begin; i < end; ++i) {
      const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
      acc += (OP == OP_MSE) ? static_cast<int64_t>(d) * static_cast<int64_t>(d)
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

torch::Tensor pixel_sums(torch::Tensor a, torch::Tensor b, int64_t n_images,
                         int64_t op, double param) {
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
  TORCH_CHECK(op >= 0 && op <= 3, "unknown pixel op ", op);
  a = a.contiguous();
  b = b.contiguous();
  const int64_t total = a.numel();
  TORCH_CHECK(n_images > 0 && total % n_images == 0, "numel not divisible by batch");
  const int64_t per = total / n_images;

  auto out = torch::empty({n_images}, torch::dtype(torch::kFloat64));
  double* op_out = out.data_ptr<double>();

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

// Defined outside the AT_DISPATCH call below, not inside it: a preprocessor
// directive may not appear within a macro's argument list, and AT_DISPATCH is a
// macro, so a #define nested in its lambda argument is ill-formed.
#define FA_CASE(OP)                                                            \
  case OP:                                                                     \
    v = is_u8 ? pix_sum_u8<OP>(au, bu, s, e, param)                            \
              : pix_sum_range<scalar_t, OP>(ap, bp, s, e, param);              \
    break;

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             a.scalar_type(), "pixel_sums", [&] {
    const scalar_t* ap = a.data_ptr<scalar_t>();
    const scalar_t* bp = b.data_ptr<scalar_t>();
    const bool is_u8 = (a.scalar_type() == torch::kUInt8);
    const uint8_t* au = reinterpret_cast<const uint8_t*>(ap);
    const uint8_t* bu = reinterpret_cast<const uint8_t*>(bp);
    pool().run(ntasks, [&](int64_t t) {
      const int64_t img = t / per_img_chunks;
      const int64_t k = t - img * per_img_chunks;
      const int64_t s = img * per + k * chunk;
      const int64_t e = std::min(s + chunk, img * per + per);
      double v = 0.0;
      switch ((int)op) {
        FA_CASE(OP_MSE)
        FA_CASE(OP_L1)
        FA_CASE(OP_CHARB)
        default: FA_CASE(OP_HUBER)
      }
      partial[t] = v;
    });
  });

#undef FA_CASE

  for (int64_t i = 0; i < n_images; ++i) {
    double acc = 0.0;
    const int64_t base = i * per_img_chunks;
    for (int64_t k = 0; k < per_img_chunks; ++k) acc += partial[base + k];
    op_out[i] = acc;
  }
  return out;
}

torch::Tensor mse_sums(torch::Tensor a, torch::Tensor b, int64_t n_images) {
  return pixel_sums(a, b, n_images, OP_MSE, 0.0);
}

// psnr_bias == PSNR_NONE means "skip the dB conversion", matching the CUDA
// side's sentinel.
constexpr double PSNR_NONE = -1.0e300;

// Sums, divided, and optionally converted to dB -- the whole reduction, before
// returning to Python.  Doing those last three steps as torch ops on scalar
// tensors cost ~10us of dispatch per call, which at 512x512 was half the total
// runtime of `psnr()`.  The CUDA path has always folded them into its finalize
// kernel; this is the CPU counterpart.
//
// One tensor out, not the CUDA path's four.  There the four are free -- a
// single kernel writes all of them and the caller picks -- but here each one is
// an allocation plus a pybind wrapper, ~1us apiece, and at this size that was
// again half the runtime.  So the CPU side takes the selector as an argument
// and allocates only what was asked for.
torch::Tensor pixel_reduce(torch::Tensor a, torch::Tensor b, int64_t n_images,
                           int64_t op, double param, double psnr_bias,
                           bool per_image) {
  const int64_t total = a.numel();
  auto sums = pixel_sums(a, b, n_images, op, param);
  const double* s = sums.data_ptr<double>();
  const bool want_psnr = (psnr_bias != PSNR_NONE);

  // divide, do not multiply by a precomputed reciprocal: the last ulp is the
  // difference between an exact integer uint8 sum and one that is off by one
  const double n_each = static_cast<double>(total / n_images);

  auto opts = torch::dtype(torch::kFloat64);
  if (!per_image) {
    double acc = 0.0;
    for (int64_t i = 0; i < n_images; ++i) acc += s[i];
    const double mean = acc / static_cast<double>(total);
    auto out = torch::empty({}, opts);
    // mean == 0 gives log10(0) == -inf, hence +inf dB, which is correct
    *out.data_ptr<double>() = want_psnr ? psnr_bias - 10.0 * std::log10(mean) : mean;
    return out;
  }

  auto out = torch::empty({n_images}, opts);
  double* o = out.data_ptr<double>();
  for (int64_t i = 0; i < n_images; ++i) {
    const double m = s[i] / n_each;
    o[i] = want_psnr ? psnr_bias - 10.0 * std::log10(m) : m;
  }
  return out;
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

std::vector<torch::Tensor> ssim_sums_impl(torch::Tensor x, torch::Tensor y,
                                          torch::Tensor win, double shift,
                                          double C1, double C2, bool want_map,
                                          bool want_cs) {
  TORCH_CHECK(x.dim() == 4 && x.sizes() == y.sizes(), "expected matching NCHW");
  x = x.contiguous();
  y = y.contiguous();
  auto winf = win.to(torch::kFloat32).contiguous();
  const int K = static_cast<int>(winf.numel());
  TORCH_CHECK(K <= MAX_WIN && K >= 1, "unsupported window size");

  const int N = (int)x.size(0), C = (int)x.size(1);
  const int H = (int)x.size(2), W = (int)x.size(3);
  const int Hout = H - K + 1, Wout = W - K + 1;
  TORCH_CHECK(Hout > 0 && Wout > 0, "image smaller than window");
  const int planes = N * C;

  std::vector<Task> tasks;
  for (int p = 0; p < planes; ++p)
    for (int r = 0; r < Hout; r += ROW_TILE)
      for (int c = 0; c < Wout; c += COL_TILE)
        tasks.push_back(Task{p, r, std::min(ROW_TILE, Hout - r), c,
                             std::min(COL_TILE, Wout - c)});

  std::vector<double> sums(tasks.size(), 0.0);
  std::vector<double> sums_cs(want_cs ? tasks.size() : 0, 0.0);
  torch::Tensor map;
  float* map_ptr = nullptr;
  if (want_map) {
    map = torch::empty({N, C, Hout, Wout}, torch::dtype(torch::kFloat32));
    map_ptr = map.data_ptr<float>();
  }

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             x.scalar_type(), "ssim_sums", [&] {
    const scalar_t* xp = x.data_ptr<scalar_t>();
    const scalar_t* yp = y.data_ptr<scalar_t>();
    const float* wp = winf.data_ptr<float>();
    double* csp = want_cs ? sums_cs.data() : nullptr;
    pool().run((int64_t)tasks.size(), [&](int64_t i) {
      if (csp) {
        ssim_tile<scalar_t, true>(xp, yp, H, W, Hout, Wout, wp, K, (float)shift,
                                  (float)C1, (float)C2, tasks[i], nullptr,
                                  &sums[i], &csp[i]);
      } else {
        ssim_tile<scalar_t, false>(xp, yp, H, W, Hout, Wout, wp, K, (float)shift,
                                   (float)C1, (float)C2, tasks[i], map_ptr,
                                   &sums[i], nullptr);
      }
    });
  });

  // fold task sums back to one value per plane, in a fixed order
  auto per_plane = torch::zeros({planes}, torch::dtype(torch::kFloat64));
  double* pp = per_plane.data_ptr<double>();
  for (size_t i = 0; i < tasks.size(); ++i) pp[tasks[i].plane] += sums[i];

  if (want_cs) {
    auto per_plane_cs = torch::zeros({planes}, torch::dtype(torch::kFloat64));
    double* pc = per_plane_cs.data_ptr<double>();
    for (size_t i = 0; i < tasks.size(); ++i) pc[tasks[i].plane] += sums_cs[i];
    const double npix = static_cast<double>(Hout) * Wout;
    return {per_plane.div_(npix), per_plane_cs.div_(npix)};
  }
  if (want_map) return {per_plane, map};
  return {per_plane};
}

std::vector<torch::Tensor> ssim_sums(torch::Tensor x, torch::Tensor y,
                                     torch::Tensor win, double shift,
                                     double C1, double C2, bool want_map) {
  return ssim_sums_impl(x, y, win, shift, C1, C2, want_map, false);
}

// Plane means of the SSIM map and of its contrast-structure factor -- the
// per-scale primitive MS-SSIM is built out of.
std::vector<torch::Tensor> ssim_cs(torch::Tensor x, torch::Tensor y,
                                   torch::Tensor win, double shift,
                                   double C1, double C2) {
  return ssim_sums_impl(x, y, win, shift, C1, C2, false, true);
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
static void gmsd_tile(const T* xp, const T* yp, int H, int W, int Hd, int Wd,
                      int Hout, int Wout, float Tc, float eps, const Task& t,
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
  (void)Hd;
  (void)Wd;
  (void)Hout;
  (void)Wout;
}

std::vector<torch::Tensor> gmsd_sums(torch::Tensor x, torch::Tensor y,
                                     double Tc, double eps, bool downsample) {
  TORCH_CHECK(x.dim() == 4 && x.sizes() == y.sizes(), "expected matching NCHW");
  x = x.contiguous();
  y = y.contiguous();

  const int N = (int)x.size(0), C = (int)x.size(1);
  const int H = (int)x.size(2), W = (int)x.size(3);
  const int Hd = downsample ? H / 2 : H;
  const int Wd = downsample ? W / 2 : W;
  const int Hout = Hd - 2, Wout = Wd - 2;
  TORCH_CHECK(Hout > 0 && Wout > 0, "image too small for a 3x3 gradient");
  const int planes = N * C;

  std::vector<Task> tasks;
  for (int p = 0; p < planes; ++p)
    for (int r = 0; r < Hout; r += ROW_TILE)
      for (int c = 0; c < Wout; c += COL_TILE)
        tasks.push_back(Task{p, r, std::min(ROW_TILE, Hout - r), c,
                             std::min(COL_TILE, Wout - c)});

  std::vector<double> sums(tasks.size(), 0.0), sumsq(tasks.size(), 0.0);
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             x.scalar_type(), "gmsd_sums", [&] {
    const scalar_t* xp = x.data_ptr<scalar_t>();
    const scalar_t* yp = y.data_ptr<scalar_t>();
    pool().run((int64_t)tasks.size(), [&](int64_t i) {
      if (downsample) {
        gmsd_tile<scalar_t, true>(xp, yp, H, W, Hd, Wd, Hout, Wout, (float)Tc,
                                  (float)eps, tasks[i], &sums[i], &sumsq[i]);
      } else {
        gmsd_tile<scalar_t, false>(xp, yp, H, W, Hd, Wd, Hout, Wout, (float)Tc,
                                   (float)eps, tasks[i], &sums[i], &sumsq[i]);
      }
    });
  });

  auto mean = torch::zeros({N}, torch::dtype(torch::kFloat64));
  auto dev = torch::zeros({N}, torch::dtype(torch::kFloat64));
  double* mp = mean.data_ptr<double>();
  double* dp = dev.data_ptr<double>();
  for (size_t i = 0; i < tasks.size(); ++i) {
    const int img = tasks[i].plane / C;
    mp[img] += sums[i];
    dp[img] += sumsq[i];
  }
  // divide, do not multiply by a precomputed reciprocal: N * (1/N) is not
  // exactly 1, and that last ulp is the difference between gmsd(x, x) == 0
  // and gmsd(x, x) == 3e-08
  const double npix = static_cast<double>(Hout) * Wout * C;
  for (int i = 0; i < N; ++i) {
    const double m = mp[i] / npix;
    const double var = dp[i] / npix - m * m;
    mp[i] = m;
    dp[i] = std::sqrt(var > 0.0 ? var : 0.0);
  }
  return {mean, dev};
}

}  // namespace cpu
}  // namespace fa

// -------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mse_sums_cpu", &fa::cpu::mse_sums, "per-image squared-error sums (CPU)");
  m.def("pixel_sums_cpu", &fa::cpu::pixel_sums, "per-image pixel-loss sums (CPU)");
  m.def("pixel_reduce_cpu", &fa::cpu::pixel_reduce,
        "per-image and batch pixel-loss means, optionally in dB (CPU)");
  m.def("ssim_sums_cpu", &fa::cpu::ssim_sums, "per-plane SSIM sums (CPU)");
  m.def("ssim_cs_cpu", &fa::cpu::ssim_cs, "per-plane SSIM and cs means (CPU)");
  m.def("gmsd_sums_cpu", &fa::cpu::gmsd_sums, "per-image GMS mean and deviation (CPU)");
#ifdef WITH_CUDA
  m.def("mse_partial_cuda", &fa::mse_partial, "block partial squared-error sums (CUDA)");
  m.def("pixel_partial_cuda", &fa::pixel_partial, "block partial pixel-loss sums (CUDA)");
  m.def("ssim_fused_cuda", &fa::ssim_fused, "fused SSIM block sums (CUDA)");
  m.def("ssim_backward_cuda", &fa::ssim_backward, "fused SSIM backward (CUDA)");
  m.def("ssim_cs_cuda", &fa::ssim_cs, "fused SSIM and cs plane means (CUDA)");
  m.def("ssim_cs_backward_cuda", &fa::ssim_cs_backward, "fused MS-SSIM backward (CUDA)");
  m.def("gmsd_cuda", &fa::gmsd_fused, "fused GMS mean and deviation (CUDA)");
  m.attr("has_cuda") = true;
#else
  m.attr("has_cuda") = false;
#endif
}
