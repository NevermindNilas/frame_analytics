// frame_analytics -- stable C ABI.
//
// Nothing in this header mentions libtorch, pybind11 or Python.  That is the
// entire point of it: a shared library built against this boundary links only
// against the C runtime (and, for the CUDA half, a statically linked cudart
// that dlopens the driver), so one binary per platform is valid for every
// Python version and every torch version.  A torch C++ extension is ABI-locked
// to the exact {python} x {torch} x {CUDA} build it was compiled against, and a
// wheel filename has no field in which to encode that -- which is why the
// extension route cannot be shipped on PyPI at all.
//
// Conventions
// -----------
//   * every entry point returns an int status; FA_OK means success
//   * negative values are CUDA runtime errors, negated (see
//     fa_cuda_error_string)
//   * the caller owns every buffer.  Outputs are allocated on the Python side
//     (through torch, so they come from the caching allocator and remain legal
//     inside a CUDA graph capture) and passed in as raw pointers
//   * shapes travel as ints, dtypes as an int code, streams as an opaque handle
//   * nothing here allocates device memory, and nothing synchronises

#ifndef FRAME_ANALYTICS_ABI_H
#define FRAME_ANALYTICS_ABI_H

#include <stdint.h>

#if defined(_WIN32)
#define FA_API __declspec(dllexport)
#else
#define FA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Bumped whenever a signature in this header changes meaning.  The loader
// checks it after dlopen/LoadLibrary, so a stale prebuilt binary sitting next
// to newer Python sources is rejected loudly instead of corrupting results.
#define FA_ABI_VERSION 1

// dtype codes -- mirrors frame_analytics.backend._DTYPE_CODE
#define FA_U8    0
#define FA_F16   1
#define FA_BF16  2
#define FA_F32   3
#define FA_F64   4

// pixel-loss selector -- mirrors functional._OP_*
#define FA_OP_MSE    0
#define FA_OP_L1     1
#define FA_OP_CHARB  2   /* param = eps^2   */
#define FA_OP_HUBER  3   /* param = delta   */

// status codes.  Negative returns are -(cudaError_t).
#define FA_OK          0
#define FA_E_DTYPE     1
#define FA_E_SHAPE     2
#define FA_E_WINDOW    3
#define FA_E_OP        4
#define FA_E_ARG       5
#define FA_E_INTERNAL  6
#define FA_E_NOCUDA    7

// "not an MSE reduction, skip the dB conversion"
#define FA_PSNR_NONE (-1.0e300)

// Largest supported Gaussian window (the kernels are instantiated for odd
// widths 3..11).
#define FA_MAX_WIN 11

// ------------------------------------------------------------------------- //
// CPU
// ------------------------------------------------------------------------- //

FA_API int fa_cpu_abi_version(void);

// Does *this* binary's host CPU support AVX2?  Always compiled into the
// baseline library, which is safe to load anywhere; the loader uses the answer
// to decide whether the AVX2 sibling may be used instead.
FA_API int fa_cpu_has_avx2(void);

// The task pool is created on first use.  Call this before then (the loader
// does, with torch.get_num_threads()); afterwards it is a no-op.  Returns the
// pool size actually in effect, counting the calling thread.
FA_API int fa_cpu_set_num_threads(int n);
FA_API int fa_cpu_num_threads(void);

// out: n_images doubles when per_image, otherwise exactly one.
FA_API int fa_cpu_pixel_reduce(const void* a, const void* b, int dtype,
                               int64_t n_images, int64_t n_per_image,
                               int op, double param, double psnr_bias,
                               int per_image, double* out);

// out_per_plane: N*C doubles, the *sum* of each plane's SSIM map (not divided).
// map_out: optional, [N, C, H-K+1, W-K+1] float32.
FA_API int fa_cpu_ssim(const void* x, const void* y, int dtype,
                       int N, int C, int H, int W,
                       const float* win, int K,
                       double shift, double C1, double C2,
                       float* map_out, double* out_per_plane);

// out_ssim / out_cs: N*C doubles each, already divided by the map area.
FA_API int fa_cpu_ssim_cs(const void* x, const void* y, int dtype,
                          int N, int C, int H, int W,
                          const float* win, int K,
                          double shift, double C1, double C2,
                          double* out_ssim, double* out_cs);

// out_mean / out_std: N doubles each.
FA_API int fa_cpu_gmsd(const void* x, const void* y, int dtype,
                       int N, int C, int H, int W,
                       double Tc, double eps, int downsample,
                       double* out_mean, double* out_std);

// ------------------------------------------------------------------------- //
// CUDA
//
// Every entry point takes the device ordinal and an opaque stream handle
// (cast a torch.cuda.Stream's cuda_stream to void*).  Passing the device
// explicitly rather than relying on the ambient current device is what
// replaces at::cuda::CUDAGuard.
//
// The *_workspace queries exist because the block counts these kernels use are
// a property of the kernel, not of the shapes alone -- but the scratch buffer
// has to come from torch's allocator to stay capture-safe, so Python has to be
// told how big it is.
// ------------------------------------------------------------------------- //

FA_API int fa_cuda_abi_version(void);
FA_API const char* fa_cuda_error_string(int code);
FA_API int fa_cuda_device_count(int* out_count);

FA_API int fa_cuda_pixel_workspace(int64_t n_images, int64_t n_per_image,
                                   int device, int* out_bpi);

// partial: [n_images, bpi] float64 scratch.
// out_psnr_* may be null when psnr_bias == FA_PSNR_NONE.
FA_API int fa_cuda_pixel_reduce(const void* a, const void* b, int dtype,
                                int64_t n_images, int64_t n_per_image,
                                int op, double param, double psnr_bias,
                                double* partial, int bpi,
                                double* out_per_image, double* out_total,
                                double* out_psnr_per_image,
                                double* out_psnr_total,
                                int device, void* stream);

FA_API int fa_cuda_ssim_workspace(int H, int W, int K, int* out_bpp);

// partial: [N*C, bpp] float64.  out_per_image: N.  out_total: 1.
FA_API int fa_cuda_ssim(const void* x, const void* y, int dtype,
                        int N, int C, int H, int W,
                        const float* win, int K,
                        double shift, double C1, double C2,
                        float* map_out,
                        double* partial, int bpp,
                        double* out_per_image, double* out_total,
                        int device, void* stream);

// out_s / out_cs: N*C doubles each, per-plane means.
FA_API int fa_cuda_ssim_cs(const void* x, const void* y, int dtype,
                           int N, int C, int H, int W,
                           const float* win, int K,
                           double shift, double C1, double C2,
                           double* partial, double* partial_cs, int bpp,
                           double* out_s, double* out_cs,
                           int device, void* stream);

// aux: (dy ? 4 : 3) * N*C * (H-K+1) * (W-K+1) floats of scratch.
// Exactly one of dL_dmap (a full map) and grad_scalar (a 1-element *device*
// tensor) must be non-null; the scalar form is what keeps a training step free
// of device->host syncs.
FA_API int fa_cuda_ssim_backward(const float* x, const float* y,
                                 const float* dL_dmap,
                                 const float* grad_scalar,
                                 const float* win, int K,
                                 int N, int C, int H, int W,
                                 double shift, double C1, double C2,
                                 float* aux, float* dx, float* dy,
                                 int device, void* stream);

// gs / gcs: N*C floats each, on the device.
FA_API int fa_cuda_ssim_cs_backward(const float* x, const float* y,
                                    const float* gs, const float* gcs,
                                    const float* win, int K,
                                    int N, int C, int H, int W,
                                    double shift, double C1, double C2,
                                    float* aux, float* dx, float* dy,
                                    int device, void* stream);

FA_API int fa_cuda_gmsd_workspace(int H, int W, int downsample, int* out_bpp);

// psum / psumsq: [N*C, bpp] float64 each.  out_mean / out_std: N each.
FA_API int fa_cuda_gmsd(const void* x, const void* y, int dtype,
                        int N, int C, int H, int W,
                        double Tc, double eps, int downsample,
                        double* psum, double* psumsq, int bpp,
                        double* out_mean, double* out_std,
                        int device, void* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FRAME_ANALYTICS_ABI_H
