#pragma once
/**
 * @file cuda.hpp
 * @brief Host interface to the CUDA GEMM kernels (src/cuda/gemm_kernels.cu).
 *
 * Eight levels, each adding one technique to the one it builds on. Levels
 * 0-4 run on the FMA (SIMT) cores in f32 and f64, Levels 5-7 on Tensor Cores:
 *
 *   0  gemm_cuda_naive           one thread per C(i,j), global memory only
 *   1  gemm_cuda_blocked         16x16 shared-memory tiles
 *   2  gemm_cuda_reg_tile        128x128 block tile, 8x8 register tile per thread
 *   3  gemm_cuda_double_buf      Level 2 + double-buffered shared memory (cp.async on sm_80+)
 *   4  gemm_cuda_vectorized      Level 2 + 128-bit loads + XOR-swizzled shared memory
 *   5  gemm_cuda_wmma            Tensor Cores through the wmma:: API, 64x64 tiles
 *   6  gemm_cuda_mma_ldmatrix    Level 5 through raw ldmatrix + mma.sync PTX
 *   7  gemm_cuda_wmma_pipelined  wmma:: with 128x128 tiles, 8 fragments per warp,
 *                                cp.async double buffering
 *   8  gemm_cuda_tma             raw mma.sync fed by TMA into swizzled shared
 *                                memory, 4-6-stage mbarrier pipeline, producer warp,
 *                                128x256 tiles (128x128 for small grids)
 *                                (src/cuda/gemm_lowp.cu, shared with cuda_lowp.hpp)
 *
 * Levels 5-8 are float-only: fp32 in and out, fp16 multiply, fp32
 * accumulate (input rounding at roughly the 1e-3 scale; output relative
 * error is not bounded near cancellation, and fp16 has a smaller range). A level whose requirement is not met
 * runs a level below it instead:
 *
 *   Level 4  needs K, N multiples of 4 (f32) / 2 (f64)      else Level 2
 *   Level 5  needs sm_70+                                  else Level 3
 *   Level 6  needs sm_80+                                  else Level 5
 *   Level 7  needs sm_70+, M and N multiples of 128, K of 32 else Level 5
 *   Level 8  needs sm_90+ (TMA)                             else Level 7
 *
 * gemm_cuda_cublas (SGEMM/DGEMM), gemm_cuda_cublas_tf32 and
 * gemm_cuda_cublas_fp16 are not part of the ladder: they are the vendor
 * reference the hand-written kernels are measured against.
 *
 * Every Matrix-based entry point allocates device buffers, copies A and B
 * in, runs the kernel and copies C back. The *_device entry points below
 * take buffers already on the GPU and only launch the GEMM: they are what
 * the benchmarks time, so GPU numbers, like CPU ones, measure the
 * multiplication alone.
 *
 * Without a CUDA toolkit the same API is provided by
 * src/cuda/gemm_kernels_stub.cpp: cuda_device_count() returns 0 and the
 * kernels throw.
 */
#include "hpc/matrix.hpp"

#include <cstddef>
#include <functional>

namespace hpc::gemm {

/// Number of CUDA devices; 0 in a build without CUDA.
int cuda_device_count() noexcept;
/// True if the selected device is sm_70+ (Tensor Cores).
bool cuda_has_tensor_cores() noexcept;
/// True if the selected device is sm_80+ (cp.async, mma.sync m16n8k16, TF32).
bool cuda_has_ampere() noexcept;
/// True if the selected device is sm_90+ (TMA, mbarrier transactions): Level 8.
bool cuda_has_tma() noexcept;

// Levels 0-7 (see the file comment).
template <typename T>
void gemm_cuda_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
template <typename T>
void gemm_cuda_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
template <typename T>
void gemm_cuda_reg_tile(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
template <typename T>
void gemm_cuda_double_buf(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
template <typename T>
void gemm_cuda_vectorized(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
void gemm_cuda_wmma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
void gemm_cuda_mma_ldmatrix(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
void gemm_cuda_wmma_pipelined(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
void gemm_cuda_tma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);

// cuBLAS reference. _tf32 and _fp16 run on Tensor Cores via cublasGemmEx;
// _fp16 converts the inputs to fp16 first. cuBLAS wrappers use one
// process-lifetime handle: one host thread and a fixed device after first use.
template <typename T>
void gemm_cuda_cublas(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
void gemm_cuda_cublas_tf32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
void gemm_cuda_cublas_fp16(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);

// ---------------------------------------------------------------------------
// Preconditions: dimensions and linear indices fit signed int; input/output
// fixed 2-D launch grids fit device limits; buffers do not overlap and are aligned for the selected path (16-byte
// vector loads, 32-byte WMMA output). Negative dimensions throw. K=0 clears
// C; M=0 or N=0 returns. Matrix wrappers also check the indexing limit.
//
// Device entry points: the same GEMMs on buffers the caller has allocated and
// filled on the GPU (row-major, M x K times K x N). Each enqueues work on the default
// stream, asynchronously, with the same fallback as its Matrix-based
// counterpart.
//
// fp16 buffers are typed void*, not __half*: this header is also compiled
// against the CPU-only stub, where <cuda_fp16.h> does not exist. Each such
// buffer holds `count` (or M*K / K*N) 2-byte halves.
//
// gemm_cuda_wmma_pipelined_device takes fp16 inputs (convert them with
// gemm_cuda_convert_f32_to_f16_device) and has no fallback: M and N must be
// multiples of 128 and K a multiple of 32.
// ---------------------------------------------------------------------------
template <typename T>
void gemm_cuda_naive_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
template <typename T>
void gemm_cuda_blocked_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
template <typename T>
void gemm_cuda_reg_tile_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
template <typename T>
void gemm_cuda_double_buf_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
template <typename T>
void gemm_cuda_vectorized_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
void gemm_cuda_wmma_device(const float* dA, const float* dB, float* dC, int M, int K, int N);
void gemm_cuda_mma_ldmatrix_device(const float* dA, const float* dB, float* dC, int M, int K, int N);
void gemm_cuda_wmma_pipelined_device(const void* dA16, const void* dB16, float* dC,
                                     int M, int K, int N);
// Level 8 takes fp16 A (M x K) and fp16 B transposed (N x K, "TN"): the
// layout its TMA copies and non-transposing ldmatrix read. No fallback: M
// and N multiples of 128, K of 32, sm_90+.
void gemm_cuda_tma_device(const void* dA16, const void* dBt16, float* dC, int M, int K, int N);
template <typename T>
void gemm_cuda_cublas_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
void gemm_cuda_cublas_tf32_device(const float* dA, const float* dB, float* dC,
                                  int M, int K, int N);
void gemm_cuda_cublas_fp16_device(const void* dA16, const void* dB16, float* dC,
                                  int M, int K, int N);
void gemm_cuda_convert_f32_to_f16_device(const float* src, void* dst, int count);

// GPU time of the work `launch` enqueues, in milliseconds: CUDA events are
// recorded on the GPU before and after it, so CPU work and
// scheduling noise outside the event interval are not counted. GPU idle
// gaps while the host submits work can be included, especially for tiny
// kernels. Waits for the work to finish.
double gemm_cuda_time_ms(const std::function<void()>& launch);

// Device-memory helpers for the benchmarks, so they need not include
// <cuda_runtime.h> (also unavailable on CPU-only builds).
void* gemm_cuda_malloc(std::size_t bytes);
void  gemm_cuda_free(void* ptr);
void  gemm_cuda_memcpy_h2d(void* dst, const void* src, std::size_t bytes);
void  gemm_cuda_memcpy_d2h(void* dst, const void* src, std::size_t bytes);
void  gemm_cuda_device_synchronize();

// ---------------------------------------------------------------------------
// Explicit instantiation declarations (definitions in gemm_kernels.cu)
// ---------------------------------------------------------------------------
extern template void gemm_cuda_naive<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
extern template void gemm_cuda_naive<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
extern template void gemm_cuda_blocked<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
extern template void gemm_cuda_blocked<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
extern template void gemm_cuda_reg_tile<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
extern template void gemm_cuda_reg_tile<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
extern template void gemm_cuda_double_buf<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
extern template void gemm_cuda_double_buf<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
extern template void gemm_cuda_vectorized<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
extern template void gemm_cuda_vectorized<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
extern template void gemm_cuda_cublas<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
extern template void gemm_cuda_cublas<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
#define HPC_CUDA_DEVICE_EXTERN(fn)                                                \
    extern template void fn<float>(const float*, const float*, float*, int, int, int); \
    extern template void fn<double>(const double*, const double*, double*, int, int, int);
HPC_CUDA_DEVICE_EXTERN(gemm_cuda_naive_device)
HPC_CUDA_DEVICE_EXTERN(gemm_cuda_blocked_device)
HPC_CUDA_DEVICE_EXTERN(gemm_cuda_reg_tile_device)
HPC_CUDA_DEVICE_EXTERN(gemm_cuda_double_buf_device)
HPC_CUDA_DEVICE_EXTERN(gemm_cuda_vectorized_device)
HPC_CUDA_DEVICE_EXTERN(gemm_cuda_cublas_device)
#undef HPC_CUDA_DEVICE_EXTERN
}  // namespace hpc::gemm
