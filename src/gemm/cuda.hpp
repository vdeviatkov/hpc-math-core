#pragma once
/**
 * @file cuda.hpp
 * @brief Host interface to the CUDA GEMM kernels (src/cuda/gemm_kernels.cu).
 *
 * Eight levels, each adding one technique to the one it builds on:
 *
 *   0  gemm_cuda_naive           one thread per C(i,j), global memory only
 *   1  gemm_cuda_blocked         16x16 shared-memory tiles
 *   2  gemm_cuda_reg_tile        128x128 block tile, 8x8 register tile per thread
 *   3  gemm_cuda_double_buf      Level 2 + double-buffered shared memory (cp.async on sm_80+)
 *   4  gemm_cuda_wmma            Tensor Cores through the wmma:: API, 64x64 tiles
 *   5  gemm_cuda_vectorized      Level 2 + 128-bit loads + XOR-swizzled shared memory
 *   6  gemm_cuda_mma_ldmatrix    Tensor Cores through raw ldmatrix + mma.sync PTX
 *   7  gemm_cuda_wmma_pipelined  wmma:: with 128x128 tiles, 8 fragments per warp,
 *                                cp.async double buffering
 *
 * Levels 4, 6 and 7 are float-only: fp32 in and out, fp16 multiply, fp32
 * accumulate (~1e-3 relative error). A level whose requirement is not met
 * runs the level below it instead:
 *
 *   Level 4  needs sm_70+                                  else Level 3
 *   Level 5  needs K, N multiples of 4 (f32) / 2 (f64)      else Level 2
 *   Level 6  needs sm_80+                                  else Level 4
 *   Level 7  needs sm_70+, M and N multiples of 128, K of 32 else Level 4
 *
 * gemm_cuda_cublas (SGEMM/DGEMM), gemm_cuda_cublas_tf32 and
 * gemm_cuda_cublas_fp16 are not part of the ladder: they are the vendor
 * reference the hand-written kernels are measured against.
 *
 * Every Matrix-based entry point allocates device buffers, copies A and B
 * in, runs the kernel and copies C back, so its timing includes transfers.
 * The *_device entry points below take device buffers instead, for
 * compute-only timing.
 *
 * Without a CUDA toolkit the same API is provided by
 * src/cuda/gemm_kernels_stub.cpp: cuda_device_count() returns 0 and the
 * kernels throw.
 */
#include "hpc/matrix.hpp"

#include <cstddef>

namespace hpc::gemm {

/// Number of CUDA devices; 0 in a build without CUDA.
int cuda_device_count() noexcept;
/// True if any device is sm_70+ (Tensor Cores).
bool cuda_has_tensor_cores() noexcept;
/// True if any device is sm_80+ (cp.async, mma.sync m16n8k16, TF32).
bool cuda_has_ampere() noexcept;

// Levels 0-7 (see the file comment).
template <typename T>
void gemm_cuda_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
template <typename T>
void gemm_cuda_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
template <typename T>
void gemm_cuda_reg_tile(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
template <typename T>
void gemm_cuda_double_buf(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
void gemm_cuda_wmma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
template <typename T>
void gemm_cuda_vectorized(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
void gemm_cuda_mma_ldmatrix(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
void gemm_cuda_wmma_pipelined(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);

// cuBLAS reference. _tf32 and _fp16 run on Tensor Cores via cublasGemmEx;
// _fp16 converts the inputs to fp16 first.
template <typename T>
void gemm_cuda_cublas(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
void gemm_cuda_cublas_tf32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
void gemm_cuda_cublas_fp16(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);

// ---------------------------------------------------------------------------
// Compute-only entry points: operate on device buffers the caller has already
// allocated and filled, so a benchmark can time the GEMM alone, without the
// per-call cudaMalloc and PCIe copies that dominate at large N.
//
// fp16 buffers are typed void*, not __half*: this header is also compiled
// against the CPU-only stub, where <cuda_fp16.h> does not exist. Each such
// buffer holds `count` (or M*K / K*N) 2-byte halves.
//
// gemm_cuda_wmma_pipelined_device has no fallback: M and N must be multiples
// of 128 and K a multiple of 32.
// ---------------------------------------------------------------------------
void gemm_cuda_cublas_device_f32(const float* dA, const float* dB, float* dC,
                                  int M, int K, int N);
void gemm_cuda_cublas_tf32_device(const float* dA, const float* dB, float* dC,
                                   int M, int K, int N);
void gemm_cuda_convert_f32_to_f16_device(const float* src, void* dst, int count);
void gemm_cuda_cublas_fp16_device(const void* dA16, const void* dB16, float* dC,
                                   int M, int K, int N);
void gemm_cuda_wmma_pipelined_device(const void* dA16, const void* dB16, float* dC,
                                      int M, int K, int N);

// Device-memory helpers for the compute-only benchmarks, so they need not
// include <cuda_runtime.h> (also unavailable on CPU-only builds).
void* gemm_cuda_malloc(std::size_t bytes);
void  gemm_cuda_free(void* ptr);
void  gemm_cuda_memcpy_h2d(void* dst, const void* src, std::size_t bytes);
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
}  // namespace hpc::gemm
