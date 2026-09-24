#pragma once
/**
 * @file cuda.hpp
 * @brief Host-side C++ interface for CUDA GEMM kernels.
 *
 * Kernel ladder (nine levels of optimization):
 *
 *   Level 0  -- gemm_cuda_naive
 *     One thread per C(i,j), no shared memory. DRAM-bound at all sizes.
 *
 *   Level 0b -- gemm_cuda_reordered
 *     CPU-symmetry naming; structurally identical to naive on GPU.
 *
 *   Level 1  -- gemm_cuda_blocked  (TILE=16)
 *     TILE x TILE shared-memory tiling, 16x fewer HBM loads.
 *     Each thread owns 1 output element -- low arithmetic intensity (~2 FLOP/byte).
 *
 *   Level 2  -- gemm_cuda_reg_tile  (128x128 block, 8x8 register tile)
 *     Each thread owns 64 outputs. Arithmetic intensity ~32 FLOP/byte.
 *     Typical: 50-75% of GPU peak FP32 throughput.
 *
 *   Level 3  -- gemm_cuda_double_buf
 *     Level 2 + double-buffered shared memory to overlap load and compute.
 *     On Ampere+ (sm_80+): uses cp.async for hardware async DMA.
 *     Typical: 75-85% of GPU peak FP32 throughput.
 *
 *   Level 4  -- gemm_cuda_wmma  (Tensor Cores, fp32 only, sm_70+)
 *     Converts fp32->fp16 on-the-fly, wmma::mma_sync Tensor Core MMA,
 *     accumulates in fp32. ~8x throughput vs SIMT FP32.
 *     fp16 conversion introduces ~1e-3 relative error.
 *     Falls back to gemm_cuda_double_buf on pre-Volta hardware.
 *
 *   Level 5  -- gemm_cuda_vectorized  (float4/double2 loads + smem XOR swizzle)
 *     Same register-tile shape as Level 2, but global->shared loads use
 *     128-bit vector instructions and shared memory uses a self-consistent
 *     XOR swizzle instead of +1 padding. Requires K and N to be multiples
 *     of the vector width (4 for float, 2 for double); falls back to
 *     gemm_cuda_reg_tile otherwise.
 *
 *   Level 6  -- gemm_cuda_mma_ldmatrix  (raw Tensor Cores, fp32 only, sm_80+)
 *     Same computation as Level 4 (fp16 in, fp32 accumulate) one level
 *     below the WMMA C++ API: hand-issued ldmatrix.sync + mma.sync PTX.
 *     UNVERIFIED (see file header in gemm_kernels.cu) -- no CUDA hardware
 *     or toolkit was available to compile or run this kernel. Falls back
 *     to gemm_cuda_wmma on sm_70-75.
 *
 *   Level 7  -- gemm_cuda_wmma_pipelined  (bigger tiles + cp.async double
 *               buffering, fp32 only, sm_70+; NEW, added after the
 *               Reference -- cuBLAS entry below measured this GPU's real
 *               Tensor Core ceiling)
 *     Same wmma:: C++ API as Level 4, but a 128x128 block tile (vs 64x64),
 *     8 warps each owning a 32x64 (8-fragment) output region instead of
 *     one 16x16 fragment, and cp.async-driven double-buffered shared
 *     memory (Ampere+; falls back to a synchronous double buffer on
 *     older Tensor-Core hardware). VERIFIED on RTX 5080 -- see
 *     gemm_kernels.cu's kernel_wmma_pipelined file comment for the full
 *     design rationale and the cuBLAS numbers that motivated it. Requires
 *     M, N exact multiples of 128 and K an exact multiple of 32 (no tail
 *     handling); falls back to the
 *     always-correct gemm_cuda_wmma otherwise.
 *
 *   Reference -- gemm_cuda_cublas / gemm_cuda_cublas_tf32 / gemm_cuda_cublas_fp16
 *     Not part of the ladder above -- vendor-tuned cuBLAS, used as a
 *     realistic achievable-peak ceiling for the hand-written kernels
 *     (this is what motivated writing Level 7 above). gemm_cuda_cublas is
 *     plain SGEMM/DGEMM; gemm_cuda_cublas_tf32/_fp16 (float only) use
 *     TF32/dense-FP16 Tensor Core compute via cublasGemmEx. Raw-device-
 *     pointer, compute-only variants of all three exist for peak-
 *     throughput measurement -- see the block below.
 *
 * Runtime guards:
 *   cuda_device_count()     -- returns 0 on CPU-only builds.
 *   cuda_has_tensor_cores() -- true if any device is sm_70+ (Volta+).
 *   cuda_has_ampere()       -- true if any device is sm_80+ (Ampere+).
 */
#include "hpc/matrix.hpp"
#include <cstddef>
namespace hpc::gemm {
// ---------------------------------------------------------------------------
// Runtime capability queries
// ---------------------------------------------------------------------------
/** Returns the number of CUDA-capable devices (0 on CPU-only builds). */
int  cuda_device_count()     noexcept;
/** True if any device has compute capability >= 7.0 (Volta+, Tensor Cores). */
bool cuda_has_tensor_cores() noexcept;
/** True if any device has compute capability >= 8.0 (Ampere+, cp.async). */
bool cuda_has_ampere()       noexcept;
// ---------------------------------------------------------------------------
// Level 0 -- Naive: one thread per C(i,j), global memory only.
// ---------------------------------------------------------------------------
template <typename T>
void gemm_cuda_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
// ---------------------------------------------------------------------------
// Level 0b -- Reordered: CPU-symmetry baseline, same as naive on GPU.
// ---------------------------------------------------------------------------
template <typename T>
void gemm_cuda_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
// ---------------------------------------------------------------------------
// Level 1 -- Blocked: TILE=16 shared-memory tiling, +1 column padding.
// ---------------------------------------------------------------------------
template <typename T>
void gemm_cuda_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
// ---------------------------------------------------------------------------
// Level 2 -- Register tile: 128x128 thread block, each thread owns 8x8 C tile.
// ---------------------------------------------------------------------------
template <typename T>
void gemm_cuda_reg_tile(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
// ---------------------------------------------------------------------------
// Level 3 -- Double-buffered register tile.
//   On Ampere+ (sm_80+): uses cp.async for asynchronous global->shared DMA.
//   On older GPUs: synchronous loads with __syncthreads barriers.
// ---------------------------------------------------------------------------
template <typename T>
void gemm_cuda_double_buf(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
// ---------------------------------------------------------------------------
// Level 4 -- Tensor Cores via WMMA.
//   fp32 input -> fp16 MMA -> fp32 accumulate.
//   Requires sm_70+ (Volta+). Falls back to double_buf on older hardware.
//   float only: WMMA does not support double precision in this configuration.
// ---------------------------------------------------------------------------
void gemm_cuda_wmma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
// ---------------------------------------------------------------------------
// Level 5 -- Vectorized loads (float4/double2) + shared-memory XOR swizzle.
//   Falls back to gemm_cuda_reg_tile when K or N isn't a multiple of the
//   vector width (4 for float, 2 for double).
// ---------------------------------------------------------------------------
template <typename T>
void gemm_cuda_vectorized(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
// ---------------------------------------------------------------------------
// Level 6 -- Raw Tensor Core MMA via mma.sync + ldmatrix (separate from WMMA).
//   UNVERIFIED -- no CUDA hardware/toolkit was available to compile or run
//   this. Requires sm_80+; falls back to gemm_cuda_wmma on sm_70-75.
//   float only, like WMMA.
// ---------------------------------------------------------------------------
void gemm_cuda_mma_ldmatrix(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
// ---------------------------------------------------------------------------
// Level 7 -- Pipelined WMMA (bigger tiles + cp.async double buffering).
//   NEW kernel -- see gemm_kernels.cu's kernel_wmma_pipelined file
//   comment for the full design rationale (motivated by the cuBLAS
//   reference below measuring ~118 TFLOP/s dense-FP16 achievable on this
//   GPU vs Level 4/6's ~5 TFLOP/s). VERIFIED on RTX 5080. Requires sm_70+
//   (Tensor Cores) and M/N exact multiples of 128, K an exact multiple of
//   32 (no tail handling); falls back to gemm_cuda_wmma otherwise.
//   float only.
// ---------------------------------------------------------------------------
void gemm_cuda_wmma_pipelined(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
// Raw-device-pointer, compute-only entry point (void* fp16 buffers -- same
// toolkit-independence reasoning as the cuBLAS raw-device-pointer
// functions below). Caller must guarantee M/N multiples of 128, K a
// multiple of 32; no fallback at this layer (see gemm_kernels.cu).
void gemm_cuda_wmma_pipelined_device(const void* dA16, const void* dB16, float* dC,
                                      int M, int K, int N);
// ---------------------------------------------------------------------------
// Reference -- cuBLAS (vendor-tuned upper bound, not part of the ladder
// above). Added to measure the realistic achievable peak on this GPU
// before attempting a larger rewrite of the hand-written Tensor Core
// kernels (Levels 4/6, which measured only ~5 TFLOP/s on RTX 5080 --
// see gemm_kernels.cu's "Reference -- cuBLAS" section for the full story).
//
//   gemm_cuda_cublas<T>   -- plain SGEMM/DGEMM. Ceiling for the FMA-based
//                            kernels (naive/blocked/reg_tile/double_buf/
//                            vectorized).
//   gemm_cuda_cublas_tf32 -- fp32 in/out, TF32 Tensor Core compute
//                            (10-bit mantissa). Ceiling for the Tensor
//                            Core kernels (wmma/mma_ldmatrix). float only.
// ---------------------------------------------------------------------------
template <typename T>
void gemm_cuda_cublas(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
void gemm_cuda_cublas_tf32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
// ---------------------------------------------------------------------------
// Reference -- cuBLAS, raw-device-pointer entry points. Take pre-allocated,
// already host->device-copied buffers and do nothing but issue the GEMM
// call -- for measuring peak compute throughput unconfounded by per-call
// cudaMalloc/PCIe-transfer overhead (which dominates at large N and badly
// understates achievable throughput in the Matrix<T>-based wrappers
// above). Same math as gemm_cuda_cublas<float>/gemm_cuda_cublas_tf32; see
// gemm_kernels.cu for the full rationale.
// ---------------------------------------------------------------------------
void gemm_cuda_cublas_device_f32(const float* dA, const float* dB, float* dC,
                                  int M, int K, int N);
void gemm_cuda_cublas_tf32_device(const float* dA, const float* dB, float* dC,
                                   int M, int K, int N);
// ---------------------------------------------------------------------------
// Reference -- cuBLAS, dense FP16 (fp16-in/fp32-accumulate) via
// cublasGemmEx. TF32 compute-only tops out around 59 TFLOP/s on RTX
// 5080 -- still under the "100-200 TFLOP/s" range dense FP16 Tensor
// Cores should reach (roughly 2x TF32, since TF32 elements occupy twice
// the bits FP16 does through the same tensor pipe). gemm_cuda_cublas_fp16
// is the Matrix<float>-based wrapper (converts to fp16 internally, for
// correctness testing); gemm_cuda_convert_f32_to_f16_device +
// gemm_cuda_cublas_fp16_device are the raw-device-pointer, compute-only
// pair (convert once outside the timed region, then call the GEMM
// repeatedly), matching the TF32/plain cuBLAS pattern above.
//
// The fp16 buffers below are typed `void*`, not `__half*`: this header
// (via gemm_kernels_stub.cpp and every host .cpp that includes it, e.g.
// bench_gemm_cuda.cpp/test_gemm_cuda.cpp) must still compile on a
// genuinely CPU-only machine with no CUDA toolkit at all, where
// <cuda_fp16.h> would not be found -- see gemm_kernels.cu for the same
// reasoning. Internally these are real __half buffers of `count`/`M*K`
// elements (2 bytes each); gemm_cuda_convert_f32_to_f16_device is the
// only thing that needs to write into one.
// ---------------------------------------------------------------------------
void gemm_cuda_cublas_fp16(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
void gemm_cuda_convert_f32_to_f16_device(const float* src, void* dst, int count);
void gemm_cuda_cublas_fp16_device(const void* dA16, const void* dB16, float* dC,
                                   int M, int K, int N);
// ---------------------------------------------------------------------------
// Reference -- generic device-memory helpers (void*/size_t only, same
// toolkit-independence reasoning as above) so bench_gemm_cuda.cpp's
// compute-only benchmarks can pre-stage device buffers without including
// <cuda_runtime.h> itself.
// ---------------------------------------------------------------------------
void* gemm_cuda_malloc(std::size_t bytes);
void  gemm_cuda_free(void* ptr);
void  gemm_cuda_memcpy_h2d(void* dst, const void* src, std::size_t bytes);
void  gemm_cuda_device_synchronize();
// ---------------------------------------------------------------------------
// Explicit instantiation declarations (definitions in gemm_kernels.cu)
// ---------------------------------------------------------------------------
extern template void gemm_cuda_naive<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
extern template void gemm_cuda_naive<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
extern template void gemm_cuda_reordered<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
extern template void gemm_cuda_reordered<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
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
