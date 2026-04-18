#pragma once
/**
 * @file cuda.hpp
 * @brief Host-side C++ interface for CUDA GEMM kernels.
 *
 * Kernel ladder (five levels of optimization):
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
}  // namespace hpc::gemm
