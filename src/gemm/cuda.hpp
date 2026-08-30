#pragma once
/**
 * @file cuda.hpp
 * @brief Host-side C++ interface for CUDA GEMM kernels.
 *
 * Kernel ladder (eight levels of optimization):
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
 *   Level 7  -- gemm_cuda_hopper_wgmma  (warp specialization + TMA, fp32 only, sm_90a)
 *     Producer/consumer warpgroups: one warpgroup issues TMA bulk-tensor
 *     loads (built via the driver API's cuTensorMapEncodeTiled), the other
 *     runs wgmma.mma_async directly against shared memory. BEST-EFFORT,
 *     EXPLICITLY UNVERIFIED AND LIKELY NON-FUNCTIONAL -- written as an
 *     honest sketch of the technique per explicit user request, not as
 *     validated working code (no Hopper hardware, no CUDA toolkit
 *     anywhere in this project -- see gemm_kernels.cu's file header for
 *     the full caveat and per-section confidence levels). Requires M, N,
 *     K to be exact multiples of the wgmma tile (64x64x16) -- no tail
 *     handling. Falls back to gemm_cuda_mma_ldmatrix, then gemm_cuda_wmma,
 *     then gemm_cuda_double_buf, in that order.
 *
 * Runtime guards:
 *   cuda_device_count()     -- returns 0 on CPU-only builds.
 *   cuda_has_tensor_cores() -- true if any device is sm_70+ (Volta+).
 *   cuda_has_ampere()       -- true if any device is sm_80+ (Ampere+).
 *   cuda_has_hopper()       -- true if any device is sm_90+ (Hopper+).
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
/** True if any device has compute capability >= 9.0 (Hopper+, wgmma/TMA). */
bool cuda_has_hopper()       noexcept;
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
// Level 7 -- Hopper warp specialization + TMA (wgmma).
//   BEST-EFFORT, EXPLICITLY UNVERIFIED, LIKELY NON-FUNCTIONAL -- see
//   gemm_kernels.cu's file header for the full caveat. Requires sm_90a and
//   M/N/K exact multiples of 64/64/16 (no tail handling); falls back to
//   gemm_cuda_mma_ldmatrix, then gemm_cuda_wmma, then gemm_cuda_double_buf.
//   float only.
// ---------------------------------------------------------------------------
void gemm_cuda_hopper_wgmma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
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
}  // namespace hpc::gemm
