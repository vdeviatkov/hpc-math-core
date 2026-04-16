#pragma once

/**
 * @file cuda.hpp
 * @brief Host-side C++ interface for CUDA GEMM kernels.
 *
 * ============================================================
 *  Overview
 * ============================================================
 *
 * This header is always included by both CUDA-enabled and CUDA-disabled
 * builds.  The actual __global__ kernels live in src/cuda/gemm_kernels.cu
 * and are compiled only when CUDA is available (CMake LANGUAGES CUDA).
 *
 * Three kernel strategies, mirroring the CPU progression:
 *
 *   gemm_cuda_naive<T>(A, B, C)
 *   ─────────────────────────────────────────────────────────
 *   One CUDA thread computes one C(i,j) element.
 *   Inner k-loop runs fully in registers.
 *   No shared memory, no tiling.  Extremely memory-bandwidth bound
 *   because every thread re-reads entire rows/columns of A and B from
 *   global memory (effective bandwidth: O(N³) global loads for an N×N
 *   GEMM instead of the optimal O(N²) with full reuse).
 *
 *   Launch grid: ceil(N/16) × ceil(N/16) 2-D blocks of 16×16 threads.
 *
 *
 *   gemm_cuda_reordered<T>(A, B, C)
 *   ─────────────────────────────────────────────────────────
 *   Same one-thread-per-element mapping but the innermost memory access
 *   pattern is restructured so that the warp reads contiguous addresses:
 *     - C(i,j) thread reads A row i (coalesced if same i in warp) and
 *       B column j (strided — not coalesced, but better L2 reuse).
 *   This is essentially the same as gemm_naive on GPU; labelled "reordered"
 *   to maintain naming symmetry with the CPU kernels.
 *   Real improvement vs naive on GPU is minor (L2 cache absorbs repeats
 *   within a warp), but it establishes the comparison baseline.
 *
 *
 *   gemm_cuda_blocked<T>(A, B, C)
 *   ─────────────────────────────────────────────────────────
 *   Classic tiled GEMM using shared memory.  A TILE×TILE block of threads
 *   cooperatively loads TILE×TILE sub-tiles of A and B into __shared__
 *   memory before computing partial dot-products.
 *
 *   Memory traffic analysis (TILE = 16):
 *     Without shared memory: each C(i,j) loads 2N elements → 2N³ loads.
 *     With TILE-sized shared tile: each element loaded N/TILE times from
 *     global memory → (2N³/TILE) global loads — a factor-TILE reduction.
 *
 *   For TILE=16 and f32: theoretical speedup ≈ 16× vs naive.
 *   Actual speedup is lower (bounded by compute, not just bandwidth).
 *
 *   The __shared__ buffers are declared with padding (+1 column) to avoid
 *   bank conflicts when threads in the same warp access the same column.
 *
 *   Launch grid: same ceil(N/TILE) × ceil(N/TILE) grid;
 *   block dim: TILE × TILE threads.
 *
 *
 * ============================================================
 *  Memory layout
 * ============================================================
 *
 * All three kernels expect ROW-MAJOR layout — consistent with the CPU
 * hpc::Matrix<T> class.  Element (i,j) of an N×N matrix is at
 * base_ptr + i*N + j.
 *
 *
 * ============================================================
 *  Usage from host code
 * ============================================================
 *
 * The three functions declared below are defined in gemm_kernels.cu.
 * They are callable from normal C++ translation units because they are
 * declared `extern "C++"` and the .cu file is compiled by nvcc/clang-cuda
 * which produces standard object files.
 *
 * On non-CUDA builds (HPC_HAVE_CUDA not defined) calling these functions
 * is a link error — the benchmark guards them with `cuda_device_count() > 0`
 * which is itself only defined when CUDA is available.
 *
 *
 * ============================================================
 *  Runtime device detection
 * ============================================================
 *
 * `cuda_device_count()` returns the number of CUDA-capable GPUs at runtime.
 * Returns 0 if CUDA is not present or no GPU is found.
 * The benchmark calls this before registering / running CUDA tests so that
 * the binary compiles and runs on CPU-only machines without any crash.
 */

#include "hpc/matrix.hpp"

#include <cstddef>

namespace hpc::gemm {

// ---------------------------------------------------------------------------
// Runtime device availability query
// ---------------------------------------------------------------------------

/**
 * @brief Returns the number of CUDA-capable devices visible at runtime.
 *
 * Always returns 0 when the library was built without CUDA support.
 * Use this to guard benchmark/test registration:
 *
 *   if (hpc::gemm::cuda_device_count() == 0) {
 *       state.SkipWithMessage("No CUDA device available");
 *       return;
 *   }
 */
int cuda_device_count() noexcept;

// ---------------------------------------------------------------------------
// CUDA GEMM kernel launchers
// ---------------------------------------------------------------------------

/**
 * @brief Naive CUDA GEMM: one thread per C(i,j), no shared memory.
 *
 * @tparam T  float or double.
 * @param A   M×K input matrix (row-major, host pointer).
 * @param B   K×N input matrix (row-major, host pointer).
 * @param C   M×N output matrix (row-major, host pointer).
 *
 * Internally: allocates device memory, copies A/B to device, runs kernel,
 * copies C back, frees device memory.  Synchronises before returning.
 */
template <typename T>
void gemm_cuda_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

/**
 * @brief Reordered CUDA GEMM: same thread mapping as naive, explicit
 *        row-major read order for slightly better warp coalescence.
 *
 * In practice results are very close to naive on modern GPUs because
 * the L2 cache absorbs the repeated accesses within a warp.
 * Provided for apples-to-apples comparison with the CPU kernels.
 */
template <typename T>
void gemm_cuda_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

/**
 * @brief Tiled CUDA GEMM using TILE×TILE shared-memory sub-tiles.
 *
 * Uses compile-time TILE=16.  Each thread block cooperatively loads
 * sub-tiles of A and B into __shared__ memory to maximise L1/shared
 * bandwidth utilisation and minimise global memory traffic.
 *
 * For best results N should be a multiple of TILE; non-multiples are
 * handled by boundary guards in the kernel (no UB, slightly lower perf).
 */
template <typename T>
void gemm_cuda_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

// ---------------------------------------------------------------------------
// Explicit instantiation declarations (definitions in gemm_kernels.cu)
// ---------------------------------------------------------------------------
extern template void gemm_cuda_naive<float>(const Matrix<float>&,
                                             const Matrix<float>&,
                                             Matrix<float>&);
extern template void gemm_cuda_naive<double>(const Matrix<double>&,
                                              const Matrix<double>&,
                                              Matrix<double>&);

extern template void gemm_cuda_reordered<float>(const Matrix<float>&,
                                                 const Matrix<float>&,
                                                 Matrix<float>&);
extern template void gemm_cuda_reordered<double>(const Matrix<double>&,
                                                  const Matrix<double>&,
                                                  Matrix<double>&);

extern template void gemm_cuda_blocked<float>(const Matrix<float>&,
                                               const Matrix<float>&,
                                               Matrix<float>&);
extern template void gemm_cuda_blocked<double>(const Matrix<double>&,
                                                const Matrix<double>&,
                                                Matrix<double>&);

}  // namespace hpc::gemm

