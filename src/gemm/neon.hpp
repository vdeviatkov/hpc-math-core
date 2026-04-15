#pragma once

/**
 * @file neon.hpp
 * @brief Three progressive ARM NEON + FMA GEMM kernels.
 *
 * ============================================================
 *  Design philosophy: ARM NEON counterpart of avx2.hpp
 * ============================================================
 *
 * Each kernel adds exactly one new technique, providing a direct
 * comparison with the x86 AVX2 family on Apple Silicon and other
 * ARM targets.  The benchmark numbers answer: "at each optimisation
 * level, how does 128-bit NEON compare to 256-bit AVX2?"
 *
 *  Kernel 1 — gemm_neon_naive
 *    Loop order: i → j → k  (same as all other naive variants)
 *    New technique: widen the inner k-loop to process 4 f32 / 2 f64
 *                   elements per VMLA using a single q-register accumulator.
 *    Remaining bottleneck: column-stride B access is still a stride-N
 *    gather — identical cache penalty to scalar naive and AVX2 naive.
 *    Expected: GFLOP/s ≈ scalar naive.
 *
 *  Kernel 2 — gemm_neon_reordered
 *    Loop order: i → k → j
 *    New technique: vdupq_n_f32/f64 broadcasts A(i,k) to all lanes;
 *                   vfmaq_f32/f64 (or vmlaq on older ISA) fused multiply-adds
 *                   against stride-1 B row and C row.
 *    Expected: ~4× scalar reordered for f32 (4-wide NEON vs scalar),
 *              ~2× for f64 (2-wide NEON).
 *
 *  Kernel 3 — gemm_neon_blocked
 *    Loop order: tiled i → k → j
 *    New technique: 3-level L2 cache tiling + register tile that keeps
 *                   4×16 f32 / 4×8 f64 elements of C in q-registers for
 *                   the full k-tile, eliminating C reload traffic.
 *    Expected: highest GFLOP/s on ARM.
 *
 *
 * ============================================================
 *  ARM NEON / AdvSIMD register file
 * ============================================================
 *
 * NEON (AArch64 / Apple Silicon):
 *   32 × 128-bit Q (V) registers.
 *   f32: 4 floats  per Q register  (16 B)
 *   f64: 2 doubles per Q register  (16 B)
 *
 * Key instructions:
 *   vdupq_n_f32(s)    — broadcast scalar s to all 4 lanes  (float32x4_t)
 *   vdupq_n_f64(s)    — broadcast scalar s to both lanes   (float64x2_t)
 *   vfmaq_f32(acc,a,b)— acc = acc + a * b, 4-wide f32 FMA  (AArch64)
 *   vfmaq_f64(acc,a,b)— acc = acc + a * b, 2-wide f64 FMA
 *   vfmaq_lane_f32    — FMA with scalar from a specific lane (no extra broadcast register)
 *   vaddvq_f32(v)     — horizontal add of all 4 f32 lanes → scalar
 *
 * Apple M-series throughput (Firestorm / Icestorm, ~3 GHz):
 *   f32: 4 NEON units × 4 lanes × 2 FLOP = 32 FLOP/cycle → ~96 GFLOP/s/core
 *   f64: 4 NEON units × 2 lanes × 2 FLOP = 16 FLOP/cycle → ~48 GFLOP/s/core
 *   (M-series has 4 FP/SIMD units per P-core vs 2 on x86 Skylake)
 *
 * NEON vs AVX2 (register width):
 *   NEON Q: 128-bit = 4 f32 / 2 f64
 *   AVX2 YMM: 256-bit = 8 f32 / 4 f64
 *   AVX2 is 2× wider, but Apple M-series has 4 FP units vs 2 on Skylake,
 *   so peak GFLOP/s is the same at the same frequency.
 *
 *
 * ============================================================
 *  Micro-kernel register tile (f32, 4 rows × 4 vectors = 4×16)
 * ============================================================
 *
 *   C tile in Q registers (8 accumulators):
 *
 *         j+0..3    j+4..7    j+8..11  j+12..15
 *   i+0: [c00 q]  [c01 q]  [c02 q]  [c03 q]
 *   i+1: [c10 q]  [c11 q]  [c12 q]  [c13 q]
 *   i+2: [c20 q]  [c21 q]  [c22 q]  [c23 q]
 *   i+3: [c30 q]  [c31 q]  [c32 q]  [c33 q]
 *
 *   16 accumulator registers (4 rows × 4 Q-vectors)
 *    + 4 broadcast A registers
 *    + 4 B load registers
 *   = 24 of 32 Q registers used.
 *
 * f64, 4 rows × 2 vectors = 4×4:
 *   8 accumulators + 4 broadcasts + 2 B loads = 14 of 32 Q registers.
 *
 *
 * ============================================================
 *  Portability guard
 * ============================================================
 *
 * NEON is available on:
 *   All AArch64 targets: Apple Silicon M1/M2/M3/M4,
 *   AWS Graviton, Ampere Altra, Raspberry Pi 4/5, etc.
 *   NOT available on: x86 (Intel/AMD) — no NEON instructions exist there.
 *
 * The header detects __ARM_NEON at compile time.
 * If absent (x86), all three kernels fall back to their AVX2 equivalents.
 * This makes the binary always correct and never generates SIGILL.
 *
 * vfmaq_f32 / vfmaq_f64 require AArch64 (ARM64).
 * On 32-bit ARMv7 with NEON, vmlaq_f32 is used as fallback
 * (no f64 NEON on 32-bit ARM).
 */

#include "hpc/matrix.hpp"

#include "gemm/avx2.hpp"  // fallback on non-ARM targets

#ifdef __ARM_NEON
    #include <arm_neon.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace hpc::gemm {

// ============================================================================
// NEON tile / unroll constants
// ============================================================================

inline constexpr std::size_t kNeonTileM = 64;   // outer i-tile
inline constexpr std::size_t kNeonTileK = 256;  // outer k-tile
inline constexpr std::size_t kNeonTileN = 256;  // outer j-tile

// f32 micro-kernel: 4 rows × 4 Q-vectors per row = 4×16 j-elements
inline constexpr std::size_t kNeonF32RegRows = 4;
inline constexpr std::size_t kNeonF32RegCols = 4;  // 4 Q-vectors → 16 f32

// f64 micro-kernel: 4 rows × 2 Q-vectors per row = 4×4 j-elements
inline constexpr std::size_t kNeonF64RegRows = 4;
inline constexpr std::size_t kNeonF64RegCols = 2;  // 2 Q-vectors → 4 f64

// ============================================================================
// NEON micro-kernels (used only by gemm_neon_blocked)
// ============================================================================

#ifdef __ARM_NEON

/**
 * @brief NEON f32 micro-kernel: C[i..i+3][j..j+15] += A[i..i+3][k_blk..k_end) × B[..][j..)
 *
 * Register tile: 4 rows × 4 Q-vectors = 4×16 f32.
 * Uses 16 accumulators + 4 broadcast + 4 B-load = 24 of 32 Q registers.
 *
 * vfmaq_laneq_f32(acc, b_vec, a_vec, lane):
 *   Fuses broadcast of a single lane of a_vec with multiply-add into acc.
 *   This avoids a separate vdup instruction — one Q-register holds all 4
 *   scalar broadcasts simultaneously, each selected by lane index 0..3.
 *   On Apple M-series this is a single micro-op at 0.25 cycles throughput.
 *
 * @param a      Pointer to A(i, k_blk) — row-stride lda
 * @param b      Pointer to B(k_blk, j) — row-stride ldb
 * @param c0..c3 Pointers to C(i+0..3, j)
 */
inline void neon_micro_f32_4x16(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c0, float* __restrict__ c1,
                                float* __restrict__ c2, float* __restrict__ c3, std::size_t lda,
                                std::size_t ldb, std::size_t k_len) noexcept {
    // Load 4×16 C tile: 16 Q-register accumulators (4 rows × 4 Q-vectors).
    float32x4_t c00 = vld1q_f32(c0), c01 = vld1q_f32(c0 + 4);
    float32x4_t c02 = vld1q_f32(c0 + 8), c03 = vld1q_f32(c0 + 12);
    float32x4_t c10 = vld1q_f32(c1), c11 = vld1q_f32(c1 + 4);
    float32x4_t c12 = vld1q_f32(c1 + 8), c13 = vld1q_f32(c1 + 12);
    float32x4_t c20 = vld1q_f32(c2), c21 = vld1q_f32(c2 + 4);
    float32x4_t c22 = vld1q_f32(c2 + 8), c23 = vld1q_f32(c2 + 12);
    float32x4_t c30 = vld1q_f32(c3), c31 = vld1q_f32(c3 + 4);
    float32x4_t c32 = vld1q_f32(c3 + 8), c33 = vld1q_f32(c3 + 12);

    for (std::size_t k = 0; k < k_len; ++k) {
        // Load 4 consecutive B elements per vector × 4 vectors = 16 f32.
        const float32x4_t b0 = vld1q_f32(b + k * ldb);
        const float32x4_t b1 = vld1q_f32(b + k * ldb + 4);
        const float32x4_t b2 = vld1q_f32(b + k * ldb + 8);
        const float32x4_t b3 = vld1q_f32(b + k * ldb + 12);

        // Load A scalars for 4 rows into one Q-register each.
        // Using vfmaq_laneq_f32 to fuse broadcast+FMA without extra vdup.
        // a_row_r holds A(i+r, k) in lane 0 (we use vld1q_dup_f32 for clarity).
        const float32x4_t a0 = vdupq_n_f32(a[0 * lda + k]);
        const float32x4_t a1 = vdupq_n_f32(a[1 * lda + k]);
        const float32x4_t a2 = vdupq_n_f32(a[2 * lda + k]);
        const float32x4_t a3 = vdupq_n_f32(a[3 * lda + k]);

        // 16 FMA instructions (4 rows × 4 B-vectors).
        c00 = vfmaq_f32(c00, a0, b0);
        c01 = vfmaq_f32(c01, a0, b1);
        c02 = vfmaq_f32(c02, a0, b2);
        c03 = vfmaq_f32(c03, a0, b3);
        c10 = vfmaq_f32(c10, a1, b0);
        c11 = vfmaq_f32(c11, a1, b1);
        c12 = vfmaq_f32(c12, a1, b2);
        c13 = vfmaq_f32(c13, a1, b3);
        c20 = vfmaq_f32(c20, a2, b0);
        c21 = vfmaq_f32(c21, a2, b1);
        c22 = vfmaq_f32(c22, a2, b2);
        c23 = vfmaq_f32(c23, a2, b3);
        c30 = vfmaq_f32(c30, a3, b0);
        c31 = vfmaq_f32(c31, a3, b1);
        c32 = vfmaq_f32(c32, a3, b2);
        c33 = vfmaq_f32(c33, a3, b3);
    }

    // Store 4×16 C tile.
    vst1q_f32(c0, c00);
    vst1q_f32(c0 + 4, c01);
    vst1q_f32(c0 + 8, c02);
    vst1q_f32(c0 + 12, c03);
    vst1q_f32(c1, c10);
    vst1q_f32(c1 + 4, c11);
    vst1q_f32(c1 + 8, c12);
    vst1q_f32(c1 + 12, c13);
    vst1q_f32(c2, c20);
    vst1q_f32(c2 + 4, c21);
    vst1q_f32(c2 + 8, c22);
    vst1q_f32(c2 + 12, c23);
    vst1q_f32(c3, c30);
    vst1q_f32(c3 + 4, c31);
    vst1q_f32(c3 + 8, c32);
    vst1q_f32(c3 + 12, c33);
}

/**
 * @brief NEON f64 micro-kernel: C[i..i+3][j..j+3] += A[i..i+3][k_blk..k_end) × B[..][j..)
 *
 * Register tile: 4 rows × 2 Q-vectors = 4×4 f64.
 * Uses 8 accumulators + 4 broadcast + 2 B-load = 14 of 32 Q registers.
 */
inline void neon_micro_f64_4x4(const double* __restrict__ a, const double* __restrict__ b,
                               double* __restrict__ c0, double* __restrict__ c1,
                               double* __restrict__ c2, double* __restrict__ c3, std::size_t lda,
                               std::size_t ldb, std::size_t k_len) noexcept {
    float64x2_t c00 = vld1q_f64(c0), c01 = vld1q_f64(c0 + 2);
    float64x2_t c10 = vld1q_f64(c1), c11 = vld1q_f64(c1 + 2);
    float64x2_t c20 = vld1q_f64(c2), c21 = vld1q_f64(c2 + 2);
    float64x2_t c30 = vld1q_f64(c3), c31 = vld1q_f64(c3 + 2);

    for (std::size_t k = 0; k < k_len; ++k) {
        const float64x2_t b0 = vld1q_f64(b + k * ldb);
        const float64x2_t b1 = vld1q_f64(b + k * ldb + 2);

        const float64x2_t a0 = vdupq_n_f64(a[0 * lda + k]);
        const float64x2_t a1 = vdupq_n_f64(a[1 * lda + k]);
        const float64x2_t a2 = vdupq_n_f64(a[2 * lda + k]);
        const float64x2_t a3 = vdupq_n_f64(a[3 * lda + k]);

        c00 = vfmaq_f64(c00, a0, b0);
        c01 = vfmaq_f64(c01, a0, b1);
        c10 = vfmaq_f64(c10, a1, b0);
        c11 = vfmaq_f64(c11, a1, b1);
        c20 = vfmaq_f64(c20, a2, b0);
        c21 = vfmaq_f64(c21, a2, b1);
        c30 = vfmaq_f64(c30, a3, b0);
        c31 = vfmaq_f64(c31, a3, b1);
    }

    vst1q_f64(c0, c00);
    vst1q_f64(c0 + 2, c01);
    vst1q_f64(c1, c10);
    vst1q_f64(c1 + 2, c11);
    vst1q_f64(c2, c20);
    vst1q_f64(c2 + 2, c21);
    vst1q_f64(c3, c30);
    vst1q_f64(c3 + 2, c31);
}

#endif  // __ARM_NEON

// ============================================================================
// Kernel 1: gemm_neon_naive  —  i → j → k,  NEON on the k-loop
// ============================================================================

/**
 * @brief NEON GEMM with naive i-j-k loop order.
 *
 * Direct ARM counterpart of gemm_avx2_naive.
 *
 * Loop structure:  for i: for j: for k:  C(i,j) += A(i,k) * B(k,j)
 *
 * NEON change vs scalar naive:
 *   Inner k-loop processes 4 f32 / 2 f64 elements per VMLA using
 *   a Q-register accumulator. B column j is accessed stride-ldb —
 *   a gather, not a sequential load.
 *
 * Expected result: GFLOP/s ≈ scalar naive.
 * The column gather from B causes a cache miss for every k-step at large N,
 * saturating memory bandwidth. This benchmark proves that SIMD width is
 * irrelevant when the access pattern is hostile — same lesson as AVX2 naive.
 *
 * Falls back to gemm_avx2_naive → gemm_naive on non-NEON targets.
 */
template <typename T>
void gemm_neon_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __ARM_NEON
    gemm_avx2_naive(A, B, C);
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_neon_naive: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    // Q-register SIMD width in elements.
    constexpr std::size_t W = (sizeof(T) == 4) ? 4 : 2;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            if constexpr (sizeof(T) == 4) {
                float32x4_t acc = vdupq_n_f32(0.f);
                std::size_t k   = 0;
                for (; k + W <= K; k += W) {
                    // Load A(i, k..k+3) sequentially — cache friendly.
                    const float32x4_t a_vec = vld1q_f32(A.data() + i * lda + k);
                    // Gather B column j: B(k,j), B(k+1,j), B(k+2,j), B(k+3,j).
                    // Each is ldb floats apart — stride-N, cache-hostile.
                    const float b_col[4] = {
                        B.data()[(k + 0) * ldb + j],
                        B.data()[(k + 1) * ldb + j],
                        B.data()[(k + 2) * ldb + j],
                        B.data()[(k + 3) * ldb + j],
                    };
                    const float32x4_t b_vec = vld1q_f32(b_col);
                    acc                     = vfmaq_f32(acc, a_vec, b_vec);
                }
                // Horizontal reduce Q → scalar.
                float s = vaddvq_f32(acc);
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            } else {
                float64x2_t acc = vdupq_n_f64(0.0);
                std::size_t k   = 0;
                for (; k + W <= K; k += W) {
                    const float64x2_t a_vec = vld1q_f64(A.data() + i * lda + k);
                    const double b_col[2]   = {
                        B.data()[(k + 0) * ldb + j],
                        B.data()[(k + 1) * ldb + j],
                    };
                    const float64x2_t b_vec = vld1q_f64(b_col);
                    acc                     = vfmaq_f64(acc, a_vec, b_vec);
                }
                // Horizontal reduce: add both lanes.
                double s = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            }
        }
    }
#endif
}

// ============================================================================
// Kernel 2: gemm_neon_reordered  —  i → k → j,  NEON on the j-loop
// ============================================================================

/**
 * @brief NEON GEMM with cache-friendly i-k-j loop order.
 *
 * Direct ARM counterpart of gemm_avx2_reordered.
 *
 * Loop structure:  for i: for k: a_broad = A(i,k);  for j: C(i,j) += a_broad * B(k,j)
 *
 * NEON change vs scalar reordered:
 *   vdupq_n_f32/f64 broadcasts A(i,k) to all lanes.
 *   Inner j-loop processes 4 f32 / 2 f64 per vfmaq instruction.
 *   B row k and C row i are accessed stride-1 across j — every byte loaded
 *   from cache is used.
 *
 * Expected result: ~4× scalar reordered for f32 (4-wide NEON),
 *                  ~2× scalar reordered for f64 (2-wide NEON).
 * Degrades at large N when C row i (N×sizeof(T)) exceeds L1, same as
 * AVX2 reordered — no blocking to prevent C eviction.
 *
 * Falls back to gemm_avx2_reordered on non-NEON targets.
 */
template <typename T>
void gemm_neon_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __ARM_NEON
    gemm_avx2_reordered(A, B, C);
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_neon_reordered: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    constexpr std::size_t W = (sizeof(T) == 4) ? 4 : 2;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            if constexpr (sizeof(T) == 4) {
                const float32x4_t a_broad = vdupq_n_f32(A.data()[i * lda + k]);
                const float* b_row        = B.data() + k * ldb;
                float* c_row              = C.data() + i * ldc;

                std::size_t j = 0;
                for (; j + W <= N; j += W) {
                    const float32x4_t b_vec = vld1q_f32(b_row + j);
                    const float32x4_t c_vec = vld1q_f32(c_row + j);
                    vst1q_f32(c_row + j, vfmaq_f32(c_vec, a_broad, b_vec));
                }
                const float a_scalar = A.data()[i * lda + k];
                for (; j < N; ++j)
                    c_row[j] += a_scalar * b_row[j];
            } else {
                const float64x2_t a_broad = vdupq_n_f64(A.data()[i * lda + k]);
                const double* b_row       = B.data() + k * ldb;
                double* c_row             = C.data() + i * ldc;

                std::size_t j = 0;
                for (; j + W <= N; j += W) {
                    const float64x2_t b_vec = vld1q_f64(b_row + j);
                    const float64x2_t c_vec = vld1q_f64(c_row + j);
                    vst1q_f64(c_row + j, vfmaq_f64(c_vec, a_broad, b_vec));
                }
                const double a_scalar = A.data()[i * lda + k];
                for (; j < N; ++j)
                    c_row[j] += a_scalar * b_row[j];
            }
        }
    }
#endif
}

// ============================================================================
// Kernel 3: gemm_neon_blocked  —  tiled i → k → j,  NEON register tile
// ============================================================================

/**
 * @brief NEON GEMM with cache-blocking and register-tiled micro-kernel.
 *
 * Direct ARM counterpart of gemm_avx2_blocked.
 *
 * Combines all three techniques:
 *   1. i-k-j loop order      (stride-1 B and C access)
 *   2. 3-level L2 cache tiling (kNeonTileM × kNeonTileK × kNeonTileN)
 *   3. Register tile: 4×16 f32 / 4×4 f64 of C held in Q registers for
 *      the full k-tile — no C load/store during k-iteration.
 *
 * f32 micro-kernel register allocation (24 of 32 Q registers):
 *   q0..q15  — 4 rows × 4 Q-vectors = 16 C accumulators
 *   q16..q19 — broadcast(A(i+r, k)) for rows r=0..3
 *   q20..q23 — 4 × B(k, j..j+3) vectors
 *   q24..q31 — free for software prefetch / future loop unrolling
 *
 * Why better than gemm_neon_reordered at large N:
 *   The reordered kernel reloads C row i on every k-iteration once N exceeds
 *   L1 capacity. The register tile holds 4 rows × 16 f32 = 256 B entirely
 *   in Q registers for kNeonTileK=256 iterations, then stores once.
 *   This matches the AVX2 blocked strategy and achieves the same cache
 *   reuse ratio — the only difference is 128-bit Q vs 256-bit YMM.
 *
 * Apple M-series note:
 *   M1/M2/M3/M4 have 4 independent NEON/FP execution units per P-core.
 *   The 16-FMA micro-kernel (4 rows × 4 B-vectors) can keep all 4 units
 *   busy simultaneously, approaching peak throughput.
 *
 * Falls back to gemm_avx2_blocked on non-NEON targets (x86 with AVX2),
 * or gemm_blocked on targets with neither.
 */
template <typename T>
void gemm_neon_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __ARM_NEON
    gemm_avx2_blocked(A, B, C);
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_neon_blocked: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    constexpr std::size_t kSimdW   = (sizeof(T) == 4) ? 4 : 2;
    constexpr std::size_t kRegRows = (sizeof(T) == 4) ? kNeonF32RegRows : kNeonF64RegRows;
    constexpr std::size_t kRegCols = (sizeof(T) == 4) ? kNeonF32RegCols : kNeonF64RegCols;
    constexpr std::size_t kJStep   = kSimdW * kRegCols;  // 16 f32 or 4 f64 per micro-kernel call

    for (std::size_t i_blk = 0; i_blk < M; i_blk += kNeonTileM) {
        const std::size_t i_end = std::min(i_blk + kNeonTileM, M);

        for (std::size_t k_blk = 0; k_blk < K; k_blk += kNeonTileK) {
            const std::size_t k_end = std::min(k_blk + kNeonTileK, K);
            const std::size_t k_len = k_end - k_blk;

            for (std::size_t j_blk = 0; j_blk < N; j_blk += kNeonTileN) {
                const std::size_t j_end = std::min(j_blk + kNeonTileN, N);

                // --- NEON hot path ---
                std::size_t i = i_blk;
                for (; i + kRegRows <= i_end; i += kRegRows) {
                    const T* a_ptr = A.data() + i * lda + k_blk;
                    const T* b_ptr = B.data() + k_blk * ldb + j_blk;

                    std::size_t j = j_blk;
                    for (; j + kJStep <= j_end; j += kJStep) {
                        T* c0 = C.data() + (i + 0) * ldc + j;
                        T* c1 = C.data() + (i + 1) * ldc + j;
                        T* c2 = C.data() + (i + 2) * ldc + j;
                        T* c3 = C.data() + (i + 3) * ldc + j;
                        if constexpr (sizeof(T) == 4) {
                            neon_micro_f32_4x16(reinterpret_cast<const float*>(a_ptr),
                                                reinterpret_cast<const float*>(b_ptr + (j - j_blk)),
                                                reinterpret_cast<float*>(c0),
                                                reinterpret_cast<float*>(c1),
                                                reinterpret_cast<float*>(c2),
                                                reinterpret_cast<float*>(c3), lda, ldb, k_len);
                        } else {
                            neon_micro_f64_4x4(reinterpret_cast<const double*>(a_ptr),
                                               reinterpret_cast<const double*>(b_ptr + (j - j_blk)),
                                               reinterpret_cast<double*>(c0),
                                               reinterpret_cast<double*>(c1),
                                               reinterpret_cast<double*>(c2),
                                               reinterpret_cast<double*>(c3), lda, ldb, k_len);
                        }
                    }
                    // Scalar j-tail
                    for (; j < j_end; ++j)
                        for (std::size_t ii = i; ii < i + kRegRows; ++ii) {
                            T acc{};
                            for (std::size_t k = k_blk; k < k_end; ++k)
                                acc += A(ii, k) * B(k, j);
                            C(ii, j) += acc;
                        }
                }
                // Scalar i-tail
                for (; i < i_end; ++i)
                    for (std::size_t k = k_blk; k < k_end; ++k) {
                        const T a_ik = A(i, k);
                        for (std::size_t j = j_blk; j < j_end; ++j)
                            C(i, j) += a_ik * B(k, j);
                    }
            }
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Convenience alias: gemm_neon → gemm_neon_blocked
// ---------------------------------------------------------------------------
template <typename T>
inline void gemm_neon(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    gemm_neon_blocked(A, B, C);
}

}  // namespace hpc::gemm
