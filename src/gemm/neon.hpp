#pragma once

/**
 * @file neon.hpp
 * @brief ARM NEON counterparts of the three AVX2 kernels (avx2.hpp).
 *
 *  gemm_neon_naive      i → j → k, 4 f32 / 2 f64 per FMA on the k-loop. B is
 *                       still a stride-N gather, so it runs at about
 *                       scalar-naive speed.
 *  gemm_neon_reordered  i → k → j: broadcast A(i,k), FMA against stride-1 B
 *                       and C rows. Measured on M4 Max: no faster than scalar
 *                       gemm_reordered, which the compiler auto-vectorises.
 *  gemm_neon_blocked    tiled i → k → j with a C tile held in Q registers for
 *                       a whole k-tile: 4×16 f32 (4 rows × 4 Q) or 4×4 f64
 *                       (4 rows × 2 Q). The fastest of the three.
 *
 * AArch64 has 32 × 128-bit Q registers (4 f32 / 2 f64). The f32 micro-kernel
 * keeps 16 accumulators, 4 broadcasts of A and 4 B vectors live (24 of 32);
 * the f64 one 8 + 4 + 2 (14 of 32).
 *
 * Apple M4 P-core peak (~4.5 GHz, 4 FP/SIMD pipes vs 2 FMA ports on x86):
 *   f32: 4 pipes × 4 lanes × 2 FLOP = 32 FLOP/cycle → ~144 GFLOP/s
 *   f64: 4 pipes × 2 lanes × 2 FLOP = 16 FLOP/cycle →  ~72 GFLOP/s
 * gemm_neon_blocked reaches ~97 / ~36 GFLOP/s (≈67% / 50% of peak). NEON is
 * half the width of AVX2, but twice the FMA pipes give the same FLOP/cycle.
 *
 * HPC_HAS_NEON (hpc/isa.hpp) requires AArch64: the kernels use f64 lanes
 * and vfmaq_f64, which 32-bit ARMv7 NEON lacks. Where it is 0 the kernels
 * are declared `= delete`.
 */

#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"

#if HPC_HAS_NEON
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

#if HPC_HAS_NEON

/**
 * @brief NEON f32 micro-kernel: C[i..i+3][j..j+15] += A[i..i+3][k_blk..k_end) × B[..][j..)
 *
 * The source broadcasts each A(i+r, k) with vdupq_n_f32 and uses plain
 * vfmaq_f32. Clang folds most of those broadcasts into the by-element form
 * of FMLA (fmla v.4s, v.4s, v.s[lane], what vfmaq_laneq_f32 spells
 * explicitly): at -O3 -mcpu=apple-m4, 12 of the 16 FMAs per k step are
 * by-element, with no separate dup instruction.
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

        // Broadcast A(i+r, k) for the 4 rows (see the note above on how the
        // compiler folds these into by-element FMAs).
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

#endif  // HPC_HAS_NEON

// ============================================================================
// Kernel 1: gemm_neon_naive  —  i → j → k,  NEON on the k-loop
// ============================================================================

/**
 * @brief NEON GEMM with naive i-j-k loop order.
 *
 * Counterpart of gemm_avx2_naive: the k-loop processes 4 f32 / 2 f64 per FMA,
 * but column j of B is gathered with stride ldb, so each k step can miss
 * and it runs at about scalar-naive speed.
 */
#if !HPC_HAS_NEON
template <typename T>
void gemm_neon_naive(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_NEON not available on this target
#else
template <typename T>
void gemm_neon_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
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
}
#endif  // HPC_HAS_NEON

// ============================================================================
// Kernel 2: gemm_neon_reordered  —  i → k → j,  NEON on the j-loop
// ============================================================================

/**
 * @brief NEON GEMM with cache-friendly i-k-j loop order.
 *
 * Counterpart of gemm_avx2_reordered: vdupq_n broadcasts A(i,k), and the
 * j-loop processes 4 f32 / 2 f64 per vfmaq against stride-1 B and C rows.
 *
 * Measured on M4 Max: no faster than scalar gemm_reordered — the compiler
 * auto-vectorises that loop too (f32 N=256: 29.6 vs 32.3 GFLOP/s).
 * Degrades at large N when C row i (N×sizeof(T)) exceeds L1, same as
 * AVX2 reordered — no blocking to prevent C eviction.
 */
#if !HPC_HAS_NEON
template <typename T>
void gemm_neon_reordered(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_NEON not available on this target
#else
template <typename T>
void gemm_neon_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
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
}
#endif  // HPC_HAS_NEON

// ============================================================================
// Kernel 3: gemm_neon_blocked  —  tiled i → k → j,  NEON register tile
// ============================================================================

/**
 * @brief NEON GEMM with cache-blocking and register-tiled micro-kernel.
 *
 * Counterpart of gemm_avx2_blocked: i-k-j order, 3-level tiling
 * (kNeonTileM × kNeonTileK × kNeonTileN), and a 4×16 f32 / 4×4 f64 C tile
 * held in Q registers for a whole k-tile, so C is stored once per
 * kNeonTileK steps instead of being reloaded every k-iteration as in
 * gemm_neon_reordered at large N.
 */
#if !HPC_HAS_NEON
template <typename T>
void gemm_neon_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_NEON not available on this target
#else
template <typename T>
void gemm_neon_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
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
}
#endif  // HPC_HAS_NEON

}  // namespace hpc::gemm
