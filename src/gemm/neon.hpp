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
 *                       a whole k-tile: 4×16 f32 (4 rows × 4 Q) or 8×4 f64
 *                       (8 rows × 2 Q). The fastest of the three.
 *
 * AArch64 has 32 × 128-bit Q registers (4 f32 / 2 f64). Both micro-kernels
 * keep 16 accumulators: f32 with 4 broadcasts of A and 4 B vectors live (24
 * of 32), f64 with 2 B vectors and a broadcast (19 of 32).
 *
 * Apple M4 P-core (measured with fma_peak at ~4.05 GHz, docs/experiments/
 * cpu-simd): up to 4 FMAs per cycle, ~3-cycle latency.
 *   f32: 4 pipes × 4 lanes × 2 FLOP = 32 FLOP/cycle → ~130 GFLOP/s
 *   f64: 4 pipes × 2 lanes × 2 FLOP = 16 FLOP/cycle →  ~65 GFLOP/s
 * gemm_neon_blocked reaches ~95 / ~40 GFLOP/s (≈73% / 61%). NEON is half the
 * width of AVX2, but twice the FMA pipes give the same FLOP/cycle.
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
// For Kernel 3: tile sizes (also used by gemm_neon_blocked_prefetch)
// ============================================================================

inline constexpr std::size_t kNeonTileM = 64;   // outer i-tile
inline constexpr std::size_t kNeonTileK = 256;  // outer k-tile
inline constexpr std::size_t kNeonTileN = 256;  // outer j-tile

// f32 micro-kernel: 4 rows × 4 Q-vectors per row = 4×16 j-elements
inline constexpr std::size_t kNeonF32RegRows = 4;
inline constexpr std::size_t kNeonF32RegCols = 4;  // 4 Q-vectors → 16 f32

// f64 micro-kernel: 8 rows × 2 Q-vectors per row = 8×4 j-elements
inline constexpr std::size_t kNeonF64RegRows = 8;
inline constexpr std::size_t kNeonF64RegCols = 2;  // 2 Q-vectors → 4 f64

// ============================================================================
// For Kernel 3: the register-tile micro-kernel
// ============================================================================

#if HPC_HAS_NEON

/// Q-register (128-bit) operations per element type, for neon_micro.
template <typename T> struct Q;
template <> struct Q<float> {
    using V = float32x4_t;
    static V load(const float* p) { return vld1q_f32(p); }
    static void store(float* p, V v) { vst1q_f32(p, v); }
    static V splat(float x) { return vdupq_n_f32(x); }
    static V fma(V acc, V x, V y) { return vfmaq_f32(acc, x, y); }
};
template <> struct Q<double> {
    using V = float64x2_t;
    static V load(const double* p) { return vld1q_f64(p); }
    static void store(double* p, V v) { vst1q_f64(p, v); }
    static V splat(double x) { return vdupq_n_f64(x); }
    static V fma(V acc, V x, V y) { return vfmaq_f64(acc, x, y); }
};

/**
 * @brief NEON micro-kernel: C[0..R)[0..V·W) += A[0..R)[0..k_len) × B[0..k_len)[0..V·W)
 *
 * W = 4 f32 / 2 f64 per Q register; R rows × V Q-vectors of C stay in
 * R·V accumulators for the whole k loop. Per k step: V loads of B row k,
 * R broadcasts of A(i+r, k), R·V FMAs. The library uses two shapes, both
 * with 16 accumulators:
 *
 *   f32  R = 4, V = 4   a 4 × 16 tile
 *   f64  R = 8, V = 2   an 8 × 4 tile
 *
 * The P-core issues up to 4 FMAs per cycle with a ~3.3-cycle latency, so
 * it takes 12–16 independent accumulators to keep the pipes busy
 * (fma_peak). The f64 kernel had a 4 × 4 tile with 8 accumulators before,
 * and ran at half this one's rate on L1-resident data (simd_microkernel,
 * docs/experiments/cpu-simd).
 *
 * The source broadcasts each A(i+r, k) with vdup and uses plain vfmaq;
 * Clang 17 (-O3 -mcpu=apple-m4) turns each broadcast into one ld1r
 * (load and replicate), so a k step of the f32 kernel is 2 ldp of B,
 * 4 ld1r of A and 16 fmla.
 *
 * @param a  Pointer to A(i, k_blk) — row-stride lda
 * @param b  Pointer to B(k_blk, j) — row-stride ldb
 * @param c  Pointer to C(i, j)     — row-stride ldc
 */
template <typename T, std::size_t R, std::size_t V>
inline void neon_micro(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                       std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t k_len) noexcept {
    using N                 = Q<T>;
    using Vec               = typename N::V;
    constexpr std::size_t W = 16 / sizeof(T);

    Vec acc[R][V];
#pragma GCC unroll 8
    for (std::size_t r = 0; r < R; ++r)
#pragma GCC unroll 4
        for (std::size_t v = 0; v < V; ++v)
            acc[r][v] = N::load(c + r * ldc + v * W);
    for (std::size_t k = 0; k < k_len; ++k) {
        Vec bv[V];
#pragma GCC unroll 4
        for (std::size_t v = 0; v < V; ++v)
            bv[v] = N::load(b + k * ldb + v * W);
#pragma GCC unroll 8
        for (std::size_t r = 0; r < R; ++r) {
            const Vec ar = N::splat(a[r * lda + k]);
#pragma GCC unroll 4
            for (std::size_t v = 0; v < V; ++v)
                acc[r][v] = N::fma(acc[r][v], ar, bv[v]);
        }
    }
#pragma GCC unroll 8
    for (std::size_t r = 0; r < R; ++r)
#pragma GCC unroll 4
        for (std::size_t v = 0; v < V; ++v)
            N::store(c + r * ldc + v * W, acc[r][v]);
}

#endif  // HPC_HAS_NEON

// ============================================================================
// Kernel 3: gemm_neon_blocked  —  tiled i → k → j,  NEON register tile
// ============================================================================

/**
 * @brief NEON GEMM with cache-blocking and register-tiled micro-kernel.
 *
 * Counterpart of gemm_avx2_blocked: i-k-j order, 3-level tiling
 * (kNeonTileM × kNeonTileK × kNeonTileN), and a 4×16 f32 / 8×4 f64 C tile
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
                    for (; j + kJStep <= j_end; j += kJStep)
                        neon_micro<T, kRegRows, kRegCols>(a_ptr, b_ptr + (j - j_blk), C.data() + i * ldc + j,
                                                          lda, ldb, ldc, k_len);
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
