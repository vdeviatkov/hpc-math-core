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
 *                       a whole k-tile: 6 rows × 4 Q by default (6×16 f32,
 *                       6×8 f64). The fastest of the three.
 *
 * AArch64 has 32 × 128-bit Q registers (4 f32 / 2 f64). The 6 × 4 tile keeps
 * 24 accumulators, 4 B vectors and one element of A live (29 of 32).
 *
 * Apple M4 P-core (measured with fma_peak, docs/experiments/cpu-simd): up
 * to 4 FMAs per cycle, ~3-cycle latency.
 *   f32: 4 pipes × 4 lanes × 2 FLOP = 32 FLOP/cycle → ~143 GFLOP/s at 4.49 GHz
 *   f64: 4 pipes × 2 lanes × 2 FLOP = 16 FLOP/cycle →  ~72 GFLOP/s
 * gemm_neon_blocked reaches ~121 / ~60 GFLOP/s at N = 256–1024 (≈85% /
 * 84%; docs/kernels/neon.md). NEON is half the width of AVX2, but twice the
 * FMA pipes give the same FLOP/cycle.
 *
 * HPC_HAS_NEON (hpc/isa.hpp) requires AArch64: the kernels use f64 lanes
 * and vfmaq_f64, which 32-bit ARMv7 NEON lacks. Where it is 0 the kernels
 * are declared `= delete`.
 */

#include "gemm/blocked_driver.hpp"
#include "gemm/vector_micro.hpp"
#include "gemm/packed_driver.hpp"
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
// For Kernel 3: cache tiles (also used by gemm_neon_blocked_prefetch)
// ============================================================================

// Cache block, the default template arguments of gemm_neon_blocked: TileM
// rows of A × TileK steps of k × TileN columns of B (B block 256 KiB f32,
// 512 KiB f64). A tall A block means each B block, re-read from memory once
// per i-block, serves 64 row strips. Measured on M4: docs/kernels/blocks.md.
inline constexpr std::size_t kNeonTileM = 384;
inline constexpr std::size_t kNeonTileK = 256;
inline constexpr std::size_t kNeonTileN = 256;

// ============================================================================
// For Kernel 3: the register-tile micro-kernel
// ============================================================================

#if HPC_HAS_NEON

/// Q-register (128-bit) operations per element type, for neon_micro. S is a
/// 64-bit D register holding one element of A in lane 0; fma multiplies a
/// B vector by that lane (FMLA by element).
template <typename T> struct Q;
template <> struct Q<float> {
    using V = float32x4_t;
    using S = float32x2_t;
    static V load(const float* p) { return vld1q_f32(p); }
    static void store(float* p, V v) { vst1q_f32(p, v); }
    static S load_a(const float* p) {
#if defined(__GNUC__)
        // A plain LDR. From intrinsics (vld1_lane, vset_lane) Clang 17 folds
        // the load and the by-element FMA back into LD1R, which neon_micro
        // measured slower for this tile on M4 with Clang 17.
        S x;
        asm("ldr %s0, %1" : "=w"(x) : "m"(*p));
        return x;
#else
        return vld1_dup_f32(p);
#endif
    }
    static V fma(S a, V b, V acc) { return vfmaq_lane_f32(acc, b, a, 0); }
};
template <> struct Q<double> {
    using V = float64x2_t;
    using S = float64x1_t;
    static V load(const double* p) { return vld1q_f64(p); }
    static void store(double* p, V v) { vst1q_f64(p, v); }
    static S load_a(const double* p) { return vld1_f64(p); }  // LDR d
    static V fma(S a, V b, V acc) { return vfmaq_lane_f64(acc, b, a, 0); }
};

/**
 * @brief NEON micro-kernel: C[0..R)[0..V·W) += A[0..R)[0..k_len) × B[0..k_len)[0..V·W)
 *
 * W = 4 f32 / 2 f64 per Q register; R rows × V Q-vectors of C stay in
 * R·V accumulators for the whole k loop. Per k step: V loads of B row k,
 * R scalar loads of A(i+r, k), R·V FMAs. The P-core issues up to 4 FMAs
 * per cycle with a ~3.3-cycle latency, so it takes 12–16 independent
 * accumulators to keep the pipes busy (fma_peak). gemm_neon_blocked uses
 * R = 6, V = 4 (docs/kernels/neon.md).
 *
 * A(i+r, k) is loaded with a plain LDR into lane 0 of a D register and
 * multiplied by element (FMLA v.4s, v.4s, v.s[0]); a k step of the 6 × 4
 * tile is 2 ldp of B, 6 ldr of A and 24 fmla. Broadcasting A with LD1R
 * (load and replicate), which Clang emits for vdupq + vfmaq, held the
 * tiles with many rows back on M4: with LDR the 6 × 4 tile is 23–30%
 * faster in the full multiplication (docs/kernels/neon.md).
 *
 * @param a  Pointer to A(i, k_blk) — row-stride lda
 * @param b  Pointer to B(k_blk, j) — row-stride ldb
 * @param c  Pointer to C(i, j)     — row-stride ldc
 */
template <int R, int V, typename T>
inline void neon_micro(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                       std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t k_len) noexcept {
    vector_micro<Q<T>, R, V>(a, b, c, lda, ldb, ldc, k_len);
}

/// The micro-kernel as blocked_gemm (blocked_driver.hpp) calls it.
using NeonMicro = VectorMicro<Q>;

#endif  // HPC_HAS_NEON

// ============================================================================
// Kernel 3: gemm_neon_blocked  —  tiled i → k → j,  NEON register tile
// ============================================================================

/**
 * @brief NEON GEMM with cache-blocking and register-tiled micro-kernel.
 *
 * Counterpart of gemm_avx2_blocked: i-k-j order, 3-level tiling
 * (kNeonTileM × kNeonTileK × kNeonTileN), and an R rows × V Q-register tile
 * of C held for a whole k-tile, so C is stored once per kNeonTileK steps
 * instead of being reloaded every k-iteration as in gemm_neon_reordered at
 * large N. The default, 6 rows × 4 Q-vectors (6×16 f32, 6×8 f64), is the
 * measured best on Apple M4 (docs/kernels/neon.md); other shapes,
 * gemm_neon_blocked<T, R, V>, run through the same driver
 * (blocked_driver.hpp).
 */
#if !HPC_HAS_NEON
template <typename T, int R = 6, int V = 4, std::size_t TileM = kNeonTileM,
          std::size_t TileK = kNeonTileK, std::size_t TileN = kNeonTileN>
void gemm_neon_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_NEON not available on this target
#else
template <typename T, int R = 6, int V = 4, std::size_t TileM = kNeonTileM,
          std::size_t TileK = kNeonTileK, std::size_t TileN = kNeonTileN>
void gemm_neon_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_neon_blocked: T must be float or double");
    detail::blocked_gemm<NeonMicro, R, V, TileM, TileK, TileN>(A, B, C);
}
#endif  // HPC_HAS_NEON

// ============================================================================
// Kernel 4: gemm_neon_packed  —  GotoBLAS order, B packed into strips
// ============================================================================

// Packed cache block (elements). On M4 every Mc from 256 to 1024 with
// Kc = 512 and Nc = 512 or 1024 is within 3% (docs/kernels/blocks.md);
// this one is best or tied in most cells: an A block of 256 × 512 (512 KiB
// f32), one B strip of 512 × 16 floats (32 KiB) in the 128 KiB L1.
inline constexpr std::size_t kNeonPackMc = 256;
inline constexpr std::size_t kNeonPackKc = 512;
inline constexpr std::size_t kNeonPackNc = 512;

/**
 * @brief gemm_neon_blocked's micro-kernel with B packed (packed_driver.hpp).
 *
 * Each Kc × Nc block of B is copied once into contiguous V-vector (16 f32 / 8 f64
 * column) strips, which removes the cache-set conflicts of B's row stride at
 * power-of-two N; PackA = true copies the A block too. Packing is timed.
 */
#if !HPC_HAS_NEON
template <typename T, int R = 6, int V = 4, std::size_t Mc = kNeonPackMc,
          std::size_t Kc = kNeonPackKc, std::size_t Nc = kNeonPackNc, bool PackA = false>
void gemm_neon_packed(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_NEON not available on this target
#else
template <typename T, int R = 6, int V = 4, std::size_t Mc = kNeonPackMc,
          std::size_t Kc = kNeonPackKc, std::size_t Nc = kNeonPackNc, bool PackA = false>
void gemm_neon_packed(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_neon_packed: T must be float or double");
    detail::packed_gemm<NeonMicro, R, V, Mc, Kc, Nc, PackA>(A, B, C);
}
#endif  // HPC_HAS_NEON

}  // namespace hpc::gemm
