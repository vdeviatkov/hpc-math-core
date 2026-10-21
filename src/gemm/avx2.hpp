#pragma once

/**
 * @file avx2.hpp
 * @brief Three AVX2 + FMA GEMM kernels, one optimisation step apart.
 *
 *  gemm_avx2_naive      i → j → k, 8 f32 / 4 f64 per FMA on the k-loop. B is
 *                       read down a column (stride N), so cache misses cap it
 *                       near scalar-naive speed.
 *  gemm_avx2_reordered  i → k → j: broadcast A(i,k), FMA against stride-1
 *                       B and C rows. About the same as scalar
 *                       gemm_reordered, which the compiler auto-vectorises.
 *  gemm_avx2_blocked    tiled i → k → j with a 4×16 f32 / 4×8 f64 C tile
 *                       held in YMM registers for a whole k-tile.
 *
 * 16 × 256-bit YMM registers (8 f32 / 4 f64). With 2 FMA ports:
 * 2 × 8 lanes × 2 FLOP = 32 FLOP/cycle f32, 16 f64.
 *
 * Available on Intel Haswell+ and AMD Zen+; not on Apple Silicon.
 * HPC_HAS_AVX2 (hpc/isa.hpp) requires AVX2 and FMA; where it is 0 the kernels
 * are declared `= delete`.
 */

#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"

#if HPC_HAS_AVX2
    #include <immintrin.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace hpc::gemm {

// ============================================================================
// Kernel 1: gemm_avx2_naive  —  i → j → k,  SIMD on the k-loop
// ============================================================================

/**
 * @brief AVX2 GEMM with naive i-j-k loop order.
 *
 * The k-loop is 8 wide (f32): a YMM holds A(i, k..k+7) and B(k..k+7, j),
 * the products accumulate in a YMM and are summed horizontally at the end.
 * B(k, j) with fixed j and varying k is a column of B: consecutive k are N
 * elements apart, so each B element sits on a different cache line and has
 * to be gathered. SIMD width doesn't change that; GFLOP/s stays close to
 * scalar naive. The kernel is kept as a measurement point for exactly that.
 */
#if !HPC_HAS_AVX2
template <typename T>
void gemm_avx2_naive(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // AVX2 not available on this target
#else
template <typename T>
void gemm_avx2_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_avx2_naive: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    // SIMD width in elements for this type.
    constexpr std::size_t W = (sizeof(T) == 4) ? 8 : 4;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            // Accumulate over k using SIMD.
            // B(k, j) is accessed with stride ldb — a column walk.

            if constexpr (sizeof(T) == 4) {
                __m256 acc    = _mm256_setzero_ps();
                std::size_t k = 0;
                // Process 8 k-elements per iteration: load A(i, k..k+7) sequentially,
                // but B(k..k+7, j) must be gathered (stride ldb apart).
                // We explicitly gather into a temporary to make the memory pattern visible.
                for (; k + W <= K; k += W) {
                    const __m256 a_vec = _mm256_loadu_ps(A.data() + i * lda + k);
                    // Manual gather of B column j: B(k,j), B(k+1,j), ..., B(k+7,j)
                    // Each is ldb floats apart — not sequential; cache reuse depends on the working set.
                    alignas(32) float b_col[8] = {
                        B.data()[(k + 0) * ldb + j], B.data()[(k + 1) * ldb + j],
                        B.data()[(k + 2) * ldb + j], B.data()[(k + 3) * ldb + j],
                        B.data()[(k + 4) * ldb + j], B.data()[(k + 5) * ldb + j],
                        B.data()[(k + 6) * ldb + j], B.data()[(k + 7) * ldb + j],
                    };
                    const __m256 b_vec = _mm256_load_ps(b_col);
                    acc                = _mm256_fmadd_ps(a_vec, b_vec, acc);
                }
                // Horizontal sum of the YMM accumulator → scalar.
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, acc);
                float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                // Scalar tail for remaining k elements.
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            } else {
                __m256d acc   = _mm256_setzero_pd();
                std::size_t k = 0;
                for (; k + W <= K; k += W) {
                    const __m256d a_vec         = _mm256_loadu_pd(A.data() + i * lda + k);
                    alignas(32) double b_col[4] = {
                        B.data()[(k + 0) * ldb + j],
                        B.data()[(k + 1) * ldb + j],
                        B.data()[(k + 2) * ldb + j],
                        B.data()[(k + 3) * ldb + j],
                    };
                    const __m256d b_vec = _mm256_load_pd(b_col);
                    acc                 = _mm256_fmadd_pd(a_vec, b_vec, acc);
                }
                alignas(32) double tmp[4];
                _mm256_store_pd(tmp, acc);
                double s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            }
        }
    }
}
#endif  // HPC_HAS_AVX2

// ============================================================================
// Kernel 2: gemm_avx2_reordered  —  i → k → j,  SIMD on the j-loop
// ============================================================================

/**
 * @brief AVX2 GEMM with cache-friendly i-k-j loop order.
 *
 * A(i,k) is broadcast once per k; B(k, j..) and C(i, j..) are read with
 * stride 1, so no gather is needed and every cache line is fully used. No
 * blocking: it degrades at large N when C row i no longer fits in L1.
 */
#if !HPC_HAS_AVX2
template <typename T>
void gemm_avx2_reordered(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // AVX2 not available on this target
#else
template <typename T>
void gemm_avx2_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_avx2_reordered: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    constexpr std::size_t W = (sizeof(T) == 4) ? 8 : 4;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            if constexpr (sizeof(T) == 4) {
                const __m256 a_broad = _mm256_broadcast_ss(A.data() + i * lda + k);
                const float* b_row   = B.data() + k * ldb;
                float* c_row         = C.data() + i * ldc;

                std::size_t j = 0;
                for (; j + W <= N; j += W) {
                    const __m256 b_vec = _mm256_loadu_ps(b_row + j);
                    const __m256 c_vec = _mm256_loadu_ps(c_row + j);
                    _mm256_storeu_ps(c_row + j, _mm256_fmadd_ps(a_broad, b_vec, c_vec));
                }
                // Scalar tail.
                const float a_scalar = A.data()[i * lda + k];
                for (; j < N; ++j)
                    c_row[j] += a_scalar * b_row[j];
            } else {
                const __m256d a_broad = _mm256_broadcast_sd(A.data() + i * lda + k);
                const double* b_row   = B.data() + k * ldb;
                double* c_row         = C.data() + i * ldc;

                std::size_t j = 0;
                for (; j + W <= N; j += W) {
                    const __m256d b_vec = _mm256_loadu_pd(b_row + j);
                    const __m256d c_vec = _mm256_loadu_pd(c_row + j);
                    _mm256_storeu_pd(c_row + j, _mm256_fmadd_pd(a_broad, b_vec, c_vec));
                }
                const double a_scalar = A.data()[i * lda + k];
                for (; j < N; ++j)
                    c_row[j] += a_scalar * b_row[j];
            }
        }
    }
}
#endif  // HPC_HAS_AVX2

// ============================================================================
// For Kernel 3: tile sizes (also used by gemm_avx2_blocked_prefetch)
// ============================================================================

inline constexpr std::size_t kAvx2TileM = 64;   // outer i-tile
inline constexpr std::size_t kAvx2TileK = 256;  // outer k-tile
inline constexpr std::size_t kAvx2TileN = 256;  // outer j-tile

inline constexpr std::size_t kF32RegRows = 4;   // C rows held in registers (f32 micro-kernel)
inline constexpr std::size_t kF32RegCols = 2;   // YMM vectors per C row   (2×8  = 16 j-elems)
inline constexpr std::size_t kF64RegRows = 4;   // C rows held in registers (f64 micro-kernel)
inline constexpr std::size_t kF64RegCols = 2;   // YMM vectors per C row   (2×4  =  8 j-elems)

// ============================================================================
// For Kernel 3: the register-tile micro-kernel
// ============================================================================

#if HPC_HAS_AVX2

/// YMM operations per element type, for avx2_micro_4x2.
template <typename T> struct Ymm;
template <> struct Ymm<float> {
    using V = __m256;
    static V load(const float* p) { return _mm256_loadu_ps(p); }
    static void store(float* p, V v) { _mm256_storeu_ps(p, v); }
    static V broadcast(const float* p) { return _mm256_broadcast_ss(p); }
    static V fma(V x, V y, V acc) { return _mm256_fmadd_ps(x, y, acc); }
};
template <> struct Ymm<double> {
    using V = __m256d;
    static V load(const double* p) { return _mm256_loadu_pd(p); }
    static void store(double* p, V v) { _mm256_storeu_pd(p, v); }
    static V broadcast(const double* p) { return _mm256_broadcast_sd(p); }
    static V fma(V x, V y, V acc) { return _mm256_fmadd_pd(x, y, acc); }
};

/**
 * @brief AVX2 micro-kernel: C[0..4)[0..2W) += A[0..4)[0..k_len) × B[0..k_len)[0..2W)
 *
 * W = 8 f32 / 4 f64 per YMM: a 4 × 16 f32 or 4 × 8 f64 tile of C in 8
 * accumulators. Per k step: 2 YMM loads of B row k, 4 broadcasts of
 * A(i+r, k), 8 FMAs. The fixed-count loops are fully unrolled, so the
 * accumulators stay in registers (8 + 2 + 4 of the 16 YMM registers).
 *
 * @param a  Pointer to A(i, k_blk) — row-stride lda
 * @param b  Pointer to B(k_blk, j) — row-stride ldb
 * @param c  Pointer to C(i, j)     — row-stride ldc
 */
template <typename T>
inline void avx2_micro_4x2(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                           std::size_t lda, std::size_t ldb, std::size_t ldc,
                           std::size_t k_len) noexcept {
    using Y                 = Ymm<T>;
    using V                 = typename Y::V;
    constexpr std::size_t W = 32 / sizeof(T);
    constexpr int R         = 4;

    V acc[R][2];
#pragma GCC unroll 4
    for (int r = 0; r < R; ++r) {
        acc[r][0] = Y::load(c + r * ldc);
        acc[r][1] = Y::load(c + r * ldc + W);
    }
    for (std::size_t k = 0; k < k_len; ++k) {
        const V b0 = Y::load(b + k * ldb);
        const V b1 = Y::load(b + k * ldb + W);
#pragma GCC unroll 4
        for (int r = 0; r < R; ++r) {
            const V ar = Y::broadcast(a + r * lda + k);
            acc[r][0]  = Y::fma(ar, b0, acc[r][0]);
            acc[r][1]  = Y::fma(ar, b1, acc[r][1]);
        }
    }
#pragma GCC unroll 4
    for (int r = 0; r < R; ++r) {
        Y::store(c + r * ldc, acc[r][0]);
        Y::store(c + r * ldc + W, acc[r][1]);
    }
}

#endif  // HPC_HAS_AVX2

// ============================================================================
// Kernel 3: gemm_avx2_blocked  —  tiled i → k → j,  register-tiled micro-kernel
// ============================================================================

/**
 * @brief AVX2 GEMM with cache-blocking and register-tiled micro-kernel.
 *
 * i-k-j order and 3-level tiling (as in gemm_blocked), plus a register tile:
 * 4 rows × 2 YMM of C stay in registers for a whole k-tile. Per k step the
 * micro-kernel keeps 8 accumulators, 4 broadcasts and 2 B vectors live
 * (14 of 16 YMM).
 *
 * Versus gemm_avx2_reordered at large N: there, C row i (N×sizeof(T)) can
 * fall out of L1 between k-iterations. Here the 4×16 f32 C sub-tile stays in
 * registers for kAvx2TileK iterations before it is stored, and the outer
 * tiling keeps the A and B tiles in L2.
 */
#if !HPC_HAS_AVX2
template <typename T>
void gemm_avx2_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // AVX2 not available on this target
#else
template <typename T>
void gemm_avx2_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_avx2_blocked: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    constexpr std::size_t kSimdW   = (sizeof(T) == 4) ? 8 : 4;
    constexpr std::size_t kRegRows = (sizeof(T) == 4) ? kF32RegRows : kF64RegRows;
    constexpr std::size_t kRegCols = (sizeof(T) == 4) ? kF32RegCols : kF64RegCols;
    constexpr std::size_t kJStep   = kSimdW * kRegCols;  // j-elements per micro-kernel call

    for (std::size_t i_blk = 0; i_blk < M; i_blk += kAvx2TileM) {
        const std::size_t i_end = std::min(i_blk + kAvx2TileM, M);

        for (std::size_t k_blk = 0; k_blk < K; k_blk += kAvx2TileK) {
            const std::size_t k_end = std::min(k_blk + kAvx2TileK, K);
            const std::size_t k_len = k_end - k_blk;

            for (std::size_t j_blk = 0; j_blk < N; j_blk += kAvx2TileN) {
                const std::size_t j_end = std::min(j_blk + kAvx2TileN, N);

                // --- AVX2 hot path: kRegRows rows × kJStep cols at once ---
                std::size_t i = i_blk;
                for (; i + kRegRows <= i_end; i += kRegRows) {
                    const T* a_ptr = A.data() + i * lda + k_blk;
                    const T* b_ptr = B.data() + k_blk * ldb + j_blk;

                    std::size_t j = j_blk;
                    for (; j + kJStep <= j_end; j += kJStep)
                        avx2_micro_4x2(a_ptr, b_ptr + (j - j_blk), C.data() + i * ldc + j, lda, ldb,
                                       ldc, k_len);
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
#endif  // HPC_HAS_AVX2

}  // namespace hpc::gemm
