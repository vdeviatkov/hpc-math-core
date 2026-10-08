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
 * HPC_HAS_AVX2 (hpc/isa.hpp) follows __AVX2__; where it is 0 the kernels
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
// Shared micro-kernel constants (used by gemm_avx2_blocked)
// ============================================================================

inline constexpr std::size_t kAvx2TileM = 64;   // outer i-tile
inline constexpr std::size_t kAvx2TileK = 256;  // outer k-tile
inline constexpr std::size_t kAvx2TileN = 256;  // outer j-tile

inline constexpr std::size_t kF32RegRows = 4;   // C rows held in registers (f32 micro-kernel)
inline constexpr std::size_t kF32RegCols = 2;   // YMM vectors per C row   (2×8  = 16 j-elems)
inline constexpr std::size_t kF64RegRows = 4;   // C rows held in registers (f64 micro-kernel)
inline constexpr std::size_t kF64RegCols = 2;   // YMM vectors per C row   (2×4  =  8 j-elems)

// ============================================================================
// Shared AVX2 micro-kernels (used only by gemm_avx2_blocked)
// ============================================================================

#if HPC_HAS_AVX2

/// f32 register-tiled micro-kernel: C[i..i+3][j..j+15] += A[i..i+3][k..k+k_len) × B[k..][j..)
inline void avx2_micro_f32_4x16(const float* __restrict__ a,  // A(i, k_blk) — stride lda
                                const float* __restrict__ b,  // B(k_blk, j) — stride ldb
                                float* __restrict__ c0,       // C(i,   j)
                                float* __restrict__ c1,       // C(i+1, j)
                                float* __restrict__ c2,       // C(i+2, j)
                                float* __restrict__ c3,       // C(i+3, j)
                                std::size_t lda, std::size_t ldb, std::size_t k_len) noexcept {
    // Load the 4×16 C tile into 8 YMM accumulators.
    __m256 c00 = _mm256_loadu_ps(c0), c01 = _mm256_loadu_ps(c0 + 8);
    __m256 c10 = _mm256_loadu_ps(c1), c11 = _mm256_loadu_ps(c1 + 8);
    __m256 c20 = _mm256_loadu_ps(c2), c21 = _mm256_loadu_ps(c2 + 8);
    __m256 c30 = _mm256_loadu_ps(c3), c31 = _mm256_loadu_ps(c3 + 8);

    for (std::size_t k = 0; k < k_len; ++k) {
        const __m256 b0 = _mm256_loadu_ps(b + k * ldb);
        const __m256 b1 = _mm256_loadu_ps(b + k * ldb + 8);
        const __m256 a0 = _mm256_broadcast_ss(a + 0 * lda + k);
        const __m256 a1 = _mm256_broadcast_ss(a + 1 * lda + k);
        const __m256 a2 = _mm256_broadcast_ss(a + 2 * lda + k);
        const __m256 a3 = _mm256_broadcast_ss(a + 3 * lda + k);
        c00             = _mm256_fmadd_ps(a0, b0, c00);
        c01             = _mm256_fmadd_ps(a0, b1, c01);
        c10             = _mm256_fmadd_ps(a1, b0, c10);
        c11             = _mm256_fmadd_ps(a1, b1, c11);
        c20             = _mm256_fmadd_ps(a2, b0, c20);
        c21             = _mm256_fmadd_ps(a2, b1, c21);
        c30             = _mm256_fmadd_ps(a3, b0, c30);
        c31             = _mm256_fmadd_ps(a3, b1, c31);
    }

    _mm256_storeu_ps(c0, c00);
    _mm256_storeu_ps(c0 + 8, c01);
    _mm256_storeu_ps(c1, c10);
    _mm256_storeu_ps(c1 + 8, c11);
    _mm256_storeu_ps(c2, c20);
    _mm256_storeu_ps(c2 + 8, c21);
    _mm256_storeu_ps(c3, c30);
    _mm256_storeu_ps(c3 + 8, c31);
}

/// f64 register-tiled micro-kernel: C[i..i+3][j..j+7] += A[i..i+3][k..k+k_len) × B[k..][j..)
inline void avx2_micro_f64_4x8(const double* __restrict__ a, const double* __restrict__ b,
                               double* __restrict__ c0, double* __restrict__ c1,
                               double* __restrict__ c2, double* __restrict__ c3, std::size_t lda,
                               std::size_t ldb, std::size_t k_len) noexcept {
    __m256d c00 = _mm256_loadu_pd(c0), c01 = _mm256_loadu_pd(c0 + 4);
    __m256d c10 = _mm256_loadu_pd(c1), c11 = _mm256_loadu_pd(c1 + 4);
    __m256d c20 = _mm256_loadu_pd(c2), c21 = _mm256_loadu_pd(c2 + 4);
    __m256d c30 = _mm256_loadu_pd(c3), c31 = _mm256_loadu_pd(c3 + 4);

    for (std::size_t k = 0; k < k_len; ++k) {
        const __m256d b0 = _mm256_loadu_pd(b + k * ldb);
        const __m256d b1 = _mm256_loadu_pd(b + k * ldb + 4);
        const __m256d a0 = _mm256_broadcast_sd(a + 0 * lda + k);
        const __m256d a1 = _mm256_broadcast_sd(a + 1 * lda + k);
        const __m256d a2 = _mm256_broadcast_sd(a + 2 * lda + k);
        const __m256d a3 = _mm256_broadcast_sd(a + 3 * lda + k);
        c00              = _mm256_fmadd_pd(a0, b0, c00);
        c01              = _mm256_fmadd_pd(a0, b1, c01);
        c10              = _mm256_fmadd_pd(a1, b0, c10);
        c11              = _mm256_fmadd_pd(a1, b1, c11);
        c20              = _mm256_fmadd_pd(a2, b0, c20);
        c21              = _mm256_fmadd_pd(a2, b1, c21);
        c30              = _mm256_fmadd_pd(a3, b0, c30);
        c31              = _mm256_fmadd_pd(a3, b1, c31);
    }

    _mm256_storeu_pd(c0, c00);
    _mm256_storeu_pd(c0 + 4, c01);
    _mm256_storeu_pd(c1, c10);
    _mm256_storeu_pd(c1 + 4, c11);
    _mm256_storeu_pd(c2, c20);
    _mm256_storeu_pd(c2 + 4, c21);
    _mm256_storeu_pd(c3, c30);
    _mm256_storeu_pd(c3 + 4, c31);
}

#endif  // HPC_HAS_AVX2

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
                    // Each is ldb floats apart — not sequential, not cacheable.
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
                    for (; j + kJStep <= j_end; j += kJStep) {
                        T* c0 = C.data() + (i + 0) * ldc + j;
                        T* c1 = C.data() + (i + 1) * ldc + j;
                        T* c2 = C.data() + (i + 2) * ldc + j;
                        T* c3 = C.data() + (i + 3) * ldc + j;
                        if constexpr (sizeof(T) == 4) {
                            avx2_micro_f32_4x16(reinterpret_cast<const float*>(a_ptr),
                                                reinterpret_cast<const float*>(b_ptr + (j - j_blk)),
                                                reinterpret_cast<float*>(c0),
                                                reinterpret_cast<float*>(c1),
                                                reinterpret_cast<float*>(c2),
                                                reinterpret_cast<float*>(c3), lda, ldb, k_len);
                        } else {
                            avx2_micro_f64_4x8(reinterpret_cast<const double*>(a_ptr),
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
#endif  // HPC_HAS_AVX2

}  // namespace hpc::gemm
