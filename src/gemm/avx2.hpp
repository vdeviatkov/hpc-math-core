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
 *  gemm_avx2_blocked    tiled i → k → j with a 6×16 f32 / 6×8 f64 C tile
 *                       held in YMM registers for a whole k-tile.
 *
 * 16 × 256-bit YMM registers (8 f32 / 4 f64). With 2 FMA ports:
 * 2 × 8 lanes × 2 FLOP = 32 FLOP/cycle f32, 16 f64.
 *
 * Available on Intel Haswell+ and AMD Zen+; not on Apple Silicon.
 * HPC_HAS_AVX2 (hpc/isa.hpp) requires AVX2 and FMA; where it is 0 the kernels
 * are declared `= delete`.
 */

#include "gemm/blocked_driver.hpp"
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
// For Kernel 3: cache tiles (also used by gemm_avx2_blocked_prefetch)
// ============================================================================

// Cache block, the default template arguments of gemm_avx2_blocked: TileM
// rows of A × TileK steps of k × TileN columns of B. The B block is 256 KiB
// in either type (1 KiB of each of 256 rows) and stays in L2; a tall A block
// means each B block, re-read from memory once per i-block, serves 64 row
// strips. Measured on Zen 5: docs/kernels/blocks.md.
inline constexpr std::size_t kAvx2TileM = 384;
inline constexpr std::size_t kAvx2TileK = 256;
template <typename T>
inline constexpr std::size_t kAvx2TileN = 1024 / sizeof(T);  // 256 f32, 128 f64


// ============================================================================
// For Kernel 3: the register-tile micro-kernel
// ============================================================================

#if HPC_HAS_AVX2

/// YMM operations per element type, for avx2_micro.
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
 * @brief AVX2 micro-kernel: C[0..R)[0..V·W) += A[0..R)[0..k_len) × B[0..k_len)[0..V·W)
 *
 * W = 8 f32 / 4 f64 per YMM: an R × V-vector tile of C in R·V
 * accumulators. Per k step: V YMM loads of B row k, R broadcasts of
 * A(i+r, k), R·V FMAs. The library uses R = 6, V = 2 (and R = 4 for the
 * strip that completes a 64-row block). The fixed-count loops are fully unrolled, so the
 * accumulators stay in registers.
 *
 * Why R = 6. Zen 5 needs 12 independent FMA chains for its full rate
 * (2 FMAs per cycle × 4 cycles latency); 4 rows give 8. And B, the operand
 * streamed from L2, is reused by R rows per load: 6 rows read a third less
 * B per FMA than 4. Six rows still fit AVX2's 16 YMM registers without
 * spilling: 12 accumulators and 2 B vectors are live for the whole k step,
 * and each broadcast of A is dead after its two FMAs (GCC 13 alternates two
 * broadcast registers: all 16 in use). Seven rows spill (docs/kernels/avx2.md).
 *
 * @param a  Pointer to A(i, k_blk) — row-stride lda
 * @param b  Pointer to B(k_blk, j) — row-stride ldb
 * @param c  Pointer to C(i, j)     — row-stride ldc
 */
template <int R, int V, typename T>
inline void avx2_micro(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                       std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t k_len) noexcept {
    // The loops below must unroll completely so acc[][] stays in registers.
    static_assert(R >= 1 && R <= 16 && V >= 1 && V <= 16, "tile dimensions up to 16 (the unroll factor)");
    using Ops               = Ymm<T>;
    using Vec               = typename Ops::V;
    constexpr std::size_t W = sizeof(Vec) / sizeof(T);  // lanes per vector

    Vec acc[R][V];
#pragma GCC unroll 16
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            acc[r][v] = Ops::load(c + r * ldc + v * W);
    for (std::size_t k = 0; k < k_len; ++k) {
        Vec bv[V];
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            bv[v] = Ops::load(b + k * ldb + v * W);
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r) {
            const Vec ar = Ops::broadcast(a + r * lda + k);
#pragma GCC unroll 16
            for (int v = 0; v < V; ++v)
                acc[r][v] = Ops::fma(ar, bv[v], acc[r][v]);
        }
    }
#pragma GCC unroll 16
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            Ops::store(c + r * ldc + v * W, acc[r][v]);
}

/// The micro-kernel as blocked_gemm (blocked_driver.hpp) calls it.
struct Avx2Micro {
    template <typename T>
    static constexpr std::size_t kLanes = 32 / sizeof(T);
    template <int R, int V, typename T>
    static void run(const T* a, const T* b, T* c, std::size_t lda, std::size_t ldb, std::size_t ldc,
                    std::size_t k_len) noexcept {
        avx2_micro<R, V>(a, b, c, lda, ldb, ldc, k_len);
    }
};

#endif  // HPC_HAS_AVX2

// ============================================================================
// Kernel 3: gemm_avx2_blocked  —  tiled i → k → j,  register-tiled micro-kernel
// ============================================================================

/**
 * @brief AVX2 GEMM with cache-blocking and register-tiled micro-kernel.
 *
 * i-k-j order and 3-level tiling (as in gemm_blocked), plus a register tile:
 * R rows × V YMM vectors of C stay in registers for a whole k-tile
 * (avx2_micro). The default, 6 × 2, is the measured best on Zen 5
 * (docs/kernels/avx2.md); other shapes, gemm_avx2_blocked<T, R, V>, run
 * through the same driver (blocked_driver.hpp), which covers the edges with
 * smaller tiles of the same kernel.
 *
 * Versus gemm_avx2_reordered at large N: there, C row i (N×sizeof(T)) can
 * fall out of L1 between k-iterations. Here each 6 × 16 f32 C tile stays in
 * registers for kAvx2TileK iterations before it is stored, and the outer
 * tiling keeps the A and B tiles in L2.
 */
#if !HPC_HAS_AVX2
template <typename T, int R = 6, int V = 2, std::size_t TileM = kAvx2TileM,
          std::size_t TileK = kAvx2TileK, std::size_t TileN = kAvx2TileN<T>>
void gemm_avx2_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // AVX2 not available on this target
#else
template <typename T, int R = 6, int V = 2, std::size_t TileM = kAvx2TileM,
          std::size_t TileK = kAvx2TileK, std::size_t TileN = kAvx2TileN<T>>
void gemm_avx2_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_avx2_blocked: T must be float or double");
    detail::blocked_gemm<Avx2Micro, R, V, TileM, TileK, TileN>(A, B, C);
}
#endif  // HPC_HAS_AVX2

}  // namespace hpc::gemm
