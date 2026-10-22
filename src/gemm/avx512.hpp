#pragma once

/**
 * @file avx512.hpp
 * @brief AVX-512 counterparts of the three AVX2 kernels (avx2.hpp).
 *
 * Same loop structures, 512-bit ZMM registers (16 f32 / 8 f64 per FMA)
 * instead of 256-bit YMM, so the numbers show what doubling the vector width
 * buys at each optimisation level:
 *
 *  gemm_avx512_naive      i → j → k. B is still a stride-N gather, so it
 *                         runs at about scalar-naive speed.
 *  gemm_avx512_reordered  i → k → j. Measured within a few percent of
 *                         AVX2/scalar reordered on Zen 5: the loop streams B
 *                         and C, so it is limited by memory traffic, not FMA
 *                         width.
 *  gemm_avx512_blocked    tiled i → k → j with an 8×48 f32 / 8×24 f64 register
 *                         tile (8 rows × 3 ZMM). The fastest of the three.
 *
 * Peak (2 FMA ports): 2 × 16 lanes × 2 FLOP = 64 FLOP/cycle f32, 32 f64.
 * The 32 ZMM registers leave room for the 24 accumulators and 3 B vectors
 * of the micro-kernel without spilling.
 *
 * Available on Intel Skylake-SP/X and later server/workstation parts
 * (Cascade Lake, Ice Lake, Sapphire Rapids, Rocket Lake) and AMD Zen 4+.
 * Not on Apple Silicon, Alder Lake/Raptor Lake (disabled by Intel), or AMD
 * before Zen 4.
 *
 * HPC_HAS_AVX512 (hpc/isa.hpp) follows __AVX512F__; where it is 0 the
 * kernels are declared `= delete`. -march=native enables AVX-512 on a
 * capable CPU; HPC_ENABLE_AVX512=ON forces the flags otherwise.
 */

#include "gemm/blocked_driver.hpp"
#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"

#if HPC_HAS_AVX512
    #include <immintrin.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace hpc::gemm {

// ============================================================================
// Kernel 1: gemm_avx512_naive  —  i → j → k,  512-bit SIMD on the k-loop
// ============================================================================

/**
 * @brief AVX-512 GEMM with naive i-j-k loop order.
 *
 * Counterpart of gemm_avx2_naive with 16 f32 / 8 f64 per FMA. The B column
 * is still a stride-N gather, so it runs at about scalar-naive speed: vector
 * width doesn't help a cache-miss-bound loop.
 */
#if !HPC_HAS_AVX512
template <typename T>
void gemm_avx512_naive(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // AVX512F not available on this target
#else
template <typename T>
void gemm_avx512_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_avx512_naive: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    // ZMM SIMD width in elements.
    constexpr std::size_t W = (sizeof(T) == 4) ? 16 : 8;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            if constexpr (sizeof(T) == 4) {
                __m512 acc    = _mm512_setzero_ps();
                std::size_t k = 0;
                for (; k + W <= K; k += W) {
                    // A(i, k..k+15): sequential load from row i.
                    const __m512 a_vec = _mm512_loadu_ps(A.data() + i * lda + k);
                    // B column j: gather 16 elements spaced ldb apart.
                    // Each access is a separate cache line for large N.
                    alignas(64) float b_col[16] = {
                        B.data()[(k + 0) * ldb + j],  B.data()[(k + 1) * ldb + j],
                        B.data()[(k + 2) * ldb + j],  B.data()[(k + 3) * ldb + j],
                        B.data()[(k + 4) * ldb + j],  B.data()[(k + 5) * ldb + j],
                        B.data()[(k + 6) * ldb + j],  B.data()[(k + 7) * ldb + j],
                        B.data()[(k + 8) * ldb + j],  B.data()[(k + 9) * ldb + j],
                        B.data()[(k + 10) * ldb + j], B.data()[(k + 11) * ldb + j],
                        B.data()[(k + 12) * ldb + j], B.data()[(k + 13) * ldb + j],
                        B.data()[(k + 14) * ldb + j], B.data()[(k + 15) * ldb + j],
                    };
                    const __m512 b_vec = _mm512_load_ps(b_col);
                    acc                = _mm512_fmadd_ps(a_vec, b_vec, acc);
                }
                // Horizontal reduce ZMM → scalar.
                float s = _mm512_reduce_add_ps(acc);
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            } else {
                __m512d acc   = _mm512_setzero_pd();
                std::size_t k = 0;
                for (; k + W <= K; k += W) {
                    const __m512d a_vec         = _mm512_loadu_pd(A.data() + i * lda + k);
                    alignas(64) double b_col[8] = {
                        B.data()[(k + 0) * ldb + j], B.data()[(k + 1) * ldb + j],
                        B.data()[(k + 2) * ldb + j], B.data()[(k + 3) * ldb + j],
                        B.data()[(k + 4) * ldb + j], B.data()[(k + 5) * ldb + j],
                        B.data()[(k + 6) * ldb + j], B.data()[(k + 7) * ldb + j],
                    };
                    const __m512d b_vec = _mm512_load_pd(b_col);
                    acc                 = _mm512_fmadd_pd(a_vec, b_vec, acc);
                }
                double s = _mm512_reduce_add_pd(acc);
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            }
        }
    }
}
#endif  // HPC_HAS_AVX512

// ============================================================================
// Kernel 2: gemm_avx512_reordered  —  i → k → j,  512-bit SIMD on the j-loop
// ============================================================================

/**
 * @brief AVX-512 GEMM with cache-friendly i-k-j loop order.
 *
 * Counterpart of gemm_avx2_reordered: A(i,k) is broadcast to 16 lanes and the
 * j-loop processes 16 f32 / 8 f64 per FMA, with B and C read stride-1.
 *
 * Measured: within a few percent of AVX2/scalar reordered on Zen 5 — not
 * the 2× the wider FMA would suggest, since the loop streams B and C rather
 * than being compute-bound. C row eviction from L1 at large N is the same
 * as AVX2.
 */
#if !HPC_HAS_AVX512
template <typename T>
void gemm_avx512_reordered(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // AVX512F not available on this target
#else
template <typename T>
void gemm_avx512_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_avx512_reordered: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    constexpr std::size_t W = (sizeof(T) == 4) ? 16 : 8;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            if constexpr (sizeof(T) == 4) {
                const __m512 a_broad = _mm512_set1_ps(A.data()[i * lda + k]);
                const float* b_row   = B.data() + k * ldb;
                float* c_row         = C.data() + i * ldc;

                std::size_t j = 0;
                for (; j + W <= N; j += W) {
                    const __m512 b_vec = _mm512_loadu_ps(b_row + j);
                    const __m512 c_vec = _mm512_loadu_ps(c_row + j);
                    _mm512_storeu_ps(c_row + j, _mm512_fmadd_ps(a_broad, b_vec, c_vec));
                }
                const float a_scalar = A.data()[i * lda + k];
                for (; j < N; ++j)
                    c_row[j] += a_scalar * b_row[j];
            } else {
                const __m512d a_broad = _mm512_set1_pd(A.data()[i * lda + k]);
                const double* b_row   = B.data() + k * ldb;
                double* c_row         = C.data() + i * ldc;

                std::size_t j = 0;
                for (; j + W <= N; j += W) {
                    const __m512d b_vec = _mm512_loadu_pd(b_row + j);
                    const __m512d c_vec = _mm512_loadu_pd(c_row + j);
                    _mm512_storeu_pd(c_row + j, _mm512_fmadd_pd(a_broad, b_vec, c_vec));
                }
                const double a_scalar = A.data()[i * lda + k];
                for (; j < N; ++j)
                    c_row[j] += a_scalar * b_row[j];
            }
        }
    }
}
#endif  // HPC_HAS_AVX512

// ============================================================================
// For Kernel 3: cache tiles (also used by gemm_avx512_blocked_prefetch)
// ============================================================================

// Cache block, the default template arguments of gemm_avx512_blocked: TileM
// rows of A × TileK steps of k × TileN columns of B. The B block is 512 KiB
// in either type (2 KiB of each of 256 rows), half of Zen 5's 1 MiB L2; a
// tall A block means each B block, re-read from memory once per i-block,
// serves 64 row strips. Measured on Zen 5: docs/kernels/blocks.md.
inline constexpr std::size_t kAvx512TileM = 512;
inline constexpr std::size_t kAvx512TileK = 256;
template <typename T>
inline constexpr std::size_t kAvx512TileN = 2048 / sizeof(T);  // 512 f32, 256 f64

// ============================================================================
// For Kernel 3: the register-tile micro-kernel
// ============================================================================

#if HPC_HAS_AVX512

/**
 * @brief AVX-512 micro-kernel: C[0..R)[0..V·W) += A[0..R)[0..k_len) × B[0..k_len)[0..V·W)
 *
 * W = 16 f32 / 8 f64 per ZMM. Per k step: V ZMM loads of B row k, R
 * broadcasts of A(i+r, k), R·V FMAs into R·V accumulators. The default,
 * R = 8, V = 3, uses 24 accumulators and 28 of the 32 ZMM registers.
 *
 * Why at least 6 rows. Zen 5 retires 2 FMAs per cycle with a 4-cycle latency,
 * so 8 independent accumulators are the bare minimum and 12 are needed for
 * the full rate (fma_peak). With 4 rows × 2 ZMM, GCC 13 also folds the B
 * loads into the FMAs as memory operands (vfmadd231ps (mem), …), re-loading
 * B for every FMA: 12 loads per k step for 8 FMAs, so the kernel ran into
 * the load ports at 60% of peak. With 6 or more rows it loads B once per k
 * step and every FMA is register-only: 95–97% of peak on L1-resident data
 * (simd_microkernel). Why the third vector: docs/kernels/avx512.md.
 *
 * @param a      Pointer to A(i, k_blk) — row-stride lda
 * @param b      Pointer to B(k_blk, j) — row-stride ldb
 * @param c      Pointer to C(i, j)     — row-stride ldc
 */
/// ZMM operations per element type, for avx512_micro.
template <typename T> struct Zmm;
template <> struct Zmm<float> {
    using V = __m512;
    static V load(const float* p) { return _mm512_loadu_ps(p); }
    static void store(float* p, V v) { _mm512_storeu_ps(p, v); }
    static V splat(float x) { return _mm512_set1_ps(x); }
    static V fma(V x, V y, V acc) { return _mm512_fmadd_ps(x, y, acc); }
};
template <> struct Zmm<double> {
    using V = __m512d;
    static V load(const double* p) { return _mm512_loadu_pd(p); }
    static void store(double* p, V v) { _mm512_storeu_pd(p, v); }
    static V splat(double x) { return _mm512_set1_pd(x); }
    static V fma(V x, V y, V acc) { return _mm512_fmadd_pd(x, y, acc); }
};

template <int R, int V, typename T>
inline void avx512_micro(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                         std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t k_len) noexcept {
    // The loops below must unroll completely so acc[][] stays in registers.
    static_assert(R >= 1 && R <= 16 && V >= 1 && V <= 16, "tile dimensions up to 16 (the unroll factor)");
    using Ops               = Zmm<T>;
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
            const Vec ar = Ops::splat(a[r * lda + k]);
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
struct Avx512Micro {
    template <typename T>
    static constexpr std::size_t kLanes = 64 / sizeof(T);
    template <int R, int V, typename T>
    static void run(const T* a, const T* b, T* c, std::size_t lda, std::size_t ldb, std::size_t ldc,
                    std::size_t k_len) noexcept {
        avx512_micro<R, V>(a, b, c, lda, ldb, ldc, k_len);
    }
};

#endif  // HPC_HAS_AVX512

// ============================================================================
// Kernel 3: gemm_avx512_blocked  —  tiled i → k → j,  512-bit register tile
// ============================================================================

/**
 * @brief AVX-512 GEMM with cache-blocking and 512-bit register-tiled micro-kernel.
 *
 * Counterpart of gemm_avx2_blocked: i-k-j order, 3-level tiling
 * (kAvx512TileM × kAvx512TileK × kAvx512TileN = 512 × 256 × 512 f32 or
 * × 256 f64), and an
 * R rows × V ZMM register tile of C held for a whole k-tile, so C is stored
 * once per kAvx512TileK steps instead of every step. The default, 8 × 3
 * (8×48 f32 / 8×24 f64, 24 accumulators, 28 of 32 ZMM registers), is the
 * measured best on Zen 5 (docs/kernels/avx512.md); other shapes,
 * gemm_avx512_blocked<T, R, V>, run through the same driver
 * (blocked_driver.hpp).
 *
 * Block footprints: B 512 KiB in either type (with 512 f64 columns it was
 * 1 MiB, all of Zen 5's L2, and f64 ran up to 30% slower), A 512 KiB f32 /
 * 1 MiB f64. The cache block and the register tile are template parameters:
 * gemm_avx512_blocked<T, R, V, TileM, TileK, TileN>.
 */
#if !HPC_HAS_AVX512
template <typename T, int R = 8, int V = 3, std::size_t TileM = kAvx512TileM,
          std::size_t TileK = kAvx512TileK, std::size_t TileN = kAvx512TileN<T>>
void gemm_avx512_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // AVX512F not available on this target
#else
template <typename T, int R = 8, int V = 3, std::size_t TileM = kAvx512TileM,
          std::size_t TileK = kAvx512TileK, std::size_t TileN = kAvx512TileN<T>>
void gemm_avx512_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_avx512_blocked: T must be float or double");
    detail::blocked_gemm<Avx512Micro, R, V, TileM, TileK, TileN>(A, B, C);
}
#endif  // HPC_HAS_AVX512

}  // namespace hpc::gemm
