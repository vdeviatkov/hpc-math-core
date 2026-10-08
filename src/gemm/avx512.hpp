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
 *  gemm_avx512_blocked    tiled i → k → j with a 4×32 f32 / 4×16 f64 register
 *                         tile (4 rows × 2 ZMM). The fastest of the three.
 *
 * Peak (2 FMA ports): 2 × 16 lanes × 2 FLOP = 64 FLOP/cycle f32, 32 f64.
 * The 32 ZMM registers leave room for the 8 accumulators, 4 broadcasts and
 * 2 B vectors of the micro-kernel without spilling.
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
// AVX-512 tile / unroll constants
// ============================================================================

inline constexpr std::size_t kAvx512TileM = 64;   // outer i-tile (same as AVX2)
inline constexpr std::size_t kAvx512TileK = 256;  // outer k-tile (same as AVX2)
inline constexpr std::size_t kAvx512TileN = 512;  // outer j-tile — wider: 512 f32 = 2 KB/row

// Micro-kernel unroll: 4 C rows × 2 ZMM vectors per row.
// f32: 2 × 16 = 32 j-elements per call  (vs 16 in AVX2)
// f64: 2 ×  8 = 16 j-elements per call  (vs  8 in AVX2)
inline constexpr std::size_t kAvx512F32RegRows = 4;
inline constexpr std::size_t kAvx512F32RegCols = 2;  // 2 ZMM → 32 f32
inline constexpr std::size_t kAvx512F64RegRows = 4;
inline constexpr std::size_t kAvx512F64RegCols = 2;  // 2 ZMM → 16 f64

// ============================================================================
// AVX-512 micro-kernels
// ============================================================================

#if HPC_HAS_AVX512

/**
 * @brief AVX-512 f32 micro-kernel: C[i..i+3][j..j+31] += A[i..i+3][k_blk..k_end) × B[..][j..)
 *
 * Live vectors per k step: 8 C accumulators (4 rows × 2 ZMM), 4 broadcasts
 * of A(i+r, k), 2 B vectors — 14 of 32 ZMM.
 *
 * Note on embedded broadcast:
 *   AVX-512 supports a memory-source broadcast operand in FMA:
 *     vfmadd231ps zmm_acc, zmm_b, [mem]{1to16}
 *   which encodes broadcast + FMA in one instruction. It doesn't pay here:
 *   each broadcast A(i+r, k) feeds 2 FMAs (two ZMM columns), so the compiler
 *   broadcasts it once into a register instead. GCC 13 -O3 -march=native on
 *   Zen 5 emits vbroadcastss/sd and no {1to16} operands for this kernel.
 *
 * @param a       Pointer to A(i, k_blk) — row-stride lda
 * @param b       Pointer to B(k_blk, j) — row-stride ldb
 * @param c0..c3  Pointers to C(i+0..3, j)
 * @param lda     Leading dimension of A (= K for row-major)
 * @param ldb     Leading dimension of B (= N for row-major)
 * @param k_len   Number of k iterations to process
 */
inline void avx512_micro_f32_4x32(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ c0, float* __restrict__ c1,
                                  float* __restrict__ c2, float* __restrict__ c3, std::size_t lda,
                                  std::size_t ldb, std::size_t k_len) noexcept {
    // Load 4×32 C tile: 8 ZMM accumulators.
    // Each ZMM covers 16 f32 = 64 bytes = 1 cache line.
    __m512 c00 = _mm512_loadu_ps(c0), c01 = _mm512_loadu_ps(c0 + 16);
    __m512 c10 = _mm512_loadu_ps(c1), c11 = _mm512_loadu_ps(c1 + 16);
    __m512 c20 = _mm512_loadu_ps(c2), c21 = _mm512_loadu_ps(c2 + 16);
    __m512 c30 = _mm512_loadu_ps(c3), c31 = _mm512_loadu_ps(c3 + 16);

    for (std::size_t k = 0; k < k_len; ++k) {
        // Load two ZMM vectors of B row k (32 consecutive floats).
        const __m512 b0 = _mm512_loadu_ps(b + k * ldb);
        const __m512 b1 = _mm512_loadu_ps(b + k * ldb + 16);

        // Broadcast A(i+r, k) — _mm512_set1_ps replicates one scalar to all 16 lanes.
        // On Skylake/ICL this compiles to vpbroadcastd + vfmadd or the embedded-broadcast form.
        const __m512 a0 = _mm512_set1_ps(a[0 * lda + k]);
        const __m512 a1 = _mm512_set1_ps(a[1 * lda + k]);
        const __m512 a2 = _mm512_set1_ps(a[2 * lda + k]);
        const __m512 a3 = _mm512_set1_ps(a[3 * lda + k]);

        // 8 FMA instructions — fills both FMA execution ports each cycle.
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c11 = _mm512_fmadd_ps(a1, b1, c11);
        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c21 = _mm512_fmadd_ps(a2, b1, c21);
        c30 = _mm512_fmadd_ps(a3, b0, c30);
        c31 = _mm512_fmadd_ps(a3, b1, c31);
    }

    // Store 4×32 C tile back.
    _mm512_storeu_ps(c0, c00);
    _mm512_storeu_ps(c0 + 16, c01);
    _mm512_storeu_ps(c1, c10);
    _mm512_storeu_ps(c1 + 16, c11);
    _mm512_storeu_ps(c2, c20);
    _mm512_storeu_ps(c2 + 16, c21);
    _mm512_storeu_ps(c3, c30);
    _mm512_storeu_ps(c3 + 16, c31);
}

/**
 * @brief AVX-512 f64 micro-kernel: C[i..i+3][j..j+15] += A[i..i+3][k_blk..k_end) × B[..][j..)
 *
 * Same structure as the f32 kernel, with 8 f64 per ZMM.
 */
inline void avx512_micro_f64_4x16(const double* __restrict__ a, const double* __restrict__ b,
                                  double* __restrict__ c0, double* __restrict__ c1,
                                  double* __restrict__ c2, double* __restrict__ c3, std::size_t lda,
                                  std::size_t ldb, std::size_t k_len) noexcept {
    __m512d c00 = _mm512_loadu_pd(c0), c01 = _mm512_loadu_pd(c0 + 8);
    __m512d c10 = _mm512_loadu_pd(c1), c11 = _mm512_loadu_pd(c1 + 8);
    __m512d c20 = _mm512_loadu_pd(c2), c21 = _mm512_loadu_pd(c2 + 8);
    __m512d c30 = _mm512_loadu_pd(c3), c31 = _mm512_loadu_pd(c3 + 8);

    for (std::size_t k = 0; k < k_len; ++k) {
        const __m512d b0 = _mm512_loadu_pd(b + k * ldb);
        const __m512d b1 = _mm512_loadu_pd(b + k * ldb + 8);

        const __m512d a0 = _mm512_set1_pd(a[0 * lda + k]);
        const __m512d a1 = _mm512_set1_pd(a[1 * lda + k]);
        const __m512d a2 = _mm512_set1_pd(a[2 * lda + k]);
        const __m512d a3 = _mm512_set1_pd(a[3 * lda + k]);

        c00 = _mm512_fmadd_pd(a0, b0, c00);
        c01 = _mm512_fmadd_pd(a0, b1, c01);
        c10 = _mm512_fmadd_pd(a1, b0, c10);
        c11 = _mm512_fmadd_pd(a1, b1, c11);
        c20 = _mm512_fmadd_pd(a2, b0, c20);
        c21 = _mm512_fmadd_pd(a2, b1, c21);
        c30 = _mm512_fmadd_pd(a3, b0, c30);
        c31 = _mm512_fmadd_pd(a3, b1, c31);
    }

    _mm512_storeu_pd(c0, c00);
    _mm512_storeu_pd(c0 + 8, c01);
    _mm512_storeu_pd(c1, c10);
    _mm512_storeu_pd(c1 + 8, c11);
    _mm512_storeu_pd(c2, c20);
    _mm512_storeu_pd(c2 + 8, c21);
    _mm512_storeu_pd(c3, c30);
    _mm512_storeu_pd(c3 + 8, c31);
}

#endif  // HPC_HAS_AVX512

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
// Kernel 3: gemm_avx512_blocked  —  tiled i → k → j,  512-bit register tile
// ============================================================================

/**
 * @brief AVX-512 GEMM with cache-blocking and 512-bit register-tiled micro-kernel.
 *
 * Counterpart of gemm_avx2_blocked: i-k-j order, 3-level tiling
 * (kAvx512TileM × kAvx512TileK × kAvx512TileN = 64 × 256 × 512), and a
 * 4×32 f32 / 4×16 f64 C tile held in ZMM registers for a whole k-tile, so C
 * is stored once per kAvx512TileK steps instead of every step.
 *
 * Tile footprints (f32): A 64 KB, B 512 KB, C 128 KB. The f64 B panel is
 * 1 MiB — exactly Zen 5's per-core L2, which matches the f64 drop between
 * N=256 and N=512 in docs/benchmarks.md.
 */
#if !HPC_HAS_AVX512
template <typename T>
void gemm_avx512_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // AVX512F not available on this target
#else
template <typename T>
void gemm_avx512_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_avx512_blocked: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    // ZMM SIMD width and micro-kernel step.
    constexpr std::size_t kSimdW   = (sizeof(T) == 4) ? 16 : 8;
    constexpr std::size_t kRegRows = (sizeof(T) == 4) ? kAvx512F32RegRows : kAvx512F64RegRows;
    constexpr std::size_t kRegCols = (sizeof(T) == 4) ? kAvx512F32RegCols : kAvx512F64RegCols;
    constexpr std::size_t kJStep   = kSimdW * kRegCols;  // 32 f32 or 16 f64 per micro-kernel call

    for (std::size_t i_blk = 0; i_blk < M; i_blk += kAvx512TileM) {
        const std::size_t i_end = std::min(i_blk + kAvx512TileM, M);

        for (std::size_t k_blk = 0; k_blk < K; k_blk += kAvx512TileK) {
            const std::size_t k_end = std::min(k_blk + kAvx512TileK, K);
            const std::size_t k_len = k_end - k_blk;

            for (std::size_t j_blk = 0; j_blk < N; j_blk += kAvx512TileN) {
                const std::size_t j_end = std::min(j_blk + kAvx512TileN, N);

                // --- AVX-512 hot path ---
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
                            avx512_micro_f32_4x32(
                                reinterpret_cast<const float*>(a_ptr),
                                reinterpret_cast<const float*>(b_ptr + (j - j_blk)),
                                reinterpret_cast<float*>(c0), reinterpret_cast<float*>(c1),
                                reinterpret_cast<float*>(c2), reinterpret_cast<float*>(c3), lda,
                                ldb, k_len);
                        } else {
                            avx512_micro_f64_4x16(
                                reinterpret_cast<const double*>(a_ptr),
                                reinterpret_cast<const double*>(b_ptr + (j - j_blk)),
                                reinterpret_cast<double*>(c0), reinterpret_cast<double*>(c1),
                                reinterpret_cast<double*>(c2), reinterpret_cast<double*>(c3), lda,
                                ldb, k_len);
                        }
                    }
                    // Scalar j-tail (j not a multiple of kJStep)
                    for (; j < j_end; ++j)
                        for (std::size_t ii = i; ii < i + kRegRows; ++ii) {
                            T acc{};
                            for (std::size_t k = k_blk; k < k_end; ++k)
                                acc += A(ii, k) * B(k, j);
                            C(ii, j) += acc;
                        }
                }
                // Scalar i-tail (M not a multiple of kRegRows)
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
#endif  // HPC_HAS_AVX512

}  // namespace hpc::gemm
