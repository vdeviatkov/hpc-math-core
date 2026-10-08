#pragma once

/**
 * @file sve.hpp
 * @brief ARM SVE counterparts of the three AVX2/NEON kernels.
 *
 * Unlike NEON (128-bit), AVX2 (256-bit) and AVX-512 (512-bit), SVE's vector
 * length is implementation-defined (128–2048 bits) and read at runtime with
 * svcntw() (f32 lanes) / svcntd() (f64 lanes). Loop steps are computed from
 * it, so one binary runs on 128-bit (Neoverse N2/V2, Graviton4), 256-bit
 * (Neoverse V1, Graviton3) and 512-bit (A64FX) hardware.
 *
 * Every load, store and FMA takes a predicate. svwhilelt_b32(j, N) is
 * active only for lanes with j + lane < N, so the last partial vector is
 * handled by the same loop body — there is no scalar tail loop.
 *
 *  gemm_sve_naive      i → j → k. B column j is gathered into a buffer and
 *                      loaded as a vector; svaddv reduces the accumulator.
 *  gemm_sve_reordered  i → k → j: broadcast A(i,k), FMA against B and C rows.
 *  gemm_sve_blocked    tiled i → k → j with a 4-row × 2-vector C register
 *                      tile whose width follows the hardware vector length.
 *
 * Not measured: no SVE hardware was available. Apple Silicon has no
 * non-streaming SVE (see sme.hpp).
 *
 * HPC_HAS_SVE (hpc/isa.hpp) follows __ARM_FEATURE_SVE, set by e.g.
 * -march=armv8.2-a+sve, -mcpu=neoverse-v1, or -march=native on SVE hardware.
 * Where it is 0 the kernels are declared `= delete`.
 */

#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"

#if HPC_HAS_SVE
    #include <arm_sve.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace hpc::gemm {

// ============================================================================
// SVE tile constants
// ============================================================================
// The j-step is svcntw()/svcntd() × kSveRegCols, read at runtime; only the
// outer blocking tile sizes are compile-time constants.

inline constexpr std::size_t kSveTileM = 64;   // outer i-tile (rows of A / C)
inline constexpr std::size_t kSveTileK = 256;  // outer k-tile (contraction width)
inline constexpr std::size_t kSveTileN = 512;  // outer j-tile (cols of B / C)
                                               // 512 × 4B = 2 KB per B row-tile;
                                               // on 256-bit SVE (8 f32/vec) this is
                                               // 64 vectors — well within L1 TLB.

// Number of C rows accumulated simultaneously in the blocked micro-kernel.
// 4 rows × 2 vectors; at 256-bit VL that is 4 × 16 f32 (256 B), the same
// tile as AVX2 blocked.
inline constexpr std::size_t kSveRegRows = 4;
// Number of SVE vectors per C row in the micro-kernel.
// 2 vectors × VL elements = 2*svcntw() f32 or 2*svcntd() f64 per row.
inline constexpr std::size_t kSveRegCols = 2;

// ============================================================================
// SVE micro-kernels (used only by gemm_sve_blocked)
// ============================================================================

#if HPC_HAS_SVE

/**
 * @brief SVE f32 micro-kernel: C[i..i+3][j..j+2*VL) += A[i..i+3][k_blk..k_end) × B[..][j..)
 *
 * Register tile: 4 rows × 2 SVE vectors = 4 × (2 * svcntw()) f32.
 *
 * 128-bit SVE: 4 × 8 f32 (128 B); 256-bit: 4 × 16 (256 B); 512-bit: 4 × 32 (512 B).
 *
 * pg0/pg1 are all-true except in the j-tail, where they come from
 * svwhilelt_b32. svmla_f32_x leaves inactive lanes undefined; that is safe
 * because the final stores use the same predicates, so those lanes are
 * never written to C.
 *
 * @param a      A(i, k_blk) — row stride lda
 * @param b0     B(k_blk, j) — first SVE-width block
 * @param b1     B(k_blk, j + svcntw()) — second SVE-width block
 * @param c0..c3 C(i+0..3, j)
 * @param pg0/1  Predicates for the two B/C vector blocks (all-true or tail)
 */
inline void sve_micro_f32_4x2v(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ c0, float* __restrict__ c1,
                               float* __restrict__ c2, float* __restrict__ c3, std::size_t lda,
                               std::size_t ldb, std::size_t k_len, svbool_t pg0,
                               svbool_t pg1) noexcept {
    const std::uint64_t vl = svcntw();  // elements per SVE vector (runtime)

    // Load 4×2 SVE-vector C tile.
    svfloat32_t c00 = svld1_f32(pg0, c0), c01 = svld1_f32(pg1, c0 + vl);
    svfloat32_t c10 = svld1_f32(pg0, c1), c11 = svld1_f32(pg1, c1 + vl);
    svfloat32_t c20 = svld1_f32(pg0, c2), c21 = svld1_f32(pg1, c2 + vl);
    svfloat32_t c30 = svld1_f32(pg0, c3), c31 = svld1_f32(pg1, c3 + vl);

    for (std::size_t k = 0; k < k_len; ++k) {
        // Load 2 SVE vectors of B row k.
        const svfloat32_t b0 = svld1_f32(pg0, b + k * ldb);
        const svfloat32_t b1 = svld1_f32(pg1, b + k * ldb + vl);

        // Broadcast A(i+r, k) to all lanes — single scalar → full vector.
        const svfloat32_t a0 = svdup_n_f32(a[0 * lda + k]);
        const svfloat32_t a1 = svdup_n_f32(a[1 * lda + k]);
        const svfloat32_t a2 = svdup_n_f32(a[2 * lda + k]);
        const svfloat32_t a3 = svdup_n_f32(a[3 * lda + k]);

        // 8 predicated FMA instructions.
        c00 = svmla_f32_x(pg0, c00, a0, b0);
        c01 = svmla_f32_x(pg1, c01, a0, b1);
        c10 = svmla_f32_x(pg0, c10, a1, b0);
        c11 = svmla_f32_x(pg1, c11, a1, b1);
        c20 = svmla_f32_x(pg0, c20, a2, b0);
        c21 = svmla_f32_x(pg1, c21, a2, b1);
        c30 = svmla_f32_x(pg0, c30, a3, b0);
        c31 = svmla_f32_x(pg1, c31, a3, b1);
    }

    // Store 4×2 SVE-vector C tile back.
    svst1_f32(pg0, c0, c00);
    svst1_f32(pg1, c0 + vl, c01);
    svst1_f32(pg0, c1, c10);
    svst1_f32(pg1, c1 + vl, c11);
    svst1_f32(pg0, c2, c20);
    svst1_f32(pg1, c2 + vl, c21);
    svst1_f32(pg0, c3, c30);
    svst1_f32(pg1, c3 + vl, c31);
}

/**
 * @brief SVE f64 micro-kernel: C[i..i+3][j..j+2*VL) += A[i..i+3][k_blk..k_end) × B[..][j..)
 *
 * Register tile: 4 rows × 2 SVE vectors = 4 × (2 * svcntd()) f64.
 * Same structure as f32 but using f64 intrinsics and svcntd() for VL.
 */
inline void sve_micro_f64_4x2v(const double* __restrict__ a, const double* __restrict__ b,
                               double* __restrict__ c0, double* __restrict__ c1,
                               double* __restrict__ c2, double* __restrict__ c3, std::size_t lda,
                               std::size_t ldb, std::size_t k_len, svbool_t pg0,
                               svbool_t pg1) noexcept {
    const std::uint64_t vl = svcntd();

    svfloat64_t c00 = svld1_f64(pg0, c0), c01 = svld1_f64(pg1, c0 + vl);
    svfloat64_t c10 = svld1_f64(pg0, c1), c11 = svld1_f64(pg1, c1 + vl);
    svfloat64_t c20 = svld1_f64(pg0, c2), c21 = svld1_f64(pg1, c2 + vl);
    svfloat64_t c30 = svld1_f64(pg0, c3), c31 = svld1_f64(pg1, c3 + vl);

    for (std::size_t k = 0; k < k_len; ++k) {
        const svfloat64_t b0 = svld1_f64(pg0, b + k * ldb);
        const svfloat64_t b1 = svld1_f64(pg1, b + k * ldb + vl);

        const svfloat64_t a0 = svdup_n_f64(a[0 * lda + k]);
        const svfloat64_t a1 = svdup_n_f64(a[1 * lda + k]);
        const svfloat64_t a2 = svdup_n_f64(a[2 * lda + k]);
        const svfloat64_t a3 = svdup_n_f64(a[3 * lda + k]);

        c00 = svmla_f64_x(pg0, c00, a0, b0);
        c01 = svmla_f64_x(pg1, c01, a0, b1);
        c10 = svmla_f64_x(pg0, c10, a1, b0);
        c11 = svmla_f64_x(pg1, c11, a1, b1);
        c20 = svmla_f64_x(pg0, c20, a2, b0);
        c21 = svmla_f64_x(pg1, c21, a2, b1);
        c30 = svmla_f64_x(pg0, c30, a3, b0);
        c31 = svmla_f64_x(pg1, c31, a3, b1);
    }

    svst1_f64(pg0, c0, c00);
    svst1_f64(pg1, c0 + vl, c01);
    svst1_f64(pg0, c1, c10);
    svst1_f64(pg1, c1 + vl, c11);
    svst1_f64(pg0, c2, c20);
    svst1_f64(pg1, c2 + vl, c21);
    svst1_f64(pg0, c3, c30);
    svst1_f64(pg1, c3 + vl, c31);
}

#endif  // HPC_HAS_SVE

// ============================================================================
// Kernel 1: gemm_sve_naive  —  i → j → k,  SVE on the k-loop
// ============================================================================

/**
 * @brief SVE GEMM with naive i-j-k loop order.
 *
 * Counterpart of gemm_neon_naive / gemm_avx2_naive with a runtime k-step of
 * svcntw()/svcntd(). B(k..k+VL-1, j) is gathered with stride ldb into a
 * std::vector buffer and loaded with svld1 (svld1_gather_index would touch
 * the same cache lines). svaddv reduces the accumulator at the end.
 */
#if !HPC_HAS_SVE
template <typename T>
void gemm_sve_naive(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_FEATURE_SVE not available on this target
#else
template <typename T>
void gemm_sve_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sve_naive: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    if constexpr (sizeof(T) == 4) {
        const std::size_t vl = svcntw();  // elements per SVE vector (runtime)
        // Buffer for one gathered B column segment (std::vector: VLAs are not
        // standard C++).
        std::vector<float> b_col(vl);
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                svfloat32_t acc = svdup_n_f32(0.f);
                std::size_t k   = 0;
                for (; k + vl <= K; k += vl) {
                    // A(i, k..k+vl): sequential load — cache friendly.
                    const svfloat32_t a_vec = svld1_f32(svptrue_b32(), A.data() + i * lda + k);
                    // Gather B column j: fill buffer element by element,
                    // then load as a contiguous SVE vector.
                    for (std::size_t lane = 0; lane < vl; ++lane)
                        b_col[lane] = B.data()[(k + lane) * ldb + j];
                    const svfloat32_t b_vec = svld1_f32(svptrue_b32(), b_col.data());
                    acc = svmla_f32_x(svptrue_b32(), acc, a_vec, b_vec);
                }
                // Horizontal reduce the SVE accumulator to a scalar.
                float s = svaddv_f32(svptrue_b32(), acc);
                // Scalar tail for k % vl elements.
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            }
        }
    } else {
        const std::size_t vl = svcntd();
        std::vector<double> b_col(vl);
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                svfloat64_t acc = svdup_n_f64(0.0);
                std::size_t k   = 0;
                for (; k + vl <= K; k += vl) {
                    const svfloat64_t a_vec = svld1_f64(svptrue_b64(), A.data() + i * lda + k);
                    for (std::size_t lane = 0; lane < vl; ++lane)
                        b_col[lane] = B.data()[(k + lane) * ldb + j];
                    const svfloat64_t b_vec = svld1_f64(svptrue_b64(), b_col.data());
                    acc = svmla_f64_x(svptrue_b64(), acc, a_vec, b_vec);
                }
                double s = svaddv_f64(svptrue_b64(), acc);
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            }
        }

    }
}
#endif  // HPC_HAS_SVE

// ============================================================================
// Kernel 2: gemm_sve_reordered  —  i → k → j,  SVE on the j-loop (VLA)
// ============================================================================

/**
 * @brief SVE GEMM with cache-friendly i-k-j loop order — vector-length agnostic.
 *
 * Counterpart of gemm_neon_reordered / gemm_avx2_reordered. The j-step is
 * svcntw()/svcntd(); the last partial vector uses an svwhilelt_b32(j, N)
 * predicate, so one loop covers any N.
 *
 * Not measured (no SVE hardware available). On every measured family the
 * explicit-SIMD reordered kernel performs about the same as the
 * auto-vectorised scalar one.
 */
#if !HPC_HAS_SVE
template <typename T>
void gemm_sve_reordered(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_FEATURE_SVE not available on this target
#else
template <typename T>
void gemm_sve_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sve_reordered: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    if constexpr (sizeof(T) == 4) {
        const std::uint64_t vl = svcntw();
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t k = 0; k < K; ++k) {
                const svfloat32_t a_broad = svdup_n_f32(A.data()[i * lda + k]);
                const float* b_row        = B.data() + k * ldb;
                float* c_row              = C.data() + i * ldc;

                // VLA j-loop: step = vl, last iteration uses tail predicate.
                std::uint64_t j = 0;
                for (svbool_t pg = svwhilelt_b32(j, (std::uint64_t)N);
                     svptest_any(svptrue_b32(), pg);
                     j += vl, pg = svwhilelt_b32(j, (std::uint64_t)N)) {
                    const svfloat32_t b_vec = svld1_f32(pg, b_row + j);
                    const svfloat32_t c_vec = svld1_f32(pg, c_row + j);
                    svst1_f32(pg, c_row + j, svmla_f32_x(pg, c_vec, a_broad, b_vec));
                }
            }
        }
    } else {
        const std::uint64_t vl = svcntd();
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t k = 0; k < K; ++k) {
                const svfloat64_t a_broad = svdup_n_f64(A.data()[i * lda + k]);
                const double* b_row       = B.data() + k * ldb;
                double* c_row             = C.data() + i * ldc;

                std::uint64_t j = 0;
                for (svbool_t pg = svwhilelt_b64(j, (std::uint64_t)N);
                     svptest_any(svptrue_b64(), pg);
                     j += vl, pg = svwhilelt_b64(j, (std::uint64_t)N)) {
                    const svfloat64_t b_vec = svld1_f64(pg, b_row + j);
                    const svfloat64_t c_vec = svld1_f64(pg, c_row + j);
                    svst1_f64(pg, c_row + j, svmla_f64_x(pg, c_vec, a_broad, b_vec));
                }
            }
        }
    }
}
#endif  // HPC_HAS_SVE

// ============================================================================
// Kernel 3: gemm_sve_blocked  —  tiled i → k → j,  VLA register tile
// ============================================================================

/**
 * @brief SVE GEMM with cache-blocking and VLA register-tiled micro-kernel.
 *
 * Counterpart of gemm_neon_blocked / gemm_avx2_blocked: i-k-j order, 3-level
 * tiling (kSveTileM × kSveTileK × kSveTileN), and a C tile of 4 rows ×
 * kSveRegCols vectors held in registers for a whole k-tile. Its width,
 * kSveRegCols × svcntw()/svcntd(), follows the hardware vector length; the
 * j-tail is handled by predicates inside the micro-kernel.
 *
 * Tile footprints (f32): A 64 KB, B 512 KB (L2-resident on Neoverse V1's
 * 1 MB L2), C 128 KB streamed.
 */
#if !HPC_HAS_SVE
template <typename T>
void gemm_sve_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_FEATURE_SVE not available on this target
#else
template <typename T>
void gemm_sve_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sve_blocked: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    // VL is queried once here; it is guaranteed constant for the process lifetime.
    const std::size_t vl     = (sizeof(T) == 4) ? svcntw() : svcntd();
    const std::size_t kJStep = kSveRegCols * vl;  // j-elements per micro-kernel call

    for (std::size_t i_blk = 0; i_blk < M; i_blk += kSveTileM) {
        const std::size_t i_end = std::min(i_blk + kSveTileM, M);

        for (std::size_t k_blk = 0; k_blk < K; k_blk += kSveTileK) {
            const std::size_t k_end = std::min(k_blk + kSveTileK, K);
            const std::size_t k_len = k_end - k_blk;

            for (std::size_t j_blk = 0; j_blk < N; j_blk += kSveTileN) {
                const std::size_t j_end = std::min(j_blk + kSveTileN, N);

                // --- SVE hot path: kSveRegRows rows × kJStep cols per call ---
                std::size_t i = i_blk;
                for (; i + kSveRegRows <= i_end; i += kSveRegRows) {
                    const T* a_ptr = A.data() + i * lda + k_blk;
                    const T* b_ptr = B.data() + k_blk * ldb + j_blk;

                    std::size_t j = j_blk;
                    for (; j < j_end; j += kJStep) {
                        // Compute remaining j elements to build predicates.
                        const std::size_t rem0 = (j_end > j) ? j_end - j : 0;
                        const std::size_t rem1 = (rem0 > vl) ? rem0 - vl : 0;

                        T* c0 = C.data() + (i + 0) * ldc + j;
                        T* c1 = C.data() + (i + 1) * ldc + j;
                        T* c2 = C.data() + (i + 2) * ldc + j;
                        T* c3 = C.data() + (i + 3) * ldc + j;

                        if constexpr (sizeof(T) == 4) {
                            // pg0: first VL elements — full if rem0 >= vl, else tail.
                            const svbool_t pg0 = (rem0 >= vl) ? svptrue_b32()
                                                              : svwhilelt_b32((std::uint64_t)0,
                                                                              (std::uint64_t)rem0);
                            // pg1: second VL elements — full if rem1 >= vl, else tail.
                            const svbool_t pg1 = (rem1 >= vl) ? svptrue_b32()
                                                              : svwhilelt_b32((std::uint64_t)0,
                                                                              (std::uint64_t)rem1);

                            // Only call the micro-kernel if the first block has work.
                            if (svptest_any(svptrue_b32(), pg0)) {
                                sve_micro_f32_4x2v(
                                    reinterpret_cast<const float*>(a_ptr),
                                    reinterpret_cast<const float*>(b_ptr + (j - j_blk)),
                                    reinterpret_cast<float*>(c0), reinterpret_cast<float*>(c1),
                                    reinterpret_cast<float*>(c2), reinterpret_cast<float*>(c3), lda,
                                    ldb, k_len, pg0, pg1);
                            }
                        } else {
                            const svbool_t pg0 = (rem0 >= vl) ? svptrue_b64()
                                                              : svwhilelt_b64((std::uint64_t)0,
                                                                              (std::uint64_t)rem0);
                            const svbool_t pg1 = (rem1 >= vl) ? svptrue_b64()
                                                              : svwhilelt_b64((std::uint64_t)0,
                                                                              (std::uint64_t)rem1);

                            if (svptest_any(svptrue_b64(), pg0)) {
                                sve_micro_f64_4x2v(
                                    reinterpret_cast<const double*>(a_ptr),
                                    reinterpret_cast<const double*>(b_ptr + (j - j_blk)),
                                    reinterpret_cast<double*>(c0), reinterpret_cast<double*>(c1),
                                    reinterpret_cast<double*>(c2), reinterpret_cast<double*>(c3),
                                    lda, ldb, k_len, pg0, pg1);
                            }
                        }
                    }
                }

                // Scalar i-tail (M not a multiple of kSveRegRows).
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
#endif  // HPC_HAS_SVE

}  // namespace hpc::gemm
