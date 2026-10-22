#pragma once

/**
 * @file prefetch.hpp
 * @brief Software-prefetch variants of every blocked GEMM kernel.
 *
 * Hardware prefetchers follow sequential streams; these hints target the
 * three streams they might not predict:
 *
 *   1. The next A rows — K*sizeof(T) bytes apart.
 *   2. The next B k-tile — a new, cold address at each k-tile boundary.
 *   3. The next C rows — a write to a C row no longer in L1 first has to
 *      load the line.
 *
 * Measured, the hints are close to neutral on both Apple M4 Max and Zen 5:
 * those hardware prefetchers already follow these streams (see
 * docs/benchmarks.md, "Prefetch distance sweep").
 *
 *
 * ============================================================
 *  __builtin_prefetch(addr, rw, locality)
 * ============================================================
 *
 *   rw        0 = prefetch for read       (PREFETCHT* / PRFM PLDL*KEEP)
 *             1 = prefetch for read+write  (PREFETCHW  / PRFM PSTL*KEEP)
 *   locality  2 = moderate temporal locality (A/B)
 *             3 = high temporal locality (C)
 * These are target-dependent hints, not guaranteed cache placement.
 *
 *
 * ============================================================
 *  Prefetch distance
 * ============================================================
 *
 * Distance D = register-tile strips ahead of the current one (D · R rows).
 * D is exposed as template parameter PfDist (default = 4).
 * The benchmark sweeps PfDist in {2, 4, 8, 16} per ISA family.
 *
 *
 * ============================================================
 *  Design
 * ============================================================
 *
 * The AVX2, AVX-512 and NEON variants are their base kernel's driver
 * (blocked_driver.hpp) with PfDist > 0, at the base kernel's default
 * register tile. The scalar and SVE variants repeat their base kernel's
 * loop with the hints added. Like its base kernel, each is declared
 * `= delete` on an unsupported ISA.
 */

#include "gemm/avx2.hpp"
#include "gemm/avx512.hpp"
#include "gemm/blocked.hpp"
#include "gemm/neon.hpp"
#include "gemm/sve.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace hpc::gemm {

/// Default prefetch distance in micro-kernel rows.
inline constexpr std::size_t kDefaultPrefetchDist = 4;

using detail::pf;  // software prefetch helper, blocked_driver.hpp

// ============================================================================
// 1.  Scalar blocked + prefetch
// ============================================================================
template <typename T, std::size_t PfDist = kDefaultPrefetchDist,
          std::size_t Tile = kDefaultTile>
void gemm_blocked_prefetch(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(Tile > 0, "GEMM tile size must be positive");
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    for (std::size_t i_blk = 0; i_blk < M; i_blk += Tile) {
        const std::size_t i_end = std::min(i_blk + Tile, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += Tile) {
            const std::size_t k_end = std::min(k_blk + Tile, K);
            // [PF-B] prefetch next B k-tile into L2
            if (k_blk + Tile < K)
                pf<0, 2>(B.data() + (k_blk + Tile) * ldb);
            for (std::size_t j_blk = 0; j_blk < N; j_blk += Tile) {
                const std::size_t j_end = std::min(j_blk + Tile, N);
                for (std::size_t i = i_blk; i < i_end; ++i) {
                    // [PF-A] next A row into L2
                    if (i + PfDist < i_end)
                        pf<0, 2>(A.data() + (i + PfDist) * lda + k_blk);
                    // [PF-C] next C write row into L1
                    if (i + PfDist < i_end)
                        pf<1, 3>(C.data() + (i + PfDist) * ldc + j_blk);
                    for (std::size_t k = k_blk; k < k_end; ++k) {
                        const T a_ik = A(i, k);
                        for (std::size_t j = j_blk; j < j_end; ++j)
                            C(i, j) += a_ik * B(k, j);
                    }
                }
            }
        }
    }
}

// ============================================================================
// 2.  AVX2 blocked + prefetch
// ============================================================================
#if !HPC_HAS_AVX2
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_avx2_blocked_prefetch(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // AVX2 not available on this target
#else
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_avx2_blocked_prefetch(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    detail::blocked_gemm<Avx2Micro, 6, 2, kAvx2TileM, kAvx2TileK, kAvx2TileN<T>, PfDist>(A, B, C);
}
#endif  // HPC_HAS_AVX2

// ============================================================================
// 3.  AVX-512 blocked + prefetch
// ============================================================================
#if !HPC_HAS_AVX512
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_avx512_blocked_prefetch(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // AVX512F not available on this target
#else
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_avx512_blocked_prefetch(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    detail::blocked_gemm<Avx512Micro, 8, 3, kAvx512TileM, kAvx512TileK, kAvx512TileN<T>, PfDist>(A, B, C);
}
#endif  // HPC_HAS_AVX512

// ============================================================================
// 4.  NEON blocked + prefetch
// ============================================================================
#if !HPC_HAS_NEON
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_neon_blocked_prefetch(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_NEON not available on this target
#else
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_neon_blocked_prefetch(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    detail::blocked_gemm<NeonMicro, 6, 4, kNeonTileM, kNeonTileK, kNeonTileN, PfDist>(A, B, C);
}
#endif  // HPC_HAS_NEON

// ============================================================================
// 5.  SVE blocked + prefetch
// ============================================================================
#if !HPC_HAS_SVE
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_sve_blocked_prefetch(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_FEATURE_SVE not available on this target
#else
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_sve_blocked_prefetch(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K, ldb = N, ldc = N;
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    const std::size_t vl     = (sizeof(T) == 4) ? svcntw() : svcntd();
    const std::size_t kJStep = kSveRegCols * vl;

    for (std::size_t i_blk = 0; i_blk < M; i_blk += kSveTileM) {
        const std::size_t i_end = std::min(i_blk + kSveTileM, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += kSveTileK) {
            const std::size_t k_end = std::min(k_blk + kSveTileK, K);
            const std::size_t k_len = k_end - k_blk;
            if (k_blk + kSveTileK < K)
                pf<0, 2>(B.data() + (k_blk + kSveTileK) * ldb);  // [PF-B]
            for (std::size_t j_blk = 0; j_blk < N; j_blk += kSveTileN) {
                const std::size_t j_end = std::min(j_blk + kSveTileN, N);
                std::size_t i = i_blk;
                for (; i + kSveRegRows <= i_end; i += kSveRegRows) {
                    if (i + kSveRegRows * PfDist < i_end) {
                        pf<0, 2>(A.data() + (i + kSveRegRows * PfDist) * lda + k_blk);  // [PF-A]
                        pf<1, 3>(C.data() + (i + kSveRegRows * PfDist) * ldc + j_blk);  // [PF-C]
                    }
                    const T* a_ptr = A.data() + i * lda + k_blk;
                    const T* b_ptr = B.data() + k_blk * ldb + j_blk;
                    for (std::size_t j = j_blk; j < j_end; j += kJStep) {
                        const std::size_t rem  = (j_end > j) ? j_end - j : 0;
                        const std::size_t rem1 = (rem > vl) ? rem - vl : 0;
                        T* c0 = C.data() + (i + 0) * ldc + j;
                        T* c1 = C.data() + (i + 1) * ldc + j;
                        T* c2 = C.data() + (i + 2) * ldc + j;
                        T* c3 = C.data() + (i + 3) * ldc + j;
                        if constexpr (sizeof(T) == 4) {
                            const svbool_t pg0 = (rem >= vl) ? svptrue_b32()
                                : svwhilelt_b32((uint64_t)0, (uint64_t)rem);
                            const svbool_t pg1 = (rem1 >= vl) ? svptrue_b32()
                                : svwhilelt_b32((uint64_t)0, (uint64_t)rem1);
                            if (svptest_any(svptrue_b32(), pg0))
                                sve_micro_f32_4x2v(reinterpret_cast<const float*>(a_ptr),
                                    reinterpret_cast<const float*>(b_ptr + (j - j_blk)),
                                    reinterpret_cast<float*>(c0), reinterpret_cast<float*>(c1),
                                    reinterpret_cast<float*>(c2), reinterpret_cast<float*>(c3),
                                    lda, ldb, k_len, pg0, pg1);
                        } else {
                            const svbool_t pg0 = (rem >= vl) ? svptrue_b64()
                                : svwhilelt_b64((uint64_t)0, (uint64_t)rem);
                            const svbool_t pg1 = (rem1 >= vl) ? svptrue_b64()
                                : svwhilelt_b64((uint64_t)0, (uint64_t)rem1);
                            if (svptest_any(svptrue_b64(), pg0))
                                sve_micro_f64_4x2v(reinterpret_cast<const double*>(a_ptr),
                                    reinterpret_cast<const double*>(b_ptr + (j - j_blk)),
                                    reinterpret_cast<double*>(c0), reinterpret_cast<double*>(c1),
                                    reinterpret_cast<double*>(c2), reinterpret_cast<double*>(c3),
                                    lda, ldb, k_len, pg0, pg1);
                        }
                    }
                }
                for (; i < i_end; ++i)
                    for (std::size_t k = k_blk; k < k_end; ++k) {
                        const T a_ik = A(i, k);
                        for (std::size_t j = j_blk; j < j_end; ++j) C(i, j) += a_ik * B(k, j);
                    }
            }
        }
    }
}
#endif  // HPC_HAS_SVE

}  // namespace hpc::gemm

