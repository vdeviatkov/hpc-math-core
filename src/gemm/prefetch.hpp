#pragma once

/**
 * @file prefetch.hpp
 * @brief Software-prefetch variant of the scalar blocked kernel.
 *
 * The AVX2, AVX-512 and NEON kernels need no separate variant:
 * gemm_simd_blocked's PfDist parameter (simd.hpp) adds the same hints.
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
 * The distance is the template parameter PfDist. gemm_blocked_prefetch
 * prefetches the A and C rows PfDist rows ahead of the current one (default
 * 4); gemm_simd_blocked prefetches PfDist register-tile strips (PfDist · R
 * rows) ahead. bench_gemm sweeps PfDist in {2, 4, 8, 16} for the scalar,
 * AVX2, AVX-512 and NEON blocked kernels.
 *
 *
 * ============================================================
 *  Design
 * ============================================================
 *
 * gemm_blocked_prefetch repeats gemm_blocked's loop with the hints added.
 */

#include "gemm/blocked.hpp"
#include "gemm/simd.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace hpc::gemm {

/// Default prefetch distance of gemm_blocked_prefetch, in rows.
inline constexpr std::size_t kDefaultPrefetchDist = 4;

using detail::pf;  // software prefetch helper, simd.hpp

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

}  // namespace hpc::gemm

