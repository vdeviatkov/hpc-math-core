#pragma once

/**
 * @file blocked.hpp
 * @brief Cache-blocked (tiled) i-k-j GEMM.
 *
 *   for i_blk in [0, M, T):
 *     for k_blk in [0, K, T):
 *       for j_blk in [0, N, T):
 *         for i in i_blk tile:
 *           for k in k_blk tile:
 *             a_ik = A(i, k)
 *             for j in j_blk tile:
 *               C(i, j) += a_ik * B(k, j)
 *
 * gemm_reordered streams whole rows: for large N, row i of C (N elements)
 * is evicted between consecutive k-iterations and every row of B is reread
 * from far away for each i. Tiling bounds the working set of the three
 * inner loops, independent of N:
 *
 *   WS = (T×T for A + T×T for B + T×T for C) × sizeof(T)
 *      = 3 × 64 × 64 × 8 B ≈ 96 KB   for the default T=64 and double
 *
 * which stays in L2 on both benchmark machines, so the B tile is reused
 * from cache across all T rows of the i-tile.
 *
 * The cost is shorter inner loops, which hurts auto-vectorisation: on Zen 5
 * gemm_blocked loses to gemm_reordered at N ≤ 512 and wins at N=4096
 * (docs/benchmarks.md). T is a compile-time parameter; the best value is
 * CPU-specific.
 */

#include "hpc/matrix.hpp"

#include <algorithm>

namespace hpc::gemm {

/// Default tile dimension (elements per side). Override with -DHPC_GEMM_TILE=<N>.
#ifndef HPC_GEMM_TILE
    #define HPC_GEMM_TILE 64
#endif

inline constexpr std::size_t kDefaultTile = HPC_GEMM_TILE;

/**
 * @brief C = A × B with tiled i-k-j loops (C is overwritten).
 *
 * @tparam T     Element type (float or double).
 * @tparam TILE  Tile size (elements per side of the square tile).
 *
 * @param A  Input matrix, M×K
 * @param B  Input matrix, K×N
 * @param C  Output matrix, M×N (must already be allocated)
 *
 * @pre  A.cols() == B.rows()
 * @pre  C.rows() == A.rows() && C.cols() == B.cols()
 */
template <typename T, std::size_t TILE = kDefaultTile>
void gemm_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();  // == B.rows()
    const std::size_t N = B.cols();

    assert(B.rows() == K && "Inner dimensions must agree");
    assert(C.rows() == M && C.cols() == N && "C must be M×N");

    C.zero();

    for (std::size_t i_blk = 0; i_blk < M; i_blk += TILE) {
        const std::size_t i_end = std::min(i_blk + TILE, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += TILE) {
            const std::size_t k_end = std::min(k_blk + TILE, K);
            // The B tile [k_blk:k_end, j_blk:j_end] is reused by every row i
            // of the i-tile while it is still in cache.
            for (std::size_t j_blk = 0; j_blk < N; j_blk += TILE) {
                const std::size_t j_end = std::min(j_blk + TILE, N);
                for (std::size_t i = i_blk; i < i_end; ++i) {
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
