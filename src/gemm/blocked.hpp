#pragma once

/**
 * @file blocked.hpp
 * @brief Cache-blocked (tiled) General Matrix Multiply — GEMM.
 *
 * ============================================================
 *  Algorithm: 3-level loop tiling (i-k-j with tile blocks)
 * ============================================================
 *
 *   for i_blk in [0, M, T_r):
 *     for k_blk in [0, K, T_k):
 *       for j_blk in [0, N, T_c):
 *         // --- micro-kernel: T_r × T_k × T_c tile ---
 *         for i in [i_blk, min(i_blk+T_r, M)):
 *           for k in [k_blk, min(k_blk+T_k, K)):
 *             a_ik = A(i, k)           // scalar in register
 *             for j in [j_blk, min(j_blk+T_c, N)):
 *               C(i, j) += a_ik * B(k, j)
 *
 *
 * WHY the reordered kernel still leaves performance on the table
 * ==============================================================
 *
 * The i-k-j reordered kernel streams B row-k and C row-i sequentially, which
 * achieves 100% cache-line utilisation inside the inner j-loop.  However, for
 * large matrices it still has a subtle inefficiency at the k-loop level:
 *
 *   • After the inner j-loop finishes one pass over C row-i (all N columns),
 *     the k-loop increments k.  For large N, row-i of C may have been partially
 *     or fully evicted from L1 between consecutive k-iterations, forcing a
 *     reload on the next pass.
 *
 *   • Similarly, each row of B (row k) is touched exactly once per (i, k) pair.
 *     For N >> L2 capacity, those rows are cold on every i-iteration.
 *
 * Concrete numbers (double, N=1024):
 *   • C row i    = 1024 × 8 B =   8 KB  (fits in L1, but only just)
 *   • B row k    = 1024 × 8 B =   8 KB  (L1 pressure with C already resident)
 *   • Together   = 16 KB — tight in a 32 KB L1; any other activity evicts them.
 *
 * Loop tiling (blocking) fixes this by constraining the working set to a
 * user-chosen tile that fits comfortably in L1 (or L2), regardless of N.
 *
 *
 * Tile size selection and the working-set equation
 * ================================================
 *
 * Consider a tile of:
 *   T_r rows of A and C  (T_r × T_k and T_r × T_c elements respectively)
 *   T_k × T_c elements of B
 *
 * Working set of the innermost 3 loops (all simultaneously resident):
 *
 *   WS = (T_r × T_k  +  T_k × T_c  +  T_r × T_c) × sizeof(T)
 *
 * For the default tile T_r=T_k=T_c=64 and double (8 B):
 *   WS = (64×64 + 64×64 + 64×64) × 8 = 3 × 4096 × 8 = 98 304 B ≈ 96 KB
 *
 * This fits comfortably in L2 (4096 KiB per core on the benchmark machine)
 * and dramatically reduces L3/DRAM traffic compared to the reordered kernel.
 *
 *
 * Access pattern inside the tile (innermost j-loop, fixed i and k)
 * =================================================================
 *
 *   A(i, k)  — scalar register, free                          ✔
 *   B(k, j)  — stride-1 across j within tile, fully in L1/L2  ✔✔
 *   C(i, j)  — stride-1 across j within tile, fully in L1/L2  ✔✔
 *
 * Because the tile fits in L2, B row k and C row i stay hot across all
 * T_r i-iterations of the tile.  Reuse distance for a cache line of B:
 *
 *   naïve:      M × N  accesses between reuses of the same B line  (always cold)
 *   reordered:  N      accesses between reuses (warm in L2 for small N)
 *   blocked:    T_r    accesses between reuses (always in L1 for T_r ≤ 8)
 *
 *
 * Tile size constants
 * ===================
 *
 * The default tile (64×64) is a reasonable starting point for a 4096 KiB L2.
 * Optimal tile sizes are CPU-specific and should be tuned empirically.
 * Rules of thumb:
 *   • T_k × T_c × sizeof(T) should fit in one L1 way (≈ 2–4 KB)
 *   • T_r × T_c × sizeof(T) (C tile) should also fit in L1
 *   • Total working set ≤ 50–75% of L2 to leave room for prefetch buffers
 *
 * Complexity: O(M × N × K) — identical to naïve and reordered.
 * The improvement is purely in cache-miss rate (constants in the O notation).
 */

#include "hpc/matrix.hpp"

#include <algorithm>

namespace hpc::gemm {

/// Default tile dimension (elements per side).  Override at compile time with
/// -DHPC_GEMM_TILE=<N> if you want to experiment without rewriting code.
#ifndef HPC_GEMM_TILE
    #define HPC_GEMM_TILE 64
#endif

inline constexpr std::size_t kDefaultTile = HPC_GEMM_TILE;

/**
 * @brief Cache-blocked i-k-j GEMM: C = A × B  (C is overwritten).
 *
 * @tparam T     Element type (float or double).
 * @tparam TILE  Tile size (elements per side of the square tile).
 *               Defaults to kDefaultTile (64).  Pass a smaller value (e.g. 32)
 *               for L1-resident tiles on CPUs with small L2.
 *
 * @param A  Input matrix, M×K
 * @param B  Input matrix, K×N
 * @param C  Output matrix, M×N  (must already be allocated; will be zeroed)
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

    // ------------------------------------------------------------------
    // Outer tile loops — step through the matrix in TILE-sized chunks.
    // The three blocking dimensions correspond to:
    //   i_blk : tile rows of A and C
    //   k_blk : tile columns of A / tile rows of B  (contraction axis)
    //   j_blk : tile columns of B and C
    // ------------------------------------------------------------------
    for (std::size_t i_blk = 0; i_blk < M; i_blk += TILE) {
        const std::size_t i_end = std::min(i_blk + TILE, M);  // handle edge tiles

        for (std::size_t k_blk = 0; k_blk < K; k_blk += TILE) {
            const std::size_t k_end = std::min(k_blk + TILE, K);

            // At this point the tile of B [ k_blk:k_end , j_blk:j_end ] will
            // be accessed T_r times (once per i in the tile).  By placing the
            // j_blk loop *inside* k_blk we ensure the B tile stays in L1/L2
            // for all T_r passes.
            for (std::size_t j_blk = 0; j_blk < N; j_blk += TILE) {
                const std::size_t j_end = std::min(j_blk + TILE, N);

                // ----------------------------------------------------------
                // Micro-kernel: compute the (i_blk, j_blk) output tile.
                // Working set: T_r×T_k (A) + T_k×T_c (B) + T_r×T_c (C)
                //              ≈ 3 × TILE² × sizeof(T) bytes.
                // For TILE=64, T=double: ≈ 96 KB — fits in L2.
                // ----------------------------------------------------------
                for (std::size_t i = i_blk; i < i_end; ++i) {
                    for (std::size_t k = k_blk; k < k_end; ++k) {
                        // Hoist A(i,k) — invariant across the j tile.
                        const T a_ik = A(i, k);

                        // Stream B row k and C row i sequentially within the
                        // tile.  Both fit in L1 (TILE × 8 B = 512 B each).
                        for (std::size_t j = j_blk; j < j_end; ++j) {
                            C(i, j) += a_ik * B(k, j);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace hpc::gemm
