#pragma once

/**
 * @file naive.hpp
 * @brief Naïve (i-j-k) General Matrix Multiply — GEMM.
 *
 * ============================================================
 *  Algorithm: canonical triple-loop, order i → j → k
 * ============================================================
 *
 *   for i in [0, M):            // row of A and C
 *     for j in [0, N):          // column of B and C
 *       for k in [0, K):        // contraction (inner) dimension
 *         C(i,j) += A(i,k) * B(k,j)
 *
 *
 * Cache behaviour — WHY this is slow
 * ===================================
 *
 * All three matrices are stored in row-major order.  Index arithmetic:
 *
 *   A(i,k) → A.data()[ i*K + k ]   — row i, element k along that row
 *   B(k,j) → B.data()[ k*N + j ]   — row k, element j along that row
 *   C(i,j) → C.data()[ i*N + j ]   — row i, element j along that row
 *
 * Inner-loop access pattern (fixed i, fixed j, k varies):
 *
 *   A(i, k)  :  k advances by 1 → stride-1 (sequential) ✔ cache-friendly
 *   B(k, j)  :  k advances by 1 → stride-N (jumps N doubles per step) ✘ cache-hostile
 *   C(i, j)  :  k does NOT change j/i → same address every iteration ✔ (register)
 *
 *        Memory layout of B (row-major, N=4 example)
 *        ┌───────────────────────────────────┐
 *   row0 │ B(0,0)  B(0,1)  B(0,2)  B(0,3)  │  ← cache line 0
 *   row1 │ B(1,0)  B(1,1)  B(1,2)  B(1,3)  │  ← cache line 1
 *   row2 │ B(2,0)  B(2,1)  B(2,2)  B(2,3)  │  ← cache line 2
 *   row3 │ B(3,0)  B(3,1)  B(3,2)  B(3,3)  │  ← cache line 3
 *        └───────────────────────────────────┘
 *
 * For a fixed column j, the inner k-loop reads B(0,j), B(1,j), B(2,j) …
 * Each read is on a *different* cache line.  For N=1024 doubles (8 bytes each)
 * one cache line holds 8 elements.  The k-loop therefore triggers N/8 = 128
 * cache-line loads per (i,j) pair, and barely reuses any of them because the
 * next (i, j+1) pair starts the same pattern on the adjacent column.
 *
 * Effective reuse distance of B:
 *   Between two accesses to B(k, j) and B(k, j') the loop visits M*K other
 *   elements — easily exceeding L1 (32 KB) and often L2 (256 KB) capacity.
 *   This results in L3 or DRAM traffic on every access to B.
 *
 * Complexity: O(M * N * K) multiply-add operations.
 * Memory traffic (worst case, no cache reuse): O(M*K + M*N*K/8 + M*N) cache lines.
 */

#include "hpc/matrix.hpp"

namespace hpc::gemm {

/**
 * @brief Naïve i-j-k GEMM: C = A × B  (C is overwritten, not accumulated).
 *
 * @param A  Input matrix, M×K
 * @param B  Input matrix, K×N
 * @param C  Output matrix, M×N  (must already be allocated; will be zeroed)
 *
 * @pre  A.cols() == B.rows()
 * @pre  C.rows() == A.rows() && C.cols() == B.cols()
 *
 * This implementation is intentionally unoptimised.  It serves as the
 * reference baseline against which all other kernels are validated and
 * benchmarked.
 */
template <typename T>
void gemm_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();  // == B.rows()
    const std::size_t N = B.cols();

    assert(B.rows() == K && "Inner dimensions must agree");
    assert(C.rows() == M && C.cols() == N && "C must be M×N");

    C.zero();  // Ensure C starts as zero before accumulation.

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            // C(i,j) accumulates a dot product of row i of A with col j of B.
            T acc{};
            for (std::size_t k = 0; k < K; ++k) {
                // A(i,k): stride-1 across k — good spatial locality.
                // B(k,j): stride-N across k — poor spatial locality (column walk).
                acc += A(i, k) * B(k, j);
            }
            C(i, j) = acc;
        }
    }
}

}  // namespace hpc::gemm
