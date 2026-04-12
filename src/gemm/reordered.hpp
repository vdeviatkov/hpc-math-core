#pragma once

/**
 * @file reordered.hpp
 * @brief Cache-friendly (i-k-j) General Matrix Multiply — GEMM.
 *
 * ============================================================
 *  Algorithm: loop reorder i → k → j
 * ============================================================
 *
 *   for i in [0, M):            // row of A and C
 *     for k in [0, K):          // contraction dimension
 *       a_ik = A(i, k)          // scalar hoisted out of the j-loop
 *       for j in [0, N):        // column of B and C
 *         C(i,j) += a_ik * B(k,j)
 *
 *
 * Cache behaviour — WHY this is faster
 * ======================================
 *
 * The *only* change compared to gemm_naive is the swap of the j and k loops.
 * This has a dramatic effect on the memory access pattern inside the hottest
 * (innermost) loop.
 *
 * Inner-loop access pattern (fixed i, fixed k, j varies):
 *
 *   A(i, k)  :  k and i are fixed → INVARIANT, hoisted to register ✔
 *   B(k, j)  :  j advances by 1  → stride-1, row-k of B sequentially ✔✔
 *   C(i, j)  :  j advances by 1  → stride-1, row-i of C sequentially ✔✔
 *
 *        Memory layout of B (row-major, N=4 example)
 *        ┌───────────────────────────────────┐
 *   row0 │ B(0,0)  B(0,1)  B(0,2)  B(0,3)  │  ← inner loop reads entire row k
 *   row1 │ B(1,0)  B(1,1)  B(1,2)  B(1,3)  │    sequentially: optimal prefetch
 *   row2 │ B(2,0)  B(2,1)  B(2,2)  B(2,3)  │
 *   row3 │ B(3,0)  B(3,1)  B(3,2)  B(3,3)  │
 *        └───────────────────────────────────┘
 *
 * B reuse: once a cache line of row k is loaded, *all 8 doubles on that line*
 * are consumed before the line is evicted. Compare this to the naïve version
 * where each cache-line load yielded only 1 useful element.
 * Utilisation ratio: 8/8 = 100 % vs. 1/8 = 12.5 %.
 *
 * C reuse: similarly, row i of C is streamed left-to-right, so each cache line
 * is both loaded *and* written back in one pass — write-combining hardware can
 * coalesce the stores efficiently.
 *
 * A(i,k) hoisting: the compiler can keep the scalar a_ik in a register for the
 * entire inner j-loop, eliminating any memory traffic for A inside the hottest
 * loop body. Without the explicit hoist, a less aggressive compiler might
 * re-read A(i,k) from L1 on every j iteration.
 *
 * Effective L1 working set (inner j-loop):
 *   • 1 cache line for the current segment of B row k   (64 bytes)
 *   • 1 cache line for the current segment of C row i   (64 bytes)
 *   Total ≈ 128 bytes — easily fits in L1 (32 KB).
 *
 * Expected speedup: 4–8× over the naïve kernel for large N on a modern CPU,
 * depending on CPU model, cache sizes and compiler optimisation level.
 *
 * Complexity: O(M * N * K) — same algorithmic complexity; the improvement is
 * purely in constants (cache-miss rate, hardware prefetch effectiveness).
 */

#include "hpc/matrix.hpp"

namespace hpc::gemm {

/**
 * @brief Cache-friendly i-k-j GEMM: C = A × B  (C is overwritten).
 *
 * @param A  Input matrix, M×K
 * @param B  Input matrix, K×N
 * @param C  Output matrix, M×N  (must already be allocated; will be zeroed)
 *
 * @pre  A.cols() == B.rows()
 * @pre  C.rows() == A.rows() && C.cols() == B.cols()
 */
template <typename T>
void gemm_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();  // == B.rows()
    const std::size_t N = B.cols();

    assert(B.rows() == K && "Inner dimensions must agree");
    assert(C.rows() == M && C.cols() == N && "C must be M×N");

    C.zero();  // Ensure C starts as zero before accumulation.

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            // Hoist A(i,k) into a scalar register.
            // This value is used for every j in [0, N), so keeping it in a
            // register avoids N redundant memory reads of the same address.
            const T a_ik = A(i, k);

            // Both B row k and C row i are now traversed sequentially.
            // Hardware prefetchers can predict the access pattern and issue
            // prefetch requests ahead of the computation.
            for (std::size_t j = 0; j < N; ++j) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }
}

}  // namespace hpc::gemm
