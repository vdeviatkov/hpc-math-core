#pragma once

/**
 * @file naive.hpp
 * @brief Naïve i-j-k GEMM — the reference baseline every other kernel is
 *        validated and benchmarked against.
 *
 *   for i in [0, M):
 *     for j in [0, N):
 *       for k in [0, K):
 *         C(i,j) += A(i,k) * B(k,j)
 *
 * Row-major storage, so in the inner k-loop:
 *
 *   A(i,k) → A[i*K + k]   stride 1: sequential, cache-friendly
 *   B(k,j) → B[k*N + j]   stride N: a new cache line on every k
 *   C(i,j)                invariant: stays in a register
 *
 *        B, row-major (N=4)
 *        ┌───────────────────────────────────┐
 *   row0 │ B(0,0)  B(0,1)  B(0,2)  B(0,3)  │  ← cache line 0
 *   row1 │ B(1,0)  B(1,1)  B(1,2)  B(1,3)  │  ← cache line 1
 *   row2 │ B(2,0)  B(2,1)  B(2,2)  B(2,3)  │  ← cache line 2
 *   row3 │ B(3,0)  B(3,1)  B(3,2)  B(3,3)  │  ← cache line 3
 *        └───────────────────────────────────┘
 *
 * Walking column j touches K different cache lines of B and uses one
 * element from each. The line holding B(k,j) is needed again for B(k,j+1),
 * but only after the k-loop has touched ~K other lines of B; at K=1024 and
 * 64-byte lines that is 64 KB, more than a typical L1. In the worst case
 * every B access misses: O(M*N*K) cache-line loads for O(M*N*K) FMAs.
 */

#include "hpc/matrix.hpp"

namespace hpc::gemm {

/**
 * @brief C = A × B with the naïve i-j-k loop order (C is overwritten).
 *
 * @param A  Input matrix, M×K
 * @param B  Input matrix, K×N
 * @param C  Output matrix, M×N (must already be allocated)
 *
 * @pre  A.cols() == B.rows()
 * @pre  C.rows() == A.rows() && C.cols() == B.cols()
 */
template <typename T>
void gemm_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();  // == B.rows()
    const std::size_t N = B.cols();

    assert(B.rows() == K && "Inner dimensions must agree");
    assert(C.rows() == M && C.cols() == N && "C must be M×N");

    C.zero();

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            T acc{};
            for (std::size_t k = 0; k < K; ++k)
                acc += A(i, k) * B(k, j);  // B walks a column: stride N
            C(i, j) = acc;
        }
    }
}

}  // namespace hpc::gemm
