#pragma once

/**
 * @file reordered.hpp
 * @brief Cache-friendly i-k-j GEMM: gemm_naive with the j and k loops swapped.
 *
 *   for i in [0, M):
 *     for k in [0, K):
 *       a_ik = A(i, k)          // invariant in the j-loop
 *       for j in [0, N):
 *         C(i,j) += a_ik * B(k,j)
 *
 * In the inner j-loop, B row k and C row i are both walked with stride 1,
 * so every element of each loaded cache line is used (8 of 8 doubles,
 * against 1 of 8 in gemm_naive) and the hardware prefetcher can run ahead.
 * A(i,k) stays in a register for the whole j-loop.
 *
 *        B, row-major (N=4)
 *        ┌───────────────────────────────────┐
 *   row0 │ B(0,0)  B(0,1)  B(0,2)  B(0,3)  │  ← the j-loop reads row k
 *   row1 │ B(1,0)  B(1,1)  B(1,2)  B(1,3)  │    left to right
 *   row2 │ B(2,0)  B(2,1)  B(2,2)  B(2,3)  │
 *   row3 │ B(3,0)  B(3,1)  B(3,2)  B(3,3)  │
 *        └───────────────────────────────────┘
 *
 * The stride-1 inner loop also lets the compiler auto-vectorise it at -O3,
 * which is a large part of the gain.
 *
 * Measured speedup over gemm_naive: 6–13× at N=256–1024 and 25–65× at
 * N=4096 (Apple M4 Max and AMD Zen 5; f32 gains more than f64, with twice the
 * elements per cache line). See docs/benchmarks.md.
 */

#include "hpc/matrix.hpp"

namespace hpc::gemm {

/**
 * @brief C = A × B with the cache-friendly i-k-j loop order (C is overwritten).
 *
 * @param A  Input matrix, M×K
 * @param B  Input matrix, K×N
 * @param C  Output matrix, M×N (must already be allocated)
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

    C.zero();

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const T a_ik = A(i, k);
            for (std::size_t j = 0; j < N; ++j)
                C(i, j) += a_ik * B(k, j);  // B and C rows: stride 1
        }
    }
}

}  // namespace hpc::gemm
