#pragma once

/**
 * @file blocked_driver.hpp
 * @brief The cache-blocked, register-tiled GEMM driver behind
 *        gemm_simd_blocked (simd.hpp), with or without software prefetch.
 *
 * C = A · B, row-major, i-k-j order with three levels of blocking:
 *
 *   for i_blk (TileM rows of A and C)
 *     for k_blk (TileK: the k extent of one register-tile pass)
 *       for j_blk (TileN columns of B and C)
 *         for i in the block, R rows at a time
 *           for j in the block, V vectors at a time
 *             Micro::run<R, V>: C(R × V·W) += A(R × TileK) · B(TileK × V·W)
 *
 * The register tile, R rows × V vectors of C kept in R·V accumulators for
 * the whole k pass, is a template parameter of every family, so one driver
 * serves all tile shapes and benchmarks can sweep them (bench_gemm,
 * docs/kernels/). The edges use the same micro-kernel with fewer rows (when
 * TileM or M is not a multiple of R) or fewer vectors; only the last
 * columns that do not fill one vector take a scalar loop.
 *
 * Micro is a struct with
 *   template <typename T> static constexpr std::size_t kLanes;   // W, elements per vector
 *   template <int R, int V, typename T> static void run(a, b, c, lda, ldb, ldc, k_len);
 * where run adds A(R × k_len) · B(k_len × V·W) into C.
 *
 * PfDist > 0 adds software prefetch (prefetch.hpp): the next k block of B,
 * and the A and C rows PfDist register-tile strips ahead.
 */

#include "hpc/matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

namespace hpc::gemm::detail {

/// Software prefetch with compile-time hints. Rw: 0 read, 1 write; Loc:
/// temporal locality 0 (none) … 3 (keep in all levels).
template <int Rw = 0, int Loc = 2>
inline void pf(const void* addr) noexcept {
#if defined(_MSC_VER)
    // _MM_HINT_T0 (L1), _MM_HINT_T1 (L2), _MM_HINT_T2 (L3), _MM_HINT_NTA
    constexpr int hint = (Loc == 3) ? _MM_HINT_T0 : (Loc == 2) ? _MM_HINT_T1 : (Loc == 1) ? _MM_HINT_T2 : _MM_HINT_NTA;
    _mm_prefetch(static_cast<const char*>(addr), hint);
#else
    __builtin_prefetch(addr, Rw, Loc);
#endif
}

// Run Micro::run<rows, V> for a runtime `rows` in 1 … R.
template <typename Micro, int R, int V, typename T>
inline void run_rows(int rows, const T* a, const T* b, T* c, std::size_t lda, std::size_t ldb,
                     std::size_t ldc, std::size_t k_len) noexcept {
    if constexpr (R >= 1) {
        if (rows == R)
            Micro::template run<R, V>(a, b, c, lda, ldb, ldc, k_len);
        else
            run_rows<Micro, R - 1, V>(rows, a, b, c, lda, ldb, ldc, k_len);
    }
}

// Run Micro::run<rows, vecs> for runtime `vecs` in 1 … V.
template <typename Micro, int R, int V, typename T>
inline void run_tile(int rows, int vecs, const T* a, const T* b, T* c, std::size_t lda, std::size_t ldb,
                     std::size_t ldc, std::size_t k_len) noexcept {
    if constexpr (V >= 1) {
        if (vecs == V)
            run_rows<Micro, R, V>(rows, a, b, c, lda, ldb, ldc, k_len);
        else
            run_tile<Micro, R, V - 1>(rows, vecs, a, b, c, lda, ldb, ldc, k_len);
    }
}

/// `rows` (≤ R) rows of one k pass across j_len columns of a block: R × V
/// tiles, then one narrower tile for the remaining whole vectors, then
/// scalar columns.
template <typename Micro, int R, int V, typename T>
inline void row_strip(int rows, const T* a, const T* b, T* c, std::size_t j_len, std::size_t lda,
                      std::size_t ldb, std::size_t ldc, std::size_t k_len) noexcept {
    constexpr std::size_t W = Micro::template kLanes<T>;
    std::size_t j           = 0;
    for (; j + V * W <= j_len; j += V * W)
        run_rows<Micro, R, V>(rows, a, b + j, c + j, lda, ldb, ldc, k_len);
    if (const int vecs = static_cast<int>((j_len - j) / W); vecs > 0) {
        run_tile<Micro, R, V - 1>(rows, vecs, a, b + j, c + j, lda, ldb, ldc, k_len);
        j += static_cast<std::size_t>(vecs) * W;
    }
    for (; j < j_len; ++j)  // fewer columns than one vector
        for (int r = 0; r < rows; ++r) {
            T acc = c[r * ldc + j];
            for (std::size_t k = 0; k < k_len; ++k)
                acc += a[r * lda + k] * b[k * ldb + j];
            c[r * ldc + j] = acc;
        }
}

template <typename Micro, int R, int V, std::size_t TileM, std::size_t TileK, std::size_t TileN,
          std::size_t PfDist = 0, typename T>
void blocked_gemm(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(R >= 1 && V >= 1, "the register tile needs at least one row and one vector");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    const std::size_t lda = K, ldb = N, ldc = N;
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    for (std::size_t i_blk = 0; i_blk < M; i_blk += TileM) {
        const std::size_t i_end = std::min(i_blk + TileM, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += TileK) {
            const std::size_t k_len = std::min(TileK, K - k_blk);
            if constexpr (PfDist > 0)
                if (k_blk + TileK < K)
                    pf<0, 2>(B.data() + (k_blk + TileK) * ldb);  // next k block of B
            for (std::size_t j_blk = 0; j_blk < N; j_blk += TileN) {
                const std::size_t j_len = std::min(TileN, N - j_blk);
                for (std::size_t i = i_blk; i < i_end; i += R) {
                    if constexpr (PfDist > 0)
                        if (i + R * PfDist < i_end) {
                            pf<0, 2>(A.data() + (i + R * PfDist) * lda + k_blk);  // A strip ahead
                            pf<1, 3>(C.data() + (i + R * PfDist) * ldc + j_blk);  // C strip ahead
                        }
                    const int rows = static_cast<int>(std::min<std::size_t>(R, i_end - i));
                    row_strip<Micro, R, V>(rows, A.data() + i * lda + k_blk, B.data() + k_blk * ldb + j_blk,
                                           C.data() + i * ldc + j_blk, j_len, lda, ldb, ldc, k_len);
                }
            }
        }
    }
}

}  // namespace hpc::gemm::detail
