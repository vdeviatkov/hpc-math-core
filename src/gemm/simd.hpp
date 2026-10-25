#pragma once

/**
 * @file simd.hpp
 * @brief The FP32/FP64 SIMD GEMM kernels, generic over the ISA.
 *
 * Each kernel takes the ISA as a primitive struct, Ops = Ymm (AVX2,
 * avx2.hpp), Zmm (AVX-512, avx512.hpp) or Q (NEON, neon.hpp), one
 * optimisation step apart:
 *
 *  gemm_simd_reordered<Ops>   i → k → j: A(i, k) broadcast, FMA against
 *                             stride-1 B and C rows. About the same as
 *                             scalar gemm_reordered, which the compiler
 *                             auto-vectorises; no blocking, so it degrades
 *                             once C row i no longer stays in L1.
 *  gemm_simd_blocked<Ops, R, V, TileM, TileK, TileN, PfDist = 0>
 *                             tiled i → k → j (blocked_driver.hpp) with an
 *                             R × V-vector tile of C held in registers for a
 *                             whole k block (vector_micro.hpp). PfDist > 0
 *                             adds software prefetch, PfDist strips ahead.
 *  gemm_simd_packed<Ops, R, V, Mc, Kc, Nc, PackA = false>
 *                             the same micro-kernel in GotoBLAS order with
 *                             each Kc × Nc block of B packed into contiguous
 *                             strips (packed_driver.hpp).
 *
 * The register tile and the cache blocks are template parameters with no
 * defaults: the right values depend on the machine and are chosen by
 * measurement (benchmarks/, docs/kernels/). T is deduced from the matrices:
 *
 *   gemm_simd_blocked<Zmm, 8, 3, 512, 256, 512>(A, B, C);   // Matrix<float>
 *
 * Ops<T> is defined only where its ISA is compiled (hpc/isa.hpp), so using
 * an absent ISA is a compile-time error.
 */

#include "gemm/avx2.hpp"
#include "gemm/avx512.hpp"
#include "gemm/blocked_driver.hpp"
#include "gemm/neon.hpp"
#include "gemm/packed_driver.hpp"
#include "gemm/vector_micro.hpp"
#include "hpc/matrix.hpp"

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace hpc::gemm {

template <template <typename> class Ops, typename T>
void gemm_simd_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_simd_reordered: T must be float or double");
    using O                 = Ops<T>;
    constexpr std::size_t W = sizeof(typename O::V) / sizeof(T);  // lanes per vector

    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    for (std::size_t i = 0; i < M; ++i) {
        T* c_row = C.data() + i * N;
        for (std::size_t k = 0; k < K; ++k) {
            const T* a_ik  = A.data() + i * K + k;
            const T* b_row = B.data() + k * N;
            const auto a   = O::load_a(a_ik);
            std::size_t j  = 0;
            for (; j + W <= N; j += W)
                O::store(c_row + j, O::fma(a, O::load(b_row + j), O::load(c_row + j)));
            for (; j < N; ++j)  // columns that do not fill a vector
                c_row[j] += *a_ik * b_row[j];
        }
    }
}

template <template <typename> class Ops, int R, int V, std::size_t TileM, std::size_t TileK,
          std::size_t TileN, std::size_t PfDist = 0, typename T>
void gemm_simd_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_simd_blocked: T must be float or double");
    detail::blocked_gemm<VectorMicro<Ops>, R, V, TileM, TileK, TileN, PfDist>(A, B, C);
}

template <template <typename> class Ops, int R, int V, std::size_t Mc, std::size_t Kc,
          std::size_t Nc, bool PackA = false, typename T>
void gemm_simd_packed(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_simd_packed: T must be float or double");
    detail::packed_gemm<VectorMicro<Ops>, R, V, Mc, Kc, Nc, PackA>(A, B, C);
}

}  // namespace hpc::gemm
