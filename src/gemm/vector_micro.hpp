#pragma once

/**
 * @file vector_micro.hpp
 * @brief The register-tiled FP32/FP64 micro-kernel of gemm_simd_blocked and
 *        gemm_simd_packed (simd.hpp), for any ISA primitive struct.
 *
 * C[0..R)[0..V·W) += A[0..R)[0..k_len) × B[0..k_len)[0..V·W), with an
 * R × V-vector tile of C held in R·V accumulators for the whole k loop.
 * Per k step: V loads of B row k, one load of A(i+r, k) per row, R·V FMAs:
 * an outer product of a column of A and a row of B added into the tile.
 *
 * The three ISAs differ only in their instructions, so each supplies an Ops
 * struct per element type (Ymm<T> in avx2.hpp, Zmm<T> in avx512.hpp, Q<T>
 * in neon.hpp):
 *
 *   using V;                                  // vector register type
 *   static V    load(const T* p);             // W contiguous elements
 *   static void store(T* p, V v);
 *   static auto load_a(const T* p);           // A(i+r, k) as fma's first operand
 *   static V    fma(decltype(load_a(p)) a, V b, V acc);   // acc + a · b
 *
 * load_a is where the ISAs really differ: AVX2 broadcasts from memory
 * (vbroadcastss), AVX-512 broadcasts a scalar (vbroadcastss / set1), and
 * NEON loads one element with a plain LDR for an FMLA by element. Each Ops
 * struct documents why.
 *
 * The fixed-count loops must unroll completely so acc[][] stays in
 * registers; R and V are template parameters for that, and so that the
 * drivers and benchmarks can sweep tile shapes (docs/kernels/).
 */

#include <cstddef>

namespace hpc::gemm {

template <typename Ops, int R, int V, typename T>
inline void vector_micro(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                         std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t k_len) noexcept {
    static_assert(R >= 1 && R <= 16 && V >= 1 && V <= 16, "tile dimensions up to 16 (the unroll factor)");
    using Vec               = typename Ops::V;
    constexpr std::size_t W = sizeof(Vec) / sizeof(T);  // lanes per vector

    Vec acc[R][V];
#pragma GCC unroll 16
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            acc[r][v] = Ops::load(c + r * ldc + v * W);
    for (std::size_t k = 0; k < k_len; ++k) {
        Vec bv[V];
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            bv[v] = Ops::load(b + k * ldb + v * W);
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r) {
            const auto ar = Ops::load_a(a + r * lda + k);
#pragma GCC unroll 16
            for (int v = 0; v < V; ++v)
                acc[r][v] = Ops::fma(ar, bv[v], acc[r][v]);
        }
    }
#pragma GCC unroll 16
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            Ops::store(c + r * ldc + v * W, acc[r][v]);
}

/// The micro-kernel as blocked_gemm (blocked_driver.hpp) and packed_gemm
/// (packed_driver.hpp) call it, e.g. VectorMicro<Zmm>.
template <template <typename> class Ops>
struct VectorMicro {
    template <typename T>
    static constexpr std::size_t kLanes = sizeof(typename Ops<T>::V) / sizeof(T);
    template <int R, int V, typename T>
    static void run(const T* a, const T* b, T* c, std::size_t lda, std::size_t ldb, std::size_t ldc,
                    std::size_t k_len) noexcept {
        vector_micro<Ops<T>, R, V>(a, b, c, lda, ldb, ldc, k_len);
    }
};

}  // namespace hpc::gemm
