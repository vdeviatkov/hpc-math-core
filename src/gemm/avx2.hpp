#pragma once

/**
 * @file avx2.hpp
 * @brief AVX2 + FMA primitives for the generic SIMD kernels (simd.hpp).
 *
 *   gemm_simd_reordered<Ymm>(A, B, C);
 *   gemm_simd_blocked<Ymm, R, V, TileM, TileK, TileN[, PfDist]>(A, B, C);
 *   gemm_simd_packed<Ymm, R, V, Mc, Kc, Nc[, PackA]>(A, B, C);
 *
 * 16 × 256-bit YMM registers (8 f32 / 4 f64). With 2 FMA ports:
 * 2 × 8 lanes × 2 FLOP = 32 FLOP/cycle f32, 16 f64.
 *
 * Available on Intel Haswell+ and AMD Zen+; not on Apple Silicon.
 * HPC_HAS_AVX2 (hpc/isa.hpp) requires AVX2 and FMA. Ymm<T> is declared on
 * every target and defined only where HPC_HAS_AVX2 is 1, so a kernel
 * instantiated with Ymm elsewhere is a compile-time error.
 */

#include "hpc/isa.hpp"

#if HPC_HAS_AVX2
    #include <immintrin.h>
#endif

namespace hpc::gemm {

template <typename T> struct Ymm;

#if HPC_HAS_AVX2

// load_a broadcasts A(i, k) from memory into all lanes (vbroadcastss).
template <> struct Ymm<float> {
    using V = __m256;
    static V load(const float* p) { return _mm256_loadu_ps(p); }
    static void store(float* p, V v) { _mm256_storeu_ps(p, v); }
    static V load_a(const float* p) { return _mm256_broadcast_ss(p); }
    static V fma(V x, V y, V acc) { return _mm256_fmadd_ps(x, y, acc); }
};
template <> struct Ymm<double> {
    using V = __m256d;
    static V load(const double* p) { return _mm256_loadu_pd(p); }
    static void store(double* p, V v) { _mm256_storeu_pd(p, v); }
    static V load_a(const double* p) { return _mm256_broadcast_sd(p); }
    static V fma(V x, V y, V acc) { return _mm256_fmadd_pd(x, y, acc); }
};

#endif  // HPC_HAS_AVX2

}  // namespace hpc::gemm
