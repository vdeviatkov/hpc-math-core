#pragma once

/**
 * @file avx512.hpp
 * @brief AVX-512 primitives for the generic SIMD kernels (simd.hpp).
 *
 *   gemm_simd_reordered<Zmm>(A, B, C);
 *   gemm_simd_blocked<Zmm, R, V, TileM, TileK, TileN>(A, B, C);
 *   gemm_simd_packed<Zmm, R, V, Mc, Kc, Nc>(A, B, C);
 *
 * 32 × 512-bit ZMM registers (16 f32 / 8 f64). With 2 FMA ports:
 * 2 × 16 lanes × 2 FLOP = 64 FLOP/cycle f32, 32 f64.
 *
 * Available on Intel Skylake-SP/X and later server/workstation parts
 * (Cascade Lake, Ice Lake, Sapphire Rapids, Rocket Lake) and AMD Zen 4+.
 * Not on Apple Silicon, Alder Lake/Raptor Lake (disabled by Intel), or AMD
 * before Zen 4.
 *
 * HPC_HAS_AVX512 (hpc/isa.hpp) follows __AVX512F__. -march=native enables
 * AVX-512 on a capable CPU; HPC_ENABLE_AVX512=ON forces the flags
 * otherwise. Zmm<T> is declared on every target and defined only where
 * HPC_HAS_AVX512 is 1.
 */

#include "hpc/isa.hpp"

#if HPC_HAS_AVX512
    #include <immintrin.h>
#endif

namespace hpc::gemm {

template <typename T> struct Zmm;

#if HPC_HAS_AVX512

// load_a broadcasts A(i, k) into all lanes (vbroadcastss from memory).
//
// The register tile matters more here than on AVX2: with 4 rows × 2 ZMM,
// GCC 13 folds the B loads into the FMAs as memory operands and re-loads B
// for every FMA, so the kernel runs into the load ports at 60% of peak.
// With 6 or more rows it loads B once per k step (docs/kernels/avx512.md).
template <> struct Zmm<float> {
    using V = __m512;
    static V load(const float* p) { return _mm512_loadu_ps(p); }
    static void store(float* p, V v) { _mm512_storeu_ps(p, v); }
    static V load_a(const float* p) { return _mm512_set1_ps(*p); }
    static V fma(V x, V y, V acc) { return _mm512_fmadd_ps(x, y, acc); }
};
template <> struct Zmm<double> {
    using V = __m512d;
    static V load(const double* p) { return _mm512_loadu_pd(p); }
    static void store(double* p, V v) { _mm512_storeu_pd(p, v); }
    static V load_a(const double* p) { return _mm512_set1_pd(*p); }
    static V fma(V x, V y, V acc) { return _mm512_fmadd_pd(x, y, acc); }
};

#endif  // HPC_HAS_AVX512

}  // namespace hpc::gemm
