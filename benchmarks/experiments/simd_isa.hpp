#pragma once

/**
 * @file simd_isa.hpp
 * @brief One vector type per ISA and precision, for the SIMD experiments
 *        (fma_peak.cpp, simd_microkernel.cpp): lanes, splat, load,
 *        broadcast-from-memory and fused multiply-add.
 */

#include "hpc/isa.hpp"

#if HPC_HAS_NEON
    #include <arm_neon.h>
#endif
#if HPC_HAS_AVX2 || HPC_HAS_AVX512
    #include <immintrin.h>
#endif

namespace hpc::experiments {

#if HPC_HAS_NEON
struct NeonF32 {
    using T = float;
    using V = float32x4_t;
    static constexpr const char* isa = "neon";
    static constexpr int lanes = 4, bits = 32, regs = 32;
    static V splat(float x) { return vdupq_n_f32(x); }
    static V load(const float* p) { return vld1q_f32(p); }
    static void store(float* p, V v) { vst1q_f32(p, v); }
    static V fma(V acc, V a, V b) { return vfmaq_f32(acc, a, b); }
};
struct NeonF64 {
    using T = double;
    using V = float64x2_t;
    static constexpr const char* isa = "neon";
    static constexpr int lanes = 2, bits = 64, regs = 32;
    static V splat(double x) { return vdupq_n_f64(x); }
    static V load(const double* p) { return vld1q_f64(p); }
    static void store(double* p, V v) { vst1q_f64(p, v); }
    static V fma(V acc, V a, V b) { return vfmaq_f64(acc, a, b); }
};
#endif
#if HPC_HAS_AVX2
struct Avx2F32 {
    using T = float;
    using V = __m256;
    static constexpr const char* isa = "avx2";
    static constexpr int lanes = 8, bits = 32, regs = 16;
    static V splat(float x) { return _mm256_set1_ps(x); }
    static V load(const float* p) { return _mm256_loadu_ps(p); }
    static void store(float* p, V v) { _mm256_storeu_ps(p, v); }
    static V fma(V acc, V a, V b) { return _mm256_fmadd_ps(a, b, acc); }
};
struct Avx2F64 {
    using T = double;
    using V = __m256d;
    static constexpr const char* isa = "avx2";
    static constexpr int lanes = 4, bits = 64, regs = 16;
    static V splat(double x) { return _mm256_set1_pd(x); }
    static V load(const double* p) { return _mm256_loadu_pd(p); }
    static void store(double* p, V v) { _mm256_storeu_pd(p, v); }
    static V fma(V acc, V a, V b) { return _mm256_fmadd_pd(a, b, acc); }
};
#endif
#if HPC_HAS_AVX512
struct Avx512F32 {
    using T = float;
    using V = __m512;
    static constexpr const char* isa = "avx512";
    static constexpr int lanes = 16, bits = 32, regs = 32;
    static V splat(float x) { return _mm512_set1_ps(x); }
    static V load(const float* p) { return _mm512_loadu_ps(p); }
    static void store(float* p, V v) { _mm512_storeu_ps(p, v); }
    static V fma(V acc, V a, V b) { return _mm512_fmadd_ps(a, b, acc); }
};
struct Avx512F64 {
    using T = double;
    using V = __m512d;
    static constexpr const char* isa = "avx512";
    static constexpr int lanes = 8, bits = 64, regs = 32;
    static V splat(double x) { return _mm512_set1_pd(x); }
    static V load(const double* p) { return _mm512_loadu_pd(p); }
    static void store(double* p, V v) { _mm512_storeu_pd(p, v); }
    static V fma(V acc, V a, V b) { return _mm512_fmadd_pd(a, b, acc); }
};
#endif

}  // namespace hpc::experiments
