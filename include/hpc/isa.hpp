#pragma once

/**
 * @file isa.hpp
 * @brief Which ISA-specific kernel families are compiled into this
 *        translation unit.
 *
 * Every HPC_HAS_* macro is always defined, to 0 or 1, so `#if HPC_HAS_AVX2`
 * tests the feature value. `#ifdef` would also accept a macro defined to 0.
 *
 * No silent fallback: where an ISA is absent, its primitive structs (Ymm,
 * ZmmDot, …) are only declared and its other gemm_* functions are declared
 * `= delete`, so calling gemm_simd_blocked<Ymm> on an ARM build is a
 * compile-time error rather than a scalar kernel under an AVX2 name. Benchmarks report such families as
 * SKIPPED without instantiating them, and their tests are not compiled, so a
 * machine's test count is exactly the set of kernels that ran there.
 *
 * Detection is compile-time: the default build uses -march=native (or an
 * -mcpu=), so the build CPU is the run CPU.
 *
 *   AVX2      __AVX2__ and FMA support (__FMA__, or MSVC /arch:AVX2)
 *   AVX-512   __AVX512F__ (-march=native on a capable CPU, or HPC_ENABLE_AVX512=ON)
 *   AVX-512 low precision
 *             __AVX512F__ && __AVX512BF16__ && __AVX512VNNI__ (VDPBF16PS, VPDPBUSD)
 *   NEON      __ARM_NEON && __aarch64__ (32-bit ARMv7 NEON lacks f64 lanes)
 *   NEON low precision
 *             NEON and the BF16, I8MM, FP16 FML and FP16 vector extensions
 *             (BFMMLA, SMMLA, FMLAL, FMLA .8h)
 *   SME       __ARM_FEATURE_SME && __ARM_FEATURE_SME2 (HPC_ENABLE_SME=ON and
 *             the configure-time probe passed; gemm_sme uses SME2 loads)
 *   AMX       HPC_ACCELERATE_AVAILABLE && __APPLE__ (HPC_ENABLE_AMX=ON and
 *             Accelerate.framework found)
 *   KleidiAI  HPC_KLEIDIAI_AVAILABLE && __ARM_FEATURE_SME2 (HPC_ENABLE_KLEIDIAI=ON)
 */

#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
    #define HPC_HAS_AVX2 1
#else
    #define HPC_HAS_AVX2 0
#endif

#if defined(__AVX512F__)
    #define HPC_HAS_AVX512 1
#else
    #define HPC_HAS_AVX512 0
#endif

#if defined(__AVX512F__) && defined(__AVX512BF16__) && defined(__AVX512VNNI__)
    #define HPC_HAS_AVX512_LOWP 1
#else
    #define HPC_HAS_AVX512_LOWP 0
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
    #define HPC_HAS_NEON 1
#else
    #define HPC_HAS_NEON 0
#endif

#if HPC_HAS_NEON && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_MATMUL_INT8) && \
    defined(__ARM_FEATURE_FP16_FML) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #define HPC_HAS_NEON_LOWP 1
#else
    #define HPC_HAS_NEON_LOWP 0
#endif

#if defined(__ARM_FEATURE_SME) && defined(__ARM_FEATURE_SME2)
    #define HPC_HAS_SME 1
#else
    #define HPC_HAS_SME 0
#endif

#if defined(HPC_ACCELERATE_AVAILABLE) && defined(__APPLE__)
    #define HPC_HAS_AMX 1
#else
    #define HPC_HAS_AMX 0
#endif

#if defined(HPC_KLEIDIAI_AVAILABLE) && defined(__ARM_FEATURE_SME2)
    #define HPC_HAS_KLEIDIAI 1
#else
    #define HPC_HAS_KLEIDIAI 0
#endif

namespace hpc {

/// Compile-time mirrors of the macros above, for `if constexpr` in templates.
inline constexpr bool kHaveAvx2   = HPC_HAS_AVX2 != 0;
inline constexpr bool kHaveAvx512 = HPC_HAS_AVX512 != 0;
inline constexpr bool kHaveAvx512Lowp = HPC_HAS_AVX512_LOWP != 0;
inline constexpr bool kHaveNeon   = HPC_HAS_NEON != 0;
inline constexpr bool kHaveNeonLowp = HPC_HAS_NEON_LOWP != 0;
inline constexpr bool kHaveSme    = HPC_HAS_SME != 0;
inline constexpr bool kHaveAmx    = HPC_HAS_AMX != 0;
inline constexpr bool kHaveKleidiAI = HPC_HAS_KLEIDIAI != 0;

}  // namespace hpc
