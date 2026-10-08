#pragma once

/**
 * @file isa.hpp
 * @brief Which ISA-specific kernel families are compiled into this
 *        translation unit.
 *
 * Every HPC_HAS_* macro is always defined, to 0 or 1, so `#if HPC_HAS_AVX2`
 * works and a typo is not silently false the way `#ifdef` would be.
 *
 * No silent fallback: where an ISA is absent, that family's gemm_* functions
 * are declared `= delete` (see each src/gemm/<isa>.hpp), so calling
 * gemm_avx2_blocked on an ARM build is a compile-time error rather than a
 * scalar kernel under an AVX2 name. Benchmarks report such families as
 * SKIPPED without instantiating them, and their tests are not compiled, so a
 * machine's test count is exactly the set of kernels that ran there.
 *
 * Detection is compile-time: the default build uses -march=native (or an
 * -mcpu=), so the build CPU is the run CPU.
 *
 *   AVX2      __AVX2__
 *   AVX-512   __AVX512F__ (-march=native on a capable CPU, or HPC_ENABLE_AVX512=ON)
 *   NEON      __ARM_NEON && __aarch64__ (32-bit ARMv7 NEON lacks f64 lanes)
 *   SVE       __ARM_FEATURE_SVE
 *   SME       __ARM_FEATURE_SME && __ARM_FEATURE_SME2 (HPC_ENABLE_SME=ON and
 *             the configure-time probe passed; gemm_sme uses SME2 loads)
 *   AMX       HPC_ACCELERATE_AVAILABLE && __APPLE__ (HPC_ENABLE_AMX=ON and
 *             Accelerate.framework found)
 *   KleidiAI  HPC_KLEIDIAI_AVAILABLE && __ARM_FEATURE_SME2 (HPC_ENABLE_KLEIDIAI=ON)
 */

#if defined(__AVX2__)
    #define HPC_HAS_AVX2 1
#else
    #define HPC_HAS_AVX2 0
#endif

#if defined(__AVX512F__)
    #define HPC_HAS_AVX512 1
#else
    #define HPC_HAS_AVX512 0
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
    #define HPC_HAS_NEON 1
#else
    #define HPC_HAS_NEON 0
#endif

#if defined(__ARM_FEATURE_SVE)
    #define HPC_HAS_SVE 1
#else
    #define HPC_HAS_SVE 0
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
inline constexpr bool kHaveNeon   = HPC_HAS_NEON != 0;
inline constexpr bool kHaveSve    = HPC_HAS_SVE != 0;
inline constexpr bool kHaveSme    = HPC_HAS_SME != 0;
inline constexpr bool kHaveAmx    = HPC_HAS_AMX != 0;
inline constexpr bool kHaveKleidiAI = HPC_HAS_KLEIDIAI != 0;

}  // namespace hpc
