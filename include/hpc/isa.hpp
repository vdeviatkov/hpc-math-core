#pragma once

/**
 * @file isa.hpp
 * @brief Single source of truth for which ISA-specific kernel families are
 *        compiled into this translation unit.
 *
 * Every `HPC_HAS_*` macro below is always defined, to exactly 0 or 1, so it
 * is safe in `#if HPC_HAS_AVX2` and never silently false because of a typo
 * (`#ifdef HPC_HAS_AVX2` would be — an undefined macro is not an error).
 *
 * Policy: **no silent fallback.**
 *   On a target where an ISA is absent, that family's `gemm_*` functions are
 *   declared `= delete` (see each src/gemm/<isa>.hpp). Calling
 *   `gemm_avx2_blocked` on an ARM build is therefore a compile-time error,
 *   never a scalar kernel wearing an AVX2 name. Benchmarks report such
 *   families as SKIPPED without instantiating them (bench_gemm.cpp); tests
 *   for them are not compiled at all (test_gemm.cpp), so the test count on
 *   any given machine is the number of kernels that actually ran there.
 *
 * All flags are decided at compile time because every build uses
 * `-march=native` (or `-mcpu=`) — the build CPU is the run CPU. Runtime
 * dispatch (cpuid → best kernel) would be a separate, explicit facility,
 * not something hidden inside each kernel.
 *
 * Detection sources:
 *   AVX2     __AVX2__            GCC/Clang -mavx2 / -march=native, MSVC /arch:AVX2
 *   AVX-512  __AVX512F__         GCC/Clang -mavx512f, MSVC /arch:AVX512 (HPC_ENABLE_AVX512=ON)
 *   NEON     __ARM_NEON && __aarch64__   any AArch64 target (32-bit ARMv7 NEON lacks
 *                                        vfmaq_f64 / f64 lanes, so it is excluded)
 *   SVE      __ARM_FEATURE_SVE   -march=armv8-a+sve or a -mcpu= that implies it
 *   SME      __ARM_FEATURE_SME   HPC_ENABLE_SME=ON and the configure-time probe passed,
 *            && __ARM_FEATURE_SME2   gemm_sme uses SME2 multi-vector loads (every SME CPU this
 *                                repo targets, e.g. Apple M4, has SME2)
 *   AMX      HPC_ACCELERATE_AVAILABLE (CMake: HPC_ENABLE_AMX=ON and Accelerate.framework found)
 *            && __APPLE__        the Apple AMX coprocessor is reached only through Accelerate
 *   KleidiAI HPC_KLEIDIAI_AVAILABLE (CMake: HPC_ENABLE_KLEIDIAI=ON, fetched and built)
 *            && __ARM_FEATURE_SME2   its f32 matmul micro-kernels are SME2 assembly
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

#if defined(__ARM_NEON) && defined(__aarch64__)  // the kernels use AArch64-only f64 NEON
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
