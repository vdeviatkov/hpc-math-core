cmake_minimum_required(VERSION 3.25)
project(hpc-math-core LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++20 — required for std::span, concepts, and other modern facilities used
# in later stages of this benchmark suite.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build type defaults
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type specified — defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Compiler diagnostics
# ---------------------------------------------------------------------------
add_compile_options(
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wextra>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wpedantic>
)

# MSVC uses __restrict instead of __restrict__ (GCC/Clang extension).
# Only apply to C++ sources — nvcc handles __restrict__ natively.
if(MSVC)
    add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__restrict__=__restrict>)
endif()

# ---------------------------------------------------------------------------
# Aggressive optimisation flags (Release / RelWithDebInfo builds only)
#
# -O3               : Full optimisation — auto-vectorisation, loop transforms,
#                     function inlining beyond -O2 thresholds.
# -march=native     : Emit instructions for the exact CPU this binary is built
#                     on (AVX2, FMA, BMI2, …). The resulting binary is NOT
#                     portable to other machines.
# -ffast-math       : Allows the compiler to reorder, contract and approximate
#                     floating-point operations in ways that violate strict
#                     IEEE-754. Enables:
#                       -fno-signed-zeros, -fno-trapping-math,
#                       -fassociative-math, -freciprocal-math,
#                       -ffinite-math-only (assumes no NaN / Inf).
#                     ⚠ Do NOT use in production code that relies on exact
#                     FP semantics (e.g. compensated summation, NaN detection).
#                     For benchmarking raw throughput it is safe and often
#                     unlocks FMA contraction and vectoriser cost-models.
# -funroll-loops    : Unroll loops whose trip count is known at compile time.
#                     Reduces loop-overhead instructions; pairs well with SIMD.
# -fno-omit-frame-pointer : Keep frame pointers so that profilers (perf,
#                     Instruments) can unwind the call stack correctly even
#                     in optimised builds.
#
# Inter-Procedural Optimisation (LTO / LTCG):
#   cmake -DHPC_ENABLE_LTO=ON enables link-time optimisation so that the
#   compiler can inline and optimise across translation-unit boundaries.
#   Increases link time significantly for large projects.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# -march target selection
#
# Default: -march=native — emit instructions for the exact build CPU.
#   Maximises performance locally and in CI on self-hosted runners.
#   ⚠ Binary is NOT portable to other CPUs.
#
# Override for CI / cross-compilation:
#   cmake -DHPC_MARCH=x86-64-v3 ..   → AVX2 + FMA (safe on all GitHub x86 runners)
#   cmake -DHPC_MARCH=armv8-a+simd .. → baseline AArch64
#   cmake -DHPC_MARCH=      ..       → empty string disables the flag entirely
# ---------------------------------------------------------------------------
set(HPC_MARCH "native" CACHE STRING "Value passed to -march=. Set to x86-64-v3 for portable x86 CI builds.")

# ---------------------------------------------------------------------------
# Optional: ARM SME (Scalable Matrix Extension)
#
# Enable with:  cmake -DHPC_ENABLE_SME=ON ..
#
# ⚠ Unlike every other ISA flag in this file, a successful *compile* does
# NOT imply SME will work: on Apple Silicon (M4/M4 Pro/M4 Max, the only
# broadly available SME2 hardware as of this writing), the generic
# "-march=armv9-a+sme2" flag produces a binary that SIGILLs on its very
# first instruction inside the streaming region. Clang emits a CNTD
# instruction outside streaming mode to size the ZA-save prologue buffer;
# CNTD is an ordinary (non-streaming) SVE instruction, and Apple Silicon
# implements NO non-streaming SVE unit — only Streaming SVE via SME. See
# src/gemm/sme.hpp for the full writeup. "-mcpu=apple-m4" sidesteps this by
# generating a prologue that doesn't need an outside-streaming SVE
# instruction. Because this failure mode is a runtime SIGILL rather than a
# compile error, we compile AND RUN a tiny SME probe for each candidate
# flag at configure time and use the first one that actually executes.
#
# This block MUST run before -march=${HPC_MARCH} is baked into
# add_compile_options() below: passing both "-march=native" AND
# "-mcpu=apple-m4" to Apple Clang silently drops the SME/SVE target
# features that "-mcpu=apple-m4" alone provides (verified empirically —
# "-march=native" has no notion of SME and clobbers it when combined). So
# when an "-mcpu=..." candidate wins, HPC_MARCH is forced empty here,
# disabling the plain -march flag entirely in favour of the CPU-name flag
# (which already implies the full baseline architecture).
# ---------------------------------------------------------------------------
option(HPC_ENABLE_SME "Enable ARM SME (Scalable Matrix Extension) instructions" OFF)

if(HPC_ENABLE_SME)
    include(CheckCXXSourceRuns)

    # Minimal probe: zero ZA, one FMOPA outer-product accumulate, one
    # horizontal-slice store out of ZA. Exercises exactly the instruction
    # sequence gemm_sme_* relies on (see src/gemm/sme.hpp).
    set(HPC_SME_PROBE_SRC [[
#include <arm_sme.h>
__arm_locally_streaming __arm_new("za")
static void probe() {
    svzero_za();
    svfloat32_t a = svdup_f32(1.0f);
    svfloat32_t b = svdup_f32(2.0f);
    svbool_t pg = svptrue_b32();
    svmopa_za32_f32_m(0, pg, pg, a, b);
    float out[64] = {0};
    svst1_hor_za32(0, 0, pg, out);
}
int main() { probe(); return 0; }
]])

    set(_HPC_SME_FOUND FALSE)
    # Try Apple-specific CPU names first (correct prologue codegen), then
    # generic native/architectural flags (correct on non-Apple SME hardware,
    # which does implement non-streaming SVE alongside SME).
    foreach(FLAG "-mcpu=apple-m4" "-mcpu=native" "-march=armv9-a+sme2")
        if(_HPC_SME_FOUND)
            break()
        endif()
        string(MAKE_C_IDENTIFIER "HPC_SME_RUNS_${FLAG}" _HPC_SME_CHECK_VAR)
        set(CMAKE_REQUIRED_FLAGS "${FLAG} -std=c++20")
        check_cxx_source_runs("${HPC_SME_PROBE_SRC}" ${_HPC_SME_CHECK_VAR})
        unset(CMAKE_REQUIRED_FLAGS)
        if(${_HPC_SME_CHECK_VAR})
            if(FLAG MATCHES "^-mcpu=")
                set(HPC_MARCH "" CACHE STRING "" FORCE)
                message(STATUS "SME flag '${FLAG}' supersedes HPC_MARCH — clearing -march (see comment above)")
            endif()
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
            set(_HPC_SME_FOUND TRUE)
            message(STATUS "SME enabled — verified working (compiled AND executed) with flag '${FLAG}'")
        endif()
    endforeach()

    if(NOT _HPC_SME_FOUND)
        message(WARNING "HPC_ENABLE_SME=ON but no candidate flag produced SME code that "
                         "actually runs on this (compiler, hardware) pair — gemm_sme_* will "
                         "fall back to SVE/NEON/scalar at compile time (no SIGILL risk).")
    endif()
endif()

add_compile_options(
    # --- GCC / Clang / AppleClang ---
    # COMPILE_LANGUAGE:CXX guards are required on every entry here: without
    # it, $<CXX_COMPILER_ID:...> only asserts "the project's CXX compiler
    # is this vendor" -- it does NOT restrict the flag to CXX sources -- so
    # on a CUDA-enabled build these host flags leak into every nvcc
    # invocation too. On Linux/macOS this went unnoticed because nvcc's
    # -forward-unknown-to-host-compiler quietly hands GCC/Clang-style flags
    # like -ffast-math/-march=native straight to the host compiler. On
    # Windows it does NOT go unnoticed: nvcc's own argument parser chokes
    # on the unguarded MSVC block below (see next comment) with "nvcc fatal
    # : A single input file is required for a non-link phase when an
    # outputfile is specified" -- caught only once real CUDA hardware
    # (Windows + MSVC + nvcc) was available to build on.
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${HPC_MARCH}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-march=${HPC_MARCH}>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-ffast-math>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-funroll-loops>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-fno-omit-frame-pointer>
    # --- MSVC ---
    # /O2          : Full optimisation (equivalent to -O2/-O3)
    # /fp:fast     : Fast floating-point model (equivalent to -ffast-math)
    # /Oy-         : Preserve frame pointers for profiling
    # NOTE: /arch flag is added below after AVX-512 detection.
    # Unguarded, nvcc (13.2, Windows) misparses these raw "/"-prefixed
    # tokens on its own command line and fails with "A single input file is
    # required for a non-link phase when an outputfile is specified" --
    # they must reach nvcc only via -Xcompiler, never directly, hence the
    # COMPILE_LANGUAGE:CXX guard (nvcc's CUDA-language compile rule adds
    # its own /O2 etc. separately, already correctly wrapped).
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/O2>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/fp:fast>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/Oy->
)

# Default MSVC SIMD arch — may be overridden to /arch:AVX512 below.
set(HPC_MSVC_ARCH "AVX2" CACHE INTERNAL "")

option(HPC_ENABLE_LTO "Enable Link-Time Optimisation (increases link time)" OFF)
if(HPC_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_err)
    if(_ipo_ok)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        message(STATUS "LTO enabled for Release builds")
    else()
        message(WARNING "LTO not supported by this toolchain: ${_ipo_err}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Optional: AVX-512 compilation flags
#
# Enable with:  cmake -DHPC_ENABLE_AVX512=ON ..
#
# ⚠ WARNING: Only enable on CPUs that support AVX-512 (Intel Skylake-SP,
# Ice Lake, Sapphire Rapids; AMD Zen 4+). Using these flags on unsupported
# hardware will cause an Illegal Instruction (SIGILL) crash at runtime.
# ---------------------------------------------------------------------------
option(HPC_ENABLE_AVX512 "Enable AVX-512 instruction set extensions" OFF)

if(HPC_ENABLE_AVX512)
    if(MSVC)
        # MSVC 17.8+ supports /arch:AVX512 which defines __AVX512F__ etc.
        # /arch:AVX512 implies AVX2, so it supersedes /arch:AVX2.
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("/arch:AVX512" HAS_MSVC_AVX512)
        if(HAS_MSVC_AVX512)
            set(HPC_MSVC_ARCH "AVX512" CACHE INTERNAL "" FORCE)
            message(STATUS "AVX-512 enabled via MSVC /arch:AVX512")
        else()
            message(WARNING "MSVC does not support /arch:AVX512 — upgrade to VS 2022 17.8+")
        endif()
    else()
        include(CheckCXXCompilerFlag)
        foreach(FLAG -mavx512f -mavx512dq -mavx512bw -mavx512vl)
            string(MAKE_C_IDENTIFIER "HAS_${FLAG}" FLAG_VAR)
            check_cxx_compiler_flag(${FLAG} ${FLAG_VAR})
            if(${FLAG_VAR})
                add_compile_options(${FLAG})
            endif()
        endforeach()
    endif()
    add_compile_definitions(HPC_HAS_AVX512)
    message(STATUS "AVX-512 extensions enabled")
endif()

# ---------------------------------------------------------------------------
# Optional: Apple AMX, accessed via Accelerate.framework
#
# Enable with:  cmake -DHPC_ENABLE_AMX=ON ..   (defaults to ON on Apple platforms)
#
# Apple Silicon has its own AMX (Apple Matrix coprocessor) — architecturally
# unrelated to Intel's AMX despite the shared acronym, and NOT exposed via
# any public ACLE-style intrinsic header the way ARM SME is (see
# HPC_ENABLE_SME above). Apple does not document raw AMX instructions for
# third-party use. The one sanctioned, stable way to benefit from it is
# Accelerate.framework's BLAS (cblas_sgemm / cblas_dgemm) — Apple's own
# highly-tuned implementation, understood (Apple does not publish which
# execution unit it targets) to dispatch to the AMX coprocessor blocks
# on Apple Silicon. Unlike the Intel-AMX design this replaced, this
# requires no special compiler flags, no runtime CPUID/permission dance,
# and no risk of SIGILL — Accelerate.framework ships in every macOS SDK,
# so this option is safe to default ON.
# ---------------------------------------------------------------------------
option(HPC_ENABLE_AMX "Enable Apple AMX GEMM path via Accelerate.framework (Apple platforms only)" ${APPLE})

set(HPC_AMX_ENABLED FALSE)
if(HPC_ENABLE_AMX)
    if(NOT APPLE)
        message(WARNING "HPC_ENABLE_AMX=ON but not building for Apple — Accelerate.framework "
                         "is Apple-only; gemm_amx_* will fall back to gemm_avx512_*/scalar.")
    else()
        find_library(HPC_ACCELERATE_FRAMEWORK Accelerate)
        if(HPC_ACCELERATE_FRAMEWORK)
            set(HPC_AMX_ENABLED TRUE)
            add_compile_definitions(HPC_HAS_AMX ACCELERATE_NEW_LAPACK)
            message(STATUS "Apple AMX enabled via Accelerate.framework (${HPC_ACCELERATE_FRAMEWORK})")
        else()
            message(WARNING "HPC_ENABLE_AMX=ON but Accelerate.framework was not found — "
                             "gemm_amx_* will fall back to gemm_avx512_*/scalar.")
        endif()
    endif()
endif()

# Apply the chosen MSVC SIMD architecture flag (AVX2 or AVX512).
# Use COMPILE_LANGUAGE:CXX so nvcc doesn't receive conflicting /arch flags.
if(MSVC)
    add_compile_options(
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/arch:${HPC_MSVC_ARCH}>
    )
    message(STATUS "MSVC SIMD architecture: /arch:${HPC_MSVC_ARCH}")
endif()

# ---------------------------------------------------------------------------
# Dependencies via FetchContent
#
# Both Google Benchmark and Google Test are pinned to specific release tags
# to ensure reproducible builds. Update tags deliberately rather than using
# HEAD or LATEST.
# ---------------------------------------------------------------------------
include(FetchContent)

# ---- Google Test (v1.15.2) ------------------------------------------------
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE   # only fetch the specified commit, not the full history
)
# Prevent GTest from overriding our compiler/linker settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---- Google Benchmark (v1.9.1) --------------------------------------------
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
    GIT_SHALLOW    TRUE
)
# Disable benchmark's own tests to speed up the configure step.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# ---------------------------------------------------------------------------
# Core library target
#
# "hpc_math_core" is a header-only INTERFACE library for now.  All public
# headers live under include/ and are exposed via target_include_directories.
# Switch to STATIC when .cpp translation units are added (e.g. explicit
# SIMD template instantiations).
# ---------------------------------------------------------------------------
add_library(hpc_math_core INTERFACE)
target_include_directories(hpc_math_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(hpc_math_core INTERFACE cxx_std_20)

if(HPC_AMX_ENABLED)
    target_link_libraries(hpc_math_core INTERFACE ${HPC_ACCELERATE_FRAMEWORK})
endif()

# ---------------------------------------------------------------------------
# Sub-directories
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)

# ---------------------------------------------------------------------------
# Optional CUDA support
#
# CUDA is enabled automatically if a CUDA toolkit is found on the system.
# When found:
#   - CUDA language is enabled for the project.
#   - A shared library `hpc_math_cuda` is compiled from src/cuda/*.cu.
#     It links against hpc_math_core (for the Matrix<T> header).
#   - bench_gemm_cuda and hpc_tests_cuda targets are added.
#   - The preprocessor macro HPC_HAVE_CUDA=1 is propagated to all consumers.
#
# When NOT found (macOS CPU-only, CI, …):
#   - A stub library `hpc_math_cuda` is created from
#     src/cuda/gemm_kernels_stub.cpp that provides cuda_device_count()→0
#     and throws on any kernel call.
#   - bench_gemm_cuda and hpc_tests_cuda still build; all benchmarks/tests
#     are skipped at runtime.
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA toolkit found — enabling GPU kernels (${CMAKE_CUDA_COMPILER})")
    enable_language(CUDA)

    # Detect compute capability for the installed GPUs so we can pass
    # the right -arch flag.  Falls back to sm_70 (Volta) if nothing is found.
    #
    # gemm_cuda_hopper_wgmma (Level 7, see src/gemm/README.md) needs the
    # Hopper-specific "90a" architecture target (wgmma/TMA are excluded
    # from the portable "90" family-compatible feature set) to actually
    # emit those instructions -- "native" only picks that up automatically
    # if built ON a Hopper GPU. Cross-compiling for Hopper from elsewhere:
    #   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90a ..
    # On any other target, the wgmma/TMA code path is compiled out (see the
    # __CUDA_ARCH__ >= 900 guard in gemm_kernels.cu) and gemm_cuda_hopper_wgmma
    # transparently falls back to gemm_cuda_mma_ldmatrix / gemm_cuda_wmma.
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
        message(STATUS "CUDA architectures not specified — compiling for: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    # The CUDA kernel library — compiled from .cu source.
    add_library(hpc_math_cuda STATIC
        src/cuda/gemm_kernels.cu
    )
    target_include_directories(hpc_math_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    )
    target_compile_features(hpc_math_cuda PUBLIC cxx_std_20)
    target_compile_definitions(hpc_math_cuda PUBLIC HPC_HAVE_CUDA=1)
    # Allow .cu files to use C++20 features.
    set_target_properties(hpc_math_cuda PROPERTIES
        CUDA_STANDARD 17        # nvcc C++17 is the max widely supported standard
        CUDA_STANDARD_REQUIRED  ON
        CUDA_SEPARABLE_COMPILATION OFF
    )
    # CUDA::cuda_driver (libcuda) is needed in addition to the usual
    # CUDA::cudart -- gemm_cuda_hopper_wgmma's TMA descriptor setup calls
    # cuTensorMapEncodeTiled, which is a driver-API-only function with no
    # CUDA-runtime-API equivalent. CUDA::cublas backs the vendor-tuned
    # reference kernels (gemm_cuda_cublas / gemm_cuda_cublas_tf32) used to
    # measure this GPU's realistic achievable peak against the hand-written
    # kernels above.
    target_link_libraries(hpc_math_cuda PUBLIC hpc_math_core CUDA::cudart CUDA::cuda_driver CUDA::cublas)

    message(STATUS "Building CUDA benchmark  : benchmarks/bench_gemm_cuda")
    message(STATUS "Building CUDA test suite : tests/hpc_tests_cuda")

else()
    message(STATUS "No CUDA compiler found — building CPU-only stub for CUDA symbols")

    # Stub implementation so that bench_gemm_cuda / hpc_tests_cuda compile
    # on machines without CUDA (they will just skip at runtime).
    add_library(hpc_math_cuda STATIC
        src/cuda/gemm_kernels_stub.cpp
    )
    target_include_directories(hpc_math_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    )
    target_compile_features(hpc_math_cuda PUBLIC cxx_std_20)
    # HPC_HAVE_CUDA is intentionally NOT defined here.
    target_link_libraries(hpc_math_cuda PUBLIC hpc_math_core)
endif()

add_subdirectory(tests/cuda)
add_subdirectory(benchmarks/cuda)
