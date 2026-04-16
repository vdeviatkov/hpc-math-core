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
add_compile_options(
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-O3>
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-march=native>
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-ffast-math>
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-funroll-loops>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-omit-frame-pointer>
)

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
    include(CheckCXXCompilerFlag)
    foreach(FLAG -mavx512f -mavx512dq -mavx512bw -mavx512vl)
        string(MAKE_C_IDENTIFIER "HAS_${FLAG}" FLAG_VAR)
        check_cxx_compiler_flag(${FLAG} ${FLAG_VAR})
        if(${FLAG_VAR})
            add_compile_options(${FLAG})
        endif()
    endforeach()
    add_compile_definitions(HPC_HAS_AVX512)
    message(STATUS "AVX-512 extensions enabled")
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
    include(FindCUDA OPTIONAL)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
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
    target_link_libraries(hpc_math_cuda PUBLIC hpc_math_core CUDA::cudart)

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
