cmake_minimum_required(VERSION 3.25)
project(hpc-math-core LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++20
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
# Compiler diagnostics, for C++ sources only: nvcc would pass them to its
# host compiler, where -Wpedantic flags every line directive nvcc generates.
# ---------------------------------------------------------------------------
add_compile_options(
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Wall;-Wextra;-Wpedantic>"
)

# MSVC uses __restrict instead of __restrict__ (GCC/Clang extension).
# Only apply to C++ sources — nvcc handles __restrict__ natively.
if(MSVC)
    add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__restrict__=__restrict>)
endif()

# ---------------------------------------------------------------------------
# -march target. The default, native, builds for the build CPU only.
# x86-64-v3 gives a portable AVX2 + FMA build; an empty value omits -march.
# ---------------------------------------------------------------------------
set(HPC_MARCH "native" CACHE STRING "Value passed to -march= (empty: no -march flag)")

# ---------------------------------------------------------------------------
# Optional: ARM SME (Scalable Matrix Extension)
#
# Enable with:  cmake -DHPC_ENABLE_SME=ON ..
#
# A successful compile does not mean SME works. On Apple Silicon,
# -march=armv9-a+sme2 builds a binary that SIGILLs on the first instruction
# of a streaming function: Clang sizes the ZA save buffer with CNTD, a
# non-streaming SVE instruction, and Apple Silicon has no non-streaming SVE.
# -mcpu=apple-m4 avoids it. So each candidate flag below is tried by compiling
# and running a small SME probe, and the first that runs is used.
#
# This must run before -march=${HPC_MARCH} is added: Apple Clang given both
# -march=native and -mcpu=apple-m4 drops the SME features, so when an -mcpu
# candidate wins, HPC_MARCH is cleared.
# ---------------------------------------------------------------------------
option(HPC_ENABLE_SME "Enable ARM SME (Scalable Matrix Extension) instructions" OFF)

if(HPC_ENABLE_SME)
    include(CheckCXXSourceRuns)

    # Probe every required feature: SME2 multi-vector loads, fp32/fp64
    # FMOPA and ZA slice stores. Observe the result so optimization cannot
    # remove the instructions whose execution is being checked.
    set(HPC_SME_PROBE_SRC [[
#include <arm_sme.h>
#ifndef __ARM_FEATURE_SME2
#error SME2 required
#endif
__arm_locally_streaming __arm_new("za")
static double probe() {
    svzero_za();
    float input[128];
    for (float& value : input) value = 1.0f;
    const auto a = svld1_f32_x2(svptrue_c32(), input);
    svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), svget2(a, 0), svget2(a, 1));
    float out32[64] = {};
    svst1_hor_za32(0, 0, svptrue_b32(), out32);
    svzero_za();
    const auto d = svdup_f64(2.0);
    svmopa_za64_f64_m(0, svptrue_b64(), svptrue_b64(), d, d);
    double out64[32] = {};
    svst1_hor_za64(0, 0, svptrue_b64(), out64);
    return out32[0] + out64[0];
}
int main() { return probe() == 5.0 ? 0 : 1; }
]])

    set(_HPC_SME_FOUND FALSE)
    # Try Apple-specific CPU names first (correct prologue codegen), then
    # generic native/architectural flags (correct on non-Apple SME hardware,
    # which does implement non-streaming SVE alongside SME).
    foreach(FLAG "-mcpu=apple-m4" "-mcpu=native" "-march=armv9-a+sme2")
        if(_HPC_SME_FOUND)
            break()
        endif()
        string(MAKE_C_IDENTIFIER "HPC_SME2_F64_RUNS_V2_${FLAG}" _HPC_SME_CHECK_VAR)
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
            message(STATUS "SME enabled — the probe compiled and ran with flag '${FLAG}'")
        endif()
    endforeach()

    if(NOT _HPC_SME_FOUND)
        message(WARNING "HPC_ENABLE_SME=ON but no candidate flag produced SME code that "
                         "actually runs on this (compiler, hardware) pair — gemm_sme will "
                         "not be compiled (declared = delete; benchmarks report SKIPPED).")
    endif()
endif()

# Release flags (rationale: docs/build.md). Every entry is limited to CXX
# sources: $<CXX_COMPILER_ID:...> alone does not restrict a flag to C++, so
# on a CUDA build these would also reach nvcc, and on Windows nvcc rejects
# the raw /-prefixed MSVC flags ("A single input file is required for a
# non-link phase when an outputfile is specified").
add_compile_options(
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${HPC_MARCH}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-march=${HPC_MARCH}>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-ffast-math>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-funroll-loops>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-fno-omit-frame-pointer>
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
# Needed only for MSVC or a non-native HPC_MARCH: -march=native already
# enables AVX-512 on a capable CPU. On a CPU without AVX-512 the resulting
# binary dies with SIGILL.
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
                add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
            endif()
        endforeach()
    endif()
    message(STATUS "AVX-512 extensions enabled")
endif()

# ---------------------------------------------------------------------------
# Optional: Apple AMX, accessed via Accelerate.framework
#
# Enable with:  cmake -DHPC_ENABLE_AMX=ON ..   (defaults to ON on Apple platforms)
#
# gemm_amx calls Accelerate's BLAS, the supported route to Apple's AMX
# coprocessor (see src/gemm/amx.hpp). It needs no special compiler flags and
# carries no SIGILL risk — Accelerate ships in every macOS SDK — so it is on
# by default on Apple platforms.
# ---------------------------------------------------------------------------
option(HPC_ENABLE_AMX "Enable Apple AMX GEMM path via Accelerate.framework (Apple platforms only)" ${APPLE})

set(HPC_AMX_ENABLED FALSE)
if(HPC_ENABLE_AMX)
    if(NOT APPLE)
        message(WARNING "HPC_ENABLE_AMX=ON but not building for Apple — Accelerate.framework "
                         "is Apple-only; gemm_amx will not be compiled.")
    else()
        find_library(HPC_ACCELERATE_FRAMEWORK Accelerate)
        if(HPC_ACCELERATE_FRAMEWORK)
            set(HPC_AMX_ENABLED TRUE)
            add_compile_definitions(HPC_ACCELERATE_AVAILABLE ACCELERATE_NEW_LAPACK)
            message(STATUS "Apple AMX enabled via Accelerate.framework (${HPC_ACCELERATE_FRAMEWORK})")
        else()
            message(WARNING "HPC_ENABLE_AMX=ON but Accelerate.framework was not found — "
                             "gemm_amx will not be compiled.")
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
# Pinned to release tags for reproducible builds.
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
# hpc_math_core is a header-only INTERFACE library: it carries the include
# paths and the optional Accelerate / KleidiAI link dependencies.
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
# Optional: Arm KleidiAI reference GEMM (gemm_kleidiai, src/gemm/kleidiai.hpp)
#
# Enable with:  cmake -DHPC_ENABLE_KLEIDIAI=ON ..   (default: ON when the SME
# probe above passed). KleidiAI's f32 matmul micro-kernels are SME2 assembly,
# so there is nothing to build without SME. Fetched and pinned like the other
# FetchContent deps; only the library target is built (no tests/benchmarks).
# ---------------------------------------------------------------------------
if(_HPC_SME_FOUND)
    set(_HPC_KLEIDIAI_DEFAULT ON)
else()
    set(_HPC_KLEIDIAI_DEFAULT OFF)
endif()
option(HPC_ENABLE_KLEIDIAI "Fetch Arm KleidiAI and build its SME2 reference GEMM (gemm_kleidiai)"
       ${_HPC_KLEIDIAI_DEFAULT})

if(HPC_ENABLE_KLEIDIAI)
    if(NOT _HPC_SME_FOUND)
        message(WARNING "HPC_ENABLE_KLEIDIAI=ON but SME is not enabled/working — "
                         "gemm_kleidiai will not be compiled.")
    else()
        FetchContent_Declare(
            kleidiai
            GIT_REPOSITORY https://github.com/ARM-software/kleidiai.git
            GIT_TAG        v1.31.0
            GIT_SHALLOW    TRUE
        )
        set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(kleidiai)
        target_link_libraries(hpc_math_core INTERFACE kleidiai)
        target_compile_definitions(hpc_math_core INTERFACE HPC_KLEIDIAI_AVAILABLE)
        message(STATUS "KleidiAI enabled (v1.31.0, SME2 f32 matmul)")
    endif()
endif()

# ---------------------------------------------------------------------------
# CUDA (optional, detected automatically)
#
# With nvcc, hpc_math_cuda is built from src/cuda/gemm.cu. Without
# it, hpc_math_cuda is src/cuda/gemm_stub.cpp: cuda_device_count()
# returns 0, so the CUDA tests and benchmarks still build and report SKIPPED.
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA toolkit found — building GPU kernels (${CMAKE_CUDA_COMPILER})")

    # Choose the architecture BEFORE enable_language(CUDA).
    #
    # enable_language(CUDA) initialises CMAKE_CUDA_ARCHITECTURES from nvcc's
    # own default (sm_52 on CUDA 12.x) whenever it is not already set, so an
    # `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` guard placed *after* that
    # call can never fire. The build then silently targets Maxwell, where
    # wmma does not exist, and every Tensor Core kernel fails to compile with
    # "name must be a namespace name" on `using namespace nvcuda`.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Verify this toolkit can actually target the requested architecture.
    #
    # nvcc refuses any architecture newer than itself, and "native" resolves
    # to whatever GPU is installed: on a machine whose card is newer than its
    # toolkit (e.g. CUDA 12.0, which predates Blackwell, driving an RTX 5080 /
    # sm_120) every .cu then dies with "Unsupported gpu architecture". Probe
    # it once here instead of failing at the first compile, and fall back to
    # the newest *virtual* architecture the toolkit can emit — the driver JITs
    # that PTX forward onto the newer GPU, at the cost of a pause on first
    # kernel launch.
    if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
        set(_hpc_cuda_probe "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/hpc_cuda_arch_probe.cu")
        file(WRITE "${_hpc_cuda_probe}" "__global__ void hpc_cuda_arch_probe() {}\n")
        execute_process(
            COMMAND "${CMAKE_CUDA_COMPILER}" -arch=native -ptx "${_hpc_cuda_probe}"
                    -o "${_hpc_cuda_probe}.ptx"
            RESULT_VARIABLE _hpc_cuda_probe_rc
            OUTPUT_QUIET ERROR_QUIET)

        if(NOT _hpc_cuda_probe_rc EQUAL 0)
            # Newest compute_XX this nvcc knows about. This branch is only
            # reached when the GPU is newer than the toolkit, so the PTX is
            # always JIT-compiled forward onto a strictly newer architecture.
            execute_process(
                COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-arch
                OUTPUT_VARIABLE _hpc_cuda_arch_list
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            string(REGEX MATCHALL "compute_[0-9]+" _hpc_cuda_archs "${_hpc_cuda_arch_list}")
            set(_hpc_cuda_fallback 0)
            foreach(_hpc_arch IN LISTS _hpc_cuda_archs)
                string(REGEX REPLACE "compute_" "" _hpc_arch_num "${_hpc_arch}")
                if(_hpc_arch_num GREATER _hpc_cuda_fallback)
                    set(_hpc_cuda_fallback ${_hpc_arch_num})
                endif()
            endforeach()

            if(_hpc_cuda_fallback GREATER 0)
                set(CMAKE_CUDA_ARCHITECTURES "${_hpc_cuda_fallback}-virtual")
                message(WARNING
                    "This GPU is newer than CUDA ${CMAKE_CUDA_COMPILER_VERSION} can target "
                    "directly. Falling back to compute_${_hpc_cuda_fallback} PTX, JIT-compiled "
                    "by the driver at run time. Kernels work, but are not tuned for the actual "
                    "architecture — install a newer CUDA toolkit for native code generation.")
            else()
                message(FATAL_ERROR
                    "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot target this GPU and no usable "
                    "fallback architecture was found. Install a newer CUDA toolkit, or pass "
                    "-DCMAKE_CUDA_ARCHITECTURES=<arch> explicitly.")
            endif()
        endif()
    endif()

    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    add_library(hpc_math_cuda STATIC src/cuda/gemm.cu)
    set_target_properties(hpc_math_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    # cuBLAS backs the reference entry points (gemm_cuda_cublas*); cuBLASLt
    # the FP8 one (gemm_cuda_cublas_lowp_device); the driver API encodes
    # Level 8's TMA tensor maps (cuTensorMapEncodeTiled).
    target_link_libraries(hpc_math_cuda PUBLIC hpc_math_core CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)
else()
    message(STATUS "No CUDA compiler found — building the CPU-only CUDA stub")
    add_library(hpc_math_cuda STATIC src/cuda/gemm_stub.cpp)
    target_link_libraries(hpc_math_cuda PUBLIC hpc_math_core)
endif()

# ---------------------------------------------------------------------------
# Metal (Apple GPUs, on by default on Apple platforms)
#
# hpc_math_metal is src/metal/gemm_metal.mm, which compiles the shaders in
# src/metal/gemm.metal at run time: their source is embedded into the binary
# here, so the build needs no offline Metal toolchain. Elsewhere it is
# src/metal/gemm_metal_stub.cpp and the Metal tests and benchmarks skip.
# ---------------------------------------------------------------------------
option(HPC_ENABLE_METAL "Build the Metal GEMM kernels (Apple platforms only)" ${APPLE})

if(HPC_ENABLE_METAL AND APPLE)
    enable_language(OBJCXX)
    set(_hpc_metal_src "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/gemm.metal")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_hpc_metal_src}")
    file(READ "${_hpc_metal_src}" HPC_METAL_SOURCE)
    file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/hpc_metal_source.inc"
         CONTENT "static const char kMetalSource[] = R\"HPCMETAL(@HPC_METAL_SOURCE@)HPCMETAL\";\n"
         @ONLY)
    add_library(hpc_math_metal STATIC src/metal/gemm_metal.mm)
    set_target_properties(hpc_math_metal PROPERTIES OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED ON)
    target_compile_options(hpc_math_metal PRIVATE -fobjc-arc -O3)
    target_include_directories(hpc_math_metal PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(hpc_math_metal PUBLIC hpc_math_core
        "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders")
    message(STATUS "Metal enabled (shaders compiled at run time from src/metal/gemm.metal)")
else()
    add_library(hpc_math_metal STATIC src/metal/gemm_metal_stub.cpp)
    target_link_libraries(hpc_math_metal PUBLIC hpc_math_core)
endif()

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
