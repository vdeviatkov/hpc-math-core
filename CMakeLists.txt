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
# -ffast-math       : Allows reassociation, FMA contraction and reciprocal
#                     approximations, and assumes no NaN/Inf. Lets the
#                     compiler vectorise the scalar reduction loops; not
#                     suitable for code that needs strict IEEE-754 semantics.
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
# Default: -march=native — instructions for the build CPU; the binary may not
# run on other CPUs.
#
# Overrides:
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

    # Probe: zero ZA, one FMOPA, one horizontal-slice store from ZA — the
    # instruction types gemm_sme uses.
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
            message(STATUS "SME enabled — the probe compiled and ran with flag '${FLAG}'")
        endif()
    endforeach()

    if(NOT _HPC_SME_FOUND)
        message(WARNING "HPC_ENABLE_SME=ON but no candidate flag produced SME code that "
                         "actually runs on this (compiler, hardware) pair — gemm_sme will "
                         "not be compiled (declared = delete; benchmarks report SKIPPED).")
    endif()
endif()

add_compile_options(
    # --- GCC / Clang / AppleClang ---
    # COMPILE_LANGUAGE:CXX guards are required on every entry here: without
    # it, $<CXX_COMPILER_ID:...> only asserts "the project's CXX compiler
    # is this vendor" -- it does NOT restrict the flag to CXX sources -- so
    # on a CUDA-enabled build these host flags leak into every nvcc
    # invocation too. On Linux/macOS nvcc's -forward-unknown-to-host-compiler
    # quietly hands GCC/Clang-style flags like -ffast-math/-march=native to
    # the host compiler, but on Windows nvcc's own argument parser rejects
    # the unguarded MSVC block below (see next comment) with "nvcc fatal :
    # A single input file is required for a non-link phase when an
    # outputfile is specified".
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
                add_compile_options(${FLAG})
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
#   - A static library `hpc_math_cuda` is compiled from src/cuda/*.cu.
#     It links against hpc_math_core (for the Matrix<T> header).
#   - bench_gemm_cuda and hpc_tests_cuda targets are added.
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

    # The CUDA kernel library — compiled from .cu source.
    add_library(hpc_math_cuda STATIC
        src/cuda/gemm_kernels.cu
    )
    target_include_directories(hpc_math_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    )
    target_compile_features(hpc_math_cuda PUBLIC cxx_std_20)
    # Allow .cu files to use C++20 features.
    set_target_properties(hpc_math_cuda PROPERTIES
        CUDA_STANDARD 17        # nvcc C++17 is the max widely supported standard
        CUDA_STANDARD_REQUIRED  ON
        CUDA_SEPARABLE_COMPILATION OFF
    )
    # CUDA::cublas backs the vendor-tuned reference kernels
    # (gemm_cuda_cublas / gemm_cuda_cublas_tf32 / gemm_cuda_cublas_fp16) used
    # to measure this GPU's realistic achievable peak against the
    # hand-written kernels above.
    target_link_libraries(hpc_math_cuda PUBLIC hpc_math_core CUDA::cudart CUDA::cublas)

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
    target_link_libraries(hpc_math_cuda PUBLIC hpc_math_core)
endif()

add_subdirectory(tests/cuda)
add_subdirectory(benchmarks/cuda)
