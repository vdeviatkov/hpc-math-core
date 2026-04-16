cmake_minimum_required(VERSION 3.25)

# ---------------------------------------------------------------------------
# CUDA GEMM benchmark binary
#
# bench_gemm_cuda links against hpc_math_cuda (which is either the real CUDA
# static library or the CPU stub — both expose the same symbols).
# On CPU-only machines all benchmarks print SKIPPED at runtime.
# ---------------------------------------------------------------------------
add_executable(bench_gemm_cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../bench_gemm_cuda.cpp
)

target_link_libraries(bench_gemm_cuda
    PRIVATE
        hpc_math_cuda
        hpc_math_core
        benchmark::benchmark
)

