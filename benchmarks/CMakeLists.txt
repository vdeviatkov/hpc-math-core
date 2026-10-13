add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE hpc_math_core benchmark::benchmark)

# CUDA kernels, or the stub when there is no CUDA toolkit (every row SKIPPED).
add_executable(bench_gemm_cuda bench_gemm_cuda.cpp)
target_link_libraries(bench_gemm_cuda PRIVATE hpc_math_cuda benchmark::benchmark)

# Experiments behind the SME2 articles in docs/articles/; their data and
# method are in docs/experiments/.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    add_executable(cache_chase experiments/cache_chase.cpp)
endif()
if(_HPC_SME_FOUND)
    add_executable(sme_b_layout experiments/sme_b_layout.cpp)
    add_executable(sme_b_stride_probe experiments/sme_b_stride_probe.cpp)
    add_executable(sme_fmopa experiments/sme_fmopa.cpp)
    target_link_libraries(sme_fmopa PRIVATE hpc_math_core)
    target_link_libraries(sme_b_layout PRIVATE hpc_math_core)
    target_link_libraries(sme_b_stride_probe PRIVATE hpc_math_core)
endif()
