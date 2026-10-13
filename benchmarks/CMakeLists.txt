add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE hpc_math_core benchmark::benchmark)

# CUDA kernels, or the stub when there is no CUDA toolkit (every row SKIPPED).
add_executable(bench_gemm_cuda bench_gemm_cuda.cpp)
target_link_libraries(bench_gemm_cuda PRIVATE hpc_math_cuda benchmark::benchmark)

# Phase timing of the SME blocked driver (docs/experiments/sme-b-layout).
if(_HPC_SME_FOUND)
    add_executable(sme_b_layout experiments/sme_b_layout.cpp)
    target_link_libraries(sme_b_layout PRIVATE hpc_math_core)
endif()
