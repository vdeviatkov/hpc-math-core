add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE hpc_math_core benchmark::benchmark)

# CUDA kernels, or the stub when there is no CUDA toolkit (every row SKIPPED).
add_executable(bench_gemm_cuda bench_gemm_cuda.cpp)
target_link_libraries(bench_gemm_cuda PRIVATE hpc_math_cuda benchmark::benchmark)

# What each low-precision format costs in accuracy (portable C++).
add_executable(lowp_accuracy experiments/lowp_accuracy.cpp)
target_link_libraries(lowp_accuracy PRIVATE hpc_math_core)

# Low-precision GEMMs (bf16 / fp16 / INT8 / FP8): the CPU kernels of this
# target with their FP32 baselines, and the CUDA ones (or the stub).
add_executable(bench_gemm_lowp bench_gemm_lowp.cpp)
target_link_libraries(bench_gemm_lowp PRIVATE hpc_math_core benchmark::benchmark)
add_executable(bench_gemm_lowp_cuda bench_gemm_lowp_cuda.cpp)
target_link_libraries(bench_gemm_lowp_cuda PRIVATE hpc_math_cuda benchmark::benchmark)

# Experiments behind the articles in docs/articles/; their data and method
# are in docs/experiments/.
if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    add_executable(cache_chase experiments/cache_chase.cpp)
endif()
# These experiments use GNU-style inline assembly, unavailable in MSVC.
if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64|x86_64|AMD64")
    add_executable(fma_peak experiments/fma_peak.cpp)
    target_link_libraries(fma_peak PRIVATE hpc_math_core)
    add_executable(simd_microkernel experiments/simd_microkernel.cpp)
    target_link_libraries(simd_microkernel PRIVATE hpc_math_core)
    add_executable(lowp_peak experiments/lowp_peak.cpp)
    target_link_libraries(lowp_peak PRIVATE hpc_math_core)
endif()
if(CMAKE_CUDA_COMPILER)
    add_executable(cuda_peak experiments/cuda_peak.cu)
    set_target_properties(cuda_peak PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()
if(_HPC_SME_FOUND)
    add_executable(sme_b_layout experiments/sme_b_layout.cpp)
    add_executable(sme_b_stride_probe experiments/sme_b_stride_probe.cpp)
    add_executable(sme_fmopa experiments/sme_fmopa.cpp)
    add_executable(sme_block_sweep experiments/sme_block_sweep.cpp)
    target_link_libraries(sme_block_sweep PRIVATE hpc_math_core)
    target_link_libraries(sme_fmopa PRIVATE hpc_math_core)
    target_link_libraries(sme_b_layout PRIVATE hpc_math_core)
    target_link_libraries(sme_b_stride_probe PRIVATE hpc_math_core)
endif()
