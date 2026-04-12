cmake_minimum_required(VERSION 3.25)

add_executable(bench_gemm bench_gemm.cpp)

target_link_libraries(bench_gemm
    PRIVATE
        hpc_math_core
        benchmark::benchmark
)

