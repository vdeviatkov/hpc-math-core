cmake_minimum_required(VERSION 3.25)

# ---------------------------------------------------------------------------
# CUDA GEMM test binary
#
# hpc_tests_cuda links against hpc_math_cuda (real or stub).
# All tests call GTEST_SKIP() when cuda_device_count() == 0, so this
# binary compiles and passes on CPU-only machines.
# ---------------------------------------------------------------------------
add_executable(hpc_tests_cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../test_gemm_cuda.cpp
)

target_link_libraries(hpc_tests_cuda
    PRIVATE
        hpc_math_cuda
        hpc_math_core
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(hpc_tests_cuda)

