include(GoogleTest)

add_executable(hpc_tests test_gemm.cpp)
target_link_libraries(hpc_tests PRIVATE hpc_math_core GTest::gtest_main)

# CUDA kernels, or the stub when there is no CUDA toolkit (tests then skip).
add_executable(hpc_tests_cuda test_gemm_cuda.cpp)
target_link_libraries(hpc_tests_cuda PRIVATE hpc_math_cuda GTest::gtest_main)

# PRE_TEST: list the tests when ctest runs, not at build time, so a binary
# built for another CPU (-march=native on a different machine) is never
# executed during the build.
gtest_discover_tests(hpc_tests DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(hpc_tests_cuda DISCOVERY_MODE PRE_TEST)
