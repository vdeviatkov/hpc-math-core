cmake_minimum_required(VERSION 3.25)

add_executable(hpc_tests test_gemm.cpp)

target_link_libraries(hpc_tests
    PRIVATE
        hpc_math_core
        GTest::gtest_main
)

# Automatically register tests with CTest so that `ctest` discovers them
# without needing to enumerate each TEST() by hand.
#
# DISCOVERY_MODE PRE_TEST: defer the --gtest_list_tests binary invocation to
# ctest run-time rather than CMake configure/link time. Without this,
# gtest_discover_tests runs the binary on the build machine to enumerate
# test names — which causes SIGILL on CI when the binary was compiled with
# -march=native for a different architecture (e.g. built locally on Apple M,
# discovered on a Linux x86-64 runner).
include(GoogleTest)
gtest_discover_tests(hpc_tests DISCOVERY_MODE PRE_TEST)

