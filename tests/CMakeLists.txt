cmake_minimum_required(VERSION 3.25)

add_executable(hpc_tests test_gemm.cpp)

target_link_libraries(hpc_tests
    PRIVATE
        hpc_math_core
        GTest::gtest_main
)

# Automatically register tests with CTest so that `ctest` discovers them
# without needing to enumerate each TEST() by hand.
include(GoogleTest)
gtest_discover_tests(hpc_tests)

