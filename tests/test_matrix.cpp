/**
 * @file test_matrix.cpp
 * @brief Storage invariants GEMM depends on: alignment, value ownership,
 *        empty/moved-from objects and allocation arithmetic overflow.
 */
#include "hpc/matrix.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <utility>

TEST(Matrix, AlignedZeroedStorageAndIndependentCopy) {
    hpc::Matrix<double> a(3, 7);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data()) % hpc::kMatrixAlignment, 0u);
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_EQ(a.data()[i], 0);
    a(2, 6) = 9;
    auto b  = a;
    b(2, 6) = 4;
    EXPECT_EQ(a(2, 6), 9);
    hpc::Matrix<double> c(1, 1);
    c = a;
    EXPECT_EQ(c.rows(), 3u);
    EXPECT_EQ(c.cols(), 7u);
    EXPECT_EQ(c(2, 6), 9);
}

TEST(Matrix, MovesLeaveUsableEmptyObjects) {
    hpc::Matrix<float> a(3, 5);
    a.fill(7);
    auto b = std::move(a);
    EXPECT_EQ(a.size(), 0u);
    a.fill(1);
    a.zero();
    EXPECT_EQ(b(2, 4), 7);
    a = std::move(b);
    EXPECT_EQ(b.size(), 0u);
    b.fill(1);
    b.zero();
    EXPECT_EQ(a(2, 4), 7);
}

TEST(Matrix, EmptyCopiesPreserveShape) {
    hpc::Matrix<float> a(0, 5);
    auto b = a;
    EXPECT_EQ(b.rows(), 0u);
    EXPECT_EQ(b.cols(), 5u);
    hpc::Matrix<float> c(3, 0);
    b = c;
    EXPECT_EQ(b.rows(), 3u);
    EXPECT_EQ(b.cols(), 0u);
}

TEST(Matrix, RejectsOverflowBeforeAllocation) {
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW((hpc::Matrix<double>(max / 2 + 1, 2)), std::bad_array_new_length);
    EXPECT_THROW((hpc::Matrix<double>(1, max / sizeof(double) + 1)), std::bad_array_new_length);
    EXPECT_THROW((hpc::Matrix<float>(1, max / sizeof(float))), std::bad_array_new_length);
}
