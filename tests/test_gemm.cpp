/**
 * @file test_gemm.cpp
 * @brief Google Test correctness suite for GEMM kernels.
 *
 * Test strategy
 * =============
 * 1. Identity test      — A × I = A for all kernels.
 * 2. Zero test          — A × 0 = 0 for all kernels.
 * 3. Small known result — hand-computable 2×2 and 3×3 cases.
 * 4. Cross-validation   — for larger random matrices, assert that
 *                         gemm_reordered produces the same result as
 *                         gemm_naive (which itself is validated above).
 *
 * All floating-point comparisons use EXPECT_NEAR with an epsilon that
 * accounts for double-precision rounding in the accumulation.
 */

#include "gemm/naive.hpp"
#include "gemm/reordered.hpp"
#include "hpc/matrix.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>

using hpc::MatrixD;
using hpc::MatrixF;

// ---------------------------------------------------------------------------
// Tolerance helpers
// ---------------------------------------------------------------------------

/// Relative epsilon for double-precision comparisons.
static constexpr double kEpsD = 1e-9;

/// Relative epsilon for single-precision comparisons (reserved for Step 2 float benchmarks).
[[maybe_unused]] static constexpr float kEpsF = 1e-4f;

// ---------------------------------------------------------------------------
// Utility: fill a matrix with random values using a fixed seed.
// ---------------------------------------------------------------------------
template <typename T>
static void fill_random(hpc::Matrix<T>& M, unsigned seed = 42, T lo = T{0}, T hi = T{1}) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(lo, hi);
    for (std::size_t i = 0; i < M.rows(); ++i)
        for (std::size_t j = 0; j < M.cols(); ++j)
            M(i, j) = dist(rng);
}

// ---------------------------------------------------------------------------
// Test fixture shared by all GEMM variants
// ---------------------------------------------------------------------------

/// Helper: build an N×N identity matrix.
static MatrixD make_identity(std::size_t N) {
    MatrixD I(N, N);
    for (std::size_t i = 0; i < N; ++i)
        I(i, i) = 1.0;
    return I;
}

// ===========================================================================
// 1. Identity tests
// ===========================================================================

TEST(GemmNaive, MultiplyByIdentityGivesOriginal) {
    constexpr std::size_t N = 32;
    MatrixD A(N, N), I = make_identity(N), C(N, N);
    fill_random(A, 1);

    hpc::gemm::gemm_naive(A, I, C);

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C(i, j), A(i, j), kEpsD) << "Mismatch at (" << i << ", " << j << ")";
}

TEST(GemmReordered, MultiplyByIdentityGivesOriginal) {
    constexpr std::size_t N = 32;
    MatrixD A(N, N), I = make_identity(N), C(N, N);
    fill_random(A, 1);

    hpc::gemm::gemm_reordered(A, I, C);

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C(i, j), A(i, j), kEpsD) << "Mismatch at (" << i << ", " << j << ")";
}

// ===========================================================================
// 2. Zero matrix tests
// ===========================================================================

TEST(GemmNaive, MultiplyByZeroGivesZero) {
    constexpr std::size_t N = 16;
    MatrixD A(N, N), Z(N, N), C(N, N);
    fill_random(A, 2);
    // Z is already zero-initialised by the Matrix constructor.

    hpc::gemm::gemm_naive(A, Z, C);

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_DOUBLE_EQ(C(i, j), 0.0) << "Expected zero at (" << i << ", " << j << ")";
}

TEST(GemmReordered, MultiplyByZeroGivesZero) {
    constexpr std::size_t N = 16;
    MatrixD A(N, N), Z(N, N), C(N, N);
    fill_random(A, 2);

    hpc::gemm::gemm_reordered(A, Z, C);

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_DOUBLE_EQ(C(i, j), 0.0) << "Expected zero at (" << i << ", " << j << ")";
}

// ===========================================================================
// 3. Small known-result tests
// ===========================================================================

TEST(GemmNaive, KnownResult2x2) {
    //  A = [1 2]   B = [5 6]   C = [19 22]
    //      [3 4]       [7 8]       [43 50]
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;

    hpc::gemm::gemm_naive(A, B, C);

    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmReordered, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;

    hpc::gemm::gemm_reordered(A, B, C);

    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmNaive, KnownResult3x3) {
    //  A = [1 0 0]   B = [1 2 3]   C = A × B = B  (A is identity)
    //      [0 1 0]       [4 5 6]
    //      [0 0 1]       [7 8 9]
    MatrixD A(3, 3), B(3, 3), C(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        A(i, i) = 1.0;
    double vals[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            B(i, j) = vals[i * 3 + j];

    hpc::gemm::gemm_naive(A, B, C);

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(C(i, j), B(i, j), kEpsD);
}

// ===========================================================================
// 4. Cross-validation: reordered must match naive for random matrices
// ===========================================================================

class GemmCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmCrossValidation, ReorderedMatchesNaive) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_naive(N, N), C_reordered(N, N);
    fill_random(A, 123);
    fill_random(B, 456);

    hpc::gemm::gemm_naive(A, B, C_naive);
    hpc::gemm::gemm_reordered(A, B, C_reordered);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            // Use a relative tolerance scaled by the magnitude of the result.
            const double expected = C_naive(i, j);
            const double got      = C_reordered(i, j);
            const double tol      = kEpsD * (1.0 + std::abs(expected));
            EXPECT_NEAR(got, expected, tol) << "N=" << N << " at (" << i << ", " << j << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmCrossValidation,
                         ::testing::Values(std::size_t{4}, std::size_t{16}, std::size_t{64},
                                           std::size_t{128}));

// ===========================================================================
// 5. Non-square (rectangular) matrix test
// ===========================================================================

TEST(GemmNaive, RectangularMatrices) {
    // A: 3×4,  B: 4×2  →  C: 3×2
    MatrixD A(3, 4), B(4, 2), C(3, 2);
    fill_random(A, 7);
    fill_random(B, 8);
    hpc::gemm::gemm_naive(A, B, C);

    // Verify C(0,0) manually.
    double expected = 0.0;
    for (std::size_t k = 0; k < 4; ++k)
        expected += A(0, k) * B(k, 0);
    EXPECT_NEAR(C(0, 0), expected, kEpsD);
}

TEST(GemmReordered, RectangularMatrices) {
    MatrixD A(3, 4), B(4, 2), C(3, 2);
    fill_random(A, 7);
    fill_random(B, 8);
    hpc::gemm::gemm_reordered(A, B, C);

    double expected = 0.0;
    for (std::size_t k = 0; k < 4; ++k)
        expected += A(0, k) * B(k, 0);
    EXPECT_NEAR(C(0, 0), expected, kEpsD);
}
