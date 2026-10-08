/**
 * @file test_gemm.cpp
 * @brief Google Test correctness suite for the CPU GEMM kernels.
 *
 * gemm_naive is checked against hand-computed results (2×2 and 3×3 cases,
 * identity, zero); every other kernel is cross-validated against gemm_naive
 * on random matrices, including sizes and shapes that hit tile and vector
 * edges. Comparisons use EXPECT_NEAR with tolerances scaled to the element
 * type and the size of the result.
 *
 * ISA / library families (AVX2, AVX-512, NEON, SVE, SME, AMX, KleidiAI) are
 * tested only where they are compiled (HPC_HAS_* from hpc/isa.hpp);
 * elsewhere their kernels are deleted, so a machine's test count is exactly
 * the set of kernels that ran on it.
 */

#include "gemm/amx.hpp"
#include "gemm/blocked.hpp"
#include "gemm/kleidiai.hpp"
#include "gemm/naive.hpp"
#include "gemm/neon.hpp"
#include "gemm/reordered.hpp"
#include "gemm/sme.hpp"
#include "gemm/sve.hpp"
#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "gemm/avx2.hpp"
#include "gemm/avx512.hpp"

using hpc::MatrixD;
using hpc::MatrixF;

// ---------------------------------------------------------------------------
// Tolerance helpers
// ---------------------------------------------------------------------------

/// Relative epsilon for double-precision comparisons.
static constexpr double kEpsD = 1e-9;

/// Relative epsilon for single-precision comparisons (only the SIMD suites test float).
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

// ===========================================================================
// 6. Blocked kernel tests
// ===========================================================================

TEST(GemmBlocked, MultiplyByIdentityGivesOriginal) {
    constexpr std::size_t N = 32;
    MatrixD A(N, N), I = make_identity(N), C(N, N);
    fill_random(A, 1);

    hpc::gemm::gemm_blocked(A, I, C);

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C(i, j), A(i, j), kEpsD) << "Mismatch at (" << i << ", " << j << ")";
}

TEST(GemmBlocked, MultiplyByZeroGivesZero) {
    constexpr std::size_t N = 16;
    MatrixD A(N, N), Z(N, N), C(N, N);
    fill_random(A, 2);

    hpc::gemm::gemm_blocked(A, Z, C);

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_DOUBLE_EQ(C(i, j), 0.0) << "Expected zero at (" << i << ", " << j << ")";
}

TEST(GemmBlocked, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;

    hpc::gemm::gemm_blocked(A, B, C);

    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmBlocked, RectangularMatrices) {
    MatrixD A(3, 4), B(4, 2), C(3, 2);
    fill_random(A, 7);
    fill_random(B, 8);
    hpc::gemm::gemm_blocked(A, B, C);

    double expected = 0.0;
    for (std::size_t k = 0; k < 4; ++k)
        expected += A(0, k) * B(k, 0);
    EXPECT_NEAR(C(0, 0), expected, kEpsD);
}

// Cross-validate blocked against naive for various sizes, including sizes that
// are not multiples of the tile width (edge-tile handling).
class GemmBlockedCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmBlockedCrossValidation, BlockedMatchesNaive) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_naive(N, N), C_blocked(N, N);
    fill_random(A, 123);
    fill_random(B, 456);

    hpc::gemm::gemm_naive(A, B, C_naive);
    hpc::gemm::gemm_blocked(A, B, C_blocked);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            const double expected = C_naive(i, j);
            const double got      = C_blocked(i, j);
            const double tol      = kEpsD * (1.0 + std::abs(expected));
            EXPECT_NEAR(got, expected, tol) << "N=" << N << " at (" << i << ", " << j << ")";
        }
    }
}

// Include non-power-of-2 sizes to exercise partial (edge) tile handling.
INSTANTIATE_TEST_SUITE_P(Sizes, GemmBlockedCrossValidation,
                         ::testing::Values(std::size_t{4},    // smaller than tile
                                           std::size_t{16},   // smaller than tile
                                           std::size_t{64},   // exactly one tile
                                           std::size_t{100},  // non-power-of-2, partial tiles
                                           std::size_t{128},  // two tiles
                                           std::size_t{256}   // four tiles
                                           ));

#if HPC_HAS_AVX2

// ===========================================================================
// 7. AVX2 Naive  (i-j-k order, SIMD on k-loop)
// ===========================================================================

TEST(GemmAvx2Naive, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;
    hpc::gemm::gemm_avx2_naive(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmAvx2Naive, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f;
    A(0, 1) = 2.f;
    A(1, 0) = 3.f;
    A(1, 1) = 4.f;
    B(0, 0) = 5.f;
    B(0, 1) = 6.f;
    B(1, 0) = 7.f;
    B(1, 1) = 8.f;
    hpc::gemm::gemm_avx2_naive(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF);
    EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF);
    EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

class GemmAvx2NaiveCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmAvx2NaiveCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 11);
    fill_random(B, 22);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx2_naive(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmAvx2NaiveCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 11);
    fill_random(B, 22);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx2_naive(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmAvx2NaiveCrossValidation,
                         ::testing::Values(std::size_t{4}, std::size_t{8}, std::size_t{13},
                                           std::size_t{16}, std::size_t{64}, std::size_t{128}));

// ===========================================================================
// 8. AVX2 Reordered  (i-k-j order, SIMD on j-loop, no blocking)
//    Cache-friendly access, SIMD width benefit without register tiling.
// ===========================================================================

TEST(GemmAvx2Reordered, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;
    hpc::gemm::gemm_avx2_reordered(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmAvx2Reordered, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f;
    A(0, 1) = 2.f;
    A(1, 0) = 3.f;
    A(1, 1) = 4.f;
    B(0, 0) = 5.f;
    B(0, 1) = 6.f;
    B(1, 0) = 7.f;
    B(1, 1) = 8.f;
    hpc::gemm::gemm_avx2_reordered(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF);
    EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF);
    EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

class GemmAvx2ReorderedCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmAvx2ReorderedCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 33);
    fill_random(B, 44);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx2_reordered(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmAvx2ReorderedCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 33);
    fill_random(B, 44);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx2_reordered(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmAvx2ReorderedCrossValidation,
                         ::testing::Values(std::size_t{4}, std::size_t{8}, std::size_t{13},
                                           std::size_t{16}, std::size_t{64}, std::size_t{128}));

// ===========================================================================
// 9. AVX2 Blocked  (tiled i-k-j + register-tiled micro-kernel)
//    Full combination: correct cache access + L2 tiling + no C reload.
// ===========================================================================

TEST(GemmAvx2Blocked, MultiplyByIdentityGivesOriginal) {
    constexpr std::size_t N = 32;
    MatrixD A(N, N), I = make_identity(N), C(N, N);
    fill_random(A, 1);
    hpc::gemm::gemm_avx2_blocked(A, I, C);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C(i, j), A(i, j), kEpsD) << "(" << i << "," << j << ")";
}

TEST(GemmAvx2Blocked, MultiplyByZeroGivesZero) {
    constexpr std::size_t N = 16;
    MatrixD A(N, N), Z(N, N), C(N, N);
    fill_random(A, 2);
    hpc::gemm::gemm_avx2_blocked(A, Z, C);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_DOUBLE_EQ(C(i, j), 0.0);
}

TEST(GemmAvx2Blocked, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;
    hpc::gemm::gemm_avx2_blocked(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmAvx2Blocked, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f;
    A(0, 1) = 2.f;
    A(1, 0) = 3.f;
    A(1, 1) = 4.f;
    B(0, 0) = 5.f;
    B(0, 1) = 6.f;
    B(1, 0) = 7.f;
    B(1, 1) = 8.f;
    hpc::gemm::gemm_avx2_blocked(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF);
    EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF);
    EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

TEST(GemmAvx2Blocked, RectangularMatrices) {
    MatrixD A(3, 4), B(4, 2), C(3, 2);
    fill_random(A, 7);
    fill_random(B, 8);
    hpc::gemm::gemm_avx2_blocked(A, B, C);
    double expected = 0.0;
    for (std::size_t k = 0; k < 4; ++k)
        expected += A(0, k) * B(k, 0);
    EXPECT_NEAR(C(0, 0), expected, kEpsD);
}

class GemmAvx2BlockedCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmAvx2BlockedCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 77);
    fill_random(B, 88);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx2_blocked(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmAvx2BlockedCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 77);
    fill_random(B, 88);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx2_blocked(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

// Include N=13 to exercise the scalar j-tail (13 % 8 != 0, 13 % 16 != 0).
INSTANTIATE_TEST_SUITE_P(Sizes, GemmAvx2BlockedCrossValidation,
                         ::testing::Values(std::size_t{4}, std::size_t{8}, std::size_t{13},
                                           std::size_t{16}, std::size_t{64}, std::size_t{128},
                                           std::size_t{256}));

#endif  // HPC_HAS_AVX2

#if HPC_HAS_AVX512

// ===========================================================================
// 10. AVX-512 Naive  (i-j-k order, 512-bit SIMD on k-loop)
// ===========================================================================

TEST(GemmAvx512Naive, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;
    hpc::gemm::gemm_avx512_naive(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmAvx512Naive, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f;
    A(0, 1) = 2.f;
    A(1, 0) = 3.f;
    A(1, 1) = 4.f;
    B(0, 0) = 5.f;
    B(0, 1) = 6.f;
    B(1, 0) = 7.f;
    B(1, 1) = 8.f;
    hpc::gemm::gemm_avx512_naive(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF);
    EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF);
    EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

class GemmAvx512NaiveCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmAvx512NaiveCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 55);
    fill_random(B, 66);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx512_naive(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmAvx512NaiveCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 55);
    fill_random(B, 66);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx512_naive(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

// N=15: not a multiple of 16 (f32 ZMM width) → exercises scalar tail.
INSTANTIATE_TEST_SUITE_P(Sizes, GemmAvx512NaiveCrossValidation,
                         ::testing::Values(std::size_t{4}, std::size_t{8}, std::size_t{15},
                                           std::size_t{16}, std::size_t{64}, std::size_t{128}));

// ===========================================================================
// 11. AVX-512 Reordered  (i-k-j order, 512-bit SIMD on j-loop, no blocking)
//     Stride-1 access to B and C; 2× FLOP/cycle vs AVX2 reordered.
// ===========================================================================

TEST(GemmAvx512Reordered, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;
    hpc::gemm::gemm_avx512_reordered(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmAvx512Reordered, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f;
    A(0, 1) = 2.f;
    A(1, 0) = 3.f;
    A(1, 1) = 4.f;
    B(0, 0) = 5.f;
    B(0, 1) = 6.f;
    B(1, 0) = 7.f;
    B(1, 1) = 8.f;
    hpc::gemm::gemm_avx512_reordered(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF);
    EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF);
    EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

class GemmAvx512ReorderedCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmAvx512ReorderedCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 99);
    fill_random(B, 111);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx512_reordered(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmAvx512ReorderedCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 99);
    fill_random(B, 111);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx512_reordered(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmAvx512ReorderedCrossValidation,
                         ::testing::Values(std::size_t{4}, std::size_t{8}, std::size_t{15},
                                           std::size_t{16}, std::size_t{64}, std::size_t{128}));

// ===========================================================================
// 12. AVX-512 Blocked  (tiled i-k-j + 512-bit register tile)
//     Full combination: stride-1 + L2 tiling + 4×32 f32 C tile in ZMM regs.
// ===========================================================================

TEST(GemmAvx512Blocked, MultiplyByIdentityGivesOriginal) {
    constexpr std::size_t N = 32;
    MatrixD A(N, N), I = make_identity(N), C(N, N);
    fill_random(A, 1);
    hpc::gemm::gemm_avx512_blocked(A, I, C);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C(i, j), A(i, j), kEpsD) << "(" << i << "," << j << ")";
}

TEST(GemmAvx512Blocked, MultiplyByZeroGivesZero) {
    constexpr std::size_t N = 16;
    MatrixD A(N, N), Z(N, N), C(N, N);
    fill_random(A, 2);
    hpc::gemm::gemm_avx512_blocked(A, Z, C);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_DOUBLE_EQ(C(i, j), 0.0);
}

TEST(GemmAvx512Blocked, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;
    hpc::gemm::gemm_avx512_blocked(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmAvx512Blocked, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f;
    A(0, 1) = 2.f;
    A(1, 0) = 3.f;
    A(1, 1) = 4.f;
    B(0, 0) = 5.f;
    B(0, 1) = 6.f;
    B(1, 0) = 7.f;
    B(1, 1) = 8.f;
    hpc::gemm::gemm_avx512_blocked(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF);
    EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF);
    EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

TEST(GemmAvx512Blocked, RectangularMatrices) {
    MatrixD A(3, 4), B(4, 2), C(3, 2);
    fill_random(A, 7);
    fill_random(B, 8);
    hpc::gemm::gemm_avx512_blocked(A, B, C);
    double expected = 0.0;
    for (std::size_t k = 0; k < 4; ++k)
        expected += A(0, k) * B(k, 0);
    EXPECT_NEAR(C(0, 0), expected, kEpsD);
}

class GemmAvx512BlockedCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmAvx512BlockedCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 123);
    fill_random(B, 456);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx512_blocked(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmAvx512BlockedCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_avx(N, N);
    fill_random(A, 123);
    fill_random(B, 456);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_avx512_blocked(A, B, C_avx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_avx(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

// N=15 → scalar j-tail (not multiple of 32 for f32 or 16 for f64).
// N=31 → also exercises the scalar tail for both types.
INSTANTIATE_TEST_SUITE_P(Sizes, GemmAvx512BlockedCrossValidation,
                         ::testing::Values(std::size_t{4}, std::size_t{8}, std::size_t{15},
                                           std::size_t{16}, std::size_t{31}, std::size_t{64},
                                           std::size_t{128}, std::size_t{256}));

#endif  // HPC_HAS_AVX512

#if HPC_HAS_NEON

// ===========================================================================
// 13. NEON Naive  (i-j-k order, 128-bit SIMD on k-loop)
// ===========================================================================

TEST(GemmNeonNaive, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;
    hpc::gemm::gemm_neon_naive(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmNeonNaive, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f;
    A(0, 1) = 2.f;
    A(1, 0) = 3.f;
    A(1, 1) = 4.f;
    B(0, 0) = 5.f;
    B(0, 1) = 6.f;
    B(1, 0) = 7.f;
    B(1, 1) = 8.f;
    hpc::gemm::gemm_neon_naive(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF);
    EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF);
    EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

class GemmNeonNaiveCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmNeonNaiveCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_neon(N, N);
    fill_random(A, 201);
    fill_random(B, 202);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_neon_naive(A, B, C_neon);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_neon(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmNeonNaiveCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_neon(N, N);
    fill_random(A, 201);
    fill_random(B, 202);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_neon_naive(A, B, C_neon);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_neon(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

// N=3: smaller than NEON width (2 f64 / 4 f32) → pure scalar tail.
// N=5: exercises scalar tail for both f32 and f64.
INSTANTIATE_TEST_SUITE_P(Sizes, GemmNeonNaiveCrossValidation,
                         ::testing::Values(std::size_t{3}, std::size_t{4}, std::size_t{5},
                                           std::size_t{8}, std::size_t{16}, std::size_t{64},
                                           std::size_t{128}));

// ===========================================================================
// 14. NEON Reordered  (i-k-j order, 128-bit SIMD on j-loop, no blocking)
//     Stride-1 B and C access; ~4× scalar f32, ~2× scalar f64.
// ===========================================================================

TEST(GemmNeonReordered, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;
    hpc::gemm::gemm_neon_reordered(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmNeonReordered, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f;
    A(0, 1) = 2.f;
    A(1, 0) = 3.f;
    A(1, 1) = 4.f;
    B(0, 0) = 5.f;
    B(0, 1) = 6.f;
    B(1, 0) = 7.f;
    B(1, 1) = 8.f;
    hpc::gemm::gemm_neon_reordered(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF);
    EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF);
    EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

class GemmNeonReorderedCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmNeonReorderedCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_neon(N, N);
    fill_random(A, 301);
    fill_random(B, 302);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_neon_reordered(A, B, C_neon);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_neon(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmNeonReorderedCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_neon(N, N);
    fill_random(A, 301);
    fill_random(B, 302);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_neon_reordered(A, B, C_neon);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_neon(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmNeonReorderedCrossValidation,
                         ::testing::Values(std::size_t{3}, std::size_t{4}, std::size_t{5},
                                           std::size_t{8}, std::size_t{16}, std::size_t{64},
                                           std::size_t{128}));

// ===========================================================================
// 15. NEON Blocked  (tiled i-k-j + 128-bit register tile)
//     Full combination: stride-1 + L2 tiling + 4×16 f32 C tile in Q regs.
// ===========================================================================

TEST(GemmNeonBlocked, MultiplyByIdentityGivesOriginal) {
    constexpr std::size_t N = 32;
    MatrixD A(N, N), I = make_identity(N), C(N, N);
    fill_random(A, 1);
    hpc::gemm::gemm_neon_blocked(A, I, C);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C(i, j), A(i, j), kEpsD) << "(" << i << "," << j << ")";
}

TEST(GemmNeonBlocked, MultiplyByZeroGivesZero) {
    constexpr std::size_t N = 16;
    MatrixD A(N, N), Z(N, N), C(N, N);
    fill_random(A, 2);
    hpc::gemm::gemm_neon_blocked(A, Z, C);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_DOUBLE_EQ(C(i, j), 0.0);
}

TEST(GemmNeonBlocked, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;
    hpc::gemm::gemm_neon_blocked(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD);
    EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD);
    EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmNeonBlocked, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f;
    A(0, 1) = 2.f;
    A(1, 0) = 3.f;
    A(1, 1) = 4.f;
    B(0, 0) = 5.f;
    B(0, 1) = 6.f;
    B(1, 0) = 7.f;
    B(1, 1) = 8.f;
    hpc::gemm::gemm_neon_blocked(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF);
    EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF);
    EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

TEST(GemmNeonBlocked, RectangularMatrices) {
    MatrixD A(3, 4), B(4, 2), C(3, 2);
    fill_random(A, 7);
    fill_random(B, 8);
    hpc::gemm::gemm_neon_blocked(A, B, C);
    double expected = 0.0;
    for (std::size_t k = 0; k < 4; ++k)
        expected += A(0, k) * B(k, 0);
    EXPECT_NEAR(C(0, 0), expected, kEpsD);
}

class GemmNeonBlockedCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmNeonBlockedCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_neon(N, N);
    fill_random(A, 401);
    fill_random(B, 402);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_neon_blocked(A, B, C_neon);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_neon(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmNeonBlockedCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_neon(N, N);
    fill_random(A, 401);
    fill_random(B, 402);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_neon_blocked(A, B, C_neon);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_neon(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

// N=3: smaller than f64 NEON width (2 lanes) → scalar tail.
// N=5: exercises tail for f32 (5 % 4 != 0) and f64 (5 % 2 != 0 after 4).
// N=11: exercises tail for f32 (11 % 16 != 0) and f64 (11 % 4 != 0).
INSTANTIATE_TEST_SUITE_P(Sizes, GemmNeonBlockedCrossValidation,
                         ::testing::Values(std::size_t{3}, std::size_t{4}, std::size_t{5},
                                           std::size_t{8}, std::size_t{11}, std::size_t{16},
                                           std::size_t{64}, std::size_t{128}, std::size_t{256}));

#endif  // HPC_HAS_NEON

#if HPC_HAS_SVE

// ===========================================================================
// 16. SVE Naive  (i-j-k order, VLA SIMD on k-loop)
// ===========================================================================

TEST(GemmSveNaive, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1; A(0, 1) = 2; A(1, 0) = 3; A(1, 1) = 4;
    B(0, 0) = 5; B(0, 1) = 6; B(1, 0) = 7; B(1, 1) = 8;
    hpc::gemm::gemm_sve_naive(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD); EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD); EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmSveNaive, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f; A(0, 1) = 2.f; A(1, 0) = 3.f; A(1, 1) = 4.f;
    B(0, 0) = 5.f; B(0, 1) = 6.f; B(1, 0) = 7.f; B(1, 1) = 8.f;
    hpc::gemm::gemm_sve_naive(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF); EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF); EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

class GemmSveNaiveCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmSveNaiveCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_sve(N, N);
    fill_random(A, 501); fill_random(B, 502);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_sve_naive(A, B, C_sve);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_sve(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmSveNaiveCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_sve(N, N);
    fill_random(A, 501); fill_random(B, 502);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_sve_naive(A, B, C_sve);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_sve(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

// N=3 / N=7 exercise scalar tails for both precisions on all SVE VL variants.
INSTANTIATE_TEST_SUITE_P(Sizes, GemmSveNaiveCrossValidation,
                         ::testing::Values(std::size_t{3},  std::size_t{4},
                                           std::size_t{7},  std::size_t{8},
                                           std::size_t{16}, std::size_t{64},
                                           std::size_t{128}));

// ===========================================================================
// 17. SVE Reordered  (i-k-j, VLA j-loop with predicated tail)
//     No scalar j-tail: svwhilelt predicates handle the remainder lanes.
// ===========================================================================

TEST(GemmSveReordered, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1; A(0, 1) = 2; A(1, 0) = 3; A(1, 1) = 4;
    B(0, 0) = 5; B(0, 1) = 6; B(1, 0) = 7; B(1, 1) = 8;
    hpc::gemm::gemm_sve_reordered(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD); EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD); EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmSveReordered, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f; A(0, 1) = 2.f; A(1, 0) = 3.f; A(1, 1) = 4.f;
    B(0, 0) = 5.f; B(0, 1) = 6.f; B(1, 0) = 7.f; B(1, 1) = 8.f;
    hpc::gemm::gemm_sve_reordered(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF); EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF); EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

class GemmSveReorderedCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmSveReorderedCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_sve(N, N);
    fill_random(A, 601); fill_random(B, 602);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_sve_reordered(A, B, C_sve);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_sve(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmSveReorderedCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_sve(N, N);
    fill_random(A, 601); fill_random(B, 602);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_sve_reordered(A, B, C_sve);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_sve(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

// N=7 / N=13 are not multiples of any common VL (4, 8, 16) → exercises the
// predicated tail on every known SVE implementation width.
INSTANTIATE_TEST_SUITE_P(Sizes, GemmSveReorderedCrossValidation,
                         ::testing::Values(std::size_t{3},  std::size_t{4},
                                           std::size_t{7},  std::size_t{8},
                                           std::size_t{13}, std::size_t{16},
                                           std::size_t{64}, std::size_t{128}));

// ===========================================================================
// 18. SVE Blocked  (tiled i-k-j + VLA register tile)
//     Tile width = kSveRegCols × svcntw/d() — scales automatically with VL.
//     All tail handling done via predicates — no scalar j-tail loop.
// ===========================================================================

TEST(GemmSveBlocked, MultiplyByIdentityGivesOriginal) {
    constexpr std::size_t N = 32;
    MatrixD A(N, N), I = make_identity(N), C(N, N);
    fill_random(A, 1);
    hpc::gemm::gemm_sve_blocked(A, I, C);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C(i, j), A(i, j), kEpsD) << "(" << i << "," << j << ")";
}

TEST(GemmSveBlocked, MultiplyByZeroGivesZero) {
    constexpr std::size_t N = 16;
    MatrixD A(N, N), Z(N, N), C(N, N);
    fill_random(A, 2);
    hpc::gemm::gemm_sve_blocked(A, Z, C);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_DOUBLE_EQ(C(i, j), 0.0);
}

TEST(GemmSveBlocked, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1; A(0, 1) = 2; A(1, 0) = 3; A(1, 1) = 4;
    B(0, 0) = 5; B(0, 1) = 6; B(1, 0) = 7; B(1, 1) = 8;
    hpc::gemm::gemm_sve_blocked(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD); EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD); EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmSveBlocked, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f; A(0, 1) = 2.f; A(1, 0) = 3.f; A(1, 1) = 4.f;
    B(0, 0) = 5.f; B(0, 1) = 6.f; B(1, 0) = 7.f; B(1, 1) = 8.f;
    hpc::gemm::gemm_sve_blocked(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF); EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF); EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

TEST(GemmSveBlocked, RectangularMatrices) {
    MatrixD A(3, 4), B(4, 2), C(3, 2);
    fill_random(A, 7); fill_random(B, 8);
    hpc::gemm::gemm_sve_blocked(A, B, C);
    double expected = 0.0;
    for (std::size_t k = 0; k < 4; ++k) expected += A(0, k) * B(k, 0);
    EXPECT_NEAR(C(0, 0), expected, kEpsD);
}

class GemmSveBlockedCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmSveBlockedCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_sve(N, N);
    fill_random(A, 701); fill_random(B, 702);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_sve_blocked(A, B, C_sve);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_sve(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmSveBlockedCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_sve(N, N);
    fill_random(A, 701); fill_random(B, 702);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_sve_blocked(A, B, C_sve);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_sve(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

// N=7 / N=13 / N=17 — not multiples of 4, 8, or 16 → predicated tail on every
// SVE VL variant (128, 256, 512-bit).
INSTANTIATE_TEST_SUITE_P(Sizes, GemmSveBlockedCrossValidation,
                         ::testing::Values(std::size_t{3},  std::size_t{4},
                                           std::size_t{7},  std::size_t{8},
                                           std::size_t{13}, std::size_t{16},
                                           std::size_t{17}, std::size_t{64},
                                           std::size_t{128}, std::size_t{256}));

#endif  // HPC_HAS_SVE

#if HPC_HAS_SME

// ===========================================================================
// 19. SME  (packed A+B, GotoBLAS cache blocking, all ZA tiles: 2×2 za32 for
//     f32, 2×4 za64 for f64 — see gemm/sme.hpp)
// ===========================================================================

/// gemm_sme vs gemm_naive on an M×K·K×N problem. C is pre-filled with junk
/// to prove the kernel overwrites (rather than accumulates into) C.
template <typename T>
static void expect_sme_matches_naive(std::size_t M, std::size_t K, std::size_t N, unsigned seed) {
    hpc::Matrix<T> A(M, K), B(K, N), C_ref(M, N), C_sme(M, N);
    fill_random(A, seed, T{-1}, T{1});
    fill_random(B, seed + 1, T{-1}, T{1});
    C_sme.fill(T{123});
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_sme(A, B, C_sme);
    const T tol = sizeof(T) == 4 ? T(1e-4) : T(1e-10);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
            ASSERT_NEAR(C_sme(i, j), C_ref(i, j), tol * (T{1} + std::abs(C_ref(i, j))))
                << (sizeof(T) == 4 ? "f32 " : "f64 ") << M << "x" << K << "x" << N << " (" << i
                << "," << j << ")";
}

TEST(GemmSme, MultiplyByIdentityGivesOriginal) {
    constexpr std::size_t N = 40;
    MatrixD A(N, N), I = make_identity(N), C(N, N);
    fill_random(A, 1);
    hpc::gemm::gemm_sme(A, I, C);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C(i, j), A(i, j), kEpsD) << "(" << i << "," << j << ")";
}

TEST(GemmSme, MultiplyByZeroGivesZero) {
    constexpr std::size_t N = 16;
    MatrixD A(N, N), Z(N, N), C(N, N);
    fill_random(A, 2);
    C.fill(7.0);
    hpc::gemm::gemm_sme(A, Z, C);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_DOUBLE_EQ(C(i, j), 0.0);
}

TEST(GemmSme, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1; A(0, 1) = 2; A(1, 0) = 3; A(1, 1) = 4;
    B(0, 0) = 5; B(0, 1) = 6; B(1, 0) = 7; B(1, 1) = 8;
    hpc::gemm::gemm_sme(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD); EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD); EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmSme, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f; A(0, 1) = 2.f; A(1, 0) = 3.f; A(1, 1) = 4.f;
    B(0, 0) = 5.f; B(0, 1) = 6.f; B(1, 0) = 7.f; B(1, 1) = 8.f;
    hpc::gemm::gemm_sme(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF); EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF); EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

TEST(GemmSme, EmptyInnerDimensionZeroesC) {
    MatrixD A(3, 0), B(0, 4), C(3, 4);
    C.fill(5.0);
    hpc::gemm::gemm_sme(A, B, C);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_DOUBLE_EQ(C(i, j), 0.0);
}

// Rectangular shapes chosen to hit every edge path: partial micro-tile rows
// (M not a multiple of 2·SVL) and only the top row of tiles in use (M ≤ SVL),
// partial/empty right-hand tiles (N not a multiple of 2·SVL / 4·SVL), K
// below the NEON transpose width, and K/M/N crossing the kSmeKc / kSmeMc /
// kSmeNc block boundaries — the K crossing exercises the reload of partial C
// sums into ZA.
struct SmeShape {
    std::size_t M, K, N;
};

class GemmSmeShapes : public ::testing::TestWithParam<SmeShape> {};

TEST_P(GemmSmeShapes, MatchesNaiveFloat) {
    const auto [M, K, N] = GetParam();
    expect_sme_matches_naive<float>(M, K, N, 1201);
}

TEST_P(GemmSmeShapes, MatchesNaiveDouble) {
    const auto [M, K, N] = GetParam();
    expect_sme_matches_naive<double>(M, K, N, 1203);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, GemmSmeShapes,
    ::testing::Values(SmeShape{1, 1, 1}, SmeShape{3, 5, 2}, SmeShape{7, 3, 9},
                      SmeShape{8, 8, 8}, SmeShape{15, 16, 17}, SmeShape{16, 16, 16},
                      SmeShape{17, 33, 15}, SmeShape{33, 1, 65}, SmeShape{32, 64, 32},
                      SmeShape{63, 65, 64}, SmeShape{129, 7, 33}, SmeShape{257, 64, 31},
                      SmeShape{40, hpc::gemm::kSmeKc + 3, 50},
                      SmeShape{20, 2 * hpc::gemm::kSmeKc + 17, 36},
                      SmeShape{hpc::gemm::kSmeMc + 5, 70, 40},
                      SmeShape{9, 11, hpc::gemm::kSmeNc + 7}),
    [](const ::testing::TestParamInfo<SmeShape>& info) {
        return std::to_string(info.param.M) + "x" + std::to_string(info.param.K) + "x" +
               std::to_string(info.param.N);
    });

// pack_a's NEON transpose relabels lanes with vreinterpretq (float <-> 64-bit
// lane views) to move float pairs as one unit. It must be a pure bit move:
// feed random bit patterns (NaN payloads, denormals, -0) and compare bytes.
template <typename T>
static void expect_pack_a_bit_exact() {
    using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
    const std::size_t mr = 2 * (sizeof(T) == 4 ? svcntsw() : svcntsd());
    std::mt19937_64 rng(77);
    for (std::size_t mc : {1, 3, 4, 5, 31, 33}) {
        for (std::size_t kc : {1, 3, 4, 7, 64}) {
            const std::size_t lda = kc + 3, strips = (mc + mr - 1) / mr;
            std::vector<T> A(mc * lda), got(strips * mr * kc), ref(strips * mr * kc);
            for (auto& x : A) {
                const Bits b = static_cast<Bits>(rng());
                std::memcpy(&x, &b, sizeof x);
            }
            hpc::gemm::sme_detail::pack_a(A.data(), lda, mc, kc, mr, got.data());
            for (std::size_t i = 0; i < strips * mr; ++i)
                for (std::size_t k = 0; k < kc; ++k)
                    ref[(i / mr) * mr * kc + k * mr + i % mr] = i < mc ? A[i * lda + k] : T{0};
            EXPECT_EQ(std::memcmp(got.data(), ref.data(), got.size() * sizeof(T)), 0)
                << "mc=" << mc << " kc=" << kc;
        }
    }
}

TEST(GemmSme, PackAIsBitExactFloat) { expect_pack_a_bit_exact<float>(); }
TEST(GemmSme, PackAIsBitExactDouble) { expect_pack_a_bit_exact<double>(); }

class GemmSmeCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmSmeCrossValidation, MatchesNaiveFloat) {
    expect_sme_matches_naive<float>(GetParam(), GetParam(), GetParam(), 1001);
}

TEST_P(GemmSmeCrossValidation, MatchesNaiveDouble) {
    expect_sme_matches_naive<double>(GetParam(), GetParam(), GetParam(), 1001);
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmSmeCrossValidation,
                         ::testing::Values(std::size_t{31}, std::size_t{64}, std::size_t{100},
                                           std::size_t{128}, std::size_t{256}));

#endif  // HPC_HAS_SME

#if HPC_HAS_AMX

// ===========================================================================
// 20. AMX (Apple AMX, via Accelerate.framework)
//
// gemm_amx is a thin wrapper around Accelerate's cblas_sgemm / cblas_dgemm
// — see src/gemm/amx.hpp. Full fp32/fp64 precision throughout (unlike
// gemm_cuda_wmma, Accelerate does not truncate to fp16), so tolerances match
// the tight ones used by every other CPU family.
// ===========================================================================

TEST(GemmAmx, KnownResult2x2) {
    MatrixD A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1; A(0, 1) = 2; A(1, 0) = 3; A(1, 1) = 4;
    B(0, 0) = 5; B(0, 1) = 6; B(1, 0) = 7; B(1, 1) = 8;
    hpc::gemm::gemm_amx(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.0, kEpsD); EXPECT_NEAR(C(0, 1), 22.0, kEpsD);
    EXPECT_NEAR(C(1, 0), 43.0, kEpsD); EXPECT_NEAR(C(1, 1), 50.0, kEpsD);
}

TEST(GemmAmx, FloatKnownResult2x2) {
    hpc::MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f; A(0, 1) = 2.f; A(1, 0) = 3.f; A(1, 1) = 4.f;
    B(0, 0) = 5.f; B(0, 1) = 6.f; B(1, 0) = 7.f; B(1, 1) = 8.f;
    hpc::gemm::gemm_amx(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF); EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF); EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

TEST(GemmAmx, RectangularAndLargeKMatrices) {
    // Non-square, large-K case: Accelerate's internal blocking is opaque to
    // us, so this is the closest equivalent to the other families'
    // K-tile-boundary tests.
    const std::size_t M = 24, N = 40, K = 777;
    MatrixD A(M, K), B(K, N), C_ref(M, N), C_amx(M, N);
    fill_random(A, 1113); fill_random(B, 1114);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_amx(A, B, C_amx);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_amx(i, j), C_ref(i, j), 1e-6 * (1.0 + std::abs(C_ref(i, j))))
                << "(" << i << "," << j << ")";
}

class GemmAmxCrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmAmxCrossValidation, MatchesNaiveDouble) {
    const std::size_t N = GetParam();
    MatrixD A(N, N), B(N, N), C_ref(N, N), C_amx(N, N);
    fill_random(A, 1101); fill_random(B, 1102);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_amx(A, B, C_amx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_amx(i, j), C_ref(i, j), 1e-8 * (1.0 + std::abs(C_ref(i, j))))
                << "f64 N=" << N << " (" << i << "," << j << ")";
}

TEST_P(GemmAmxCrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    hpc::MatrixF A(N, N), B(N, N), C_ref(N, N), C_amx(N, N);
    fill_random(A, 1103); fill_random(B, 1104);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_amx(A, B, C_amx);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(C_amx(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmAmxCrossValidation,
                         ::testing::Values(std::size_t{1},  std::size_t{15},
                                           std::size_t{16}, std::size_t{17},
                                           std::size_t{33}, std::size_t{64},
                                           std::size_t{65}, std::size_t{128},
                                           std::size_t{256}));

#endif  // HPC_HAS_AMX

#if HPC_HAS_KLEIDIAI

// ===========================================================================
// 21. KleidiAI (reference, SME2 f32 only — src/gemm/kleidiai.hpp)
// ===========================================================================

TEST(GemmKleidiAI, FloatKnownResult2x2) {
    MatrixF A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1.f; A(0, 1) = 2.f; A(1, 0) = 3.f; A(1, 1) = 4.f;
    B(0, 0) = 5.f; B(0, 1) = 6.f; B(1, 0) = 7.f; B(1, 1) = 8.f;
    hpc::gemm::gemm_kleidiai(A, B, C);
    EXPECT_NEAR(C(0, 0), 19.f, kEpsF); EXPECT_NEAR(C(0, 1), 22.f, kEpsF);
    EXPECT_NEAR(C(1, 0), 43.f, kEpsF); EXPECT_NEAR(C(1, 1), 50.f, kEpsF);
}

class GemmKleidiAICrossValidation : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmKleidiAICrossValidation, MatchesNaiveFloat) {
    const std::size_t N = GetParam();
    MatrixF A(N, N + 3), B(N + 3, N + 1), C_ref(N, N + 1), C_kai(N, N + 1);
    fill_random(A, 1401, -1.f, 1.f); fill_random(B, 1402, -1.f, 1.f);
    C_kai.fill(123.f);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_kleidiai(A, B, C_kai);
    for (std::size_t i = 0; i < C_ref.rows(); ++i)
        for (std::size_t j = 0; j < C_ref.cols(); ++j)
            EXPECT_NEAR(C_kai(i, j), C_ref(i, j), 1e-4f * (1.0f + std::abs(C_ref(i, j))))
                << "f32 N=" << N << " (" << i << "," << j << ")";
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmKleidiAICrossValidation,
                         ::testing::Values(std::size_t{1}, std::size_t{15}, std::size_t{33},
                                           std::size_t{64}, std::size_t{129}));

#endif  // HPC_HAS_KLEIDIAI
