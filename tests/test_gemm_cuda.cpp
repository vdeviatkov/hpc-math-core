/**
 * @file test_gemm_cuda.cpp
 * @brief Google Test correctness suite for CUDA GEMM kernels.
 *
 * All tests are skipped at runtime when no CUDA device is present —
 * the binary compiles and runs cleanly on CPU-only machines.
 *
 * Three kernels verified:
 *   - gemm_cuda_naive
 *   - gemm_cuda_reordered
 *   - gemm_cuda_blocked
 *
 * Each kernel is compared against gemm_naive (CPU scalar) as the reference.
 * Tolerances:
 *   float  : abs tolerance 1e-3, rel tolerance 1e-4
 *   double : abs tolerance 1e-9, rel tolerance 1e-10
 */

#include "gemm/cuda.hpp"
#include "gemm/naive.hpp"
#include "hpc/matrix.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <type_traits>

// ---------------------------------------------------------------------------
// Skip guard — skip all CUDA tests when no GPU is available.
// ---------------------------------------------------------------------------
class CudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (hpc::gemm::cuda_device_count() == 0)
            GTEST_SKIP() << "No CUDA device available on this machine";
    }
};

// ---------------------------------------------------------------------------
// Tolerance helpers
// ---------------------------------------------------------------------------
template <typename T>
constexpr T abs_tol();
template <>
constexpr float abs_tol<float>() { return 1e-3f; }
template <>
constexpr double abs_tol<double>() { return 1e-9; }

template <typename T>
constexpr T rel_tol();
template <>
constexpr float rel_tol<float>() { return 1e-4f; }
template <>
constexpr double rel_tol<double>() { return 1e-10; }

// ---------------------------------------------------------------------------
// Fill matrix with reproducible random values.
// ---------------------------------------------------------------------------
template <typename T>
static void fill_random(hpc::Matrix<T>& M, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(T{-1}, T{1});
    for (std::size_t i = 0; i < M.rows(); ++i)
        for (std::size_t j = 0; j < M.cols(); ++j)
            M(i, j) = dist(rng);
}

// ---------------------------------------------------------------------------
// Element-wise comparison helper.
// ---------------------------------------------------------------------------
template <typename T>
static void expect_near(const hpc::Matrix<T>& got,
                        const hpc::Matrix<T>& ref,
                        const char*           label) {
    ASSERT_EQ(got.rows(), ref.rows());
    ASSERT_EQ(got.cols(), ref.cols());
    for (std::size_t i = 0; i < ref.rows(); ++i) {
        for (std::size_t j = 0; j < ref.cols(); ++j) {
            const T g = got(i, j);
            const T r = ref(i, j);
            const T diff = std::abs(g - r);
            const T scale = std::max(std::abs(r), T{1});
            EXPECT_LT(diff / scale, rel_tol<T>())
                << label << " mismatch at (" << i << "," << j << ")"
                << " got=" << g << " ref=" << r;
        }
    }
}

// ===========================================================================
// Test fixture parametrised over (kernel, size, type)
// ===========================================================================

// ---- float -----------------------------------------------------------------

struct CudaNaiveFloat : CudaTest {};
struct CudaReorderedFloat : CudaTest {};
struct CudaBlockedFloat : CudaTest {};
struct CudaNaiveDouble : CudaTest {};
struct CudaReorderedDouble : CudaTest {};
struct CudaBlockedDouble : CudaTest {};

// Helper macro to avoid repetition.
#define HPC_CUDA_TEST(FIXTURE, KERNEL_FUNC, T, N, SEED_A, SEED_B)         \
    TEST_F(FIXTURE, N##x##N) {                                             \
        hpc::Matrix<T> A(N, N), B(N, N);                                   \
        hpc::Matrix<T> C_ref(N, N), C_got(N, N);                           \
        fill_random(A, SEED_A);                                             \
        fill_random(B, SEED_B);                                             \
        hpc::gemm::gemm_naive(A, B, C_ref);                                \
        hpc::gemm::KERNEL_FUNC(A, B, C_got);                               \
        expect_near(C_got, C_ref, #KERNEL_FUNC "<" #T ">/N=" #N);          \
    }

// gemm_cuda_naive — float
HPC_CUDA_TEST(CudaNaiveFloat,    gemm_cuda_naive,    float,   32,  1, 2)
HPC_CUDA_TEST(CudaNaiveFloat,    gemm_cuda_naive,    float,   64,  3, 4)
HPC_CUDA_TEST(CudaNaiveFloat,    gemm_cuda_naive,    float,  128,  5, 6)
HPC_CUDA_TEST(CudaNaiveFloat,    gemm_cuda_naive,    float,  256,  7, 8)
HPC_CUDA_TEST(CudaNaiveFloat,    gemm_cuda_naive,    float,  512,  9, 10)

// gemm_cuda_reordered — float
HPC_CUDA_TEST(CudaReorderedFloat, gemm_cuda_reordered, float,  32,  1, 2)
HPC_CUDA_TEST(CudaReorderedFloat, gemm_cuda_reordered, float,  64,  3, 4)
HPC_CUDA_TEST(CudaReorderedFloat, gemm_cuda_reordered, float, 128,  5, 6)
HPC_CUDA_TEST(CudaReorderedFloat, gemm_cuda_reordered, float, 256,  7, 8)
HPC_CUDA_TEST(CudaReorderedFloat, gemm_cuda_reordered, float, 512,  9, 10)

// gemm_cuda_blocked — float
HPC_CUDA_TEST(CudaBlockedFloat,   gemm_cuda_blocked,   float,  32,  1, 2)
HPC_CUDA_TEST(CudaBlockedFloat,   gemm_cuda_blocked,   float,  64,  3, 4)
HPC_CUDA_TEST(CudaBlockedFloat,   gemm_cuda_blocked,   float, 128,  5, 6)
HPC_CUDA_TEST(CudaBlockedFloat,   gemm_cuda_blocked,   float, 256,  7, 8)
HPC_CUDA_TEST(CudaBlockedFloat,   gemm_cuda_blocked,   float, 512,  9, 10)

// gemm_cuda_naive — double
HPC_CUDA_TEST(CudaNaiveDouble,    gemm_cuda_naive,    double,  32,  1, 2)
HPC_CUDA_TEST(CudaNaiveDouble,    gemm_cuda_naive,    double,  64,  3, 4)
HPC_CUDA_TEST(CudaNaiveDouble,    gemm_cuda_naive,    double, 128,  5, 6)
HPC_CUDA_TEST(CudaNaiveDouble,    gemm_cuda_naive,    double, 256,  7, 8)
HPC_CUDA_TEST(CudaNaiveDouble,    gemm_cuda_naive,    double, 512,  9, 10)

// gemm_cuda_reordered — double
HPC_CUDA_TEST(CudaReorderedDouble, gemm_cuda_reordered, double,  32,  1, 2)
HPC_CUDA_TEST(CudaReorderedDouble, gemm_cuda_reordered, double,  64,  3, 4)
HPC_CUDA_TEST(CudaReorderedDouble, gemm_cuda_reordered, double, 128,  5, 6)
HPC_CUDA_TEST(CudaReorderedDouble, gemm_cuda_reordered, double, 256,  7, 8)
HPC_CUDA_TEST(CudaReorderedDouble, gemm_cuda_reordered, double, 512,  9, 10)

// gemm_cuda_blocked — double
HPC_CUDA_TEST(CudaBlockedDouble,   gemm_cuda_blocked,   double,  32,  1, 2)
HPC_CUDA_TEST(CudaBlockedDouble,   gemm_cuda_blocked,   double,  64,  3, 4)
HPC_CUDA_TEST(CudaBlockedDouble,   gemm_cuda_blocked,   double, 128,  5, 6)
HPC_CUDA_TEST(CudaBlockedDouble,   gemm_cuda_blocked,   double, 256,  7, 8)
HPC_CUDA_TEST(CudaBlockedDouble,   gemm_cuda_blocked,   double, 512,  9, 10)

#undef HPC_CUDA_TEST

// ---------------------------------------------------------------------------
// Non-square matrices — verify boundary handling in kernel_blocked.
// ---------------------------------------------------------------------------
TEST_F(CudaBlockedFloat, NonSquare_100x200x50) {
    hpc::Matrix<float> A(100, 50), B(50, 200);
    hpc::Matrix<float> C_ref(100, 200), C_got(100, 200);
    fill_random(A, 11);
    fill_random(B, 12);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_blocked(A, B, C_got);
    expect_near(C_got, C_ref, "blocked float non-square");
}

TEST_F(CudaBlockedDouble, NonSquare_77x99x33) {
    hpc::Matrix<double> A(77, 33), B(33, 99);
    hpc::Matrix<double> C_ref(77, 99), C_got(77, 99);
    fill_random(A, 13);
    fill_random(B, 14);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_blocked(A, B, C_got);
    expect_near(C_got, C_ref, "blocked double non-square");
}

