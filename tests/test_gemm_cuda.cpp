/**
 * @file test_gemm_cuda.cpp
 * @brief Google Test correctness suite for all CUDA GEMM kernels.
 *
 * Kernels verified:
 *   gemm_cuda_naive, gemm_cuda_reordered, gemm_cuda_blocked  (Levels 0-1)
 *   gemm_cuda_reg_tile, gemm_cuda_double_buf                  (Levels 2-3)
 *   gemm_cuda_wmma                                            (Level 4, fp32 only)
 *
 * All tests skip at runtime when no CUDA device is present.
 * gemm_cuda_wmma additionally skips when Tensor Cores are unavailable.
 *
 * Tolerances:
 *   float  (SIMT): rel 1e-4, abs 1e-3
 *   double (SIMT): rel 1e-10, abs 1e-9
 *   float  (WMMA): rel 1e-2, abs 1e-2  -- fp16 conversion introduces ~1e-3 error
 */

#include "gemm/cuda.hpp"
#include "gemm/naive.hpp"
#include "hpc/matrix.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>

// ---------------------------------------------------------------------------
// Skip guards
// ---------------------------------------------------------------------------
class CudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (hpc::gemm::cuda_device_count() == 0)
            GTEST_SKIP() << "No CUDA device available on this machine";
    }
};

class CudaTensorCoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (hpc::gemm::cuda_device_count() == 0)
            GTEST_SKIP() << "No CUDA device available on this machine";
        if (!hpc::gemm::cuda_has_tensor_cores())
            GTEST_SKIP() << "Tensor Cores not available (requires sm_70+)";
    }
};

// ---------------------------------------------------------------------------
// Tolerance helpers
// ---------------------------------------------------------------------------
template <typename T> constexpr T abs_tol();
template <> constexpr float  abs_tol<float>()  { return 1e-3f; }
template <> constexpr double abs_tol<double>() { return 1e-9; }

template <typename T> constexpr T rel_tol();
template <> constexpr float  rel_tol<float>()  { return 1e-4f; }
template <> constexpr double rel_tol<double>() { return 1e-10; }

// ---------------------------------------------------------------------------
// Fill / compare helpers
// ---------------------------------------------------------------------------
template <typename T>
static void fill_random(hpc::Matrix<T>& M, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(T{-1}, T{1});
    for (std::size_t i = 0; i < M.rows(); ++i)
        for (std::size_t j = 0; j < M.cols(); ++j)
            M(i, j) = dist(rng);
}

template <typename T>
static void expect_near(const hpc::Matrix<T>& got, const hpc::Matrix<T>& ref,
                        const char* label,
                        double rel = -1.0, double abs_t = -1.0) {
    ASSERT_EQ(got.rows(), ref.rows());
    ASSERT_EQ(got.cols(), ref.cols());
    const double r = (rel   < 0) ? double(rel_tol<T>()) : rel;
    const double a = (abs_t < 0) ? double(abs_tol<T>()) : abs_t;
    for (std::size_t i = 0; i < ref.rows(); ++i) {
        for (std::size_t j = 0; j < ref.cols(); ++j) {
            const double g    = double(got(i, j));
            const double rf   = double(ref(i, j));
            const double diff = std::abs(g - rf);
            const double scale = std::max(std::abs(rf), 1.0);
            EXPECT_TRUE(diff / scale < r || diff < a)
                << label << " mismatch at (" << i << "," << j << ")"
                << " got=" << g << " ref=" << rf
                << " rel_err=" << diff/scale << " (tol=" << r << ")";
        }
    }
}

// ===========================================================================
// Macro: generate test fixture + test body
// ===========================================================================
#define HPC_CUDA_TEST(FIXTURE, KERNEL_FUNC, T, N, SA, SB)    \
    TEST_F(FIXTURE, N##x##N) {                                \
        hpc::Matrix<T> A(N, N), B(N, N);                      \
        hpc::Matrix<T> C_ref(N, N), C_got(N, N);              \
        fill_random(A, SA); fill_random(B, SB);               \
        hpc::gemm::gemm_naive(A, B, C_ref);                   \
        hpc::gemm::KERNEL_FUNC(A, B, C_got);                  \
        expect_near(C_got, C_ref, #KERNEL_FUNC "<" #T ">/N=" #N); \
    }

// ===========================================================================
// Level 0 -- Naive
// ===========================================================================
struct CudaNaiveFloat   : CudaTest {};
struct CudaNaiveDouble  : CudaTest {};
HPC_CUDA_TEST(CudaNaiveFloat,  gemm_cuda_naive, float,   32, 1, 2)
HPC_CUDA_TEST(CudaNaiveFloat,  gemm_cuda_naive, float,   64, 3, 4)
HPC_CUDA_TEST(CudaNaiveFloat,  gemm_cuda_naive, float,  128, 5, 6)
HPC_CUDA_TEST(CudaNaiveFloat,  gemm_cuda_naive, float,  256, 7, 8)
HPC_CUDA_TEST(CudaNaiveDouble, gemm_cuda_naive, double,  32, 1, 2)
HPC_CUDA_TEST(CudaNaiveDouble, gemm_cuda_naive, double,  64, 3, 4)
HPC_CUDA_TEST(CudaNaiveDouble, gemm_cuda_naive, double, 128, 5, 6)
HPC_CUDA_TEST(CudaNaiveDouble, gemm_cuda_naive, double, 256, 7, 8)

// ===========================================================================
// Level 0b -- Reordered
// ===========================================================================
struct CudaReorderedFloat  : CudaTest {};
struct CudaReorderedDouble : CudaTest {};
HPC_CUDA_TEST(CudaReorderedFloat,  gemm_cuda_reordered, float,   32, 1, 2)
HPC_CUDA_TEST(CudaReorderedFloat,  gemm_cuda_reordered, float,   64, 3, 4)
HPC_CUDA_TEST(CudaReorderedFloat,  gemm_cuda_reordered, float,  128, 5, 6)
HPC_CUDA_TEST(CudaReorderedFloat,  gemm_cuda_reordered, float,  256, 7, 8)
HPC_CUDA_TEST(CudaReorderedDouble, gemm_cuda_reordered, double,  32, 1, 2)
HPC_CUDA_TEST(CudaReorderedDouble, gemm_cuda_reordered, double, 128, 5, 6)

// ===========================================================================
// Level 1 -- Blocked
// ===========================================================================
struct CudaBlockedFloat  : CudaTest {};
struct CudaBlockedDouble : CudaTest {};
HPC_CUDA_TEST(CudaBlockedFloat,  gemm_cuda_blocked, float,   32, 1, 2)
HPC_CUDA_TEST(CudaBlockedFloat,  gemm_cuda_blocked, float,   64, 3, 4)
HPC_CUDA_TEST(CudaBlockedFloat,  gemm_cuda_blocked, float,  128, 5, 6)
HPC_CUDA_TEST(CudaBlockedFloat,  gemm_cuda_blocked, float,  256, 7, 8)
HPC_CUDA_TEST(CudaBlockedDouble, gemm_cuda_blocked, double,  32, 1, 2)
HPC_CUDA_TEST(CudaBlockedDouble, gemm_cuda_blocked, double, 128, 5, 6)

TEST_F(CudaBlockedFloat, NonSquare_100x200x50) {
    hpc::Matrix<float> A(100,50), B(50,200), C_ref(100,200), C_got(100,200);
    fill_random(A,11); fill_random(B,12);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_blocked(A, B, C_got);
    expect_near(C_got, C_ref, "blocked float non-square");
}

// ===========================================================================
// Level 2 -- Register tile
// ===========================================================================
struct CudaRegTileFloat  : CudaTest {};
struct CudaRegTileDouble : CudaTest {};
HPC_CUDA_TEST(CudaRegTileFloat,  gemm_cuda_reg_tile, float,   64, 1, 2)
HPC_CUDA_TEST(CudaRegTileFloat,  gemm_cuda_reg_tile, float,  128, 3, 4)
HPC_CUDA_TEST(CudaRegTileFloat,  gemm_cuda_reg_tile, float,  256, 5, 6)
HPC_CUDA_TEST(CudaRegTileFloat,  gemm_cuda_reg_tile, float,  512, 7, 8)
HPC_CUDA_TEST(CudaRegTileDouble, gemm_cuda_reg_tile, double,  64, 1, 2)
HPC_CUDA_TEST(CudaRegTileDouble, gemm_cuda_reg_tile, double, 256, 3, 4)
HPC_CUDA_TEST(CudaRegTileDouble, gemm_cuda_reg_tile, double, 512, 5, 6)

TEST_F(CudaRegTileFloat, NonSquare_200x300x100) {
    hpc::Matrix<float> A(200,100), B(100,300), C_ref(200,300), C_got(200,300);
    fill_random(A,20); fill_random(B,21);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_reg_tile(A, B, C_got);
    expect_near(C_got, C_ref, "reg_tile float non-square");
}

// ===========================================================================
// Level 3 -- Double-buffered register tile
// ===========================================================================
struct CudaDoubleBufFloat  : CudaTest {};
struct CudaDoubleBufDouble : CudaTest {};
HPC_CUDA_TEST(CudaDoubleBufFloat,  gemm_cuda_double_buf, float,   64, 1, 2)
HPC_CUDA_TEST(CudaDoubleBufFloat,  gemm_cuda_double_buf, float,  128, 3, 4)
HPC_CUDA_TEST(CudaDoubleBufFloat,  gemm_cuda_double_buf, float,  256, 5, 6)
HPC_CUDA_TEST(CudaDoubleBufFloat,  gemm_cuda_double_buf, float,  512, 7, 8)
HPC_CUDA_TEST(CudaDoubleBufDouble, gemm_cuda_double_buf, double,  64, 1, 2)
HPC_CUDA_TEST(CudaDoubleBufDouble, gemm_cuda_double_buf, double, 256, 3, 4)
HPC_CUDA_TEST(CudaDoubleBufDouble, gemm_cuda_double_buf, double, 512, 5, 6)

// ===========================================================================
// Level 4 -- Tensor Cores (WMMA) -- fp32 only, sm_70+
// Uses relaxed tolerance because fp32->fp16 conversion introduces ~1e-3 error.
// ===========================================================================
struct CudaWmmaFloat : CudaTensorCoreTest {};

TEST_F(CudaWmmaFloat, N64) {
    hpc::Matrix<float> A(64,64), B(64,64), C_ref(64,64), C_got(64,64);
    fill_random(A,1); fill_random(B,2);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_wmma(A, B, C_got);
    expect_near(C_got, C_ref, "wmma/N=64", 1e-2, 1e-2);
}
TEST_F(CudaWmmaFloat, N128) {
    hpc::Matrix<float> A(128,128), B(128,128), C_ref(128,128), C_got(128,128);
    fill_random(A,3); fill_random(B,4);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_wmma(A, B, C_got);
    expect_near(C_got, C_ref, "wmma/N=128", 1e-2, 1e-2);
}
TEST_F(CudaWmmaFloat, N256) {
    hpc::Matrix<float> A(256,256), B(256,256), C_ref(256,256), C_got(256,256);
    fill_random(A,5); fill_random(B,6);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_wmma(A, B, C_got);
    expect_near(C_got, C_ref, "wmma/N=256", 1e-2, 1e-2);
}
TEST_F(CudaWmmaFloat, N512) {
    hpc::Matrix<float> A(512,512), B(512,512), C_ref(512,512), C_got(512,512);
    fill_random(A,7); fill_random(B,8);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_wmma(A, B, C_got);
    expect_near(C_got, C_ref, "wmma/N=512", 1e-2, 1e-2);
}

#undef HPC_CUDA_TEST
