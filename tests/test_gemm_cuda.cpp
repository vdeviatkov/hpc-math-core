/**
 * @file test_gemm_cuda.cpp
 * @brief Google Test correctness suite for all CUDA GEMM kernels.
 *
 * Kernels verified:
 *   gemm_cuda_naive, gemm_cuda_reordered, gemm_cuda_blocked  (Levels 0-1)
 *   gemm_cuda_reg_tile, gemm_cuda_double_buf                  (Levels 2-3)
 *   gemm_cuda_wmma                                            (Level 4, fp32 only)
 *   gemm_cuda_vectorized                                      (Level 5)
 *   gemm_cuda_mma_ldmatrix                                    (Level 6, fp32 only)
 *   gemm_cuda_hopper_wgmma                                    (Level 7, fp32 only)
 *
 * All tests skip at runtime when no CUDA device is present.
 * gemm_cuda_wmma / gemm_cuda_mma_ldmatrix additionally skip when Tensor
 * Cores / sm_80+ are unavailable. gemm_cuda_hopper_wgmma additionally
 * skips without sm_90a.
 *
 * STATUS: all kernels above except gemm_cuda_hopper_wgmma have now been
 * verified on real hardware (RTX 5080, Blackwell sm_120, CUDA 13.2) --
 * every test in this file that runs on that hardware (all but the two
 * CudaHopperWgmmaFloat cases, which SKIP: this GPU is not Hopper) passes.
 * That verification run found and fixed real bugs in gemm_cuda_double_buf
 * (cp.async source in the wrong address space), the DoubleBuf launch
 * config (hardcoded thread count, wrong for double), gemm_cuda_wmma
 * (shared-memory padding broke load_matrix_sync's alignment requirement,
 * and the A/B fragment major-order tags were swapped relative to the
 * physical layout), and gemm_cuda_mma_ldmatrix (the A-fragment
 * ldmatrix.x4 quadrant mapping had its row/col bits swapped) -- see each
 * kernel's file comment in gemm_kernels.cu for the full writeup.
 * gemm_cuda_hopper_wgmma remains genuinely UNVERIFIED: it requires sm_90a
 * (Hopper) specifically, and no Hopper hardware has been available to
 * test it on (it is an explicitly best-effort, likely-non-functional
 * sketch of Hopper warp specialization + TMA, written per direct user
 * request with that understanding).
 *
 * Tolerances:
 *   float  (SIMT): rel 1e-4, abs 1e-3
 *   double (SIMT): rel 1e-10, abs 1e-9
 *   float  (WMMA / mma.sync / wgmma): rel 1e-2, abs 1e-2  -- fp16
 *     conversion introduces ~1e-3 error on top of whatever error, if any,
 *     an incorrect fragment/descriptor mapping might additionally add for
 *     the still-UNVERIFIED wgmma kernel.
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

// gemm_cuda_mma_ldmatrix requires sm_80+ specifically (mma.sync m16n8k16
// f16 shape). Verified on real hardware -- see gemm_kernels.cu's file
// comment for the bug that was found and fixed.
class CudaAmpereMmaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (hpc::gemm::cuda_device_count() == 0)
            GTEST_SKIP() << "No CUDA device available on this machine";
        if (!hpc::gemm::cuda_has_ampere())
            GTEST_SKIP() << "mma.sync m16n8k16 requires sm_80+ (Ampere)";
    }
};

// gemm_cuda_hopper_wgmma requires sm_90a. BEST-EFFORT, EXPLICITLY
// UNVERIFIED, LIKELY NON-FUNCTIONAL -- see gemm_kernels.cu's file comment.
// This fixture will SKIP on every machine this repo has actually been
// tested on; it exists so the test compiles and is ready to run the
// moment someone with real Hopper hardware builds this project.
class CudaHopperTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (hpc::gemm::cuda_device_count() == 0)
            GTEST_SKIP() << "No CUDA device available on this machine";
        if (!hpc::gemm::cuda_has_hopper())
            GTEST_SKIP() << "wgmma/TMA requires sm_90a (Hopper) -- UNVERIFIED code path";
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

// ===========================================================================
// Level 5 -- Vectorized loads (float4/double2) + shared-memory XOR swizzle.
// Full fp32/fp64 precision (no bf16/fp16 truncation) -- uses the same
// tight tolerances as every non-Tensor-Core kernel above. Sizes deliberately
// include N=100/300 (not multiples of the vector width) to exercise the
// fallback-to-RegTile path alongside the vectorized path itself.
// ===========================================================================
struct CudaVectorizedFloat  : CudaTest {};
struct CudaVectorizedDouble : CudaTest {};
HPC_CUDA_TEST(CudaVectorizedFloat,  gemm_cuda_vectorized, float,   32, 1, 2)
HPC_CUDA_TEST(CudaVectorizedFloat,  gemm_cuda_vectorized, float,   64, 3, 4)
HPC_CUDA_TEST(CudaVectorizedFloat,  gemm_cuda_vectorized, float,  128, 5, 6)
HPC_CUDA_TEST(CudaVectorizedFloat,  gemm_cuda_vectorized, float,  256, 7, 8)
HPC_CUDA_TEST(CudaVectorizedFloat,  gemm_cuda_vectorized, float,  100, 9, 10)  // not a multiple of 4 -- fallback path
HPC_CUDA_TEST(CudaVectorizedDouble, gemm_cuda_vectorized, double,  32, 1, 2)
HPC_CUDA_TEST(CudaVectorizedDouble, gemm_cuda_vectorized, double, 128, 5, 6)
HPC_CUDA_TEST(CudaVectorizedDouble, gemm_cuda_vectorized, double,  99, 9, 10)  // not a multiple of 2 -- fallback path

TEST_F(CudaVectorizedFloat, NonSquare_100x200x50) {
    hpc::Matrix<float> A(100,50), B(50,200), C_ref(100,200), C_got(100,200);
    fill_random(A,11); fill_random(B,12);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_vectorized(A, B, C_got);
    expect_near(C_got, C_ref, "vectorized float non-square");
}

// ===========================================================================
// Level 6 -- Raw Tensor Cores via mma.sync + ldmatrix -- fp32 only, sm_80+.
// Verified on real hardware (RTX 5080, Blackwell sm_120) -- all three
// cases below pass; see gemm_kernels.cu's kernel_mma_ldmatrix file comment
// for the ldmatrix quadrant-mapping bug that was found and fixed here.
// Relaxed tolerance for the same reason as WMMA (fp16 conversion).
// ===========================================================================
struct CudaMmaLdmatrixFloat : CudaAmpereMmaTest {};

TEST_F(CudaMmaLdmatrixFloat, N64) {
    hpc::Matrix<float> A(64,64), B(64,64), C_ref(64,64), C_got(64,64);
    fill_random(A,1); fill_random(B,2);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_mma_ldmatrix(A, B, C_got);
    expect_near(C_got, C_ref, "mma_ldmatrix/N=64", 1e-2, 1e-2);
}
TEST_F(CudaMmaLdmatrixFloat, N128) {
    hpc::Matrix<float> A(128,128), B(128,128), C_ref(128,128), C_got(128,128);
    fill_random(A,3); fill_random(B,4);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_mma_ldmatrix(A, B, C_got);
    expect_near(C_got, C_ref, "mma_ldmatrix/N=128", 1e-2, 1e-2);
}
TEST_F(CudaMmaLdmatrixFloat, N256) {
    hpc::Matrix<float> A(256,256), B(256,256), C_ref(256,256), C_got(256,256);
    fill_random(A,5); fill_random(B,6);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_mma_ldmatrix(A, B, C_got);
    expect_near(C_got, C_ref, "mma_ldmatrix/N=256", 1e-2, 1e-2);
}

// ===========================================================================
// Level 7 -- Hopper warp specialization + TMA (wgmma) -- fp32 only, sm_90a.
// BEST-EFFORT, EXPLICITLY UNVERIFIED, LIKELY NON-FUNCTIONAL: see
// gemm_kernels.cu's kernel_hopper_wgmma file comment. This test will SKIP
// on every machine this repo has actually been run on (no Hopper hardware
// was available anywhere in this project); it exists so there is
// something to run the moment someone with real sm_90a hardware builds
// this project -- if it fails there, that is genuinely new information,
// not a regression. Sizes are exact multiples of 64/64/16 as required.
// ===========================================================================
struct CudaHopperWgmmaFloat : CudaHopperTest {};

TEST_F(CudaHopperWgmmaFloat, N64) {
    hpc::Matrix<float> A(64,64), B(64,64), C_ref(64,64), C_got(64,64);
    fill_random(A,1); fill_random(B,2);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_hopper_wgmma(A, B, C_got);
    expect_near(C_got, C_ref, "hopper_wgmma/N=64", 1e-2, 1e-2);
}
TEST_F(CudaHopperWgmmaFloat, N128) {
    hpc::Matrix<float> A(128,128), B(128,128), C_ref(128,128), C_got(128,128);
    fill_random(A,3); fill_random(B,4);
    hpc::gemm::gemm_naive(A, B, C_ref);
    hpc::gemm::gemm_cuda_hopper_wgmma(A, B, C_got);
    expect_near(C_got, C_ref, "hopper_wgmma/N=128", 1e-2, 1e-2);
}

#undef HPC_CUDA_TEST
