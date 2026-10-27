/**
 * @file test_gemm_lowp_cuda.cpp
 * @brief Correctness tests for the low-precision CUDA GEMMs (gemm/cuda.hpp).
 *
 * Each format runs through both hand-written kernels (the cp.async one and
 * Level 8, TMA) and through cuBLAS, and is compared with gemm_lowp_naive on the same converted
 * inputs, so input rounding is identical on both sides.
 *
 *   INT8 → INT32: exact equality.
 *   fp32 accumulation: |C − C_ref| ≤ 2·K·u·Σ_k |A(i,k)·B(k,j)| + 1e-30,
 *   u = 2^-24, the bound the CPU kernels meet. Tensor Cores do not add in
 *   IEEE FP32 order, so this is checked, not assumed; the test prints the
 *   worst error in units of u·Σ|ab|.
 *   fp16 accumulation: the same bound with u = 2^-11.
 *
 * Shapes are not tile multiples, so the Matrix wrappers' zero padding is
 * exercised in every dimension; K crosses several pipeline stages. A device
 * without the format's mma.sync (or a CPU-only build) skips.
 */
#include "gemm/cuda.hpp"
#include "gemm/lowp_naive.hpp"
#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

namespace {

using hpc::bf16;
using hpc::fp16;
using hpc::fp8_e4m3;
using hpc::Matrix;
using namespace hpc::gemm;

struct Shape {
    std::size_t M, K, N;
};

const Shape kShapes[] = {
    {1, 1, 1},     {3, 5, 7},       {128, 32, 128}, {129, 65, 130}, {256, 512, 384},
    {100, 1000, 60}, {64, 4100, 72}, {300, 70, 260}, {0, 5, 3},     {4, 0, 5},
    {2688, 200, 1024},  // 84 blocks of 128 x 256: Level 8's wide tile on an 84-SM GPU
};

template <typename In>
Matrix<In> random_matrix(std::size_t rows, std::size_t cols, unsigned seed) {
    std::mt19937 rng(seed);
    Matrix<In> m(rows, cols);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::uniform_int_distribution<int> di(-128, 127);
    for (std::size_t i = 0; i < m.size(); ++i) {
        if constexpr (std::is_same_v<In, std::int8_t>)
            m.data()[i] = static_cast<std::int8_t>(di(rng));
        else if constexpr (std::is_same_v<In, bf16>)
            m.data()[i] = hpc::to_bf16(d(rng));
        else if constexpr (std::is_same_v<In, fp16>)
            m.data()[i] = hpc::to_fp16(d(rng));
        else
            m.data()[i] = hpc::to_fp8_e4m3(d(rng));
    }
    return m;
}

template <typename In>
using Kernel = void (*)(const Matrix<In>&, const Matrix<In>&, Matrix<lowp_out_t<In>>&, CudaLowpImpl);

template <typename In>
void expect_matches_naive(CudaLowp format, Kernel<In> kernel, CudaLowpImpl impl, const char* name,
                              double unit_roundoff = 0x1p-24) {
    if (cuda_device_count() == 0 || !cuda_lowp_supported(format))
        GTEST_SKIP() << "no CUDA device with this format's mma.sync";
    if (impl == CudaLowpImpl::tma && !cuda_has_tma())
        GTEST_SKIP() << "Level 8 needs sm_90+ (TMA)";
    double worst = 0;  // largest |C - C_ref| / (2^-24 · Σ|ab|)
    for (const Shape& s : kShapes) {
        const Matrix<In> A = random_matrix<In>(s.M, s.K, unsigned(s.M * 131 + s.K));
        const Matrix<In> B = random_matrix<In>(s.K, s.N, unsigned(s.K * 137 + s.N));
        Matrix<lowp_out_t<In>> ref(s.M, s.N), C(s.M, s.N);
        gemm_lowp_naive(A, B, ref);
        C.fill(lowp_out_t<In>(123));
        kernel(A, B, C, impl);
        const std::string what = std::string(name) + " " + std::to_string(s.M) + "x" +
                                 std::to_string(s.K) + "x" + std::to_string(s.N);
        for (std::size_t i = 0; i < s.M; ++i)
            for (std::size_t j = 0; j < s.N; ++j) {
                if constexpr (std::is_same_v<In, std::int8_t>) {
                    ASSERT_EQ(C(i, j), ref(i, j)) << what << " at (" << i << "," << j << ")";
                } else {
                    double abs_sum = 0;
                    for (std::size_t k = 0; k < s.K; ++k)
                        abs_sum += std::fabs(double(hpc::to_float(A(i, k))) * double(hpc::to_float(B(k, j))));
                    const double err = std::fabs(double(C(i, j)) - double(ref(i, j)));
                    if (abs_sum > 0)
                        worst = std::max(worst, err / (abs_sum * 0x1p-24));
                    ASSERT_LE(err, 2.0 * double(s.K) * unit_roundoff * abs_sum + 1e-30)
                        << what << " at (" << i << "," << j << ")";
                }
            }
    }
    if constexpr (!std::is_same_v<In, std::int8_t>)
        std::printf("[ %s ] worst error %.3g x 2^-24 * sum|ab|\n", name, worst);
}

TEST(LowpCuda, Bf16) { expect_matches_naive<bf16>(CudaLowp::bf16, gemm_cuda_bf16, CudaLowpImpl::mma, "mma bf16"); }
TEST(LowpCuda, Fp16) { expect_matches_naive<fp16>(CudaLowp::fp16, gemm_cuda_fp16, CudaLowpImpl::mma, "mma fp16"); }
TEST(LowpCuda, Fp16Accumulate16) {
    expect_matches_naive<fp16>(CudaLowp::fp16_acc16, gemm_cuda_fp16_acc16, CudaLowpImpl::mma, "mma fp16 acc16", 0x1p-11);
}
TEST(LowpCuda, Int8) { expect_matches_naive<std::int8_t>(CudaLowp::int8, gemm_cuda_i8, CudaLowpImpl::mma, "mma int8"); }
TEST(LowpCuda, Fp8) { expect_matches_naive<fp8_e4m3>(CudaLowp::fp8_e4m3, gemm_cuda_fp8, CudaLowpImpl::mma, "mma fp8"); }

TEST(LowpCudaTma, Bf16) { expect_matches_naive<bf16>(CudaLowp::bf16, gemm_cuda_bf16, CudaLowpImpl::tma, "tma bf16"); }
TEST(LowpCudaTma, Fp16) { expect_matches_naive<fp16>(CudaLowp::fp16, gemm_cuda_fp16, CudaLowpImpl::tma, "tma fp16"); }
TEST(LowpCudaTma, Fp16Accumulate16) {
    expect_matches_naive<fp16>(CudaLowp::fp16_acc16, gemm_cuda_fp16_acc16, CudaLowpImpl::tma, "tma fp16 acc16", 0x1p-11);
}
TEST(LowpCudaTma, Int8) { expect_matches_naive<std::int8_t>(CudaLowp::int8, gemm_cuda_i8, CudaLowpImpl::tma, "tma int8"); }
TEST(LowpCudaTma, Fp8) { expect_matches_naive<fp8_e4m3>(CudaLowp::fp8_e4m3, gemm_cuda_fp8, CudaLowpImpl::tma, "tma fp8"); }

TEST(LowpCublas, Bf16) { expect_matches_naive<bf16>(CudaLowp::bf16, gemm_cuda_bf16, CudaLowpImpl::cublas, "cublas bf16"); }
TEST(LowpCublas, Fp16) { expect_matches_naive<fp16>(CudaLowp::fp16, gemm_cuda_fp16, CudaLowpImpl::cublas, "cublas fp16"); }
TEST(LowpCublas, Fp16Accumulate16) {
    expect_matches_naive<fp16>(CudaLowp::fp16_acc16, gemm_cuda_fp16_acc16, CudaLowpImpl::cublas, "cublas fp16 acc16", 0x1p-11);
}
TEST(LowpCublas, Int8) { expect_matches_naive<std::int8_t>(CudaLowp::int8, gemm_cuda_i8, CudaLowpImpl::cublas, "cublas int8"); }
TEST(LowpCublas, Fp8) { expect_matches_naive<fp8_e4m3>(CudaLowp::fp8_e4m3, gemm_cuda_fp8, CudaLowpImpl::cublas, "cublas fp8"); }

}  // namespace
