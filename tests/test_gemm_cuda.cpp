/**
 * @file test_gemm_cuda.cpp
 * @brief Correctness tests for the CUDA GEMM kernels (gemm/cuda.hpp).
 *
 * Each kernel is compared with gemm_naive on its own list of shapes. A
 * result passes when its relative error (against max(|ref|, 1)) or its
 * absolute error is under the tolerance: 1e-4 / 1e-3 for f32, 1e-10 / 1e-9
 * for f64, and 1e-2 for the kernels that multiply in fp16 or TF32, where
 * the input conversion alone costs ~1e-3.
 *
 * Every test skips without a CUDA device (the stub build has none), and the
 * Tensor Core / TF32 ones also on GPUs older than they need.
 */

#include "gemm/cuda.hpp"
#include "gemm/naive.hpp"
#include "hpc/matrix.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>

namespace {

using hpc::Matrix;
using namespace hpc::gemm;

template <typename T>
void fill_random(Matrix<T>& M, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(T{-1}, T{1});
    for (std::size_t i = 0; i < M.size(); ++i)
        M.data()[i] = dist(rng);
}

enum class Needs { kDevice, kTensorCores, kAmpere };

struct Shape {
    std::size_t M, K, N;
};

template <typename T>
struct Case {
    const char* name;
    void (*kernel)(const Matrix<T>&, const Matrix<T>&, Matrix<T>&);
    Needs needs;
    bool reduced_precision;  ///< fp16 or TF32 multiply: relaxed tolerance
    std::vector<Shape> shapes;
};

std::vector<Shape> square(std::initializer_list<std::size_t> sizes) {
    std::vector<Shape> shapes;
    for (const std::size_t n : sizes)
        shapes.push_back({n, n, n});
    return shapes;
}

std::vector<Shape> operator+(std::vector<Shape> a, const Shape& b) {
    a.push_back(b);
    return a;
}

std::vector<Shape> operator+(std::vector<Shape> a, const std::vector<Shape>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// Partial tiles in every dimension (M, K, N): all odd; one past the 128-wide
// block tiles with K not a multiple of the 16-deep k-tile; and multiples of
// the vector width that are not multiples of any tile, which take
// gemm_cuda_vectorized's vector path rather than its fallback.
const std::vector<Shape> kEdges = {{17, 33, 65}, {129, 37, 131}, {132, 68, 260}};

// Every kernel with bounds checks also runs kEdges. Sizes that are not
// multiples of 16 check WMMA's staged edge-tile store; N=99/100 check
// gemm_cuda_vectorized's fallback (K, N not multiples of the vector width);
// N=192 checks gemm_cuda_wmma_pipelined's fallback (not a multiple of 128)
// and 384x160x256 its fast path with M, K, N all different.
const std::vector<Case<float>> kF32 = {
    {"Naive", gemm_cuda_naive<float>, Needs::kDevice, false, square({32, 64, 128, 256}) + kEdges},
    {"Blocked", gemm_cuda_blocked<float>, Needs::kDevice, false,
     square({32, 64, 128, 256}) + Shape{100, 50, 200} + kEdges},
    {"RegTile", gemm_cuda_reg_tile<float>, Needs::kDevice, false,
     square({64, 128, 256, 512}) + Shape{200, 100, 300} + kEdges},
    {"DoubleBuf", gemm_cuda_double_buf<float>, Needs::kDevice, false,
     square({64, 128, 256, 512}) + kEdges},
    {"Vectorized", gemm_cuda_vectorized<float>, Needs::kDevice, false,
     square({32, 64, 128, 256, 100}) + Shape{100, 50, 200} + kEdges},
    {"Wmma", gemm_cuda_wmma, Needs::kTensorCores, true,
     square({64, 128, 256, 512, 17, 50, 72, 100}) + Shape{40, 24, 56} + kEdges},
    {"MmaLdmatrix", gemm_cuda_mma_ldmatrix, Needs::kAmpere, true,
     square({64, 128, 256}) + kEdges},
    {"WmmaPipelined", gemm_cuda_wmma_pipelined, Needs::kTensorCores, true,
     square({128, 256, 512, 192}) + Shape{384, 160, 256}},
    {"Cublas", gemm_cuda_cublas<float>, Needs::kDevice, false, square({64, 256, 512})},
    {"CublasTf32", gemm_cuda_cublas_tf32, Needs::kAmpere, true, square({64, 256, 512})},
    {"CublasFp16", gemm_cuda_cublas_fp16, Needs::kTensorCores, true, square({64, 256, 512})},
};

const std::vector<Case<double>> kF64 = {
    {"Naive", gemm_cuda_naive<double>, Needs::kDevice, false, square({32, 64, 128, 256}) + kEdges},
    {"Blocked", gemm_cuda_blocked<double>, Needs::kDevice, false, square({32, 128}) + kEdges},
    {"RegTile", gemm_cuda_reg_tile<double>, Needs::kDevice, false, square({64, 256, 512}) + kEdges},
    {"DoubleBuf", gemm_cuda_double_buf<double>, Needs::kDevice, false,
     square({64, 256, 512}) + kEdges},
    {"Vectorized", gemm_cuda_vectorized<double>, Needs::kDevice, false,
     square({32, 128, 99}) + kEdges},
    {"Cublas", gemm_cuda_cublas<double>, Needs::kDevice, false, square({64, 256})},
};

template <typename T>
void run_case(const Case<T>& c) {
    if (cuda_device_count() == 0)
        GTEST_SKIP() << "No CUDA device available";
    if (c.needs == Needs::kTensorCores && !cuda_has_tensor_cores())
        GTEST_SKIP() << "needs Tensor Cores (sm_70+)";
    if (c.needs == Needs::kAmpere && !cuda_has_ampere())
        GTEST_SKIP() << "needs sm_80+";

    const double rel = c.reduced_precision ? 1e-2 : (sizeof(T) == 4 ? 1e-4 : 1e-10);
    const double abs = c.reduced_precision ? 1e-2 : (sizeof(T) == 4 ? 1e-3 : 1e-9);
    for (const Shape& s : c.shapes + std::vector<Shape>{{3, 0, 4}, {0, 5, 3}, {4, 5, 0},
                                                        {0, 0, 0}, {11, 19, 23}}) {
        Matrix<T> A(s.M, s.K), B(s.K, s.N), C_ref(s.M, s.N), C(s.M, s.N);
        fill_random(A, unsigned(s.M + s.K));
        fill_random(B, unsigned(s.K + s.N + 1));
        gemm_naive(A, B, C_ref);
        C.fill(T{123});  // every path must overwrite existing output
        c.kernel(A, B, C);
        for (std::size_t i = 0; i < s.M; ++i) {
            for (std::size_t j = 0; j < s.N; ++j) {
                const double ref  = C_ref(i, j);
                const double diff = std::abs(double(C(i, j)) - ref);
                ASSERT_TRUE(diff / std::max(std::abs(ref), 1.0) < rel || diff < abs)
                    << s.M << "x" << s.K << "x" << s.N << " at (" << i << "," << j
                    << "): got " << C(i, j) << ", expected " << ref;
            }
        }
    }
}

class CudaF32 : public ::testing::TestWithParam<Case<float>> {};
class CudaF64 : public ::testing::TestWithParam<Case<double>> {};

TEST_P(CudaF32, MatchesNaive) { run_case(GetParam()); }
TEST_P(CudaF64, MatchesNaive) { run_case(GetParam()); }

auto case_name = [](const auto& info) { return std::string(info.param.name); };
INSTANTIATE_TEST_SUITE_P(Kernels, CudaF32, ::testing::ValuesIn(kF32), case_name);
INSTANTIATE_TEST_SUITE_P(Kernels, CudaF64, ::testing::ValuesIn(kF64), case_name);

TEST(CudaDevice, EmptyProductsAndInvalidPipelinedShape) {
    if (cuda_device_count() == 0)
        GTEST_SKIP() << "No CUDA device available";
    using Fn = void (*)(const float*, const float*, float*, int, int, int);
    const Fn kernels[] = {gemm_cuda_naive_device<float>, gemm_cuda_blocked_device<float>,
                          gemm_cuda_reg_tile_device<float>, gemm_cuda_double_buf_device<float>,
                          gemm_cuda_vectorized_device<float>, gemm_cuda_wmma_device,
                          gemm_cuda_mma_ldmatrix_device, gemm_cuda_cublas_device<float>,
                          gemm_cuda_cublas_tf32_device};
    Matrix<float> out(3, 4);
    auto* device = static_cast<float*>(gemm_cuda_malloc(out.size() * sizeof(float)));
    struct FreeDevice {
        float* p;
        ~FreeDevice() { gemm_cuda_free(p); }
    } cleanup{device};
    for (const auto fn : kernels) {
        out.fill(123);
        gemm_cuda_memcpy_h2d(device, out.data(), out.size() * sizeof(float));
        fn(nullptr, nullptr, device, 3, 0, 4);
        gemm_cuda_memcpy_d2h(out.data(), device, out.size() * sizeof(float));
        for (std::size_t i = 0; i < out.size(); ++i)
            EXPECT_EQ(out.data()[i], 0);
        fn(nullptr, nullptr, nullptr, 0, 5, 3);
        fn(nullptr, nullptr, nullptr, 4, 5, 0);
        EXPECT_THROW(fn(nullptr, nullptr, nullptr, -1, 5, 3), std::invalid_argument);
    }
    EXPECT_NO_THROW(gemm_cuda_convert_f32_to_f16_device(nullptr, nullptr, 0));
    EXPECT_THROW(gemm_cuda_wmma_pipelined_device(nullptr, nullptr, device, 3, 32, 4),
                 std::invalid_argument);
    gemm_cuda_wmma_pipelined_device(nullptr, nullptr, device, 3, 0, 4);
    gemm_cuda_cublas_fp16_device(nullptr, nullptr, device, 3, 0, 4);
    gemm_cuda_device_synchronize();
}

}  // namespace
