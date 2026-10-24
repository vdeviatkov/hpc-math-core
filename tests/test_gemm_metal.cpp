/**
 * @file test_gemm_metal.cpp
 * @brief Correctness tests for the Metal GEMM kernels (gemm/metal.hpp).
 *
 * Each kernel and tile variant is compared with gemm_naive, with the
 * tolerances of test_gemm_cuda.cpp: relative error (against max(|ref|, 1))
 * under 1e-4 or absolute error under 1e-3 for f32, 1e-2 for the kernels that
 * multiply in fp16. Every test skips without Metal (the stub build).
 */

#include "gemm/metal.hpp"
#include "gemm/naive.hpp"
#include "hpc/matrix.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

using hpc::Matrix;
using namespace hpc::gemm;

void fill_random(Matrix<float>& M, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < M.size(); ++i)
        M.data()[i] = dist(rng);
}

struct Shape {
    std::size_t M, K, N;
};

// Tile multiples with M, K, N all different (fast paths of Levels 2-5, and
// the 128-wide variants), then shapes off every tile, which fall back to
// Level 1, and empty products.
const std::vector<Shape> kShapes = {{64, 64, 64},   {128, 256, 128}, {256, 128, 384}, {512, 512, 512},
                                    {17, 33, 65},   {129, 37, 131},  {100, 50, 200},  {1, 1, 1},
                                    {3, 0, 4},      {0, 5, 3},       {4, 5, 0}};

/// Run `run` on each shape and compare with gemm_naive. Callers run each
/// GEMM twice into the same C, so a kernel that accumulates into C instead of
/// overwriting it fails.
void check(const std::function<void(MetalGemm&)>& run, bool fp16, const std::vector<Shape>& shapes) {
    if (!metal_available())
        GTEST_SKIP() << "Metal not available";
    const double tol = fp16 ? 1e-2 : 1e-4, abs_tol = fp16 ? 1e-2 : 1e-3;
    for (const Shape& s : shapes) {
        Matrix<float> A(s.M, s.K), B(s.K, s.N), C_ref(s.M, s.N), C(s.M, s.N);
        fill_random(A, unsigned(s.M + s.K));
        fill_random(B, unsigned(s.K + s.N + 1));
        gemm_naive(A, B, C_ref);
        MetalGemm g(s.M, s.K, s.N);
        g.set_inputs(A.data(), B.data());
        run(g);
        g.get_output(C.data());
        for (std::size_t i = 0; i < s.M; ++i)
            for (std::size_t j = 0; j < s.N; ++j) {
                const double ref  = C_ref(i, j);
                const double diff = std::abs(double(C(i, j)) - ref);
                ASSERT_TRUE(diff / std::max(std::abs(ref), 1.0) < tol || diff < abs_tol)
                    << s.M << "x" << s.K << "x" << s.N << " at (" << i << "," << j << "): got " << C(i, j)
                    << ", expected " << ref;
            }
    }
}

struct Case {
    const char* name;
    MetalKernel kernel;
    bool fp16;
};

const Case kCases[] = {
    {"Naive", MetalKernel::kNaive, false},       {"Tiled", MetalKernel::kTiled, false},
    {"RegTile", MetalKernel::kRegTile, false},   {"Simd", MetalKernel::kSimd, false},
    {"SimdTg", MetalKernel::kSimdTg, false},     {"SimdTgF16", MetalKernel::kSimdTgF16, true},
    {"Mps", MetalKernel::kMps, false},           {"MpsF16", MetalKernel::kMpsF16, true},
};

class Metal : public ::testing::TestWithParam<Case> {};

TEST_P(Metal, MatchesNaive) {
    const MetalKernel k = GetParam().kernel;
    check([k](MetalGemm& g) { g.run_ms(k, 2); }, GetParam().fp16, kShapes);
}

INSTANTIATE_TEST_SUITE_P(Kernels, Metal, ::testing::ValuesIn(kCases),
                         [](const auto& info) { return std::string(info.param.name); });

// Every tile variant on shapes that are multiples of all their tiles.
TEST(MetalVariants, MatchNaive) {
    if (!metal_available())
        GTEST_SKIP() << "Metal not available";
    for (const std::string& fn : metal_kernel_variants()) {
        SCOPED_TRACE(fn);
        check([&fn](MetalGemm& g) { g.run_variant_ms(fn, 2); }, fn.find("f16") != std::string::npos,
              {{128, 64, 128}, {256, 160, 384}});
    }
}

TEST(MetalGemm, FallsBackOffTile) {
    if (!metal_available())
        GTEST_SKIP() << "Metal not available";
    MetalGemm g(100, 50, 200);
    EXPECT_EQ(g.effective(MetalKernel::kSimdTg), MetalKernel::kTiled);
    MetalGemm h(128, 64, 256);
    EXPECT_EQ(h.effective(MetalKernel::kSimdTg), MetalKernel::kSimdTg);
}

}  // namespace
