/**
 * @file test_gemm.cpp
 * @brief Correctness tests for the CPU GEMM kernels.
 *
 * gemm_naive is checked against hand-computed products. Every other kernel
 * is compared with gemm_naive on the shapes in kShapes, which cover each
 * kernel's edges: sizes below and around the SIMD widths, register tiles and
 * cache blocks, rectangular and empty matrices. C is pre-filled with junk,
 * so a kernel that accumulates into C instead of overwriting it fails.
 *
 * A kernel family is in kKernels only where it is compiled (HPC_HAS_* from
 * hpc/isa.hpp); elsewhere its functions are deleted. The test list on a
 * machine is therefore exactly the set of kernels that run there.
 */

#include "gemm/amx.hpp"
#include "gemm/avx2.hpp"
#include "gemm/avx512.hpp"
#include "gemm/blocked.hpp"
#include "gemm/kleidiai.hpp"
#include "gemm/naive.hpp"
#include "gemm/neon.hpp"
#include "gemm/prefetch.hpp"
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

namespace {

using hpc::Matrix;
using namespace hpc::gemm;

template <typename T>
void fill_random(Matrix<T>& M, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(T{-1}, T{1});
    for (std::size_t i = 0; i < M.size(); ++i)
        M.data()[i] = dist(rng);
}

/// Relative tolerance: f32 sums of up to a few thousand terms in a
/// different order than gemm_naive differ by ~1e-5.
template <typename T>
constexpr T kTol = sizeof(T) == 4 ? T(1e-4) : T(1e-10);

template <typename T>
void expect_near(const Matrix<T>& got, const Matrix<T>& ref, const std::string& what) {
    for (std::size_t i = 0; i < ref.rows(); ++i)
        for (std::size_t j = 0; j < ref.cols(); ++j)
            ASSERT_NEAR(got(i, j), ref(i, j), kTol<T> * (T{1} + std::abs(ref(i, j))))
                << what << " at (" << i << "," << j << ")";
}

// ===========================================================================
// gemm_naive against hand-computed products
// ===========================================================================

TEST(GemmNaive, KnownResult2x2) {
    //  [1 2] [5 6]   [19 22]
    //  [3 4] [7 8] = [43 50]
    Matrix<double> A(2, 2), B(2, 2), C(2, 2);
    const double a[] = {1, 2, 3, 4}, b[] = {5, 6, 7, 8}, c[] = {19, 22, 43, 50};
    std::memcpy(A.data(), a, sizeof a);
    std::memcpy(B.data(), b, sizeof b);
    gemm_naive(A, B, C);
    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(C.data()[i], c[i]);
}

TEST(GemmNaive, Rectangular2x3Times3x2) {
    //  [1 2 3] [ 7  8]   [ 58  64]
    //  [4 5 6] [ 9 10] = [139 154]
    //          [11 12]
    Matrix<double> A(2, 3), B(3, 2), C(2, 2);
    const double a[] = {1, 2, 3, 4, 5, 6}, b[] = {7, 8, 9, 10, 11, 12}, c[] = {58, 64, 139, 154};
    std::memcpy(A.data(), a, sizeof a);
    std::memcpy(B.data(), b, sizeof b);
    gemm_naive(A, B, C);
    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(C.data()[i], c[i]);
}

TEST(GemmNaive, IdentityAndZero) {
    constexpr std::size_t N = 17;
    Matrix<double> A(N, N), I(N, N), Z(N, N), C(N, N);
    fill_random(A, 1);
    for (std::size_t i = 0; i < N; ++i)
        I(i, i) = 1.0;
    gemm_naive(A, I, C);
    for (std::size_t i = 0; i < N * N; ++i)
        EXPECT_DOUBLE_EQ(C.data()[i], A.data()[i]);
    gemm_naive(A, Z, C);
    for (std::size_t i = 0; i < N * N; ++i)
        EXPECT_DOUBLE_EQ(C.data()[i], 0.0);
}

// ===========================================================================
// Every other kernel against gemm_naive
// ===========================================================================

template <typename T>
using KernelFn = void (*)(const Matrix<T>&, const Matrix<T>&, Matrix<T>&);

struct Kernel {
    const char* name;
    KernelFn<float> f32;
    KernelFn<double> f64;  ///< nullptr: the kernel has no f64 version
};

#define HPC_BOTH(name, fn) Kernel{name, &fn<float>, &fn<double>}

const Kernel kKernels[] = {
    HPC_BOTH("Reordered", gemm_reordered),
    HPC_BOTH("Blocked", gemm_blocked),
    HPC_BOTH("BlockedPrefetch", gemm_blocked_prefetch),
#if HPC_HAS_AVX2
    HPC_BOTH("Avx2Naive", gemm_avx2_naive),
    HPC_BOTH("Avx2Reordered", gemm_avx2_reordered),
    HPC_BOTH("Avx2Blocked", gemm_avx2_blocked),
    HPC_BOTH("Avx2BlockedPrefetch", gemm_avx2_blocked_prefetch),
#endif
#if HPC_HAS_AVX512
    HPC_BOTH("Avx512Naive", gemm_avx512_naive),
    HPC_BOTH("Avx512Reordered", gemm_avx512_reordered),
    HPC_BOTH("Avx512Blocked", gemm_avx512_blocked),
    HPC_BOTH("Avx512BlockedPrefetch", gemm_avx512_blocked_prefetch),
#endif
#if HPC_HAS_NEON
    HPC_BOTH("NeonNaive", gemm_neon_naive),
    HPC_BOTH("NeonReordered", gemm_neon_reordered),
    HPC_BOTH("NeonBlocked", gemm_neon_blocked),
    HPC_BOTH("NeonBlockedPrefetch", gemm_neon_blocked_prefetch),
#endif
#if HPC_HAS_SVE
    HPC_BOTH("SveNaive", gemm_sve_naive),
    HPC_BOTH("SveReordered", gemm_sve_reordered),
    HPC_BOTH("SveBlocked", gemm_sve_blocked),
    HPC_BOTH("SveBlockedPrefetch", gemm_sve_blocked_prefetch),
#endif
#if HPC_HAS_SME
    HPC_BOTH("Sme1Tile", gemm_sme_1tile),
    HPC_BOTH("SmeTiles", gemm_sme_tiles),
    HPC_BOTH("SmeBlocked", gemm_sme_blocked),
    HPC_BOTH("Sme", gemm_sme),
#endif
#if HPC_HAS_AMX
    HPC_BOTH("Amx", gemm_amx),
#endif
#if HPC_HAS_KLEIDIAI
    Kernel{"KleidiAI", &gemm_kleidiai<float>, nullptr},
#endif
};

struct Shape {
    std::size_t M, K, N;
};

// Square sizes below, at and just past the vector widths (4/8/16 lanes),
// register tiles (16, 32) and cache tiles (64, 128, 256); rectangular shapes
// with partial tiles in every dimension; K across the SME Kc block (twice,
// so partial sums are reloaded into ZA) and N across its Nc block; empty
// matrices.
const Shape kShapes[] = {
    {1, 1, 1},       {2, 2, 2},       {3, 3, 3},     {4, 4, 4},         {5, 5, 5},
    {7, 7, 7},       {8, 8, 8},       {13, 13, 13},  {15, 15, 15},      {16, 16, 16},
    {17, 17, 17},    {31, 31, 31},    {32, 32, 32},  {33, 33, 33},      {64, 64, 64},
    {100, 100, 100}, {128, 128, 128}, {256, 256, 256},
    {3, 5, 2},       {7, 3, 9},       {17, 33, 15},  {33, 1, 65},       {63, 65, 64},
    {129, 7, 33},    {257, 64, 31},   {24, 777, 40}, {20, 2065, 36},    {133, 70, 40},
    {9, 11, 4103},   {3, 0, 4},       {0, 5, 3},     {4, 5, 0},
};

class GemmKernel : public ::testing::TestWithParam<Kernel> {};

template <typename T>
void expect_matches_naive(KernelFn<T> kernel, const Shape& s, const char* name) {
    Matrix<T> A(s.M, s.K), B(s.K, s.N), C_ref(s.M, s.N), C(s.M, s.N);
    fill_random(A, unsigned(s.M * 131 + s.K));
    fill_random(B, unsigned(s.K * 137 + s.N));
    gemm_naive(A, B, C_ref);
    C.fill(T{123});  // must be overwritten, not accumulated into
    kernel(A, B, C);
    expect_near(C, C_ref,
                std::string(name) + (sizeof(T) == 4 ? " f32 " : " f64 ") + std::to_string(s.M) +
                    "x" + std::to_string(s.K) + "x" + std::to_string(s.N));
}

TEST_P(GemmKernel, MatchesNaive) {
    const Kernel& k = GetParam();
    for (const Shape& s : kShapes) {
        expect_matches_naive(k.f32, s, k.name);
        if (k.f64)
            expect_matches_naive(k.f64, s, k.name);
        if (HasFatalFailure())
            return;
    }
}

INSTANTIATE_TEST_SUITE_P(All, GemmKernel, ::testing::ValuesIn(kKernels),
                         [](const ::testing::TestParamInfo<Kernel>& info) {
                             return std::string(info.param.name);
                         });

// ===========================================================================
// SME internals
// ===========================================================================

#if HPC_HAS_SME

// pack_a's NEON transpose must be a pure bit move: feed random bit patterns
// (NaN payloads, denormals, -0) and compare bytes.
template <typename T>
void expect_pack_a_bit_exact() {
    using Bits           = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
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
            sme_detail::pack_a(A.data(), lda, mc, kc, mr, got.data());
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

// blocked_driver with B rows ldb > N apart, as the BStride benchmarks run it:
// both B paths must ignore the padding columns.
template <typename T>
void expect_padded_b_matches_naive(std::size_t M, std::size_t K, std::size_t N, std::size_t ldb) {
    Matrix<T> A(M, K), B(K, N), Bpad(K, ldb), C_ref(M, N), C(M, N);
    fill_random(A, 31);
    fill_random(Bpad, 32);  // the padding columns hold junk too
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j)
            B(k, j) = Bpad(k, j);
    gemm_naive(A, B, C_ref);
    C.fill(T{123});
    sme_detail::blocked_driver<T, false>(A, Bpad.data(), ldb, C);
    expect_near(C, C_ref, "in place, ldb=" + std::to_string(ldb));
    C.fill(T{123});
    sme_detail::blocked_driver<T, true>(A, Bpad.data(), ldb, C);
    expect_near(C, C_ref, "packed, ldb=" + std::to_string(ldb));
}

TEST(GemmSme, PaddedBMatchesNaive) {
    expect_padded_b_matches_naive<float>(70, 90, 50, 53);
    expect_padded_b_matches_naive<float>(33, kSmeKc + 5, 65, 97);
    expect_padded_b_matches_naive<double>(70, 90, 50, 53);
    expect_padded_b_matches_naive<double>(33, kSmeKc + 5, 65, 97);
}

#endif  // HPC_HAS_SME

}  // namespace
