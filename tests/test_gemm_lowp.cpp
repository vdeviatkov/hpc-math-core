/**
 * @file test_gemm_lowp.cpp
 * @brief Correctness tests for the low-precision CPU GEMMs.
 *
 * Every kernel is compared with gemm_lowp_reference on the same converted
 * inputs, so the comparison isolates the kernel: input rounding to bf16 /
 * fp16 is identical on both sides.
 *
 *   INT8 → INT32: exact equality.
 *   bf16 / fp16 → FP32: |C − C_ref| ≤ 2·K·u · Σ_k |A(i,k)·B(k,j)| + 1e-30,
 *   u = 2^-24, the standard bound for a length-K FP32 sum in any order.
 *
 * Shapes cover partial register tiles in every dimension, K not a multiple
 * of the k-group (2 or 4), and K across the cache blocks (Kc = 4096 for
 * 16-bit inputs and 8192 for INT8 in the SME2 kernel). C is pre-filled with
 * junk, so a kernel that accumulates into C instead of overwriting it fails.
 * A kernel family is listed only where it is compiled.
 */

#include "gemm/avx512_lowp.hpp"
#include "gemm/lowp_reference.hpp"
#include "gemm/neon_lowp.hpp"
#include "gemm/sme_lowp.hpp"
#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

using hpc::bf16;
using hpc::fp16;
using hpc::Matrix;
using namespace hpc::gemm;

struct Shape {
    std::size_t M, K, N;
};

const Shape kShapes[] = {
    {1, 1, 1},    {2, 3, 2},     {3, 5, 7},      {8, 8, 8},       {16, 16, 16},  {17, 33, 65},
    {32, 32, 32}, {33, 3, 31},   {64, 64, 64},   {65, 66, 67},    {100, 7, 100}, {129, 130, 131},
    {20, 4099, 36}, {12, 8195, 20}, {1031, 70, 40}, {9, 11, 4103}, {0, 5, 3},     {4, 0, 5},
};

template <typename In>
Matrix<In> random_matrix(std::size_t rows, std::size_t cols, unsigned seed) {
    std::mt19937 rng(seed);
    Matrix<In> m(rows, cols);
    if constexpr (std::is_same_v<In, std::int8_t>) {
        std::uniform_int_distribution<int> d(-128, 127);
        for (std::size_t i = 0; i < m.size(); ++i)
            m.data()[i] = static_cast<std::int8_t>(d(rng));
    } else {
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (std::size_t i = 0; i < m.size(); ++i) {
            if constexpr (std::is_same_v<In, bf16>)
                m.data()[i] = hpc::to_bf16(d(rng));
            else
                m.data()[i] = hpc::to_fp16(d(rng));
        }
    }
    return m;
}

template <typename In>
using Kernel = void (*)(const Matrix<In>&, const Matrix<In>&, Matrix<lowp_out_t<In>>&);

/// unit_roundoff: of the accumulation, 2^-24 for FP32 and 2^-11 for fp16.
template <typename In>
void expect_matches_reference(Kernel<In> kernel, const char* name, double unit_roundoff = 0x1p-24) {
    for (const Shape& s : kShapes) {
        const Matrix<In> A = random_matrix<In>(s.M, s.K, unsigned(s.M * 131 + s.K));
        const Matrix<In> B = random_matrix<In>(s.K, s.N, unsigned(s.K * 137 + s.N));
        Matrix<lowp_out_t<In>> ref(s.M, s.N), C(s.M, s.N);
        gemm_lowp_reference(A, B, ref);
        C.fill(lowp_out_t<In>(123));
        kernel(A, B, C);
        const std::string what = std::string(name) + " " + std::to_string(s.M) + "x" +
                                 std::to_string(s.K) + "x" + std::to_string(s.N);
        for (std::size_t i = 0; i < s.M; ++i)
            for (std::size_t j = 0; j < s.N; ++j) {
                if constexpr (std::is_same_v<In, std::int8_t>) {
                    ASSERT_EQ(C(i, j), ref(i, j)) << what << " at (" << i << "," << j << ")";
                } else {
                    double abs_sum = 0;
                    for (std::size_t k = 0; k < s.K; ++k)
                        abs_sum += std::fabs(double(hpc::to_float(A(i, k))) *
                                             double(hpc::to_float(B(k, j))));
                    const double tol = 2.0 * double(s.K) * unit_roundoff * abs_sum + 1e-30;
                    ASSERT_LE(std::fabs(double(C(i, j)) - double(ref(i, j))), tol)
                        << what << " at (" << i << "," << j << ")";
                }
            }
    }
}

// The conversions themselves: exhaustive over all 65,536 bit patterns. NaNs
// are recognised by their bits: the tests build with -ffast-math, under which
// std::isnan may be folded to false.
TEST(Lowp, Fp16RoundTripIsExact) {
    for (std::uint32_t b = 0; b < 0x10000; ++b) {
        const fp16 h{static_cast<std::uint16_t>(b)};
        if ((b & 0x7c00u) == 0x7c00u && (b & 0x03ffu) != 0)  // NaN
            continue;
        const float f = hpc::to_float(h);
        EXPECT_EQ(hpc::to_fp16(f).bits, h.bits) << std::hex << b;
    }
}

TEST(Lowp, Bf16RoundTripIsExact) {
    for (std::uint32_t b = 0; b < 0x10000; ++b) {
        const bf16 h{static_cast<std::uint16_t>(b)};
        if ((b & 0x7fffu) > 0x7f80u)  // NaN
            continue;
        const float f = hpc::to_float(h);
        EXPECT_EQ(hpc::to_bf16(f).bits, h.bits) << std::hex << b;
    }
}

TEST(Lowp, Fp8RoundTripIsExact) {
    for (std::uint32_t b = 0; b < 0x100; ++b) {
        if ((b & 0x7fu) == 0x7fu)  // NaN
            continue;
        const hpc::fp8_e4m3 h{static_cast<std::uint8_t>(b)};
        EXPECT_EQ(hpc::to_fp8_e4m3(hpc::to_float(h)).bits, h.bits) << std::hex << b;
    }
}

TEST(Lowp, Fp8RoundsAndSaturates) {
    EXPECT_EQ(hpc::to_float(hpc::fp8_e4m3{0x7e}), 448.0f);                        // largest finite
    EXPECT_EQ(hpc::to_float(hpc::fp8_e4m3{0x01}), 0x1p-9f);                      // smallest subnormal
    EXPECT_EQ(hpc::to_fp8_e4m3(1.0f + 0x1p-4f).bits, hpc::to_fp8_e4m3(1.0f).bits);  // tie → even
    EXPECT_EQ(hpc::to_fp8_e4m3(1.0f + 0x1p-4f + 0x1p-10f).bits, 0x39u);            // above the tie
    EXPECT_EQ(hpc::to_fp8_e4m3(1e6f).bits, 0x7eu);                                // saturates
    EXPECT_EQ(hpc::to_fp8_e4m3(-1e6f).bits, 0xfeu);
    EXPECT_EQ(hpc::to_fp8_e4m3(0x1p-10f).bits, 0x00u);                            // tie → even (0)
    EXPECT_EQ(hpc::to_fp8_e4m3(3 * 0x1p-10f).bits, 0x02u);                        // tie → even (2)
}

TEST(Lowp, ConversionsRoundToNearestEven) {
    EXPECT_EQ(hpc::to_fp16(1.0f + 0x1p-11f).bits, hpc::to_fp16(1.0f).bits);          // tie → even
    EXPECT_EQ(hpc::to_fp16(1.0f + 0x1p-11f + 0x1p-20f).bits, 0x3c01u);               // above the tie
    EXPECT_EQ(hpc::to_fp16(65520.0f).bits, 0x7c00u);                                 // overflows to inf
    EXPECT_EQ(hpc::to_fp16(0x1p-24f).bits, 0x0001u);                                 // smallest subnormal
    EXPECT_EQ(hpc::to_bf16(1.0f + 0x1p-8f).bits, hpc::to_bf16(1.0f).bits);           // tie → even
    EXPECT_EQ(hpc::to_bf16(1.0f + 0x1p-8f + 0x1p-20f).bits, 0x3f81u);
}

#if HPC_HAS_AVX512_LOWP
TEST(LowpAvx512, Bf16MatchesReference) { expect_matches_reference<bf16>(gemm_avx512_bf16, "avx512 bf16"); }
TEST(LowpAvx512, Int8MatchesReference) {
    expect_matches_reference<std::int8_t>(gemm_avx512_i8, "avx512 int8");
}
// Other register tiles and k blocks: R not dividing Mc = 128 (6, 12), wide
// strips (V = 3, 4), one vector (V = 1), and k blocks small enough that the
// shapes above span many of them (KcBytes = 4: one k group per block).
template <int R, int V, std::size_t KcBytes>
void expect_avx512_tile_matches_reference() {
    const std::string tile = std::to_string(R) + "x" + std::to_string(V) + " kc" + std::to_string(KcBytes);
    expect_matches_reference<bf16>(gemm_avx512_bf16<R, V, KcBytes>, ("avx512 bf16 " + tile).c_str());
    expect_matches_reference<std::int8_t>(gemm_avx512_i8<R, V, KcBytes>, ("avx512 int8 " + tile).c_str());
}
TEST(LowpAvx512, OtherTilesMatchReference) {
    expect_avx512_tile_matches_reference<8, 3, 1024>();
    expect_avx512_tile_matches_reference<6, 4, 512>();
    expect_avx512_tile_matches_reference<12, 2, 1024>();
    expect_avx512_tile_matches_reference<16, 1, 64>();
    expect_avx512_tile_matches_reference<1, 1, 4>();
}
#endif

#if HPC_HAS_NEON_LOWP
TEST(LowpNeon, Bf16MatchesReference) { expect_matches_reference<bf16>(gemm_neon_bf16, "neon bf16"); }
TEST(LowpNeon, Int8MatchesReference) { expect_matches_reference<std::int8_t>(gemm_neon_i8, "neon int8"); }
TEST(LowpNeon, Fp16MatchesReference) { expect_matches_reference<fp16>(gemm_neon_f16, "neon fp16"); }
// FP16 accumulation: the bound uses fp16's unit roundoff, 2^-11.
TEST(LowpNeon, Fp16AccumulateWithinFp16Bound) {
    expect_matches_reference<fp16>(gemm_neon_f16_acc, "neon fp16 acc", 0x1p-11);
}
#endif

#if HPC_HAS_SME
TEST(LowpSme, Fp16MatchesReference) { expect_matches_reference<fp16>(gemm_sme_widening<fp16>, "sme fp16"); }
TEST(LowpSme, Bf16MatchesReference) { expect_matches_reference<bf16>(gemm_sme_widening<bf16>, "sme bf16"); }
TEST(LowpSme, Int8MatchesReference) {
    expect_matches_reference<std::int8_t>(gemm_sme_widening<std::int8_t>, "sme int8");
}
#endif

}  // namespace
