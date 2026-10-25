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
#include <iterator>
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
// A blocked SIMD kernel with an explicit register tile: R rows × V vectors.
// Shapes other than the defaults exercise the driver's edge handling: rows
// left over from the 64-row block, fewer vectors, scalar columns.
#define HPC_TILE(name, fn, R, V) Kernel{name, &fn<float, R, V>, &fn<double, R, V>}
// Cache blocks that are not multiples of the tile rows or the vector width:
// strips and tiles cut short at every block edge, not only at the matrix's.
#define HPC_BLOCK(name, fn, R, V, M, K, N) \
    Kernel{name, &fn<float, R, V, M, K, N>, &fn<double, R, V, M, K, N>}

const Kernel kKernels[] = {
    HPC_BOTH("Reordered", gemm_reordered),
    HPC_BOTH("Blocked", gemm_blocked),
    HPC_BOTH("BlockedPrefetch", gemm_blocked_prefetch),
#if HPC_HAS_AVX2
    HPC_BOTH("Avx2Naive", gemm_avx2_naive),
    HPC_BOTH("Avx2Reordered", gemm_avx2_reordered),
    HPC_BOTH("Avx2Blocked", gemm_avx2_blocked),
    HPC_BOTH("Avx2BlockedPrefetch", gemm_avx2_blocked_prefetch),
    HPC_TILE("Avx2Blocked1x1", gemm_avx2_blocked, 1, 1),
    HPC_TILE("Avx2Blocked4x2", gemm_avx2_blocked, 4, 2),
    HPC_TILE("Avx2Blocked2x4", gemm_avx2_blocked, 2, 4),
    HPC_TILE("Avx2Blocked4x3", gemm_avx2_blocked, 4, 3),
    HPC_TILE("Avx2Blocked7x2", gemm_avx2_blocked, 7, 2),
    HPC_BLOCK("Avx2Blocked_20x7x37", gemm_avx2_blocked, 6, 2, 20, 7, 37),
    HPC_BOTH("Avx2Packed", gemm_avx2_packed),
    Kernel{"Avx2PackedA", &gemm_avx2_packed<float, 6, 2, kAvx2PackMc, kAvx2PackKc, kAvx2PackNc, true>,
           &gemm_avx2_packed<double, 6, 2, kAvx2PackMc, kAvx2PackKc, kAvx2PackNc, true>},
    HPC_TILE("Avx2Packed4x2", gemm_avx2_packed, 4, 2),
    HPC_BLOCK("Avx2Packed_5x7x50", gemm_avx2_packed, 6, 2, 5, 7, 50),
    Kernel{"Avx2PackedA_5x7x50", &gemm_avx2_packed<float, 6, 2, 5, 7, 50, true>,
           &gemm_avx2_packed<double, 6, 2, 5, 7, 50, true>},
#endif
#if HPC_HAS_AVX512
    HPC_BOTH("Avx512Naive", gemm_avx512_naive),
    HPC_BOTH("Avx512Reordered", gemm_avx512_reordered),
    HPC_BOTH("Avx512Blocked", gemm_avx512_blocked),
    HPC_BOTH("Avx512BlockedPrefetch", gemm_avx512_blocked_prefetch),
    HPC_TILE("Avx512Blocked8x2", gemm_avx512_blocked, 8, 2),
    HPC_TILE("Avx512Blocked4x2", gemm_avx512_blocked, 4, 2),
    HPC_TILE("Avx512Blocked6x3", gemm_avx512_blocked, 6, 3),
    HPC_TILE("Avx512Blocked12x2", gemm_avx512_blocked, 12, 2),
    HPC_BLOCK("Avx512Blocked_20x7x37", gemm_avx512_blocked, 8, 3, 20, 7, 37),
    HPC_BOTH("Avx512Packed", gemm_avx512_packed),
    Kernel{"Avx512PackedA", &gemm_avx512_packed<float, 8, 3, kAvx512PackMc, kAvx512PackKc, kAvx512PackNc, true>,
           &gemm_avx512_packed<double, 8, 3, kAvx512PackMc, kAvx512PackKc, kAvx512PackNc, true>},
    HPC_TILE("Avx512Packed4x2", gemm_avx512_packed, 4, 2),
    HPC_BLOCK("Avx512Packed_5x7x50", gemm_avx512_packed, 8, 3, 5, 7, 50),
    Kernel{"Avx512PackedA_5x7x50", &gemm_avx512_packed<float, 8, 3, 5, 7, 50, true>,
           &gemm_avx512_packed<double, 8, 3, 5, 7, 50, true>},
#endif
#if HPC_HAS_NEON
    HPC_BOTH("NeonNaive", gemm_neon_naive),
    HPC_BOTH("NeonReordered", gemm_neon_reordered),
    HPC_BOTH("NeonBlocked", gemm_neon_blocked),
    HPC_BOTH("NeonBlockedPrefetch", gemm_neon_blocked_prefetch),
    HPC_TILE("NeonBlocked1x1", gemm_neon_blocked, 1, 1),
    HPC_TILE("NeonBlocked4x2", gemm_neon_blocked, 4, 2),
    HPC_TILE("NeonBlocked6x3", gemm_neon_blocked, 6, 3),
    HPC_TILE("NeonBlocked3x5", gemm_neon_blocked, 3, 5),
    HPC_BLOCK("NeonBlocked_20x7x37", gemm_neon_blocked, 6, 4, 20, 7, 37),
    HPC_BOTH("NeonPacked", gemm_neon_packed),
    Kernel{"NeonPackedA", &gemm_neon_packed<float, 6, 4, kNeonPackMc, kNeonPackKc, kNeonPackNc, true>,
           &gemm_neon_packed<double, 6, 4, kNeonPackMc, kNeonPackKc, kNeonPackNc, true>},
    HPC_TILE("NeonPacked4x2", gemm_neon_packed, 4, 2),
    HPC_BLOCK("NeonPacked_5x7x50", gemm_neon_packed, 6, 4, 5, 7, 50),
    Kernel{"NeonPackedA_5x7x50", &gemm_neon_packed<float, 6, 4, 5, 7, 50, true>,
           &gemm_neon_packed<double, 6, 4, 5, 7, 50, true>},
#endif
#if HPC_HAS_SVE
    HPC_BOTH("SveNaive", gemm_sve_naive),
    HPC_BOTH("SveReordered", gemm_sve_reordered),
    HPC_BOTH("SveBlocked", gemm_sve_blocked),
    HPC_BOTH("SveBlockedPrefetch", gemm_sve_blocked_prefetch),
    HPC_TILE("SveBlocked1x1", gemm_sve_blocked, 1, 1),
    HPC_TILE("SveBlocked2x4", gemm_sve_blocked, 2, 4),
    HPC_TILE("SveBlocked4x4", gemm_sve_blocked, 4, 4),
    HPC_TILE("SveBlocked6x3", gemm_sve_blocked, 6, 3),
    HPC_TILE("SveBlocked8x2", gemm_sve_blocked, 8, 2),
    HPC_TILE("SveBlocked8x3", gemm_sve_blocked, 8, 3),
    HPC_TILE("SveBlocked6x4", gemm_sve_blocked, 6, 4),
    HPC_TILE("SveBlocked12x2", gemm_sve_blocked, 12, 2),
    HPC_TILE("SveBlocked2x8", gemm_sve_blocked, 2, 8),
    HPC_TILE("SveBlocked16x1", gemm_sve_blocked, 16, 1),
    HPC_BLOCK("SveBlocked_5x7x13", gemm_sve_blocked, 6, 4, 5, 7, 13),
    Kernel{"SvePrefetch6x4", &gemm_sve_blocked_prefetch<float, 2, 6, 4>,
           &gemm_sve_blocked_prefetch<double, 2, 6, 4>},
    Kernel{"SvePrefetchOddBlocks", &gemm_sve_blocked_prefetch<float, 2, 6, 4, 5, 7, 13>,
           &gemm_sve_blocked_prefetch<double, 2, 6, 4, 5, 7, 13>},
    Kernel{"SveBlockedPrefetch0", &gemm_sve_blocked_prefetch<float, 0>,
           &gemm_sve_blocked_prefetch<double, 0>},
    Kernel{"SveBlockedPrefetch2", &gemm_sve_blocked_prefetch<float, 2>,
           &gemm_sve_blocked_prefetch<double, 2>},
    Kernel{"SveBlockedPrefetch16", &gemm_sve_blocked_prefetch<float, 16>,
           &gemm_sve_blocked_prefetch<double, 16>},
#endif
#if HPC_HAS_SME
    HPC_BOTH("Sme1Tile", gemm_sme_1tile),
    HPC_BOTH("SmeTiles", gemm_sme_tiles),
    HPC_BOTH("SmeBlocked", gemm_sme_blocked),
    HPC_BOTH("Sme", gemm_sme),
    HPC_BLOCK("SmeOddBlocks", gemm_sme, 2, 2, 5, 7, 13),
    HPC_BLOCK("SmeBlockedOddBlocks", gemm_sme_blocked, 2, 2, 5, 7, 13),
    // Several threads: the default, a thread count that does not divide the
    // micro-tile count, and odd A blocks with more threads than some shapes
    // have micro-tile rows (those use fewer threads).
    Kernel{"SmeMt",
           +[](const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) { gemm_sme_mt(A, B, C); },
           +[](const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) { gemm_sme_mt(A, B, C); }},
    Kernel{"SmeMt3Threads",
           +[](const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) { gemm_sme_mt(A, B, C, 3); },
           +[](const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) { gemm_sme_mt(A, B, C, 3); }},
    Kernel{"SmeMtOddBlocks8Threads",
           +[](const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) { gemm_sme_mt<float, 5>(A, B, C, 8); },
           +[](const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) { gemm_sme_mt<double, 5>(A, B, C, 8); }},
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
// so partial sums are reloaded into ZA), M across its Mc block and N across
// its Nc block; empty matrices.
const Shape kShapes[] = {
    {1, 1, 1},       {2, 2, 2},       {3, 3, 3},     {4, 4, 4},         {5, 5, 5},
    {7, 7, 7},       {8, 8, 8},       {13, 13, 13},  {15, 15, 15},      {16, 16, 16},
    {17, 17, 17},    {31, 31, 31},    {32, 32, 32},  {33, 33, 33},      {64, 64, 64},
    {100, 100, 100}, {128, 128, 128}, {256, 256, 256},
    {3, 5, 2},       {7, 3, 9},       {17, 33, 15},  {33, 1, 65},       {63, 65, 64},
    {129, 7, 33},    {257, 64, 31},   {24, 777, 40}, {20, 4113, 36},    {1031, 70, 40},
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
// SVE column tails follow the runtime vector length, not one fixed ISA width.
// ===========================================================================

#if HPC_HAS_SVE
template <typename T, int R, int V>
void expect_sve_vector_edges() {
    const std::size_t vl = sizeof(T) == 4 ? svcntw() : svcntd();
    const KernelFn<T> kernels[] = {
        &gemm_sve_blocked<T, R, V>, &gemm_sve_blocked_prefetch<T, 0, R, V>,
        &gemm_sve_blocked_prefetch<T, 2, R, V>, &gemm_sve_blocked_prefetch<T, 4, R, V>,
        &gemm_sve_blocked_prefetch<T, 8, R, V>, &gemm_sve_blocked_prefetch<T, 16, R, V>};
    for (const std::size_t n : {vl - 1, vl, vl + 1, V * vl - 1, V * vl, V * vl + 1})
        for (std::size_t kernel = 0; kernel < std::size(kernels); ++kernel) {
            SCOPED_TRACE(kernel);
            // One full register tile, a partial row tile and two K blocks.
            expect_matches_naive(kernels[kernel], {R + 1, kSveTileK + 1, n}, "SVE vector edge");
            if (::testing::Test::HasFatalFailure())
                return;
        }
}

TEST(GemmSve, BlockedVectorEdgesFloat) {
    expect_sve_vector_edges<float, 4, 2>();
    expect_sve_vector_edges<float, 6, 4>();
    expect_sve_vector_edges<float, 2, 8>();
}
TEST(GemmSve, BlockedVectorEdgesDouble) {
    expect_sve_vector_edges<double, 4, 2>();
    expect_sve_vector_edges<double, 6, 4>();
    expect_sve_vector_edges<double, 2, 8>();
}
#endif

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

template <typename T, int R, int V>
void expect_sme_shape() {
    // Odd blocks split all three dimensions, forcing packed padding, C reloads,
    // and partial ZA tiles for every legal shape at the current streaming VL.
    const KernelFn<T> kernels[] = {&gemm_sme<T, R, V>, &gemm_sme_blocked<T, R, V>,
                                   &gemm_sme_tiles<T, R, V>, &gemm_sme<T, R, V, 5, 7, 13>,
                                   &gemm_sme_blocked<T, R, V, 5, 7, 13>};
    const std::size_t s         = sizeof(T) == 4 ? svcntsw() : svcntsd();
    const Shape edges[]         = {
        {R * s - 1, 15, V * s + 1},
        {    R * s,  9,     V * s},
        {R * s + 1, 17, V * s - 1},
        {        3,  0,         4},
        {        0,  5,         3},
        {        4,  5,         0}
    };
    for (auto kernel : kernels)
        for (auto shape : edges)
            expect_matches_naive(kernel, shape,
                                 ("SME " + std::to_string(R) + "x" + std::to_string(V)).c_str());
}

template <typename T, int R, int... Col>
void expect_sme_columns(std::integer_sequence<int, Col...>) {
    (expect_sme_shape<T, R, Col + 1>(), ...);
}

template <typename T, int... Row>
void expect_sme_rows(std::integer_sequence<int, Row...>) {
    (expect_sme_columns<T, Row + 1>(std::make_integer_sequence<int, sizeof(T) / (Row + 1)>{}), ...);
}

TEST(GemmSme, AllFloatTileShapes) {
    expect_sme_rows<float>(std::make_integer_sequence<int, 4>{});
}
TEST(GemmSme, AllDoubleTileShapes) {
    expect_sme_rows<double>(std::make_integer_sequence<int, 8>{});
}

// gemm_sme_mt over several (jc, pc) passes: K and N one past kSmeKc and
// kSmeNc, so the shared B panel is repacked between passes and the second
// pc pass accumulates into C, with uneven chunks across 3 threads.
TEST(GemmSme, MtSeveralPasses) {
    expect_matches_naive<float>(
        +[](const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) { gemm_sme_mt(A, B, C, 3); },
        Shape{97, kSmeKc + 1, kSmeNc + 1}, "SmeMt passes");
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
