#pragma once

/**
 * @file neon_lowp.hpp
 * @brief Low-precision GEMM on ARM NEON (Apple M4 P-core): four instruction
 *        shapes for 16-bit and 8-bit inputs.
 *
 *   gemm_neon_bf16      C (FP32)  = A (bf16) · B (bf16)   BFMMLA
 *   gemm_neon_i8        C (INT32) = A (int8) · B (int8)   SMMLA
 *   gemm_neon_f16       C (FP32)  = A (fp16) · B (fp16)   FMLAL / FMLAL2, FP32 accumulate
 *   gemm_neon_f16_acc   C (FP32)  = A (fp16) · B (fp16)   FMLA .8h, FP16 accumulate
 *
 * BFMMLA and SMMLA are small matrix multiplies, not vector FMAs. Each takes
 * a 2 × G block of A and a 2 × G block of Bᵀ (G = 4 bf16 or 8 INT8, one
 * 128-bit register each) and adds their 2 × 2 product into a 2 × 2 tile of
 * C held in one register:
 *
 *   C[2×2] += A[2×G] · B[G×2]     BFMMLA: 32 flops   SMMLA: 64 integer ops
 *
 * The micro-kernel keeps an 8 × 8 block of C in 16 such registers (4 row
 * pairs × 4 column pairs) and per k group loads 4 registers of A and 4 of
 * B for 16 instructions, the outer-product structure of the FP32 kernels
 * at 2 × 2-tile granularity. Both operands are packed so that each
 * register is one load: per k group, 8 bytes of each row of A (G elements
 * along k) and 8 bytes of each column of B (G rows of B transposed).
 *
 * FMLAL is the FP32 kernel's FMA with fp16 inputs widened to FP32 (4 lanes),
 * so it does the FP32 kernel's work per instruction and saves only memory.
 * FMLA .8h computes and accumulates in fp16 (8 lanes): twice the work per
 * instruction, at fp16's precision and range; the C tile is converted to
 * FP32 on store. Both use a 4-row tile with 16 accumulators, the shape of
 * gemm_neon_blocked's FP32 tile before it moved to 6 × 4.
 *
 * Blocking (elements): Mc = 256 rows; Kc = 1024 for 16-bit inputs and 2048
 * for INT8 (the same bytes); Nc = 512. K is zero-padded to a whole k group.
 */

#include "gemm/lowp_reference.hpp"
#include "hpc/isa.hpp"
#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"

#if HPC_HAS_NEON
    #include <arm_neon.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if HPC_HAS_NEON && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && \
    defined(__ARM_FEATURE_MATMUL_INT8) && defined(__ARM_FEATURE_FP16_FML) && \
    defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #define HPC_HAS_NEON_LOWP 1
#else
    #define HPC_HAS_NEON_LOWP 0
#endif

namespace hpc::gemm {

#if !HPC_HAS_NEON_LOWP

void gemm_neon_bf16(const Matrix<bf16>&, const Matrix<bf16>&, Matrix<float>&) = delete;  // needs AArch64 BF16
void gemm_neon_i8(const Matrix<std::int8_t>&, const Matrix<std::int8_t>&, Matrix<std::int32_t>&) = delete;  // needs AArch64 I8MM
void gemm_neon_f16(const Matrix<fp16>&, const Matrix<fp16>&, Matrix<float>&) = delete;  // needs AArch64 FP16 FML
void gemm_neon_f16_acc(const Matrix<fp16>&, const Matrix<fp16>&, Matrix<float>&) = delete;  // needs AArch64 FP16

#else

namespace neon_lowp_detail {

inline constexpr std::size_t kMc = 256, kNc = 512;

inline std::size_t round_up(std::size_t x, std::size_t m) { return (x + m - 1) / m * m; }

// ----------------------------------------------------------------------------
// BFMMLA / SMMLA: 8 × 8 C tile of 2 × 2 tiles
// ----------------------------------------------------------------------------

/// G = k elements per 8-byte word: 4 bf16 or 8 INT8.
template <typename In>
inline constexpr std::size_t kG = 8 / sizeof(In);
template <typename In>
inline constexpr std::size_t kKc = 2048 / sizeof(In);  // 1024 bf16, 2048 INT8

/// Pack mc × kc of A into 8-row strips: word [w·8 + r] = A(i0+r, G·w … G·w+G−1).
template <typename In>
void pack_a_mmla(const In* A, std::size_t lda, std::size_t mc, std::size_t kc,
                 std::uint64_t* dst) {
    constexpr std::size_t G = kG<In>;
    const std::size_t kw = (kc + G - 1) / G, full = kc / G;
    for (std::size_t i0 = 0; i0 < mc; i0 += 8) {
        const std::size_t rows = std::min<std::size_t>(8, mc - i0);
        std::uint64_t* strip   = dst + i0 * kw;
        for (std::size_t r = 0; r < 8; ++r) {
            if (r >= rows) {
                for (std::size_t w = 0; w < kw; ++w)
                    strip[w * 8 + r] = 0;
                continue;
            }
            const In* src = A + (i0 + r) * lda;
            for (std::size_t w = 0; w < full; ++w)
                std::memcpy(&strip[w * 8 + r], src + w * G, 8);
            if (full < kw) {
                In g[8] = {};
                std::memcpy(g, src + full * G, (kc - full * G) * sizeof(In));
                std::memcpy(&strip[full * 8 + r], g, 8);
            }
        }
    }
}

/// 8 × 8 transpose of bytes (rows r[0..7] → columns), three vtrn stages.
inline void transpose8x8_u8(uint8x8_t (&r)[8]) {
    const uint8x8x2_t a01 = vtrn_u8(r[0], r[1]), a23 = vtrn_u8(r[2], r[3]);
    const uint8x8x2_t a45 = vtrn_u8(r[4], r[5]), a67 = vtrn_u8(r[6], r[7]);
    const uint16x4x2_t b02 = vtrn_u16(vreinterpret_u16_u8(a01.val[0]), vreinterpret_u16_u8(a23.val[0]));
    const uint16x4x2_t b13 = vtrn_u16(vreinterpret_u16_u8(a01.val[1]), vreinterpret_u16_u8(a23.val[1]));
    const uint16x4x2_t b46 = vtrn_u16(vreinterpret_u16_u8(a45.val[0]), vreinterpret_u16_u8(a67.val[0]));
    const uint16x4x2_t b57 = vtrn_u16(vreinterpret_u16_u8(a45.val[1]), vreinterpret_u16_u8(a67.val[1]));
    const uint32x2x2_t c04 = vtrn_u32(vreinterpret_u32_u16(b02.val[0]), vreinterpret_u32_u16(b46.val[0]));
    const uint32x2x2_t c26 = vtrn_u32(vreinterpret_u32_u16(b02.val[1]), vreinterpret_u32_u16(b46.val[1]));
    const uint32x2x2_t c15 = vtrn_u32(vreinterpret_u32_u16(b13.val[0]), vreinterpret_u32_u16(b57.val[0]));
    const uint32x2x2_t c37 = vtrn_u32(vreinterpret_u32_u16(b13.val[1]), vreinterpret_u32_u16(b57.val[1]));
    r[0] = vreinterpret_u8_u32(c04.val[0]);
    r[1] = vreinterpret_u8_u32(c15.val[0]);
    r[2] = vreinterpret_u8_u32(c26.val[0]);
    r[3] = vreinterpret_u8_u32(c37.val[0]);
    r[4] = vreinterpret_u8_u32(c04.val[1]);
    r[5] = vreinterpret_u8_u32(c15.val[1]);
    r[6] = vreinterpret_u8_u32(c26.val[1]);
    r[7] = vreinterpret_u8_u32(c37.val[1]);
}

/// Pack kc × nc of B into 8-column strips: word [w·8 + c] = B(G·w … G·w+G−1, j0+c).
template <typename In>
void pack_b_mmla(const In* B, std::size_t ldb, std::size_t kc, std::size_t nc,
                 std::uint64_t* dst) {
    constexpr std::size_t G = kG<In>;
    const std::size_t kw    = (kc + G - 1) / G;
    for (std::size_t j0 = 0; j0 < nc; j0 += 8) {
        const std::size_t cols = std::min<std::size_t>(8, nc - j0);
        std::uint64_t* strip   = dst + j0 * kw;
        for (std::size_t w = 0; w < kw; ++w) {
            std::uint64_t* out = strip + w * 8;
            const In* row      = B + w * G * ldb + j0;
            if (cols == 8 && (w + 1) * G <= kc) {
                if constexpr (G == 4) {  // 4 rows × 8 bf16 columns → 8 words of 4
                    const uint16x8_t r0 = vld1q_u16(reinterpret_cast<const std::uint16_t*>(row));
                    const uint16x8_t r1 = vld1q_u16(reinterpret_cast<const std::uint16_t*>(row + ldb));
                    const uint16x8_t r2 = vld1q_u16(reinterpret_cast<const std::uint16_t*>(row + 2 * ldb));
                    const uint16x8_t r3 = vld1q_u16(reinterpret_cast<const std::uint16_t*>(row + 3 * ldb));
                    const uint32x4_t lo01 = vreinterpretq_u32_u16(vzip1q_u16(r0, r1));
                    const uint32x4_t hi01 = vreinterpretq_u32_u16(vzip2q_u16(r0, r1));
                    const uint32x4_t lo23 = vreinterpretq_u32_u16(vzip1q_u16(r2, r3));
                    const uint32x4_t hi23 = vreinterpretq_u32_u16(vzip2q_u16(r2, r3));
                    vst1q_u32(reinterpret_cast<std::uint32_t*>(out), vzip1q_u32(lo01, lo23));
                    vst1q_u32(reinterpret_cast<std::uint32_t*>(out + 2), vzip2q_u32(lo01, lo23));
                    vst1q_u32(reinterpret_cast<std::uint32_t*>(out + 4), vzip1q_u32(hi01, hi23));
                    vst1q_u32(reinterpret_cast<std::uint32_t*>(out + 6), vzip2q_u32(hi01, hi23));
                } else {  // 8 rows × 8 INT8 columns → 8 words of 8
                    uint8x8_t r[8];
                    for (std::size_t e = 0; e < 8; ++e)
                        r[e] = vld1_u8(reinterpret_cast<const std::uint8_t*>(row + e * ldb));
                    transpose8x8_u8(r);
                    for (std::size_t c = 0; c < 8; ++c)
                        vst1_u8(reinterpret_cast<std::uint8_t*>(out + c), r[c]);
                }
                continue;
            }
            for (std::size_t c = 0; c < 8; ++c) {
                In g[8] = {};
                if (c < cols)
                    for (std::size_t e = 0; e < G && w * G + e < kc; ++e)
                        g[e] = row[e * ldb + c];
                std::memcpy(&out[c], g, 8);
            }
        }
    }
}

template <typename In> struct Mmla;
template <> struct Mmla<bf16> {
    using Acc = float32x4_t;
    using Out = float;
    static Acc zero() { return vdupq_n_f32(0.f); }
    static Acc mmla(Acc c, const std::uint64_t* a, const std::uint64_t* b) {
        return vbfmmlaq_f32(c, vreinterpretq_bf16_u64(vld1q_u64(a)), vreinterpretq_bf16_u64(vld1q_u64(b)));
    }
    static Acc load2(const float* r0, const float* r1) { return vcombine_f32(vld1_f32(r0), vld1_f32(r1)); }
    static void store2(float* r0, float* r1, Acc v) {
        vst1_f32(r0, vget_low_f32(v));
        vst1_f32(r1, vget_high_f32(v));
    }
};
template <> struct Mmla<std::int8_t> {
    using Acc = int32x4_t;
    using Out = std::int32_t;
    static Acc zero() { return vdupq_n_s32(0); }
    static Acc mmla(Acc c, const std::uint64_t* a, const std::uint64_t* b) {
        return vmmlaq_s32(c, vreinterpretq_s8_u64(vld1q_u64(a)), vreinterpretq_s8_u64(vld1q_u64(b)));
    }
    static Acc load2(const std::int32_t* r0, const std::int32_t* r1) {
        return vcombine_s32(vld1_s32(r0), vld1_s32(r1));
    }
    static void store2(std::int32_t* r0, std::int32_t* r1, Acc v) {
        vst1_s32(r0, vget_low_s32(v));
        vst1_s32(r1, vget_high_s32(v));
    }
};

/// 8 × 8 tile: C (+)= A strip · B strip over kw k groups.
template <typename In>
void micro_mmla(const std::uint64_t* __restrict a, const std::uint64_t* __restrict b,
                typename Mmla<In>::Out* __restrict c, std::size_t ldc, std::size_t kw,
                std::size_t rows, std::size_t cols, bool accumulate) {
    using M     = Mmla<In>;
    using Out   = typename M::Out;
    const bool full = rows == 8 && cols == 8;
    Out tmp[8][8];
    Out* base         = full ? c : &tmp[0][0];
    const std::size_t ld = full ? ldc : 8;
    if (!full && accumulate)
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < cols; ++j)
                tmp[r][j] = c[r * ldc + j];

    typename M::Acc acc[4][4];
#pragma GCC unroll 4
    for (int p = 0; p < 4; ++p)
#pragma GCC unroll 4
        for (int q = 0; q < 4; ++q)
            acc[p][q] = accumulate
                            ? M::load2(base + (2 * p) * ld + 2 * q, base + (2 * p + 1) * ld + 2 * q)
                            : M::zero();
    for (std::size_t w = 0; w < kw; ++w) {
        const std::uint64_t* aw = a + w * 8;
        const std::uint64_t* bw = b + w * 8;
#pragma GCC unroll 4
        for (int p = 0; p < 4; ++p)
#pragma GCC unroll 4
            for (int q = 0; q < 4; ++q)
                acc[p][q] = M::mmla(acc[p][q], aw + 2 * p, bw + 2 * q);
    }
#pragma GCC unroll 4
    for (int p = 0; p < 4; ++p)
#pragma GCC unroll 4
        for (int q = 0; q < 4; ++q)
            M::store2(base + (2 * p) * ld + 2 * q, base + (2 * p + 1) * ld + 2 * q, acc[p][q]);
    if (!full)
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < cols; ++j)
                c[r * ldc + j] = tmp[r][j];
}

template <typename In>
void driver_mmla(const Matrix<In>& A, const Matrix<In>& B, Matrix<lowp_out_t<In>>& C) {
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;
    constexpr std::size_t G = kG<In>, Kc = kKc<In>;
    const std::size_t kw_max = (std::min(Kc, K) + G - 1) / G;
    Matrix<std::uint64_t> a_pack(round_up(std::min(kMc, M), 8), kw_max);
    Matrix<std::uint64_t> b_pack(kw_max, round_up(std::min(kNc, N), 8));
    for (std::size_t jc = 0; jc < N; jc += kNc) {
        const std::size_t nc = std::min(kNc, N - jc);
        for (std::size_t pc = 0; pc < K; pc += Kc) {
            const std::size_t kc = std::min(Kc, K - pc), kw = (kc + G - 1) / G;
            pack_b_mmla(B.data() + pc * N + jc, N, kc, nc, b_pack.data());
            for (std::size_t ic = 0; ic < M; ic += kMc) {
                const std::size_t mc = std::min(kMc, M - ic);
                pack_a_mmla(A.data() + ic * K + pc, K, mc, kc, a_pack.data());
                for (std::size_t jr = 0; jr < nc; jr += 8)
                    for (std::size_t ir = 0; ir < mc; ir += 8)
                        micro_mmla<In>(a_pack.data() + ir * kw, b_pack.data() + jr * kw,
                                       C.data() + (ic + ir) * N + jc + jr, N, kw,
                                       std::min<std::size_t>(8, mc - ir),
                                       std::min<std::size_t>(8, nc - jr), pc > 0);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// FMLAL (fp16 → FP32) and FMLA .8h (fp16 accumulate): 4-row tiles
// ----------------------------------------------------------------------------

inline constexpr std::size_t kKcFma = 512;

/// Pack mc × kc of fp16 A into 4-row strips, k-major: dst[k·4 + r] = A(i0+r, k).
inline void pack_a_fma(const fp16* A, std::size_t lda, std::size_t mc, std::size_t kc, fp16* dst) {
    for (std::size_t i0 = 0; i0 < mc; i0 += 4) {
        fp16* strip = dst + i0 * kc;
        for (std::size_t k = 0; k < kc; ++k)
            for (std::size_t r = 0; r < 4; ++r)
                strip[k * 4 + r] = i0 + r < mc ? A[(i0 + r) * lda + k] : fp16{0};
    }
}

/// Pack kc × nc of fp16 B into nr-column strips, row-major within a strip.
inline void pack_b_fma(const fp16* B, std::size_t ldb, std::size_t kc, std::size_t nc,
                       std::size_t nr, fp16* dst) {
    for (std::size_t j0 = 0; j0 < nc; j0 += nr) {
        const std::size_t cols = std::min(nr, nc - j0);
        fp16* strip            = dst + j0 * kc;
        for (std::size_t k = 0; k < kc; ++k) {
            std::memcpy(strip + k * nr, B + k * ldb + j0, cols * sizeof(fp16));
            std::fill(strip + k * nr + cols, strip + (k + 1) * nr, fp16{0});
        }
    }
}

/// 4 × 16 tile, FMLAL: C(r, 0..15) += A(r, k) · B(k, 0..15), widened to FP32.
inline void micro_fmlal(const fp16* __restrict a, const fp16* __restrict b, float* __restrict c,
                        std::size_t ldc, std::size_t kc, std::size_t rows, std::size_t cols,
                        bool accumulate) {
    float tmp[4][16];
    const bool full    = rows == 4 && cols == 16;
    float* base        = full ? c : &tmp[0][0];
    const std::size_t ld = full ? ldc : 16;
    if (!full && accumulate)
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < cols; ++j)
                tmp[r][j] = c[r * ldc + j];
    float32x4_t acc[4][4];
#pragma GCC unroll 4
    for (int r = 0; r < 4; ++r)
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v)
            acc[r][v] = accumulate ? vld1q_f32(base + r * ld + 4 * v) : vdupq_n_f32(0.f);
    const auto* ap = reinterpret_cast<const float16_t*>(a);
    const auto* bp = reinterpret_cast<const float16_t*>(b);
    for (std::size_t k = 0; k < kc; ++k) {
        const float16x8_t b0 = vld1q_f16(bp + k * 16), b1 = vld1q_f16(bp + k * 16 + 8);
        const float16x4_t av = vld1_f16(ap + k * 4);
#define HPC_FMLAL_ROW(r)                                          \
    acc[r][0] = vfmlalq_lane_low_f16(acc[r][0], b0, av, r);       \
    acc[r][1] = vfmlalq_lane_high_f16(acc[r][1], b0, av, r);      \
    acc[r][2] = vfmlalq_lane_low_f16(acc[r][2], b1, av, r);       \
    acc[r][3] = vfmlalq_lane_high_f16(acc[r][3], b1, av, r);
        HPC_FMLAL_ROW(0)
        HPC_FMLAL_ROW(1)
        HPC_FMLAL_ROW(2)
        HPC_FMLAL_ROW(3)
#undef HPC_FMLAL_ROW
    }
#pragma GCC unroll 4
    for (int r = 0; r < 4; ++r)
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v)
            vst1q_f32(base + r * ld + 4 * v, acc[r][v]);
    if (!full)
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < cols; ++j)
                c[r * ldc + j] = tmp[r][j];
}

/// 4 × 32 tile, FMLA .8h: C(r, 0..31) += A(r, k) · B(k, 0..31) in fp16.
inline void micro_f16acc(const fp16* __restrict a, const fp16* __restrict b, float* __restrict c,
                         std::size_t ldc, std::size_t kc, std::size_t rows, std::size_t cols,
                         bool accumulate) {
    float tmp[4][32];
    const bool full    = rows == 4 && cols == 32;
    float* base        = full ? c : &tmp[0][0];
    const std::size_t ld = full ? ldc : 32;
    if (!full && accumulate)
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < cols; ++j)
                tmp[r][j] = c[r * ldc + j];
    float16x8_t acc[4][4];
#pragma GCC unroll 4
    for (int r = 0; r < 4; ++r)
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v)
            acc[r][v] = accumulate ? vcombine_f16(vcvt_f16_f32(vld1q_f32(base + r * ld + 8 * v)),
                                                  vcvt_f16_f32(vld1q_f32(base + r * ld + 8 * v + 4)))
                                   : vdupq_n_f16(0);
    const auto* ap = reinterpret_cast<const float16_t*>(a);
    const auto* bp = reinterpret_cast<const float16_t*>(b);
    for (std::size_t k = 0; k < kc; ++k) {
        const float16x8_t b0 = vld1q_f16(bp + k * 32), b1 = vld1q_f16(bp + k * 32 + 8);
        const float16x8_t b2 = vld1q_f16(bp + k * 32 + 16), b3 = vld1q_f16(bp + k * 32 + 24);
        const float16x4_t av = vld1_f16(ap + k * 4);
#define HPC_FMLA_ROW(r)                                \
    acc[r][0] = vfmaq_lane_f16(acc[r][0], b0, av, r);  \
    acc[r][1] = vfmaq_lane_f16(acc[r][1], b1, av, r);  \
    acc[r][2] = vfmaq_lane_f16(acc[r][2], b2, av, r);  \
    acc[r][3] = vfmaq_lane_f16(acc[r][3], b3, av, r);
        HPC_FMLA_ROW(0)
        HPC_FMLA_ROW(1)
        HPC_FMLA_ROW(2)
        HPC_FMLA_ROW(3)
#undef HPC_FMLA_ROW
    }
#pragma GCC unroll 4
    for (int r = 0; r < 4; ++r)
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) {
            vst1q_f32(base + r * ld + 8 * v, vcvt_f32_f16(vget_low_f16(acc[r][v])));
            vst1q_f32(base + r * ld + 8 * v + 4, vcvt_high_f32_f16(acc[r][v]));
        }
    if (!full)
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < cols; ++j)
                c[r * ldc + j] = tmp[r][j];
}

template <std::size_t NR, typename Micro>
void driver_fma(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C, Micro micro) {
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;
    const std::size_t kc_max = std::min(kKcFma, K);
    Matrix<fp16> a_pack(round_up(std::min(kMc, M), 4), kc_max);
    Matrix<fp16> b_pack(kc_max, round_up(std::min(kNc, N), NR));
    for (std::size_t jc = 0; jc < N; jc += kNc) {
        const std::size_t nc = std::min(kNc, N - jc);
        for (std::size_t pc = 0; pc < K; pc += kKcFma) {
            const std::size_t kc = std::min(kKcFma, K - pc);
            pack_b_fma(B.data() + pc * N + jc, N, kc, nc, NR, b_pack.data());
            for (std::size_t ic = 0; ic < M; ic += kMc) {
                const std::size_t mc = std::min(kMc, M - ic);
                pack_a_fma(A.data() + ic * K + pc, K, mc, kc, a_pack.data());
                for (std::size_t jr = 0; jr < nc; jr += NR)
                    for (std::size_t ir = 0; ir < mc; ir += 4)
                        micro(a_pack.data() + ir * kc, b_pack.data() + jr * kc,
                              C.data() + (ic + ir) * N + jc + jr, N, kc,
                              std::min<std::size_t>(4, mc - ir), std::min(NR, nc - jr), pc > 0);
            }
        }
    }
}

}  // namespace neon_lowp_detail

/// C (FP32) = A · B, bf16 inputs, BFMMLA. See the file header.
inline void gemm_neon_bf16(const Matrix<bf16>& A, const Matrix<bf16>& B, Matrix<float>& C) {
    neon_lowp_detail::driver_mmla(A, B, C);
}

/// C (INT32) = A · B, INT8 inputs, SMMLA. See the file header.
inline void gemm_neon_i8(const Matrix<std::int8_t>& A, const Matrix<std::int8_t>& B,
                         Matrix<std::int32_t>& C) {
    neon_lowp_detail::driver_mmla(A, B, C);
}

/// C (FP32) = A · B, fp16 inputs widened to FP32 (FMLAL). See the file header.
inline void gemm_neon_f16(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C) {
    neon_lowp_detail::driver_fma<16>(A, B, C, neon_lowp_detail::micro_fmlal);
}

/// C (FP32) = A · B, fp16 inputs, fp16 accumulation (FMLA .8h): faster, much
/// less accurate. See the file header.
inline void gemm_neon_f16_acc(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C) {
    neon_lowp_detail::driver_fma<32>(A, B, C, neon_lowp_detail::micro_f16acc);
}

#endif  // HPC_HAS_NEON_LOWP

}  // namespace hpc::gemm
