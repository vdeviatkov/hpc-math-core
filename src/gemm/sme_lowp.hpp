#pragma once

/**
 * @file sme_lowp.hpp
 * @brief Low-precision GEMM on ARM SME2: widening outer products.
 *
 *   gemm_sme_widening<fp16>        C (FP32)  = A (fp16) · B (fp16)
 *   gemm_sme_widening<bf16>        C (FP32)  = A (bf16) · B (bf16)
 *   gemm_sme_widening<std::int8_t> C (INT32) = A (int8) · B (int8)
 *
 * The structure is gemm_sme's (sme.hpp): all four 32-bit ZA tiles hold a
 * 2·SVL × 2·SVL block of C, GotoBLAS blocking, A and B packed. What changes
 * is the instruction. A widening FMOPA / SMOPA reads two vectors whose
 * elements come in groups of G along k (G = 2 for 16-bit inputs, 4 for
 * INT8) and adds a dot product of length G into each 32-bit accumulator:
 *
 *   ZA[t][r][c] += Σ_g a[r·G + g] · b[c·G + g]      g = 0 … G−1
 *
 * so one instruction does G times the work of the FP32 FMOPA: 1,024 flops
 * (16-bit) or 2,048 integer operations (INT8) on M4's 512-bit vectors, and
 * C stays in the same 32-bit tiles. The packed strips must hold each k
 * group contiguously — one 32-bit word per (row or column, k group):
 *
 *   A strip: word [kg][r]  = A(r, G·kg … G·kg+G−1)   a k group of a row: contiguous in A
 *   B strip: word [kg][c]  = B(G·kg … G·kg+G−1, c)   G rows of B interleaved
 *
 * A is therefore packed by transposing 32-bit words (NEON 4 × 4 word
 * transposes, as pack_a in sme.hpp), B by interleaving G rows (NEON zips).
 * Without packing B there is no way to feed these instructions: B in place
 * has its k groups N elements apart.
 *
 * Blocks keep the FP32 kernel's sizes in bytes: an 8 MiB A block
 * (Mc = 1024 rows × Kc = 4096 16-bit or 8192 INT8 elements) and
 * Nc = 4096. K is zero-padded to a multiple of G.
 *
 * M4 implements these instructions (SME_F16F32, SME_B16F32, SME_I8I32), but
 * not the non-widening FP16 → FP16 or BF16 → BF16 forms.
 */

#include "gemm/sme.hpp"
#include "hpc/lowp.hpp"
#include "gemm/lowp_reference.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace hpc::gemm {

#if !HPC_HAS_SME

template <typename In>
void gemm_sme_widening(const Matrix<In>&, const Matrix<In>&, Matrix<lowp_out_t<In>>&) = delete;  // SME2 not available on this target

#else

namespace sme_lowp_detail {

/// k elements per 32-bit word: 2 for fp16 / bf16, 4 for INT8.
template <typename In>
inline constexpr std::size_t kG = 4 / sizeof(In);

/// Blocking, in elements, for an 8 MiB A block like the FP32 kernel's.
inline constexpr std::size_t kMc = 1024;
template <typename In>
inline constexpr std::size_t kKc = 8192 / sizeof(In);
inline constexpr std::size_t kNc = 4096;

static_assert(kKc<std::int8_t> == 8192 && kKc<bf16> == 4096);

/// One 32-bit word: the G elements of a k group.
inline std::uint32_t word(const void* p) noexcept {
    std::uint32_t w;
    std::memcpy(&w, p, 4);
    return w;
}

/// Pack an mc × kc block of A (row-major, lda elements) into mr-row strips
/// of k-group words: strip word [kg·mr + r] = A(i0 + r, G·kg … G·kg + G−1),
/// zero-padded to mr rows and to a whole last group.
template <typename In>
void pack_a(const In* A, std::size_t lda, std::size_t mc, std::size_t kc, std::size_t mr,
            std::uint32_t* dst) {
    constexpr std::size_t G = kG<In>;
    const std::size_t kw    = (kc + G - 1) / G;  // words per row
    const std::size_t full  = kc / G;            // whole groups
    for (std::size_t i0 = 0; i0 < mc; i0 += mr) {
        const std::size_t rows = std::min(mr, mc - i0);
        std::uint32_t* strip   = dst + i0 * kw;
        std::size_t r          = 0;
        for (; r + 4 <= rows; r += 4) {  // 4 rows × 4 words: NEON 4 × 4 word transpose
            const In* src = A + (i0 + r) * lda;
            std::size_t w = 0;
            for (; w + 4 <= full; w += 4) {
                const uint32x4_t r0 = vreinterpretq_u32_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>(src + w * G)));
                const uint32x4_t r1 = vreinterpretq_u32_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>(src + lda + w * G)));
                const uint32x4_t r2 = vreinterpretq_u32_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>(src + 2 * lda + w * G)));
                const uint32x4_t r3 = vreinterpretq_u32_u8(vld1q_u8(reinterpret_cast<const std::uint8_t*>(src + 3 * lda + w * G)));
                const uint32x4_t t0 = vtrn1q_u32(r0, r1), t1 = vtrn2q_u32(r0, r1);
                const uint32x4_t t2 = vtrn1q_u32(r2, r3), t3 = vtrn2q_u32(r2, r3);
                std::uint32_t* out = strip + w * mr + r;
                vst1q_u32(out,          vcombine_u32(vget_low_u32(t0),  vget_low_u32(t2)));
                vst1q_u32(out + mr,     vcombine_u32(vget_low_u32(t1),  vget_low_u32(t3)));
                vst1q_u32(out + 2 * mr, vcombine_u32(vget_high_u32(t0), vget_high_u32(t2)));
                vst1q_u32(out + 3 * mr, vcombine_u32(vget_high_u32(t1), vget_high_u32(t3)));
            }
            for (; w < full; ++w)
                for (std::size_t rr = 0; rr < 4; ++rr)
                    strip[w * mr + r + rr] = word(src + rr * lda + w * G);
        }
        for (; r < rows; ++r)
            for (std::size_t w = 0; w < full; ++w)
                strip[w * mr + r] = word(A + (i0 + r) * lda + w * G);
        if (full < kw)  // the last, partial k group: copy what exists, zero the rest
            for (std::size_t rr = 0; rr < rows; ++rr) {
                In g[4] = {};
                for (std::size_t e = 0; e < kc - full * G; ++e)
                    g[e] = A[(i0 + rr) * lda + full * G + e];
                strip[full * mr + rr] = word(g);
            }
        if (rows < mr)
            for (std::size_t w = 0; w < kw; ++w)
                std::fill(strip + w * mr + rows, strip + (w + 1) * mr, 0u);
    }
}

/// Pack a kc × nc panel of B (row-major, ldb elements) into nr-column strips
/// of k-group words: strip word [kg·nr + c] = B(G·kg … G·kg + G−1, j0 + c),
/// zero-padded to nr columns and to a whole last group.
template <typename In>
void pack_b(const In* B, std::size_t ldb, std::size_t kc, std::size_t nc, std::size_t nr,
            std::uint32_t* dst) {
    constexpr std::size_t G = kG<In>;
    const std::size_t kw    = (kc + G - 1) / G;
    for (std::size_t j0 = 0; j0 < nc; j0 += nr) {
        const std::size_t cols = std::min(nr, nc - j0);
        std::uint32_t* strip   = dst + j0 * kw;
        for (std::size_t w = 0; w < kw; ++w) {
            std::uint32_t* out = strip + w * nr;
            const In* row      = B + w * G * ldb + j0;
            const bool whole   = (w + 1) * G <= kc;
            std::size_t c      = 0;
            if (whole) {
                if constexpr (G == 2) {  // zip two rows of 16-bit elements: 8 columns per step
                    for (; c + 8 <= cols; c += 8) {
                        const uint16x8_t r0 = vld1q_u16(reinterpret_cast<const std::uint16_t*>(row + c));
                        const uint16x8_t r1 = vld1q_u16(reinterpret_cast<const std::uint16_t*>(row + ldb + c));
                        vst1q_u16(reinterpret_cast<std::uint16_t*>(out + c), vzip1q_u16(r0, r1));
                        vst1q_u16(reinterpret_cast<std::uint16_t*>(out + c + 4), vzip2q_u16(r0, r1));
                    }
                } else {  // interleave four rows of bytes: 16 columns per step
                    for (; c + 16 <= cols; c += 16) {
                        const auto* p      = reinterpret_cast<const std::uint8_t*>(row + c);
                        const uint8x16_t r0 = vld1q_u8(p), r1 = vld1q_u8(p + ldb);
                        const uint8x16_t r2 = vld1q_u8(p + 2 * ldb), r3 = vld1q_u8(p + 3 * ldb);
                        const uint16x8_t lo01 = vreinterpretq_u16_u8(vzip1q_u8(r0, r1));
                        const uint16x8_t hi01 = vreinterpretq_u16_u8(vzip2q_u8(r0, r1));
                        const uint16x8_t lo23 = vreinterpretq_u16_u8(vzip1q_u8(r2, r3));
                        const uint16x8_t hi23 = vreinterpretq_u16_u8(vzip2q_u8(r2, r3));
                        vst1q_u16(reinterpret_cast<std::uint16_t*>(out + c), vzip1q_u16(lo01, lo23));
                        vst1q_u16(reinterpret_cast<std::uint16_t*>(out + c + 4), vzip2q_u16(lo01, lo23));
                        vst1q_u16(reinterpret_cast<std::uint16_t*>(out + c + 8), vzip1q_u16(hi01, hi23));
                        vst1q_u16(reinterpret_cast<std::uint16_t*>(out + c + 12), vzip2q_u16(hi01, hi23));
                    }
                }
            }
            for (; c < cols; ++c) {
                In g[4] = {};
                for (std::size_t e = 0; e < G && w * G + e < kc; ++e)
                    g[e] = row[e * ldb + c];
                out[c] = word(g);
            }
            std::fill(out + cols, out + nr, 0u);
        }
    }
}

/// ZA[Tile] += widening outer product of two k-group vectors.
template <typename In, int Tile>
[[gnu::always_inline]] inline void mopa(svuint32_t a, svuint32_t b) __arm_streaming __arm_inout("za") {
    if constexpr (std::is_same_v<In, fp16>)
        svmopa_za32_f16_m(Tile, svptrue_b16(), svptrue_b16(), svreinterpret_f16(a), svreinterpret_f16(b));
    else if constexpr (std::is_same_v<In, bf16>)
        svmopa_za32_bf16_m(Tile, svptrue_b16(), svptrue_b16(), svreinterpret_bf16(a), svreinterpret_bf16(b));
    else
        svmopa_za32_s8_m(Tile, svptrue_b8(), svptrue_b8(), svreinterpret_s8(a), svreinterpret_s8(b));
}

/// Macro-kernel (streaming): C[mc × nc] (+)= Apack · Bpack over kw k groups,
/// the 2 × 2 tiles of gemm_sme. C is FP32 or INT32; ZA moves are bit copies.
template <typename In>
__arm_locally_streaming __arm_new("za")
void macro_kernel(const std::uint32_t* __restrict Ap, const std::uint32_t* __restrict Bp,
                  lowp_out_t<In>* __restrict C, std::int64_t ldc, std::int64_t mc,
                  std::int64_t nc, std::int64_t kw, bool accumulate) {
    const std::int64_t s  = svcntsw();  // 32-bit lanes per vector, rows/columns per tile
    const std::int64_t mr = 2 * s, nr = 2 * s;
    const svcount_t pn    = svptrue_c32();

    for (std::int64_t jr = 0; jr < nc; jr += nr) {
        const std::int64_t cols      = std::min(nr, nc - jr);
        const std::uint32_t* b_strip = Bp + jr * kw;
        for (std::int64_t ir = 0; ir < mc; ir += mr) {
            const std::int64_t rows      = std::min(mr, mc - ir);
            const std::uint32_t* a_strip = Ap + ir * kw;
            float* c                     = reinterpret_cast<float*>(C + ir * ldc + jr);
            if (accumulate)
                sme_detail::move_c<float, true>(c, ldc, s, rows, cols);
            else
                svzero_za();
            for (std::int64_t w = 0; w < kw; ++w) {
                const svuint32x2_t a  = svld1_x2(pn, a_strip + w * mr);
                const svuint32x2_t b  = svld1_x2(pn, b_strip + w * nr);
                const svuint32_t a0 = svget2(a, 0), a1 = svget2(a, 1);
                mopa<In, 0>(a0, svget2(b, 0));
                mopa<In, 1>(a0, svget2(b, 1));
                mopa<In, 2>(a1, svget2(b, 0));
                mopa<In, 3>(a1, svget2(b, 1));
            }
            sme_detail::move_c<float, false>(c, ldc, s, rows, cols);
        }
    }
}

}  // namespace sme_lowp_detail

/**
 * @brief C = A · B on ARM SME2 with widening outer products: fp16 or bf16
 *        inputs with FP32 accumulation and output, or INT8 inputs with
 *        INT32 accumulation and output. See the file header.
 */
template <typename In>
void gemm_sme_widening(const Matrix<In>& A, const Matrix<In>& B, Matrix<lowp_out_t<In>>& C) {
    static_assert(std::is_same_v<In, fp16> || std::is_same_v<In, bf16> ||
                      std::is_same_v<In, std::int8_t>,
                  "gemm_sme_widening: In must be fp16, bf16 or std::int8_t");
    namespace d = sme_lowp_detail;
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;

    constexpr std::size_t G = d::kG<In>;
    const std::size_t s     = svcntsw();
    const std::size_t mr = 2 * s, nr = 2 * s;
    auto round_up        = [](std::size_t x, std::size_t m) { return (x + m - 1) / m * m; };
    const std::size_t kw_max = (std::min(d::kKc<In>, K) + G - 1) / G;
    Matrix<std::uint32_t> a_pack(round_up(std::min(d::kMc, M), mr), kw_max);
    Matrix<std::uint32_t> b_pack(kw_max, round_up(std::min(d::kNc, N), nr));

    for (std::size_t jc = 0; jc < N; jc += d::kNc) {
        const std::size_t nc = std::min(d::kNc, N - jc);
        for (std::size_t pc = 0; pc < K; pc += d::kKc<In>) {
            const std::size_t kc = std::min(d::kKc<In>, K - pc);
            const std::size_t kw = (kc + G - 1) / G;
            d::pack_b(B.data() + pc * N + jc, N, kc, nc, nr, b_pack.data());
            for (std::size_t ic = 0; ic < M; ic += d::kMc) {
                const std::size_t mc = std::min(d::kMc, M - ic);
                d::pack_a(A.data() + ic * K + pc, K, mc, kc, mr, a_pack.data());
                d::macro_kernel<In>(a_pack.data(), b_pack.data(), C.data() + ic * N + jc,
                                    static_cast<std::int64_t>(N), static_cast<std::int64_t>(mc),
                                    static_cast<std::int64_t>(nc), static_cast<std::int64_t>(kw),
                                    pc > 0);
            }
        }
    }
}

#endif  // HPC_HAS_SME

}  // namespace hpc::gemm
