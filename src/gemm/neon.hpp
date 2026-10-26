#pragma once

/**
 * @file neon.hpp
 * @brief ARM NEON primitives for the generic SIMD kernels (simd.hpp).
 *
 *   gemm_simd_reordered<Q>(A, B, C);
 *   gemm_simd_blocked<Q, R, V, TileM, TileK, TileN[, PfDist]>(A, B, C);
 *   gemm_simd_packed<Q, R, V, Mc, Kc, Nc[, PackA]>(A, B, C);
 *   gemm_simd_lowp<QMmla, R, V, Mc, Kc, Nc>(A, B, C);      // bf16 or INT8: BFMMLA, SMMLA
 *   gemm_simd_lowp<QFmlal, 4, V, Mc, Kc, Nc>(A, B, C);     // fp16 → FP32: FMLAL
 *   gemm_simd_lowp<QFmlaF16, 4, V, Mc, Kc, Nc>(A, B, C);   // fp16 accumulate: FMLA .8h
 *
 * AArch64 has 32 × 128-bit Q registers (4 f32 / 2 f64). Apple M4 P-core
 * (measured with fma_peak, docs/experiments/cpu-simd): up to 4 FMAs per
 * cycle, ~3-cycle latency.
 *   f32: 4 pipes × 4 lanes × 2 FLOP = 32 FLOP/cycle → ~143 GFLOP/s at 4.49 GHz
 *   f64: 4 pipes × 2 lanes × 2 FLOP = 16 FLOP/cycle →  ~72 GFLOP/s
 * NEON is half the width of AVX2, but twice the FMA pipes give the same
 * FLOP/cycle.
 *
 * HPC_HAS_NEON (hpc/isa.hpp) requires AArch64: Q<double> uses f64 lanes
 * and vfmaq_f64, which 32-bit ARMv7 NEON lacks. Q<T> is declared on every
 * target and defined only where HPC_HAS_NEON is 1; the low-precision structs
 * need the BF16, I8MM and FP16 extensions too (HPC_HAS_NEON_LOWP).
 */

#include "hpc/isa.hpp"
#include "hpc/lowp.hpp"

#if HPC_HAS_NEON
    #include <arm_neon.h>
#endif

#include <cstddef>
#include <cstdint>

namespace hpc::gemm {

template <typename T> struct Q;

#if HPC_HAS_NEON

// load_a puts A(i, k) in lane 0 of a 64-bit D register (S) with a plain LDR,
// and fma multiplies a B vector by that lane (FMLA by element). Broadcasting
// A with LD1R (load and replicate), which Clang emits for vdupq + vfmaq,
// held tiles with many rows back on M4: with LDR the 6 × 4 tile is 23–30%
// faster in the full multiplication (docs/kernels/neon.md).
template <> struct Q<float> {
    using V = float32x4_t;
    using S = float32x2_t;
    static V load(const float* p) { return vld1q_f32(p); }
    static void store(float* p, V v) { vst1q_f32(p, v); }
    static S load_a(const float* p) {
#if defined(__GNUC__)
        // From intrinsics (vld1_lane, vset_lane) Clang 17 folds the load and
        // the by-element FMA back into LD1R, so the LDR is inline assembly.
        S x;
        asm("ldr %s0, %1" : "=w"(x) : "m"(*p));
        return x;
#else
        return vld1_dup_f32(p);
#endif
    }
    static V fma(S a, V b, V acc) { return vfmaq_lane_f32(acc, b, a, 0); }
};
template <> struct Q<double> {
    using V = float64x2_t;
    using S = float64x1_t;
    static V load(const double* p) { return vld1q_f64(p); }
    static void store(double* p, V v) { vst1q_f64(p, v); }
    static S load_a(const double* p) { return vld1_f64(p); }  // LDR d
    static V fma(S a, V b, V acc) { return vfmaq_lane_f64(acc, b, a, 0); }
};

#endif  // HPC_HAS_NEON

template <typename In> struct QMmla;
template <typename In> struct QFmlal;
template <typename In> struct QFmlaF16;

#if HPC_HAS_NEON_LOWP

// BFMMLA and SMMLA are small matrix multiplies, not vector FMAs. Each takes
// a 2 × G block of A and a 2 × G block of Bᵀ (G = 4 bf16 or 8 INT8, one
// 128-bit register each) and adds their 2 × 2 product into a 2 × 2 tile of
// C held in one register:
//
//   C[2×2] += A[2×G] · B[G×2]     BFMMLA: 32 flops   SMMLA: 64 integer ops
//
// So one accumulator is 2 rows × 2 columns of C, an A operand is a row pair
// and a B operand a column pair: the outer product of the FP32 kernels at
// 2 × 2-tile granularity. Both operands are packed so that each register is
// one load: per k group, 8 bytes of each row of A (G elements along k) and
// 8 bytes of each column of B (G rows of B transposed). Tile edges go
// through a buffer. R row pairs × V column pairs: R·V accumulators, R
// registers of A and V of B in 32.
namespace neon_detail {

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

/// What the BFMMLA and SMMLA structs share: one 8-byte word per k group of
/// a row of A or a column of B, 2 × 2 accumulators.
struct MmlaBase {
    using Word = std::uint64_t;
    using AReg = uint64x2_t;  // two words: a row pair of A
    using BReg = uint64x2_t;  // a column pair of B
    struct Edge {};
    static constexpr bool kPackA = true, kMaskedEdges = false, kBPlus128 = false;
    static constexpr int kRowsPerAcc = 2, kColsPerAcc = 2;
    static constexpr Word kFlip      = 0;
    template <int R, int V>
    static constexpr bool kTileFits = R * V + R + V <= 32;
    static constexpr std::size_t a_offset(int r) { return 2 * std::size_t(r); }
    static constexpr std::size_t b_offset(int v) { return 2 * std::size_t(v); }
    static AReg load_a(const Word* p) { return vld1q_u64(p); }
    static BReg load_b(const Word* p) { return vld1q_u64(p); }
    static Edge edge(std::size_t) { return {}; }
};

}  // namespace neon_detail

template <> struct QMmla<bf16> : neon_detail::MmlaBase {
    using In  = bf16;
    using Out = float;
    using Acc = float32x4_t;
    static constexpr std::size_t G = 4;
    static Acc madd(Acc acc, AReg a, BReg b, int, int) {
        return vbfmmlaq_f32(acc, vreinterpretq_bf16_u64(a), vreinterpretq_bf16_u64(b));
    }
    static Acc init(std::int32_t) { return vdupq_n_f32(0.f); }
    static Acc load_c(const float* c, std::size_t ldc, Edge) { return vcombine_f32(vld1_f32(c), vld1_f32(c + ldc)); }
    static void store_c(float* c, std::size_t ldc, Acc v, Edge) {
        vst1_f32(c, vget_low_f32(v));
        vst1_f32(c + ldc, vget_high_f32(v));
    }
    /// 4 rows × 8 bf16 columns → 8 words of 4, by zips (packing a full k group).
    static std::size_t pack_b_fast(const bf16* row, std::size_t ldb, Word* out, std::size_t n) {
        std::size_t c = 0;
        for (; c + 8 <= n; c += 8) {
            const auto* p       = reinterpret_cast<const std::uint16_t*>(row + c);
            const uint16x8_t r0 = vld1q_u16(p), r1 = vld1q_u16(p + ldb);
            const uint16x8_t r2 = vld1q_u16(p + 2 * ldb), r3 = vld1q_u16(p + 3 * ldb);
            const uint32x4_t lo01 = vreinterpretq_u32_u16(vzip1q_u16(r0, r1));
            const uint32x4_t hi01 = vreinterpretq_u32_u16(vzip2q_u16(r0, r1));
            const uint32x4_t lo23 = vreinterpretq_u32_u16(vzip1q_u16(r2, r3));
            const uint32x4_t hi23 = vreinterpretq_u32_u16(vzip2q_u16(r2, r3));
            auto* o = reinterpret_cast<std::uint32_t*>(out + c);
            vst1q_u32(o, vzip1q_u32(lo01, lo23));
            vst1q_u32(o + 4, vzip2q_u32(lo01, lo23));
            vst1q_u32(o + 8, vzip1q_u32(hi01, hi23));
            vst1q_u32(o + 12, vzip2q_u32(hi01, hi23));
        }
        return c;
    }
};

template <> struct QMmla<std::int8_t> : neon_detail::MmlaBase {
    using In  = std::int8_t;
    using Out = std::int32_t;
    using Acc = int32x4_t;
    static constexpr std::size_t G = 8;
    static Acc madd(Acc acc, AReg a, BReg b, int, int) {
        return vmmlaq_s32(acc, vreinterpretq_s8_u64(a), vreinterpretq_s8_u64(b));
    }
    static Acc init(std::int32_t) { return vdupq_n_s32(0); }
    static Acc load_c(const std::int32_t* c, std::size_t ldc, Edge) {
        return vcombine_s32(vld1_s32(c), vld1_s32(c + ldc));
    }
    static void store_c(std::int32_t* c, std::size_t ldc, Acc v, Edge) {
        vst1_s32(c, vget_low_s32(v));
        vst1_s32(c + ldc, vget_high_s32(v));
    }
    /// 8 rows × 8 INT8 columns → 8 words of 8, by a byte transpose.
    static std::size_t pack_b_fast(const std::int8_t* row, std::size_t ldb, Word* out, std::size_t n) {
        std::size_t c = 0;
        for (; c + 8 <= n; c += 8) {
            uint8x8_t r[8];
            for (std::size_t e = 0; e < 8; ++e)
                r[e] = vld1_u8(reinterpret_cast<const std::uint8_t*>(row + c + e * ldb));
            neon_detail::transpose8x8_u8(r);
            for (std::size_t j = 0; j < 8; ++j)
                vst1_u8(reinterpret_cast<std::uint8_t*>(out + c + j), r[j]);
        }
        return c;
    }
};

// FMLAL is the FP32 kernel's FMA with fp16 inputs widened to FP32 (4 lanes),
// so it does the FP32 kernel's work per instruction and saves only memory.
// FMLA .8h computes and accumulates in fp16 (8 lanes): twice the work per
// instruction, at fp16's precision and range; the C tile is converted to
// FP32 on load and store. Both multiply by element: A is packed 4 rows per
// k (one D register, G = 1), and row r of the tile uses lane r, so R is 4.
// Tile edges go through a buffer.
template <> struct QFmlal<fp16> {
    using In   = fp16;
    using Out  = float;
    using Word = fp16;
    using Acc  = float32x4_t;
    using AReg = float16x4_t;  // A(i0 … i0+3, k)
    using BReg = float16x8_t;  // one B register feeds two accumulators (low, high halves)
    struct Edge {};
    static constexpr std::size_t G = 1;
    static constexpr bool kPackA = true, kMaskedEdges = false, kBPlus128 = false;
    static constexpr int kRowsPerAcc = 1, kColsPerAcc = 4;
    static constexpr Word kFlip      = {};
    template <int R, int V>
    static constexpr bool kTileFits = R == 4 && V % 2 == 0 && R * V + V / 2 + 1 <= 32;
    static constexpr std::size_t a_offset(int) { return 0; }
    static constexpr std::size_t b_offset(int v) { return 8 * std::size_t(v / 2); }
    static AReg load_a(const fp16* p) { return vld1_f16(reinterpret_cast<const float16_t*>(p)); }
    static BReg load_b(const fp16* p) { return vld1q_f16(reinterpret_cast<const float16_t*>(p)); }
    // The lane is an instruction immediate; r and v are constants once the
    // micro-kernel's loops are unrolled, so the switch folds away.
    static Acc madd(Acc acc, AReg a, BReg b, int r, int v) {
        if (v % 2 == 0)
            switch (r) {
                case 0: return vfmlalq_lane_low_f16(acc, b, a, 0);
                case 1: return vfmlalq_lane_low_f16(acc, b, a, 1);
                case 2: return vfmlalq_lane_low_f16(acc, b, a, 2);
                default: return vfmlalq_lane_low_f16(acc, b, a, 3);
            }
        switch (r) {
            case 0: return vfmlalq_lane_high_f16(acc, b, a, 0);
            case 1: return vfmlalq_lane_high_f16(acc, b, a, 1);
            case 2: return vfmlalq_lane_high_f16(acc, b, a, 2);
            default: return vfmlalq_lane_high_f16(acc, b, a, 3);
        }
    }
    static Edge edge(std::size_t) { return {}; }
    static Acc init(std::int32_t) { return vdupq_n_f32(0.f); }
    static Acc load_c(const float* c, std::size_t, Edge) { return vld1q_f32(c); }
    static void store_c(float* c, std::size_t, Acc v, Edge) { vst1q_f32(c, v); }
    static std::size_t pack_b_fast(const fp16*, std::size_t, Word*, std::size_t) { return 0; }
};

template <> struct QFmlaF16<fp16> {
    using In   = fp16;
    using Out  = float;
    using Word = fp16;
    using Acc  = float16x8_t;
    using AReg = float16x4_t;
    using BReg = float16x8_t;
    struct Edge {};
    static constexpr std::size_t G = 1;
    static constexpr bool kPackA = true, kMaskedEdges = false, kBPlus128 = false;
    static constexpr int kRowsPerAcc = 1, kColsPerAcc = 8;
    static constexpr Word kFlip      = {};
    template <int R, int V>
    static constexpr bool kTileFits = R == 4 && R * V + V + 1 <= 32;
    static constexpr std::size_t a_offset(int) { return 0; }
    static constexpr std::size_t b_offset(int v) { return 8 * std::size_t(v); }
    static AReg load_a(const fp16* p) { return vld1_f16(reinterpret_cast<const float16_t*>(p)); }
    static BReg load_b(const fp16* p) { return vld1q_f16(reinterpret_cast<const float16_t*>(p)); }
    static Acc madd(Acc acc, AReg a, BReg b, int r, int) {
        switch (r) {  // the lane is an immediate; folds once the loops are unrolled
            case 0: return vfmaq_lane_f16(acc, b, a, 0);
            case 1: return vfmaq_lane_f16(acc, b, a, 1);
            case 2: return vfmaq_lane_f16(acc, b, a, 2);
            default: return vfmaq_lane_f16(acc, b, a, 3);
        }
    }
    static Edge edge(std::size_t) { return {}; }
    static Acc init(std::int32_t) { return vdupq_n_f16(0); }
    static Acc load_c(const float* c, std::size_t, Edge) {
        return vcombine_f16(vcvt_f16_f32(vld1q_f32(c)), vcvt_f16_f32(vld1q_f32(c + 4)));
    }
    static void store_c(float* c, std::size_t, Acc v, Edge) {
        vst1q_f32(c, vcvt_f32_f16(vget_low_f16(v)));
        vst1q_f32(c + 4, vcvt_high_f32_f16(v));
    }
    static std::size_t pack_b_fast(const fp16*, std::size_t, Word*, std::size_t) { return 0; }
};

#endif  // HPC_HAS_NEON_LOWP

}  // namespace hpc::gemm
