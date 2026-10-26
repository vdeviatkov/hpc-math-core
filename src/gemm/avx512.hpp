#pragma once

/**
 * @file avx512.hpp
 * @brief AVX-512 primitives for the generic SIMD kernels (simd.hpp).
 *
 *   gemm_simd_reordered<Zmm>(A, B, C);
 *   gemm_simd_blocked<Zmm, R, V, TileM, TileK, TileN[, PfDist]>(A, B, C);
 *   gemm_simd_packed<Zmm, R, V, Mc, Kc, Nc[, PackA]>(A, B, C);
 *   gemm_simd_lowp<ZmmDot, R, V, Mc, Kc, Nc>(A, B, C);   // bf16 or INT8 A, B
 *
 * 32 × 512-bit ZMM registers (16 f32 / 8 f64). With 2 FMA ports:
 * 2 × 16 lanes × 2 FLOP = 64 FLOP/cycle f32, 32 f64.
 *
 * Available on Intel Skylake-SP/X and later server/workstation parts
 * (Cascade Lake, Ice Lake, Sapphire Rapids, Rocket Lake) and AMD Zen 4+.
 * Not on Apple Silicon, Alder Lake/Raptor Lake (disabled by Intel), or AMD
 * before Zen 4.
 *
 * HPC_HAS_AVX512 (hpc/isa.hpp) follows __AVX512F__. -march=native enables
 * AVX-512 on a capable CPU; HPC_ENABLE_AVX512=ON forces the flags
 * otherwise. Zmm<T> is declared on every target and defined only where
 * HPC_HAS_AVX512 is 1; ZmmDot<In> needs AVX512_BF16 and AVX512_VNNI too
 * (HPC_HAS_AVX512_LOWP).
 */

#include "hpc/isa.hpp"
#include "hpc/lowp.hpp"

#if HPC_HAS_AVX512
    #include <immintrin.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace hpc::gemm {

template <typename T> struct Zmm;

#if HPC_HAS_AVX512

// load_a broadcasts A(i, k) into all lanes (vbroadcastss from memory).
//
// The register tile matters more here than on AVX2: with 4 rows × 2 ZMM,
// GCC 13 folds the B loads into the FMAs as memory operands and re-loads B
// for every FMA, so the kernel runs into the load ports at 60% of peak.
// With 6 or more rows it loads B once per k step (docs/kernels/avx512.md).
template <> struct Zmm<float> {
    using V = __m512;
    static V load(const float* p) { return _mm512_loadu_ps(p); }
    static void store(float* p, V v) { _mm512_storeu_ps(p, v); }
    static V load_a(const float* p) { return _mm512_set1_ps(*p); }
    static V fma(V x, V y, V acc) { return _mm512_fmadd_ps(x, y, acc); }
};
template <> struct Zmm<double> {
    using V = __m512d;
    static V load(const double* p) { return _mm512_loadu_pd(p); }
    static void store(double* p, V v) { _mm512_storeu_pd(p, v); }
    static V load_a(const double* p) { return _mm512_set1_pd(*p); }
    static V fma(V x, V y, V acc) { return _mm512_fmadd_pd(x, y, acc); }
};

#endif  // HPC_HAS_AVX512

template <typename In> struct ZmmDot;

#if HPC_HAS_AVX512_LOWP

// Low precision with the dot-product instructions: VDPBF16PS (bf16 → FP32)
// and VPDPBUSD (INT8 → INT32) add, into each 32-bit lane, a dot product of
// G consecutive k (G = 2 bf16, 4 INT8): G times the FP32 FMA's work on 16
// lanes. One accumulator is 1 row × 16 columns of C; per k group the
// micro-kernel loads V vectors of B and broadcasts one 32-bit word of A per
// row. A is read in place, because a broadcast needs G consecutive k of one
// row, which row-major A already has. Tile edges use masked loads and
// stores of C. 32 ZMM registers: R·V accumulators, V vectors of B and one
// broadcast.
template <> struct ZmmDot<bf16> {
    using In   = bf16;
    using Out  = float;
    using Word = std::uint32_t;  // one k group of a column of B
    using Acc  = __m512;
    using AReg = __m512i;
    using BReg = __m512i;
    using Edge = __mmask16;
    static constexpr std::size_t G = 2;
    static constexpr bool kPackA = false, kMaskedEdges = true, kBPlus128 = false;
    static constexpr int kRowsPerAcc = 1, kColsPerAcc = 16;
    static constexpr Word kFlip      = 0;
    template <int R, int V>
    static constexpr bool kTileFits = R * V + V + 1 <= 32;
    static constexpr std::size_t a_offset(int) { return 0; }
    static constexpr std::size_t b_offset(int v) { return 16 * std::size_t(v); }
    static AReg load_a(const bf16* p) {
        std::uint32_t word;
        std::memcpy(&word, p, 4);
        return _mm512_set1_epi32(static_cast<int>(word));
    }
    static BReg load_b(const Word* p) { return _mm512_loadu_si512(p); }
    static Acc madd(Acc acc, AReg a, BReg b, int, int) { return _mm512_dpbf16_ps(acc, (__m512bh)a, (__m512bh)b); }
    static Edge edge(std::size_t cols) { return cols >= 16 ? 0xFFFF : static_cast<__mmask16>((1u << cols) - 1); }
    static Acc init(std::int32_t) { return _mm512_setzero_ps(); }
    static Acc load_c(const float* c, std::size_t, Edge m) { return _mm512_maskz_loadu_ps(m, c); }
    static void store_c(float* c, std::size_t, Acc v, Edge m) { _mm512_mask_storeu_ps(c, m, v); }
    /// Interleave two rows of B, 16 columns per step (packing a full k group).
    static std::size_t pack_b_fast(const bf16* row, std::size_t ldb, Word* out, std::size_t n) {
        std::size_t c = 0;
        for (; c + 16 <= n; c += 16) {
            const __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + c));
            const __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + ldb + c));
            const __m256i lo = _mm256_unpacklo_epi16(r0, r1);  // columns 0-3, 8-11
            const __m256i hi = _mm256_unpackhi_epi16(r0, r1);  // columns 4-7, 12-15
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + c), _mm256_permute2x128_si256(lo, hi, 0x20));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + c + 8), _mm256_permute2x128_si256(lo, hi, 0x31));
        }
        return c;
    }
};

// INT8 signedness: VPDPBUSD multiplies unsigned bytes by signed bytes, and
// Zen 5 has no signed × signed form. B is packed as b + 128 (its sign bit
// flipped, as uint8) and A stays signed; the extra 128 · Σ_k A(i, k) is
// subtracted by starting row i's accumulators at minus that (kBPlus128).
// Exact in INT32.
template <> struct ZmmDot<std::int8_t> {
    using In   = std::int8_t;
    using Out  = std::int32_t;
    using Word = std::uint32_t;
    using Acc  = __m512i;
    using AReg = __m512i;
    using BReg = __m512i;
    using Edge = __mmask16;
    static constexpr std::size_t G = 4;
    static constexpr bool kPackA = false, kMaskedEdges = true, kBPlus128 = true;
    static constexpr int kRowsPerAcc = 1, kColsPerAcc = 16;
    static constexpr Word kFlip      = 0x80808080u;  // B packed as b + 128 (unsigned)
    template <int R, int V>
    static constexpr bool kTileFits = R * V + V + 1 <= 32;
    static constexpr std::size_t a_offset(int) { return 0; }
    static constexpr std::size_t b_offset(int v) { return 16 * std::size_t(v); }
    static AReg load_a(const std::int8_t* p) {
        std::uint32_t word;
        std::memcpy(&word, p, 4);
        return _mm512_set1_epi32(static_cast<int>(word));
    }
    static BReg load_b(const Word* p) { return _mm512_loadu_si512(p); }
    // With the intrinsic, GCC 13 copies every accumulator to another register
    // around each VPDPBUSD (30 extra moves per 16 dot products), and the
    // loop then runs out of front-end bandwidth: 935 instead of 1,195 GOP/s
    // on Zen 5. The one-instruction asm keeps each accumulator in place.
    static Acc madd(Acc acc, AReg a, BReg b, int, int) {
#if defined(__GNUC__)
        asm("vpdpbusd %2, %1, %0" : "+v"(acc) : "v"(b), "v"(a));
        return acc;
#else
        return _mm512_dpbusd_epi32(acc, b, a);
#endif
    }
    static Edge edge(std::size_t cols) { return cols >= 16 ? 0xFFFF : static_cast<__mmask16>((1u << cols) - 1); }
    static Acc init(std::int32_t x) { return _mm512_set1_epi32(x); }
    static Acc load_c(const std::int32_t* c, std::size_t, Edge m) { return _mm512_maskz_loadu_epi32(m, c); }
    static void store_c(std::int32_t* c, std::size_t, Acc v, Edge m) { _mm512_mask_storeu_epi32(c, m, v); }
    static std::size_t pack_b_fast(const std::int8_t*, std::size_t, Word*, std::size_t) { return 0; }
};

#endif  // HPC_HAS_AVX512_LOWP

}  // namespace hpc::gemm
