#pragma once

/**
 * @file avx512_lowp.hpp
 * @brief Low-precision GEMM with AVX-512 dot-product instructions (Zen 5).
 *
 *   gemm_avx512_bf16   C (FP32)  = A (bf16) · B (bf16)   VDPBF16PS (AVX512_BF16)
 *   gemm_avx512_i8     C (INT32) = A (int8) · B (int8)   VPDPBUSD  (AVX512_VNNI)
 *
 * Both instructions add, into each 32-bit lane, a dot product of G
 * consecutive elements (G = 2 bf16, 4 INT8): one instruction does G times
 * the FP32 FMA's work on 16 lanes. The micro-kernel is an 8 × 2-ZMM tile
 * (16 accumulators, 8 rows × 32 columns of C); per k group it loads 2
 * vectors of B and broadcasts one 32-bit word of A per row.
 *
 * Only B is packed, because the instruction asks for it: a lane needs G
 * consecutive k of one column of B, so B is packed with G rows interleaved
 * (word [kg][c] = B(G·kg … G·kg+G−1, c)), once per kc × nc block, and the
 * block serves every row of A. A broadcast needs G consecutive k of one row
 * of A, which row-major A already has, so A is read in place.
 *
 * INT8 signedness: VPDPBUSD multiplies unsigned bytes by signed bytes, and
 * Zen 5 has no signed × signed form. B is packed as b + 128 (its sign bit
 * flipped, as uint8) and A stays signed; the extra 128 · Σ_k A(i, k) is
 * subtracted by starting row i's accumulators at minus that. Exact in INT32.
 *
 * Edges: a tile narrower than 32 columns uses masked loads and stores of C;
 * rows past the edge repeat the last row and are not stored; a last k
 * group shorter than G is padded with zeros.
 *
 * Blocking (elements): Mc = 128, Kc = 512 bf16 / 1024 INT8 (the same
 * bytes), Nc = 512; a 32-column B strip (32 KiB) stays in the 48 KiB L1.
 */

#include "gemm/lowp_reference.hpp"
#include "hpc/isa.hpp"
#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"

#if defined(__AVX512F__) && defined(__AVX512BF16__) && defined(__AVX512VNNI__)
    #define HPC_HAS_AVX512_LOWP 1
    #include <immintrin.h>
#else
    #define HPC_HAS_AVX512_LOWP 0
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace hpc::gemm {

#if !HPC_HAS_AVX512_LOWP

void gemm_avx512_bf16(const Matrix<bf16>&, const Matrix<bf16>&, Matrix<float>&) = delete;  // needs AVX512_BF16
void gemm_avx512_i8(const Matrix<std::int8_t>&, const Matrix<std::int8_t>&, Matrix<std::int32_t>&) = delete;  // needs AVX512_VNNI

#else

namespace avx512_lowp_detail {

inline constexpr std::size_t kMc = 128, kNc = 512, kMr = 8, kNr = 32;

template <typename In>
inline constexpr std::size_t kG = 4 / sizeof(In);  // 2 bf16, 4 INT8 per 32-bit word
template <typename In>
inline constexpr std::size_t kKc = 1024 / sizeof(In);  // 512 bf16, 1024 INT8

inline std::size_t round_up(std::size_t x, std::size_t m) { return (x + m - 1) / m * m; }

/// What differs between the two formats: the accumulator vector, the
/// dot-product instruction and how C is loaded and stored (masked, so a
/// tile narrower than 32 columns needs no special case).
template <typename In> struct Dot;
template <> struct Dot<bf16> {
    using V = __m512;
    static constexpr std::uint32_t kFlip = 0;  // B packed as is
    static V load(const float* p, __mmask16 m) { return _mm512_maskz_loadu_ps(m, p); }
    static void store(float* p, V v, __mmask16 m) { _mm512_mask_storeu_ps(p, m, v); }
    static V set1(std::int32_t) { return _mm512_setzero_ps(); }
    static V dot(V acc, __m512i a, __m512i b) { return _mm512_dpbf16_ps(acc, (__m512bh)a, (__m512bh)b); }
};
template <> struct Dot<std::int8_t> {
    using V = __m512i;
    static constexpr std::uint32_t kFlip = 0x80808080u;  // B packed as b + 128 (unsigned)
    static V load(const std::int32_t* p, __mmask16 m) { return _mm512_maskz_loadu_epi32(m, p); }
    static void store(std::int32_t* p, V v, __mmask16 m) { _mm512_mask_storeu_epi32(p, m, v); }
    static V set1(std::int32_t x) { return _mm512_set1_epi32(x); }
    // VPDPBUSD multiplies unsigned bytes (B + 128) by signed bytes (A). With
    // the intrinsic, GCC 13 copies every accumulator to another register
    // around each VPDPBUSD (30 extra moves per 16 dot products), and the
    // loop then runs out of front-end bandwidth: 935 instead of 1,195 GOP/s
    // on Zen 5. The one-instruction asm keeps each accumulator in place.
    static V dot(V acc, __m512i a, __m512i b) {
#if defined(__GNUC__)
        asm("vpdpbusd %2, %1, %0" : "+v"(acc) : "v"(b), "v"(a));
        return acc;
#else
        return _mm512_dpbusd_epi32(acc, b, a);
#endif
    }
};

/// k groups [0, kw) of an 8 × 32 tile: row r of A starts at a[r] (G
/// consecutive elements per group), B at b (packed, 32 words per group).
template <typename In, typename V>
[[gnu::always_inline]] inline void k_loop(V (&acc)[kMr][2], const In* const (&a)[kMr],
                                          const std::uint32_t* b, std::size_t kw) {
    constexpr std::size_t G = kG<In>;
    for (std::size_t w = 0; w < kw; ++w) {
        const __m512i b0 = _mm512_loadu_si512(b + w * kNr);
        const __m512i b1 = _mm512_loadu_si512(b + w * kNr + 16);
#pragma GCC unroll 8
        for (std::size_t r = 0; r < kMr; ++r) {
            std::uint32_t word;
            std::memcpy(&word, a[r] + w * G, 4);
            const __m512i av = _mm512_set1_epi32(static_cast<int>(word));
            acc[r][0]        = Dot<In>::dot(acc[r][0], av, b0);
            acc[r][1]        = Dot<In>::dot(acc[r][1], av, b1);
        }
    }
}

/// Pack kc × nc of B into 32-column strips of k-group words, [w·32 + c]:
/// word w of column c holds B(G·w … G·w+G−1, c), the G values one lane of
/// the dot product multiplies. Missing rows and columns are zero (before
/// the INT8 offset, which their zero A partner cancels).
template <typename In>
void pack_b(const In* B, std::size_t ldb, std::size_t kc, std::size_t nc, std::uint32_t* dst) {
    constexpr std::size_t G = kG<In>;
    const std::size_t kw    = (kc + G - 1) / G;
    for (std::size_t j0 = 0; j0 < nc; j0 += kNr) {
        const std::size_t cols = std::min(kNr, nc - j0);
        for (std::size_t w = 0; w < kw; ++w) {
            std::uint32_t* out = dst + j0 * kw + w * kNr;
            const In* row      = B + w * G * ldb + j0;
            std::size_t c      = 0;
            if constexpr (G == 2) {  // bf16: interleave two rows, 16 columns per step
                if (cols == kNr && (w + 1) * G <= kc)
                    for (; c < kNr; c += 16) {
                        const __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + c));
                        const __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + ldb + c));
                        const __m256i lo = _mm256_unpacklo_epi16(r0, r1);  // columns 0-3, 8-11
                        const __m256i hi = _mm256_unpackhi_epi16(r0, r1);  // columns 4-7, 12-15
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + c), _mm256_permute2x128_si256(lo, hi, 0x20));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + c + 8), _mm256_permute2x128_si256(lo, hi, 0x31));
                    }
            }
            for (; c < kNr; ++c) {
                In g[4] = {};
                for (std::size_t e = 0; c < cols && e < G && w * G + e < kc; ++e)
                    g[e] = row[e * ldb + c];
                std::uint32_t word;
                std::memcpy(&word, g, 4);
                out[c] = word ^ Dot<In>::kFlip;
            }
        }
    }
}

/// C(rows × cols) (+)= A(rows × kc, in place) · B strip (packed), for one
/// 8 × 32 tile. `init` is the start value of every accumulator of row r
/// when `accumulate` is false (INT8: −128 · Σ_k A(r, k)), else C is loaded.
template <typename In>
void micro(const In* a, std::size_t lda, const std::uint32_t* __restrict b, lowp_out_t<In>* c,
           std::size_t ldc, std::size_t kc, std::size_t rows, std::size_t cols, bool accumulate,
           const std::int32_t* init) {
    using Ops                = Dot<In>;
    using V                  = typename Ops::V;
    constexpr std::size_t G  = kG<In>;
    const std::size_t full   = kc / G;  // whole k groups; a partial one follows if kc % G
    const __mmask16 m0       = cols >= 16 ? 0xFFFF : static_cast<__mmask16>((1u << cols) - 1);
    const __mmask16 m1       = cols >= 32 ? 0xFFFF : cols <= 16 ? 0 : static_cast<__mmask16>((1u << (cols - 16)) - 1);

    // Rows past the edge repeat the last row; they are computed, not stored.
    const In* ar[kMr];
    V acc[kMr][2];
    for (std::size_t r = 0; r < kMr; ++r) {
        const std::size_t rr = std::min(r, rows - 1);
        ar[r]                = a + rr * lda;
        if (accumulate) {
            acc[r][0] = Ops::load(c + rr * ldc, m0);
            acc[r][1] = Ops::load(c + rr * ldc + 16, m1);
        } else {
            acc[r][0] = acc[r][1] = Ops::set1(init ? init[rr] : 0);
        }
    }
    k_loop<In>(acc, ar, b, full);
    if (full * G < kc) {  // a last k group shorter than G, padded with zeros
        In tail[kMr][4] = {};
        const In* tp[kMr];
        for (std::size_t r = 0; r < kMr; ++r) {
            std::memcpy(tail[r], ar[r] + full * G, (kc - full * G) * sizeof(In));
            tp[r] = tail[r];
        }
        k_loop<In>(acc, tp, b + full * kNr, 1);
    }
    for (std::size_t r = 0; r < rows; ++r) {
        Ops::store(c + r * ldc, acc[r][0], m0);
        Ops::store(c + r * ldc + 16, acc[r][1], m1);
    }
}

/// GotoBLAS order: each kc × nc block of B is packed once and serves every
/// row of A, which is read in place. For INT8 the +128 offset on B adds
/// 128 · Σ_k A(i, k) to row i; the accumulators start at minus that.
template <typename In>
void driver(const Matrix<In>& A, const Matrix<In>& B, Matrix<lowp_out_t<In>>& C) {
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;
    constexpr std::size_t G = kG<In>, Kc = kKc<In>;
    std::vector<std::int32_t> offset;  // −128 · row sums of A (INT8 only)
    if constexpr (std::is_same_v<In, std::int8_t>) {
        offset.resize(M);
        for (std::size_t i = 0; i < M; ++i) {
            std::int32_t s = 0;
            for (std::size_t k = 0; k < K; ++k)
                s += A(i, k);
            offset[i] = -128 * s;
        }
    }
    std::vector<std::uint32_t> b_pack(((std::min(Kc, K) + G - 1) / G) * round_up(std::min(kNc, N), kNr));
    for (std::size_t jc = 0; jc < N; jc += kNc) {
        const std::size_t nc = std::min(kNc, N - jc);
        for (std::size_t pc = 0; pc < K; pc += Kc) {
            const std::size_t kc = std::min(Kc, K - pc), kw = (kc + G - 1) / G;
            pack_b<In>(B.data() + pc * N + jc, N, kc, nc, b_pack.data());
            for (std::size_t ic = 0; ic < M; ic += kMc)
                for (std::size_t jr = 0; jr < nc; jr += kNr)
                    for (std::size_t ir = ic; ir < std::min(ic + kMc, M); ir += kMr)
                        micro<In>(A.data() + ir * K + pc, K, b_pack.data() + jr * kw,
                                  C.data() + ir * N + jc + jr, N, kc, std::min(kMr, M - ir),
                                  std::min(kNr, nc - jr), pc > 0,
                                  offset.empty() ? nullptr : offset.data() + ir);
        }
    }
}

}  // namespace avx512_lowp_detail

/// C (FP32) = A · B, bf16 inputs, VDPBF16PS. See the file header.
inline void gemm_avx512_bf16(const Matrix<bf16>& A, const Matrix<bf16>& B, Matrix<float>& C) {
    avx512_lowp_detail::driver(A, B, C);
}

/// C (INT32) = A · B, INT8 inputs, VPDPBUSD with a +128 offset. See the file header.
inline void gemm_avx512_i8(const Matrix<std::int8_t>& A, const Matrix<std::int8_t>& B,
                           Matrix<std::int32_t>& C) {
    avx512_lowp_detail::driver(A, B, C);
}

#endif  // HPC_HAS_AVX512_LOWP

}  // namespace hpc::gemm
