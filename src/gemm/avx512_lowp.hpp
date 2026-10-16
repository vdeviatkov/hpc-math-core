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
 * the FP32 FMA's work on 16 lanes. The micro-kernel is the FP32 kernel's
 * 8 × 2-ZMM tile (16 accumulators, 8 rows × 32 columns of C, avx512.hpp);
 * per k group it loads 2 vectors of B and broadcasts 8 words of A.
 *
 * Packing follows from the instruction: a lane needs G consecutive k of one
 * column of B, so B is packed with G rows interleaved (word [kg][c] =
 * B(G·kg … G·kg+G−1, c)); a broadcast needs G consecutive k of one row of
 * A, which row-major A already has, so A is packed only to collect the 8
 * rows of a strip and pad K to a whole group.
 *
 * INT8 signedness: VPDPBUSD multiplies unsigned bytes (first operand) by
 * signed bytes. A is packed with its sign bit flipped (a + 128, as uint8),
 * and the extra 128 · Σ_k B(k, j) is subtracted from every C(i, j); the
 * column sums are computed while packing B. The result is exact in INT32.
 * (Zen 5 has no AVX-VNNI-INT8 signed × signed form.)
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

/// Pack mc × kc of A into 8-row strips of k-group words: [w·8 + r]. INT8
/// words have their sign bits flipped (a + 128 as uint8).
template <typename In>
void pack_a(const In* A, std::size_t lda, std::size_t mc, std::size_t kc, std::uint32_t* dst) {
    constexpr std::size_t G = kG<In>;
    const std::size_t kw = (kc + G - 1) / G, full = kc / G;
    constexpr std::uint32_t flip = std::is_same_v<In, std::int8_t> ? 0x80808080u : 0u;
    for (std::size_t i0 = 0; i0 < mc; i0 += kMr) {
        std::uint32_t* strip = dst + i0 * kw;
        for (std::size_t r = 0; r < kMr; ++r) {
            if (i0 + r >= mc) {
                for (std::size_t w = 0; w < kw; ++w)
                    strip[w * kMr + r] = flip;  // a zero row: 0 + 128 for INT8
                continue;
            }
            const In* src = A + (i0 + r) * lda;
            for (std::size_t w = 0; w < full; ++w) {
                std::uint32_t word;
                std::memcpy(&word, src + w * G, 4);
                strip[w * kMr + r] = word ^ flip;
            }
            if (full < kw) {
                In g[4] = {};
                std::memcpy(g, src + full * G, (kc - full * G) * sizeof(In));
                std::uint32_t word;
                std::memcpy(&word, g, 4);
                strip[full * kMr + r] = word ^ flip;
            }
        }
    }
}

/// Pack kc × nc of B into 32-column strips of k-group words: [w·32 + c].
/// For INT8, colsum[j] = Σ_k B(k, j) over this k block.
template <typename In>
void pack_b(const In* B, std::size_t ldb, std::size_t kc, std::size_t nc, std::uint32_t* dst,
            std::int32_t* colsum) {
    constexpr std::size_t G = kG<In>;
    const std::size_t kw    = (kc + G - 1) / G;
    for (std::size_t j0 = 0; j0 < nc; j0 += kNr) {
        const std::size_t cols = std::min(kNr, nc - j0);
        std::uint32_t* strip   = dst + j0 * kw;
        for (std::size_t w = 0; w < kw; ++w) {
            std::uint32_t* out = strip + w * kNr;
            const In* row      = B + w * G * ldb + j0;
            std::size_t c      = 0;
            if ((w + 1) * G <= kc) {
                if constexpr (G == 2) {  // interleave two rows of 16-bit: 16 columns per step
                    for (; c + 16 <= cols; c += 16) {
                        const __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + c));
                        const __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + ldb + c));
                        const __m256i lo = _mm256_unpacklo_epi16(r0, r1);  // cols 0-3, 8-11
                        const __m256i hi = _mm256_unpackhi_epi16(r0, r1);  // cols 4-7, 12-15
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + c),
                                            _mm256_permute2x128_si256(lo, hi, 0x20));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + c + 8),
                                            _mm256_permute2x128_si256(lo, hi, 0x31));
                    }
                }
            }
            for (; c < cols; ++c) {
                In g[4] = {};
                for (std::size_t e = 0; e < G && w * G + e < kc; ++e)
                    g[e] = row[e * ldb + c];
                std::memcpy(&out[c], g, 4);
            }
            std::fill(out + cols, out + kNr, 0u);
        }
    }
    if constexpr (std::is_same_v<In, std::int8_t>) {
        std::fill(colsum, colsum + round_up(nc, kNr), 0);
        for (std::size_t k = 0; k < kc; ++k)
            for (std::size_t j = 0; j < nc; ++j)
                colsum[j] += B[k * ldb + j];
    }
}

/// Accumulator vector per input type (a traits struct, not std::conditional_t,
/// which drops the vector types' attributes).
template <typename In> struct AccVec;
template <> struct AccVec<bf16> { using V = __m512; };
template <> struct AccVec<std::int8_t> { using V = __m512i; };

/// 8 × 32 tile: C (+)= A strip · B strip over kw k groups.
template <typename In>
void micro(const std::uint32_t* __restrict a, const std::uint32_t* __restrict b,
           lowp_out_t<In>* __restrict c, std::size_t ldc, std::size_t kw, std::size_t rows,
           std::size_t cols, bool accumulate, const std::int32_t* colsum) {
    using Out          = lowp_out_t<In>;
    constexpr bool kI8 = std::is_same_v<In, std::int8_t>;
    using V            = typename AccVec<In>::V;
    const bool full    = rows == kMr && cols == kNr;
    alignas(64) Out tmp[kMr][kNr];
    Out* base            = full ? c : &tmp[0][0];
    const std::size_t ld = full ? ldc : kNr;
    if (!full && accumulate)
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < cols; ++j)
                tmp[r][j] = c[r * ldc + j];

    V acc[kMr][2];
    if constexpr (kI8) {
        // Subtract 128 · colsum: A was offset by +128.
        const __m512i corr0 = _mm512_slli_epi32(_mm512_loadu_si512(colsum), 7);
        const __m512i corr1 = _mm512_slli_epi32(_mm512_loadu_si512(colsum + 16), 7);
        for (std::size_t r = 0; r < kMr; ++r) {
            const __m512i c0 = accumulate ? _mm512_loadu_si512(base + r * ld) : _mm512_setzero_si512();
            const __m512i c1 = accumulate ? _mm512_loadu_si512(base + r * ld + 16) : _mm512_setzero_si512();
            acc[r][0] = _mm512_sub_epi32(c0, corr0);
            acc[r][1] = _mm512_sub_epi32(c1, corr1);
        }
    } else {
        for (std::size_t r = 0; r < kMr; ++r) {
            acc[r][0] = accumulate ? _mm512_loadu_ps(base + r * ld) : _mm512_setzero_ps();
            acc[r][1] = accumulate ? _mm512_loadu_ps(base + r * ld + 16) : _mm512_setzero_ps();
        }
    }
    for (std::size_t w = 0; w < kw; ++w) {
        const __m512i b0 = _mm512_loadu_si512(b + w * kNr);
        const __m512i b1 = _mm512_loadu_si512(b + w * kNr + 16);
#pragma GCC unroll 8
        for (std::size_t r = 0; r < kMr; ++r) {
            const __m512i ar = _mm512_set1_epi32(static_cast<int>(a[w * kMr + r]));
            if constexpr (kI8) {
                acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], ar, b0);
                acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], ar, b1);
            } else {
                acc[r][0] = _mm512_dpbf16_ps(acc[r][0], (__m512bh)ar, (__m512bh)b0);
                acc[r][1] = _mm512_dpbf16_ps(acc[r][1], (__m512bh)ar, (__m512bh)b1);
            }
        }
    }
    for (std::size_t r = 0; r < kMr; ++r) {
        if constexpr (kI8) {
            _mm512_storeu_si512(base + r * ld, acc[r][0]);
            _mm512_storeu_si512(base + r * ld + 16, acc[r][1]);
        } else {
            _mm512_storeu_ps(base + r * ld, acc[r][0]);
            _mm512_storeu_ps(base + r * ld + 16, acc[r][1]);
        }
    }
    if (!full)
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < cols; ++j)
                c[r * ldc + j] = tmp[r][j];
}

template <typename In>
void driver(const Matrix<In>& A, const Matrix<In>& B, Matrix<lowp_out_t<In>>& C) {
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;
    constexpr std::size_t G = kG<In>, Kc = kKc<In>;
    const std::size_t kw_max = (std::min(Kc, K) + G - 1) / G;
    Matrix<std::uint32_t> a_pack(round_up(std::min(kMc, M), kMr), kw_max);
    Matrix<std::uint32_t> b_pack(kw_max, round_up(std::min(kNc, N), kNr));
    std::vector<std::int32_t> colsum(round_up(std::min(kNc, N), kNr) + kNr, 0);
    for (std::size_t jc = 0; jc < N; jc += kNc) {
        const std::size_t nc = std::min(kNc, N - jc);
        for (std::size_t pc = 0; pc < K; pc += Kc) {
            const std::size_t kc = std::min(Kc, K - pc), kw = (kc + G - 1) / G;
            pack_b<In>(B.data() + pc * N + jc, N, kc, nc, b_pack.data(), colsum.data());
            for (std::size_t ic = 0; ic < M; ic += kMc) {
                const std::size_t mc = std::min(kMc, M - ic);
                pack_a<In>(A.data() + ic * K + pc, K, mc, kc, a_pack.data());
                for (std::size_t jr = 0; jr < nc; jr += kNr)
                    for (std::size_t ir = 0; ir < mc; ir += kMr)
                        micro<In>(a_pack.data() + ir * kw, b_pack.data() + jr * kw,
                                  C.data() + (ic + ir) * N + jc + jr, N, kw,
                                  std::min(kMr, mc - ir), std::min(kNr, nc - jr), pc > 0,
                                  colsum.data() + jr);
            }
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
