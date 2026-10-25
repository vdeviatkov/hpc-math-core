#pragma once

/**
 * @file simd_lowp.hpp
 * @brief The low-precision SIMD GEMM, generic over the instruction.
 *
 *   gemm_simd_lowp<Ops, R, V, Mc, Kc, Nc>(A, B, C)
 *
 * C (FP32, or INT32 for INT8) = A · B with bf16, fp16 or INT8 inputs. Ops is
 * a primitive struct for one instruction shape, defined with its ISA:
 *
 *   ZmmDot<In>    VDPBF16PS, VPDPBUSD (avx512.hpp)    bf16, INT8
 *   QMmla<In>     BFMMLA, SMMLA (neon.hpp)            bf16, INT8
 *   QFmlal<In>    FMLAL, fp16 widened to FP32         fp16
 *   QFmlaF16<In>  FMLA .8h, fp16 accumulation         fp16
 *
 * In is deduced from the matrices. R × V is the register tile in
 * accumulators (rows × columns of them), Mc × Kc × Nc the cache block in
 * elements; none has a default, the values are measured (benchmarks/,
 * docs/articles/lowp-gemm.md).
 *
 * What the instructions share is a k group: G consecutive k (G = 1 for the
 * fp16 FMAs, 2–8 for the dot products and matrix multiplies) that one
 * instruction consumes per lane or per register. B is always packed by k
 * group, and A is packed the same way where the instruction needs it (Ops::
 * kPackA); the dot products broadcast G consecutive k of a row of A, which
 * row-major A already has, so they read A in place.
 *
 * The file is laid out bottom-up: packing, the micro-kernel, the kernel.
 * SME's widening outer products (sme_lowp.hpp) are not part of it: they run
 * in streaming mode and accumulate in ZA.
 */

#include "gemm/avx512.hpp"
#include "gemm/lowp_naive.hpp"
#include "gemm/neon.hpp"
#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace hpc::gemm {

// ============================================================================
// The Ops interface
// ============================================================================
//
// Formats and layout:
//   using In, Out;                     // input element; FP32 or INT32 output
//   using Word;                        // one k group: G elements of In
//   static constexpr std::size_t G;    // k per word
//   static constexpr bool kPackA;      // A packed into k-group strips, or read in place
//   static constexpr int kRowsPerAcc, kColsPerAcc;   // the C block one accumulator holds
//   static constexpr Word kFlip;       // XORed into every packed word of B
//   static constexpr bool kBPlus128;   // kFlip adds 128 to INT8 B: subtract 128 · Σ_k A(i, k)
//   template <int R, int V> static constexpr bool kTileFits;   // the tile fits the registers
//
// The micro-kernel's operands, for accumulator (r, v) of the tile:
//   using Acc, AReg, BReg;
//   static constexpr std::size_t a_offset(int r);   // packed A: row unit r's operand within a k group
//   static constexpr std::size_t b_offset(int v);   // column unit v's operand within a k group of B
//   static AReg load_a(const AWord*);  // AWord = Word if kPackA, else In
//   static BReg load_b(const Word*);
//   static Acc madd(Acc acc, AReg a, BReg b, int r, int v);   // r, v constant once unrolled
//
// C: tile edges are masked (kMaskedEdges: Edge is a mask) or the tile goes
// through a buffer (Edge is empty):
//   using Edge; static Edge edge(std::size_t cols);   // the columns present of one accumulator
//   static Acc init(std::int32_t row_start);   // INT8 with kBPlus128: the row's offset; else zero
//   static Acc load_c(const Out* c, std::size_t ldc, Edge);
//   static void store_c(Out* c, std::size_t ldc, Acc, Edge);
//
// Packing B quickly where the instruction set has a transpose or zip:
//   static std::size_t pack_b_fast(const In* row, std::size_t ldb, Word* out, std::size_t n);
//   // packs a whole k group of the first columns of n, returns how many

namespace detail {

inline std::size_t round_up(std::size_t x, std::size_t m) { return (x + m - 1) / m * m; }

// ============================================================================
// Packing
// ============================================================================

/// Pack kc × nc of B into Nr-column strips of k-group words, [w·Nr + c]:
/// word w of column c holds B(G·w … G·w+G−1, c), the G values one lane or
/// register of the instruction multiplies. Missing rows and columns are zero
/// (before kFlip, whose INT8 offset their zero A partner cancels).
template <typename O, std::size_t Nr>
void pack_b(const typename O::In* B, std::size_t ldb, std::size_t kc, std::size_t nc, typename O::Word* dst) {
    using In = typename O::In;
    using Word = typename O::Word;
    constexpr std::size_t G = O::G;
    const std::size_t kw    = (kc + G - 1) / G;
    for (std::size_t j0 = 0; j0 < nc; j0 += Nr) {
        const std::size_t cols = std::min(Nr, nc - j0);
        for (std::size_t w = 0; w < kw; ++w) {
            Word* out     = dst + j0 * kw + w * Nr;
            const In* row = B + w * G * ldb + j0;
            if constexpr (G == 1) {  // a row of the strip is a row of B
                std::memcpy(out, row, cols * sizeof(In));
                std::fill(out + cols, out + Nr, Word{});
            } else {
                std::size_t c = cols == Nr && (w + 1) * G <= kc ? O::pack_b_fast(row, ldb, out, Nr) : 0;
                for (; c < Nr; ++c) {
                    In g[G] = {};
                    for (std::size_t e = 0; c < cols && e < G && w * G + e < kc; ++e)
                        g[e] = row[e * ldb + c];
                    Word word;
                    std::memcpy(&word, g, sizeof(Word));
                    out[c] = word ^ O::kFlip;
                }
            }
        }
    }
}

/// Pack mc × kc of A into Mr-row strips of k-group words, [w·Mr + r]: word w
/// of row r holds A(i0+r, G·w … G·w+G−1). Rows past mc and k past kc are zero.
template <typename O, std::size_t Mr>
void pack_a(const typename O::In* A, std::size_t lda, std::size_t mc, std::size_t kc, typename O::Word* dst) {
    using In = typename O::In;
    using Word = typename O::Word;
    constexpr std::size_t G = O::G;
    const std::size_t kw = (kc + G - 1) / G, full = kc / G;
    for (std::size_t i0 = 0; i0 < mc; i0 += Mr) {
        const std::size_t rows = std::min(Mr, mc - i0);
        Word* strip            = dst + i0 * kw;
        for (std::size_t r = 0; r < Mr; ++r) {
            if (r >= rows) {
                for (std::size_t w = 0; w < kw; ++w)
                    strip[w * Mr + r] = Word{};
                continue;
            }
            const In* src = A + (i0 + r) * lda;
            for (std::size_t w = 0; w < full; ++w)
                std::memcpy(&strip[w * Mr + r], src + w * G, sizeof(Word));
            if (full < kw) {
                In g[G] = {};
                std::memcpy(g, src + full * G, (kc - full * G) * sizeof(In));
                std::memcpy(&strip[full * Mr + r], g, sizeof(Word));
            }
        }
    }
}

// ============================================================================
// The micro-kernel
// ============================================================================

/// k groups [0, kw) of an R × V-accumulator tile: row unit r's A operands
/// start at ar[r] and advance by a_step per group; B (packed, Nr words per
/// group) at b. Per group: V loads of B, one A operand per row unit, R·V
/// instructions: an outer product at the granularity of one accumulator.
/// The fixed-count loops must unroll completely, so the accumulators stay in
/// registers and r, v are constants in madd (an FMLA lane is an immediate).
template <typename O, int R, int V, std::size_t Nr, typename AWord>
[[gnu::always_inline]] inline void k_loop(typename O::Acc (&acc)[R][V], const AWord* const (&ar)[R],
                                          std::size_t a_step, const typename O::Word* b, std::size_t kw) {
    for (std::size_t w = 0; w < kw; ++w) {
        typename O::BReg bv[V];
#pragma GCC unroll 32
        for (int v = 0; v < V; ++v)
            bv[v] = O::load_b(b + w * Nr + O::b_offset(v));
#pragma GCC unroll 32
        for (int r = 0; r < R; ++r) {
            const typename O::AReg a = O::load_a(ar[r] + w * a_step);
#pragma GCC unroll 32
            for (int v = 0; v < V; ++v)
                acc[r][v] = O::madd(acc[r][v], a, bv[v], r, v);
        }
    }
}

/// C(rows × cols) (+)= A · B strip for one tile of R × V accumulators
/// (Mr × Nr elements of C). A is a packed strip (kPackA) or rows of A in
/// place, lda apart. `init` is the start value of row r's accumulators when
/// `accumulate` is false (INT8 with kBPlus128), else C is loaded. Kept out
/// of line: inlined into the kernel, the tile's row pointers compete with the
/// kernel's own values for registers, and the k loop reloads them (seen with
/// Clang on the AVX-512 bf16 kernel: a 56-instruction loop became 420).
template <typename O, int R, int V>
[[gnu::noinline]] void micro(const void* a_ptr, std::size_t lda, const typename O::Word* __restrict b, typename O::Out* c,
           std::size_t ldc, std::size_t kc, std::size_t rows, std::size_t cols, bool accumulate,
           const std::int32_t* init) {
    using In  = typename O::In;
    using Out = typename O::Out;
    using Acc = typename O::Acc;
    constexpr int Rpa = O::kRowsPerAcc, Cpa = O::kColsPerAcc;
    constexpr std::size_t G = O::G, Mr = R * Rpa, Nr = V * Cpa;

    typename O::Edge e[V];  // the columns of each accumulator present in this tile
#pragma GCC unroll 32
    for (int v = 0; v < V; ++v)
        e[v] = O::edge(cols > std::size_t(v * Cpa) ? cols - v * Cpa : 0);

    // Buffered edges: a partial tile is computed in tmp and copied out.
    Out tmp[O::kMaskedEdges ? 1 : Mr][O::kMaskedEdges ? 1 : Nr];
    const bool full      = O::kMaskedEdges || (rows == Mr && cols == Nr);
    Out* base            = full ? c : &tmp[0][0];
    const std::size_t ld = full ? ldc : Nr;
    if (!full && accumulate)
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < cols; ++j)
                tmp[r][j] = c[r * ldc + j];

    // Row unit r's first C row: masked edges repeat the last row (computed,
    // not stored); buffered tiles compute every row into tmp.
    const auto c_row = [rows](int r) -> std::size_t {
        return O::kMaskedEdges ? std::min(std::size_t(r * Rpa), rows - 1) : std::size_t(r * Rpa);
    };

    Acc acc[R][V];
#pragma GCC unroll 32
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 32
        for (int v = 0; v < V; ++v)
            acc[r][v] = accumulate ? O::load_c(base + c_row(r) * ld + v * Cpa, ld, e[v])
                                   : O::init(init ? init[c_row(r)] : 0);

    if constexpr (O::kPackA) {  // K is zero-padded to whole k groups in the strip
        const auto* a = static_cast<const typename O::Word*>(a_ptr);
        const typename O::Word* ar[R];
        for (int r = 0; r < R; ++r)
            ar[r] = a + O::a_offset(r);
        k_loop<O, R, V, Nr>(acc, ar, Mr, b, (kc + G - 1) / G);
    } else {
        const auto* a           = static_cast<const In*>(a_ptr);
        const std::size_t whole = kc / G;  // whole k groups; a partial one follows if kc % G
        const In* ar[R];
        for (int r = 0; r < R; ++r)
            ar[r] = a + c_row(r) * lda;
        k_loop<O, R, V, Nr>(acc, ar, G, b, whole);
        if (whole * G < kc) {  // a last k group shorter than G, padded with zeros
            In tail[R][G] = {};
            const In* tp[R];
            for (int r = 0; r < R; ++r) {
                std::memcpy(tail[r], ar[r] + whole * G, (kc - whole * G) * sizeof(In));
                tp[r] = tail[r];
            }
            k_loop<O, R, V, Nr>(acc, tp, G, b + whole * Nr, 1);
        }
    }

#pragma GCC unroll 32
    for (int r = 0; r < R; ++r) {
        if (O::kMaskedEdges && std::size_t(r * Rpa) >= rows)
            break;
#pragma GCC unroll 32
        for (int v = 0; v < V; ++v)
            O::store_c(base + r * Rpa * ld + v * Cpa, ld, acc[r][v], e[v]);
    }
    if (!full)
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < cols; ++j)
                c[r * ldc + j] = tmp[r][j];
}

}  // namespace detail

// ============================================================================
// The kernel
// ============================================================================
//
// GotoBLAS order: each kc × nc block of B is packed once and serves every
// row of A; A is packed per mc × kc block where the instruction needs it:
//
//   for jc (Nc columns of B and C)
//     for pc (Kc steps of k)
//       pack B(pc.., jc..) into Nr-column strips of k-group words
//       for ic (Mc rows of A and C)
//         [kPackA: pack A(ic.., pc..) into Mr-row strips]
//         for jr (one Nr-column strip)
//           for ir (Mr rows at a time)
//             micro: C(Mr × Nr) (+)= A(Mr × kc) · strip(kc × Nr)
//
// Mr = R · kRowsPerAcc and Nr = V · kColsPerAcc elements of C per tile. Rows
// stop at the Mc block: with Mr not dividing Mc, a tile must not run into
// the next block's rows, or they would be accumulated twice. With kBPlus128
// (INT8 B packed as b + 128), row i's accumulators start at −128 · Σ_k A(i, k).

template <template <typename> class Ops, int R, int V, std::size_t Mc, std::size_t Kc, std::size_t Nc,
          typename In>
void gemm_simd_lowp(const Matrix<In>& A, const Matrix<In>& B, Matrix<lowp_out_t<In>>& C) {
    using O = Ops<In>;
    static_assert(std::is_same_v<typename O::Out, lowp_out_t<In>>, "Ops<In>::Out must be lowp_out_t<In>");
    static_assert(R >= 1 && V >= 1 && O::template kTileFits<R, V>,
                  "this R × V tile does not fit the instruction's registers or lanes");
    constexpr std::size_t G = O::G, Mr = R * O::kRowsPerAcc, Nr = V * O::kColsPerAcc;
    static_assert(Kc >= G && Kc % G == 0, "Kc must be a whole number of k groups");

    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;

    std::vector<std::int32_t> offset;  // −128 · row sums of A
    if constexpr (O::kBPlus128) {
        offset.resize(M);
        for (std::size_t i = 0; i < M; ++i) {
            std::int32_t s = 0;
            for (std::size_t k = 0; k < K; ++k)
                s += A(i, k);
            offset[i] = -128 * s;
        }
    }
    const std::size_t kw_max = (std::min(Kc, K) + G - 1) / G;
    Matrix<typename O::Word> b_pack(1, kw_max * detail::round_up(std::min(Nc, N), Nr));
    Matrix<typename O::Word> a_pack(O::kPackA ? 1 : 0, detail::round_up(std::min(Mc, M), Mr) * kw_max);

    for (std::size_t jc = 0; jc < N; jc += Nc) {
        const std::size_t nc = std::min(Nc, N - jc);
        for (std::size_t pc = 0; pc < K; pc += Kc) {
            const std::size_t kc = std::min(Kc, K - pc), kw = (kc + G - 1) / G;
            detail::pack_b<O, Nr>(B.data() + pc * N + jc, N, kc, nc, b_pack.data());
            for (std::size_t ic = 0; ic < M; ic += Mc) {
                const std::size_t mc = std::min(Mc, M - ic);
                if constexpr (O::kPackA)
                    detail::pack_a<O, Mr>(A.data() + ic * K + pc, K, mc, kc, a_pack.data());
                for (std::size_t jr = 0; jr < nc; jr += Nr)
                    for (std::size_t ir = 0; ir < mc; ir += Mr) {
                        const void* a = O::kPackA ? static_cast<const void*>(a_pack.data() + ir * kw)
                                                  : static_cast<const void*>(A.data() + (ic + ir) * K + pc);
                        detail::micro<O, R, V>(a, K, b_pack.data() + jr * kw, C.data() + (ic + ir) * N + jc + jr, N,
                                               kc, std::min(Mr, mc - ir), std::min(Nr, nc - jr), pc > 0,
                                               offset.empty() ? nullptr : offset.data() + ic + ir);
                    }
            }
        }
    }
}

}  // namespace hpc::gemm
