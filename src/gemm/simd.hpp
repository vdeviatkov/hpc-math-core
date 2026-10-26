#pragma once

/**
 * @file simd.hpp
 * @brief The SIMD GEMM kernels: FP32/FP64, generic over the ISA, and
 *        low precision (bf16, fp16, INT8), generic over the instruction.
 *
 * The FP kernels take the ISA as a primitive struct, Ops = Ymm (AVX2,
 * avx2.hpp), Zmm (AVX-512, avx512.hpp) or Q (NEON, neon.hpp), one
 * optimisation step apart:
 *
 *  gemm_simd_reordered<Ops>   i → k → j: A(i, k) broadcast, FMA against
 *                             stride-1 B and C rows. About the same as
 *                             scalar gemm_reordered, which the compiler
 *                             auto-vectorises; no blocking, so it degrades
 *                             once C row i no longer stays in L1.
 *  gemm_simd_blocked<Ops, R, V, TileM, TileK, TileN, PfDist = 0>
 *                             tiled i → k → j with an R × V-vector tile of C
 *                             held in registers for a whole k block
 *                             (vector_micro). PfDist > 0 adds software
 *                             prefetch, PfDist strips ahead.
 *  gemm_simd_packed<Ops, R, V, Mc, Kc, Nc, PackA = false>
 *                             the same outer product per k step in GotoBLAS
 *                             order, with each Kc × Nc block of B packed
 *                             into contiguous strips: detail::gemm_kgroup,
 *                             the kernel of gemm_simd_lowp, with the FMA as
 *                             an instruction of k group G = 1.
 *
 * The low-precision kernel takes a struct for one instruction shape,
 * defined with its ISA:
 *
 *  gemm_simd_lowp<Ops, R, V, Mc, Kc, Nc>
 *                             C (FP32, or INT32 for INT8) = A · B with bf16,
 *                             fp16 or INT8 inputs: detail::gemm_kgroup with
 *                             the instruction's k group, G = 1–8.
 *
 *   ZmmDot<In>    VDPBF16PS, VPDPBUSD (avx512.hpp)          bf16, INT8
 *   QMmla<In>     BFMMLA, SMMLA (neon.hpp)                  bf16, INT8
 *   QFmlal<In>    FMLAL, fp16 widened to FP32 (neon.hpp)    fp16
 *   QFmlaF16<In>  FMLA .8h, fp16 accumulation (neon.hpp)    fp16
 *
 * The register tile and the cache blocks are template parameters with no
 * defaults: the right values depend on the machine and are chosen by
 * measurement (benchmarks/, docs/kernels/, docs/articles/lowp-gemm.md). For
 * gemm_simd_lowp, R × V counts accumulators (rows × columns of them) and
 * Mc × Kc × Nc are in elements. T (or In) is deduced from the matrices:
 *
 *   gemm_simd_blocked<Zmm, 8, 3, 512, 256, 512>(A, B, C);   // Matrix<float>
 *
 * Ops<T> is defined only where its ISA is compiled (hpc/isa.hpp), so using
 * an absent ISA is a compile-time error.
 *
 * The file is laid out bottom-up: vector_micro and the tile edges of
 * gemm_simd_blocked, the reordered and blocked kernels, then the k-group
 * kernel (packing, micro-kernel, driver), gemm_simd_packed and
 * gemm_simd_lowp. SME's widening outer products (gemm_sme, sme.hpp) are not
 * part of it: they run in streaming mode and accumulate in ZA. detail::pf,
 * the prefetch helper, is also used by prefetch.hpp and sve.hpp.
 */

#include "gemm/avx2.hpp"
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

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

namespace hpc::gemm {

// ============================================================================
// The micro-kernel
// ============================================================================
//
// C[0..R)[0..V·W) += A[0..R)[0..k_len) × B[0..k_len)[0..V·W), with an
// R × V-vector tile of C held in R·V accumulators for the whole k loop.
// Per k step: V loads of B row k, one load of A(i+r, k) per row, R·V FMAs:
// an outer product of a column of A and a row of B added into the tile.
//
// The three ISAs differ only in their instructions, so each supplies an Ops
// struct per element type (Ymm<T> in avx2.hpp, Zmm<T> in avx512.hpp, Q<T>
// in neon.hpp):
//
//   using V;                                  // vector register type
//   static V    load(const T* p);             // W contiguous elements
//   static void store(T* p, V v);
//   static auto load_a(const T* p);           // A(i+r, k) as fma's first operand
//   static V    fma(decltype(load_a(p)) a, V b, V acc);   // acc + a · b
//
// load_a is where the ISAs really differ: AVX2 broadcasts from memory
// (vbroadcastss), AVX-512 broadcasts a scalar (vbroadcastss / set1), and
// NEON loads one element with a plain LDR for an FMLA by element. Each Ops
// struct documents why.
//
// The fixed-count loops must unroll completely so acc[][] stays in
// registers; R and V are template parameters for that, and so that the
// benchmarks can sweep tile shapes (docs/kernels/).

template <typename Ops, int R, int V, typename T>
inline void vector_micro(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                         std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t k_len) noexcept {
    static_assert(R >= 1 && R <= 16 && V >= 1 && V <= 16, "tile dimensions up to 16 (the unroll factor)");
    using Vec               = typename Ops::V;
    constexpr std::size_t W = sizeof(Vec) / sizeof(T);  // lanes per vector

    Vec acc[R][V];
#pragma GCC unroll 16
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            acc[r][v] = Ops::load(c + r * ldc + v * W);
    for (std::size_t k = 0; k < k_len; ++k) {
        Vec bv[V];
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            bv[v] = Ops::load(b + k * ldb + v * W);
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r) {
            const auto ar = Ops::load_a(a + r * lda + k);
#pragma GCC unroll 16
            for (int v = 0; v < V; ++v)
                acc[r][v] = Ops::fma(ar, bv[v], acc[r][v]);
        }
    }
#pragma GCC unroll 16
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            Ops::store(c + r * ldc + v * W, acc[r][v]);
}

namespace detail {

/// Software prefetch with compile-time hints. Rw: 0 read, 1 write; Loc:
/// temporal locality 0 (none) … 3 (keep in all levels).
template <int Rw = 0, int Loc = 2>
inline void pf(const void* addr) noexcept {
#if defined(_MSC_VER)
    // _MM_HINT_T0 (L1), _MM_HINT_T1 (L2), _MM_HINT_T2 (L3), _MM_HINT_NTA
    constexpr int hint = (Loc == 3) ? _MM_HINT_T0 : (Loc == 2) ? _MM_HINT_T1 : (Loc == 1) ? _MM_HINT_T2 : _MM_HINT_NTA;
    _mm_prefetch(static_cast<const char*>(addr), hint);
#else
    __builtin_prefetch(addr, Rw, Loc);
#endif
}

// ============================================================================
// Tile edges of gemm_simd_blocked
// ============================================================================
//
// The edges of a block use the same micro-kernel with fewer rows (when the
// block or M is not a multiple of R) or fewer vectors; only the last columns
// that do not fill one vector take a scalar loop.

// Run vector_micro<Ops<T>, rows, V> for a runtime `rows` in 1 … R.
template <template <typename> class Ops, int R, int V, typename T>
inline void run_rows(int rows, const T* a, const T* b, T* c, std::size_t lda, std::size_t ldb,
                     std::size_t ldc, std::size_t k_len) noexcept {
    if constexpr (R >= 1) {
        if (rows == R)
            vector_micro<Ops<T>, R, V>(a, b, c, lda, ldb, ldc, k_len);
        else
            run_rows<Ops, R - 1, V>(rows, a, b, c, lda, ldb, ldc, k_len);
    }
}

// Run vector_micro<Ops<T>, rows, vecs> for runtime `vecs` in 1 … V.
template <template <typename> class Ops, int R, int V, typename T>
inline void run_tile(int rows, int vecs, const T* a, const T* b, T* c, std::size_t lda, std::size_t ldb,
                     std::size_t ldc, std::size_t k_len) noexcept {
    if constexpr (V >= 1) {
        if (vecs == V)
            run_rows<Ops, R, V>(rows, a, b, c, lda, ldb, ldc, k_len);
        else
            run_tile<Ops, R, V - 1>(rows, vecs, a, b, c, lda, ldb, ldc, k_len);
    }
}

/// `rows` (≤ R) rows of one k pass across j_len columns of a block: R × V
/// tiles, then one narrower tile for the remaining whole vectors, then
/// scalar columns.
template <template <typename> class Ops, int R, int V, typename T>
inline void row_strip(int rows, const T* a, const T* b, T* c, std::size_t j_len, std::size_t lda,
                      std::size_t ldb, std::size_t ldc, std::size_t k_len) noexcept {
    constexpr std::size_t W = sizeof(typename Ops<T>::V) / sizeof(T);  // lanes per vector
    std::size_t j           = 0;
    for (; j + V * W <= j_len; j += V * W)
        run_rows<Ops, R, V>(rows, a, b + j, c + j, lda, ldb, ldc, k_len);
    if (const int vecs = static_cast<int>((j_len - j) / W); vecs > 0) {
        run_tile<Ops, R, V - 1>(rows, vecs, a, b + j, c + j, lda, ldb, ldc, k_len);
        j += static_cast<std::size_t>(vecs) * W;
    }
    for (; j < j_len; ++j)  // fewer columns than one vector
        for (int r = 0; r < rows; ++r) {
            T acc = c[r * ldc + j];
            for (std::size_t k = 0; k < k_len; ++k)
                acc += a[r * lda + k] * b[k * ldb + j];
            c[r * ldc + j] = acc;
        }
}

}  // namespace detail

// ============================================================================
// gemm_simd_reordered
// ============================================================================
//
// i → k → j: A(i, k) broadcast once per k, FMA against stride-1 B and C rows.

template <template <typename> class Ops, typename T>
void gemm_simd_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_simd_reordered: T must be float or double");
    using O                 = Ops<T>;
    constexpr std::size_t W = sizeof(typename O::V) / sizeof(T);  // lanes per vector

    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    for (std::size_t i = 0; i < M; ++i) {
        T* c_row = C.data() + i * N;
        for (std::size_t k = 0; k < K; ++k) {
            const T* a_ik  = A.data() + i * K + k;
            const T* b_row = B.data() + k * N;
            const auto a   = O::load_a(a_ik);
            std::size_t j  = 0;
            for (; j + W <= N; j += W)
                O::store(c_row + j, O::fma(a, O::load(b_row + j), O::load(c_row + j)));
            for (; j < N; ++j)  // columns that do not fill a vector
                c_row[j] += *a_ik * b_row[j];
        }
    }
}

// ============================================================================
// gemm_simd_blocked
// ============================================================================
//
// C = A · B, row-major, i-k-j order with three levels of blocking:
//
//   for i_blk (TileM rows of A and C)
//     for k_blk (TileK: the k extent of one register-tile pass)
//       for j_blk (TileN columns of B and C)
//         for i in the block, R rows at a time
//           for j in the block, V vectors at a time
//             vector_micro<Ops<T>, R, V>: C(R × V·W) += A(R × TileK) · B(TileK × V·W)
//
// The register tile, R rows × V vectors of C kept in R·V accumulators for
// the whole k pass, is a template parameter, so one kernel serves all tile
// shapes and benchmarks can sweep them (bench_gemm, docs/kernels/). Edges:
// detail::row_strip.
//
// PfDist > 0 adds software prefetch: the next k block of B, and the A and C
// rows PfDist register-tile strips ahead (detail::pf, as in prefetch.hpp).

template <template <typename> class Ops, int R, int V, std::size_t TileM, std::size_t TileK,
          std::size_t TileN, std::size_t PfDist = 0, typename T>
void gemm_simd_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_simd_blocked: T must be float or double");
    static_assert(R >= 1 && V >= 1, "the register tile needs at least one row and one vector");
    using detail::pf;
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    const std::size_t lda = K, ldb = N, ldc = N;
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    for (std::size_t i_blk = 0; i_blk < M; i_blk += TileM) {
        const std::size_t i_end = std::min(i_blk + TileM, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += TileK) {
            const std::size_t k_len = std::min(TileK, K - k_blk);
            if constexpr (PfDist > 0)
                if (k_blk + TileK < K)
                    pf<0, 2>(B.data() + (k_blk + TileK) * ldb);  // next k block of B
            for (std::size_t j_blk = 0; j_blk < N; j_blk += TileN) {
                const std::size_t j_len = std::min(TileN, N - j_blk);
                for (std::size_t i = i_blk; i < i_end; i += R) {
                    if constexpr (PfDist > 0)
                        if (i + R * PfDist < i_end) {
                            pf<0, 2>(A.data() + (i + R * PfDist) * lda + k_blk);  // A strip ahead
                            pf<1, 3>(C.data() + (i + R * PfDist) * ldc + j_blk);  // C strip ahead
                        }
                    const int rows = static_cast<int>(std::min<std::size_t>(R, i_end - i));
                    detail::row_strip<Ops, R, V>(rows, A.data() + i * lda + k_blk, B.data() + k_blk * ldb + j_blk,
                                                 C.data() + i * ldc + j_blk, j_len, lda, ldb, ldc, k_len);
                }
            }
        }
    }
}

// ============================================================================
// The k-group kernel: gemm_simd_packed and gemm_simd_lowp
// ============================================================================
//
// gemm_simd_packed (FP32, FP64, below) and gemm_simd_lowp (bf16, fp16, INT8,
// at the end of this file) are one GotoBLAS kernel, detail::gemm_kgroup. The
// instructions differ in their k group: the G consecutive k that one
// instruction consumes per lane or per register. An FMA consumes one
// (G = 1); the dot products and matrix multiplies of the low-precision
// kernels consume 2–8. The kernel handles any G in packing B into strips of
// k-group words, packing A where the instruction needs it, the micro-kernel
// and the tile edges. With G = 1, a word is one element and packing B is a
// plain copy of rows.
//
// The instruction comes as an Ops struct. For the FP kernels, detail::Fma
// builds one from the ISA's Ymm<T>, Zmm<T> or Q<T>. G is a member of the
// Ops struct, not a separate template parameter, because the instruction
// fixes it.
//
// The Ops interface. Formats and layout:
//   using In, Out;                     // input element; output: T for FP, else FP32 or INT32
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
// Packing B quickly where the instruction set has a transpose or zip (G > 1 only):
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
        const In* src          = A + i0 * lda;
        // k groups outermost: the strip is written in order, each of the
        // rows read in order.
        for (std::size_t w = 0; w < full; ++w)
            for (std::size_t r = 0; r < rows; ++r)
                std::memcpy(&strip[w * Mr + r], src + r * lda + w * G, sizeof(Word));
        if (full < kw)
            for (std::size_t r = 0; r < rows; ++r) {
                In g[G] = {};
                std::memcpy(g, src + r * lda + full * G, (kc - full * G) * sizeof(In));
                std::memcpy(&strip[full * Mr + r], g, sizeof(Word));
            }
        if (rows < Mr)
            for (std::size_t w = 0; w < kw; ++w)
                std::fill(strip + w * Mr + rows, strip + (w + 1) * Mr, Word{});
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
/// (Mr × Nr elements of C) in a B strip Ns ≥ Nr words wide (narrower tiles
/// serve a strip's last columns). A is a packed strip (kPackA) or rows of A in
/// place, lda apart. `init` is the start value of row r's accumulators when
/// `accumulate` is false (INT8 with kBPlus128), else C is loaded. Kept out
/// of line: inlined into the kernel, the tile's row pointers compete with the
/// kernel's own values for registers, and the k loop reloads them (seen with
/// Clang on the AVX-512 bf16 kernel: a 56-instruction loop became 420).
template <typename O, int R, int V, std::size_t Ns>
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
    // not stored), and so does A read in place, whose rows past the tile may
    // be past the matrix; otherwise buffered tiles compute every row into tmp.
    const auto c_row = [rows](int r) -> std::size_t {
        return O::kMaskedEdges || !O::kPackA ? std::min(std::size_t(r * Rpa), rows - 1) : std::size_t(r * Rpa);
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
        k_loop<O, R, V, Ns>(acc, ar, Mr, b, (kc + G - 1) / G);
    } else {
        const auto* a           = static_cast<const In*>(a_ptr);
        const std::size_t whole = kc / G;  // whole k groups; a partial one follows if kc % G
        const In* ar[R];
        for (int r = 0; r < R; ++r)
            ar[r] = a + c_row(r) * lda;
        k_loop<O, R, V, Ns>(acc, ar, G, b, whole);
        if (whole * G < kc) {  // a last k group shorter than G, padded with zeros
            In tail[R][G] = {};
            const In* tp[R];
            for (int r = 0; r < R; ++r) {
                std::memcpy(tail[r], ar[r] + whole * G, (kc - whole * G) * sizeof(In));
                tp[r] = tail[r];
            }
            k_loop<O, R, V, Ns>(acc, tp, G, b + whole * Ns, 1);
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

/// micro with the fewest accumulator columns, at most V, that cover `cols`:
/// the last strip of a block is computed only as wide as it is.
template <typename O, int R, int V, std::size_t Ns, typename... Args>
void micro_cols(std::size_t cols, Args... args) {
    if constexpr (V > 1)
        if (cols <= std::size_t(V - 1) * O::kColsPerAcc)
            return micro_cols<O, R, V - 1, Ns>(cols, args...);
    micro<O, R, V, Ns>(args...);
}

// GotoBLAS order: each kc × nc block of B is packed once and serves every
// row of A; A is packed per mc × kc block where the instruction needs it:
//
//   for jc (Nc columns of B and C, rounded down to whole strips)
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
template <typename O, int R, int V, std::size_t Mc, std::size_t Kc, std::size_t Nc>
void gemm_kgroup(const Matrix<typename O::In>& A, const Matrix<typename O::In>& B, Matrix<typename O::Out>& C) {
    static_assert(R >= 1 && V >= 1 && O::template kTileFits<R, V>,
                  "this R × V tile does not fit the instruction's registers or lanes");
    constexpr std::size_t G = O::G, Mr = R * O::kRowsPerAcc, Nr = V * O::kColsPerAcc;
    constexpr std::size_t Ncs = Nc / Nr * Nr;  // whole strips (an 8 × 3 AVX-512 strip is 48 floats wide)
    static_assert(Kc >= G && Kc % G == 0, "Kc must be a whole number of k groups");
    static_assert(Ncs > 0, "Nc must hold at least one strip");

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
    // Matrix gives 64-byte aligned buffers; one row is enough.
    const std::size_t kw_max = (std::min(Kc, K) + G - 1) / G;
    Matrix<typename O::Word> b_pack(1, kw_max * round_up(std::min(Ncs, N), Nr));
    Matrix<typename O::Word> a_pack(O::kPackA ? 1 : 0, round_up(std::min(Mc, M), Mr) * kw_max);

    for (std::size_t jc = 0; jc < N; jc += Ncs) {
        const std::size_t nc = std::min(Ncs, N - jc);
        for (std::size_t pc = 0; pc < K; pc += Kc) {
            const std::size_t kc = std::min(Kc, K - pc), kw = (kc + G - 1) / G;
            pack_b<O, Nr>(B.data() + pc * N + jc, N, kc, nc, b_pack.data());
            for (std::size_t ic = 0; ic < M; ic += Mc) {
                const std::size_t mc = std::min(Mc, M - ic);
                if constexpr (O::kPackA)
                    pack_a<O, Mr>(A.data() + ic * K + pc, K, mc, kc, a_pack.data());
                for (std::size_t jr = 0; jr < nc; jr += Nr)
                    for (std::size_t ir = 0; ir < mc; ir += Mr) {
                        const void* a = O::kPackA ? static_cast<const void*>(a_pack.data() + ir * kw)
                                                  : static_cast<const void*>(A.data() + (ic + ir) * K + pc);
                        const std::size_t cols = std::min(Nr, nc - jr);
                        micro_cols<O, R, V, Nr>(cols, a, K, b_pack.data() + jr * kw, C.data() + (ic + ir) * N + jc + jr,
                                                N, kc, std::min(Mr, mc - ir), cols, pc > 0,
                                                offset.empty() ? nullptr : offset.data() + ic + ir);
                    }
            }
        }
    }
}

/// The FP32/FP64 FMA of an ISA (Isa<T> = Ymm<T>, Zmm<T>, Q<T>) as the Ops
/// of gemm_kgroup, with G = 1: one accumulator is one vector of a row of C,
/// fed by one broadcast of A (load_a) and one vector of B. Tile edges go
/// through a buffer. PackA packs A into strips (a_offset(r) = row r of a k
/// step); otherwise A is read in place.
template <template <typename> class Isa, bool PackA>
struct Fma {
    template <typename T>
    struct Ops {
        using I    = Isa<T>;
        using In   = T;
        using Out  = T;
        using Word = T;
        using Acc  = typename I::V;
        using AReg = decltype(I::load_a(static_cast<const T*>(nullptr)));
        using BReg = Acc;
        struct Edge {};
        static constexpr std::size_t G = 1;
        static constexpr bool kPackA = PackA, kMaskedEdges = false, kBPlus128 = false;
        static constexpr int kRowsPerAcc = 1, kColsPerAcc = int(sizeof(Acc) / sizeof(T));
        static constexpr Word kFlip      = 0;
        template <int R, int V>
        static constexpr bool kTileFits = R <= 32 && V <= 32;  // the unroll factor; spills are the caller's choice
        static constexpr std::size_t a_offset(int r) { return std::size_t(r); }
        static constexpr std::size_t b_offset(int v) { return std::size_t(v) * kColsPerAcc; }
        static AReg load_a(const T* p) { return I::load_a(p); }
        static BReg load_b(const T* p) { return I::load(p); }
        static Acc madd(Acc acc, AReg a, BReg b, int, int) { return I::fma(a, b, acc); }
        static Edge edge(std::size_t) { return {}; }
        static Acc init(std::int32_t) { return Acc{}; }
        static Acc load_c(const T* c, std::size_t, Edge) { return I::load(c); }
        static void store_c(T* c, std::size_t, Acc v, Edge) { I::store(c, v); }
    };
};

}  // namespace detail

// ============================================================================
// gemm_simd_packed
// ============================================================================
//
// gemm_kgroup with the FMA (G = 1): the same outer product per k step as
// gemm_simd_blocked's micro-kernel, in GotoBLAS order, with each Kc × Nc
// block of B packed into contiguous strips Nr = V·W columns wide. That copy
// serves every row of A.
//
// Why pack B at all for FP32: row-major B already has the layout the
// micro-kernel wants (row k of a strip is V contiguous vectors), so packing
// is not needed for the instruction, as it is for the dot-product kernels
// (gemm_simd_lowp). It is for the caches: in place, row k and row k+1 of
// a strip are N elements apart, and at N·sizeof(T) a multiple of 4 KiB all
// kc rows fall in the same L1 set (and few L2 sets), so the strip is evicted
// long before its reuse (docs/articles/sme2-cache-associativity.md). Packed,
// the strip is kc·Nr contiguous elements and uses every set.
//
// PackA does the same for A: in place, the R rows a micro-kernel streams
// are K elements apart (16 KiB at N = 4096 FP32), so they share one L1 set
// too. Packed into R-row strips, the R values of one k step are contiguous.
//
// Strips are zero-padded to Nr columns; a tile cut short by the matrix or a
// block is computed whole into a buffer and its present part copied out.
// Packing is inside the call and is timed.

template <template <typename> class Ops, int R, int V, std::size_t Mc, std::size_t Kc,
          std::size_t Nc, bool PackA = false, typename T>
void gemm_simd_packed(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_simd_packed: T must be float or double");
    detail::gemm_kgroup<typename detail::Fma<Ops, PackA>::template Ops<T>, R, V, Mc, Kc, Nc>(A, B, C);
}

// ============================================================================
// gemm_simd_lowp
// ============================================================================
//
// gemm_kgroup with a low-precision instruction. What the instructions share
// is a k group: G consecutive k (G = 1 for the fp16 FMAs, 2–8 for the dot
// products and matrix multiplies) that one instruction consumes per lane or
// per register. B is always packed by k group, and A is packed the same way
// where the instruction needs it (Ops::kPackA); the dot products broadcast
// G consecutive k of a row of A, which row-major A already has, so they
// read A in place.

template <template <typename> class Ops, int R, int V, std::size_t Mc, std::size_t Kc, std::size_t Nc,
          typename In>
void gemm_simd_lowp(const Matrix<In>& A, const Matrix<In>& B, Matrix<lowp_out_t<In>>& C) {
    static_assert(std::is_same_v<typename Ops<In>::Out, lowp_out_t<In>>, "Ops<In>::Out must be lowp_out_t<In>");
    detail::gemm_kgroup<Ops<In>, R, V, Mc, Kc, Nc>(A, B, C);
}

}  // namespace hpc::gemm
