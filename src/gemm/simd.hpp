#pragma once

/**
 * @file simd.hpp
 * @brief The FP32/FP64 SIMD GEMM kernels, generic over the ISA.
 *
 * Each kernel takes the ISA as a primitive struct, Ops = Ymm (AVX2,
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
 *                             the same micro-kernel in GotoBLAS order with
 *                             each Kc × Nc block of B packed into contiguous
 *                             strips.
 *
 * The register tile and the cache blocks are template parameters with no
 * defaults: the right values depend on the machine and are chosen by
 * measurement (benchmarks/, docs/kernels/). T is deduced from the matrices:
 *
 *   gemm_simd_blocked<Zmm, 8, 3, 512, 256, 512>(A, B, C);   // Matrix<float>
 *
 * Ops<T> is defined only where its ISA is compiled (hpc/isa.hpp), so using
 * an absent ISA is a compile-time error.
 *
 * The file is laid out bottom-up: the micro-kernel, the helpers for tile
 * edges and packing, then the three kernels. detail::pf, the prefetch
 * helper, is also used by prefetch.hpp and sve.hpp.
 */

#include "gemm/avx2.hpp"
#include "gemm/avx512.hpp"
#include "gemm/neon.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

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
// Tile edges, shared by gemm_simd_blocked and gemm_simd_packed
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

/// Copy kc × nc of B (row stride ldb) into NR-column strips, each kc rows of
/// NR contiguous elements; columns past nc are zero.
template <std::size_t NR, typename T>
inline void pack_b_strips(const T* B, std::size_t ldb, std::size_t kc, std::size_t nc, T* dst) noexcept {
    for (std::size_t j0 = 0; j0 < nc; j0 += NR) {
        const std::size_t cols = std::min(NR, nc - j0);
        T* strip               = dst + j0 * kc;
        for (std::size_t k = 0; k < kc; ++k) {
            std::memcpy(strip + k * NR, B + k * ldb + j0, cols * sizeof(T));
            if (cols < NR)
                std::memset(strip + k * NR + cols, 0, (NR - cols) * sizeof(T));
        }
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
// gemm_simd_packed
// ============================================================================
//
// Same micro-kernel as gemm_simd_blocked, different loop order and B
// layout. GotoBLAS order: each Kc × Nc block of B is copied once into
// contiguous strips NR = V·W columns wide, and that copy serves every row
// of A:
//
//   for jc (Nc columns of B and C, rounded down to whole strips)
//     for pc (Kc steps of k)
//       pack B(pc.., jc..) into strips: strip s holds kc rows of NR values
//       for ic (Mc rows of A and C)
//         [PackA: copy A(ic.., pc..) into an mc × kc row-major buffer]
//         for jr (one NR-column strip, read from L1)
//           for ir (R rows at a time)
//             vector_micro<Ops<T>, R, V>: C(R × NR) += A(R × kc) · strip(kc × NR)
//
// Why pack B at all for FP32: row-major B already has the layout the
// micro-kernel wants (row k of a strip is V contiguous vectors), so packing
// is not needed for the instruction, as it is for the dot-product kernels
// (simd_lowp.hpp). It is for the caches: in place, row k and row k+1 of
// a strip are N elements apart, and at N·sizeof(T) a multiple of 4 KiB all
// kc rows fall in the same L1 set (and few L2 sets), so the strip is evicted
// long before its reuse (docs/articles/sme2-cache-associativity.md). Packed,
// the strip is kc·NR contiguous elements and uses every set.
//
// PackA does the same for A: in place, the R rows a micro-kernel streams
// are K elements apart (16 KiB at N = 4096 FP32), so they share one L1 set
// too. Packed, they are kc elements apart.
//
// The strip is zero-padded to NR columns, but a micro-kernel never computes
// padding: the last strip of a block runs detail::row_strip with its true
// width (fewer vectors, then scalar columns), as gemm_simd_blocked's edges
// do. Packing is inside the call and is timed.

template <template <typename> class Ops, int R, int V, std::size_t Mc, std::size_t Kc,
          std::size_t Nc, bool PackA = false, typename T>
void gemm_simd_packed(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_simd_packed: T must be float or double");
    static_assert(R >= 1 && V >= 1, "the register tile needs at least one row and one vector");
    constexpr std::size_t NR  = V * (sizeof(typename Ops<T>::V) / sizeof(T));
    constexpr std::size_t Ncs = Nc / NR * NR;  // whole strips (an 8 × 3 AVX-512 strip is 48 floats wide)
    static_assert(Ncs > 0, "Nc must hold at least one strip");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;

    // Matrix gives a 64-byte aligned, zeroed buffer; one row is enough.
    Matrix<T> b_pack(1, std::min(Kc, K) * std::min(Ncs, (N + NR - 1) / NR * NR));
    Matrix<T> a_pack(PackA ? 1 : 0, std::min(Mc, M) * std::min(Kc, K));

    for (std::size_t jc = 0; jc < N; jc += Ncs) {
        const std::size_t nc = std::min(Ncs, N - jc);
        for (std::size_t pc = 0; pc < K; pc += Kc) {
            const std::size_t kc = std::min(Kc, K - pc);
            detail::pack_b_strips<NR>(B.data() + pc * N + jc, N, kc, nc, b_pack.data());
            for (std::size_t ic = 0; ic < M; ic += Mc) {
                const std::size_t mc = std::min(Mc, M - ic);
                const T* a           = A.data() + ic * K + pc;
                std::size_t lda      = K;
                if constexpr (PackA) {
                    for (std::size_t i = 0; i < mc; ++i)
                        std::memcpy(a_pack.data() + i * kc, a + i * K, kc * sizeof(T));
                    a   = a_pack.data();
                    lda = kc;
                }
                for (std::size_t jr = 0; jr < nc; jr += NR) {
                    const T* strip = b_pack.data() + jr * kc;
                    T* c           = C.data() + ic * N + jc + jr;
                    const std::size_t cols = std::min(NR, nc - jr);
                    for (std::size_t ir = 0; ir < mc; ir += R) {
                        const int rows = static_cast<int>(std::min<std::size_t>(R, mc - ir));
                        detail::row_strip<Ops, R, V>(rows, a + ir * lda, strip, c + ir * N, cols, lda, NR, N, kc);
                    }
                }
            }
        }
    }
}

}  // namespace hpc::gemm
