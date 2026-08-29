#pragma once

/**
 * @file sme.hpp
 * @brief Three progressive ARM SME (Scalable Matrix Extension) GEMM kernels.
 *
 * ============================================================
 *  Why SME is fundamentally different from SVE/NEON/AVX
 * ============================================================
 *
 * Every other kernel family in this repo (AVX2, AVX-512, NEON, SVE) computes
 * a C(i,j) accumulator with vector *FMA*: broadcast one scalar, multiply it
 * against a vector, add into another vector. The vector register holds one
 * row (or part of one row) of C.
 *
 * SME instead computes GEMM with a hardware **outer product**: given a
 * column vector `a` (SVL elements) and a row vector `b` (SVL elements), a
 * single FMOPA instruction computes the full SVL×SVL outer product
 * `a ⊗ b` and accumulates it into a dedicated 2-D accumulator register
 * called ZA — not a normal vector register, a whole tile of them.
 *
 *   ZA[r][c] += a[r] * b[c]     for r, c in [0, SVL)
 *
 * Looping this over k = 0..K-1 with a[k] = A(i0+r, k) and b[k] = B(k, j0+c)
 * computes exactly C(i0+r, j0+c) = sum_k A(i0+r,k)*B(k,j0+c) — a full
 * SVL×SVL tile of C — in K instructions instead of K×SVL FMAs. This is the
 * same computational primitive NVIDIA tensor cores (WMMA, already used in
 * gemm_cuda_wmma) and Intel AMX (see amx.hpp) use, just for CPU SIMD.
 *
 * SME reuses SVE's vector-length-agnostic model: SVL is implementation
 * defined and queried at runtime via svcntsw() (f32) / svcntsd() (f64) —
 * the *streaming* vector length, which can differ from the non-streaming
 * SVE length (and on Apple Silicon, there IS no non-streaming SVE — see
 * "Apple Silicon note" below).
 *
 *
 * ============================================================
 *  Streaming mode
 * ============================================================
 *
 * SME instructions (FMOPA, LD1/ST1 into ZA, ZERO {za}, …) are only legal
 * while the CPU is in "Streaming SVE mode". A function enters this mode via
 * the `SMSTART` instruction and must leave it via `SMSTOP` before returning
 * to normal code — Clang generates both automatically for a function marked
 * `__arm_locally_streaming`. The ZA accumulator array must additionally be
 * declared as "owned" by the function via `__arm_new("za")`, which emits a
 * `zero {za}`-safe prologue/epilogue (saving/restoring any caller ZA state).
 *
 *   __arm_locally_streaming __arm_new("za")
 *   void my_sme_kernel(...) { ... }
 *
 * ============================================================
 *  Hardware finding: gather-loads are NOT legal in streaming mode
 * ============================================================
 *
 * The obvious way to build the column vector `a` for the outer product is
 * `svld1_gather_index` with a stride-lda index vector (A is row-major, so a
 * column is strided). On this hardware (Apple M4 Max, SME2) Clang rejects
 * this: "builtin can only be called from a non-streaming function". Gather/
 * scatter addressing modes are excluded from the Streaming SVE instruction
 * subset by the architecture (Arm ARM, SME chapter). This is NOT a NEON/SVE
 * limitation — it is specific to streaming-mode execution.
 *
 * Consequence: the column vector must be assembled with ordinary scalar
 * loads into a stack/heap buffer, then loaded contiguously with svld1. This
 * reframes the classic "naive vs reordered" cache lesson already present in
 * every other kernel family in this repo: gemm_sme_naive re-does this
 * column-assembly scalar loop on *every* k (redundant work repeated for
 * every (i0,j0) tile pair); gemm_sme_reordered does it *once* per i0-tile
 * for the full K range and reuses the packed panel across all j0-tiles;
 * gemm_sme_blocked additionally bounds the packed panel's size so it stays
 * cache-resident even when K is very large.
 *
 *
 * ============================================================
 *  Three kernels — same pedagogical structure as AVX2/NEON/SVE
 * ============================================================
 *
 *  Kernel 1 — gemm_sme_naive
 *    For each SVLxSVL output tile (i0, j0): zero the ZA tile, then for each
 *    k re-gather A(i0..i0+SVL, k) via a scalar loop into a stack buffer,
 *    load B(k, j0..j0+SVL) contiguously, and accumulate one outer product.
 *    The A-column gather is redone for every (i0, j0, k) triple — O(N/SVL)
 *    times more scalar work than necessary. Measured: ~3 GFLOP/s, flat
 *    across N (scalar-gather bound, same lesson as every other *_naive).
 *
 *  Kernel 2 — gemm_sme_reordered
 *    For each i0-tile: pack A(i0..i0+SVL, 0..K) into a contiguous SVL×K
 *    buffer ONCE (one scalar pass), then loop all j0-tiles reusing that
 *    packed panel with pure contiguous loads for both operands. Removes
 *    the O(N/SVL) redundant gathering. Measured: 218-380 GFLOP/s
 *    (single-threaded, Apple M4 Max, f32) — far above any other CPU kernel
 *    in this repo. Degrades at very large N once the packed panel
 *    (SVL x K x 4B) exceeds L1/L2.
 *
 *  Kernel 3 — gemm_sme_blocked
 *    Adds K-tiling (kSmeTileK) on top of the panel-packing scheme: the
 *    packed A buffer is bounded to SVL x kSmeTileK regardless of K, keeping
 *    it cache-resident. Partial C sums are carried across k-tiles by
 *    reloading them into ZA via svld1_hor_za (rather than re-deriving them
 *    from scratch), at the cost of extra C traffic. Wins over "reordered"
 *    once K is large enough that the unbounded packed panel would spill
 *    L2 (measured crossover ~N=2048 on this hardware: 254 vs 210 GFLOP/s
 *    at N=2048; 172 vs 127 GFLOP/s at N=4096).
 *
 *
 * ============================================================
 *  Apple Silicon note
 * ============================================================
 *
 * Apple M4 (and M4 Pro/Max) is, as of 2026, essentially the only shipping
 * SME2 hardware widely available to individual developers (server-class SME
 * implementations are expected from other vendors but are not yet common).
 * Apple Silicon does *not* implement general (non-streaming) SVE — only
 * Streaming SVE via SME. This has a concrete build implication:
 * `-march=armv9-a+sme2` compiles correctly but the resulting binary SIGILLs
 * at runtime on the very first `cntd`/`cntw`-family instruction Clang emits
 * in the function prologue (used to size the ZA-save spill buffer) — those
 * are ordinary (non-streaming) SVE instructions, and Apple hardware has no
 * non-streaming SVE unit to execute them on. `-mcpu=apple-m4` (or any Apple
 * CPU name that implies SME2) avoids this by making the compiler size the
 * prologue buffer without an outside-streaming SVE instruction. This repo's
 * CMake SME probe (see CMakeLists.txt / HPC_ENABLE_SME) actually COMPILES
 * AND RUNS a tiny SME snippet at configure time for exactly this reason —
 * a compile-only check is not sufficient to prove SME works on a given
 * (compiler, flags, hardware) triple.
 *
 *
 * ============================================================
 *  Hardware availability
 * ============================================================
 *
 * SME is available on:
 *   Apple M4 / M4 Pro / M4 Max (SME2, 512-bit SVL, f32/f64/bf16/f16/int8/int16)
 *   Future Armv9.2+ server/mobile SoCs implementing FEAT_SME (announced by
 *   several vendors, not yet broadly available at time of writing)
 *   NOT on: Apple M1/M2/M3, AWS Graviton3/4, Fujitsu A64FX, x86 (Intel/AMD)
 *
 * The header detects __ARM_FEATURE_SME at compile time.
 * If absent, all three kernels fall back to their SVE equivalents (which
 * themselves fall back to NEON, then AVX2, then scalar).
 */

#include "gemm/sve.hpp"  // fallback chain: SME → SVE → NEON → AVX2 → scalar
#include "hpc/matrix.hpp"

#ifdef __ARM_FEATURE_SME
    #include <arm_sme.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace hpc::gemm {

// ============================================================================
// SME tile constants
// ============================================================================

// K-tile width used by gemm_sme_blocked to bound the packed-A panel size.
// At SVL=16 f32 (Apple M4 Max, 512-bit SVL) this caps the panel at
// 16 * 256 * 4B = 16 KB — comfortably L1-resident regardless of K.
inline constexpr std::size_t kSmeTileK = 256;

#ifdef __ARM_FEATURE_SME
// Forward declarations: the gemm_sme_* templates below call these before
// their (streaming-mode) definitions appear later in the file. Two-phase
// name lookup requires either a prior declaration or ADL; ADL does not
// apply here since the arguments are plain pointers/ints, not Matrix<T>.
void sme_gemm_naive_f32_impl(const float* __restrict A, const float* __restrict B,
                              float* __restrict C, std::int64_t M, std::int64_t N, std::int64_t K,
                              std::int64_t lda, std::int64_t ldb, std::int64_t ldc);
void sme_gemm_naive_f64_impl(const double* __restrict A, const double* __restrict B,
                              double* __restrict C, std::int64_t M, std::int64_t N,
                              std::int64_t K, std::int64_t lda, std::int64_t ldb,
                              std::int64_t ldc);
void sme_gemm_reordered_f32_impl(const float* __restrict A, const float* __restrict B,
                                  float* __restrict C, std::int64_t M, std::int64_t N,
                                  std::int64_t K, std::int64_t lda, std::int64_t ldb,
                                  std::int64_t ldc);
void sme_gemm_reordered_f64_impl(const double* __restrict A, const double* __restrict B,
                                  double* __restrict C, std::int64_t M, std::int64_t N,
                                  std::int64_t K, std::int64_t lda, std::int64_t ldb,
                                  std::int64_t ldc);
void sme_gemm_blocked_f32_impl(const float* __restrict A, const float* __restrict B,
                                float* __restrict C, std::int64_t M, std::int64_t N,
                                std::int64_t K, std::int64_t lda, std::int64_t ldb,
                                std::int64_t ldc);
void sme_gemm_blocked_f64_impl(const double* __restrict A, const double* __restrict B,
                                double* __restrict C, std::int64_t M, std::int64_t N,
                                std::int64_t K, std::int64_t lda, std::int64_t ldb,
                                std::int64_t ldc);
#endif  // __ARM_FEATURE_SME

// ============================================================================
// Kernel 1: gemm_sme_naive — per-(i0,j0,k) scalar column gather + FMOPA
// ============================================================================

/**
 * @brief SME GEMM using single-ZA-tile outer-product accumulation, with the
 *        A-column vector re-gathered via a scalar loop on every k.
 *
 * Loop structure:
 *   for i0 in steps of SVL:
 *     for j0 in steps of SVL:
 *       zero ZA
 *       for k in [0, K):
 *         a_col[r] = A(i0+r, k)  for r in [0, SVL)   ← scalar gather, redone
 *                                                        every (i0,j0,k)
 *         b_row = B(k, j0..j0+SVL)                    ← contiguous load
 *         ZA += outer(a_col, b_row)                   ← single FMOPA
 *       store ZA rows to C(i0.., j0..)
 *
 * Pedagogical purpose: same lesson as every other *_naive kernel in this
 * repo — SIMD/matrix-engine width does not help when memory access is
 * cache-hostile. Here the hostility is self-inflicted: gather-load
 * intrinsics are illegal in SME streaming mode (see file header), so this
 * kernel pays the strided-A cost with 1 scalar load per row per k, redone
 * for every j0-tile — O(N/SVL) more scalar work than gemm_sme_reordered.
 *
 * Falls back to gemm_sve_naive on non-SME targets.
 */
template <typename T>
void gemm_sme_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __ARM_FEATURE_SME
    gemm_sve_naive(A, B, C);
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sme_naive: T must be float or double");

    const std::int64_t M   = static_cast<std::int64_t>(A.rows());
    const std::int64_t K   = static_cast<std::int64_t>(A.cols());
    const std::int64_t N   = static_cast<std::int64_t>(B.cols());
    const std::int64_t lda = K;
    const std::int64_t ldb = N;
    const std::int64_t ldc = N;

    assert(static_cast<std::int64_t>(B.rows()) == K && static_cast<std::int64_t>(C.rows()) == M &&
           static_cast<std::int64_t>(C.cols()) == N);
    C.zero();

    const T* a_ptr = A.data();
    const T* b_ptr = B.data();
    T* c_ptr       = C.data();

    if constexpr (sizeof(T) == 4) {
        sme_gemm_naive_f32_impl(a_ptr, b_ptr, c_ptr, M, N, K, lda, ldb, ldc);
    } else {
        sme_gemm_naive_f64_impl(a_ptr, b_ptr, c_ptr, M, N, K, lda, ldb, ldc);
    }
#endif
}

#ifdef __ARM_FEATURE_SME

__arm_locally_streaming __arm_new("za")
inline void sme_gemm_naive_f32_impl(const float* __restrict A, const float* __restrict B,
                                     float* __restrict C, std::int64_t M, std::int64_t N,
                                     std::int64_t K, std::int64_t lda, std::int64_t ldb,
                                     std::int64_t ldc) {
    const std::int64_t svl = static_cast<std::int64_t>(svcntsw());
    std::vector<float> col_buf(static_cast<std::size_t>(svl));

    for (std::int64_t i0 = 0; i0 < M; i0 += svl) {
        svbool_t pg_rows       = svwhilelt_b32(static_cast<std::uint32_t>(i0),
                                                static_cast<std::uint32_t>(M));
        const std::int64_t rows_here = std::min<std::int64_t>(svl, M - i0);
        for (std::int64_t j0 = 0; j0 < N; j0 += svl) {
            svbool_t pg_cols = svwhilelt_b32(static_cast<std::uint32_t>(j0),
                                              static_cast<std::uint32_t>(N));
            svzero_za();
            for (std::int64_t k = 0; k < K; ++k) {
                for (std::int64_t r = 0; r < rows_here; ++r)
                    col_buf[static_cast<std::size_t>(r)] = A[(i0 + r) * lda + k];
                svfloat32_t a_col = svld1(pg_rows, col_buf.data());
                svfloat32_t b_row = svld1(pg_cols, B + k * ldb + j0);
                svmopa_za32_f32_m(0, pg_rows, pg_cols, a_col, b_row);
            }
            for (std::int64_t r = 0; r < svl && i0 + r < M; ++r)
                svst1_hor_za32(0, static_cast<std::uint32_t>(r), pg_cols,
                                C + (i0 + r) * ldc + j0);
        }
    }
}

__arm_locally_streaming __arm_new("za")
inline void sme_gemm_naive_f64_impl(const double* __restrict A, const double* __restrict B,
                                     double* __restrict C, std::int64_t M, std::int64_t N,
                                     std::int64_t K, std::int64_t lda, std::int64_t ldb,
                                     std::int64_t ldc) {
    const std::int64_t svl = static_cast<std::int64_t>(svcntsd());
    std::vector<double> col_buf(static_cast<std::size_t>(svl));

    for (std::int64_t i0 = 0; i0 < M; i0 += svl) {
        svbool_t pg_rows       = svwhilelt_b64(static_cast<std::uint64_t>(i0),
                                                static_cast<std::uint64_t>(M));
        const std::int64_t rows_here = std::min<std::int64_t>(svl, M - i0);
        for (std::int64_t j0 = 0; j0 < N; j0 += svl) {
            svbool_t pg_cols = svwhilelt_b64(static_cast<std::uint64_t>(j0),
                                              static_cast<std::uint64_t>(N));
            svzero_za();
            for (std::int64_t k = 0; k < K; ++k) {
                for (std::int64_t r = 0; r < rows_here; ++r)
                    col_buf[static_cast<std::size_t>(r)] = A[(i0 + r) * lda + k];
                svfloat64_t a_col = svld1(pg_rows, col_buf.data());
                svfloat64_t b_row = svld1(pg_cols, B + k * ldb + j0);
                svmopa_za64_f64_m(0, pg_rows, pg_cols, a_col, b_row);
            }
            for (std::int64_t r = 0; r < svl && i0 + r < M; ++r)
                svst1_hor_za64(0, static_cast<std::uint32_t>(r), pg_cols,
                                C + (i0 + r) * ldc + j0);
        }
    }
}

#endif  // __ARM_FEATURE_SME

// ============================================================================
// Kernel 2: gemm_sme_reordered — pack A panel once per i0-tile, reuse
// ============================================================================

/**
 * @brief SME GEMM that packs the A(i0..i0+SVL, 0..K) panel once per i0-tile
 *        into a contiguous buffer, then reuses it across every j0-tile.
 *
 * Loop structure:
 *   for i0 in steps of SVL:
 *     pack A(i0..i0+SVL, 0..K) → packed[k*SVL + r] = A(i0+r, k)   (once)
 *     for j0 in steps of SVL:
 *       zero ZA
 *       for k in [0, K):
 *         a_col = contiguous load from packed[k*SVL..]
 *         b_row = contiguous load from B(k, j0..)
 *         ZA += outer(a_col, b_row)
 *       store ZA rows to C(i0.., j0..)
 *
 * This is the direct analogue of the AVX2/NEON/SVE "i-k-j" reordering: it
 * eliminates the O(N/SVL) redundant A-gathering that gemm_sme_naive pays,
 * by doing the (currently unavoidable, see file header) scalar-gather pass
 * exactly once per i0-tile instead of once per (i0,j0)-tile pair.
 *
 * Measured (Apple M4 Max, single-threaded, f32): 218-380 GFLOP/s for
 * N in [256, 1024] — the highest CPU throughput in this repo by a wide
 * margin. Degrades once the packed panel (SVL*K*sizeof(T) bytes) exceeds
 * L1/L2 — see gemm_sme_blocked for the fix.
 *
 * Falls back to gemm_sve_reordered on non-SME targets.
 */
template <typename T>
void gemm_sme_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __ARM_FEATURE_SME
    gemm_sve_reordered(A, B, C);
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sme_reordered: T must be float or double");

    const std::int64_t M   = static_cast<std::int64_t>(A.rows());
    const std::int64_t K   = static_cast<std::int64_t>(A.cols());
    const std::int64_t N   = static_cast<std::int64_t>(B.cols());
    const std::int64_t lda = K;
    const std::int64_t ldb = N;
    const std::int64_t ldc = N;

    assert(static_cast<std::int64_t>(B.rows()) == K && static_cast<std::int64_t>(C.rows()) == M &&
           static_cast<std::int64_t>(C.cols()) == N);
    C.zero();

    if constexpr (sizeof(T) == 4) {
        sme_gemm_reordered_f32_impl(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);
    } else {
        sme_gemm_reordered_f64_impl(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);
    }
#endif
}

#ifdef __ARM_FEATURE_SME

__arm_locally_streaming __arm_new("za")
inline void sme_gemm_reordered_f32_impl(const float* __restrict A, const float* __restrict B,
                                         float* __restrict C, std::int64_t M, std::int64_t N,
                                         std::int64_t K, std::int64_t lda, std::int64_t ldb,
                                         std::int64_t ldc) {
    const std::int64_t svl = static_cast<std::int64_t>(svcntsw());
    std::vector<float> packed(static_cast<std::size_t>(svl * K));  // packed[k*svl + r] = A(i0+r, k)

    for (std::int64_t i0 = 0; i0 < M; i0 += svl) {
        svbool_t pg_rows = svwhilelt_b32(static_cast<std::uint32_t>(i0),
                                          static_cast<std::uint32_t>(M));
        const std::int64_t rows_here = std::min<std::int64_t>(svl, M - i0);
        for (std::int64_t k = 0; k < K; ++k)
            for (std::int64_t r = 0; r < rows_here; ++r)
                packed[static_cast<std::size_t>(k * svl + r)] = A[(i0 + r) * lda + k];

        for (std::int64_t j0 = 0; j0 < N; j0 += svl) {
            svbool_t pg_cols = svwhilelt_b32(static_cast<std::uint32_t>(j0),
                                              static_cast<std::uint32_t>(N));
            svzero_za();
            for (std::int64_t k = 0; k < K; ++k) {
                svfloat32_t a_col = svld1(pg_rows, packed.data() + k * svl);
                svfloat32_t b_row = svld1(pg_cols, B + k * ldb + j0);
                svmopa_za32_f32_m(0, pg_rows, pg_cols, a_col, b_row);
            }
            for (std::int64_t r = 0; r < svl && i0 + r < M; ++r)
                svst1_hor_za32(0, static_cast<std::uint32_t>(r), pg_cols,
                                C + (i0 + r) * ldc + j0);
        }
    }
}

__arm_locally_streaming __arm_new("za")
inline void sme_gemm_reordered_f64_impl(const double* __restrict A, const double* __restrict B,
                                         double* __restrict C, std::int64_t M, std::int64_t N,
                                         std::int64_t K, std::int64_t lda, std::int64_t ldb,
                                         std::int64_t ldc) {
    const std::int64_t svl = static_cast<std::int64_t>(svcntsd());
    std::vector<double> packed(static_cast<std::size_t>(svl * K));

    for (std::int64_t i0 = 0; i0 < M; i0 += svl) {
        svbool_t pg_rows = svwhilelt_b64(static_cast<std::uint64_t>(i0),
                                          static_cast<std::uint64_t>(M));
        const std::int64_t rows_here = std::min<std::int64_t>(svl, M - i0);
        for (std::int64_t k = 0; k < K; ++k)
            for (std::int64_t r = 0; r < rows_here; ++r)
                packed[static_cast<std::size_t>(k * svl + r)] = A[(i0 + r) * lda + k];

        for (std::int64_t j0 = 0; j0 < N; j0 += svl) {
            svbool_t pg_cols = svwhilelt_b64(static_cast<std::uint64_t>(j0),
                                              static_cast<std::uint64_t>(N));
            svzero_za();
            for (std::int64_t k = 0; k < K; ++k) {
                svfloat64_t a_col = svld1(pg_rows, packed.data() + k * svl);
                svfloat64_t b_row = svld1(pg_cols, B + k * ldb + j0);
                svmopa_za64_f64_m(0, pg_rows, pg_cols, a_col, b_row);
            }
            for (std::int64_t r = 0; r < svl && i0 + r < M; ++r)
                svst1_hor_za64(0, static_cast<std::uint32_t>(r), pg_cols,
                                C + (i0 + r) * ldc + j0);
        }
    }
}

#endif  // __ARM_FEATURE_SME

// ============================================================================
// Kernel 3: gemm_sme_blocked — K-tiled panel pack, ZA reloaded across tiles
// ============================================================================

/**
 * @brief SME GEMM with K-blocking so the packed A panel stays cache-resident
 *        regardless of K, at the cost of extra C read-modify-write traffic.
 *
 * Loop structure:
 *   for i0 in steps of SVL:
 *     for k_blk in steps of kSmeTileK:
 *       pack A(i0..i0+SVL, k_blk..k_blk+tk) → small buffer (bounded size)
 *       for j0 in steps of SVL:
 *         if k_blk == 0: zero ZA
 *         else:          load existing partial C(i0.., j0..) into ZA
 *                         (svld1_hor_za — resumes accumulation in hardware)
 *         for k in [0, tk): ZA += outer(packed[..], B(k_blk+k, j0..))
 *         store ZA rows back to C(i0.., j0..)
 *
 * gemm_sme_reordered packs the *entire* A(i0.., :) row-panel (SVL*K
 * elements) up front; for large K this exceeds L1 (and eventually L2),
 * causing repeated cache spills as it's re-read once per j0-tile.
 * gemm_sme_blocked instead packs only SVL*kSmeTileK elements at a time
 * (16 KB at SVL=16 f32, kSmeTileK=256 — comfortably L1-resident on any
 * known SME implementation) and pays for that bound with one extra
 * load+store of each C tile per k-block (K/kSmeTileK round-trips instead
 * of one).
 *
 * Measured (Apple M4 Max, single-threaded, f32): matches gemm_sme_reordered
 * for N <= 1024 (where the unbounded panel already fits cache) and wins
 * clearly once it doesn't: 254 vs 210 GFLOP/s at N=2048; 172 vs 127 GFLOP/s
 * at N=4096.
 *
 * Falls back to gemm_sve_blocked on non-SME targets.
 */
template <typename T>
void gemm_sme_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __ARM_FEATURE_SME
    gemm_sve_blocked(A, B, C);
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sme_blocked: T must be float or double");

    const std::int64_t M   = static_cast<std::int64_t>(A.rows());
    const std::int64_t K   = static_cast<std::int64_t>(A.cols());
    const std::int64_t N   = static_cast<std::int64_t>(B.cols());
    const std::int64_t lda = K;
    const std::int64_t ldb = N;
    const std::int64_t ldc = N;

    assert(static_cast<std::int64_t>(B.rows()) == K && static_cast<std::int64_t>(C.rows()) == M &&
           static_cast<std::int64_t>(C.cols()) == N);
    C.zero();

    if constexpr (sizeof(T) == 4) {
        sme_gemm_blocked_f32_impl(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);
    } else {
        sme_gemm_blocked_f64_impl(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);
    }
#endif
}

#ifdef __ARM_FEATURE_SME

__arm_locally_streaming __arm_new("za")
inline void sme_gemm_blocked_f32_impl(const float* __restrict A, const float* __restrict B,
                                       float* __restrict C, std::int64_t M, std::int64_t N,
                                       std::int64_t K, std::int64_t lda, std::int64_t ldb,
                                       std::int64_t ldc) {
    const std::int64_t svl = static_cast<std::int64_t>(svcntsw());
    const std::int64_t tile_k = static_cast<std::int64_t>(kSmeTileK);
    std::vector<float> packed(static_cast<std::size_t>(svl * tile_k));

    for (std::int64_t i0 = 0; i0 < M; i0 += svl) {
        svbool_t pg_rows = svwhilelt_b32(static_cast<std::uint32_t>(i0),
                                          static_cast<std::uint32_t>(M));
        const std::int64_t rows_here = std::min<std::int64_t>(svl, M - i0);

        for (std::int64_t k_blk = 0; k_blk < K; k_blk += tile_k) {
            const std::int64_t tk = std::min<std::int64_t>(tile_k, K - k_blk);
            for (std::int64_t k = 0; k < tk; ++k)
                for (std::int64_t r = 0; r < rows_here; ++r)
                    packed[static_cast<std::size_t>(k * svl + r)] = A[(i0 + r) * lda + (k_blk + k)];

            for (std::int64_t j0 = 0; j0 < N; j0 += svl) {
                svbool_t pg_cols = svwhilelt_b32(static_cast<std::uint32_t>(j0),
                                                  static_cast<std::uint32_t>(N));
                if (k_blk == 0) {
                    svzero_za();
                } else {
                    for (std::int64_t r = 0; r < svl && i0 + r < M; ++r)
                        svld1_hor_za32(0, static_cast<std::uint32_t>(r), pg_cols,
                                        C + (i0 + r) * ldc + j0);
                }
                for (std::int64_t k = 0; k < tk; ++k) {
                    svfloat32_t a_col = svld1(pg_rows, packed.data() + k * svl);
                    svfloat32_t b_row = svld1(pg_cols, B + (k_blk + k) * ldb + j0);
                    svmopa_za32_f32_m(0, pg_rows, pg_cols, a_col, b_row);
                }
                for (std::int64_t r = 0; r < svl && i0 + r < M; ++r)
                    svst1_hor_za32(0, static_cast<std::uint32_t>(r), pg_cols,
                                    C + (i0 + r) * ldc + j0);
            }
        }
    }
}

__arm_locally_streaming __arm_new("za")
inline void sme_gemm_blocked_f64_impl(const double* __restrict A, const double* __restrict B,
                                       double* __restrict C, std::int64_t M, std::int64_t N,
                                       std::int64_t K, std::int64_t lda, std::int64_t ldb,
                                       std::int64_t ldc) {
    const std::int64_t svl = static_cast<std::int64_t>(svcntsd());
    const std::int64_t tile_k = static_cast<std::int64_t>(kSmeTileK);
    std::vector<double> packed(static_cast<std::size_t>(svl * tile_k));

    for (std::int64_t i0 = 0; i0 < M; i0 += svl) {
        svbool_t pg_rows = svwhilelt_b64(static_cast<std::uint64_t>(i0),
                                          static_cast<std::uint64_t>(M));
        const std::int64_t rows_here = std::min<std::int64_t>(svl, M - i0);

        for (std::int64_t k_blk = 0; k_blk < K; k_blk += tile_k) {
            const std::int64_t tk = std::min<std::int64_t>(tile_k, K - k_blk);
            for (std::int64_t k = 0; k < tk; ++k)
                for (std::int64_t r = 0; r < rows_here; ++r)
                    packed[static_cast<std::size_t>(k * svl + r)] = A[(i0 + r) * lda + (k_blk + k)];

            for (std::int64_t j0 = 0; j0 < N; j0 += svl) {
                svbool_t pg_cols = svwhilelt_b64(static_cast<std::uint64_t>(j0),
                                                  static_cast<std::uint64_t>(N));
                if (k_blk == 0) {
                    svzero_za();
                } else {
                    for (std::int64_t r = 0; r < svl && i0 + r < M; ++r)
                        svld1_hor_za64(0, static_cast<std::uint32_t>(r), pg_cols,
                                        C + (i0 + r) * ldc + j0);
                }
                for (std::int64_t k = 0; k < tk; ++k) {
                    svfloat64_t a_col = svld1(pg_rows, packed.data() + k * svl);
                    svfloat64_t b_row = svld1(pg_cols, B + (k_blk + k) * ldb + j0);
                    svmopa_za64_f64_m(0, pg_rows, pg_cols, a_col, b_row);
                }
                for (std::int64_t r = 0; r < svl && i0 + r < M; ++r)
                    svst1_hor_za64(0, static_cast<std::uint32_t>(r), pg_cols,
                                    C + (i0 + r) * ldc + j0);
            }
        }
    }
}

#endif  // __ARM_FEATURE_SME

// ---------------------------------------------------------------------------
// Convenience alias: gemm_sme → gemm_sme_blocked
// ---------------------------------------------------------------------------
template <typename T>
inline void gemm_sme(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    gemm_sme_blocked(A, B, C);
}

}  // namespace hpc::gemm
