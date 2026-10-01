#pragma once

/**
 * @file sme.hpp
 * @brief ARM SME (Scalable Matrix Extension) GEMM — one packed, multi-tile kernel.
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
 * gemm_cuda_wmma) use, just for CPU SIMD.
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
 * Gather loads are NOT legal in streaming mode (Clang: "builtin can only be
 * called from a non-streaming function"), so the strided A column the outer
 * product needs cannot be loaded directly — A has to be packed.
 *
 *
 * ============================================================
 *  The kernel — gemm_sme
 * ============================================================
 *
 * An earlier version of this file had three kernels (naive / reordered /
 * blocked) that all used a single ZA tile and an unpacked B. They peaked at
 * ~380 GFLOP/s f32 on one M4 Max core while Accelerate reached ~1650 GFLOP/s
 * on the same core, so they were replaced by this one kernel (one template
 * for f32 and f64), which fixes the four reasons for that gap:
 *
 *  1. All ZA tiles in use. A single tile serialises every FMOPA on the
 *     previous one's accumulator, so the loop runs at FMOPA *latency*, not
 *     throughput. ZA holds 4 f32 tiles (SVL×SVL each) or 8 f64 tiles; the
 *     micro-kernel keeps all of them busy with independent outer products:
 *
 *       f32: 2×2 tiles → (2·SVL)×(2·SVL) = 32×32 C block on a 512-bit SVL
 *            per k: 2 A vectors + 2 B vectors → 4 FMOPA
 *       f64: 2×4 tiles → (2·SVL)×(4·SVL) = 16×32 C block
 *            per k: 2 A vectors + 4 B vectors → 8 FMOPA
 *
 *     Each loaded vector now feeds 2 (or 4) FMOPAs instead of 1. One template
 *     serves both types; kCols<T> (2 or 4) is the only shape difference.
 *     Using just 2×2 for f64 too (4 of 8 tiles) measured 9% slower at N=4096.
 *
 *  2. A *and* B packed, GotoBLAS-style cache blocking. Loop nest
 *       jc (kSmeNc) → pc (kSmeKc) → pack B panel → ic (kSmeMc) → pack A block
 *       → macro-kernel: jr (nr) → ir (mr) → k
 *     The packed B strip is kc×nr and the packed A strip kc×mr, both read
 *     with unit stride, so the k loop touches two contiguous streams instead
 *     of one B row per k spaced ldb apart (which is what made the old kernels
 *     fall off at N ≥ 2048). Partial C sums across pc blocks are carried by
 *     loading C into ZA before the k loop.
 *
 *  3. Packing outside streaming mode. Scalar/NEON code is slow in streaming
 *     mode on M4, so packing is ordinary C++ in the non-streaming driver; only
 *     the macro-kernel (pure ZA/vector work) runs streaming. One SMSTART/
 *     SMSTOP per Mc×Nc×Kc block is negligible. A is transposed with NEON
 *     4×4 / 2×2 in-register transposes — ~10% faster overall than a scalar
 *     packing loop.
 *
 * Pitfall: every streaming helper called from the ZA-owning macro-kernel
 * needs a ZA attribute (__arm_inout("za") etc.). Without one it is
 * "private-ZA": Clang won't inline it and wraps each call in a lazy ZA save
 * (TPIDR2 + smstart za) — inside the k loop that cut throughput ~5×.
 *
 *  4. SME2 multi-vector loads. The A and B operands of one k step are each
 *     fetched with LD1W/LD1D {z0-z1} (svld1_x2) — one instruction per 2
 *     vectors, halving the load instruction count. SME2 is therefore required (HPC_HAS_SME checks
 *     __ARM_FEATURE_SME2 too).
 *
 * Edges: packing zero-pads partial strips to full mr/nr, so FMOPAs always
 * run with an all-true predicate; only the C load/store into/out of ZA is
 * predicated (columns) and bounded (rows).
 *
 * Single-threaded. Compare against Accelerate with VECLIB_MAXIMUM_THREADS=1.
 *
 *
 * ============================================================
 *  Apple Silicon note
 * ============================================================
 *
 * Apple M4 (and M4 Pro/Max) is, as of 2026, essentially the only shipping
 * SME2 hardware widely available to individual developers.
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
 * svcntsw()/svcntsd() compile to RDSVL, an SME (not SVE) instruction that is
 * legal outside streaming mode, so the non-streaming driver can size its
 * packing buffers with them.
 *
 *
 * ============================================================
 *  Hardware availability
 * ============================================================
 *
 * SME is available on:
 *   Apple M4 / M4 Pro / M4 Max (SME2, 512-bit SVL, f32/f64/bf16/f16/int8/int16)
 *   Future Armv9.2+ server/mobile SoCs implementing FEAT_SME
 *   NOT on: Apple M1/M2/M3, AWS Graviton3/4, Fujitsu A64FX, x86 (Intel/AMD)
 *
 * The header detects __ARM_FEATURE_SME at compile time (HPC_HAS_SME in hpc/isa.hpp).
 * If the ISA is absent gemm_sme is declared `= delete`: calling it is a
 * compile-time error, never a silent substitution of a slower kernel.
 */

#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"

#if HPC_HAS_SME
    #include <arm_neon.h>
    #include <arm_sme.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace hpc::gemm {

// ============================================================================
// Cache-blocking parameters (elements)
//
// On M4 the SME unit is shared by a P-core cluster and streams operands from
// L2, so blocks are sized for L2 rather than L1:
//   packed B panel  kSmeKc × kSmeNc  (f32: 1024 × 4096 × 4B = 16 MB)
//   packed A block  kSmeMc × kSmeKc  (f32:  128 × 1024 × 4B = 512 KB)
// kSmeKc also sets how often a C block is reloaded into ZA (once per Kc).
// Values picked by a sweep on M4 Max (Mc 64–512, Kc 256–2048, Nc 512–4096):
// large Kc/Nc win because they amortise C reloads and A repacking; the
// gain from Nc 2048 → 4096 shows only at N ≥ 4096 (+5%).
// ============================================================================

inline constexpr std::size_t kSmeMc = 128;
inline constexpr std::size_t kSmeKc = 1024;
inline constexpr std::size_t kSmeNc = 4096;

#if !HPC_HAS_SME

template <typename T>
void gemm_sme(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_FEATURE_SME not available on this target

#else

namespace sme_detail {

/// Pack an mc×kc block of row-major A into mr-row strips, column-major within
/// each strip: dst[s*mr*kc + k*mr + r] = A(s*mr + r, k), zero-padded to mr rows.
template <typename T>
inline void pack_a(const T* A, std::size_t lda, std::size_t mc, std::size_t kc, std::size_t mr,
                   T* dst) {
    for (std::size_t i0 = 0; i0 < mc; i0 += mr) {
        const std::size_t rows = std::min(mr, mc - i0);
        T* strip               = dst + i0 * kc;
        std::size_t r          = 0;
        // NEON in-register transpose: L×L tiles (L = 4 f32 / 2 f64 lanes),
        // ~2× faster than the scalar strided-store loop below.
        constexpr std::size_t L = 16 / sizeof(T);
        const std::size_t kc_v  = kc - kc % L;
        for (; r + L <= rows; r += L) {
            const T* src = A + (i0 + r) * lda;
            for (std::size_t k = 0; k < kc_v; k += L) {
                T* out = strip + k * mr + r;
                if constexpr (sizeof(T) == 4) {
                    const float32x4_t r0 = vld1q_f32(src + k);
                    const float32x4_t r1 = vld1q_f32(src + lda + k);
                    const float32x4_t r2 = vld1q_f32(src + 2 * lda + k);
                    const float32x4_t r3 = vld1q_f32(src + 3 * lda + k);
                    const float64x2_t t0 = vreinterpretq_f64_f32(vtrn1q_f32(r0, r1));
                    const float64x2_t t1 = vreinterpretq_f64_f32(vtrn2q_f32(r0, r1));
                    const float64x2_t t2 = vreinterpretq_f64_f32(vtrn1q_f32(r2, r3));
                    const float64x2_t t3 = vreinterpretq_f64_f32(vtrn2q_f32(r2, r3));
                    vst1q_f32(out, vreinterpretq_f32_f64(vtrn1q_f64(t0, t2)));
                    vst1q_f32(out + mr, vreinterpretq_f32_f64(vtrn1q_f64(t1, t3)));
                    vst1q_f32(out + 2 * mr, vreinterpretq_f32_f64(vtrn2q_f64(t0, t2)));
                    vst1q_f32(out + 3 * mr, vreinterpretq_f32_f64(vtrn2q_f64(t1, t3)));
                } else {
                    const float64x2_t r0 = vld1q_f64(src + k);
                    const float64x2_t r1 = vld1q_f64(src + lda + k);
                    vst1q_f64(out, vtrn1q_f64(r0, r1));
                    vst1q_f64(out + mr, vtrn2q_f64(r0, r1));
                }
            }
            for (std::size_t rr = r; rr < r + L; ++rr)
                for (std::size_t k = kc_v; k < kc; ++k)
                    strip[k * mr + rr] = A[(i0 + rr) * lda + k];
        }
        for (; r < rows; ++r) {
            const T* src = A + (i0 + r) * lda;
            for (std::size_t k = 0; k < kc; ++k)
                strip[k * mr + r] = src[k];
        }
        if (rows < mr)
            for (std::size_t k = 0; k < kc; ++k)
                std::fill(strip + k * mr + rows, strip + (k + 1) * mr, T{0});
    }
}

/// Pack a kc×nc panel of row-major B into nr-column strips, row-major within
/// each strip: dst[s*nr*kc + k*nr + c] = B(k, s*nr + c), zero-padded to nr cols.
template <typename T>
inline void pack_b(const T* B, std::size_t ldb, std::size_t kc, std::size_t nc, std::size_t nr,
                   T* dst) {
    for (std::size_t j0 = 0; j0 < nc; j0 += nr) {
        const std::size_t cols = std::min(nr, nc - j0);
        T* strip               = dst + j0 * kc;
        for (std::size_t k = 0; k < kc; ++k) {
            std::memcpy(strip + k * nr, B + k * ldb + j0, cols * sizeof(T));
            std::fill(strip + k * nr + cols, strip + (k + 1) * nr, T{0});
        }
    }
}

template <typename T>
inline constexpr bool kF32 = sizeof(T) == 4;

/// Micro-tile = 2 × kCols ZA tiles = (2·SVL)×(kCols·SVL) block of C, using
/// every tile: 2×2 za32 (f32) or 2×4 za64 (f64). Tile index = row·kCols + col.
template <typename T>
inline constexpr int kCols = kF32<T> ? 2 : 4;

/// ZA[Tile] += a ⊗ b.
template <typename T, int Tile, typename V>
[[gnu::always_inline]] inline void mopa(svbool_t pt, V a, V b) __arm_streaming __arm_inout("za") {
    if constexpr (kF32<T>)
        svmopa_za32_m(Tile, pt, pt, a, b);
    else
        svmopa_za64_m(Tile, pt, pt, a, b);
}

/// Copy rows [0, rows) of ZA tile `Tile` to C (Load=false) or from C (Load=true).
template <typename T, int Tile, bool Load>
[[gnu::always_inline]] inline void move_tile(T* c, std::int64_t ldc, std::int64_t rows,
                                             svbool_t pg) __arm_streaming __arm_inout("za") {
    for (std::int64_t r = 0; r < rows; ++r) {
        const auto slice = static_cast<std::uint32_t>(r);
        if constexpr (kF32<T> && Load)  svld1_hor_za32(Tile, slice, pg, c + r * ldc);
        if constexpr (kF32<T> && !Load) svst1_hor_za32(Tile, slice, pg, c + r * ldc);
        if constexpr (!kF32<T> && Load)  svld1_hor_za64(Tile, slice, pg, c + r * ldc);
        if constexpr (!kF32<T> && !Load) svst1_hor_za64(Tile, slice, pg, c + r * ldc);
    }
}

/// Move one column of tiles (top tile Col, bottom tile kCols+Col) between ZA and C.
template <typename T, int Col, bool Load>
[[gnu::always_inline]] inline void move_column(T* c, std::int64_t ldc, std::int64_t s,
                                               std::int64_t rows, std::int64_t cols) __arm_streaming
    __arm_inout("za") {
    const std::int64_t top = std::min(rows, s);
    const svbool_t pg      = kF32<T> ? svwhilelt_b32(Col * s, cols) : svwhilelt_b64(Col * s, cols);
    move_tile<T, Col, Load>(c + Col * s, ldc, top, pg);
    move_tile<T, kCols<T> + Col, Load>(c + s * ldc + Col * s, ldc, rows - top, pg);
}

/// Load (Load=true) or store the whole micro-tile accumulator.
template <typename T, bool Load>
[[gnu::always_inline]] inline void move_c(T* c, std::int64_t ldc, std::int64_t s, std::int64_t rows,
                                          std::int64_t cols) __arm_streaming __arm_inout("za") {
    move_column<T, 0, Load>(c, ldc, s, rows, cols);
    move_column<T, 1, Load>(c, ldc, s, rows, cols);
    if constexpr (kCols<T> == 4) {
        move_column<T, 2, Load>(c, ldc, s, rows, cols);
        move_column<T, 3, Load>(c, ldc, s, rows, cols);
    }
}

/// Macro-kernel (streaming): C[mc×nc] (+)= Apack[mc×kc] · Bpack[kc×nc].
template <typename T>
__arm_locally_streaming __arm_new("za")
void macro_kernel(const T* __restrict Ap, const T* __restrict Bp, T* __restrict C,
                  std::int64_t ldc, std::int64_t mc, std::int64_t nc, std::int64_t kc,
                  bool accumulate) {
    constexpr int kC      = kCols<T>;
    const std::int64_t s  = kF32<T> ? svcntsw() : svcntsd();
    const std::int64_t mr = 2 * s, nr = kC * s;
    const svbool_t pt     = kF32<T> ? svptrue_b32() : svptrue_b64();
    const svcount_t pn    = kF32<T> ? svptrue_c32() : svptrue_c64();

    for (std::int64_t jr = 0; jr < nc; jr += nr) {
        const std::int64_t cols = std::min(nr, nc - jr);
        const T* b_strip        = Bp + jr * kc;

        for (std::int64_t ir = 0; ir < mc; ir += mr) {
            const std::int64_t rows = std::min(mr, mc - ir);
            const T* a_strip        = Ap + ir * kc;
            T* c                    = C + ir * ldc + jr;

            if (accumulate)
                move_c<T, true>(c, ldc, s, rows, cols);
            else
                svzero_za();

            for (std::int64_t k = 0; k < kc; ++k) {
                // SME2 multi-vector loads: one instruction fetches 2 vectors.
                const auto a  = svld1_x2(pn, a_strip + k * mr);
                const auto b  = svld1_x2(pn, b_strip + k * nr);
                const auto a0 = svget2(a, 0), a1 = svget2(a, 1);
                mopa<T, 0>(pt, a0, svget2(b, 0));
                mopa<T, 1>(pt, a0, svget2(b, 1));
                mopa<T, kC + 0>(pt, a1, svget2(b, 0));
                mopa<T, kC + 1>(pt, a1, svget2(b, 1));
                if constexpr (kC == 4) {  // f64: columns 2–3 use the other 4 za64 tiles
                    const auto b2 = svld1_x2(pn, b_strip + k * nr + 2 * s);
                    mopa<T, 2>(pt, a0, svget2(b2, 0));
                    mopa<T, 3>(pt, a0, svget2(b2, 1));
                    mopa<T, 6>(pt, a1, svget2(b2, 0));
                    mopa<T, 7>(pt, a1, svget2(b2, 1));
                }
            }

            move_c<T, false>(c, ldc, s, rows, cols);
        }
    }
}

}  // namespace sme_detail

/**
 * @brief C = A · B on ARM SME — packed, cache-blocked, all ZA tiles in use.
 *
 * See the file header for the design. Single-threaded.
 */
template <typename T>
void gemm_sme(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sme: T must be float or double");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;

    // Micro-tile is mr × nr; RDSVL (svcnts*) is legal outside streaming mode.
    const std::size_t s  = sme_detail::kF32<T> ? svcntsw() : svcntsd();
    const std::size_t mr = 2 * s, nr = sme_detail::kCols<T> * s;
    auto round_up        = [](std::size_t x, std::size_t m) { return (x + m - 1) / m * m; };
    const std::size_t kc_max = std::min(kSmeKc, K);
    Matrix<T> a_pack(round_up(std::min(kSmeMc, M), mr), kc_max);  // 64-byte aligned scratch
    Matrix<T> b_pack(kc_max, round_up(std::min(kSmeNc, N), nr));

    for (std::size_t jc = 0; jc < N; jc += kSmeNc) {
        const std::size_t nc = std::min(kSmeNc, N - jc);
        for (std::size_t pc = 0; pc < K; pc += kSmeKc) {
            const std::size_t kc = std::min(kSmeKc, K - pc);
            sme_detail::pack_b(B.data() + pc * N + jc, N, kc, nc, nr, b_pack.data());
            for (std::size_t ic = 0; ic < M; ic += kSmeMc) {
                const std::size_t mc = std::min(kSmeMc, M - ic);
                sme_detail::pack_a(A.data() + ic * K + pc, K, mc, kc, mr, a_pack.data());
                sme_detail::macro_kernel<T>(a_pack.data(), b_pack.data(), C.data() + ic * N + jc,
                                            static_cast<std::int64_t>(N),
                                            static_cast<std::int64_t>(mc),
                                            static_cast<std::int64_t>(nc),
                                            static_cast<std::int64_t>(kc), pc > 0);
            }
        }
    }
}

#endif  // HPC_HAS_SME

}  // namespace hpc::gemm
