#pragma once

/**
 * @file sme.hpp
 * @brief ARM SME (Scalable Matrix Extension) GEMM — one packed, multi-tile kernel.
 *
 * ============================================================
 *  Outer products instead of FMA
 * ============================================================
 *
 * The AVX2, AVX-512, NEON and SVE kernels compute C with vector FMA:
 * broadcast one scalar, multiply it against a vector, add into a vector that
 * holds part of a row of C. SME's FMOPA instead takes a column vector `a` and
 * a row vector `b` (SVL elements each) and adds their whole SVL×SVL outer
 * product into ZA, a 2-D accumulator array:
 *
 *   ZA[r][c] += a[r] * b[c]     for r, c in [0, SVL)
 *
 * With a[k] = A(i0+r, k) and b[k] = B(k, j0+c), K of these instructions
 * produce an SVL×SVL tile of C — the same primitive as GPU Tensor Cores.
 * SVL (the streaming vector length) is read at runtime with svcntsw() /
 * svcntsd(): 16 f32 / 8 f64 on Apple M4.
 *
 *
 * ============================================================
 *  Streaming mode
 * ============================================================
 *
 * FMOPA, loads/stores of ZA and `zero {za}` are only legal in streaming SVE
 * mode. `__arm_locally_streaming` makes Clang wrap a function in SMSTART /
 * SMSTOP; `__arm_new("za")` gives it its own ZA state (any live caller ZA
 * state is saved, and ZA starts zeroed).
 *
 * Gather loads are not legal in streaming mode (Clang: "builtin can only be
 * called from a non-streaming function"), so the strided A column the outer
 * product needs cannot be loaded directly — A has to be packed.
 *
 *
 * ============================================================
 *  The kernel — gemm_sme
 * ============================================================
 *
 * A straightforward SME kernel — one ZA tile, B read in place — peaks at
 * ~380 GFLOP/s f32 on one M4 Max core, against ~1650 GFLOP/s for
 * Accelerate on the same core. gemm_sme (one template for f32 and f64)
 * closes most of that gap with four design choices:
 *
 *  1. All ZA tiles in use. With one tile, every FMOPA waits for the previous
 *     one's accumulator, so the loop runs at FMOPA latency, not throughput.
 *     ZA holds 4 f32 tiles (SVL×SVL each) or 8 f64 tiles; the micro-kernel
 *     keeps all of them busy with independent outer products:
 *
 *       f32: 2×2 tiles → (2·SVL)×(2·SVL) = 32×32 C block on a 512-bit SVL
 *            per k: 2 A vectors + 2 B vectors → 4 FMOPA
 *       f64: 2×4 tiles → (2·SVL)×(4·SVL) = 16×32 C block
 *            per k: 2 A vectors + 4 B vectors → 8 FMOPA
 *
 *     Each loaded vector feeds 2 (or 4) FMOPAs instead of 1. kCols<T> (2 or
 *     4) is the only shape difference between the types. A 2×2 layout for
 *     f64 too (4 of 8 tiles) measured 9% slower at N=4096.
 *
 *  2. A and B packed, GotoBLAS-style cache blocking (loop nest and why it
 *     has this order: "The algorithm" below). The packed B strip (kc×nr) and
 *     A strip (kc×mr) are both read with unit stride.
 *
 *     A must be packed: FMOPA needs a column of A as one vector and
 *     streaming mode has no gather loads. Read in place, the column
 *     A(i0..i0+31, k) is 32 values K·sizeof(T) apart — 32 cache lines per
 *     k step. pack_a uses the same strip layout as B, transposed:
 *
 *       dst[s·mr·kc + k·mr + r] = A(s·mr + r, k)
 *
 *       ┌ strip 0: rows 0..31 ┬ strip 1: rows 32..63 ┬ strip 2 ┬ strip 3 ┐
 *       │ k = 0..1023, 128 B  │ k = 0..1023          │         │         │
 *       └────── 128 KB ───────┴─────── 128 KB ───────┴─────────┴─────────┘
 *       (f32 A block, Mc × Kc = 128 × 1024, 512 KB)
 *
 *     Each k step then reads one full 128 B line (f64: mr = 16 doubles, also
 *     128 B) with one svld1_x2, consecutive k steps go to consecutive cache
 *     sets, and the strip streams through the prefetcher like B.
 *
 *     B is packed to keep each strip cached. The packed panel (f32: 1024 ×
 *     4096 floats, 16 MB) is not stored row-major but strip by strip, each
 *     strip contiguous:
 *
 *       dst[s·nr·kc + k·nr + c] = B(k, s·nr + c)
 *
 *       ┌ strip 0: cols 0..31 ┬ strip 1: cols 32..63 ┬ ... ┬ strip 127 ┐
 *       │ row 0..1023, 128 B  │ row 0..1023          │     │           │
 *       └────── 128 KB ───────┴─────── 128 KB ───────┴ ... ┴───────────┘
 *
 *     Reuse happens at two levels (f32):
 *
 *       what             size     reused by
 *       one B strip      128 KB   the 4 ir iterations of a jr step (Mc/mr)
 *       whole B panel    16 MB    every ic block (M/Mc of them)
 *
 *     The panel is one contiguous block, so it is limited only by capacity
 *     (it fills the 16 MB P-cluster L2; this is why kSmeNc stops at 4096).
 *     The strip is where layout matters. Read in place, strip row k is at
 *     B + k·N + jr, so its 1,024 rows are N·sizeof(T) bytes apart instead
 *     of 128 B.
 *
 *     A cache splits an address into tag | set index | line offset. The
 *     offset is bits 0–6 (128 B lines); the set index is the next bits; a
 *     line may only live in one of the few ways of its own set. A stride of
 *     2^14 (N=4096, f32) never changes bits 0–13, so every row of the strip
 *     lands in the same set:
 *
 *       row 0     base + 0·16384      set X
 *       row 1     base + 1·16384      set X
 *       ...
 *       row 1023  base + 1023·16384   set X   → 1,024 lines, a few ways
 *
 *     Packed, row k is at strip + k·128 and each row takes the next set, so
 *     the strip spreads over all sets. At N=4000 the stride is 16,000 B =
 *     125 lines; 125 is odd, so the set index walks every set before
 *     repeating and the strip also fits.
 *
 *     Pointer chase on an M4 Max P-core (1,024 lines, random order, ns per
 *     load): 128 B apart 0.78, 16,000 B apart 2.4 (TLB), 16 KB apart 24.4.
 *     At 16 KB apart only 8 lines stay in L1 (128 KB, 8-way, 128 sets) and
 *     512 but not 1,024 stay in L2, so the strip does not survive between
 *     ir iterations at any level. In gemm_sme this costs −55% at N=4096,
 *     −12% at N=2048 (an 8 KB stride uses twice as many sets), −7% at N=4000. See
 *     src/gemm/README.md, "Why A and B are packed".
 *
 *  3. Packing outside streaming mode. Scalar/NEON code is slow in streaming
 *     mode on M4, so packing runs in the ordinary (non-streaming) driver and
 *     only the macro-kernel runs streaming — one SMSTART/SMSTOP per
 *     Mc×Nc×Kc block. A is transposed with NEON 4×4 / 2×2 in-register
 *     transposes, ~10% faster overall than a scalar packing loop.
 *
 *  4. SME2 multi-vector loads. The A and B operands of one k step are each
 *     fetched with LD1W/LD1D {z0-z1} (svld1_x2), halving the load
 *     instruction count. SME2 is therefore required (HPC_HAS_SME checks
 *     __ARM_FEATURE_SME2 too).
 *
 * Pitfall: every streaming helper called from the ZA-owning macro-kernel
 * needs a ZA attribute (__arm_inout("za") etc.). Without one it is
 * "private-ZA": Clang won't inline it and wraps each call in a lazy ZA save
 * (TPIDR2 + smstart za) — inside the k loop that cuts throughput ~5×.
 *
 * Single-threaded. Compare against Accelerate with VECLIB_MAXIMUM_THREADS=1.
 *
 *
 * ============================================================
 *  The algorithm
 * ============================================================
 *
 *   for jc (Nc = 4096)      pack_b → 16 MB B panel        normal mode
 *    for pc (Kc = 1024)
 *     for ic (Mc = 128)     pack_a → 512 KB A block       normal mode
 *      macro_kernel:                                      streaming mode
 *       for jr              one B strip, 128 KB
 *        for ir             4 A strips, each re-reads the same B strip
 *         C tile → ZA
 *         for k = 0..1023   2 A + 2 B vectors → 4 FMOPA
 *         ZA → C tile
 *
 *  1. Setup. s = svcntsw()/svcntsd() (16 f32 / 8 f64); micro-tile
 *     mr = 2s, nr = kCols·s (32×32 f32, 16×32 f64); allocate a_pack
 *     (Mc×Kc) and b_pack (Kc×Nc).
 *  2. pack_b (per jc, pc): strip by strip, each strip's rows contiguous;
 *     partial strips zero-padded to nr.
 *  3. pack_a (per ic): transpose into strips of mr rows so that column
 *     A(i0..i0+mr−1, k) is one contiguous vector; NEON 4×4 / 2×2
 *     transposes; rows zero-padded to mr.
 *  4. macro_kernel (__arm_locally_streaming __arm_new("za")), for each
 *     jr, ir:
 *       - pc == 0: svzero_za(); otherwise move_c loads the partial sums of
 *         C into ZA, so sums carry across pc blocks.
 *       - k loop: svld1_x2 for A, svld1_x2 for B (f64: a second one for
 *         B columns 2–3), then 4 (f32) or 8 (f64) FMOPAs into separate
 *         tiles.
 *       - move_c stores ZA back to C.
 *  5. Edges: thanks to the zero padding, FMOPAs always run with an
 *     all-true predicate; only the C transfers into and out of ZA are
 *     predicated (columns) and bounded (rows).
 *
 * Why this loop order. Each level holds one operand fixed while the levels
 * inside it reuse it (f32, N=4096):
 *
 *   loop  trips  fixed during it       size     reused by        lives in
 *   k     1024   C micro-tile          4 KB     1,024 k steps    ZA
 *   ir    4      B strip kc×nr         128 KB   4 A strips       cache
 *   jr    128    A block mc×kc         512 KB   128 B strips     L2
 *   ic    32     B panel kc×nc         16 MB    32 A blocks      L2 / SLC
 *   pc    4      — (splits K)
 *   jc    1      — (splits N)
 *
 *   - k innermost: C stays in ZA for the whole k loop and touches memory
 *     once per pc pass. Per k step f32 loads 256 B for 2,048 flops
 *     (8 flop/B); f64 loads 384 B for 1,024 flops (~2.7 flop/B).
 *   - ir inside jr: a B strip is reused after only one A strip (128 KB)
 *     has passed through the cache, and an A strip after ~640 KB. Swapped,
 *     each A strip would walk the whole 16 MB B panel before the next one
 *     reuses it: the same bytes read, but at a reuse distance of the whole
 *     L2.
 *   - ic inside pc: B is packed once per (jc, pc) and serves all M/Mc A
 *     blocks. A is the operand repacked often, which is cheap: each A
 *     element is packed N/Nc times, each B element once — O(MK + KN)
 *     against O(MNK) flops.
 *   - pc bounds the strips and panels in K. The cost is one C round trip
 *     through ZA per pc (K/Kc times), so Kc is large.
 *   - jc caps the B panel at the 16 MB P-cluster L2; at N ≤ 4096 it runs
 *     once.
 *
 * Blocks are sized for L2, not L1 as in classic GotoBLAS, because on M4 the
 * SME unit is shared by the P-cluster and reads operands from L2.
 *
 * f64: mr = 16, nr = 32, so 8 ir steps per jr, a 256 KB B strip, a 1 MB A
 * block and a 32 MB B panel.
 *
 *
 * ============================================================
 *  Build notes and availability
 * ============================================================
 *
 * Apple Silicon implements streaming SVE (via SME) but no non-streaming SVE.
 * `-march=armv9-a+sme2` compiles, but the binary SIGILLs on the first
 * cntd/cntw Clang emits in a function prologue to size the ZA save buffer:
 * those are non-streaming SVE instructions. `-mcpu=apple-m4` avoids them.
 * Because a compile-only check cannot catch this, the CMake SME probe
 * (HPC_ENABLE_SME) compiles and runs a small SME program at configure time.
 *
 * svcntsw()/svcntsd() compile to RDSVL, an SME instruction that is legal
 * outside streaming mode, so the non-streaming driver can size its packing
 * buffers with them.
 *
 * SME2 is available on Apple M4 / M4 Pro / M4 Max (512-bit SVL); not on
 * Apple M1–M3, AWS Graviton3/4, Fujitsu A64FX or x86. HPC_HAS_SME
 * (hpc/isa.hpp) follows __ARM_FEATURE_SME && __ARM_FEATURE_SME2; where it is
 * 0, gemm_sme is declared `= delete`.
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
                    // r0..r3 = rows a..d. Swap single floats, then 2-float halves.
                    const float32x4_t t0 = vtrn1q_f32(r0, r1);  // a0 b0 a2 b2
                    const float32x4_t t1 = vtrn2q_f32(r0, r1);  // a1 b1 a3 b3
                    const float32x4_t t2 = vtrn1q_f32(r2, r3);  // c0 d0 c2 d2
                    const float32x4_t t3 = vtrn2q_f32(r2, r3);  // c1 d1 c3 d3
                    vst1q_f32(out, vcombine_f32(vget_low_f32(t0), vget_low_f32(t2)));
                    vst1q_f32(out + mr, vcombine_f32(vget_low_f32(t1), vget_low_f32(t3)));
                    vst1q_f32(out + 2 * mr, vcombine_f32(vget_high_f32(t0), vget_high_f32(t2)));
                    vst1q_f32(out + 3 * mr, vcombine_f32(vget_high_f32(t1), vget_high_f32(t3)));
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
