#pragma once

/**
 * @file sme.hpp
 * @brief ARM SME (Scalable Matrix Extension) GEMM in four steps: one ZA tile,
 *        all ZA tiles, cache blocking, packed B.
 *
 * ============================================================
 *  SME in brief
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
 * SVL is read at runtime with svcntsw() / svcntsd(): 16 f32 / 8 f64 on
 * Apple M4, where ZA holds 4 f32 or 8 f64 tiles of SVL×SVL.
 *
 * FMOPA, ZA loads/stores and `zero {za}` are only legal in streaming SVE
 * mode. `__arm_locally_streaming` makes Clang wrap a function in SMSTART /
 * SMSTOP; `__arm_new("za")` gives it its own ZA state (any live caller ZA
 * state is saved, and ZA starts zeroed). Gather loads are not legal in
 * streaming mode, so the strided column of A that FMOPA needs cannot be
 * loaded directly: every kernel here packs A ("Packing" below).
 *
 *
 * ============================================================
 *  The four kernels
 * ============================================================
 *
 *   kernel            ZA tiles   B          cache blocking
 *   gemm_sme_1tile    1          in place   none
 *   gemm_sme_tiles    all        in place   none
 *   gemm_sme_blocked  all        in place   GotoBLAS
 *   gemm_sme          all        packed     GotoBLAS
 *
 * All four pack A, so each step changes one thing. Each is one template for
 * f32 and f64; the last three share macro_kernel. All are single-threaded:
 * compare against Accelerate with VECLIB_MAXIMUM_THREADS=1 (~1650 GFLOP/s
 * f32, ~410 f64 on one M4 Max core).
 *
 * One M4 Max core, GFLOP/s, median of 3 runs:
 *
 *   f32                N=256   512  1024  2048  4000  4096
 *   gemm_sme_1tile       420   472   474   225   125   131
 *   gemm_sme_tiles       976  1461  1381   624   429   445
 *   gemm_sme_blocked     911  1400  1401  1249  1373   597
 *   gemm_sme             752  1283  1421  1438  1419  1372
 *
 *   f64                N=256   512  1024  2048  4000  4096
 *   gemm_sme_1tile       121   121   110    40    42    37
 *   gemm_sme_tiles       350   420   403   232   210   208
 *   gemm_sme_blocked     342   422   442   284   373   271
 *   gemm_sme             315   397   414   406   410   403
 *
 *  - 1tile → tiles, 2.3–3.4× f32 / 2.9–5.8× f64: FMOPA latency hidden
 *    ("All ZA tiles" below). At large N the one-tile kernel is also
 *    bandwidth bound, since each vector it loads feeds one FMOPA.
 *  - tiles → blocked, 2.0× f32 at N=2048 / 3.2× at N=4000: without
 *    blocking, all of A is re-read for every B strip and falls out of
 *    cache. N=4000 is as slow as N=4096, so this is not a stride effect;
 *    capacity is the likely cause.
 *  - blocked → gemm_sme: within 3% at N=4000, but 2.3× f32 / 1.5× f64 at
 *    N=4096 and 1.15× f32 at N=2048 — the power-of-two stride effect that
 *    packing B removes ("Packing" below).
 *  - At N ≤ 512 the kernels without pack_b are faster (+30% f32 at N=256):
 *    everything fits in cache, and gemm_sme pays for copying B.
 *
 *
 * ============================================================
 *  gemm_sme
 * ============================================================
 *
 * All ZA tiles. With one tile every FMOPA waits for the previous one's
 * accumulator, so the loop runs at FMOPA latency, not throughput. The
 * micro-kernel keeps every tile busy with independent outer products:
 *
 *   f32: 2×2 tiles → 32×32 C block, per k: 2 A + 2 B vectors → 4 FMOPA
 *   f64: 2×4 tiles → 16×32 C block, per k: 2 A + 4 B vectors → 8 FMOPA
 *
 * Each loaded vector feeds 2 (or 4) FMOPAs. kCols<T> (2 or 4) is the only
 * shape difference between the types; a 2×2 layout for f64 too (4 of 8
 * tiles) measured 9% slower at N=4096. The A and B vectors of one k step
 * are each fetched with one SME2 svld1_x2 (LD1W/LD1D {z0-z1}), halving the
 * load count; SME2 is therefore required (HPC_HAS_SME checks
 * __ARM_FEATURE_SME2 too).
 *
 * Cache blocking, GotoBLAS-style (f32 sizes):
 *
 *   for jc (Nc = 4096)      pack_b → 16 MB B panel        normal mode
 *    for pc (Kc = 1024)
 *     for ic (Mc = 128)     pack_a → 512 KB A block       normal mode
 *      macro_kernel:                                      streaming mode
 *       for jr              one B strip, 128 KB
 *        for ir             4 A strips, each re-reads the same B strip
 *         C tile → ZA       (zeroed when pc == 0, else loaded from C)
 *         for k = 0..1023   2 A + 2 B vectors → 4 FMOPA
 *         ZA → C tile
 *
 * Each level holds one operand fixed while the levels inside it reuse it
 * (f32, N=4096; f64 has 16-row A strips, so 8 ir steps, and twice the
 * bytes per block):
 *
 *   loop  trips  fixed during it       size     reused by        lives in
 *   k     1024   C micro-tile          4 KB     1,024 k steps    ZA
 *   ir    4      B strip kc×nr         128 KB   4 A strips       cache
 *   jr    128    A block mc×kc         512 KB   128 B strips     L2
 *   ic    32     B panel kc×nc         16 MB    32 A blocks      L2 / SLC
 *   pc    4      — (splits K)
 *   jc    1      — (splits N)
 *
 *  - k innermost: C stays in ZA for the whole k loop and touches memory
 *    once per pc pass. Per k step f32 loads 256 B for 2,048 flops
 *    (8 flop/B), f64 384 B for 1,024 flops (~2.7 flop/B).
 *  - ir inside jr: a B strip is reused after one A strip (128 KB) has
 *    passed through the cache, an A strip after ~640 KB. Swapped, each A
 *    strip would walk the whole 16 MB panel before the next one reused it.
 *  - ic inside pc: B is packed once per (jc, pc) and serves all M/Mc A
 *    blocks; A, the small operand, is repacked N/Nc times. Packing is
 *    O(MK + KN) against O(MNK) flops.
 *  - pc: C makes one round trip through ZA per Kc, so Kc is large.
 *  - jc: caps the B panel at the 16 MB P-cluster L2.
 *
 * Blocks are sized for L2, not L1 as in classic GotoBLAS: on M4 the SME
 * unit is shared by the P-cluster and reads operands from L2.
 *
 * Packing runs in the non-streaming driver, because scalar and NEON code is
 * slow in streaming mode; only macro_kernel runs streaming, with one
 * SMSTART/SMSTOP per Mc×Nc×Kc block.
 *
 * Edges: packing zero-pads strips to mr / nr, so FMOPAs always run with an
 * all-true predicate; only the C transfers into and out of ZA are
 * predicated (columns) and bounded (rows).
 *
 * Pitfall: every streaming helper called from the ZA-owning macro_kernel
 * needs a ZA attribute (__arm_inout("za") etc.). Without one it is
 * "private-ZA": Clang won't inline it and wraps each call in a lazy ZA save
 * (TPIDR2 + smstart za); inside the k loop that cuts throughput ~5×.
 *
 *
 * ============================================================
 *  Packing
 * ============================================================
 *
 * Both operands become strips with one contiguous row per k step:
 *
 *   A: dst[s·mr·kc + k·mr + r] = A(s·mr + r, k)   row k = column k of A,
 *                                                  rows s·mr .. s·mr+mr−1
 *   B: dst[s·nr·kc + k·nr + c] = B(k, s·nr + c)   row k = B(k, s·nr .. +nr−1)
 *
 *   ┌ strip 0 ┬ strip 1 ┬ ... ┐   f32: kc = 1024 rows × 128 B = 128 KB each
 *
 * Row widths follow the micro-tile, not cache tuning: mr = 2·SVL (128 B
 * for both types), nr = kCols·SVL (128 B f32, 256 B f64). Each k step reads
 * one row of each strip with svld1_x2.
 *
 * A must be packed: read in place, the column A(i0..i0+31, k) is 32 values
 * K·sizeof(T) apart — 32 cache lines per k step, with no gather load to
 * fetch them. pack_a transposes with NEON 4×4 / 2×2 in-register transposes,
 * ~10% faster overall than a scalar loop.
 *
 * B could be read in place, as its row segments are contiguous, but then
 * the strip's rows are ldb·sizeof(T) bytes apart (ldb = N: every run here
 * is square N×N, so "N=4000" means a 16,000 B stride for f32). At
 * power-of-two strides this appears to cause severe cache-set conflicts.
 * An address splits into tag | set index | line offset (bits 0–6 for 128 B
 * lines), and a line may only live in one of the few ways of its own set.
 * A 2^14 B stride (f32 N=4096) never changes bits 0–13, so if the set index
 * comes from those bits (Apple does not document M4's indexing), all 1,024
 * rows of the strip land in one set:
 *
 *   in place, N=4096:  row k at base + k·16 KB    same set every row
 *   packed:            row k at strip + k·128 B   next set every row
 *
 * More generally, rows s bytes apart visit 128 / gcd(s/128, 128) of the 128
 * L1 sets: 1 at 16 KB, 2 at 8 KB, all 128 at 16,000 B (125 lines, odd).
 * Contiguity is what fixes it; any strip width would.
 *
 * Evidence, gemm_sme_blocked (in place) vs gemm_sme (packed), f32 / f64:
 *  - N swept around powers of two: −17% / −31% at N=2048 and −55% / −33%
 *    at N=4096, but within 3% at N = 1984–2080 and −3…−7% / −7…−11% at
 *    N = 4000–4128.
 *  - The same 4096×4096 problem with only ldb changed: ldb = 4096 −55% /
 *    −33%, ldb = 4128 (+128 B per row) −3…−8% / −6…−8% over two runs. The
 *    collapse follows the byte stride, not N, the work or the pages
 *    touched; padded B still stays a few percent behind packed strips.
 *  - ldb = 4097, whose rows are no longer 128 B aligned and so span two
 *    lines each: −16% / −10%. Line-sized, line-aligned rows matter too.
 *  - A pointer chase from a P-core over 1,024 lines 16 KB apart keeps only
 *    8 of them in L1 and fewer than 1,024 in L2 (24 ns per load, against
 *    0.78 ns for lines 128 B apart).
 * Only large problems are affected: at N ≤ 1024 in place is within 2% of
 * packed or faster (f64 N=1024, an 8 KB stride: +7%), as everything stays
 * in L2. The conflicts are in L2: with rows 16 KiB apart the macro-kernel
 * keeps pace with a padded stride up to 384 rows per strip and breaks by
 * 448 rows, far beyond L1's 8 ways. With a padded stride, in place still
 * trails packed by 0% at N=1024 up to ~10% at N=4096. Padding B would not
 * replace packing in a library anyway: the caller owns B's layout, while
 * packing works for any N. Data: docs/experiments/sme-b-stride.
 *
 * The packing buffers are Matrix<T> storage, 64-byte aligned by contract;
 * the ≥ 128 KB buffers came back 128-byte aligned in all 600 allocations
 * tried on macOS, but nothing guarantees it.
 *
 * Full tables: docs/benchmarks.md, "SME: B row stride". Reproduce with
 * bench_gemm --benchmark_filter=BStride.
 *
 *
 * ============================================================
 *  Build and availability
 * ============================================================
 *
 * SME2 is available on Apple M4 / M4 Pro / M4 Max (512-bit SVL), not on
 * Apple M1–M3, AWS Graviton3/4, Fujitsu A64FX or x86. HPC_HAS_SME
 * (hpc/isa.hpp) follows __ARM_FEATURE_SME && __ARM_FEATURE_SME2; where it is
 * 0, all four kernels are declared `= delete`. The flags and why a wrong
 * combination SIGILLs on Apple Silicon: docs/build.md, "SME and AMX build
 * flags".
 *
 * svcntsw()/svcntsd() compile to RDSVL, an SME instruction that is legal
 * outside streaming mode, so the non-streaming driver can size its packing
 * buffers with them.
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
// Cache-blocking parameters (elements) for gemm_sme and gemm_sme_blocked;
// what each block holds is in the file header ("gemm_sme"). Values picked by
// a sweep on M4 Max (Mc 64–512, Kc 256–2048, Nc 512–4096): large Kc/Nc win
// because they amortise C reloads and A repacking; the gain from Nc 2048 →
// 4096 shows only at N ≥ 4096 (+5%).
// ============================================================================

inline constexpr std::size_t kSmeMc = 128;
inline constexpr std::size_t kSmeKc = 1024;
inline constexpr std::size_t kSmeNc = 4096;

#if !HPC_HAS_SME

template <typename T>
void gemm_sme_1tile(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // SME2 not available on this target
template <typename T>
void gemm_sme_tiles(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // SME2 not available on this target
template <typename T>
void gemm_sme_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // SME2 not available on this target
template <typename T>
void gemm_sme(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // SME2 not available on this target

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
                    // 4×4 transpose: swap single floats, then 2-float halves.
                    const float32x4_t r0 = vld1q_f32(src + k);            // a0 a1 a2 a3
                    const float32x4_t r1 = vld1q_f32(src + lda + k);      // b0 b1 b2 b3
                    const float32x4_t r2 = vld1q_f32(src + 2 * lda + k);  // c0 c1 c2 c3
                    const float32x4_t r3 = vld1q_f32(src + 3 * lda + k);  // d0 d1 d2 d3
                    const float32x4_t t0 = vtrn1q_f32(r0, r1);            // a0 b0 a2 b2
                    const float32x4_t t1 = vtrn2q_f32(r0, r1);            // a1 b1 a3 b3
                    const float32x4_t t2 = vtrn1q_f32(r2, r3);            // c0 d0 c2 d2
                    const float32x4_t t3 = vtrn2q_f32(r2, r3);            // c1 d1 c3 d3
                    vst1q_f32(out,          vcombine_f32(vget_low_f32(t0),  vget_low_f32(t2)));   // a0 b0 c0 d0
                    vst1q_f32(out + mr,     vcombine_f32(vget_low_f32(t1),  vget_low_f32(t3)));   // a1 b1 c1 d1
                    vst1q_f32(out + 2 * mr, vcombine_f32(vget_high_f32(t0), vget_high_f32(t2)));  // a2 b2 c2 d2
                    vst1q_f32(out + 3 * mr, vcombine_f32(vget_high_f32(t1), vget_high_f32(t3)));  // a3 b3 c3 d3
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

/// Macro-kernel (streaming): C[mc×nc] (+)= Apack[mc×kc] · B[kc×nc], all ZA
/// tiles. A is packed (pack_a). Row k of the B strip starting at column jr
/// is at B + jr·b_step + k·ldb: packed strips (b_step = kc, ldb = nr) for
/// gemm_sme, or B in place (b_step = 1, ldb = N) for gemm_sme_tiles and
/// gemm_sme_blocked. B loads are predicated to the strip's real columns, so
/// reading B in place never runs past the end of a row.
template <typename T>
__arm_locally_streaming __arm_new("za")
void macro_kernel(const T* __restrict Ap, const T* __restrict B, std::int64_t b_step,
                  std::int64_t ldb, T* __restrict C, std::int64_t ldc, std::int64_t mc,
                  std::int64_t nc, std::int64_t kc, bool accumulate) {
    constexpr int kC      = kCols<T>;
    const std::int64_t s  = kF32<T> ? svcntsw() : svcntsd();
    const std::int64_t mr = 2 * s, nr = kC * s;
    const svbool_t pt     = kF32<T> ? svptrue_b32() : svptrue_b64();
    const svcount_t pn    = kF32<T> ? svptrue_c32() : svptrue_c64();

    for (std::int64_t jr = 0; jr < nc; jr += nr) {
        const std::int64_t cols = std::min(nr, nc - jr);
        const T* b_strip        = B + jr * b_step;
        // Columns [0, 2s) and, for f64, [2s, 4s) of the strip.
        const svcount_t pb0 = kF32<T> ? svwhilelt_c32(std::int64_t{0}, cols, 2)
                                      : svwhilelt_c64(std::int64_t{0}, cols, 2);
        const svcount_t pb1 = svwhilelt_c64(2 * s, cols, 2);

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
                const auto b  = svld1_x2(pb0, b_strip + k * ldb);
                const auto a0 = svget2(a, 0), a1 = svget2(a, 1);
                mopa<T, 0>(pt, a0, svget2(b, 0));
                mopa<T, 1>(pt, a0, svget2(b, 1));
                mopa<T, kC + 0>(pt, a1, svget2(b, 0));
                mopa<T, kC + 1>(pt, a1, svget2(b, 1));
                if constexpr (kC == 4) {  // f64: columns 2–3 use the other 4 za64 tiles
                    const auto b2 = svld1_x2(pb1, b_strip + k * ldb + 2 * s);
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

/// One-tile kernel (streaming): C[M×N] = Apack[M×K] · B[K×N] with only ZA
/// tile 0 (s×s). Apack holds s-row strips; B is read in place.
template <typename T>
__arm_locally_streaming __arm_new("za")
void one_tile_kernel(const T* __restrict Ap, const T* __restrict B, T* __restrict C,
                     std::int64_t M, std::int64_t N, std::int64_t K) {
    const std::int64_t s = kF32<T> ? svcntsw() : svcntsd();
    const svbool_t pt    = kF32<T> ? svptrue_b32() : svptrue_b64();

    for (std::int64_t j0 = 0; j0 < N; j0 += s) {
        const svbool_t pg = kF32<T> ? svwhilelt_b32(j0, N) : svwhilelt_b64(j0, N);
        for (std::int64_t i0 = 0; i0 < M; i0 += s) {
            const T* a_strip = Ap + i0 * K;
            svzero_za();
            // Every FMOPA accumulates into the same tile, so each one waits
            // for the previous one's result: this loop runs at FMOPA latency.
            for (std::int64_t k = 0; k < K; ++k) {
                const auto a = svld1(pt, a_strip + k * s);
                const auto b = svld1(pg, B + k * N + j0);
                mopa<T, 0>(pt, a, b);
            }
            move_tile<T, 0, false>(C + i0 * N + j0, N, std::min(s, M - i0), pg);
        }
    }
}

/// Pack all of A (M×K) into mr-row strips, for the kernels without blocking.
template <typename T>
Matrix<T> pack_all_a(const Matrix<T>& A, std::size_t mr) {
    const std::size_t M = A.rows(), K = A.cols();
    Matrix<T> a_pack((M + mr - 1) / mr * mr, K);
    pack_a(A.data(), K, M, K, mr, a_pack.data());
    return a_pack;
}

}  // namespace sme_detail

/**
 * @brief C = A · B on ARM SME, step 1 of 4: one ZA tile, A packed into
 *        SVL-row strips, B read in place, no cache blocking. FMOPA latency
 *        bound; see the file header.
 */
template <typename T>
void gemm_sme_1tile(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sme_1tile: T must be float or double");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;
    const std::size_t s = sme_detail::kF32<T> ? svcntsw() : svcntsd();
    const Matrix<T> a_pack = sme_detail::pack_all_a(A, s);
    sme_detail::one_tile_kernel<T>(a_pack.data(), B.data(), C.data(),
                                   static_cast<std::int64_t>(M), static_cast<std::int64_t>(N),
                                   static_cast<std::int64_t>(K));
}

/**
 * @brief C = A · B on ARM SME, step 2 of 4: all ZA tiles (gemm_sme's
 *        macro_kernel run once over the whole problem), A packed into
 *        2·SVL-row strips, B read in place, no cache blocking. See the file
 *        header.
 */
template <typename T>
void gemm_sme_tiles(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sme_tiles: T must be float or double");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;
    const std::size_t s = sme_detail::kF32<T> ? svcntsw() : svcntsd();
    const Matrix<T> a_pack = sme_detail::pack_all_a(A, 2 * s);
    const auto n = static_cast<std::int64_t>(N);
    sme_detail::macro_kernel<T>(a_pack.data(), B.data(), 1, n, C.data(), n,
                                static_cast<std::int64_t>(M), n, static_cast<std::int64_t>(K),
                                false);
}

namespace sme_detail {

/// GotoBLAS driver shared by gemm_sme_blocked (PackB = false) and gemm_sme
/// (PackB = true); the only difference is where the macro-kernel reads B.
/// B is K×N with rows ldb ≥ N elements apart (B(k, j) = B[k·ldb + j]); the
/// public kernels pass ldb = N, the BStride benchmarks pad it.
template <typename T, bool PackB>
void blocked_driver(const Matrix<T>& A, const T* B, std::size_t ldb, Matrix<T>& C) {
    const std::size_t M = A.rows(), K = A.cols(), N = C.cols();
    assert(C.rows() == M && ldb >= N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;

    // Micro-tile is mr × nr; RDSVL (svcnts*) is legal outside streaming mode.
    const std::size_t s  = kF32<T> ? svcntsw() : svcntsd();
    const std::size_t mr = 2 * s, nr = kCols<T> * s;
    auto round_up        = [](std::size_t x, std::size_t m) { return (x + m - 1) / m * m; };
    const std::size_t kc_max = std::min(kSmeKc, K);
    Matrix<T> a_pack(round_up(std::min(kSmeMc, M), mr), kc_max);  // 64-byte aligned scratch
    Matrix<T> b_pack(PackB ? kc_max : 0, PackB ? round_up(std::min(kSmeNc, N), nr) : 0);

    for (std::size_t jc = 0; jc < N; jc += kSmeNc) {
        const std::size_t nc = std::min(kSmeNc, N - jc);
        for (std::size_t pc = 0; pc < K; pc += kSmeKc) {
            const std::size_t kc = std::min(kSmeKc, K - pc);
            // B strip rows: packed, nr apart (b_step = kc); in place, ldb apart.
            const T* b          = B + pc * ldb + jc;
            std::int64_t b_step = 1, b_ld = static_cast<std::int64_t>(ldb);
            if constexpr (PackB) {
                pack_b(b, ldb, kc, nc, nr, b_pack.data());
                b      = b_pack.data();
                b_step = static_cast<std::int64_t>(kc);
                b_ld   = static_cast<std::int64_t>(nr);
            }
            for (std::size_t ic = 0; ic < M; ic += kSmeMc) {
                const std::size_t mc = std::min(kSmeMc, M - ic);
                pack_a(A.data() + ic * K + pc, K, mc, kc, mr, a_pack.data());
                macro_kernel<T>(a_pack.data(), b, b_step, b_ld, C.data() + ic * N + jc,
                                static_cast<std::int64_t>(N), static_cast<std::int64_t>(mc),
                                static_cast<std::int64_t>(nc), static_cast<std::int64_t>(kc),
                                pc > 0);
            }
        }
    }
}

}  // namespace sme_detail

/**
 * @brief C = A · B on ARM SME, step 3 of 4: gemm_sme without pack_b (all ZA
 *        tiles, cache blocking, B read in place). See the file header,
 *        "Packing", for what that costs at power-of-two N.
 */
template <typename T>
void gemm_sme_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sme_blocked: T must be float or double");
    assert(B.rows() == A.cols() && B.cols() == C.cols());
    sme_detail::blocked_driver<T, false>(A, B.data(), B.cols(), C);
}

/**
 * @brief C = A · B on ARM SME, step 4 of 4: all ZA tiles, cache blocking,
 *        A and B packed. See the file header, "gemm_sme".
 */
template <typename T>
void gemm_sme(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sme: T must be float or double");
    assert(B.rows() == A.cols() && B.cols() == C.cols());
    sme_detail::blocked_driver<T, true>(A, B.data(), B.cols(), C);
}

#endif  // HPC_HAS_SME

}  // namespace hpc::gemm
