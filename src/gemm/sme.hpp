#pragma once

/**
 * @file sme.hpp
 * @brief ARM SME (Scalable Matrix Extension) GEMM — four steps: one ZA tile,
 *        all ZA tiles, cache blocking, packed B.
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
 *  The kernels — four steps
 * ============================================================
 *
 *   kernel            ZA tiles   A              B          cache blocking
 *   gemm_sme_1tile    1          packed strips  in place   none
 *   gemm_sme_tiles    all        packed strips  in place   none
 *   gemm_sme_blocked  all        packed strips  in place   GotoBLAS
 *   gemm_sme          all        packed strips  packed     GotoBLAS
 *
 * Every step packs A (pack_a): FMOPA needs a column of A as one vector and
 * streaming mode has no gather loads. The strips make each k step's column
 * contiguous (64 B for one tile, a full 128 B line for two), which also
 * keeps A clear of the cache-set conflicts a power-of-two row stride causes
 * (design item 2 below). So each step changes one thing: the number of
 * tiles, then blocking, then packing B. Each kernel is one template for
 * f32 and f64; the last three share one macro-kernel.
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
 *  - 1tile → tiles: 2.3–3.4× f32, 2.9–5.8× f64. All tiles hide FMOPA
 *    latency (design item 1). At large N the one-tile kernel is also
 *    bandwidth bound: each A and B vector it loads feeds one FMOPA.
 *  - tiles → blocked: 2.0× f32 at N=2048, 3.2× at N=4000. Without
 *    blocking, all of A is re-read for every B strip and no longer stays
 *    cached. N=4000 is as slow as N=4096, so this is not a stride effect;
 *    capacity is the likely cause. At N=4096 blocking gains only 1.3×,
 *    because the next problem takes over.
 *  - blocked → gemm_sme: within 3% at N=4000, but 2.3× f32 / 1.5× f64 at
 *    N=4096 and 1.15× f32 at N=2048. The loss follows the power-of-two
 *    distance between B strip rows and is consistent with cache-set
 *    conflicts; packing B removes it (design item 2).
 *  - At N ≤ 512 the kernels without pack_b are faster (+30% f32 at N=256
 *    for gemm_sme_tiles): everything fits in cache anyway, and gemm_sme
 *    pays for copying B.
 *
 *
 * ============================================================
 *  gemm_sme — design
 * ============================================================
 *
 * gemm_sme_1tile peaks at ~470 GFLOP/s f32 on one M4 Max core, against
 * ~1650 GFLOP/s for Accelerate on the same core. gemm_sme closes most of
 * that gap with four design choices. gemm_sme_tiles already has 1, 3 and 4
 * (it shares gemm_sme's macro-kernel); gemm_sme_blocked adds the blocking
 * of 2 and gemm_sme the packing of B:
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
 *     has this order: "gemm_sme — the algorithm" below). The packed B strip
 *     (kc×nr) and A strip (kc×mr) are both read with unit stride.
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
 *     At a power-of-two stride this appears to cause severe cache-set
 *     conflicts. A cache splits an address into tag | set index | line
 *     offset. The offset is bits 0–6 (128 B lines); the set index is the
 *     next bits; a line may only live in one of the few ways of its own
 *     set. A stride of 2^14 (N=4096, f32) never changes bits 0–13, so if
 *     the set index comes straight from those bits (Apple does not document
 *     M4's indexing), every row of the strip lands in the same set:
 *
 *       row 0     base + 0·16384      set X
 *       row 1     base + 1·16384      set X
 *       ...
 *       row 1023  base + 1023·16384   set X   → 1,024 lines, a few ways
 *
 *     Packed, row k is at strip + k·128 and each row takes the next set, so
 *     the strip spreads over all sets. At N=4000 the stride is 16,000 B =
 *     125 lines; 125 is odd, so the set index walks every set before
 *     repeating and the strip also fits. What helps is the compact,
 *     contiguous layout, not the strip width of 32 (that comes from the
 *     tile shape).
 *
 *     Evidence (gemm_sme_blocked = gemm_sme without pack_b, f32):
 *       - N swept around powers of two: in place loses 55% to packed at
 *         N=4096 and 17% at N=2048, but is within 7% of it at N = 2016,
 *         2080, 4000, 4032, 4064 and 4128.
 *       - Same 4096×4096 problem, only B's row stride changed: ldb = 4096
 *         loses 55%, ldb = 4128 (one extra cache line per row) 3%. So the
 *         loss follows the byte stride, not N, the work or the pages
 *         touched.
 *       - A pointer chase from a P-core (1,024 lines, random order) costs
 *         0.78 ns per load at 128 B apart, 2.4 ns at 16,000 B (likely
 *         TLB) and 24.4 ns at 16 KB; at 16 KB apart only 8 lines stay in
 *         L1 and fewer than 1,024 in L2.
 *     Not established: which cache level the SME unit's loads conflict in
 *     (it reads through L2, and the pointer chase only shows the core's
 *     view), and the cause of the 3–11% that in place still loses away
 *     from powers of two (TLB, prefetching and footprint were not
 *     separated). Full tables, f64 included: src/gemm/README.md, "Why A
 *     and B are packed"; reproduce with bench_gemm --benchmark_filter=BStride.
 *
 *     Terms. N is the matrix size: every run above is square, so B has
 *     K = N rows of N columns. ldb is the number of elements between the
 *     starts of consecutive B rows (= N unless B is stored padded), and
 *     the stride is the byte distance, ldb·sizeof(T). "N=4000" therefore
 *     means a 16,000 B stride for f32 and 32,000 B for f64.
 *
 *     Which sizes are affected. With 128 B lines and the 128-set L1
 *     above, rows s bytes apart (s a multiple of 128) step the set index
 *     by s/128 and visit 128 / gcd(s/128, 128) different sets: 1 at a
 *     16 KB stride, 2 at 8 KB, all 128 at 16,000 B. So any stride with
 *     many trailing zero bits is at risk, not only exact powers of two
 *     (f32 N=3072 and 6144 would use 4 and 2 sets; not measured). The
 *     loss also needs a large problem: it was measured only at N=2048 and
 *     4096 (f32 −17% / −55%, f64 −31% / −33%). At N ≤ 1024 in place is
 *     within 2% of packed or faster (f64 N=1024: +7%, although its stride
 *     is 8 KB like f32 N=2048), as the whole problem stays in L2.
 *
 *     Other strides vs packed strips. Away from those sizes, reading B in
 *     place comes close to packed but does not match it: −3% f32 / −7 to
 *     −8% f64 at N = 4000, 4032 and 4128 (−7% / −11% at N=4064), and
 *     padding the 4096 rows by 128 B (ldb = 4128) gives −3% / −8%. Around
 *     N=2000 f32 in place is up to 3% faster, because it skips the copy.
 *     One plausible factor in the remaining gap (not measured): a packed f32
 *     strip is 128 KB of contiguous memory, ~8 pages of 16 KB, while read
 *     in place its 1,024 rows sit on ~1,024 different pages — more TLB
 *     entries, and the prefetcher crosses page boundaries far more often.
 *     Padding is no substitute for packing in a library anyway: the caller
 *     owns B's layout (Matrix<T> has no padding), while packing works for
 *     any N and ldb.
 *
 *     How the strip widths are chosen. They follow from the micro-tile,
 *     not from cache tuning. A packed B row is nr = kCols<T>·SVL elements
 *     (2·16 floats = 128 B, 4·8 doubles = 256 B); a packed A row is
 *     mr = 2·SVL elements (128 B for both types). Contiguity is what
 *     removes the set conflicts, and any width would do that. The width
 *     matters for another reason: a row that is a whole number of cache
 *     lines, starting on a line boundary, never splits a load across two
 *     lines (the misaligned ldb = 4097 control costs −16% f32). Matrix<T>
 *     only promises 64-byte alignment; the packing buffers (≥ 128 KB)
 *     came back 128-byte aligned in all 600 allocations tried on macOS,
 *     but nothing guarantees it.
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
 *  gemm_sme — the algorithm
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
 * 0, all four kernels are declared `= delete`.
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
 * @brief C = A · B on ARM SME with a single ZA tile — the first step of the
 *        ladder: FMOPA latency bound.
 *
 * A is packed into s-row strips (s = SVL elements), which every SME kernel
 * needs: FMOPA takes a column of A as one vector and streaming mode has no
 * gather loads. The strips also keep A clear of the cache-set conflicts a
 * power-of-two row stride causes (see the file header). B is read in place
 * and there is no cache blocking. Single-threaded.
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
 * @brief C = A · B on ARM SME with all ZA tiles — the second step: FMOPA
 *        latency hidden, B still read in place.
 *
 * Same micro-kernel as gemm_sme (2×2 za32 / 2×4 za64 tiles), A packed into
 * 2·SVL-row strips once, but no B packing and no cache blocking: B strip rows
 * are N·sizeof(T) apart, and A is re-read for every B strip, so it slows
 * down from N=2048 (see the file header). Single-threaded.
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
 * @brief C = A · B on ARM SME with all ZA tiles and GotoBLAS cache blocking,
 *        B read in place — the third step.
 *
 * Exactly gemm_sme without pack_b. Blocking fixes gemm_sme_tiles' large-N
 * slowdown, but B strip rows stay N·sizeof(T) apart, and at a power-of-two
 * stride it loses up to 55% to gemm_sme, consistent with cache-set
 * conflicts (design item 2 in the file header).
 * Single-threaded.
 */
template <typename T>
void gemm_sme_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sme_blocked: T must be float or double");
    assert(B.rows() == A.cols() && B.cols() == C.cols());
    sme_detail::blocked_driver<T, false>(A, B.data(), B.cols(), C);
}

/**
 * @brief C = A · B on ARM SME — packed, cache-blocked, all ZA tiles in use.
 *
 * See the file header for the design. Single-threaded.
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
