#pragma once

/**
 * @file sme.hpp
 * @brief ARM SME (Scalable Matrix Extension) GEMM in four steps: one ZA tile,
 *        all ZA tiles, cache blocking, packed B; then packed B on several
 *        SME units (gemm_sme_mt). gemm_sme and gemm_sme_mt also take fp16,
 *        bf16 and INT8 inputs, with the widening outer products.
 *
 * ============================================================
 * Shape and cache block: gemm_sme<T, R, V, TileM, TileK, TileN>, no
 * defaults. R/V count ZA tile rows/columns, not SIMD scalar rows/vector
 * columns. The values below are M4's measured ones
 * (benchmarks/sme_params.hpp); rationale and sweeps: docs/kernels/sme.md.
 *
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
 * With a[r] = A(i0+r, k) and b[c] = B(k, j0+c), K of these instructions
 * produce an SVL×SVL tile of C — the same primitive as GPU Tensor Cores.
 * SVL is read at runtime with svcntsw() / svcntsd(): 16 f32 / 8 f64 on
 * Apple M4, where ZA holds 4 f32 or 8 f64 tiles of SVL×SVL.
 *
 * FMOPA, ZA loads/stores and `zero {za}` are only legal in streaming SVE
 * mode. `__arm_locally_streaming` makes Clang wrap a function in SMSTART /
 * SMSTOP; `__arm_new("za")` gives it its own ZA state (any live caller ZA
 * state is preserved; svzero_za explicitly initializes the accumulator). Gather loads are not legal in
 * streaming mode, so the strided column of A that FMOPA needs cannot be
 * loaded directly: every kernel here packs A ("Packing" below).
 *
 *
 * ============================================================
 *  The kernels
 * ============================================================
 *
 *   kernel            ZA tiles   B          cache blocking   threads
 *   gemm_sme_1tile    1          in place   none             1
 *   gemm_sme_tiles    all        in place   none             1
 *   gemm_sme_blocked  all        in place   GotoBLAS         1
 *   gemm_sme          all        packed     GotoBLAS         1
 *   gemm_sme_mt       all        packed     GotoBLAS         `threads`
 *
 * All pack A, so each step changes one thing. Each is one template for f32
 * and f64 (gemm_sme and gemm_sme_mt also for fp16, bf16 and INT8: "Low
 * precision" below); all but gemm_sme_1tile share macro_kernel. The first four are
 * single-threaded: compare against Accelerate with VECLIB_MAXIMUM_THREADS=1
 * (1,600–1,720 GFLOP/s f32, 340–460 f64 on one M4 Max core, N = 256–4096).
 *
 * gemm_sme_mt is gemm_sme with the B panel packed once and shared, and A
 * blocks handed out to threads on demand. An SME unit belongs to a
 * P-cluster, not a core, so one thread reaches at most one unit's
 * throughput; M4 Max has two. Details at its definition and in
 * docs/kernels/sme.md, "Several SME units".
 *
 * One M4 Max core, GFLOP/s, median of 3 runs, block sizes as below:
 *
 *   f32                N=256   512  1024  2048  4000  4096
 *   gemm_sme_1tile       420   472   464   220   125   131
 *   gemm_sme_tiles       975  1447  1319   617   427   446
 *   gemm_sme_blocked     989  1460  1333   633  1468   489
 *   gemm_sme             835  1322  1365  1452  1532  1519
 *
 *   f64                N=256   512  1024  2048  4000  4096
 *   gemm_sme_1tile       121   121   105    40    42    38
 *   gemm_sme_tiles       352   418   401   227   204   205
 *   gemm_sme_blocked     351   420   423   243   422   225
 *   gemm_sme             322   396   404   413   436   421
 *
 *  - 1tile → tiles, 2.3–3.4× f32 / 2.9–5.7× f64: FMOPA latency hidden
 *    ("All ZA tiles" below). At large N the one-tile kernel is also
 *    bandwidth bound, since each vector it loads feeds one FMOPA.
 *  - tiles → blocked, 3.4× f32 / 2.1× f64 at N=4000: without blocking, all
 *    of A is re-read for every B strip and falls out of cache. N=4000 is
 *    as slow as N=4096 without blocking, so that is capacity, not stride.
 *  - blocked → gemm_sme: +4% at N=4000, but 2.3× f32 at N=2048 and 3.1× at
 *    N=4096 (1.7× / 1.9× f64) — the power-of-two stride effect that
 *    packing B removes ("Packing" below). With 2,048-row strips it hits
 *    an 8 KiB stride (N=2048) as hard as a 16 KiB one.
 *  - At N ≤ 512 the kernels without pack_b are faster (+18% f32 at
 *    N=256): everything fits in cache, and gemm_sme pays for copying B.
 *
 *
 * ============================================================
 *  gemm_sme (at M4's measured shape and cache blocks)
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
 * Cache blocking, GotoBLAS-style (f32 sizes; f64 has Mc = 512):
 *
 *   for jc (Nc = 4096)      pack_b → 32 MiB B panel       normal mode
 *    for pc (Kc = 2048)
 *     for ic (Mc = 1024)    pack_a → 8 MiB A block        normal mode
 *      macro_kernel:                                      streaming mode
 *       for jr              one B strip, 256 KiB
 *        for ir             32 A strips, each re-reads the same B strip
 *         C tile → ZA       (zeroed when pc == 0, else loaded from C)
 *         for k = 0..2047   2 A + 2 B vectors → 4 FMOPA
 *         ZA → C tile
 *
 * Each level holds one operand fixed while the levels inside it reuse it
 * (f32, N=4096; f64 has 16-row A strips and Mc = 512, so also 32 ir steps,
 * an 8 MiB A block, 512 KiB B strips and a 64 MiB B panel):
 *
 *   loop  trips  fixed during it       size      reused by        lives in
 *   k     2048   C micro-tile          4 KiB     2,048 k steps    ZA
 *   ir    32     B strip kc×nr         256 KiB   32 A strips      L2
 *   jr    128    A block mc×kc         8 MiB     128 B strips     L2
 *   ic    4      B panel kc×nc         32 MiB    4 A blocks       SLC / DRAM
 *   pc    2      — (splits K)
 *   jc    1      — (splits N)
 *
 *  - k innermost: C stays in ZA for the whole k loop and touches memory
 *    once per pc pass. Per k step f32 loads 256 B for 2,048 flops
 *    (8 flop/B), f64 384 B for 1,024 flops (~2.7 flop/B).
 *  - ir inside jr: a B strip is reused by 32 A strips in a row, so it is
 *    fetched from beyond L2 once and read from L2 31 times. An A strip is
 *    reused once per B strip, after the whole 8 MiB A block has passed:
 *    the A block has to stay in L2, and at half the L2 it does.
 *  - ic inside pc: B is packed once per (jc, pc) and serves all M/Mc A
 *    blocks; A is repacked N/Nc times. Packing is O(MK + KN) against
 *    O(MNK) flops.
 *  - pc: C makes one round trip through ZA per Kc, so Kc is large.
 *  - jc: bounds the B panel. It does not fit in L2 and need not: each B
 *    strip is streamed in once per A block and then reused from L2.
 *
 * This is classic GotoBLAS — the A block resident in L2, B streamed — one
 * level out: the B strips are reused from L2 rather than L1, because on
 * M4 our stride/cache measurements point to operand reuse through the
 * P-cluster L2, rather than the core L1. This is an inference about M4,
 * not an SME architectural requirement or a documented internal datapath.
 * The previous sizing (Mc = 128, Kc = 1024) aimed at keeping the whole
 * B panel in L2 instead; the current one is 4–12% faster at N ≥ 2048.
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
 *   ┌ strip 0 ┬ strip 1 ┬ ... ┐   f32: kc = 2048 rows × 128 B = 256 KiB each
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
 * comes from those bits (Apple does not document M4's indexing), all 2,048
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
 *  - N swept around powers of two: −57% / −42% at N=2048 and −68% / −48%
 *    at N=4096; f32 is within 5% at N = 1984–2080 and 4000–4128.
 *  - The same 4096×4096 problem with only ldb changed: f32 −68% at
 *    ldb = 4096, −6% at ldb = 4128 (+128 B per row). The collapse follows
 *    the byte stride, not N or the work; padding adds only ~0.8% storage.
 *  - f64 in place loses 26–48% at nearly every N around 4096, padded or
 *    not (2,048 rows of 256 B per strip); with 1,024-row strips padded f64
 *    lost only 7–8%. Packing removes it.
 *  - ldb = 4097, whose rows are no longer 128 B aligned and so span an
 *    extra line each: −16% / −30%. Line-sized, line-aligned rows matter too.
 *  - A pointer chase from a P-core over 1,024 lines 16 KB apart keeps only
 *    8 of them in L1 and fewer than 1,024 in L2 (100 cycles per load,
 *    against 12 for lines 16,512 B apart).
 * The evidence points to L2 conflicts: with rows 16 KiB apart the macro-kernel breaks
 * between 576 and 640 rows per strip, far beyond L1's 8 ways. With a
 * padded stride, in place still trails packed by 4–8% (f32). Padding B
 * would not replace packing in a library anyway: the caller owns B's
 * layout, while packing works for any N. Data: docs/experiments/sme-b-stride.
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
 *  Low precision: widening outer products
 * ============================================================
 *
 *   gemm_sme<fp16>        C (FP32)  = A (fp16) · B (fp16)
 *   gemm_sme<bf16>        C (FP32)  = A (bf16) · B (bf16)
 *   gemm_sme<std::int8_t> C (INT32) = A (int8) · B (int8)
 *
 * The same kernels, with another instruction. A widening FMOPA / SMOPA reads
 * two vectors whose elements come in groups of G along k (G = 2 for 16-bit
 * inputs, 4 for INT8) and adds a dot product of length G into each 32-bit
 * accumulator:
 *
 *   ZA[t][r][c] += Σ_g a[r·G + g] · b[c·G + g]      g = 0 … G−1
 *
 * so one instruction does G times the work of the FP32 FMOPA: 1,024 flops
 * (16-bit) or 2,048 integer operations (INT8) on M4's 512-bit vectors, and
 * C stays in the same four 32-bit tiles as FP32. Everything else is FP32's
 * with a 32-bit word of G elements (sme_word_t) in place of an element: the
 * packed strips hold one word per (row or column, k group),
 *
 *   A strip: word [w][r]  = A(r, G·w … G·w+G−1)   a k group of a row: contiguous in A
 *   B strip: word [w][c]  = B(G·w … G·w+G−1, c)   G rows of B interleaved
 *
 * so pack_a is the same NEON transpose of 32-bit words, and pack_b zips G
 * rows of B. Without packing B there is no way to feed these instructions
 * (B in place has its k groups N elements apart), so gemm_sme_tiles and
 * gemm_sme_blocked stay FP32 / FP64. FP32 and FP64 are the case G = 1.
 *
 * The blocks used (benchmarks/sme_params.hpp; not swept for these types)
 * keep FP32's sizes in k groups, so in bytes: an 8 MiB A block
 * (Mc = 1024 rows × Kc = 2048 words, 4,096 fp16 / bf16 or 8,192 INT8
 * elements) and Nc = 4096. K is zero-padded to a multiple of G.
 *
 * M4 implements these instructions (SME_F16F32, SME_B16F32, SME_I8I32), but
 * not the non-widening FP16 → FP16 or BF16 → BF16 forms.
 *
 *
 * ============================================================
 *  Build and availability
 * ============================================================
 *
 * SME2 is available on Apple M4 / M4 Pro / M4 Max (512-bit SVL), not on
 * Apple M1–M3, AWS Graviton3/4, Fujitsu A64FX or x86. HPC_HAS_SME
 * (hpc/isa.hpp) follows __ARM_FEATURE_SME && __ARM_FEATURE_SME2; where it is
 * 0, all five kernels are declared `= delete`. The flags and why a wrong
 * combination SIGILLs on Apple Silicon: docs/build.md, "SME and AMX build
 * flags".
 *
 * svcntsw()/svcntsd() compile to RDSVL, an SME instruction that is legal
 * outside streaming mode, so the non-streaming driver can size its packing
 * buffers with them.
 */

#include "gemm/lowp_naive.hpp"
#include "hpc/isa.hpp"
#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"

#if HPC_HAS_SME
    #include <arm_neon.h>
    #include <arm_sme.h>
#endif
#if defined(__APPLE__)
    #include <pthread.h>
#endif

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpc::gemm {

// ============================================================================
// Input types: FP32 and FP64 with FMOPA, one k per instruction (G = 1);
// fp16, bf16 and INT8 with the widening FMOPA / SMOPA, G = 2 or 4 k per
// 32-bit lane, into FP32 or INT32 ("Low precision" in the file header).
// ============================================================================

template <typename T>
inline constexpr bool kSmeWidening = !std::is_same_v<T, float> && !std::is_same_v<T, double>;
/// What packing and the ZA loads move: one element of FP32 / FP64, else one
/// 32-bit word of G elements consecutive in k.
template <typename T>
using sme_word_t = std::conditional_t<kSmeWidening<T>, std::uint32_t, T>;
/// k per word.
template <typename T>
inline constexpr std::size_t kSmeG = sizeof(sme_word_t<T>) / sizeof(T);
/// C's element type: T for FP32 / FP64, else FP32 or INT32 (lowp_out_t).
template <typename T>
using sme_out_t = std::conditional_t<kSmeWidening<T>, lowp_out_t<T>, T>;

// R and V count square ZA tiles, not scalar rows and vector columns; TileM,
// TileK, TileN are the cache block in elements (TileK a whole number of k
// groups). None has a default: the values are measured
// (benchmarks/sme_params.hpp, docs/kernels/sme.md).

#if !HPC_HAS_SME

template <typename T>
void gemm_sme_1tile(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // SME2 not available on this target
template <typename T, int R, int V>
void gemm_sme_tiles(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // SME2 not available on this target
template <typename T, int R, int V, std::size_t TileM, std::size_t TileK, std::size_t TileN>
void gemm_sme_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // SME2 not available on this target
template <typename T, int R, int V, std::size_t TileM, std::size_t TileK, std::size_t TileN>
void gemm_sme(const Matrix<T>&, const Matrix<T>&, Matrix<sme_out_t<T>>&) = delete;  // SME2 not available on this target
template <typename T, int R, int V, std::size_t TileM, std::size_t TileK, std::size_t TileN>
void gemm_sme_mt(const Matrix<T>&, const Matrix<T>&, Matrix<sme_out_t<T>>&, unsigned threads) = delete;  // SME2 not available on this target

#else

namespace sme_detail {

/// One word, the G elements of In from p, at any alignment.
template <typename In>
inline sme_word_t<In> load_word(const In* p) noexcept {
    sme_word_t<In> w;
    std::memcpy(&w, p, sizeof w);
    return w;
}

/// Pack an mc×kc block of row-major A (lda elements) into mr-row strips of
/// k-group words, column-major within each strip:
/// dst[s·mr·kw + w·mr + r] = A(s·mr + r, G·w … G·w+G−1), kw = ⌈kc / G⌉,
/// zero-padded to mr rows and to a whole last group. For FP32 / FP64 a word
/// is one element, so row w of a strip is column k = w of A.
template <typename In>
inline void pack_a(const In* A, std::size_t lda, std::size_t mc, std::size_t kc, std::size_t mr,
                   sme_word_t<In>* dst) {
    using W                 = sme_word_t<In>;
    constexpr std::size_t G = kSmeG<In>;
    // NEON in-register transpose of words: L×L tiles (L = 4 32-bit or 2
    // 64-bit words), ~2× faster than the scalar strided-store loop below.
    // Bytes in and out, so the move is exact for any bit pattern.
    constexpr std::size_t L = 16 / sizeof(W);
    const std::size_t kw = (kc + G - 1) / G, full = kc / G;  // full: whole k groups
    const std::size_t row_bytes = lda * sizeof(In);
    for (std::size_t i0 = 0; i0 < mc; i0 += mr) {
        const std::size_t rows = std::min(mr, mc - i0);
        W* strip               = dst + i0 * kw;
        std::size_t r          = 0;
        for (; r + L <= rows; r += L) {
            const In* src = A + (i0 + r) * lda;
            std::size_t w = 0;
            for (; w + L <= full; w += L) {
                const auto* p = reinterpret_cast<const std::uint8_t*>(src + w * G);
                auto* out     = reinterpret_cast<std::uint8_t*>(strip + w * mr + r);
                const std::size_t out_row = mr * sizeof(W);
                if constexpr (L == 4) {
                    // 4×4 transpose: swap single words, then 2-word halves.
                    const uint32x4_t r0 = vreinterpretq_u32_u8(vld1q_u8(p));                  // a0 a1 a2 a3
                    const uint32x4_t r1 = vreinterpretq_u32_u8(vld1q_u8(p + row_bytes));      // b0 b1 b2 b3
                    const uint32x4_t r2 = vreinterpretq_u32_u8(vld1q_u8(p + 2 * row_bytes));  // c0 c1 c2 c3
                    const uint32x4_t r3 = vreinterpretq_u32_u8(vld1q_u8(p + 3 * row_bytes));  // d0 d1 d2 d3
                    const uint32x4_t t0 = vtrn1q_u32(r0, r1);                                 // a0 b0 a2 b2
                    const uint32x4_t t1 = vtrn2q_u32(r0, r1);                                 // a1 b1 a3 b3
                    const uint32x4_t t2 = vtrn1q_u32(r2, r3);                                 // c0 d0 c2 d2
                    const uint32x4_t t3 = vtrn2q_u32(r2, r3);                                 // c1 d1 c3 d3
                    vst1q_u8(out, vreinterpretq_u8_u32(vcombine_u32(vget_low_u32(t0), vget_low_u32(t2))));    // a0 b0 c0 d0
                    vst1q_u8(out + out_row, vreinterpretq_u8_u32(vcombine_u32(vget_low_u32(t1), vget_low_u32(t3))));
                    vst1q_u8(out + 2 * out_row,
                             vreinterpretq_u8_u32(vcombine_u32(vget_high_u32(t0), vget_high_u32(t2))));
                    vst1q_u8(out + 3 * out_row,
                             vreinterpretq_u8_u32(vcombine_u32(vget_high_u32(t1), vget_high_u32(t3))));
                } else {
                    const uint64x2_t r0 = vreinterpretq_u64_u8(vld1q_u8(p));
                    const uint64x2_t r1 = vreinterpretq_u64_u8(vld1q_u8(p + row_bytes));
                    vst1q_u8(out, vreinterpretq_u8_u64(vtrn1q_u64(r0, r1)));
                    vst1q_u8(out + out_row, vreinterpretq_u8_u64(vtrn2q_u64(r0, r1)));
                }
            }
            for (; w < full; ++w)
                for (std::size_t rr = 0; rr < L; ++rr)
                    strip[w * mr + r + rr] = load_word(src + rr * lda + w * G);
        }
        for (; r < rows; ++r)
            for (std::size_t w = 0; w < full; ++w)
                strip[w * mr + r] = load_word(A + (i0 + r) * lda + w * G);
        if constexpr (G > 1)
            if (full < kw)  // the last, partial k group: what exists, then zeros
                for (std::size_t rr = 0; rr < rows; ++rr) {
                    In g[G] = {};
                    std::memcpy(g, A + (i0 + rr) * lda + full * G, (kc - full * G) * sizeof(In));
                    strip[full * mr + rr] = load_word(g);
                }
        if (rows < mr)
            for (std::size_t w = 0; w < kw; ++w)
                std::fill(strip + w * mr + rows, strip + (w + 1) * mr, W{0});
    }
}

/// Pack a kc×nc panel of row-major B (ldb elements) into nr-column strips of
/// k-group words, row-major within each strip:
/// dst[s·nr·kw + w·nr + c] = B(G·w … G·w+G−1, s·nr + c), zero-padded to nr
/// columns and to a whole last group. For FP32 / FP64 a strip row is a row
/// of B; otherwise G rows of B are interleaved with NEON zips.
template <typename In>
inline void pack_b(const In* B, std::size_t ldb, std::size_t kc, std::size_t nc, std::size_t nr,
                   sme_word_t<In>* dst) {
    using W                 = sme_word_t<In>;
    constexpr std::size_t G = kSmeG<In>;
    const std::size_t kw    = (kc + G - 1) / G;
    for (std::size_t j0 = 0; j0 < nc; j0 += nr) {
        const std::size_t cols = std::min(nr, nc - j0);
        W* strip               = dst + j0 * kw;
        for (std::size_t w = 0; w < kw; ++w) {
            W* out        = strip + w * nr;
            const In* row = B + w * G * ldb + j0;
            std::size_t c = 0;
            if constexpr (G == 1) {
                std::memcpy(out, row, cols * sizeof(W));
                c = cols;
            } else if ((w + 1) * G <= kc) {  // a whole k group
                const auto* p               = reinterpret_cast<const std::uint8_t*>(row);
                auto* o                     = reinterpret_cast<std::uint8_t*>(out);
                const std::size_t row_bytes = ldb * sizeof(In);
                if constexpr (G == 2) {  // zip two rows of 16-bit elements: 8 columns per step
                    for (; c + 8 <= cols; c += 8) {
                        const uint16x8_t r0 = vreinterpretq_u16_u8(vld1q_u8(p + 2 * c));
                        const uint16x8_t r1 = vreinterpretq_u16_u8(vld1q_u8(p + row_bytes + 2 * c));
                        vst1q_u8(o + 4 * c, vreinterpretq_u8_u16(vzip1q_u16(r0, r1)));
                        vst1q_u8(o + 4 * c + 16, vreinterpretq_u8_u16(vzip2q_u16(r0, r1)));
                    }
                } else {  // interleave four rows of bytes: 16 columns per step
                    for (; c + 16 <= cols; c += 16) {
                        const uint8x16_t r0 = vld1q_u8(p + c), r1 = vld1q_u8(p + row_bytes + c);
                        const uint8x16_t r2 = vld1q_u8(p + 2 * row_bytes + c), r3 = vld1q_u8(p + 3 * row_bytes + c);
                        const uint16x8_t lo01 = vreinterpretq_u16_u8(vzip1q_u8(r0, r1));
                        const uint16x8_t hi01 = vreinterpretq_u16_u8(vzip2q_u8(r0, r1));
                        const uint16x8_t lo23 = vreinterpretq_u16_u8(vzip1q_u8(r2, r3));
                        const uint16x8_t hi23 = vreinterpretq_u16_u8(vzip2q_u8(r2, r3));
                        vst1q_u8(o + 4 * c, vreinterpretq_u8_u16(vzip1q_u16(lo01, lo23)));
                        vst1q_u8(o + 4 * c + 16, vreinterpretq_u8_u16(vzip2q_u16(lo01, lo23)));
                        vst1q_u8(o + 4 * c + 32, vreinterpretq_u8_u16(vzip1q_u16(hi01, hi23)));
                        vst1q_u8(o + 4 * c + 48, vreinterpretq_u8_u16(vzip2q_u16(hi01, hi23)));
                    }
                }
            }
            if constexpr (G > 1)
                for (; c < cols; ++c) {  // columns left over, or a partial last group
                    In g[G] = {};
                    for (std::size_t e = 0; e < G && w * G + e < kc; ++e)
                        g[e] = row[e * ldb + c];
                    out[c] = load_word(g);
                }
            std::fill(out + cols, out + nr, W{0});
        }
    }
}

template <typename T>
inline constexpr bool kF32 = sizeof(T) == 4;

/// ZA tiles per row when two rows of tiles use them all: 2×2 za32 (FP32 and
/// widening) or 2×4 za64 (FP64). macro_kernel has a fast path for that shape.
template <typename T>
inline constexpr int kCols = int(sizeof(sme_word_t<T>)) / 2;

/// ZA[Tile] += a ⊗ b, the outer product of In: FMOPA for FP32 / FP64 (pt
/// all-true); for fp16, bf16 and INT8, a and b hold k-group words and the
/// widening FMOPA / SMOPA adds a G-long dot product into each 32-bit lane
/// (predicates over all 16- or 8-bit elements).
template <typename In, int Tile, typename V>
[[gnu::always_inline]] inline void mopa(svbool_t pt, V a, V b) __arm_streaming __arm_inout("za") {
    if constexpr (std::is_same_v<In, float>)
        svmopa_za32_m(Tile, pt, pt, a, b);
    else if constexpr (std::is_same_v<In, double>)
        svmopa_za64_m(Tile, pt, pt, a, b);
    else if constexpr (std::is_same_v<In, fp16>)
        svmopa_za32_f16_m(Tile, svptrue_b16(), svptrue_b16(), svreinterpret_f16(a), svreinterpret_f16(b));
    else if constexpr (std::is_same_v<In, bf16>)
        svmopa_za32_bf16_m(Tile, svptrue_b16(), svptrue_b16(), svreinterpret_bf16(a), svreinterpret_bf16(b));
    else
        svmopa_za32_s8_m(Tile, svptrue_b8(), svptrue_b8(), svreinterpret_s8(a), svreinterpret_s8(b));
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
template <typename T, int Col, bool Load, bool Full>
[[gnu::always_inline]] inline void move_column(T* c, std::int64_t ldc, std::int64_t s,
                                               std::int64_t rows, std::int64_t cols) __arm_streaming
    __arm_inout("za") {
    if constexpr (Full) {
        const svbool_t pg = kF32<T> ? svptrue_b32() : svptrue_b64();
        move_tile<T, Col, Load>(c + Col * s, ldc, s, pg);
        move_tile<T, kCols<T> + Col, Load>(c + s * ldc + Col * s, ldc, s, pg);
        return;
    }
    const std::int64_t top = std::min(rows, s);
    // Predication suppresses accesses, not C++ pointer arithmetic. Clamp
    // entirely inactive columns and do not form a nonexistent bottom row.
    const std::int64_t offset = std::min<std::int64_t>(Col * s, cols);
    const svbool_t pg      = kF32<T> ? svwhilelt_b32(Col * s, cols) : svwhilelt_b64(Col * s, cols);
    move_tile<T, Col, Load>(c + offset, ldc, top, pg);
    const std::int64_t bottom = rows > top ? s * ldc + offset : 0;
    move_tile<T, kCols<T> + Col, Load>(c + bottom, ldc, rows - top, pg);
}

/// Load (Load=true) or store the whole micro-tile accumulator.
template <typename T, bool Load, bool Full>
[[gnu::always_inline]] inline void move_c_columns(T* c, std::int64_t ldc, std::int64_t s, std::int64_t rows,
                                          std::int64_t cols) __arm_streaming __arm_inout("za") {
    move_column<T, 0, Load, Full>(c, ldc, s, rows, cols);
    move_column<T, 1, Load, Full>(c, ldc, s, rows, cols);
    if constexpr (kCols<T> == 4) {
        move_column<T, 2, Load, Full>(c, ldc, s, rows, cols);
        move_column<T, 3, Load, Full>(c, ldc, s, rows, cols);
    }
}

// Each tile number is an instruction immediate. Expand at compile time.
template <typename T, int V, bool Load, int Tile>
[[gnu::always_inline]] inline void move_shaped_tile(
    T* c, std::int64_t ldc, std::int64_t s, std::int64_t rows,
    std::int64_t cols) __arm_streaming __arm_inout("za") {
    constexpr int row = Tile / V, col = Tile % V;
    if (rows <= row * s)
        return;
    const auto offset = std::min<std::int64_t>(col * s, cols);
    const svbool_t pg = kF32<T> ? svwhilelt_b32(col * s, cols) : svwhilelt_b64(col * s, cols);
    move_tile<T, Tile, Load>(c + row * s * ldc + offset, ldc, std::min(s, rows - row * s), pg);
}

template <typename T, int V, bool Load, std::size_t... I>
[[gnu::always_inline]] inline void move_shape(
    T* c, std::int64_t ldc, std::int64_t s, std::int64_t rows, std::int64_t cols,
    std::index_sequence<I...>) __arm_streaming __arm_inout("za") {
    (move_shaped_tile<T, V, Load, I>(c, ldc, s, rows, cols), ...);
}

// Keep the common full-tile transfers free of edge-address bookkeeping.
// Both paths remain inlined into the ZA owner, without mode switches.
template <typename T, bool Load, int R = 2, int V = kCols<T>>
[[gnu::always_inline]] inline void move_c(T* c, std::int64_t ldc, std::int64_t s, std::int64_t rows,
                                          std::int64_t cols) __arm_streaming __arm_inout("za") {
    if constexpr (R == 2 && V == kCols<T>) {
        if (rows == 2 * s && cols == kCols<T> * s)
            move_c_columns<T, Load, true>(c, ldc, s, rows, cols);
        else
            move_c_columns<T, Load, false>(c, ldc, s, rows, cols);
    } else {
        move_shape<T, V, Load>(c, ldc, s, rows, cols, std::make_index_sequence<R * V>{});
    }
}

template <typename T, int Row, int V, typename A, std::size_t... Col, typename... B>
[[gnu::always_inline]] inline void outer_row(svbool_t pt, A a, std::index_sequence<Col...>,
                                             B... b) __arm_streaming __arm_inout("za") {
    (mopa<T, Row * V + Col>(pt, a, b), ...);
}

template <typename T, int V, std::size_t... Row, typename... B>
[[gnu::always_inline]] inline void outer_rows(svbool_t pt, const sme_word_t<T>* a, std::int64_t s,
                                              std::index_sequence<Row...>,
                                              B... b) __arm_streaming __arm_inout("za") {
    (outer_row<T, Row, V>(pt, svld1(pt, a + Row * s), std::make_index_sequence<V>{}, b...), ...);
}

// Sizeless vectors travel as a parameter pack: V B loads reused by R rows.
// Clamp inactive B columns before forming pointers, even with false predicates.
template <typename T, int R, int V, std::size_t... Col>
[[gnu::always_inline]] inline void outer_step(
    svbool_t pt, const sme_word_t<T>* a, const sme_word_t<T>* b, std::int64_t s, std::int64_t cols,
    std::index_sequence<Col...>) __arm_streaming __arm_inout("za") {
    outer_rows<T, V>(pt, a, s, std::make_index_sequence<R>{},
                     svld1(kF32<sme_word_t<T>> ? svwhilelt_b32(std::int64_t(Col) * s, cols)
                                               : svwhilelt_b64(std::int64_t(Col) * s, cols),
                           b + std::min<std::int64_t>(Col * s, cols))...);
}

template <typename T, int R, int V>
constexpr bool check_shape() {
    static_assert(R > 0 && V > 0, "SME tile dimensions must be positive");
    static_assert(R * V <= int(sizeof(sme_word_t<T>)),
                  "SME shape exceeds available ZA tiles (4 32-bit: FP32 and widening, 8 FP64)");
    return true;
}

/// Macro-kernel (streaming): C[mc×nc] (+)= Apack · B over kw k groups (kw =
/// kc for FP32 / FP64), R×V ZA tiles. A is packed
/// (pack_a). Row w of the B strip starting at column jr is at B + jr·b_step +
/// w·ldb: packed strips (b_step = kw, ldb = nr) for gemm_sme, or B in place
/// (b_step = 1, ldb = N; FP32 / FP64 only) for gemm_sme_tiles and
/// gemm_sme_blocked. B loads are predicated to the strip's real columns, so
/// reading B in place never runs past the end of a row. C is T, or FP32 /
/// INT32 for the widening types; ZA moves are bit copies.
template <typename T, int R, int V>
__arm_locally_streaming __arm_new("za")
void macro_kernel(const sme_word_t<T>* __restrict Ap, const sme_word_t<T>* __restrict B, std::int64_t b_step,
                  std::int64_t ldb, sme_out_t<T>* __restrict C, std::int64_t ldc, std::int64_t mc,
                  std::int64_t nc, std::int64_t kw, bool accumulate) {
    using W   = sme_word_t<T>;
    using Out = sme_out_t<T>;
    static_assert(check_shape<T, R, V>());
    constexpr int kC      = V;
    const std::int64_t s  = kF32<W> ? svcntsw() : svcntsd();
    const std::int64_t mr = R * s, nr = kC * s;
    const svbool_t pt     = kF32<W> ? svptrue_b32() : svptrue_b64();
    const svcount_t pn    = kF32<W> ? svptrue_c32() : svptrue_c64();

    for (std::int64_t jr = 0; jr < nc; jr += nr) {
        const std::int64_t cols = std::min(nr, nc - jr);
        const W* b_strip        = B + jr * b_step;
        // Columns [0, 2s) and, for f64, [2s, 4s) of the strip.
        const svcount_t pb0 = kF32<W> ? svwhilelt_c32(std::int64_t{0}, cols, 2)
                                      : svwhilelt_c64(std::int64_t{0}, cols, 2);
        const svcount_t pb1 = svwhilelt_c64(2 * s, cols, 2);

        for (std::int64_t ir = 0; ir < mc; ir += mr) {
            const std::int64_t rows = std::min(mr, mc - ir);
            const W* a_strip        = Ap + ir * kw;
            Out* c                  = C + ir * ldc + jr;

            if (accumulate)
                move_c<Out, true, R, V>(c, ldc, s, rows, cols);
            else
                svzero_za();

            for (std::int64_t k = 0; k < kw; ++k) {
                if constexpr (R == 2 && V == kCols<T>) {
                    // SME2 multi-vector loads: one instruction fetches 2 vectors.
                    const auto a  = svld1_x2(pn, a_strip + k * mr);
                    const auto b  = svld1_x2(pb0, b_strip + k * ldb);
                    const auto a0 = svget2(a, 0), a1 = svget2(a, 1);
                    mopa<T, 0>(pt, a0, svget2(b, 0));
                    mopa<T, 1>(pt, a0, svget2(b, 1));
                    mopa<T, kC + 0>(pt, a1, svget2(b, 0));
                    mopa<T, kC + 1>(pt, a1, svget2(b, 1));
                    if constexpr (kC == 4) {  // f64: columns 2–3 use the other 4 za64 tiles
                        const auto b2 = svld1_x2(pb1, b_strip + k * ldb + (cols > 2 * s ? 2 * s : 0));
                        mopa<T, 2>(pt, a0, svget2(b2, 0));
                        mopa<T, 3>(pt, a0, svget2(b2, 1));
                        mopa<T, 6>(pt, a1, svget2(b2, 0));
                        mopa<T, 7>(pt, a1, svget2(b2, 1));
                    }
                } else {
                    outer_step<T, R, V>(pt, a_strip + k * mr, b_strip + k * ldb,
                                       s, cols, std::make_index_sequence<V>{});
                }
            }

            move_c<Out, false, R, V>(c, ldc, s, rows, cols);
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
 * @brief C = A · B on ARM SME, step 2 of 4: R×V ZA tiles (gemm_sme's
 *        macro_kernel run once over the whole problem), A packed into
 *        R·SVL-row strips, B read in place, no cache blocking. See the file
 *        header.
 */
template <typename T, int R, int V>
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
    static_assert(sme_detail::check_shape<T, R, V>());
    const Matrix<T> a_pack = sme_detail::pack_all_a(A, R * s);
    const auto n = static_cast<std::int64_t>(N);
    sme_detail::macro_kernel<T, R, V>(a_pack.data(), B.data(), 1, n, C.data(), n,
                                static_cast<std::int64_t>(M), n, static_cast<std::int64_t>(K),
                                false);
}

namespace sme_detail {

/// GotoBLAS driver shared by gemm_sme_blocked (PackB = false) and gemm_sme
/// (PackB = true); the only difference is where the macro-kernel reads B.
/// B is K×N with rows ldb ≥ N elements apart (B(k, j) = B[k·ldb + j]); the
/// public kernels pass ldb = N, the BStride benchmarks pad it. T is any of
/// the five input types; TileK is in elements, a whole number of k groups.
/// B in place is FP32 / FP64 only: the widening instructions need B's k
/// groups contiguous, which only packing gives.
template <typename T, bool PackB, int R, int V, std::size_t TileM, std::size_t TileK, std::size_t TileN>
void blocked_driver(const Matrix<T>& A, const T* B, std::size_t ldb, Matrix<sme_out_t<T>>& C) {
    using W                 = sme_word_t<T>;
    constexpr std::size_t G = kSmeG<T>;
    static_assert(check_shape<T, R, V>());
    static_assert(TileM > 0 && TileK > 0 && TileN > 0, "SME cache blocks must be positive");
    static_assert(TileK % G == 0, "TileK must be a whole number of k groups");
    static_assert(PackB || G == 1, "the widening kernels need B packed");
    const std::size_t M = A.rows(), K = A.cols(), N = C.cols();
    assert(C.rows() == M && ldb >= N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;

    // Micro-tile is mr × nr; RDSVL (svcnts*) is legal outside streaming mode.
    const std::size_t s  = kF32<W> ? svcntsw() : svcntsd();
    const std::size_t mr = R * s, nr = V * s;
    auto round_up        = [](std::size_t x, std::size_t m) { return (x + m - 1) / m * m; };
    const std::size_t kw_max = (std::min(TileK, K) + G - 1) / G;
    Matrix<W> a_pack(round_up(std::min(TileM, M), mr), kw_max);  // 64-byte aligned scratch
    Matrix<W> b_pack(PackB ? kw_max : 0, PackB ? round_up(std::min(TileN, N), nr) : 0);

    for (std::size_t jc = 0; jc < N; jc += TileN) {
        const std::size_t nc = std::min(TileN, N - jc);
        for (std::size_t pc = 0; pc < K; pc += TileK) {
            const std::size_t kc = std::min(TileK, K - pc), kw = (kc + G - 1) / G;
            // B strip rows: packed, nr apart (b_step = kw); in place, ldb apart.
            const W* b;
            std::int64_t b_step = 1, b_ld = static_cast<std::int64_t>(ldb);
            if constexpr (PackB) {
                pack_b(B + pc * ldb + jc, ldb, kc, nc, nr, b_pack.data());
                b      = b_pack.data();
                b_step = static_cast<std::int64_t>(kw);
                b_ld   = static_cast<std::int64_t>(nr);
            } else {
                b = B + pc * ldb + jc;  // G = 1: a word is an element
            }
            for (std::size_t ic = 0; ic < M; ic += TileM) {
                const std::size_t mc = std::min(TileM, M - ic);
                pack_a(A.data() + ic * K + pc, K, mc, kc, mr, a_pack.data());
                macro_kernel<T, R, V>(a_pack.data(), b, b_step, b_ld, C.data() + ic * N + jc,
                                static_cast<std::int64_t>(N), static_cast<std::int64_t>(mc),
                                static_cast<std::int64_t>(nc), static_cast<std::int64_t>(kw),
                                pc > 0);
            }
        }
    }
}


}  // namespace sme_detail

/**
 * @brief C = A · B on ARM SME, step 3 of 4: gemm_sme without pack_b (R×V ZA
 *        tiles, cache blocking, B read in place). See the file header,
 *        "Packing", for what that costs at power-of-two N.
 */
template <typename T, int R, int V, std::size_t TileM, std::size_t TileK, std::size_t TileN>
void gemm_sme_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sme_blocked: T must be float or double");
    assert(B.rows() == A.cols() && B.cols() == C.cols());
    sme_detail::blocked_driver<T, false, R, V, TileM, TileK, TileN>(A, B.data(), B.cols(), C);
}

/**
 * @brief C = A · B on ARM SME, step 4 of 4: R×V ZA tiles, cache blocking,
 *        A and B packed. See the file header, "gemm_sme". T is float or
 *        double, or fp16, bf16, int8_t with the widening outer products
 *        (FP32 / INT32 C; "Low precision" in the file header).
 */
template <typename T, int R, int V, std::size_t TileM, std::size_t TileK, std::size_t TileN>
void gemm_sme(const Matrix<T>& A, const Matrix<T>& B, Matrix<sme_out_t<T>>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, fp16> ||
                      std::is_same_v<T, bf16> || std::is_same_v<T, std::int8_t>,
                  "gemm_sme: T must be float, double, fp16, bf16 or std::int8_t");
    assert(B.rows() == A.cols() && B.cols() == C.cols());
    sme_detail::blocked_driver<T, true, R, V, TileM, TileK, TileN>(A, B.data(), B.cols(), C);
}

/**
 * @brief C = A · B on several SME units at once: gemm_sme's blocking with
 *        the B panel packed once and shared, and A blocks handed out to
 *        `threads` threads on demand.
 *
 * An SME unit belongs to a P-cluster, not to a core: the cores of a cluster
 * share it, and M4 Max has two P-clusters, so one thread reaches at most one
 * unit's throughput (gemm_sme). macOS has no API to pin a thread to a
 * cluster, and the scheduler tends to keep a few busy threads on one
 * cluster (so on one unit) until it is full; with equal static shares, the
 * unit that got more threads then finishes last. So the work is split
 * dynamically, as in multi-threaded GotoBLAS:
 *
 *   for jc, pc (as gemm_sme)
 *     every thread packs every team-th B strip  → one shared B panel
 *     barrier
 *     each thread takes the next chunk of rows (atomic counter), packs its
 *     A block and runs macro_kernel on it, until no chunk is left
 *     barrier                                    (B panel reused next pass)
 *
 * A unit with more threads simply takes more chunks. Chunks are TileM rows
 * or fewer, so that every thread gets about four (at least one micro-tile,
 * 2·SVL rows, each); each thread has its own A buffer, and chunks write
 * disjoint rows of C. Workers run at user-interactive QoS, which keeps them
 * on P-cores. Threads are created per call (tens of µs), so small products
 * should use gemm_sme. M4's measured thread count and A block height
 * (8, and gemm_sme's TileM): docs/kernels/sme.md, "Several SME units".
 * T, R, V and the blocks: as gemm_sme.
 */
template <typename T, int R, int V, std::size_t TileM, std::size_t TileK, std::size_t TileN>
void gemm_sme_mt(const Matrix<T>& A, const Matrix<T>& B, Matrix<sme_out_t<T>>& C, unsigned threads) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, fp16> ||
                      std::is_same_v<T, bf16> || std::is_same_v<T, std::int8_t>,
                  "gemm_sme_mt: T must be float, double, fp16, bf16 or std::int8_t");
    using W                 = sme_word_t<T>;
    constexpr std::size_t G = kSmeG<T>;
    static_assert(sme_detail::check_shape<T, R, V>());
    static_assert(TileM > 0 && TileK > 0 && TileN > 0, "SME cache blocks must be positive");
    static_assert(TileK % G == 0, "TileK must be a whole number of k groups");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    if (K == 0)
        C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;

    const std::size_t s  = sme_detail::kF32<W> ? svcntsw() : svcntsd();
    const std::size_t mr = R * s, nr = V * s;
    auto round_up        = [](std::size_t x, std::size_t m) { return (x + m - 1) / m * m; };
    const std::size_t want   = std::max<std::size_t>(threads, 1);
    const std::size_t chunk  = std::clamp(round_up((M + 4 * want - 1) / (4 * want), mr), mr,
                                          round_up(TileM, mr));
    const std::size_t chunks = (M + chunk - 1) / chunk;
    const std::size_t team   = std::min(want, chunks);
    const std::size_t kw_max = (std::min(TileK, K) + G - 1) / G;
    const std::size_t passes = ((N + TileN - 1) / TileN) * ((K + TileK - 1) / TileK);

    Matrix<W> b_pack(kw_max, round_up(std::min(TileN, N), nr));
    std::vector<std::atomic<std::size_t>> next(passes);  // chunk counter per (jc, pc) pass
    std::barrier sync(static_cast<std::ptrdiff_t>(team));

    auto worker = [&](std::size_t t) {
#if defined(__APPLE__)
        // New threads start at default QoS and may run on E-cores; the
        // caller's thread keeps its own QoS.
        if (t > 0)
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
        Matrix<W> a_pack(chunk, kw_max);
        std::size_t pass = 0;
        for (std::size_t jc = 0; jc < N; jc += TileN) {
            const std::size_t nc = std::min(TileN, N - jc);
            for (std::size_t pc = 0; pc < K; pc += TileK, ++pass) {
                const std::size_t kc = std::min(TileK, K - pc), kw = (kc + G - 1) / G;
                for (std::size_t j0 = t * nr; j0 < nc; j0 += team * nr)
                    sme_detail::pack_b(B.data() + pc * N + jc + j0, N, kc, std::min(nr, nc - j0), nr,
                                       b_pack.data() + j0 * kw);
                sync.arrive_and_wait();
                for (std::size_t c; (c = next[pass].fetch_add(1, std::memory_order_relaxed)) < chunks;) {
                    const std::size_t ic = c * chunk, mc = std::min(chunk, M - ic);
                    sme_detail::pack_a(A.data() + ic * K + pc, K, mc, kc, mr, a_pack.data());
                    sme_detail::macro_kernel<T, R, V>(
                        a_pack.data(), b_pack.data(), static_cast<std::int64_t>(kw),
                        static_cast<std::int64_t>(nr), C.data() + ic * N + jc, static_cast<std::int64_t>(N),
                        static_cast<std::int64_t>(mc), static_cast<std::int64_t>(nc),
                        static_cast<std::int64_t>(kw), pc > 0);
                }
                sync.arrive_and_wait();
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(team - 1);
    for (std::size_t t = 1; t < team; ++t)
        pool.emplace_back(worker, t);
    worker(0);
    for (std::thread& th : pool)
        th.join();
}

#endif  // HPC_HAS_SME

}  // namespace hpc::gemm
