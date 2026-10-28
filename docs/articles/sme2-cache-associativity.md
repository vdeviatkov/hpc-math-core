---
title: "A Power-of-Two Stride Cuts SME2 GEMM Throughput by Two Thirds on Apple M4"
description: "An SME2 matrix multiplication on Apple M4 Max runs at a third of its speed when B's rows are 16 KiB apart. Controlled experiments point to L2 set conflicts; padding or packing B fixes it, and packing wins. The same experiment on AMD Zen 5 finds the same mechanism at a smaller cost: 14–17%, from L1 sets."
date: 2026-10-12
series_part: 3
permalink: /articles/sme2-cache-associativity.html
---
# A power-of-two stride cuts SME2 GEMM throughput by two thirds on Apple M4

*Apple M4 Max, one thread, FP32, and for comparison one AMD Zen 5 core.
Part 3 of a series. How the kernel is built, and the machine's caches,
TLBs and SME unit: [part 2, Matrix multiplication on Apple M4's SME2
unit](sme2-gemm.md).*

A matrix multiplication written for ARM's Scalable Matrix Extension (SME2)
reached **1,494 GFLOP/s at N = 4000 and 480 GFLOP/s at N = 4096**. The larger
problem does 7% more work, and its throughput drops by 68%. This article traces that
cliff to its cause and measures both fixes.

**Key results**

- The slowdown follows the **byte distance between rows of B**, not the
  matrix size: keeping N = 4096 and padding each row of B by 128 bytes
  makes it 3.1× faster with the same arithmetic.
- At a 16 KiB row stride, the B rows the kernel reuses compete for a small
  part of the cache. The kernel collapses between 576 and 640 rows per
  reused strip, far past the 8 lines L1 keeps at this stride, and the CPU
  core's own loads hit a limit of similar size: the evidence points to
  **L2 set conflicts**.
- **Packing B** into contiguous strips removes the cliff for any stride. It
  costs 3–5.5% of the runtime (2.5–4.9 ms of about 90 ms) and cuts compute
  time by 8–9% even when the stride is already padded.
- **On AMD Zen 5 the same mechanism costs much less.** Part 1's AVX-512
  kernel (then with an 8 × 2 tile) loses 14–17% at N = 1024, 2048 and
  4096 against the same rows padded by 64 B, against 68% here. Every
  multiple of 4 KiB folds a strip's rows onto 12 lines of L1; the L2
  limits behind the M4's cliff are there too, but a fast L3 backs them.
  Packing B in GotoBLAS order ([part 1, Step 5](simd-gemm.md#step-5-pack-b))
  removes it there too.

## The kernel and the B operand

The kernel computes `C = A × B` for row-major matrices. SME's `FMOPA`
instruction adds the outer product of two vectors into ZA, a 2-D
accumulator. With M4's 512-bit streaming vector length (16 FP32 lanes), ZA
holds four 16 × 16 FP32 tiles, which together accumulate a 32 × 32 block
of C:

```text
for k in 0 .. Kc-1:
    a = A[i .. i+31, k]                  // column segment of A: 2 vectors
    b = B[k, j .. j+31]                  // row segment of B: 2 vectors, one svld1_x2
    C[i .. i+31, j .. j+31] += a ⊗ b     // 4 FMOPAs, one per ZA tile
```

A's 32 values are a row apart and streaming mode has no gather load, so A
is always packed into strips where each column segment is contiguous. B's
32 values are already adjacent, so B can be read **in place**: row `k` of
the segment is at `B + k·ldb + j`. Or B can be **packed**: each 32-column
strip of a `Kc`-row block is copied into a contiguous buffer, 128 bytes per
row ([`pack_b`](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/sme.hpp)).

Both variants run one driver, `sme_detail::blocked_driver<float, PackB>`
([source](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/sme.hpp)),
with GotoBLAS cache blocking `Mc = 1024`, `Kc = 2048`, `Nc = 4096`. Only the
B source differs:

| | B in place | B packed |
|---|---|---|
| A packing, blocking, ZA tiles, FMOPAs | identical | identical |
| Row k of a B strip | `B + k·ldb + j` | `strip + k·128 B` |
| Cost of copying B | none | included in every timing |

The quantity that matters is the **B strip**: 2,048 rows × 32 floats =
256 KiB. The macro-kernel streams one strip against the 32 A strips of an
Mc block, one after another, so the strip must stay cached between those
32 passes. Packed, the strip is 256 KiB of contiguous memory. In place at
`ldb = 4096`, it is 2,048 pieces of 128 bytes spaced 16 KiB apart, spread
over 32 MiB of address space.

## N = 4096 runs at a third of the speed of N = 4000

Square matrices around 2048 and 4096, so the row stride is `N × 4` bytes.
Median of 3 interleaved repetitions; each timed call includes scratch
allocation and all packing. (This is a separate run from the step tables
in part 2; the same kernels agree between the two within 2–3%.)

![Throughput near N = 2048 and 4096, B in place and packed](assets/sme2-size-sweep.svg)

| N | B row stride | B in place, GFLOP/s | B packed, GFLOP/s |
|---:|---:|---:|---:|
| 1984 | 7,936 B | 1,570 | 1,561 |
| 2016 | 8,064 B | 1,572 | 1,555 |
| **2048** | **8,192 B** | **636** | 1,485 |
| 2080 | 8,320 B | 1,462 | 1,440 |
| 4000 | 16,000 B | 1,494 | 1,515 |
| 4032 | 16,128 B | 1,489 | 1,529 |
| 4064 | 16,256 B | 1,476 | 1,525 |
| **4096** | **16,384 B** | **480** | 1,506 |
| 4128 | 16,512 B | 1,356 | 1,433 |

The dips sit exactly on the powers of two: −60% at 2048 and −68% at 4096
against the size just below, mostly recovered 32 columns either side. Packed
B shows no cliff at all.

## Same multiplication, different stride: the cliff follows the bytes

Changing N also changes the work and the memory footprint. The decisive
test fixes `M = N = K = 4096` and stores B in a wider buffer, so only the
row stride `ldb` changes. The padding columns are never read
(`BStride` in [`bench_gemm.cpp`](https://github.com/vdeviatkov/hpc-math-core/blob/main/benchmarks/bench_gemm.cpp)).

![Same 4096 × 4096 multiplication with four leading dimensions of B](assets/sme2-padding.svg)

| ldb | B row stride | B in place, GFLOP/s | B packed, GFLOP/s |
|---:|---:|---:|---:|
| **4096** | **16,384 B** | **480** | 1,506 |
| 4097 | 16,388 B | 1,283 | 1,527 |
| 4128 | 16,512 B | 1,465 | 1,553 |
| 4160 | 16,640 B | 1,498 | 1,532 |

One extra cache line per row (`ldb = 4128`) makes B 0.5 MiB larger and the
in-place kernel **3.1× faster**. The arithmetic and useful data are
unchanged; the allocation and page footprint grow by about 0.8%.
`ldb = 4097` recovers less because its 128-byte row segments no longer
start on a cache-line boundary, so each one spans two lines.

## Why 16 KiB is special: rows competing for one cache set

A set-associative cache stores each line in one set chosen from its
address, and each set holds a fixed number of lines (its ways). In the
simplest scheme, modulo indexing, the set is taken directly from the
address bits just above the line offset: with 128-byte lines and 128
sets, a line's set is its line number modulo 128. In such a cache (an
illustrative model, not Apple's documented L2 mapping), the rows of one B
strip land like this:

```text
B in place, rows 16 KiB apart          B packed, rows 128 B apart
(16,384 B = 128 lines per row)         (one line per row)

  row 0     ──►  set 0                   row 0     ──►  set 0
  row 1     ──►  set 0                   row 1     ──►  set 1
  row 2     ──►  set 0                   row 2     ──►  set 2
  row 3     ──►  set 0                   row 3     ──►  set 3
   …                                      …
  row 2047  ──►  set 0                   row 2047  ──►  set 127

  2,048 rows share one set's ways        16 rows per set, every set used
```

Every in-place row starts exactly 128 lines after the previous one, so in
this model all of them compete for the few ways of a single set, and the
rest of the cache cannot help this strip. (The model has the 128 sets of
the 128 KiB L1. In the 16 MiB L2 the same stride would leave a fraction
of about 1/128 of the cache to these rows: roughly a thousand lines, not
8; the measurements below find about 450–640.) Packed rows walk through
the sets one by one. Padding B by one line per row (16,512 B, 129 lines) also spreads
the rows: each one moves one set further.

More generally, rows `s` bytes apart, with `s` a multiple of the 128-byte
line, step through the sets `s / 128` at a time and so occupy only
`S / gcd(s / 128, S)` of a cache's `S` sets. For a modulo-indexed cache
with 128 sets:

| Row stride | Stride in lines | Sets the strip's rows use |
|---:|---:|---:|
| 16,384 B | 128 | 1 |
| 8,192 B | 64 | 2 |
| 16,000 B | 125 | all 128 |
| 16,512 B | 129 | all 128 |
| 128 B (packed) | 1 | all 128 |

At a power-of-two stride every row of the strip lands in the same few sets.
Once the strip has more rows than those sets have ways, it evicts itself
before the next A strip comes back for it: **conflict misses** in a cache
that has plenty of total capacity. One line more of stride spreads the rows
over every set, which matches what the padding experiment shows. An
8 KiB stride gives the rows twice as many sets as 16 KiB, but a
2,048-row strip overflows those too, which is consistent with N = 2048
losing almost as much as N = 4096.

This calculation illustrates the conflict mechanism for a conventional
modulo-indexed cache. Modern processors can instead hash higher address
bits into the set index ([how each level picks its set](#how-an-address-picks-its-set-l1-l2-and-llc-slices)), and Apple does not document the set-indexing
function of M4's L2. The table is therefore a model of the mechanism, not
a reconstruction of the M4 Max's physical cache organization. The evidence
for conflicts on this machine is the controlled stride and working-set
experiments: the padding test above and the capacity measurements below.

## Locating the conflicts: L2, not L1

Which cache? The number of rows a strip can have before the conflict bites
measures the capacity of the conflicting sets (shared with whatever else
the kernel keeps there, such as the A block), so the next experiment
varies the strip height `Kc` while timing only the macro-kernel
([`sme_b_stride_probe`](https://github.com/vdeviatkov/hpc-math-core/blob/main/benchmarks/experiments/sme_b_stride_probe.cpp); A packed
and B laid out once, N = 1024, Mc = 1024, 9 shuffled repetitions).

![Macro-kernel throughput against rows per B strip, rows 16 KiB vs 16,512 B apart](assets/sme2-strip-height.svg)

| Rows per strip | rows 16,384 B apart | rows 16,512 B apart | ratio |
|---:|---:|---:|---:|
| 64 | 630 | 786 | 0.80 |
| 256 | 1,200 | 1,434 | 0.84 |
| 512 | 1,267 | 1,652 | 0.77 |
| 576 | 1,236 | 1,677 | 0.74 |
| **640** | **946** | 1,668 | **0.57** |
| 1024 | 626 | 1,698 | 0.37 |
| 2048 | 500 | 1,584 | 0.32 |

From 64 to 576 rows the power-of-two stride costs 14–26%; between 576 and
640 rows it breaks, and from there it degrades with every row added, to a
third of the padded throughput at the library's Kc = 2,048. The pointer
chase below shows L1 keeping only 8 lines at this stride (consistent with
8-way sets, an inference: the associativity is not documented). The
conflicting structure holds roughly 600 rows, far more than L1 keeps here,
so the conflicting sets must be L2's.

A pointer chase on a CPU core, one dependent load per line, measures the
capacity at this stride independently of SME (cycles per load at
3.99 GHz; best of 5, from an earlier session than the other experiments):

| Lines, 16 KiB apart | 8 | 16 | 448 | 512 | 768 | 1,024 | 2,048 |
|---|--:|--:|--:|--:|--:|--:|--:|
| cycles per load | 3 | 24 | 33 | 41 | 52 | 100 | 144 |

Only 8 lines stay in L1 (3 cycles). Up to 448 lines the loads hit L2:
about 24 cycles (part 2's machine table, from a different chase, gives
26–27), and about 33 from 256 lines on, where each line, alone on
its own 16 KiB page, also misses the 128-entry first-level TLB. In the
archived run the next step comes between 448 and 512 lines (it moved up
to 640 in other, unarchived runs); beyond that, loads go further out. The
SME kernel breaks at a similar size, between 576 and 640 rows. The two
measurements agree on the scale: a power-of-two stride leaves this data a
few hundred lines of a 131,072-line L2.

What this does and does not prove: the matching capacity thresholds, in
two independent experiments, strongly suggest that L2 conflicts cause the
throughput collapse. Neither measurement reveals individual cache-set
mappings, and neither replaces hardware performance counters; what they
establish is a capacity scale consistent with L2 rather than L1. The
mechanism is inferred from controlled measurements, not observed
directly.

## Fix 1: pad the leading dimension

Padding B's rows by one cache line removes the cliff: 480 → 1,465 GFLOP/s
at N = 4096. It costs 0.8% more memory and no code. It is not a complete
fix, for two reasons:

- The caller owns B's layout. A library receiving a matrix with a 16 KiB
  stride cannot pad it without copying it, and a copy is what packing is.
- Even padded, B in place trails packed B by 4–8% (6–8% from N = 3072
  up). Macro-kernel only, `ldb = N + 32`:

| N | 1024 | 2048 | 3072 | 3584 | 4096 | 4224 |
|---|--:|--:|--:|--:|--:|--:|
| in place / packed | 0.96 | 0.95 | 0.92 | 0.92 | 0.92 | 0.92 |

## Fix 2: pack B, and why it wins

Packing copies each strip once per `Kc` block and lets every reuse read a
contiguous 128-byte stream, whatever the caller's stride. The cost is
measurable directly: the same driver with a clock around each phase
([`sme_b_layout`](https://github.com/vdeviatkov/hpc-math-core/blob/main/benchmarks/experiments/sme_b_layout.cpp)), N =
4096, 15 shuffled repetitions, medians in milliseconds.

| ldb | B | End to end | Pack A | Pack B | Compute |
|---:|:--|--:|--:|--:|--:|
| 4096 | in place | 281.2 | 3.75 | — | 278.9 |
| 4096 | packed | 91.1 | 3.84 | 4.94 | 82.4 |
| 4128 | in place | 92.9 | 3.74 | — | 89.0 |
| 4128 | packed | 88.5 | 3.76 | 2.54 | 82.0 |
| 4160 | in place | 93.2 | 3.77 | — | 89.6 |
| 4160 | packed | 88.4 | 3.81 | 2.49 | 81.7 |

- At the 16 KiB stride, packing costs 4.9 ms and saves 196 ms of compute.
- With an already padded stride, packing costs 2.5 ms and still saves
  7–8 ms; packed B was faster end to end in all 15 repetitions.
- With packed B, compute time is the same 82 ms for every `ldb`: the
  macro-kernel only ever sees the packed layout.

The copy is cheap because each element of B is copied once, while the
compute reads it once per A strip: 32 times per A block, `M / 32` times in
all. At these sizes it pays for itself several times over.

## The same experiment on AMD Zen 5

Part 1's AVX-512 kernel, `gemm_simd_blocked<Zmm>`, also reads B in place, and
it also loses throughput at power-of-two sizes: FP32 at N = 1024 and FP64
at N = 512, both rows 4 KiB apart ([part 1](simd-gemm.md)). The experiments
above, repeated on one core of an AMD Ryzen 9 9950X with the kernel's 8 × 2
register tile and 64-row blocks (its defaults when this was measured; in
`bench_gemm` the library kernel dropped 13% from FP32 N = 512 to 1024 with
them, and drops 5% with today's 8 × 3 tile and 512 × 256 × 512 blocks, see
[docs/kernels](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/kernels/README.md)), show when and how much
([`avx512_b_stride`](https://github.com/vdeviatkov/hpc-math-core/blob/main/benchmarks/experiments/avx512_b_stride.cpp),
data in [experiments/zen5-b-stride](https://github.com/vdeviatkov/hpc-math-core/tree/main/docs/experiments/zen5-b-stride)).
The core runs at 5.71 GHz, pinned to one core of an idle machine; GCC 13.3.

### The caches, and what a stride leaves of them

Zen 5's caches are documented and reported by Linux: 64-byte lines, a
48 KiB L1 with 64 sets of 12 ways, a 1 MiB L2 with 1,024 sets of 16 ways,
and a 32 MiB L3 shared by 8 cores. A modulo-indexed cache gives lines `s`
bytes apart `sets / gcd(s / 64, sets)` of its sets. For L1 every multiple
of 4 KiB (64 lines) is the same: one set, 12 lines. L2 aliases every
64 KiB, so its share halves with each doubling of the stride.

The pointer chase measures it directly. Unlike the M4's, this one runs on
2 MiB huge pages, so a strided walk does not also miss the TLB. Cycles per
load at 5.71 GHz:

| Lines | 12 | 16 | 64 | 96 | 128 | 192 | 256 | 384 | 16,384 | 32,768 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 64 B apart (dense) | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 14 | 43 |
| 4 KiB apart | 4 | **14** | 14 | 14 | 14 | 14 | 14 | **45** | 45 | 197 |
| 8 KiB apart | 4 | **14** | 14 | 14 | 14 | **45** | 45 | 45 | 45 | 275 |
| 16 KiB apart | 4 | **14** | 14 | **45** | 46 | 35 | 46 | 46 | 47 | 174 |
| 4 KiB + 64 B apart | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 37 | 48 |

L1 (4 cycles) keeps exactly 12 lines at every power-of-two stride, and
L2 (14 cycles) 256, 128 and 64 lines at 4, 8 and 16 KiB (16 at 64 KiB):
the modulo model's numbers, to the line. One line more of stride keeps
everything in L1 up to its 768-line capacity. L3 (about 45 cycles) keeps
16,384 lines even at these strides, at least twice what plain modulo
indexing of its 32,768 sets would allow, so its set index is not the
plain address bits. (The 192-line value at 16 KiB, 35, sits between two
levels and is one sample.)

### What it costs the GEMM

The same multiplication with only B's row stride changed, as on the M4,
using the library kernel's blocking of the time (64 × 256 × 512) and its
8 × 2 micro-kernel. This harness is not `bench_gemm`, and its absolute
numbers are about a fifth lower (220 against 282 GFLOP/s at N = 1024);
only the ratios matter here:

| N = M = K | B rows `4N` bytes apart | + 64 B | Change |
|---|--:|--:|--:|
| 1024 (4 KiB) | 220 | 255 | −14% |
| 2048 (8 KiB) | 209 | 250 | −16% |
| 4096 (16 KiB) | 178 | 214 | −17% |

GFLOP/s, median of 5. Changing N instead shows the same dips against the
mean of the neighbouring sizes: −13% at N = 1024 (992: 258, 1024: 222, 1056: 253),
−17% at 2048 and −16% at 4096. The M4 lost 68% at N = 4096; Zen 5 loses a
sixth.

### Which cache: L1 here, not L2

The strip-height sweep separates the two. It changes only how many rows
(Kc) the reused block of B has, for several strides (M = 512, K = 2048,
N = 1024, GFLOP/s):

| Rows per block | 4 KiB | 8 KiB | 16 KiB | 6 KiB | 4,352 B | 4 KiB + 64 B | 8 KiB + 64 B | 16 KiB + 64 B |
|---:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 148 | 204 | 204 | 240 | 253 | 251 | 251 | 250 |
| 64 | 251 | 225 | 207 | 296 | 306 | 305 | 305 | 271 |
| 256 | 274 | 240 | 220 | 313 | 313 | 313 | 308 | 259 |
| 384 | 261 | 202 | 217 | 308 | 311 | 307 | 302 | 256 |
| 512 | 246 | 177 | 170 | 243 | 271 | 296 | 280 | 236 |
| 1024 | 167 | 184 | 140 | 215 | 269 | 279 | 279 | 215 |

(One run per cell, [`height.csv`](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/experiments/zen5-b-stride/height.csv); the CSV also has 32, 48, 96, 128, 192 and 768 rows.)

- **Up to 384 rows**, while the block (rows × 2 KiB) fits L2 by capacity,
  power-of-two strides lose 13–33% against the same stride padded by 64 B
  from 32 rows up (41% at 4 KiB for strips of only 16 rows, which L2 holds
  at any of these strides). A 6 KiB stride, which gives a strip's rows two
  groups of L1 sets (24 lines) instead of one (12), runs within about 3% of
  a fully spread 4,352-byte stride from 64 to 384 rows. The loss follows the L1 limit, which
  is the same at every multiple of 4 KiB.
- **The L2 limits do not show.** If L2 sets were the cause, the loss would
  step up where the block passes what L2 keeps at that stride: 256, 128
  and 64 rows at 4, 8 and 16 KiB. It does not: at 8 KiB, 64 rows (inside
  the limit) lose 26% and 256 rows (twice over) 22%; at 16 KiB, 64 rows
  (at the limit) lose 24% and 256 rows (four times over) 15%. Rows that L2 conflicts evict
  come back from an L3 45 cycles away that does not conflict at these
  strides, and the kernel streams B in order, which the hardware
  prefetchers can follow; on the M4 the level behind L2 is the system
  cache at about 150 cycles (part 2). That explanation is an inference: no
  counters were read.
- **From 512 rows** the block fills L2 itself (1 MiB) with nothing left
  for A and C, and is streamed from L3 at every reuse. Every stride slows
  down, and power-of-two strides lose 17–44% against padded ones, 6 KiB
  11–20% against 4,352 B.

So on Zen 5 the stride matters whenever B's rows are a multiple of 4 KiB
apart, FP32 with N a multiple of 1024 and FP64 with N a multiple of 512,
and cost this kernel 14–17% at its block size of the time; on the M4 the same
mechanism costs two thirds. Padding by one line removes it. Packing
helps too, but less here than on the M4: in this experiment each B block
is packed once per 64-row block of A (the loop order and block of the
time), so packing still reads the strided B N / 64 times. Packed, the GEMM runs at
238, 231 and 202 GFLOP/s at the three power-of-two strides and 269, 269
and 234 when padded. A GotoBLAS loop order, as in part 2, packs each block
of B once; [part 1's Step 5](simd-gemm.md#step-5-pack-b) now does this on
Zen 5, with the current tiles and blocks. At N = 4096 AVX-512 runs at 319
GFLOP/s packed against 294 in place (+8%), and AVX2, whose 4 KiB strides
cost it most, at 167 against 104 (+60%).

## How an address picks its set: L1, L2, and LLC slices

A cache splits each address into three fields: the low bits are the byte
offset within the line, the next bits are the **set index**, and the
rest is the tag that is stored and compared. How each level forms its index
decides which strides conflict, and the levels do it differently.

```text
 physical address:  |        tag        |  set index  | line offset |
 Zen 5 L1 (64 sets, 64 B lines):          bits 11..6     bits 5..0
 M4 L1 (128 sets, 128 B lines):           bits 13..7     bits 6..0
```

**L1: plain bits, inside the page offset.** L1 is on the critical path of
every load, so it looks up the set *while* the TLB translates the address
(virtually indexed, physically tagged, VIPT). That only works if every index
bit lies inside the page offset, where virtual and physical addresses are
identical, and this limits L1 to `page size × ways`:

| | Page | Ways | Largest VIPT L1 | Actual L1 | Set index bits |
|---|--:|--:|--:|--:|---|
| Zen 5 | 4 KiB | 12 | 48 KiB | 48 KiB | 6–11 |
| M4 P-core | 16 KiB | 8 (inferred) | 128 KiB | 128 KiB | 7–13 |

Both L1s sit exactly at this limit. This is also why their conflicts are so
regular. The index is the plain address bits with no hashing (an XOR in front of the
lookup would add latency), and those bits repeat every page. So any stride
that is a multiple of the page size (4 KiB on Zen 5, 16 KiB on the M4) puts
every row into one set, which holds as many lines as the cache has ways:
12 on Zen 5 and 8 on the M4, as both pointer chases measured. Apple does
not document the M4's associativity. The 8 ways are inferred from the
8-line limit and fit the VIPT rule exactly.

**L2: physical bits, usually above the page.** L2 is looked up after
translation (physically indexed, PIPT), and it has more sets than one page
can index. Zen 5's 1,024 sets of 64 B lines need bits 6–15, and bits 12–15
come from the physical page number the OS chose, not from the program's
addresses. So the mapping a program sees depends on its pages:

- With 2 MiB huge pages, as in the Zen 5 pointer chase, physical bits 0–20
  equal the virtual ones, and L2 behaved exactly like the modulo model: 256,
  128 and 64 lines at 4, 8 and 16 KiB.
- With small pages, the upper index bits of each row follow its physical frame.
  Rows a power of two apart in virtual memory may land in more sets or
  fewer, depending on which frames the OS handed out (*page colouring*).
  Run-to-run variation in an L2-bound threshold, like the
  448-to-640-line step on the M4, is what you would expect from this.

On the M4, the 16 MiB L2's index runs well above the 16 KiB page offset,
and Apple documents neither those bits nor whether they are hashed. The
measurements only show that a 16 KiB stride still confines a strip to a few
hundred of L2's 131,072 lines.

**LLC and SLC: first a slice, then a set.** A large last-level cache is not
one array. It is built from **slices**, one per core or per stop on the
on-chip network: one per core on a Zen CCD and on Intel's ring and mesh,
one per home node (HN-F) in Arm's CMN interconnect, and blocks next to the
memory controllers in Apple's SoCs. A lookup happens in two steps:

1. **Slice = hash(physical address).** A fixed function, typically XORs of
   many address bits above the line offset, picks the slice. Its job is to
   spread traffic evenly: consecutive lines, and strided streams, go to
   different slices, so no single slice or network stop becomes a hot spot.
   Vendors do not publish these functions. Intel's were reverse-engineered
   from performance counters (Maurice et al., *Reverse Engineering Intel
   Last-Level Cache Complex Addressing*, RAID 2015). For a memory-side SLC the slice is
   tied to how addresses interleave across memory channels.
2. **Set = index bits within that slice.** These are the low bits above the
   line offset, as in any cache, sometimes with further hashing.

The hash is what makes the last level tolerant of the strides that break
L1 and L2. A stride that maps every row to one set of one slice under plain
indexing is spread over all slices by the hash, which multiplies the space
the rows can use by the number of slices. Zen 5's L3 kept 16,384 lines 4,
8 and 16 KiB apart, at least twice what plain modulo indexing of all its
32,768 sets would allow, so its slice choice and probably its set index
are hashed. That is why it can absorb the rows L2 conflicts drop, as the
next section explains.

## Where an evicted line goes: L1/L2 inclusion and the LLC

The same set conflicts cost the M4 two thirds and Zen 5 a sixth. To see
why, it helps to know what a cache does with a line it evicts, and
which level can still supply it. Four policies decide that.

**Replacement within a set.** When a set is full, the cache picks one of
its ways to evict. L1 and L2 typically use a cheap approximation of
least-recently-used (tree pseudo-LRU and similar); large last-level caches
often use adaptive policies of the RRIP family, which keep a line that has been reused
over one that has not. Every one of them evicts a line in a full set,
however hot it is. That is why the strided strip evicts itself: in a set
holding more strip rows than it has ways, each new row can only replace another row of the
strip.

**Write-back and write-allocate.** CPU caches are write-back: a store
changes only the cached copy and marks it dirty, and the line goes to the
next level only when it is evicted. A store that misses first reads the line in
(write-allocate) unless it is a non-temporal store. Evicting a clean line
can be silent. Evicting a dirty line costs a write to the next level as well.

**Inclusion between levels.** This decides whether a line evicted from one
level still exists in the next:

| Policy | What the outer level holds | Effective capacity | Cost |
|---|---|---|---|
| **Inclusive** | a copy of every line in the inner levels | outer level only | evicting from the outer level must also invalidate the inner copies (*back-invalidation*), even of hot lines |
| **Exclusive** (victim cache) | only lines the inner level evicted; a hit moves the line back in | inner + outer | every inner eviction, clean or dirty, is a fill of the outer level |
| **Non-inclusive** (NINE) | whatever it happens to keep; no guarantee either way | between the two | the coherence protocol needs a separate directory or snoop filter |

Between L1 and L2, inclusion is common and cheap because L2 is many times
larger. AMD documents Zen's L2 as inclusive of its L1 caches, and
Arm's Cortex-A76/Neoverse N1 L2 is strictly inclusive of L1 data. So a
line that falls out of L1 because of a set conflict is usually still in L2,
one L2 hit away (14 cycles on Zen 5). L1 conflicts are therefore cheap. That
is the 14–17% loss Zen 5 shows above.

**The last level.** Here designs diverge, and this is what the experiments
above actually feel:

- **Zen (AMD): an exclusive victim L3.** L3 is filled from lines the L2s of
  its CCD evict, not on the way in from memory, and it is mostly exclusive
  of them. Lines kept in several cores, such as shared code, stay in both. Shadow
  tags of the L2s let the L3 answer coherence requests without probing the
  cores. A line L2 drops because of a conflict goes straight into L3,
  whose sliced and hashed set index (see the pointer chase above) does not
  alias at these strides, and it returns in about 45 cycles.
- **Intel: from inclusive to non-inclusive.** The L3 was inclusive through
  Broadwell: simple coherence, but every L3 eviction back-invalidates the core's copies
  (*inclusion victims*), and L3 capacity was spent duplicating L2.
  Skylake-SP and later server parts enlarged L2 and made L3 non-inclusive,
  with a snoop filter to track the lines held by private caches.
- **Arm and Apple: a system-level cache (SLC).** In Arm reference systems
  (the SLC in the CMN interconnect) and in Apple's SoCs, the last level is
  shared by the CPU clusters, the GPU, and other agents, and sits on the
  memory side, in front of the DRAM controllers. Each address has one home
  slice. A CPU line that misses in L2 goes over the fabric to that slice, and
  a dirty line L2 evicts is written back through it. It is a different kind of cache
  from Zen's L3: farther away (about 150 cycles on the M4 against 45), shared with
  non-CPU traffic, and not sized or managed for one core's working set.
  Apple documents neither the inclusion policy of the M4's L2 nor
  the SLC's allocation policy.

Put together, this explains rule 5 below. On Zen 5 a strip row evicted by
an L1 conflict comes back from the inclusive L2 at 14 cycles, and one
evicted by an L2 conflict comes back from the victim L3 at 45. Every
level the kernel falls back to is close and does not conflict at these strides. On the M4 the
conflicting level is L2 itself, and the next copy of the line is in the SLC
(or DRAM) about 150 cycles away: an L2 conflict reload costs about three
times what it does on Zen 5, and ten times an L1 conflict reload there. As in the rest of this article, this is an inference from
latencies and documented policies, not from counted evictions.

## Limitations

What the experiments do not establish:

- **The L2 mapping and replacement policy are not established.** The
  set-index function, the number of ways, and how lines are chosen for
  eviction are undocumented, and nothing here measures them. The cache
  model above explains the mechanism; it does not describe M4's L2.
- **Two machines.** The M4 results come from one M4 Max system, under the
  load conditions recorded with the data; the Zen 5 results from one
  Ryzen 9 9950X core. Other chips, or other variants of these, may break
  at different thresholds.
- **One kernel configuration.** The thresholds belong to these matrix
  shapes, this micro-kernel and these block sizes. With the earlier
  blocking (Mc = 128, Kc = 1024) the kernel broke between 384 and 448
  rows instead of between 576 and 640; other shapes and blocks may move
  them again.
- **Inferred, not counted.** The case for L2 conflicts on the M4, and for
  L1 conflicts on Zen 5, rests on throughput and latency thresholds
  matching across experiments. No hardware performance counters were read
  on either machine (macOS has no public counter API; the Linux machine
  restricts them), so cache misses were not counted directly.

## Rules of thumb

1. **Look at strides in bytes, not dimensions.** A row stride that is a
   large power of two leaves few cache sets for all the rows a loop reuses.
   Which level suffers, and how much, depends on the machine: on the M4,
   L2 from 8 KiB on, at a cost of up to two thirds; on Zen 5, L1 at every
   multiple of 4 KiB, at a cost of 14–17% for part 1's kernel (with its
   earlier 8 × 2 tile and 64-row blocks). 4096 FP32
   columns and 2048 FP64 columns are both 16 KiB.
2. **Test with a padded leading dimension.** Running the same problem with
   `ldb = N + one cache line` separates a stride effect from everything
   else in one measurement.
3. **Pack operands that are reused across a loop.** In a blocked GEMM the
   B strip is reused by every A strip; copying it into a contiguous buffer
   costs 3–5.5% and makes performance independent of the caller's layout.
4. **Measure the conflicting capacity.** Sweeping the number of reused rows
   until the stride starts to hurt tells you which cache level is involved,
   without hardware counters. A threshold that halves when the stride
   doubles is L2 (or another level that aliases at large strides); a loss
   that is the same at every multiple of 4 KiB is L1.
5. **What sits behind the conflicting level decides the cost.** The same
   L2 conflicts that cost the M4 two thirds cost Zen 5 little, because
   Zen 5's evicted lines come back from a nearby L3 that does not conflict
   at these strides ([where an evicted line goes](#where-an-evicted-line-goes-l1l2-inclusion-and-the-llc)).

## Method and reproduction

The Zen 5 setup is described in its section and in
[experiments/zen5-b-stride](https://github.com/vdeviatkov/hpc-math-core/tree/main/docs/experiments/zen5-b-stride).

Apple M4 Max (P-cores: 128-byte lines, 128 KiB L1 data per core, 16 MiB L2
per cluster, 16 KiB pages), macOS 26.3.1, Apple Clang 17,
`-O3 -mcpu=apple-m4 -ffast-math -funroll-loops`, one thread. All results
except the pointer chase (best of 5, earlier session) are medians of
repetitions run in randomly interleaved order on a desktop
machine under moderate load; interquartile ranges, raw samples, source
hashes and per-experiment details are in the
[experiment data](https://github.com/vdeviatkov/hpc-math-core/tree/main/docs/experiments/sme-b-stride).
Inputs are warm in cache between calls. GFLOP/s is `2·M·N·K` divided by
the time per call.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON
cmake --build build --parallel
ctest --test-dir build                       # includes every B layout used here

# size and padding sweeps
./build/benchmarks/bench_gemm --benchmark_filter=BStride/f32 \
    --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=true
./build/benchmarks/sme_b_stride_probe height   # rows per strip
./build/benchmarks/sme_b_stride_probe width    # padded vs packed by width
./build/benchmarks/sme_b_layout 4096 15 3      # phase timing
./build/benchmarks/cache_chase                 # CPU core pointer chase

# Zen 5 (Linux, AVX-512), pinned to one core
taskset -c 2 ./build/benchmarks/avx512_b_stride chase    # pointer chase, huge pages
taskset -c 2 ./build/benchmarks/avx512_b_stride ldb      # stride at fixed N
taskset -c 2 ./build/benchmarks/avx512_b_stride size     # N sweep
taskset -c 2 ./build/benchmarks/avx512_b_stride height   # rows per block
```

The chart data is in [sme2-bstride.csv](assets/sme2-bstride.csv) and the
experiment CSVs; [render_charts.py](assets/render_charts.py) regenerates the
three figures. The kernel is part of
[hpc-math-core](https://github.com/vdeviatkov/hpc-math-core), a set of GEMM
implementations from scalar C++ to SME2 and CUDA Tensor Cores.

**Next:** [part 4](cuda-gemm.md) builds the same multiplication on an NVIDIA
GPU, where the bank structure of shared memory plays the role the cache
sets play here.
