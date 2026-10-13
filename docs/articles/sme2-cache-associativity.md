---
title: "ARM SME2 GEMM: A Power-of-Two Performance Cliff"
description: "Why an SME2 matrix multiplication on Apple M4 Max runs at half speed when B's rows are 16 KiB apart, and what packing and padding recover."
permalink: /articles/sme2-cache-associativity.html
---
# ARM SME2 GEMM: a power-of-two performance cliff

*Apple M4 Max, single thread, FP32 unless noted.*

An SME2 matrix multiplication that reads B straight from its row-major
storage runs at **1,365 GFLOP/s for N = 4000 but 608 GFLOP/s for N = 4096**.
Copying B into contiguous strips first ("packing") brings N = 4096 back to
1,344 GFLOP/s, copy included. Keeping the problem at 4096 × 4096 and only
padding each row of B by 32 unused floats also recovers most of it
(1,339 GFLOP/s). The slowdown follows the byte distance between B's rows, not
the matrix size, which is what cache-set conflicts predict. Timing the
phases shows that packing is still the better fix: with a padded stride,
copying B costs 2 ms and saves 14 ms of compute.

## The kernel

`C = A × B`, all matrices row-major. SME's `FMOPA` instruction adds the
outer product of two vectors to ZA, a two-dimensional accumulator. With a
512-bit streaming vector length (16 FP32 lanes), ZA holds four 16 × 16 FP32
tiles, which together accumulate a 32 × 32 block of C. One step of the
k loop is:

```text
a = A[i .. i+31, k]                    // a column segment of A: 2 vectors
b = B[k, j .. j+31]                    // a row segment of B: 2 vectors (svld1_x2)
C[i .. i+31, j .. j+31] += a ⊗ b       // 4 FMOPAs, one per ZA tile
```

The 32 values of A are a row stride apart and streaming mode has no gather
load, so A is always packed into strips where each column segment is
contiguous. B's 32 values are already adjacent, so B can be read in place:
row `k` of the segment is at `B + k·ldb + j`. Packing B instead copies each
32-column strip of a 1024-row block into a contiguous buffer, 128 bytes per
row.

The comparison uses one driver, `sme_detail::blocked_driver<T, PackB>`
([sme.hpp](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/sme.hpp)),
with GotoBLAS blocking `Mc = 128`, `Kc = 1024`, `Nc = 4096`:

| | B in place (`gemm_sme_blocked`) | B packed (`gemm_sme`) |
|---|---|---|
| A packing, cache blocking, ZA tiles, FMOPAs | same | same |
| B row k of a strip is at | `B + k·ldb + j` | `strip + k·128 B` |
| Copy of B | none | included in the timing |

One FP32 B strip is 1024 rows × 32 floats = **128 KiB** of useful data. The
macro-kernel reuses it for the 4 A strips of an Mc block (`jr` outer loop,
`ir` inner), so it has to survive in cache from one A strip to the next.
Read in place at `ldb = 4096`, those 128 KiB are spread over 1024 rows
16 KiB apart, a 16 MiB address range.

Built with Apple Clang 17, `-O3 -mcpu=apple-m4 -ffast-math -funroll-loops`.
The M4 Max P-cores have 128-byte cache lines, a 128 KiB L1 data cache per
core and a 16 MiB L2 per cluster. Apple does not document the SME unit; it
is shared by the cores of a cluster and, by public measurements, loads
through L2 rather than a core's L1.

## The cliff

Square matrices around 2048 and 4096, so `ldb = N`. Medians of 3 randomly
interleaved repetitions; the timed call includes allocating scratch buffers
and packing ([raw tables](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/benchmarks.md#sme-b-row-stride)).

![FP32 throughput near 2048 and 4096 with B in place and packed](assets/sme2-size-sweep.svg)

| N | B row stride | in place | packed | | FP64 in place | FP64 packed |
|---:|---:|---:|---:|---|---:|---:|
| 1984 | 7,936 B | 1,565 | 1,517 | | 407 | 418 |
| 2016 | 8,064 B | 1,556 | 1,541 | | 410 | 419 |
| **2048** | **8,192 B** | **1,196** | 1,432 | | **281** | 407 |
| 2080 | 8,320 B | 1,493 | 1,483 | | 398 | 404 |
| 4000 | 16,000 B | 1,365 | 1,406 | | 380 | 409 |
| 4032 | 16,128 B | 1,354 | 1,399 | | 380 | 408 |
| 4064 | 16,256 B | 1,303 | 1,399 | | 364 | 411 |
| **4096** | **16,384 B** | **608** | 1,344 | | **268** | 398 |
| 4128 | 16,512 B | 1,260 | 1,304 | | 362 | 395 |

GFLOP/s; the stride column is for FP32, FP64 strides are twice as large.

From N = 4000 to 4096 the work grows by 7.4%, but in-place throughput falls
by 55%, and at 4128 it is back to 1,260. The dips sit exactly on the powers
of two. FP64, whose B rows reach 16 KiB already at N = 2048, loses 31% there.

## Why a power-of-two stride can hurt

A set-associative cache maps every line to one set by bits of its address,
and each set holds only a few lines (its ways). With 128-byte lines and
`S` sets indexed by the address bits directly above the line offset, rows
`s` bytes apart (a multiple of 128) land in `S / gcd(s / 128, S)` different
sets. For an illustrative 128 KiB, 8-way cache (128 sets):

| Row stride | Stride in lines | Sets used by 1024 rows | Rows that fit |
|---:|---:|---:|---:|
| 16,384 B | 128 | 1 | 8 |
| 8,192 B | 64 | 2 | 16 |
| 16,000 B | 125 | 128 | all |
| 128 B (packed) | 1 | 128 | all |

At a 16 KiB stride every row of the strip competes for the same 8 ways, so
the strip evicts itself long before an A strip comes back for it, although
it is far smaller than the cache. These are conflict misses, not capacity
misses. Real caches may hash more address bits into the set index, and
Apple does not document M4's, so this is a model, not a claim about the
hardware.

A pointer chase on a P-core is consistent with it: 1,024 lines visited in
random order cost 0.78 ns per load when they are 128 B apart, but 24 ns
when they are 16 KiB apart. At that stride only 8 lines stay in L1, and
fewer than 1,024 in L2. That measures the core's load path, though, not
the SME unit's.

## Same multiplication, different stride

Changing N also changes the work. The cleaner test keeps
`M = N = K = 4096` and stores B with a wider leading dimension `ldb`; the
extra columns are never read.

![Same 4096 × 4096 GEMM with four leading dimensions of B](assets/sme2-padding.svg)

| ldb | B row stride | in place | packed | | FP64 in place | FP64 packed |
|---:|---:|---:|---:|---|---:|---:|
| **4096** | **16,384 B** | **608** | 1,344 | | **268** | 398 |
| 4097 | 16,388 B | 1,145 | 1,365 | | 367 | 407 |
| 4128 | 16,512 B | 1,339 | 1,386 | | 376 | 407 |
| 4160 | 16,640 B | 1,332 | 1,390 | | 376 | 406 |

One extra cache line per row (`ldb = 4128`) makes B 0.5 MiB larger and the
in-place kernel 2.2× faster: the work, the useful data and the number of
pages touched are unchanged, only the stride is. `ldb = 4097` is a control
rather than a fix: its 128-byte row segments no longer start on a line
boundary, so each spans two lines.

Padding does not make B as good as packed. In this run the padded in-place
kernel was 3% behind packed in FP32; a repeat with 5 repetitions measured
8%, and FP64 lost 6–9%. Neither ldb nor anything else about B changes the
packed kernel, which only ever sees the 128-byte packed stride.

## Where the time goes

A [phase-timing experiment](https://github.com/vdeviatkov/hpc-math-core/tree/main/docs/experiments/sme-b-layout)
runs the same driver with a clock around each phase: 4096 × 4096 FP32,
15 shuffled repetitions, medians in milliseconds.

| ldb | B | End to end | Pack A | Pack B | Compute |
|---:|:--|--:|--:|--:|--:|
| 4096 | in place | 238.1 | 2.95 | — | 230.8 |
| 4096 | packed | 104.7 | 2.96 | 4.81 | 96.9 |
| 4128 | in place | 114.1 | 2.94 | — | 110.0 |
| 4128 | packed | 102.0 | 2.96 | 2.07 | 96.0 |
| 4160 | in place | 116.6 | 2.93 | — | 107.2 |
| 4160 | packed | 101.4 | 2.98 | 2.03 | 97.0 |

- At `ldb = 4096` the loss is in the macro-kernel: 231 ms against 97 ms.
- With a padded stride, copying B costs about 2 ms and saves 11–14 ms of
  compute. Packed B was faster in all 15 repetitions at both padded strides.
- Packing from the 16 KiB stride is slower too (4.8 ms), because the copy
  reads the same strided rows, but it reads each element once instead of
  once per A strip.

## What the evidence does and does not show

It shows that the in-place slowdown depends on the exact byte stride
between B's rows, collapses at the power-of-two strides (8, 16 and, for
FP64, 32 KiB), and disappears almost entirely when the stride moves by one cache
line, with the work and the pages touched held fixed. That is the signature
of cache-set conflicts.

It does not show which cache level conflicts, or how M4 indexes its sets:
no hardware counters were collected, and the pointer chase sees the core,
not the SME unit. It does not explain the 3–9% that padded B still loses to
packed B; a strip read in place spans about 1,024 pages against 8 for a
packed strip, so TLB reach and prefetching are candidates, not separated
here. And packing is not free: around N = 2000, off the power of two, FP32 with
B in place is up to 3% faster because it skips the copy.

## Reproduce

On an M4-class Mac:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON
cmake --build build --parallel
ctest --test-dir build          # includes in-place and packed B with ldb > N

./build/benchmarks/bench_gemm --benchmark_filter=BStride \
    --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=true
./build/benchmarks/sme_b_layout 4096 15 3 > timings.csv
```

`BStride` covers 12 shapes × 2 precisions × 2 B paths. In `bench_gemm` the
inputs are created once and reused across iterations, and GFLOP/s is
2·N³ divided by CPU time. The chart data is in
[sme2-bstride.csv](assets/sme2-bstride.csv), and
[render_charts.py](assets/render_charts.py) regenerates both figures.

## Takeaway

Contiguous values within one load are not enough: what matters is the
sequence of addresses across the loop and whether the data survives until
it is reused. A 128 KiB strip that fits easily in cache stopped fitting
because its rows were 16 KiB apart. Packing gives the kernel its own
layout whatever stride the caller's matrix has, and at these sizes the
copy costs a few percent of the time it saves.
