---
title: "Matrix Multiplication with SIMD on Apple M4 and AMD Zen 5: From 0.4 to 328 GFLOP/s on One Core"
description: "One CPU core, two architectures: loop order, cache blocking and SIMD register tiles, with FMA latency and throughput measured on Apple M4 and AMD Zen 5. A compiler code-generation trap held AVX-512 at 60% of peak; measured register tiles and cache blocks take it to 91%."
date: 2026-10-15
series_part: 1
permalink: /articles/simd-gemm.html
---
# Matrix multiplication with SIMD on Apple M4 and AMD Zen 5: from 0.4 to 328 GFLOP/s on one core

*One thread on an Apple M4 Max P-core and an AMD Ryzen 9 9950X (Zen 5)
core. Every number was measured on those machines; the code is in
[hpc-math-core](https://github.com/vdeviatkov/hpc-math-core). Part 1 of a
series: [part 2](sme2-gemm.md) moves the same multiplication to Apple's
matrix unit, [part 3](sme2-cache-associativity.md) analyses its cache-set
conflicts, [part 4](cuda-gemm.md) moves it to an NVIDIA GPU and
[part 5](lowp-gemm.md) to 16- and 8-bit formats.*

Matrix multiplication, C = A · B, is the operation the rest of this series
optimises on ever more specialised hardware. Before matrix units and GPUs,
there is the ordinary CPU core: vector registers and fused multiply-add
(FMA) instructions. This article takes the textbook triple loop to 91% of
what a Zen 5 core can compute, measures every step on two quite different
cores, and isolates the three things that decide the result:
which loop runs innermost, which data stays in cache, and how many
independent accumulators the inner loop keeps in registers.

**Key results**

- The textbook i-j-k loop runs at **0.4–0.6 GFLOP/s** at N = 4096: it
  touches a new cache line of B for every multiply-add. Changing the loop
  order to i-k-j is **52–73× faster** in FP32, with no other change.
- The peak is set by the FMA units: measured, **362 GFLOP/s FP32 on a
  Zen 5 core (AVX-512)** and **~130 on an M4 P-core (NEON)**. To reach it a
  kernel needs FMA latency × FMA pipes independent accumulators: 8–12 on
  Zen 5, 12–16 on M4.
- An AVX-512 register tile with 8 accumulators ran at **60% of peak**
  because GCC folded the B loads into the FMAs, re-loading B for every FMA.
  An 8-row tile with 16 accumulators loads B once per step: **333 GFLOP/s,
  92% of peak** (+53%). The same change made the M4's FP64 kernel 26–38%
  faster for N ≥ 256.
- With the tile shape a template parameter, every family was swept over
  9–10 shapes. The winners: 6 × 2 for AVX2 (+14–35% over 4 × 2 at
  N ≥ 256), 8 × 3 for AVX-512 (+10–16% FP32 and +26–38% FP64 at N ≥ 1024
  over 8 × 2), and on the M4 a 6 × 4 tile once A is loaded with `LDR`
  instead of `LD1R`: +25% FP32 and +48% FP64 at N = 256 over the previous
  kernels at the same clock, 86% / 87% of peak with the current blocks.
- The cache blocks around the tile were sizes nobody had measured. Swept
  the same way, the best keep each block of B within half of Zen 5's L2
  and make the blocks of A 6–8 times taller: at N = 4096 +10–12% for AVX2
  and NEON, +28% (FP32) and +51% (FP64) for AVX-512.
- Copying each block of B into contiguous strips (GotoBLAS packing),
  with the same micro-kernels, removes the remaining drop at power-of-two
  sizes: at N = 4096 **+60% (AVX2 FP32), +8% (AVX-512 FP32) and +18%
  (NEON FP32)**, flat from N = 1024 to 4096. Below N = 512 the copy costs
  2–13%.

## The problem

For N × N matrices, C(i,j) = Σₖ A(i,k) · B(k,j) costs N³ multiply-adds,
2N³ flops: 137 Gflop at N = 4096. Each element of A and B is used N times,
so the arithmetic is not the hard part; feeding it is. Matrices here are
row-major: element (i,j) is at offset i·N + j, so a row is contiguous and a
column is N elements apart.

## The machines, measured

| One core | Apple M4 Max P-core (12 P + 4 E cores) | AMD Ryzen 9 9950X (Zen 5, 16 cores) |
|---|---|---|
| Clock, measured under load | 3.96–4.13 GHz (4.48–4.50 GHz on 2026-10-21) | 5.71 GHz |
| Vector registers | 32 × 128-bit (4 FP32) | 32 × 512-bit (16 FP32) with AVX-512 |
| L1 data cache | 128 KiB, 128-byte lines | 48 KiB, 64-byte lines |
| L2 cache | 16 MiB, shared by 6 cores | 1 MiB, private |
| Last-level cache | system-level cache, shared by the chip | L3, 32 MiB per 8 cores |
| FMA latency, measured | 3.3–3.5 cycles | 4.0 cycles |
| FMAs per cycle, measured | up to ~4 | 2 |
| FP32 peak | ~130 GFLOP/s (~143 at 4.49 GHz) | **362 GFLOP/s** (AVX2: 182) |
| FP64 peak | ~65 GFLOP/s | 180 GFLOP/s (AVX2: 91) |

The clock is measured with a chain of dependent integer adds, one per cycle
on both cores. The M4 clock moves with the load on the other cores:
3.94–3.99 GHz in part 2's measurements, 3.96–4.13 GHz here. Cache sizes
and line lengths are what the operating systems report (`sysctl hw` on
macOS, `lscpu` on Linux). FMA latency and throughput come from [`fma_peak`](https://github.com/vdeviatkov/hpc-math-core/blob/main/benchmarks/experiments/fma_peak.cpp), a loop of
FMAs on C independent register accumulators and no memory traffic. With
one accumulator each FMA waits for the previous one; with more, they
overlap until the FMA pipes are full:

| FMAs per cycle (FP32) | 1 | 2 | 4 | 8 | 12 | 16 |
|---|--:|--:|--:|--:|--:|--:|
| Zen 5, AVX-512 | 0.25 | 0.50 | 1.00 | 1.83 | **1.98** | 1.98 |
| M4, NEON | 0.30 | 0.58 | 1.22 | 2.01 | 2.81 | **3.46** |

Zen 5 retires two 512-bit FMAs per cycle with a latency of exactly 4
cycles: 8 accumulators come close and 12 reach the full rate. The M4 core
has four 128-bit FMA pipes and a shorter latency, 3.3–3.5 cycles; it needs
16 accumulators to come near its peak (3.97 FMAs per cycle measured in
FP64 with 20). The two cores get their throughput in different ways: Zen 5
from wide vectors, M4 from more pipes. Zen 5 still does twice the FP32
work per cycle (2 × 16 lanes against 4 × 4) and, at its higher clock,
nearly three times the GFLOP/s.

That rule, **accumulators ≥ latency × pipes**, decides the design of every
fast kernel below, and of the SME2 and GPU kernels in the rest of the
series.

## Step 1: the textbook loop — 0.4 GFLOP/s

```cpp
for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j) {
        float acc = 0;
        for (k = 0; k < N; ++k)
            acc += A[i*N + k] * B[k*N + j];   // B walks down a column
        C[i*N + j] = acc;
    }
```

Code: [`gemm_naive`](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/naive.hpp).

The inner loop reads A along a row, which is contiguous, and B down a
column, N elements apart. Every B(k,j) is on a different cache line, and
the line holding B(k,j) is needed again for B(k,j+1) only after the loop
has touched N other lines: 256 KiB at N = 4096 on Zen 5, 512 KiB on M4,
more than either L1. Each multiply-add waits for a cache miss.

| GFLOP/s, FP32 | N = 64 | 256 | 1024 | 4096 |
|---|--:|--:|--:|--:|
| Zen 5 | 15.8 | 3.4 | 0.57 | 0.42 |
| M4 | 8.9 | 2.5 | 2.1 | 0.59 |

At N = 64 the whole problem fits in L1 and the loop is merely slow; from
there it collapses. At N = 4096 one FP32 multiplication takes about 328 s
on Zen 5 (312 s for FP64), and FP32 and FP64 run at the same speed (0.42
and 0.44 GFLOP/s): element width does not matter to a loop that waits on
memory. Vector instructions cannot fix it either: what limits the loop is
the column walk through B, not the arithmetic.

## Step 2: loop order — 52–73× faster

Swapping the two inner loops makes the innermost loop walk rows
(C must be zeroed before this accumulation loop):

```cpp
for (i = 0; i < N; ++i)
    for (k = 0; k < N; ++k) {
        const float a = A[i*N + k];                // fixed for the whole j loop
        for (j = 0; j < N; ++j)
            C[i*N + j] += a * B[k*N + j];          // B and C: contiguous rows
    }
```

Code: [`gemm_reordered`](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/reordered.hpp).

B and C are now read along rows, every byte of every cache line is used,
the hardware prefetcher can run ahead, and the compiler vectorises the
inner loop by itself (both compilers, at `-O3`):

| GFLOP/s, FP32 | N = 64 | 256 | 1024 | 4096 | ×naive at 4096 |
|---|--:|--:|--:|--:|--:|
| Zen 5 | 50.4 | 93.9 | 70.2 | 30.5 | 73× |
| M4 | 82.6 | 31.0 | 31.6 | 30.7 | 52× |

Two limits remain. Each multiply-add now also loads and stores an element
of C, so the loop moves three values per FMA; on the M4 it sits at about
one 128-bit FMA per cycle from N = 256 on. And for large N, a row of C and
the whole of B no longer stay cached between uses: at N = 4096 the k loop
streams all of B (64 MiB) from memory again for every row of A, and Zen 5
falls to 30 GFLOP/s.

## Step 3: cache blocking

Tiling the three loops into 64 × 64 blocks
([`gemm_blocked`](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/blocked.hpp)) keeps the working set of the
inner loops at 3 × 64 × 64 elements (48 KiB in FP32) whatever N is, so each
tile of B is reused from cache by 64 rows of A:

| GFLOP/s, FP32 | N = 64 | 256 | 1024 | 4096 |
|---|--:|--:|--:|--:|
| Zen 5 | 48.9 | 47.8 | 46.0 | **42.5** |
| M4 | 83.0 | 80.5 | 41.1 | 28.8 |

On Zen 5 throughput is now flat, 1.4× the reordered loop at N = 4096, but
at N = 256 it is half the reordered loop's speed (the cause was not
isolated; the inner loops are only 64 elements long). On the M4 it is
fast up to N = 256 and then decays, to below the reordered loop at
N = 4096. Its 64 rows of each B tile are N elements apart, but the
row-stride penalty the M4's micro-kernel shows in isolation is small (7%
at 4 KiB), so the stride alone does not explain the decay; this article
did not isolate it. Either way the inner loop is unchanged: a load, an FMA
and a store of C per vector.

## Step 4: a register tile

The step that matters keeps a block of C in **registers** for the whole k
loop. The kernel computes, say, 8 rows × 32 columns of C at once. At each
k it loads 32 consecutive values of row k of B as two vectors, broadcasts
the 8 values A(i..i+7, k), and adds their outer product into 16 vector
accumulators:

```cpp
// AVX-512, FP32: an 8 × 32 block of C in 16 ZMM registers
for (k = 0; k < kc; ++k) {
    const __m512 b0 = _mm512_loadu_ps(B + k*ldb);         // B(k, j .. j+15)
    const __m512 b1 = _mm512_loadu_ps(B + k*ldb + 16);    // B(k, j+16 .. j+31)
    for (int r = 0; r < 8; ++r) {                         // unrolled
        const __m512 a = _mm512_set1_ps(A[r*lda + k]);    // broadcast A(i+r, k)
        acc[r][0] = _mm512_fmadd_ps(a, b0, acc[r][0]);
        acc[r][1] = _mm512_fmadd_ps(a, b1, acc[r][1]);
    }
}
```

Code: the micro-kernel
[`vector_micro<Ops, R, V>`](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/simd.hpp),
written like the loop above once for every ISA, with the tile's R rows and
V vectors as template parameters. `Ops` supplies the instructions:
[`Q<T>`](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/neon.hpp),
[`Ymm<T>`](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/avx2.hpp) or
[`Zmm<T>`](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/avx512.hpp),
for FP32 or FP64. `gemm_simd_blocked` (`simd.hpp`) calls it through the
blocked driver.

Per k step: 2 vector loads and 8 broadcasts for 16 FMAs, against a load,
a store and a load per FMA in step 2. C is loaded once before the loop and
stored once after it, every kc = 256 steps. Around the micro-kernel, a
three-level blocking like step 3's keeps the A and B blocks in cache (its
sizes are chosen [below](#cache-blocks-measured-too)). The tiles used:

| Kernel | Register tile of C | Accumulators | Registers used |
|---|---|--:|--:|
| NEON, FP32 / FP64 | 6 × 16 / 6 × 8 (6 rows × 4 vectors) | 24 | 29 of 32 |
| AVX2, FP32 / FP64 | 6 × 16 / 6 × 8 (6 rows × 2 vectors) | 12 | 15–16 of 16 |
| AVX-512, FP32 / FP64 | 8 × 48 / 8 × 24 (8 rows × 3 vectors) | 24 | 28 of 32 |

How each shape was chosen is told below, and every shape measured is in
[docs/kernels](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/kernels/README.md).

This is an outer product computed with vector FMAs: 8 values of A times
32 values of B, 256 multiply-adds per k step. Part 2 of this series runs
the same idea on a unit that performs a 16 × 16 outer product in a single
instruction.

### Why the tile has 8 rows: a code-generation trap

The AVX-512 kernel first had a 4 × 2 tile, 8 accumulators, which the FMA
measurement says is enough for 92% of peak. It ran at 60%. To find out
why, [`simd_microkernel`](https://github.com/vdeviatkov/hpc-math-core/blob/main/benchmarks/experiments/simd_microkernel.cpp) runs the micro-kernel alone on data that stays in
L1, varying the number of rows:

| Zen 5, AVX-512 FP32, L1-resident | Accumulators | GFLOP/s | % of peak |
|---|--:|--:|--:|
| 4 rows × 2 vectors | 8 | 143.7 | 40% |
| 6 rows × 2 vectors | 12 | 348.5 | 96% |
| 8 rows × 2 vectors | 16 | **352.4** | **97%** |
| 12 rows × 2 vectors | 24 | 353.2 | 98% |

The 4-row kernel is far slower than its accumulators allow, and the machine
code shows why. For 4 rows, GCC 13 folds the loads of B into the FMAs as
memory operands:

```text
vfmadd231ps  (%rdx),%zmm11,%zmm2       # B(k, j..j+15) loaded again
vfmadd132ps  0x40(%rdx),%zmm7,%zmm11   # B(k, j+16..j+31) loaded again
vfmadd231ps  (%rdx),%zmm10,%zmm1       # … and again for every row
```

Every FMA re-loads its vector of B: 8 loads of B plus 4 broadcasts per k
step for 8 FMAs, 12 loads where 6 suffice. Zen 5 performs at most two
512-bit loads per cycle (AMD's documented limit, not measured here), so
such a k step cannot take fewer than 6 cycles,
while its 8 FMAs need 4. Measured, it takes 6.7 cycles in the library
kernel, at the load bound, and 10.2 in the isolated micro-kernel, which
the load bound alone does not explain: either way the loads, not the FMA
units, set the pace. With
6 or more rows the compiler loads B into registers once per k step (4 loads
per two steps) and every FMA is register-only.

The library kernel was changed to the 8-row tile, which also divides its
64-row block evenly (and later to 8 × 3, below). In the full
multiplication, same session:

| `gemm_simd_blocked<Zmm>`, GFLOP/s | N = 256 | 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|
| FP32, 4 × 2 tile | 216.8 | 205.2 | 141.3 | 119.9 |
| FP32, 8 × 2 tile | **332.7** | **327.9** | **284.6** | **198.2** |
| FP64, 4 × 2 tile | 106.2 | 55.8 | 55.1 | 51.1 |
| FP64, 8 × 2 tile | **161.5** | **105.2** | **90.9** | **77.2** |

The NEON FP64 kernel had the opposite problem: no folding, but only 8
accumulators for a core that needs 12–16. On L1-resident data, 8 rows
instead of 4 nearly doubled it (31.0 → 58.2 GFLOP/s); in the full
multiplication it is 26–38% faster from N = 256 up (31.5 → 39.6 GFLOP/s at
N = 256, 24.0 → 33.1 at N = 4096) and unchanged at N = 64 (38.2 → 37.3).
Both NEON kernels have since moved to a 6 × 4 tile, below.

### The AVX2 tile: 6 rows, chosen by measurement

The AVX2 kernel also had a 4 × 2 tile: 8 accumulators, exactly the FMA
latency × pipes of Zen 5 and no margin, at 83% of the AVX2 peak. Which
tile is best is a trade-off between three things an R × V tile (R rows of
A, V vectors of B) fixes:

- **Independent accumulators, R·V.** At least 8 on Zen 5, 12 for the full
  rate.
- **Registers, R·V + V + 1.** The accumulators, V vectors of B live for the
  whole k step, and one broadcast of A, which feeds its V FMAs and is then
  dead. AVX2 has 16 YMM registers.
- **B traffic, 32 / R bytes per FMA in FP32.** B is the operand that comes
  from L2: during the inner j loop the same R × 256 piece of A (4–6 KiB)
  serves 16 calls in a row and stays in L1, while each call reads a
  different 256 × 16 strip of B, 16 KiB, out of a 256 KiB block that L1
  cannot hold. Every row added to the tile reuses each B load once more.

Why six rows and two vectors, rather than two rows and six vectors?
Both shapes have 12 accumulators, but their operand registers differ:

| Shape, rows × vectors | B vectors loaded per k step | A scalar broadcasts per k step | Approximate vector-register budget |
|---|--:|--:|--:|
| 6 × 2 | 2 | 6 | 12 + 2 + 1 = 15 |
| 2 × 6 | 6 | 2 | 12 + 6 + 1 = 19 |

The budget assumes all V B vectors stay live while one A broadcast at a
time feeds its V FMAs. Six rows reuse each full-vector B load six times;
two rows reuse it only twice. Increasing rows needs more scalar A loads,
while increasing vectors needs more full-vector B loads and live B
registers. On the 16-register AVX2 target, 2 × 6 cannot keep this schedule
entirely in registers: it needs spills, reloads or another schedule. This
is register-budget arithmetic, not a measured 2 × 6 benchmark result;
actual code generation still decides performance.

Also, rows × vectors is not rows × scalar columns. One AVX2 vector holds
eight FP32 elements: 6 × 2 computes a **6 × 16** block of C, while 2 × 6
computes 2 × 48. The default is already wider than it is tall in scalar
elements. Transposing the two template numbers changes operand reuse and
register pressure, rather than merely rotating the same output rectangle.

Measured on the whole multiplication: the library kernel
`gemm_simd_blocked<Ymm, R, V, …>` with each shape (`bench_gemm`'s `Avx2Tile`
benchmarks, built for 16 YMM registers, x86-64-v3; FP32 GFLOP/s, median
of 3; all shapes and FP64 in
[docs/kernels/avx2.md](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/kernels/avx2.md)):

| Tile | Accumulators | N = 256 | 1024 | 4096 |
|---|--:|--:|--:|--:|
| 4 × 2 | 8 | 151.2 | 92.7 | 75.9 |
| 2 × 4 (same accumulators, twice the B traffic) | 8 | 135.5 | 63.0 | 60.0 |
| 8 × 1 | 8 | 143.4 | 92.2 | 74.9 |
| 5 × 2 | 10 | 163.2 | 111.4 | 85.7 |
| **6 × 2** | **12** | **169.0** | **126.7** | 94.1 |
| 4 × 3 | 12 | 154.2 | 108.2 | 87.9 |
| 3 × 4 | 12 | 152.1 | 108.0 | **94.3** |
| 7 × 2 (needs 17 registers: spills) | 14 | 97.1 | 90.7 | 83.8 |

6 × 2 needs 15 registers and does not spill (GCC 13 alternates two
broadcast registers and uses all 16); 7 × 2 needs 17 and spills
accumulators to the stack in the inner loop. (With the 32 registers of
the native Zen 5 build, 7 × 2 does not spill and is 3–8% faster from
N = 512–1024 up; 6 × 2 is kept because it is the fastest that never
spills on an AVX2-only CPU. The 3 × 3 shape, also swept, is not shown:
149.5 / 84.6 / 73.9.) The library kernel now uses 6 × 2; its block, then
64 rows, was ten 6-row strips and one 4-row strip of the same kernel.
Same session, old against new, alternated three times:

| `gemm_simd_blocked<Ymm>`, GFLOP/s | N = 256 | 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|
| FP32, 4 × 2 tile | 150.4 | 142.3 | 93.2 | 75.7 |
| FP32, 6 × 2 tile | **171.2** | **167.7** | **124.0** | **95.4** |
| FP64, 4 × 2 tile | 71.1 | 46.4 | 41.6 | 37.3 |
| FP64, 6 × 2 tile | **83.8** | **62.3** | **54.3** | **48.1** |

14–35% faster from N = 256 up (11–12% at N = 64). (These are the
`-march=native` build, the one every table here uses except the sweep
above. On Zen 5 it enables AVX-512VL, which gives even 256-bit code 32 YMM
registers; the 16-register x86-64-v3 build gains about the same, 13–37%
from N = 256 up.)

### AVX-512 and NEON, swept the same way

The register tile is a template parameter of every family, so
`bench_gemm` sweeps the AVX-512 and NEON kernels over 10 shapes each as
well ([docs/kernels](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/kernels/README.md)
has all of them).

**AVX-512: 8 × 3.** The rule from the trap above holds across the sweep:
every shape with 4 or fewer rows (4 × 2, 2 × 4, 4 × 4) has GCC fold the B
loads into the FMAs and runs 20–68% behind the best shape at N ≤ 512. Of
the rest, a third vector on the 8-row tile pays where B's rows are 4 KiB
apart and B comes from L2:

| `gemm_simd_blocked<Zmm>`, GFLOP/s | N = 256 | 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|
| FP32, 8 × 2 tile | **327.6** | 325.0 | 281.8 | 197.8 |
| FP32, 8 × 3 tile | 324.6 | **328.2** | **309.9** | **229.6** |
| FP64, 8 × 2 tile | 158.9 | 102.7 | 90.1 | 76.8 |
| FP64, 8 × 3 tile | **162.0** | **150.9** ¹ | **124.1** | **97.1** |

¹ At this size the result depends on where the matrices land in memory;
run alone, 8 × 3 measures 126–128 and 8 × 2 101–103.

8 × 3 uses 28 of the 32 registers. The library kernel uses it now. The
sweep does not show why the third vector helps so much where B misses L1,
and this machine allows no hardware counters to look.

**NEON: 6 × 4, and `LDR` instead of `LD1R`.** On the M4 the sweep first
showed something else: every tile with 6 or more rows and 4 vectors, or 8
and more rows, stayed under 97 GFLOP/s FP32 and 49 FP64 whatever its
accumulator count. Those tiles broadcast A with `LD1R` (load and replicate),
one per row per k step, which is what Clang emits for `vdupq_n_f32` +
`vfmaq_f32`. Loading A with a plain `LDR` and multiplying by element
(`FMLA v.4s, v.4s, v.s[0]`) lifted them; alternated runs of the same
6 × 4 tile with each load, N = 256:

| 6 × 4 tile, GFLOP/s | FP32 | FP64 |
|---|--:|--:|
| A broadcast with `LD1R` | 94.7 | 47.2 |
| A loaded with `LDR`, FMA by element | **123.2** | **61.4** |

With `LDR`, 6 × 4 (24 accumulators, 29 of 32 registers) is the fastest
shape from N = 256 up in both types. FP32 Clang folds every intrinsic form
back into `LD1R`, so the FP32 load is one line of inline assembly.

### Cache blocks, measured too

The three cache-block sizes, TileM rows of A × TileK steps of k × TileN
columns of B, were 64 × 256 × 256 (AVX-512: × 512) from the first version
of the code, never measured. They are template parameters as well, and
`bench_gemm` swept 112 blocks per family
([docs/kernels/blocks.md](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/kernels/blocks.md)).
Each dimension has a different job in the SIMD driver's loop order
(i-block, k-block, j-block, row strip, column tile):

- **TileK × TileN controls the B block.** Every R-row strip of an i-block
  reads that block, so the target is L2 reuse across row strips.
- **R × TileK controls the A strip.** The same strip serves successive
  column-tile calls within a j-block. At the defaults it is small enough
  for L1, even though the whole B block is not.
- **TileM controls how many row strips share B.** Increasing it reduces
  repeated fetches of B from beyond L2. The selected value comes from
  the measured performance plateau, not from requiring the entire A
  block and C block to fit alongside B.
- **TileK also controls partial-C traffic.** C stays in registers for
  TileK contraction steps; a longer K block means fewer accumulator
  reloads and stores, but larger operand working sets. Cache capacity,
  bandwidth and stride conflicts limit the benefit.

For the selected AVX2 FP32 block, **384 × 256 × 256**, with its 6 × 2
register tile:

| Working set | Size | Intended reuse |
|---|---:|---|
| B block: 256 × 256 floats | 256 KiB | L2, across 64 groups of six rows |
| A strip: 6 × 256 floats | 6 KiB | L1, across the 16 column-tile calls of a j-block |
| C micro-tile: 6 × 16 floats | 12 vector registers | Registers, throughout the 256-step k loop |

These are reuse targets, not guarantees of cache residency. Associativity,
row stride, other data and the executing machine matter. The cache model
narrows the candidates; the sweep selects the final block sizes.

Two rules came out:

- **Tall blocks of A.** Each block of B is fetched again for every block
  of A rows, N / TileM times in all. Once B no longer fits in the
  last-level cache (N = 4096: 64 MiB in FP32), that traffic sets the pace,
  and every doubling of TileM helps until about 384–512 rows. At N ≤ 1024
  TileM makes no difference.
- **A block of B within half of Zen 5's L2.** The B block, TileK × TileN,
  is reused by every row strip of the A block; up to 512 KiB it stays in
  the 1 MiB L2, at 1 MiB it does not, and the kernel loses 11–27%. The old
  AVX-512 FP64 block, 256 × 512 doubles, was exactly 1 MiB. On the M4,
  whose L2 is 16 MiB, the B block size made no difference.

| Family | Old block | New block | N = 4096, FP32 | N = 4096, FP64 |
|---|---|---|--:|--:|
| AVX-512 | 64 × 256 × 512 | 512 × 256 × 512 FP32, × 256 FP64 | 230.2 → **294.7** | 95.9 → **145.0** |
| AVX2 | 64 × 256 × 256 | 384 × 256 × 256 FP32, × 128 FP64 | 95.4 → **105.6** | 48.1 → **53.8** |
| NEON | 64 × 256 × 256 | 384 × 256 × 256 | 91.9 → **102.1** | 45.8 → **51.0** |

Old blocks: the last runs of each kernel before the change
([`bench_zen5_avx512_8x3.csv`](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/experiments/tile-sweeps/bench_zen5_avx512_8x3.csv),
[avx2-tiles](https://github.com/vdeviatkov/hpc-math-core/tree/main/docs/experiments/cpu-simd/avx2-tiles),
[`bench_m4_neon_6x4.csv`](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/experiments/tile-sweeps/bench_m4_neon_6x4.csv));
new blocks: the runs in the table below.

The smaller FP64 block of B also lifts AVX-512 FP64 at N = 512 and 1024
by 22–23% (128.4 → 157.7, 123.4 → 151.0).

### What the register tile reaches

The library kernels with their current tiles and blocks, median of three
runs:

| GFLOP/s, FP32 | N = 64 | 256 | 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|--:|
| Zen 5, AVX-512, 8 × 3 | 297.7 | 324.4 | **327.9** | 312.8 | 294.7 |
| Zen 5, AVX2, 6 × 2 | 166.6 | **171.7** | 169.7 | 126.9 | 105.6 |
| M4, NEON, 6 × 4 ¹ | 122.4 | 122.9 | **123.8** | 121.2 | 102.1 |

| GFLOP/s, FP64 | N = 64 | 256 | 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|--:|
| Zen 5, AVX-512, 8 × 3 | 153.0 | **162.0** | 157.7 | 151.0 | 145.0 |
| Zen 5, AVX2, 6 × 2 | **82.8** | 81.9 | 62.3 | 54.8 | 53.8 |
| M4, NEON, 6 × 4 ¹ | 62.1 | **62.2** | 61.0 | 59.5 | 51.0 |

Sources: [`bench_zen5_defaults.csv`](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/experiments/block-sweeps/bench_zen5_defaults.csv)
and [`bench_m4_defaults.csv`](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/experiments/block-sweeps/bench_m4_defaults.csv).
¹ Run on 2026-10-22; the core clock measured in the M4 runs of
2026-10-21 was 4.48–4.50 GHz (none was recorded for this run), against 3.96–4.13 GHz for the M4's
other numbers in this article (the bars marked "before" in the chart
below).

![FP32 GEMM on one Zen 5 core, N = 512](assets/simd-zen5.svg)

![FP32 GEMM on one Apple M4 P-core, N = 512](assets/simd-m4.svg)

At its best size the AVX-512 kernel runs at **91% of the core's FP32 peak
and 90% of FP64** (of the measured 362 / 180), AVX2 at 94% / 91%, NEON at
86% / 87% (of ~144 / ~72 GFLOP/s at 4.49 GHz). The README's headline
divides by the theoretical peaks (365 / 183 for AVX-512) and quotes the
best result over every tile and block, so its percentages differ by about
a point. Against the textbook loop at N = 4096 the AVX-512
kernel is 704× faster in FP32.

## What remains: B's row stride

The register-tile kernels lose throughput as N grows, and the drops line
up with the byte distance between B's rows, which these kernels read in
place: AVX2 drops at FP32 N = 1024 and FP64 N = 512, both a 4 KiB stride,
and so did AVX-512 with its earlier tile and blocks (−13%; −5% now). The micro-kernel runs down 256 rows of B, and rows a power
of two apart compete for a few cache sets. The micro-kernel experiment
measures it directly, on data small enough for L1:

| AVX-512 FP32, 16 accumulators, L1-resident | GFLOP/s |
|---|--:|
| B packed (rows contiguous) | 352.4 |
| B rows 1 KiB apart | 335.6 (−5%) |
| B rows 4 KiB apart | 275.5 (−22%) |

The full kernel drops at the same sizes, by less: with the 8 × 2 tile
−13%, 327.9 → 284.6 FP32 from N = 512 to 1024, where B's stride is one
cost among several; with 8 × 3 and the current blocks, −5% (327.9 →
312.8).
[Part 3](sme2-cache-associativity.md#the-same-experiment-on-amd-zen-5)
isolates it on this core: with only the stride changed, B rows 4, 8 and
16 KiB apart cost the kernel (then with the 8 × 2 tile and 64-row
blocks) 14–17% against the same rows padded by one
64-byte line. Every multiple of 4 KiB folds a strip's rows onto 12 lines
of L1 (64 sets × 12 ways); L2 conflicts, which cost the M4's SME2 kernel
two thirds, appear to be absorbed by Zen 5's L3. The cure is to pad B's
rows or copy each strip of B into a contiguous buffer before the
micro-kernel uses it, as part 2 does for the SME2 kernel. The next step
does the copy.

## Step 5: pack B

Code: [`gemm_simd_packed<Ymm>`, `gemm_simd_packed<Zmm>`, `gemm_simd_packed<Q>`](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/simd.hpp).

Every SIMD kernel in this series, including the 16- and 8-bit ones of
[part 5](lowp-gemm.md), has the same inner step: load a few vectors of a
row of B, broadcast one element of A per row of the tile, and multiply-add
into R × V accumulators. The low-precision kernels must pack B, because
their dot-product instructions need G consecutive k of one column in each
32-bit lane, and row-major B does not store them that way. The FP32
kernels do not have to: a row of B is already V contiguous vectors, the
layout the FMA wants. Here packing B is purely about the caches.

The packed kernels keep the micro-kernel and the register tile, and
change two things:

1. **B's layout.** Each Kc × Nc block of B is copied into strips one tile
   wide (V·W columns: 48 floats for AVX-512, 16 for AVX2 and NEON), kc rows
   each, stored contiguously. Inside a strip, row k + 1 follows row k
   directly instead of N elements later, so no N folds the strip onto a few
   cache sets.
2. **The loop order (GotoBLAS).** The B block is copied once and then
   serves every row block of A; inside, one strip is reused by every R-row
   tile of the A block before the next strip is touched.

```text
for jc: block of Nc columns
  for pc: block of Kc steps of k
    pack B(pc.., jc..) into strips          ← once, shared by all of A
    for ic: block of Mc rows of A           (A read in place)
      for jr: one strip                     ← kc × V·W, contiguous
        for ir: R rows → micro-kernel(A rows, strip, ldb = V·W)
```

The blocks were swept as in Step 4 (Mc × Kc × Nc, 36 combinations on
Zen 5 and 45 on the M4). The defaults are 256 × 256 × 512 for AVX2,
128 × 512 × 512 for AVX-512 and 256 × 512 × 512 for NEON. The best ten
blocks of each Zen 5 grid lie within 2.5%. One result was unexpected: on
AVX-512, Kc = 512 beats 256 by ~2%, although a 512 × 48-float strip
(96 KiB) is twice the L1. Once the rows are contiguous, streaming the strip
from L2 is fast enough.

Packed against in place, GFLOP/s (Zen 5: 3 runs, M4: 5 runs, medians):

| FP32 | N = 1000 | 1024 | 2000 | 2048 | 4000 | 4096 |
|---|--:|--:|--:|--:|--:|--:|
| Zen 5, AVX-512, in place | 250.6 | 311.5 | 308.2 | 301.6 | 309.8 | 294.2 |
| Zen 5, AVX-512, packed | **288.7** | 315.7 | 320.9 | 318.2 | 322.5 | **318.7** |
| Zen 5, AVX2, in place | 168.4 | 126.3 | 163.9 | 105.2 | 163.0 | 104.2 |
| Zen 5, AVX2, packed | 167.5 | **166.1** | 168.7 | **166.5** | 169.4 | **166.9** |
| M4, NEON, in place | 120.8 | 121.5 | 118.7 | 117.6 | 114.1 | 101.6 |
| M4, NEON, packed | 121.9 | 123.2 | 118.6 | 121.1 | 120.7 | **120.4** |

| FP64 | N = 1000 | 1024 | 2000 | 2048 | 4000 | 4096 |
|---|--:|--:|--:|--:|--:|--:|
| Zen 5, AVX-512, in place | 152.2 | 149.8 | 150.5 | 144.7 | 149.9 | 142.4 |
| Zen 5, AVX-512, packed | 155.6 | 154.5 | 155.0 | 154.3 | 154.2 | **152.2** |
| Zen 5, AVX2, in place | 83.9 | 54.7 | 81.0 | 53.4 | 81.5 | 53.6 |
| Zen 5, AVX2, packed | 83.2 | **81.8** | 82.9 | **82.0** | 82.6 | **81.3** |
| M4, NEON, in place | 59.0 | 59.1 | 54.8 | 51.6 | 54.1 | 50.5 |
| M4, NEON, packed | 59.2 | 59.8 | 59.1 | **58.8** | 58.7 | **58.5** |

- **The 4 KiB cliff is gone.** In place, AVX2 loses a quarter to a third
  of its speed whenever B's rows are a multiple of 4 KiB apart. Packed, it
  runs at 166–169 GFLOP/s FP32 and 81–83 FP64 at every size from 1000 to
  4096, **+32% to +60%**, and at N = 4096 within 3% of its best small-N
  figure.
- **AVX-512 gains 3–8%** at N ≥ 2000, and 15% at N = 1000, where the
  in-place kernel is unusually slow (rows 4,000 B apart, just short of
  4 KiB). At N = 4096 it now runs at 318.7 GFLOP/s, 88% of the measured
  peak, against 81% in place.
- **NEON gains 16–19% at N = 4096**, and in FP64 8–16% from N = 2000. On
  the M4 the FP64 gain at N = 2000 and 4000 is as large as at the powers of
  two, so the stride is not the only thing the copy fixes there; what else
  it fixes was not isolated.
- **Small matrices pay for the copy.** At N = 64 the packed kernels run at
  0.87–0.95× of the in-place ones, at N = 256 at 0.94–0.98×; N = 512 is
  break-even. The copy and its buffer are inside the timed call. So the
  packed kernels are separate entry points, and the blocked kernels remain
  the defaults of the tables above.
- **Packing A too does not help.** Copying the Mc × Kc block of A as well,
  so that a tile's rows are kc elements apart instead of K, runs at
  0.95–1.00× of packing B alone on both machines.

Data and method: [experiments/packed-b](https://github.com/vdeviatkov/hpc-math-core/tree/main/docs/experiments/packed-b),
[kernels/blocks.md](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/kernels/blocks.md#packed-b).

## What does not help

- **Wider vectors on a memory-bound loop.** Explicit AVX2, AVX-512 and
  NEON versions of the reordered loop run at or below the speed of the
  scalar one, which the compiler already vectorises.
- **Software prefetch** ([`prefetch.hpp`](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/prefetch.hpp)). Prefetching A, B and C 2 to 16 tile rows ahead of
  the current register-tile kernels changes their throughput by −1% to
  +0.3% (Zen 5, AVX-512) and −3% to +0.2% (M4, NEON) at N = 256 to 1024:
  the hardware prefetchers already follow these streaming patterns. On the
  earlier 4-row AVX-512 kernel, which was bound by its loads, prefetching
  gained up to 6%, and 21% in one outlier (FP64, N = 512, 16 rows ahead).

## Correctness

Every kernel is compared element by element with the textbook loop on 32
shapes in FP32 and FP64
([`tests/test_gemm.cpp`](https://github.com/vdeviatkov/hpc-math-core/blob/main/tests/test_gemm.cpp)):
squares around every vector and tile width (1–5, 7, 8, 13, 15–17, 31–33,
64, 100, 128, 256), rectangles with partial tiles in every dimension, and
empty matrices. Row counts such as 17, 33, 63, 129 and 257 exercise the
remainder strips, which run the same micro-kernel with fewer rows. The
tests also run each family with several non-default tiles (for example
AVX-512 8 × 2, 6 × 3 and 12 × 2; NEON 1 × 1, 6 × 3 and 3 × 5). The packed
kernels of Step 5 run the same shapes at their defaults, with A packed
too, with a 4 × 2 tile, and with 5 × 7 × 50 blocks, which cut every strip
and block short. An element passes if
`|C − C_ref| ≤ tol · (1 + |C_ref|)`, tol = 1e-4 in FP32 and 1e-10 in FP64:
the kernels sum in a different order than the reference, and the tests use
the same `-ffast-math` build as the benchmarks. All pass on both machines,
before and after the tile changes.

## Limitations

- **One thread.** Both chips have 16 cores; this article is about what one
  core's vector units do.
- **Two compilers, one version each.** The folding of B loads into FMAs is
  GCC 13's choice for this loop; another compiler or version may choose
  differently, which is why the kernel now uses a tile that is fast either
  way.
- **The M4 ran with background load.** Two unrelated processes kept two
  cores busy during the 2026-10-15 measurements, and Spotlight indexing
  and an IDE (load average 2–14) during those of 2026-10-21/22; the core
  clock moved between 3.96 and 4.13 GHz from one program to the next on
  10-15 and was 4.48–4.50 GHz on 10-21. Treat M4 figures as ±5%,
  and compare across the two dates in cycles.
- **No hardware counters on the CPUs.** macOS has no public counter API,
  and the Linux machine restricts them. The explanations rest on controlled
  experiments and on reading the generated machine code.

## Takeaways

1. **Loop order before everything.** The same arithmetic in the order that
   walks rows is up to 73× faster.
2. **Measure the FMA units, then count accumulators.** Latency × pipes is
   the minimum number of independent accumulators: 8–12 on Zen 5, 12–16 on
   M4. Below that a kernel cannot reach peak, whatever else is right.
3. **Read the machine code.** The AVX-512 kernel had enough accumulators on
   paper; the compiler's choice to re-load B in every FMA held it at 60%.
   A larger tile fixed it: 92% of peak with 8 × 2, 91% with today's 8 × 3
   and blocks.
4. **Pack B when N is large.** Row-major B already suits the FP32
   micro-kernel; the copy is for the caches, and it is worth up to 60% at
   power-of-two N and a few percent below N = 512.
5. **The register tile is an outer product.** Every fast kernel in this
   series keeps a block of C in registers (or, on a matrix unit, in
   accumulator tiles) and adds outer products of a column of A and a row of
   B into it.

**Next:** [part 2](sme2-gemm.md) runs the same multiplication on Apple's
SME2 matrix unit, where one instruction is a 16 × 16 outer product, and
where the same packing of B matters far more: unpacked, a power-of-two
stride costs it two thirds of its speed ([part 3](sme2-cache-associativity.md)).

## Reproduce

```bash
git clone https://github.com/vdeviatkov/hpc-math-core && cd hpc-math-core
cmake -B build -DCMAKE_BUILD_TYPE=Release          # add -DHPC_ENABLE_AVX512=ON on x86
cmake --build build --parallel
ctest --test-dir build
./build/benchmarks/fma_peak                        # FMA latency and throughput
./build/benchmarks/simd_microkernel                # the micro-kernel alone
./build/benchmarks/bench_gemm --benchmark_filter='^(Naive|Reordered|Blocked|Neon|Avx2|Avx512)'
./build/benchmarks/bench_gemm --benchmark_filter='Tile'   # every register-tile shape
./build/benchmarks/bench_gemm --benchmark_filter='Block[0-9]'   # every cache block
./build/benchmarks/bench_gemm --benchmark_filter='Cmp/'        # packed B against in place
```

The kernels are in
[`src/gemm/`](https://github.com/vdeviatkov/hpc-math-core/tree/main/src/gemm)
(`naive.hpp`, `reordered.hpp`, `blocked.hpp`, `neon.hpp`, `avx2.hpp`,
`avx512.hpp`, `simd.hpp`). Raw data and method:
[experiments/cpu-simd](https://github.com/vdeviatkov/hpc-math-core/tree/main/docs/experiments/cpu-simd),
[tile-sweeps](https://github.com/vdeviatkov/hpc-math-core/tree/main/docs/experiments/tile-sweeps)
and [block-sweeps](https://github.com/vdeviatkov/hpc-math-core/tree/main/docs/experiments/block-sweeps).
