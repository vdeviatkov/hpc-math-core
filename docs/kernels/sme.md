# SME2 ZA tiles and cache blocks (`gemm_sme`)

`gemm_sme<T, R, V, TileM, TileK, TileN>` multiplies on the SME2 matrix
unit with an R × V arrangement of ZA tiles, each holding an S × S block of
C, where S is the streaming vector length in elements (16 FP32, 8 FP64 on
the M4). The measured defaults are **2 × 2 FP32** (a 32 × 32 block of C)
and **2 × 4 FP64** (16 × 32), with cache blocks 1024 × 2048 × 4096 FP32
and 512 × 2048 × 4096 FP64. This page gives the measurements behind those
choices on an Apple M4 Max, one SME unit, and the multi-threaded kernel
that uses both units. Code: [`sme.hpp`](../../src/gemm/sme.hpp); measured
values: [`benchmarks/sme_params.hpp`](../../benchmarks/sme_params.hpp)
(`SmeParams<T>`). The other families: [AVX2](avx2.md),
[AVX-512](avx512.md), [NEON](neon.md).

## The kernels

```cpp
// T, ZA tile rows R, ZA tile columns V, TileM, TileK, TileN; no defaults.
gemm_sme<float, 2, 2, 1024, 2048, 4096>(A, B, C);           // packs A and B
gemm_sme<double, 2, 4, 512, 2048, 4096>(A64, B64, C64);
gemm_sme_blocked<float, 2, 2, 1024, 2048, 4096>(A, B, C);   // packs A, reads B in place
gemm_sme_tiles<double, 2, 4>(A64, B64, C64);                // no cache blocking
gemm_sme_mt<float, 2, 2, 1024, 2048, 4096>(A, B, C, 8);     // several SME units
```

All of them overwrite C and reload the partial sums between K blocks.
`gemm_sme_1tile` is a fixed one-tile step kept for the article's ladder.
The ZA shape and the blocks are template parameters with no defaults;
the benchmarks pass `SmeParams<T>`.

## What a ZA shape decides

Unlike the SIMD kernels' R rows × V vectors, SME's R × V counts square ZA
tiles. C's micro-tile is (R·S) × (V·S); per K step the kernel loads R
vectors of A and V of B and issues R·V outer products (FMOPA).

- **Independent accumulators, R·V.** An FMOPA's result reaches the next
  FMOPA on the same tile after 4 cycles, and the unit issues one per
  cycle, so 4 tiles are needed for the full rate
  ([sme2-machine](../experiments/sme2-machine/README.md): 1,963 GFLOP/s
  FP32, 500 FP64).
- **ZA capacity, R·V ≤ 4 or 8.** ZA holds four 32-bit tiles (FP32 and the
  widening types) or eight 64-bit tiles (FP64). Shapes over the limit
  fail at compile time
  ([Arm SME programmer's guide](https://documentation-service.arm.com/static/664f013638084307512bb30c)).
- **Operand loads, R + V per K step.** At a fixed R·V a balanced shape
  loads the fewest vectors: 2 × 2 loads 4 for 4 FMOPAs, 1 × 4 loads 5.

Every shape runs the same code: the R + V vectors are loaded two at a time
with predicated SME2 loads (an odd last vector is the inactive half of a
pair), and tile numbers are instruction immediates, so no ZA tile is
indexed at run time.

## Shape sweep, M4 Max

`bench_gemm`'s `SmeTile` benchmarks: `gemm_sme` with every legal shape and
the default cache blocks, GFLOP/s, median of 3 interleaved repetitions
([data](../experiments/sme-tiles/README.md)). Best per column in bold.

**In this sweep only the default shapes used paired loads**; the others
loaded one vector at a time. The kernel now pairs loads for every shape,
so the non-default rows may understate it.

FP32 (all eight shapes):

| ZA shape | C block | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|---|--:|--:|--:|--:|--:|
| 1 × 1 | 16 × 16 | 199.1 | 367.6 | 447.4 | 467.3 | 470.8 |
| 1 × 2 | 16 × 32 | 253.6 | 577.2 | 822.0 | 886.0 | 887.8 |
| 2 × 1 | 32 × 16 | 240.3 | 549.4 | 782.0 | 831.2 | 834.3 |
| 1 × 3 | 16 × 48 | 231.6 | 648.4 | 1,095.3 | 1,236.0 | 1,231.0 |
| 3 × 1 | 48 × 16 | 201.8 | 612.0 | 1,020.3 | 1,164.7 | 1,108.2 |
| 1 × 4 | 16 × 64 | **298.7** | **784.6** | 1,344.3 | **1,603.6** | **1,560.0** |
| **2 × 2** (default) | 32 × 32 | 291.1 | 772.2 | **1,345.9** | 1,468.0 | 1,535.9 |
| 4 × 1 | 64 × 16 | 238.5 | 642.2 | 1,085.4 | 1,265.7 | 1,274.2 |

FP64, the shapes within 10% of the best at N = 1024 (all twenty in the
[data](../experiments/sme-tiles/README.md)):

| ZA shape | C block | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|---|--:|--:|--:|--:|--:|
| 1 × 4 | 8 × 32 | **89.0** | **323.6** | **409.7** | **443.5** | 421.5 |
| 1 × 5 | 8 × 40 | 67.4 | 303.8 | 403.3 | 426.2 | 424.0 |
| 1 × 6 | 8 × 48 | 65.0 | 296.1 | 395.9 | 429.6 | 425.1 |
| 1 × 7 | 8 × 56 | 63.4 | 299.5 | 377.2 | 424.6 | 424.5 |
| 2 × 2 | 16 × 16 | 83.1 | 319.4 | 393.8 | 422.6 | 405.4 |
| 2 × 3 | 16 × 24 | 70.9 | 314.5 | 393.4 | 426.3 | 415.0 |
| **2 × 4** (default) | 16 × 32 | 88.8 | 318.3 | 404.0 | 430.9 | **429.9** |
| 3 × 2 | 24 × 16 | 72.0 | 311.7 | 383.5 | 417.5 | 389.4 |
| 4 × 2 | 32 × 16 | 85.7 | 310.3 | 381.3 | 412.5 | 402.9 |

The leading candidate against the defaults, re-measured with 7
interleaved repetitions in one session:

| Type | N | Default | 1 × 4 | 1 × 4 vs default |
|---|--:|--:|--:|--:|
| FP32 | 256 | 770.3 | 785.1 | +1.9% |
| FP32 | 1024 | 1,452.2 | 1,589.9 | +9.5% |
| FP32 | 4096 | 1,547.3 | 1,578.4 | +2.0% |
| FP64 | 256 | 320.1 | 324.6 | +1.4% |
| FP64 | 1024 | 422.7 | 441.0 | +4.3% |
| FP64 | 4096 | 430.9 | 425.1 | −1.4% |

## Why 2 × 2 / 2 × 4

- **All tiles in use.** FP32 shapes with fewer than 4 tiles cannot hide
  the FMOPA latency and stay at or below 1,236 GFLOP/s. Of the three
  4-tile shapes, 4 × 1 trails (1,274 at N = 4096); 2 × 2 and 1 × 4 are
  within 2% of each other at N = 512 and 4096.
- **FP64: 2 × 4.** The fastest shape at N = 4096 and within 3% of the best
  at N ≥ 512. 1 × 4 (4 of 8 tiles) leads at N ≤ 1024 by up to 4.3% and
  trails at N = 4096.
- **FP32: 1 × 4 is faster at every size re-measured.** In the repeated
  comparison 1 × 4 leads 2 × 2 by 1.9–9.5%, although it loads one more
  vector per K step and, in that run, loaded them one at a time. Why it
  is faster was not investigated. `SmeParams` still uses 2 × 2; switching
  is a change of template arguments (`gemm_sme<float, 1, 4, …>`), and a
  re-measurement with today's paired loads is the step before it.

## Cache blocks

TileM × TileK × TileN = 1024 × 2048 × 4096 FP32 and 512 × 2048 × 4096
FP64: the A block (TileM × TileK) is 8 MiB in both types, half the M4
P-cluster's 16 MiB L2, and each packed B strip is reused by 32 strips of
A. The values come from the run-time block sweep in
[sme2-gemm](../experiments/sme2-gemm/README.md) (TileM 128–2048, TileK
1024–4096, TileN 1024–4096, N = 512 … 4096) and were confirmed by the
`SmeBlock` grid (TileM 256–1024, TileK and TileN 1024–4096): at N = 4096
the defaults are the best configuration in the grid, 1,538.2 GFLOP/s FP32
and 435.2 FP64; at smaller N other blocks lead. The reasoning per size:
[sizing.md](sizing.md).

**Widening types (fp16, bf16, INT8).** The same templates apply
(`gemm_sme<fp16, R, V, TileM, TileK, TileN>`), with packed strips of
k-group words and TileK in input elements, a multiple of the k group G.
`SmeParams` uses 2 × 2 and FP32's blocks in k groups (TileK = 2048·G);
their shapes and blocks have not been swept.

## Edges and correctness

A is zero-padded to R·S rows and packed B to V·S columns; B read in place
uses predicates and clamped pointers for inactive columns; C is moved
with bounded rows and predicated columns. Cache blocks need not be
multiples of the micro-tile. The tests run every legal shape through the
packed-B, unpacked-B and unblocked entry points, with 5 × 7 × 13 cache
blocks, partial tiles, empty dimensions and K accumulation. Streaming
helpers share the macro-kernel's ZA state, so no call inside the K loop
triggers a private-ZA save.

## Several SME units

`gemm_sme` runs on one thread and so uses one SME unit. The M4 Max has one
unit per P-cluster and two clusters of six P-cores
(`sysctl hw.perflevel0.cpusperl2` = 6). `gemm_sme_mt` spreads `gemm_sme`'s
blocking over several threads, as multi-threaded GotoBLAS does:

- each `(jc, pc)` pass packs the B panel once, shared, with every thread
  packing every team-th strip;
- after a barrier, threads take chunks of rows from an atomic counter (at
  most TileM rows, about four chunks per thread, never less than one
  R·S-row micro-tile), pack them into their own A buffer and run
  `macro_kernel`;
- a second barrier keeps the panel intact until every chunk is done.

macOS cannot pin a thread to a cluster, and the scheduler tends to keep a
few busy threads on one cluster until it is full. Dynamic chunks let the
unit that hosts more threads take more chunks instead of finishing last.

`SmeMtT` benchmarks, GFLOP/s, median of 3 interleaved repetitions,
wall-clock time, load average 3.9, TileM = 1024 (256 and 512 are within
4% everywhere) ([data](../experiments/sme-mt/README.md)):

| Threads | FP32 N = 1024 | FP32 N = 2048 | FP32 N = 4096 | FP64 N = 1024 | FP64 N = 2048 | FP64 N = 4096 |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1,425 | 1,530 | 1,529 | 417 | 418 | 415 |
| 2 | 1,565 | 1,781 | 2,778 | 455 | 710 | 852 |
| 3 | 1,573 | 2,047 | 2,931 | 533 | 841 | 874 |
| 4 | 1,709 | 2,094 | 3,241 | 506 | 826 | 877 |
| 6 | 1,778 | 2,660 | 3,492 | 620 | 865 | 916 |
| **8** (default) | **2,682** | **3,085** | 3,558 | **764** | **905** | **931** |
| 12 | 2,545 | 2,974 | **3,597** | 755 | 882 | 923 |
| Accelerate, all cores | 3,204 | 3,180 | 3,154 | 825 | 785 | 789 |

- **8 threads** (`SmeParams::Threads`): the best or within 1.1% of the
  best everywhere. 12 threads, one per P-core, are no faster at N = 4096
  and 5% slower at N = 1024.
- **Placement.** Up to six threads, small products stay on about one
  unit; the jump from 6 to 8 threads at N = 1024 (1.8 → 2.7 TFLOP/s) fits
  threads spilling onto the second cluster. At N = 4096 a call lasts tens
  of milliseconds, long enough for the scheduler to spread even two
  threads (1.8× one thread).
- **TileM** makes no measurable difference: with 8 threads and N ≤ 4096
  the chunks are at most N / (4 · 8) = 128 rows, below every height
  swept. `SmeParams::MtMc` stays `gemm_sme`'s TileM.
- **Against Accelerate.** From N = 2048 up `gemm_sme_mt` matches or beats
  it: at N = 4096, 3.56 TFLOP/s FP32 (113% of Accelerate, 91% of the two
  units' 3.93 TFLOP/s peak) and 931 GFLOP/s FP64 (118%, 93% of 1.00). At
  N = 1024 it reaches 84% / 93% of Accelerate: a call lasts ~1 ms, so
  creating 8 threads per call and their placement still weigh. Below
  N ≈ 512 use `gemm_sme`: at N = 64 the multi-threaded kernel measures 26
  GFLOP/s against 276.

## Reproduce

On an SME2 machine (the M4 Max supports it):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON
cmake --build build --target hpc_tests bench_gemm --parallel
./build/tests/hpc_tests --gtest_filter='*Sme*'

# ZA shapes: every legal shape, N = 64 … 4096
./build/benchmarks/bench_gemm --benchmark_filter='^SmeTile[0-9]' \
  --benchmark_repetitions=7 --benchmark_enable_random_interleaving=true \
  --benchmark_out=sme-tiles.json --benchmark_out_format=json

# Cache blocks: TileM {256, 512, 1024} × TileK, TileN {1024, 2048, 4096}, N = 512 … 4096
./build/benchmarks/bench_gemm --benchmark_filter='^SmeBlock[0-9]' \
  --benchmark_repetitions=7 --benchmark_enable_random_interleaving=true \
  --benchmark_out=sme-blocks.json --benchmark_out_format=json

# Threads
./build/benchmarks/bench_gemm \
  --benchmark_filter='^(SmeMtT[0-9]+Mc[0-9]+|Sme|Amx)/f(32|64)/N=(1024|2048|4096)(/|$)' \
  --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true
```

Packing and allocation inside the GEMM call are timed, as in the other
SME benchmarks. The shape benchmarks report the ZA rows, columns and tile
count, `svl` in elements and the output micro-tile (`mr`, `nr`). On other
ISAs the SME rows report as skipped. Record compiler flags, streaming
vector length, machine, load and repetitions with each result.
