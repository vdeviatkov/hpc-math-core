# SME2 ZA tiles and cache blocks (`gemm_sme`)

`gemm_sme<T, R, V, TileM, TileK, TileN>` multiplies on the SME2 matrix
unit with an R × V arrangement of ZA tiles, each holding an S × S block of
C, where S is the streaming vector length in elements (16 FP32, 8 FP64 on
the M4). The measured defaults are **1 × 4** in both types (a 16 × 64
block of C in FP32, 8 × 32 in FP64), with cache blocks 1024 × 2048 × 4096
FP32 and 512 × 2048 × 4096 FP64; the widening types use 2 × 2. This page
gives the measurements behind those choices on an Apple M4 Max, one SME unit, and the multi-threaded kernel
that uses both units. Code: [`sme.hpp`](../../src/gemm/sme.hpp); measured
values: [`benchmarks/sme_params.hpp`](../../benchmarks/sme_params.hpp)
(`SmeParams<T>`). The other families: [AVX2](avx2.md),
[AVX-512](avx512.md), [NEON](neon.md).

## The kernels

```cpp
// T, ZA tile rows R, ZA tile columns V, TileM, TileK, TileN; no defaults.
gemm_sme<float, 1, 4, 1024, 2048, 4096>(A, B, C);           // packs A and B
gemm_sme<double, 1, 4, 512, 2048, 4096>(A64, B64, C64);
gemm_sme_blocked<float, 1, 4, 1024, 2048, 4096>(A, B, C);   // packs A, reads B in place
gemm_sme_tiles<double, 1, 4>(A64, B64, C64);                // no cache blocking
gemm_sme_mt<float, 1, 4, 1024, 2048, 4096>(A, B, C, 8);     // several SME units
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

`bench_gemm`'s `SmeTile` benchmarks: `gemm_sme` with every 4-tile FP32
shape and the leading FP64 shapes at the default cache blocks, GFLOP/s,
median of 7 interleaved repetitions (load average 12–13; coefficient of
variation ≤ 2.7%; [data](../experiments/sme-tiles/README.md)). Best per
column in bold.

| Type | ZA shape | C block | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|---|---|--:|--:|--:|--:|
| FP32 | **1 × 4** (default) | 16 × 64 | **768.5** | 1,322.7 | **1,574.8** | **1,602.9** |
| FP32 | 2 × 2 | 32 × 32 | 758.4 | **1,333.2** | 1,467.2 | 1,557.6 |
| FP32 | 4 × 1 | 64 × 16 | 663.4 | 1,143.7 | 1,365.0 | 1,422.6 |
| FP64 | **1 × 4** (default) | 8 × 32 | **318.8** | **407.3** | **439.7** | **429.5** |
| FP64 | 2 × 4 | 16 × 32 | 314.9 | 405.7 | 424.2 | 426.3 |
| FP64 | 2 × 2 | 16 × 16 | **318.8** | 392.8 | 415.4 | 419.2 |
| FP64 | 4 × 1 | 32 × 8 | 294.9 | 356.8 | 390.8 | 387.0 |

## Why 1 × 4

- **All tiles' worth of chains.** An FMOPA takes 4 cycles before the next
  one on the same tile can start and the unit issues one per cycle
  ([sme2-machine](../experiments/sme2-machine/README.md)), so a shape needs
  4 tiles to keep it busy. FP32 has exactly 4; FP64 needs 4 of its 8, so
  1 × 4 reaches the full rate in both types.
- **1 × 4 is the fastest 4-tile shape.** It leads
  2 × 2 FP32 by 7.3% at N = 1024 and 2.9% at 4096 (−0.8% at 512, within
  1% at 256), and 2 × 4 FP64 by 0.4–3.7% at every size, although 1 × 4
  loads 5 vectors per 4 FMOPAs where 2 × 2 loads 4.
- **The difference is outside the micro-kernel.** On cache-resident packed
  operands (`sme_microkernel`, the macro-kernel alone, K = 2048) every
  shape runs at 99–100% of the FMOPA peak: FP32 1 × 4, 2 × 2, 4 × 1 at
  1,954, 1,961, 1,966 GFLOP/s, FP64 1 × 4, 2 × 4, 2 × 2, 4 × 2 at 496–497
  ([data](../experiments/sme-tiles/microkernel_m4.csv)). The full GEMM
  reaches 82% (1 × 4 FP32, N = 4096), so its last 18% and the gap between
  shapes come from moving data (packing, B from L2, C in and out of ZA),
  not from the FMOPA schedule. At K = 256 the same kernel reaches 88–89%
  FP32: C moves in and out of ZA once per call, eight times as often per
  FMOPA as at K = 2048.
- **Not the tall shape.** 4 × 1, with the same loads per FMOPA as 1 × 4,
  is 8–14% behind it in both types.

## Cache blocks

TileM × TileK × TileN = 1024 × 2048 × 4096 FP32 and 512 × 2048 × 4096
FP64: the A block (TileM × TileK) is 8 MiB in both types, half the M4
P-cluster's 16 MiB L2, and each packed B strip is reused by 64 strips of
A (1 × 4: 16 rows FP32, 8 FP64). The values come from a run-time block
sweep at the 1 × 4 shape (TileM 128–2048, TileK 1024–4096, TileN
1024–4096, N = 512 … 4096), followed by the six leading blocks compiled in
and interleaved (`SmeBlock`, 5 repetitions;
[data](../experiments/sme2-gemm/README.md)): the defaults are the best or
within 0.6% of the best at N = 512, 1024 and 4096 in both types (FP32
1,320.8 / 1,574.4 / 1,597.8, FP64 405.4 / 436.8 / 430.2 GFLOP/s). The reasoning per size:
[sizing.md](sizing.md).

**Widening types (fp16, bf16, INT8).** The same templates apply
(`gemm_sme<fp16, R, V, TileM, TileK, TileN>`), with packed strips of
k-group words and TileK in input elements, a multiple of the k group G.
`SmeParams` uses 1 × 4 and FP32's blocks in k groups (TileK = 2048·G).
1 × 4 leads 2 × 2 at every size measured, 7 interleaved repetitions
([data](../experiments/lowp/sweeps-m4/confirm.json)), GFLOP/s or GOP/s:

| Type | N = 256 | 1024 | 2048 | 4096 |
|---|--:|--:|--:|--:|
| bf16, 1 × 4 / 2 × 2 | 965.5 / 946.8 | 1,754.0 / 1,739.0 | 1,856.2 / 1,810.2 | 1,858.8 / 1,807.5 |
| fp16, 1 × 4 / 2 × 2 | 965.7 / 954.5 | 1,753.1 / 1,739.4 | 1,851.0 / 1,807.1 | 1,855.8 / 1,805.6 |
| INT8, 1 × 4 / 2 × 2 | 1,600.6 / 1,570.8 | 3,382.6 / 3,284.5 | 3,688.7 / 3,646.0 | 3,758.6 / 3,541.1 |

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
wall-clock time, load average 6–10, TileM = 1024
([data](../experiments/sme-mt/README.md)):

| Threads | FP32 N = 1024 | FP32 N = 2048 | FP32 N = 4096 | FP64 N = 1024 | FP64 N = 2048 | FP64 N = 4096 |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1,536 | 1,553 | 1,592 | 423 | 399 | 414 |
| 2 | 1,690 | 1,868 | 2,921 | 466 | 720 | 839 |
| 3 | 1,702 | 2,307 | 3,114 | 569 | 826 | 833 |
| 4 | 1,783 | 2,176 | 3,172 | 475 | 791 | 862 |
| 6 | 1,812 | 2,802 | 3,350 | 647 | 845 | 915 |
| 7 | **2,808** | 2,881 | **3,607** | **791** | 892 | 901 |
| **8** (default) | 2,768 | **3,096** | 3,500 | 767 | 885 | 920 |
| 10 | 2,680 | 2,933 | 3,570 | 748 | **899** | **935** |
| 12 | 2,606 | 3,053 | 3,551 | 779 | 876 | 926 |
| Accelerate, all cores | 3,253 | 3,263 | 3,149 | 825 | 792 | 790 |

- **8 threads** (`SmeParams::Threads`): the best at FP32 N = 2048 and
  within 3.0% of the best elsewhere. 7 threads trail by 7% at FP32
  N = 2048; 10 and 12 lead by at most 1.6%, and 12 is 6% slower at FP32
  N = 1024.
- **Placement.** Up to six threads, small products stay on about one
  unit. Seven, the smallest count that cannot fit one six-core cluster, is
  where the jump happens: FP32 N = 1024 goes from 1.8 TFLOP/s with 6
  threads to 2.8 with 7. At N = 4096 a call lasts tens of milliseconds,
  long enough for the scheduler to spread even two threads (1.9× one
  thread).
- **TileM**: with 8 threads and N ≤ 4096 the chunks are at most
  N / (4 · 8) = 128 rows, below any useful TileM, so `SmeParams::MtMc`
  stays `gemm_sme`'s TileM.
- **Against Accelerate** (5 interleaved repetitions,
  [data](../experiments/sme-mt/default.json)): from N = 2048
  up `gemm_sme_mt` matches or beats it: at N = 4096, 3.49 TFLOP/s FP32
  (110% of Accelerate, 89% of the two units' 3.93 TFLOP/s peak) and 910
  GFLOP/s FP64 (115%, 91% of 1.00). At N = 1024 it reaches 87% / 92% of
  Accelerate: a call lasts ~1 ms, so creating 8 threads per call and their
  placement still weigh. Below N ≈ 512 use `gemm_sme`: at N = 64 the
  multi-threaded kernel measures 16 GFLOP/s against 250.

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
