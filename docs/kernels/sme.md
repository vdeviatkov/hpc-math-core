# SME2: configurable ZA shapes and cache blocks

```cpp
// T, ZA tile rows, ZA tile columns, TileM, TileK, TileN.
gemm_sme<float, 1, 4>(A, B, C);
gemm_sme<double, 4, 2, 512, 2048, 4096>(A64, B64, C64);
gemm_sme_blocked<float, 2, 2, 256, 1024, 2048>(A, B, C); // B in place
gemm_sme_tiles<double, 4, 2>(A64, B64, C64); // no cache blocking
```

`gemm_sme` packs A and B; `gemm_sme_blocked` packs only A. Both overwrite C,
then reload partial sums between K blocks. Defaults are unchanged: 2×2
FP32 / 2×4 FP64 ZA shapes and M/K/N blocks 1024/2048/4096 FP32,
512/2048/4096 FP64. The original `gemm_sme_1tile` remains a fixed one-tile
teaching step. [Why the defaults were chosen](sizing.md).

## Shape units and limits

Unlike SIMD R rows × V vectors, SME R×V counts square ZA tiles. Let S be
the streaming vector length **in elements of T**. C's micro-tile is
(R·S)×(V·S), each K step loads R A and V B vectors and computes R·V outer
products. ZA contains four FP32 or eight FP64 tiles. Shapes must be
positive and R·V ≤ 4 / 8; cache sizes must be positive. Invalid choices
fail at compile time. These limits follow from
[Arm's SME programmer guide](https://documentation-service.arm.com/static/664f013638084307512bb30c).

More tiles give independent accumulation chains; with all tiles in use,
shape still affects operand reuse. At a fixed product R·V, a balanced
rectangle minimizes R+V operand vectors, but measured load scheduling,
packing, edges and reuse can favor another shape.

The default shapes preserve their paired SME2 loads and full-tile C
transfer path. Other shapes use compile-time expansions with single-vector
predicated loads, loading each B vector once per K step and reusing it
across R A vectors. Thus a sweep compares the actual implementations,
including these load schedules. A winning nondefault shape may warrant a
paired-load specialization before changing the default. No runtime ZA-tile
indexing is introduced: tile numbers are instruction immediates.

A is zero-padded to R·S rows; packed B is zero-padded to V·S columns.
Unpacked B uses predicates and clamped inactive-column pointers. C movement
bounds rows and predicates columns. Cache blocks need not be multiples of
the micro-tile; all legal shapes are tested with 5×7×13 cache blocks too.
Streaming helpers share the macro-kernel's ZA state, so calls do not force
private-ZA saves inside K.

## Reproduce and publish

On an SME2-capable machine (the local M4 Max supports it):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON
cmake --build build --target hpc_tests bench_gemm --parallel
./build/tests/hpc_tests --gtest_filter='*Sme*'
./build/benchmarks/bench_gemm --benchmark_filter='^SmeTile[0-9]' \
  --benchmark_repetitions=7 --benchmark_enable_random_interleaving=true \
  --benchmark_out=sme-tiles.json --benchmark_out_format=json
./build/benchmarks/bench_gemm --benchmark_filter='^SmeBlock[0-9]' \
  --benchmark_repetitions=7 --benchmark_enable_random_interleaving=true \
  --benchmark_out=sme-blocks.json --benchmark_out_format=json
```

The tile sweep covers every legal positive rectangular shape: eight FP32
and twenty FP64 shapes, each at N=64/256/512/1024/4096. Counters include
ZA rows/columns/tile count, `svl` in elements, and actual output `mr`/`nr`.
The cache grid uses M={256,512,1024}, K/N={1024,2048,4096}, at
N=512/1024/4096 and the default shape for each type. Packing and allocation
inside the GEMM call are timed, as in the original SME benchmarks.
Other ISAs/builds report skipped rows rather than substitute kernels.

Keep compiler flags, SVL, machine, load and repetitions with results.
[Historical cache measurements](../experiments/sme2-gemm/README.md) and
[new shape measurements](../experiments/sme-tiles/README.md) are separate.
These template controls currently cover FP32 / FP64; SME widening kernels
have a separate packed k-group layout and retain their fixed 2×2 ZA32 shape.

The new M4 sweep finds FP32 1×4 about 9.5% faster than 2×2 at N=1024,
confirmed with seven interleaved repetitions. FP64 1×4 is about 4.3% faster
at that size but 1.4% slower at N=4096. Thus the historical balanced shape
is not optimal at every size. Defaults are preserved while alternatives
can now be selected directly; see the linked results for all shapes and
conditions. The default before/after check finds −1.03% to +1.40% changes
across six cases, within desktop run variation.

## Several SME units

`gemm_sme` runs on one thread, so it reaches one SME unit at most. An SME
unit belongs to a P-cluster, and the M4 Max has two clusters of six P-cores
(`sysctl hw.perflevel0.cpusperl2` = 6). Each has its own unit. Accelerate
with default threading uses both and measures about twice the single-core
figure. `gemm_sme_mt<T, TileM>(A, B, C, threads = kSmeThreads)` is
`gemm_sme`'s blocking spread over several threads, as in multi-threaded
GotoBLAS:

- each `(jc, pc)` pass packs the B panel once, shared, with every thread
  packing every team-th strip;
- after a barrier, threads take chunks of rows from an atomic counter (at
  most `TileM` rows, about four chunks per thread, never less than one 2·SVL
  micro-tile), pack them into their own A buffer and run `macro_kernel`;
- a second barrier keeps the panel intact until every chunk is done.

macOS cannot pin a thread to a cluster, so placement is up to the
scheduler. The first version gave each thread an equal static band of rows.
It showed the scheduler keeping a few threads on one cluster: with 2, 4 or 6
threads, N ≤ 2048 ran no faster than one thread (f32 N=1024: 1,447 GFLOP/s
on one thread, 1,470 on two), and 3 threads were slower than 2 at N=4096
(2.3 against 2.7 TFLOP/s), because the unit that got two of the three
threads finished last. With dynamic chunks, a unit that hosts more threads
simply takes more chunks.

Sweep: `--benchmark_filter='^(SmeMtT[0-9]+Mc[0-9]+|Sme|Amx)/f(32|64)/N=(1024|2048|4096)(/|$)'`,
median of 3 interleaved repetitions, wall-clock time, load average 3.9
([data](../experiments/sme-mt/README.md)). GFLOP/s, `TileM` = 1024 (256 and
512 are within 3% of it everywhere):

| Threads | f32 N=1024 | f32 N=2048 | f32 N=4096 | f64 N=1024 | f64 N=2048 | f64 N=4096 |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1,425 | 1,530 | 1,529 | 417 | 418 | 415 |
| 2 | 1,565 | 1,781 | 2,778 | 455 | 710 | 852 |
| 3 | 1,573 | 2,047 | 2,931 | 533 | 841 | 874 |
| 4 | 1,709 | 2,094 | 3,241 | 506 | 826 | 877 |
| 6 | 1,778 | 2,660 | 3,492 | 620 | 865 | 916 |
| **8** | **2,682** | **3,085** | **3,558** | **764** | **905** | **931** |
| 12 | 2,545 | 2,974 | 3,597 | 755 | 882 | 923 |
| Accelerate, all cores | 3,204 | 3,180 | 3,154 | 825 | 785 | 789 |

- Up to six threads, small products stay on about one unit. The jump
  between 6 and 8 threads at N=1024 (1.8 → 2.7 TFLOP/s) fits threads
  spilling onto the second cluster once the first is full. At N=4096 each
  call lasts tens of milliseconds, enough for the scheduler to spread even
  two threads, and two threads already reach 1.8× one.
- 8 threads is the default (`kSmeThreads`): best or within 1.1% of the best everywhere.
  12 threads, one per P-core, are no better at N=4096 and 5% worse at
  N=1024.
- `TileM` makes no measurable difference, because chunks are smaller than
  any of the three heights from N ≤ 4096 / (4 · 8). `kSmeMtMc` stays
  `kSmeMc`.
- From N=2048 up, `gemm_sme_mt` matches or beats Accelerate. At N=4096 it
  reaches 3.56 TFLOP/s f32, 113% of Accelerate and 91% of the two units'
  3.93 TFLOP/s peak, and 931 GFLOP/s f64 (118%, 93% of 1.00 TFLOP/s). At
  N=1024 it reaches 84% (f32) / 93% (f64): each call lasts ~1 ms, so the
  thread start-up (8 threads are created per call) and the scheduler's
  placement still weigh. Below N≈512, use `gemm_sme`: at N=64 the
  multi-threaded kernel measures 26 GFLOP/s against 276.

```bash
./build/tests/hpc_tests --gtest_filter='*SmeMt*:GemmSme.MtSeveralPasses'
./build/benchmarks/bench_gemm --benchmark_filter='^(SmeMt|Sme|Amx)/' \
  --benchmark_repetitions=5 --benchmark_enable_random_interleaving=true
```
