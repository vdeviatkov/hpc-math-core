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
