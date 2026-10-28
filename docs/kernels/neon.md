# NEON register tiles (`gemm_simd_blocked<Q>`)

`gemm_simd_blocked<Q, R, V, …>` multiplies with an R × V register tile:
R rows of C, V Q-vectors of 4 FP32 / 2 FP64 each, kept in R·V accumulators
for a whole k block. The default is **6 × 4** (6 × 16 FP32, 6 × 8 FP64) for
both types. This page gives the measurements behind that choice on an
Apple M4 Max P-core, and a code-generation change that the sweep led to.
Code: [`neon.hpp`](../../src/gemm/neon.hpp) (`vector_micro<Q<T>, R, V>`), driver
[`simd.hpp`](../../src/gemm/simd.hpp). The other
families: [AVX2](avx2.md), [AVX-512](avx512.md).

## What a tile shape decides

The same three numbers as for [AVX2](avx2.md#what-a-tile-shape-decides),
with the M4's figures:

- **Accumulators, R·V.** Up to 4 FMAs per cycle with a 3.3–3.5-cycle
  latency (`fma_peak`, [experiments/cpu-simd](../experiments/cpu-simd/README.md)),
  so 12–16 independent accumulators. Peak at the 4.49 GHz measured during
  these runs, at fma_peak's 3.97 FMAs per cycle: ≈143 GFLOP/s FP32, ≈71
  FP64.
- **Registers, R·V + V + 1.** 32 Q registers. 6 × 4 needs 29.
- **B traffic, 16 / R bytes per FMA.** As on x86, A's R × 256 piece stays
  in L1 for the j loop while each call reads a new strip of B.

## Loading A: LDR, not LD1R

Each k step needs A(i+r, k) as a multiplier for V vector FMAs. The
straightforward source, `vdupq_n_f32(a[r*lda + k])` and `vfmaq_f32`, makes
Clang 17 emit one `LD1R` (load and replicate to all lanes) per row. The
library now loads the element with a plain `LDR` into lane 0 of a D
register and multiplies by element (`FMLA v.4s, v.4s, v.s[0]`); a k step
of the 6 × 4 tile is 2 `LDP` of B, 6 `LDR` of A and 24 `FMLA`. For FP64,
`vld1_f64` + `vfmaq_lane_f64` produce that directly. For FP32, Clang folds
every intrinsic form (`vld1_lane_f32`, `vset_lane_f32`) back into `LD1R`,
so `Q<float>::load_a` is a one-instruction inline-assembly `ldr`.

Same binary layout, the library kernel with each A load, alternated three
times on the same afternoon (median GFLOP/s, [data](../experiments/tile-sweeps/m4_ld1r_ab/)):

| Tile | FP32 N = 256 | FP32 N = 1024 | FP64 N = 256 | FP64 N = 1024 |
|---|--:|--:|--:|--:|
| 4 × 4, LD1R → LDR | 111.8 → 111.0 | 102.7 → 101.8 | 55.2 → 54.8 | 47.8 → 47.8 |
| 6 × 3, LD1R → LDR | 103.0 → 102.4 | 94.4 → 94.1 | 51.2 → 51.1 | 44.9 → 44.6 |
| 6 × 4, LD1R → LDR | 94.7 → **123.2** | 96.7 → **122.1** | 47.2 → **61.4** | 47.5 → **58.2** |

With `LD1R`, the tiles with 6 rows × 4 vectors or 8 and more rows stayed
at 78–97 GFLOP/s FP32 and 37–48 FP64 at N = 256–1024, whatever their
accumulator count, and 4 × 4 was the best shape at most sizes. With `LDR`
the same tiles gain 10–33% at N = 256–1024 (the two full
sweeps, [`tiles_m4_ld1r.csv`](../experiments/tile-sweeps/tiles_m4_ld1r.csv)
and [`tiles_m4.csv`](../experiments/tile-sweeps/tiles_m4.csv), N = 256):

| Tile | FP32, LD1R → LDR | FP64, LD1R → LDR |
|---|--:|--:|
| 8 × 2 | 82.7 → 98.8 | 41.5 → 49.5 |
| 8 × 3 | 85.9 → 114.2 | 43.3 → 57.7 |
| 6 × 4 | 94.8 → 121.2 | 47.6 → 60.9 |
| 12 × 2 | 79.0 → 102.2 | 39.4 → 51.8 |

The two sweeps ran an hour apart under different background load; shapes
the change does not affect differ between them by up to 7% (4 × 4 FP32:
115.2 and 109.7), against ≤ 1% in the alternated runs above. Apple does
not document `LD1R`'s throughput, and 6 × 3, which has 6 rows, does not
gain; the rule above is what was measured, not a model of the core.

`simd_microkernel` does not show the difference: there `lda` and `ldb`
are compile-time constants, and Clang emits by-element FMLAs from either
source.

## Tile sweep, M4

`bench_gemm`'s `NeonTile<R>x<V>` benchmarks with the `LDR` kernel: GFLOP/s,
median of 3 ([data](../experiments/tile-sweeps/README.md)). Best per
column in bold.

FP32:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 82.8 | 68.7 | 68.2 | 65.2 | 53.4 |
| 2 × 4 | 8 | 80.4 | 65.5 | 64.6 | 58.9 | 44.6 |
| 4 × 4 | 16 | 117.5 | 109.7 | 108.6 | 100.7 | 75.2 |
| 8 × 2 | 16 | 114.0 | 98.8 | 99.1 | 97.7 | 65.3 |
| 6 × 3 | 18 | 112.3 | 101.1 | 100.8 | 92.3 | 74.2 |
| 8 × 3 | 24 | 114.8 | 114.2 | 114.4 | 110.3 | 78.8 |
| 5 × 4 | 20 | **122.0** | 119.4 | 120.0 | 114.8 | 84.5 |
| **6 × 4** (default) | 24 | 121.0 | **121.2** | **122.1** | **120.4** | **90.8** |
| 12 × 2 | 24 | 109.6 | 102.2 | 103.2 | 101.9 | 59.3 |
| 2 × 8 | 16 | 121.4 | 78.9 | 75.3 | 62.3 | 52.8 |

FP64:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 42.7 | 34.9 | 33.5 | 32.8 | 27.0 |
| 2 × 4 | 8 | 41.3 | 32.7 | 29.8 | 27.5 | 22.3 |
| 4 × 4 | 16 | 58.3 | 54.1 | 50.7 | 47.1 | 37.3 |
| 8 × 2 | 16 | 56.5 | 49.5 | 49.1 | 45.3 | 36.0 |
| 6 × 3 | 18 | 59.5 | 50.5 | 46.2 | 44.1 | 36.8 |
| 8 × 3 | 24 | 61.0 | 57.7 | 55.4 | 52.7 | 42.9 |
| 5 × 4 | 20 | **61.7** | 60.2 | 57.2 | 54.2 | 41.7 |
| **6 × 4** (default) | 24 | 60.7 | **60.9** | **60.0** | **57.8** | **45.2** |
| 12 × 2 | 24 | 55.5 | 51.8 | 51.5 | 46.2 | 32.6 |
| 2 × 8 | 16 | 61.5 | 37.5 | 31.0 | 29.2 | 25.9 |

The sweep ran with the cache block of the time, 64 × 256 × 256; the blocks
were swept afterwards ([blocks.md](blocks.md)).

Variation: coefficient of variation ≤ 1.1% in every cell. Load average
8–14 on 16 cores during the sweep (Spotlight indexing and an IDE, about
2.5 of the 12 P-cores busy); core clock 4.48–4.49 GHz.

No shape spills: Clang uses up to 32 Q registers (8 × 3 and larger), and
the only stack accesses are the saves of the callee-saved d8–d15 outside
the loop.

## Why 6 × 4

- **Accumulators.** The 8-accumulator shapes run at 48–60% of 6 × 4
  from N = 256 up: the M4 needs 12–16 independent FMAs in flight.
- **Wide rows of B.** Among shapes with enough accumulators, 4 vectors
  per row beat 2 or 3 at the same row count or accumulator count: 6 × 4
  against 8 × 3 and 12 × 2 (24 accumulators each), 4 × 4 against 8 × 2
  (16). The sweep shows this; it does not isolate why. The wide but short
  2 × 8 is fast only at N = 64, where all of B fits in L1; past that it
  reads 8 bytes of B per FMA and falls to 53–79 GFLOP/s.
- **Enough rows.** From 4 × 4 to 5 × 4 to 6 × 4, B's bytes per FMA drop
  from 4 to 2.7, and the kernel gains 11–21% FP32 and 13–23% FP64 at
  N ≥ 256. 7 × 4 would need 33 registers.

6 × 4 reaches 85% of the FP32 peak and 85% of FP64 (121.2 and 60.9
GFLOP/s at N = 256). With the cache blocks chosen since
([blocks.md](blocks.md)) it runs at 123.8 / 62.2 at its best size and
102.1 / 51.0 at N = 4096 (was 91.9 / 45.8).

## History

Three changes, each measured against the one before at the same clock:

| `gemm_simd_blocked<Q>` | FP32 N = 256 | FP32 N = 1024 | FP64 N = 256 | FP64 N = 1024 |
|---|--:|--:|--:|--:|
| 4 × 4 / 8 × 2, one hand-written micro-kernel per type (to commit 3bd58c0) | 97.3 | 95.7 | 41.2 | 40.5 |
| the same shapes through the template micro-kernel and shared driver | 110.9 | 102.0 | 40.6 | 40.0 |
| 6 × 4 with `LDR` (from the sweep above) | 121.2 | 120.4 | 60.9 | 57.8 |

The first two rows are three alternated runs
([data](../experiments/tile-sweeps/m4_driver_ab/)). The template version
broadcasts all four rows of A with `LD1R`, where the hand-written one had
Clang fold 12 of 16 FMAs per k step into the by-element form; why that
made the older kernel slower was not investigated. Overall: +25% FP32 and
+48% FP64 at N = 256.

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_gemm
./build/benchmarks/bench_gemm --benchmark_filter='^NeonTile' \
    --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
```

A shape not in the sweep: `gemm_simd_blocked<Q, R, V, TileM, TileK, TileN>(A, B, C)`, or a
line `HPC_NEON_TILE(R, V);` in `add_tiles()` of
[`bench_gemm.cpp`](../../benchmarks/bench_gemm.cpp).
