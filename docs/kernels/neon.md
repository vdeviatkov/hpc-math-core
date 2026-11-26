# NEON register tiles (`gemm_simd_blocked<Q>`)

`gemm_simd_blocked<Q, R, V, …>` multiplies with an R × V register tile:
R rows of C, V Q-vectors of 4 FP32 / 2 FP64 each, kept in R·V accumulators
for a whole k block. The default is **6 × 4** for both types,
which covers a 6 × 16 block of C in FP32 and 6 × 8 in FP64. This page gives the measurements behind that choice on an
Apple M4 Max P-core, and why the kernel loads A with `LDR`.
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
library loads the element with a plain `LDR` into lane 0 of a D
register and multiplies by element (`FMLA v.4s, v.4s, v.s[0]`); a k step
of the 6 × 4 tile is 2 `LDP` of B, 6 `LDR` of A and 24 `FMLA`. For FP64,
`vld1_f64` + `vfmaq_lane_f64` produce that directly. For FP32, Clang folds
every intrinsic form (`vld1_lane_f32`, `vset_lane_f32`) back into `LD1R`,
so `Q<float>::load_a` is a one-instruction inline-assembly `ldr`.

The library kernel built with each A load, the two builds alternated
three times (median GFLOP/s, [data](../experiments/tile-sweeps/m4_ld1r_ab/)):

| Tile | FP32 N = 256 | FP32 N = 1024 | FP64 N = 256 | FP64 N = 1024 |
|---|--:|--:|--:|--:|
| 4 × 4, LD1R → LDR | 111.8 → 111.0 | 102.7 → 101.8 | 55.2 → 54.8 | 47.8 → 47.8 |
| 6 × 3, LD1R → LDR | 103.0 → 102.4 | 94.4 → 94.1 | 51.2 → 51.1 | 44.9 → 44.6 |
| 6 × 4, LD1R → LDR | 94.7 → **123.2** | 96.7 → **122.1** | 47.2 → **61.4** | 47.5 → **58.2** |

With `LD1R`, 6 × 4 is slower than 4 × 4; with `LDR` it gains 23–30%
and becomes the fastest shape. Apple does not document `LD1R`'s
throughput, and 6 × 3, which also has 6 rows, does not gain; this is what
was measured, not a model of the core.

`simd_microkernel` does not show the difference: there `lda` and `ldb`
are compile-time constants, and Clang emits by-element FMLAs from either
source.

## Tile sweep, M4

`bench_gemm`'s `NeonTile<R>x<V>` benchmarks: GFLOP/s,
median of 3 ([data](../experiments/tile-sweeps/README.md)). Best per
column in bold.

FP32:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 80.8 | 66.9 | 66.8 | 63.4 | 53.1 |
| 2 × 4 | 8 | 79.5 | 65.2 | 64.2 | 58.3 | 44.7 |
| 4 × 4 | 16 | 116.9 | 109.6 | 108.5 | 100.0 | 78.4 |
| 8 × 2 | 16 | 113.6 | 98.7 | 99.1 | 97.2 | 68.4 |
| 6 × 3 | 18 | 112.4 | 103.1 | 101.8 | 95.5 | 77.8 |
| 8 × 3 | 24 | 115.2 | 114.2 | 113.8 | 110.3 | 81.2 |
| 5 × 4 | 20 | **122.2** | 119.2 | 120.9 | 115.8 | 92.2 |
| **6 × 4** (default) | 24 | 121.3 | **122.3** | **122.8** | **122.3** | **101.0** |
| 12 × 2 | 24 | 107.9 | 105.6 | 105.2 | 104.5 | 63.9 |
| 2 × 8 | 16 | 121.1 | 79.9 | 76.1 | 63.5 | 54.2 |

FP64:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 40.4 | 33.4 | 32.1 | 30.8 | 24.9 |
| 2 × 4 | 8 | 40.5 | 32.1 | 29.0 | 27.1 | 22.1 |
| 4 × 4 | 16 | 58.1 | 54.0 | 50.3 | 47.2 | 39.1 |
| 8 × 2 | 16 | 56.4 | 49.4 | 49.0 | 45.3 | 37.8 |
| 6 × 3 | 18 | 59.5 | 51.1 | 47.0 | 45.3 | 38.9 |
| 8 × 3 | 24 | 60.9 | 57.8 | 55.2 | 52.9 | 44.7 |
| 5 × 4 | 20 | **61.9** | 60.1 | 57.4 | 52.9 | 45.1 |
| **6 × 4** (default) | 24 | 60.8 | **61.2** | **60.4** | **59.1** | **50.6** |
| 12 × 2 | 24 | 55.5 | 53.5 | 53.3 | 48.3 | 35.5 |
| 2 × 8 | 16 | 60.5 | 37.8 | 31.5 | 29.4 | 26.7 |

The sweep runs each shape at the default cache block (384 × 256 × 256;
[blocks.md](blocks.md)). Variation: coefficient of variation ≤ 3.8% in
every cell. Load average 13–15 on 16 cores during the sweep.

No shape spills: Clang uses up to 32 Q registers (8 × 3 and larger), and
the only stack accesses are the saves of the callee-saved d8–d15 outside
the loop.

## Why 6 × 4

- **Accumulators.** The 8-accumulator shapes run at 44–55% of 6 × 4
  from N = 256 up: the M4 needs 12–16 independent FMAs in flight.
- **Wide rows of B.** Among shapes with enough accumulators, 4 vectors
  per row beat 2 or 3 at the same row count or accumulator count: 6 × 4
  against 8 × 3 and 12 × 2 (24 accumulators each), 4 × 4 against 8 × 2
  (16). The sweep shows this; it does not isolate why. The wide but short
  2 × 8 is fast only at N = 64, where all of B fits in L1; past that it
  reads 8 bytes of B per FMA and falls to 54–80 GFLOP/s.
- **Enough rows.** From 4 × 4 to 5 × 4 to 6 × 4, B's bytes per FMA drop
  from 4 to 2.7, and the kernel gains 12–29% FP32 and 13–29% FP64 at
  N ≥ 256. 7 × 4 would need 33 registers.

6 × 4 reaches 85% of the FP32 peak and 85% of FP64 (122.3 and 61.2
GFLOP/s at N = 256) in the sweep. With its default cache block
([blocks.md](blocks.md)) it runs at 123.8 / 62.2 at its best size and
102.1 / 51.0 at N = 4096
([bench_m4_defaults.csv](../experiments/block-sweeps/bench_m4_defaults.csv)).

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
