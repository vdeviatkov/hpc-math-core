# Metal tiles (`gemm_reg_tile`, `gemm_simd`, `gemm_simd_tg`)

The Metal kernels on the Apple M4 Max GPU (40 cores) have two kinds of
tile. Level 2 gives each thread a TM × TN block of C in registers, as the
CPU kernels do. Levels 3–5 give each simdgroup (32 threads) an SM × SN
block of C as 8 × 8 `simdgroup_float8x8` accumulator fragments. Every
size is a template parameter of the kernel, and every variant is compiled
under its own name. This page gives the sweep behind the defaults. Code:
[`gemm.metal`](../../src/metal/gemm.metal), host and level list
[`metal.hpp`](../../src/gemm/metal.hpp). The CUDA counterpart:
[cuda.md](cuda.md).

Two sizes are fixed by Metal, not chosen: `kVec = 4` (`float4` is the
widest vector type, so tile edges that loads walk along are multiples of
4) and `kFrag = 8` (`simdgroup_matrix<T, 8, 8>` is the only fragment
shape, so simdgroup tiles and the k step are multiples of 8).

## What a tile decides

**Level 2, TM × TN outputs per thread.** Per k step a thread reads TM
values of A and TN of B from threadgroup memory and issues TM·TN FMAs.
8 × 4 reads 12 values for 32 FMAs; 4 × 4 reads 8 for 16; 8 × 8 reads 16
for 64. More outputs per thread means fewer reads per FMA, but more
accumulator registers (32, 16, 64) and fewer threads per block tile.

**Levels 3–5, SM × SN per simdgroup.** Per 8-step of k a simdgroup loads
SM/8 fragments of A and SN/8 of B and issues (SM/8)·(SN/8) fragment
multiply-adds: 32 × 32 loads 8 fragments for 16 multiply-adds, 16 × 32 and
32 × 16 load 6 for 8. A threadgroup is WM × WN simdgroups, so its tile of
C is (WM·SM) × (WN·SN). Levels 4–5 also stage BK steps of k of A and B in
threadgroup memory per barrier.

## Tile sweep, M4 Max

`bench_gemm_metal`'s `MetalVariant/<function>` benchmarks, GFLOP/s at
N = 2048 / 4096, median of 5 interleaved repetitions, 2026-10-24
([data](../experiments/metal-m4/README.md)). The default of each level is
the plain name; best of each level per column in bold.

| Level | Variant | Tile of C per threadgroup | N = 2048 | N = 4096 |
|--:|---|---|--:|--:|
| 2 | `gemm_reg_tile` (default: 64 × 64, 8 × 4 per thread) | 64 × 64 | **6,726** | **6,075** |
| 2 | `gemm_reg_tile_64x64_4x4` | 64 × 64 | 5,792 | 4,852 |
| 2 | `gemm_reg_tile_128x64_8x8` | 128 × 64 | 6,207 | 4,718 |
| 2 | `gemm_reg_tile_128x128_8x8` | 128 × 128 | 5,029 | 5,119 |
| 3 | `gemm_simd` (default: 2 × 2 simdgroups of 32 × 32) | 64 × 64 | 9,850 | **10,391** |
| 3 | `gemm_simd_2x2_16x32` | 32 × 64 | **9,929** | 8,432 |
| 3 | `gemm_simd_2x2_32x16` | 64 × 32 | 8,590 | 7,732 |
| 4 | `gemm_simd_tg` (default: 2 × 2 of 32 × 32, BK = 16) | 64 × 64 | 9,526 | 9,522 |
| 4 | `gemm_simd_tg_2x2_32x32_k32` | 64 × 64 | 9,333 | 8,374 |
| 4 | `gemm_simd_tg_2x2_64x32_k16` | 128 × 64 | 9,399 | **9,564** |
| 4 | `gemm_simd_tg_2x4_32x32_k16` | 64 × 128 | 8,749 | 9,251 |
| 4 | `gemm_simd_tg_4x2_32x32_k16` | 128 × 64 | **9,680** | 8,520 |
| 5 | `gemm_simd_tg_f16` (default, fp16 in) | 64 × 64 | 10,762 | **13,118** |
| 5 | `gemm_simd_tg_f16_2x2_32x32_k32` | 64 × 64 | **11,818** | 10,845 |
| 5 | `gemm_simd_tg_f16_2x2_64x32_k16` | 128 × 64 | 11,186 | 11,690 |
| 5 | `gemm_simd_tg_f16_2x4_32x32_k16` | 64 × 128 | 11,625 | 11,440 |

Variation: coefficient of variation up to 18% per row. Absolute GPU rates
on this laptop moved by up to ~3× between sessions (power limiting under
sustained memory-heavy load, inferred; no GPU clock reading was available),
so compare rows of this run with each other, not with other runs.

## Why these defaults

- **Level 2: 8 × 4 per thread.** The only clear result of the sweep: the
  default is the best at both sizes, 16–25% ahead of 4 × 4 and 8–34%
  ahead of the two 8 × 8 variants. It replaced 4 × 4 after an earlier single pass (7,477
  against 6,445 GFLOP/s at N = 4096), and this run confirms it. Why 8 × 8,
  with fewer reads per FMA, is slower was not investigated; this run
  records no register counts or GPU counters.
- **Levels 3–5: kept, not proven best.** No variant beats its default at
  both sizes, and most gaps are within this run's noise. Where a variant
  wins at one size it loses more at the other: Level 3's 16 × 32 is 1%
  ahead at N = 2048 and 19% behind at 4096; Level 5's BK = 32 is 10% ahead
  at 2048 and 17% behind at 4096. The defaults stay because nothing
  measured better, not because the sweep shows a reason for 32 × 32.
- **Not swept:** Level 1's 16 × 16 tile, Level 3's 2 × 2 simdgroup
  arrangement (only the per-simdgroup block varies), and any size at
  N ≤ 1024.

## A code-generation trap: unrolling

The first build of Levels 3–5 ran the 32 × 32-per-simdgroup kernels at
1.2–2.0 TFLOP/s, while the 16 × 32 variant reached 11.8 and the fp16 copy
of the same code 12.4. The fixed-trip loops over the accumulator array were
not unrolled, so the array was indexed at run time and lived in memory
instead of registers. `#pragma clang loop unroll(full)` on those loops
(`HPC_UNROLL` in `gemm.metal`) brought them to 11–13 TFLOP/s. This is the
GPU form of the CPU kernels' rule that the tile loops must unroll
completely so the accumulators stay in registers.

## How far the kernels get

Measured FMA peak in the same run: 14.2 TFLOP/s FP32 (14.9 run alone).
fp16 FMA runs at the same rate: on M1–M4 GPUs simdgroup matrices run on
the ordinary FP32 ALUs, with no separate matrix unit. Level 3 at N = 2048
reaches 10.5 TFLOP/s, 74% of that peak, and the best f32 kernel reaches
86–100% of MPS (`MPSMatrixMultiplication`) from N = 2048 to 8192. All levels and MPS:
[benchmarks.md](../benchmarks.md#apple-m4-max-gpu--metal).
