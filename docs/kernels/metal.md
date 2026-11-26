# Metal tiles (`gemm_reg_tile`, `gemm_simd`, `gemm_simd_tg`)

The Metal kernels on the Apple M4 Max GPU (40 cores) have two kinds of
tile beyond Level 1's threadgroup-memory tile (32 × 32 threads, one output
each). Level 2 gives each thread a TM × TN block of C in registers, as the
CPU kernels do. Levels 3–5 give each simdgroup (32 threads) an SM × SN
block of C as 8 × 8 `simdgroup_float8x8` accumulator fragments. Every
size is a template parameter of the kernel, and every variant is compiled
under its own name. This page gives the sweep behind the defaults. Code:
[`gemm.metal`](../../src/metal/gemm.metal), host and level list
[`metal.hpp`](../../src/gemm/metal.hpp). The CUDA counterpart:
[cuda.md](cuda.md).

Two sizes are fixed by Metal, not chosen: `kVec = 4` (`float4` is the
widest vector type, so tile edges that loads walk along are multiples of 4) and
`kFrag = 8` (`simdgroup_matrix<T, 8, 8>` is the only fragment
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

`bench_gemm_metal`'s `MetalVariant/<function>` benchmarks, GFLOP/s, median
of 5 randomly interleaved repetitions
([data](../experiments/metal-m4/variants.json)). The default of each
level is the plain name; best of each level per column in bold.

| Level | Variant | Tile of C per threadgroup | N = 256 | N = 1024 | N = 2048 | N = 4096 |
|--:|---|---|--:|--:|--:|--:|
| 1 | `gemm_tiled` (default: 32 × 32 threads) | 32 × 32 | **1,521** | **2,337** | **2,255** | **2,344** |
| 1 | `gemm_tiled_16` | 16 × 16 | 1,456 | 2,025 | 2,112 | 2,140 |
| 1 | `gemm_tiled_8` | 8 × 8 | 1,377 | 2,125 | 2,132 | 1,936 |
| 2 | `gemm_reg_tile` (default: 64 × 64, 8 × 4 per thread) | 64 × 64 | 1,239 | **6,373** | **7,176** | **7,485** |
| 2 | `gemm_reg_tile_64x64_4x4` | 64 × 64 | **1,603** | 5,841 | 6,366 | 6,677 |
| 2 | `gemm_reg_tile_128x64_8x8` | 128 × 64 | 558 | 5,069 | 6,303 | 6,455 |
| 2 | `gemm_reg_tile_128x128_8x8` | 128 × 128 | 492 | 5,189 | 5,493 | 6,422 |
| 3 | `gemm_simd` (default: 2 × 2 simdgroups of 32 × 32) | 64 × 64 | 2,068 | 10,551 | **12,106** | **12,424** |
| 3 | `gemm_simd_1x1_32x32` | 32 × 32 | 2,083 | **11,222** | 11,590 | 11,893 |
| 3 | `gemm_simd_1x4_32x32` | 32 × 128 | 2,042 | 10,752 | 11,671 | 11,939 |
| 3 | `gemm_simd_4x1_32x32` | 128 × 32 | 2,061 | 10,522 | 11,835 | 11,655 |
| 3 | `gemm_simd_2x4_32x32` | 64 × 128 | 1,825 | 9,725 | 11,694 | 11,771 |
| 3 | `gemm_simd_4x2_32x32` | 128 × 64 | 1,773 | 9,449 | 11,827 | 12,057 |
| 3 | `gemm_simd_4x4_32x32` | 128 × 128 | 1,041 | 9,706 | 11,193 | 11,622 |
| 3 | `gemm_simd_2x2_16x32` | 32 × 64 | **3,207** | 10,909 | 11,282 | 10,719 |
| 3 | `gemm_simd_2x2_32x16` | 64 × 32 | 3,114 | 10,142 | 9,803 | 9,762 |
| 4 | `gemm_simd_tg` (default: 2 × 2 of 32 × 32, BK = 16) | 64 × 64 | 1,486 | 9,616 | **12,215** | 10,598 |
| 4 | `gemm_simd_tg_2x2_32x32_k32` | 64 × 64 | **1,594** | **9,863** | 9,456 | 9,824 |
| 4 | `gemm_simd_tg_2x2_64x32_k16` | 128 × 64 | 993 | 8,936 | 11,256 | **11,448** |
| 4 | `gemm_simd_tg_2x4_32x32_k16` | 64 × 128 | 1,267 | 8,605 | 11,003 | 11,280 |
| 4 | `gemm_simd_tg_4x2_32x32_k16` | 128 × 64 | 1,271 | 8,750 | 11,537 | 10,790 |
| 5 | `gemm_simd_tg_f16` (default, fp16 in) | 64 × 64 | 1,615 | 10,320 | 12,756 | 13,153 |
| 5 | `gemm_simd_tg_f16_2x2_32x32_k32` | 64 × 64 | **1,674** | **10,594** | **13,052** | 13,152 |
| 5 | `gemm_simd_tg_f16_2x2_64x32_k16` | 128 × 64 | 1,092 | 9,496 | 12,659 | **13,580** |
| 5 | `gemm_simd_tg_f16_2x4_32x32_k16` | 64 × 128 | 1,369 | 9,379 | 12,762 | 12,097 |

Level 1 is a separate interleaved run of its three sizes
([variants_level1.json](../experiments/metal-m4/variants_level1.json)).
Variation: coefficient of variation up to 14% per row; load average 2–8
during the run. Absolute GPU rates on this laptop vary by up to ~3×
between runs ([metal-m4](../experiments/metal-m4/README.md#clock-and-power)),
so compare rows of one run, not runs.
The sweep is only valid with the tile loops fully unrolled: the Metal
compiler leaves them rolled on its own, which puts the 32 × 32 kernels at
1.2–2.0 TFLOP/s and makes 16 × 32 look 6–10× faster; `HPC_UNROLL`
(`clang loop unroll(full)`) forces the unrolling
([details](../experiments/metal-m4/README.md#unrolling)).

## Why these defaults

- **Level 1: 32 × 32.** Fastest at every size, 4–15% ahead of 16 × 16.
- **Level 2: 8 × 4 per thread.** Fastest from N = 1024 up: 9–13% ahead of
  4 × 4 and 14–31% ahead of the two 8 × 8 variants. At N = 256 the 64 × 64
  tile gives only 16 threadgroups for 40 GPU cores, and 4 × 4 (256 threads
  per threadgroup instead of 128) leads. Every pipeline reports the device
  maximum of 1,024 threads per threadgroup (`MetalMaxThreads`), so the
  8 × 8 variants are not limited by threads per threadgroup; Metal exposes
  no register count or occupancy counter to show what else limits them.
- **Levels 3–5: 32 × 32 per simdgroup, 2 × 2 simdgroups, BK = 16.** At
  N = 2048 and 4096, where the GPU is full, the default is the best or
  within 2% of it in Level 3, and within the run's variation in Levels 4–5
  (the leaders, 128 × 64 tiles at N = 4096, are 3–8% ahead and fall
  behind at N ≤ 2048). At N ≤ 1024 smaller threadgroup tiles lead (Level 3
  16 × 32: 3,207 against 2,068 GFLOP/s at N = 256), because a 64 × 64 tile
  leaves most of the 40 cores idle there (16 threadgroups at N = 256).

## How far the kernels get

Measured FMA peak in the same run: 14.2 TFLOP/s FP32 (14.9 run alone).
fp16 FMA runs at the same rate: on M1–M4 GPUs simdgroup matrices run on
the ordinary FP32 ALUs, with no separate matrix unit. Level 3 at N = 2048
reaches 10.5 TFLOP/s, 74% of that peak, and the best f32 kernel reaches
86–100% of MPS (`MPSMatrixMultiplication`) from N = 2048 to 8192. All levels and MPS:
[benchmarks.md](../benchmarks.md#apple-m4-max-gpu--metal).
