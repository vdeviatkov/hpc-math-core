# CUDA tiles (Levels 0–8)

The CUDA kernels on an RTX 5080 (sm_120, 84 SMs) choose a block tile of C
per thread block and, inside it, a tile per thread (Levels 2–4) or per
warp (Levels 5–8). Every size is a member of a parameter struct the
kernels take as a template argument (`gemm/cuda.hpp`, "Tile
parameters"); the measured values are `hpc::bench::CudaParams` in
[`benchmarks/cuda_params.hpp`](../../benchmarks/cuda_params.hpp), and the
swept variants are in
[`benchmarks/cuda_variants.hpp`](../../benchmarks/cuda_variants.hpp). The
only fixed sizes are those an API or the hardware sets: the 16 × 16 × 16
WMMA fragment, the 64-byte rows of the TMA swizzle, the 128-bit vector
width. The reasoning and profiles behind each level are in the article,
[*Matrix multiplication on an RTX 5080*](../articles/cuda-gemm.md). Code:
[`gemm_kernels.cuh`](../../src/cuda/gemm_kernels.cuh). The Metal
counterpart: [metal.md](metal.md).

## The measured parameters

| Level | Kernel | FP32 | FP64 |
|--:|---|---|---|
| 0–1 | `kernel_naive`, `kernel_blocked` | 16 × 16 threads | same |
| 2 | `kernel_reg_tile` | 128 × 128, BK 32, 8 × 8 per thread | 64 × 64, BK 16, 4 × 4 |
| 3 | `kernel_double_buf` | 128 × 128, BK 16, 8 × 8, 2 blocks per SM | 64 × 64, BK 16, 4 × 4, 2 blocks |
| 4 | `kernel_vectorized` | 128 × 128, BK 16, 16 × 8 per thread | 64 × 64, BK 16, 4 × 4 |
| 5–6 | `kernel_wmma`, `kernel_mma_ldmatrix` | 2 × 8 warps of 16 × 16 (32 × 128), 3 blocks per SM | — |
| 7 | `kernel_wmma_pipelined` | 128 × 128, BK 32, 64 × 32 per warp | — |
| lowp | `kernel_lowp` | 128 × 128, 2 × 4 warps of 64 × 32, 2 stages, groups of 8 | — |
| 8 | `kernel_tma` | 128 × 256 (4 stages) or 128 × 128 (6), chosen per launch; groups of 8 | — |

## Sweeps

`bench_gemm_cuda` / `bench_gemm_lowp_cuda` `*Var/` rows:
median of 3 randomly interleaved repetitions, GPU event time, idle host,
coefficient of variation ≤ 0.9%
([data](../experiments/cuda-machine/tile-sweeps/)). Every variant was
checked: FP32 FMA kernels within 3 × 10⁻⁵ relative of cuBLAS SGEMM, FP64
identical to cuBLAS DGEMM, Level 7 identical to cuBLAS FP16, WMMA levels
within fp16 rounding (1.2%), low-precision rows by the sampled `err`
counter. TFLOP/s; the chosen value in bold.

**Levels 0–1** (FP32, N = 1024 / 4096 / 8192): 16 × 16 threads is the
fastest of 8, 16, 32 at every size (Level 0: 3.62 / 3.67 / 3.25; Level 1:
3.32 / 3.39 / 3.38; 32 × 32 is 7–23% slower, 8 × 8 18–62%).

**Level 2** (FP32):

| Block, k step, thread tile | N = 1024 | 4096 | 8192 |
|---|--:|--:|--:|
| 128 × 128, BK 16, 8 × 8 | 14.33 | 23.63 | 24.76 |
| **128 × 128, BK 32, 8 × 8** | **14.91** | **24.29** | **25.09** |
| 128 × 128, BK 16, 8 × 4 | 16.01 | 23.60 | 23.27 |
| 128 × 128, BK 8, 8 × 8 | 13.19 | 21.64 | 23.26 |
| 128 × 128, BK 16, 16 × 8 | 11.21 | 15.69 | 16.38 |
| 128 × 128, BK 16, 4 × 8 | 11.22 | 16.62 | 17.09 |
| 64 × 64, BK 16, 4 × 4 | 13.52 | 17.80 | 17.55 |
| 64 × 64, BK 16, 8 × 8 | 11.85 | 13.56 | 14.41 |

**Level 3** (FP32): 128 × 128 / 8 × 8 with two blocks per SM is the
fastest of six at N = 4096 and 8192 (26.51, 27.03); one block per SM is
4–7% slower there; 8 × 4 with one block leads at N = 1024 (19.25 against
17.92).

**Level 4** (FP32):

| Block, k step, thread tile | N = 1024 | 4096 | 8192 |
|---|--:|--:|--:|
| 128 × 128, BK 16, 8 × 8 | 16.35 | 26.82 | 27.47 |
| **128 × 128, BK 16, 16 × 8** | **16.42** | **28.46** | **29.81** |
| 128 × 128, BK 16, 8 × 4 | 19.63 | 20.48 | 21.34 |
| 128 × 128, BK 32, 8 × 8 | 19.60 | 21.92 | 23.01 |
| 64 × 64, BK 16, 8 × 8 | 17.70 | 25.12 | 26.08 |

BK 32 is the best for Level 2 and 18% slower for Level 4, so the two
levels have separate parameters.

**Levels 2–4, FP64** (N = 1024 / 4096, GFLOP/s): 64 × 64 with 4 × 4 per
thread is the fastest or tied in each level: Level 2 729 / 961 against
722 / 874 for 128 × 128 / 8 × 8 (+10% at 4096), Level 3 736 / 965 against
725 / 944, Level 4 728 / 951 against 726 / 889 (+7%).

**Levels 5 and 6** (warps per block, each warp one 16 × 16 output):

| Warp grid (tile of C) | L5 N = 1024 | 4096 | 8192 | L6 N = 1024 | 4096 | 8192 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 4 (64 × 64) | 10.69 | 14.48 | 14.09 | 15.08 | 16.91 | 17.20 |
| **2 × 8 (32 × 128)** | **14.80** | **22.21** | **21.18** | **16.35** | **18.63** | **17.81** |
| 2 × 2 (32 × 32) | 17.36 | 20.43 | 18.37 | 19.98 | 18.03 | 15.57 |
| 2 × 4 (32 × 64) | 15.02 | 21.04 | 18.60 | 16.77 | 15.52 | 13.95 |
| 4 × 2 (64 × 32) | 11.22 | 14.54 | 14.28 | 19.33 | 17.73 | 16.22 |
| 8 × 2 (128 × 32) | 10.33 | 14.97 | 14.91 | 14.42 | 13.59 | 13.96 |
| 4 × 8 (64 × 128) | 10.99 | 11.35 | 10.60 | 14.66 | 13.86 | 12.89 |
| 8 × 4 (128 × 64) | 10.64 | 10.68 | 10.20 | 14.47 | 12.94 | 12.05 |

2 × 8 is the fastest at N ≥ 4096 in both levels and faster than 4 × 4 at
every size (Level 5 +38% to +53%, Level 6 +4% to +10%). At N = 1024 the
2 × 2 tile leads (more blocks for 84 SMs).

**Level 7** (block tile, k step, warp tile):

| Variant | N = 1024 | 4096 | 8192 |
|---|--:|--:|--:|
| 128 × 128, BK 32, warp 32 × 64 | 63.58 | 100.90 | 104.86 |
| **128 × 128, BK 32, warp 64 × 32** | **63.28** | **104.94** | **107.71** |
| 128 × 128, BK 32, warp 64 × 64 | 59.93 | 105.63 | 109.78 |
| 128 × 128, BK 32, warp 32 × 32 | 63.60 | 89.94 | 93.44 |
| 128 × 128, BK 16, warp 32 × 64 | 53.73 | 98.44 | 97.76 |
| 64 × 128, BK 32, warp 32 × 64 | 76.63 | 108.92 | 91.64 |
| 128 × 64, BK 32, warp 32 × 32 | 67.85 | 107.01 | 93.77 |
| 64 × 64, BK 32, warp 32 × 32 | 72.32 | 87.91 | 89.42 |

64 × 32 warps are within 0.5% of 32 × 64 warps at N = 1024 and 3–4%
faster from 4096 up; 64 × 64 warps are 2% faster still at N ≥ 4096 but 6%
slower at 1024; the smaller block tiles lead at N ≤ 4096 and fall behind
at 8192. Padding the shared-memory rows by 8 halves takes bank conflicts
from 85% to 1% of wavefronts and the kernel from 80.8 to 100.5 TFLOP/s at
N = 4096 (Nsight Compute timing, 32 × 64 warp tiles).

**Low-precision cp.async kernel** (7 variants × 5 formats,
N = 1024 … 8192): the default is within 1% of the best variant from
N = 4096 up in every format; group heights of 4 and 16 change nothing
beyond 1.5%. At N = 1024 32 × 32 warp tiles are 3–10% faster. 128 × 256
tiles lead only at N = 8192 for the 8-bit formats (+1–2%) and are 40–45%
slower at 1024. Stages were swept separately
([`tuning_cp_async.csv`](../experiments/lowp/tuning_cp_async.csv)): 2 are
8–18% faster than 3 or 4.

**Level 8** (6 variants × 5 formats): the tile the wave rule picks is
within 1% of the best at N = 1024, 2048 and 8192 in every format. At
N = 4096 it picks the narrow tile (512 wide blocks would fill 6.1 waves of
84 SMs): right for bf16 and fp16 (+2% over the wide tile), but the wide
tile is 3.5–4.8% faster for INT8, FP8 and fp16-accumulate there. Block rows
of 64 are slower at every size (up to 44%); group heights of 4 and 16 are
within 1.5%. Tiles and stages were swept separately
([`tuning_level8.csv`](../experiments/lowp/tuning_level8.csv)).

## How far the kernels get

With the measured parameters, at N = 8192 (TFLOP/s): Level 2 24.94,
Level 3 27.20, Level 4 29.78 (77% of cuBLAS SGEMM's 38.7), Level 5 21.18,
Level 6 17.87, Level 7 107.88, Level 8 117.60 (99% of cuBLAS FP16's
118.3). FP64 at N = 4096: 0.95–0.97 TFLOP/s for Levels 2–4
([data](../experiments/cuda-machine/bench_gemm_cuda.json)).
