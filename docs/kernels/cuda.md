# CUDA tiles (Levels 0–8)

The CUDA kernels on an RTX 5080 (sm_120, 84 SMs) choose a block tile of C
per thread block and, inside it, a tile per thread (Levels 2–4) or per
warp (Levels 5–8). This page lists every level's sizes and says how each
was chosen: **swept** (alternatives measured), **measured** (one change
measured against the version before it), **forced** (set by a hardware
limit) or **design** (chosen by reasoning, never compared with another
size). The reasoning and the profiles behind each level are in the
article, [*Matrix multiplication on an RTX 5080*](../articles/cuda-gemm.md);
this page does not repeat them. Code: [`gemm.cu`](../../src/cuda/gemm.cu),
level list [`cuda.hpp`](../../src/gemm/cuda.hpp). The Metal counterpart:
[metal.md](metal.md).

Unlike the CPU kernels, the CUDA sizes are `constexpr` constants in
`gemm.cu`, not parameters the benchmarks can sweep (Level 8's two tiles
are two instantiations of one template).

## Every level

| Level | Kernel | Block tile of C | Per thread / warp | Threads | How chosen |
|--:|---|---|---|--:|---|
| 0 | `kernel_naive` | 16 × 16 | 1 output per thread | 256 | design |
| 1 | `kernel_blocked` | 16 × 16, 16 × 16 tiles of A and B | 1 output per thread | 256 | design |
| 2 | `kernel_reg_tile` | 128 × 128, BK = 16 | 8 × 8 per thread | 256 | design |
| 3 | `kernel_double_buf` | 128 × 128 FP32, **64 × 64 FP64** | 8 × 8 per thread | 256 / 64 | FP64 tile forced; FP32 launch bound measured |
| 4 | `kernel_vectorized` | 128 × 128 | 8 × 8 per thread, read as `float4` | 256 | read width measured |
| 5 | `kernel_wmma` | 64 × 64 | 16 × 16 per warp, 4 × 4 warps | 512 | design |
| 6 | `kernel_mma_ldmatrix` | 64 × 64 | 16 × 16 per warp | 512 | design |
| 7 | `kernel_wmma_pipelined` | 128 × 128, BK = 32 | 32 × 64 per warp (2 × 4 fragments) | 256 | padding measured; tile design |
| 8 | `kernel_tma` | **128 × 256** (4 stages) or 128 × 128 (6 stages) | 64 × 64 per consumer warp | 288 / 160 | **swept** |
| lowp | `kernel_lowp` | 128 × 128, 2 stages | 64 × 32 per warp, 2 × 4 warps | 256 | **swept** |

## What was measured

- **Level 2: the 8 × 8 register tile, 128 × 128 block.** Not compared
  with other sizes. 8 × 8 gives 64 FMAs per 16 shared-memory reads and
  needs 128 registers per thread, two blocks per SM. It is the largest
  step of the ladder (6.4× Level 0), but whether 8 × 4 or 16 × 8 would do
  better on this GPU was not tested. What *was* measured is how the tiles
  are loaded: coalesced loads, transposed into shared memory, made Level 2
  27% faster and Level 3 36%.
- **Level 3: FP64 uses 64 × 64 because it must.** Two buffers of a
  128 × 128 FP64 tile need 66,048 bytes of shared memory, over the 48 KiB
  static limit; 64 × 64 needs 33,280. For FP32,
  `__launch_bounds__(256, 2)` keeps the kernel at 128 registers so two
  blocks fit per SM; one register more would allow only one block, which
  measured ~4% slower at N = 4096.
- **Level 4: 128-bit reads halve the registers.** Reading a thread's 8 A
  and 8 B values as four `float4` loads instead of sixteen scalar ones
  took the kernel from 255 to 128 registers, from one block per SM to two,
  and from 13.1 to 26.9 TFLOP/s at N = 4096.
- **Levels 5–6: 64 × 64 blocks, one fragment per warp.** Not compared with
  other sizes; the article shows why they are slow (88% and 80% of
  shared-memory loads are bank conflicts, Tensor pipe 12–14% busy) and
  Level 7 changes the structure rather than the size.
- **Level 7: padding, not the tile.** Padding each shared-memory row by 8
  halves took bank conflicts from 85% to 1% of wavefronts and the kernel
  from 80.8 to 100.5 TFLOP/s at N = 4096 (+24%, Nsight Compute timing).
  The 128 × 128 block and
  32 × 64 warp tile were not compared with other sizes.
- **Level 8: 128 × 256 against 128 × 128, and the stage count.** Swept at
  N = 2048 and 8192, all five formats
  ([`tuning_level8.csv`](../experiments/lowp/tuning_level8.csv)). FP16,
  TFLOP/s:

  | Tile, stages | N = 2048 | N = 8192 |
  |---|--:|--:|
  | 128 × 128, 4 | 84.3 | 114.5 |
  | 128 × 128, 5 | 83.7 | 113.9 |
  | 128 × 128, 6 | 83.8 | 114.4 |
  | 128 × 256, 3 | **87.9** | **117.9** |
  | 128 × 256, 4 | **87.9** | 117.8 |

  128 × 256 is 3–4% faster in FP16 (3–11% across the formats); the stage
  count changes little from 3 on. Both shipped tiles fill 96 KiB of shared
  memory, so one block runs per SM, and the host picks the tile per launch
  by how full the last wave of blocks is (`launch_tma`): 128 × 256 unless
  128 × 128 fills the waves more than 5% better. On 84 SMs that gives
  128 × 128 at N = 1024 and 4096, 128 × 256 at 2048 and from 8192 up.
- **Low-precision cp.async kernel: stages and block order.** Swept at
  N = 4096, 8192 and 16384, all formats
  ([`tuning_cp_async.csv`](../experiments/lowp/tuning_cp_async.csv)):
  2 stages beat 3 and 4 by 8–18% (two 40 KiB blocks per SM), blocks
  grouped by 8 rows of tiles beat row-major order (INT8 at N = 16384: 181
  → 308 TOP/s), and a 128 × 256 tile was not faster. Details:
  [experiments/lowp](../experiments/lowp/README.md).

## How far the kernels get

At N = 8192: Level 4 (FMA cores) 27.5 TFLOP/s, 71% of cuBLAS SGEMM;
Level 8 (FP16 Tensor Cores) 117.3 TFLOP/s, 99% of cuBLAS FP16 and 94% of
the measured `mma.sync` peak. All levels, sizes and counters:
[the article](../articles/cuda-gemm.md) and
[experiments/cuda-machine](../experiments/cuda-machine/README.md).
