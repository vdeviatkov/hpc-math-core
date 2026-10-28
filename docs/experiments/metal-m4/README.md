# Apple M4 Max GPU: Metal kernels, MPS and FMA peak

Raw data behind [benchmarks.md § Apple M4 Max GPU — Metal](../../benchmarks.md#apple-m4-max-gpu--metal).
Apple M4 Max, 40-core GPU, macOS 26.3.1, Apple Clang 17, 2026-10-24, on AC
power, automatic power mode. The kernels: [`src/metal/gemm.metal`](../../../src/metal/gemm.metal),
interface and level list: [`src/gemm/metal.hpp`](../../../src/gemm/metal.hpp).

| File | Program | What it measures |
|---|---|---|
| [`bench_gemm_metal.json`](bench_gemm_metal.json) | `bench_gemm_metal --benchmark_repetitions=5 --benchmark_min_time=1s --benchmark_enable_random_interleaving=true --benchmark_report_aggregates_only=true` | every level, MPS, every tile variant and the FMA peak; median, mean, stddev and cv of 5 interleaved repetitions. **The published dataset.** |
| [`load.txt`](load.txt) | `uptime` before and after | load averages 1.5–3.9 |
| [`session_a.json`](session_a.json), [`session_b.json`](session_b.json), [`session_log.txt`](session_log.txt) | the same without variants (`--benchmark_filter=-Variant`), median of 3, before and after the PyTorch MPS and JAX-on-Metal rows of [frameworks-m4](../frameworks-m4/README.md) | one bracketed GPU session, used for the cross-framework comparison; ran at 57–59% of the published run's speed (median over rows; see below) |

## Method

GPU time of the GEMM alone: A, B and C sit in shared-storage `MTLBuffer`s
filled before timing (plus fp16 copies for the F16 rows), and one untimed
call compiles the pipeline. Each Google Benchmark iteration is one command
buffer of `reps` back-to-back dispatches, with `reps` chosen so it lasts at
least ~5 ms (93–256 at N=256, 1 at N ≥ 4096); its `GPUStartTime..GPUEndTime`
span is the iteration time, and the flop counter is `reps × 2N³`. The FMA
peak runs 2²⁰ threads × 16,384 iterations × 8 independent 4-wide FMA
chains (2⁴⁰ flops, ~75 ms), best of 10 after a warm-up run.

## Results (bench_gemm_metal.json)

GFLOP/s, median of 5; the last column is the largest coefficient of
variation of the row.

| Kernel | N=256 | N=1024 | N=2048 | N=4096 | N=8192 | max cv |
|---|--:|--:|--:|--:|--:|--:|
| `MetalNaive` | 944 | 1,132 | 1,226 | 1,027 | 814 | 17% |
| `MetalTiled` | 1,327 | 1,605 | 1,615 | 1,598 | 1,860 | 13% |
| `MetalRegTile` | 1,248 | 5,923 | 6,494 | 6,298 | 6,469 | 12% |
| `MetalSimd` | 2,131 | 9,549 | 10,531 | 10,205 | 5,560 | 15% |
| `MetalSimdTg` | 1,364 | 8,846 | 10,717 | 9,104 | 9,373 | 14% |
| `MetalSimdTgF16` | 1,492 | 9,450 | 11,617 | 11,380 | 11,211 | 11% |
| `MetalMps` | 3,475 | 11,123 | 10,704 | 11,087 | 10,855 | 17% |
| `MetalMpsF16` | 4,974 | 10,868 | 11,479 | 13,732 | 12,671 | 12% |

FMA peak in the same run: f32 14,216 GFLOP/s (cv 5%), f16 11,967 (cv 14%).
Run alone, three times in a row, the f32 peak measured 14,858 each time and
f16 15,488–15,785: 40 cores × 128 FP32 lanes × 2 flops at about 1.45 GHz,
and fp16 at the same rate as fp32, so Levels 3–5 have no faster matrix
datapath to reach for on this GPU.

Tile variants (`MetalVariant/<function>`), GFLOP/s at N=2048 / 4096; the
defaults are the plain names:

| Level | Variant | N=2048 | N=4096 |
|---|---|--:|--:|
| 2 | `gemm_reg_tile` (64×64, 8×4 per thread) | **6,726** | **6,075** |
| 2 | `gemm_reg_tile_64x64_4x4` | 5,792 | 4,852 |
| 2 | `gemm_reg_tile_128x64_8x8` | 6,207 | 4,718 |
| 2 | `gemm_reg_tile_128x128_8x8` | 5,029 | 5,119 |
| 3 | `gemm_simd` (2×2 simdgroups of 32×32) | 9,850 | **10,391** |
| 3 | `gemm_simd_2x2_16x32` | **9,929** | 8,432 |
| 3 | `gemm_simd_2x2_32x16` | 8,590 | 7,732 |
| 4 | `gemm_simd_tg` (k step 16) | 9,526 | 9,522 |
| 4 | `gemm_simd_tg_2x2_32x32_k32` | 9,333 | 8,374 |
| 4 | `gemm_simd_tg_2x2_64x32_k16` | 9,399 | **9,564** |
| 4 | `gemm_simd_tg_2x4_32x32_k16` | 8,749 | 9,251 |
| 4 | `gemm_simd_tg_4x2_32x32_k16` | **9,680** | 8,520 |
| 5 | `gemm_simd_tg_f16` | 10,762 | **13,118** |
| 5 | `gemm_simd_tg_f16_2x2_32x32_k32` | **11,818** | 10,845 |
| 5 | `gemm_simd_tg_f16_2x2_64x32_k16` | 11,186 | 11,690 |
| 5 | `gemm_simd_tg_f16_2x4_32x32_k16` | 11,625 | 11,440 |

Bold: the best of each level at each size. In Levels 3–5 no variant beats
its default at both sizes, and the gaps are mostly within this run's noise
(cv up to 18%), so the defaults stay. The Level 2 default was changed from
4×4 to 8×4 outputs per thread after an earlier single pass (7,477 against
6,445 GFLOP/s at N=4096), which this run confirms at both sizes.

## Two things found on the way

- **Unrolling.** The first build of Levels 3–5 measured 1.2–2.0 TFLOP/s for
  the 32×32-per-simdgroup kernels, while the 16×32 variant reached 11.8 and
  the fp16 copy of the same code 12.4. The fixed-trip loops over the
  accumulator array were not unrolled, so the array was indexed dynamically
  and lived in memory. `clang loop unroll(full)` on those loops brought them
  to 11–13 TFLOP/s in the same conditions.
- **Clock and power.** Absolute GPU numbers on this laptop moved by up to
  ~3× between sessions. A first quick pass had `MetalMps` f32 at 13.5 TFLOP/s
  at N=4096; the published run 11.1; the bracketed session 6.6–7.3; a run a
  few minutes later 4.5–5.0, steady over 12 repetitions, while the
  register-only FMA peak still reached 14.7. That pattern (fast when cold,
  slower under a sustained memory-heavy load, the low-power FMA loop
  unaffected) fits power limiting rather than another process using the GPU:
  `ioreg` showed the GPU idle between runs. During those later runs CLion was
  indexing (230% CPU) and an XProtect scan ran (90%). No `sudo powermetrics`
  was available to read the GPU clock, so this is an inference. Compare rows
  within one run, not across runs.
