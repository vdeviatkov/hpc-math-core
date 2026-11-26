# Apple M4 Max GPU: Metal kernels, MPS and FMA peak

Raw data behind [benchmarks.md § Apple M4 Max GPU — Metal](../../benchmarks.md#apple-m4-max-gpu--metal)
and the tile sweep in [docs/kernels/metal.md](../../kernels/metal.md).
Apple M4 Max, 40-core GPU, macOS 26.3.1, Apple Clang 17, on AC power,
automatic power mode. The kernels: [`src/metal/gemm.metal`](../../../src/metal/gemm.metal),
interface and level list: [`src/gemm/metal.hpp`](../../../src/gemm/metal.hpp).

| File | Program | What it measures |
|---|---|---|
| [`bench_gemm_metal.json`](bench_gemm_metal.json), [`load.txt`](load.txt) | `bench_gemm_metal --benchmark_filter='^Metal(Naive\|Tiled\|RegTile\|Simd\|SimdTg\|SimdTgF16\|Mps\|MpsF16\|FmaPeak)/' --benchmark_repetitions=5 --benchmark_min_time=1s --benchmark_enable_random_interleaving=true` | every level at its default tile, MPS and the FMA peak, N = 256 … 8192; load average 0.8–4. The table in benchmarks.md |
| [`variants.json`](variants.json), [`variants_load.txt`](variants_load.txt) | `bench_gemm_metal --benchmark_filter='^(MetalVariant\|MetalMaxThreads\|MetalFmaPeak)/'`, same repetitions | every Level 1–5 tile variant, including Level 3's simdgroup arrangements (1 × 1 to 4 × 4), N = 256 … 4096; load average 2–8. `MetalMaxThreads` records each pipeline's maxTotalThreadsPerThreadgroup: 1,024 for every function |
| [`variants_level1.json`](variants_level1.json), [`variants_level1_load.txt`](variants_level1_load.txt) | the Level 1 variants alone, interleaved | Level 1 at 8, 16 and 32 threads square |
| [`session_a.json`](session_a.json), [`session_b.json`](session_b.json), [`session_log.txt`](session_log.txt) | the levels without variants, median of 3, before and after the PyTorch MPS and JAX-on-Metal rows of [frameworks-m4](../frameworks-m4/README.md) | one bracketed GPU session for the cross-framework comparison only; it ran throttled (see below) and is not comparable with the other files |

## Method

GPU time of the GEMM alone: A, B and C sit in shared-storage `MTLBuffer`s
filled before timing (plus fp16 copies for the F16 rows), and one untimed
call compiles the pipeline. Each Google Benchmark iteration is one command
buffer of `reps` back-to-back dispatches, with `reps` chosen so it lasts at
least ~5 ms (93–256 at N=256, 1 at N ≥ 4096); its `GPUStartTime..GPUEndTime`
span is the iteration time, and the flop counter is `reps × 2N³`. The FMA
peak runs 2²⁰ threads × 16,384 iterations × 8 independent 4-wide FMA
chains (2⁴⁰ flops, ~75 ms), best of 10 after a warm-up run.

## Clock and power

Absolute GPU numbers on this laptop vary by up to ~3× between runs
(`MetalMps` f32 at N=4096: 13.1 TFLOP/s in `bench_gemm_metal.json`, as low
as 4.5 in other runs), while the register-only FMA peak stays at
14.2–14.9 TFLOP/s. That pattern (fast when cold, slower under a sustained
memory-heavy load, the low-power FMA loop unaffected) fits power limiting
rather than another process using the GPU: `ioreg` showed the GPU idle
between runs. No `sudo powermetrics` was available to read the GPU clock,
so this is an inference. Compare rows within one file, not across files.

## Unrolling

The tile sweep is only meaningful with the tile loops fully unrolled. Left
to itself, the Metal compiler keeps the fixed-trip loops over the
accumulator array rolled; the array is then indexed at run time and lives
in memory, which puts the 32 × 32-per-simdgroup kernels at 1.2–2.0
TFLOP/s. `HPC_UNROLL` (`clang loop unroll(full)`) in `gemm.metal` forces
the unrolling.
