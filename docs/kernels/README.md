# Tiles and blocks of the kernels

[Why each default was chosen, including evidence and untuned sizes](sizing.md).

`gemm_simd_blocked<Ymm>`, `gemm_simd_blocked<Zmm>` and `gemm_simd_blocked<Q>` share
one driver ([`simd.hpp`](../../src/gemm/simd.hpp)) and
take their register tile, R rows × V vectors of C, as template parameters:
`gemm_simd_blocked<Ops, R, V, …>`. They have no defaults; "default" on these
pages means the measured value the benchmarks use
([`benchmarks/simd_params.hpp`](../../benchmarks/simd_params.hpp),
[`benchmarks/sme_params.hpp`](../../benchmarks/sme_params.hpp)). `bench_gemm` runs each family with every
listed shape (`<Isa>Tile<R>x<V>`), and these pages compare them and explain
the measured choices:

| Family | Page | Default tile | Machine | Best measured, FP32 / FP64 GFLOP/s | Share of the core's peak |
|---|---|---|---|--:|--:|
| AVX2 | [avx2.md](avx2.md) | 6 × 2 (6 × 16 FP32) | AMD Zen 5 | 171.7 / 82.8 | 94% / 91% |
| AVX-512 | [avx512.md](avx512.md) | 8 × 3 (8 × 48 FP32) | AMD Zen 5 | 327.9 / 162.0 | 91% / 90% |
| NEON | [neon.md](neon.md) | 6 × 4 (6 × 16 FP32) | Apple M4 Max | 123.8 / 62.2 | 86% / 87% |
| SME2 | [sme.md](sme.md) | 1×4 ZA tiles (16 × 64 FP32, 8 × 32 FP64) | Apple M4 Max, one SME unit | 1,602.9 / 429.5 | 82% / 86% |

The cache blocks around the register tile are template parameters too
(`gemm_simd_blocked<Ops, R, V, TileM, TileK, TileN>`), swept in
[blocks.md](blocks.md): taller blocks of A and a B block that fits in half
of Zen 5's L2 make the kernels 9–50% faster at N = 4096 than 64-row
blocks.

Best measured is `gemm_simd_blocked` at its measured tile and its best size
(N = 64–512) with its default cache block, from the runs in
[benchmarks.md](../benchmarks.md), or `gemm_sme` with its measured ZA shape
and blocks at its best size (N = 4096), from the shape re-measurement in
[sme.md](sme.md). Peaks
are `fma_peak`'s measured FMA rates at the clock of each run, and for SME2
`sme_fmopa`'s 1,963 / 500 GFLOP/s for one SME unit
([sme2-machine](../experiments/sme2-machine/README.md)). Raw data and
method: [experiments/tile-sweeps](../experiments/tile-sweeps/README.md),
[experiments/block-sweeps](../experiments/block-sweeps/README.md),
[experiments/sme2-gemm](../experiments/sme2-gemm/README.md).

## GPU kernels

The GPU kernels have block tiles per thread block and tiles per thread,
simdgroup or warp instead of R × V register tiles. Their pages list every
level's sizes and the sweeps behind them:

| Family | Page | Default tiles | Machine | Swept |
|---|---|---|---|---|
| Metal | [metal.md](metal.md) | Level 1: 32 × 32 threads; Level 2: 64 × 64, 8 × 4 per thread; Levels 3–5: 2 × 2 simdgroups of 32 × 32 | Apple M4 Max GPU | every level, N = 256 … 4096 |
| CUDA | [cuda.md](cuda.md) | Level 2: 128 × 128, BK 32, 8 × 8 per thread; Level 4: 16 × 8; Levels 5–6: 2 × 8 warps; Level 7: 64 × 32 warps; Level 8: 128 × 256 or 128 × 128 | NVIDIA RTX 5080 | every level, FP32 and FP64, N = 1024 … 8192 |

## CPU pages

The measured AVX2, AVX-512 and NEON pages have, per shape and size, the full-multiplication throughput,
the registers it needs, and why the default wins: the accumulator count
the FMA latency asks for, the B traffic each row of the tile saves, the
register budget, and what the compiler makes of the loop.
