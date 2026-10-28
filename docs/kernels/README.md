# Register tiles of the SIMD and SME kernels

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
| SME2 | [sme.md](sme.md) | 2×2 FP32 / 2×4 FP64 ZA tiles | Apple M4 Max | See SME measurements | — |

The cache blocks around the register tile are template parameters too
(`gemm_simd_blocked<Ops, R, V, TileM, TileK, TileN>`), swept in
[blocks.md](blocks.md): taller blocks of A and a B block that fits in half
of Zen 5's L2 made the kernels 10–51% faster at N = 4096.

Best measured is `gemm_simd_blocked` at its measured tile and its best size
(N = 64–512) with its default cache block, from the runs in
[benchmarks.md](../benchmarks.md); peaks
are `fma_peak`'s measured FMA rates at the clock of each run. Raw data and
method: [experiments/tile-sweeps](../experiments/tile-sweeps/README.md),
[experiments/block-sweeps](../experiments/block-sweeps/README.md).

The measured AVX2, AVX-512 and NEON pages have, per shape and size, the full-multiplication throughput,
the registers it needs, and why the default wins: the accumulator count
the FMA latency asks for, the B traffic each row of the tile saves, the
register budget, and what the compiler makes of the loop.
