# Cache-block sweeps

Raw data behind [docs/kernels/blocks.md](../../kernels/blocks.md): the
AVX2, AVX-512 and NEON blocked kernels at their default register tiles
(6 × 2, 8 × 3, 6 × 4) with every cache block TileM × TileK × TileN in
`bench_gemm`'s `add_blocks()` grid, and the library kernels re-measured
with the blocks chosen from it. 2026-10-21/22, one thread.

Machines, compilers and flags as in [tile-sweeps](../tile-sweeps/README.md):
Zen 5 with GCC 13.3, `-O3 -march=native`, pinned to core 2, idle (load
average ≤ 1.3); M4 Max with Apple Clang 17, load average 2–10 from other
work on 16 cores (`load_*.txt`).

| File | Run | What |
|---|---|---|
| [`blocks_zen5_stage1.csv`](blocks_zen5_stage1.csv), [`blocks_m4_stage1.csv`](blocks_m4_stage1.csv) | `bench_gemm --benchmark_filter='^(Avx2Block\|Avx512Block)[0-9]'` (Zen 5), `'^NeonBlock[0-9]'` (M4), one run each | the grid with TileM ∈ {48, 64, 96, 192, 384} (AVX-512: {32, 64, 128, 256, 512}) × TileK, TileN ∈ {128, 256, 512, 1024}; N = 512, 1024, 4096; FP32 and FP64 |
| [`blocks_zen5_stage1_ext.csv`](blocks_zen5_stage1_ext.csv), [`blocks_m4_stage1_ext.csv`](blocks_m4_stage1_ext.csv) | same, filtered to the larger TileM | TileM ∈ {768, 1536} (AVX-512: {1024, 2048}), added when the best TileM was at the edge of the first grid |
| [`blocks_zen5_stage2.csv`](blocks_zen5_stage2.csv), [`blocks_m4_stage2.csv`](blocks_m4_stage2.csv) | the best candidates and the old default, `--benchmark_repetitions=3 --benchmark_enable_random_interleaving=true` | stage 2 of the doc |
| [`blocks_m4_slices.csv`](blocks_m4_slices.csv) | `^NeonBlock((48\|…\|1536)x256x256\|384x(128\|512\|1024)x256\|384x256x(128\|512\|1024))/`, 3 runs interleaved | the M4 tables that vary one block size around the default; the one-run M4 grid is too noisy for single cells (up to 2× on a few) |
| [`bench_zen5_defaults.csv`](bench_zen5_defaults.csv), [`lowp_zen5_defaults.csv`](lowp_zen5_defaults.csv) | `bench_gemm --benchmark_filter='^(Avx2Blocked\|Avx512Blocked)'`, `bench_gemm_lowp --benchmark_filter='^Avx512Blocked'`, 3 runs | the library kernels with the new default blocks, and their prefetch variants |
| [`bench_m4_defaults.csv`](bench_m4_defaults.csv), [`lowp_m4_defaults.csv`](lowp_m4_defaults.csv) | `bench_gemm --benchmark_filter='^NeonBlocked'`, `bench_gemm_lowp --benchmark_filter='^Neon'`, 3 runs | the same on the M4; the low-precision run includes every NEON kernel, one session |
| `load_zen5.txt`, `load_m4.txt`, `load_*_defaults.txt` | `uptime` | load before and after each run |

Coefficient of variation of the time: ≤ 2.7% in every Zen 5 stage-2 cell
and ≤ 0.6% in the Zen 5 default runs; ≤ 4.3% in M4 stage 2 and ≤ 1.5% in
the M4 default runs; in the M4 slices 11% in one cell (TileK 1024, FP64,
N = 1024) and ≤ 3.7% in the rest.
