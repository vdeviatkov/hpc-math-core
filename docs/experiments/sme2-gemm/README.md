# SME2 GEMM: the kernel ladder, block sizes and the head-to-head comparison

Raw data behind the step tables, the block-size section and "Where it
stands" in
[*Matrix multiplication on Apple M4's SME2 unit*](../../articles/sme2-gemm.md).
Apple M4 Max, one thread, Apple Clang 17,
`-O3 -mcpu=apple-m4 -ffast-math -funroll-loops`, 2026-10-12. The machine's
caches, TLBs and FMOPA rates are in [sme2-machine](../sme2-machine/README.md).

| File | Program | What it measures |
|---|---|---|
| [`ladder.csv`](ladder.csv) | [`bench_gemm`](../../../benchmarks/bench_gemm.cpp) `--benchmark_filter=Sme` | the four SME steps (`Sme1Tile`, `SmeTiles`, `SmeBlocked`, `Sme`), N = 256 … 4096, FP32 and FP64: three runs and their median |
| [`blocks_old_vs_new.csv`](blocks_old_vs_new.csv) | `bench_gemm --benchmark_filter=^Sme/` with the first and the current block sizes | `gemm_sme` with Mc = 128, Kc = 1024 against Mc = 1024 (FP32) / 512 (FP64), Kc = 2048, alternated, three rounds each |
| [`block_sweep.csv`](block_sweep.csv) | [`sme_block_sweep`](../../../benchmarks/experiments/sme_block_sweep.cpp) | the complete kernel with run-time Mc, Kc, Nc: every combination of Mc ∈ {256, 512, 1024, 2048}, Kc, Nc ∈ {1024, 2048, 4096}, plus (128, 1024, 4096); N = 512 … 4096; median of 5 (3 at N = 4096) |
| [`head_to_head.csv`](head_to_head.csv) | `VECLIB_MAXIMUM_THREADS=1 bench_gemm --benchmark_filter='^(Sme\|KleidiAI\|Amx)/'` | `gemm_sme`, KleidiAI and Accelerate on one thread, 5 interleaved repetitions (Google Benchmark CSV, `_median` rows) |
| [`accelerate_all_cores.csv`](accelerate_all_cores.csv) | `bench_gemm --benchmark_filter=^Amx/` | Accelerate with its default threading, 3 repetitions |

GFLOP/s in the Google Benchmark CSVs is the `GFLOP/s` column in FLOP/s
(divide by 10⁹). `block_sweep.csv` was produced by the predecessor of
`sme_block_sweep.cpp`, a scratch copy with the same loop nest and the same
configurations; the program in the repository reproduces it.

Conditions: a desktop machine with other work running (load average 4–10);
medians of interleaved repetitions. Results of separate runs differ by up to
~2% for the same kernel and size, which is why the article notes which run
each table comes from.
