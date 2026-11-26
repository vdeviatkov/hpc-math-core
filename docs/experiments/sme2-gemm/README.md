# SME2 GEMM: the kernel ladder, block sizes and the head-to-head comparison

Raw data behind the step tables, the block-size section and "Where it
stands" in
[*Matrix multiplication on Apple M4's SME2 unit*](../../articles/sme2-gemm.md).
Apple M4 Max, one thread, Apple Clang 17,
`-O3 -mcpu=apple-m4 -ffast-math -funroll-loops`. The machine's
caches, TLBs and FMOPA rates are in [sme2-machine](../sme2-machine/README.md).

| File | Program | What it measures |
|---|---|---|
| [`ladder.json`](ladder.json), [`ladder_load.txt`](ladder_load.txt) | [`bench_gemm`](../../../benchmarks/bench_gemm.cpp) `--benchmark_filter='^(Sme1Tile\|SmeTiles\|SmeBlocked\|Sme)/'`, 3 interleaved repetitions | the four SME steps, N = 256 … 4096, FP32 and FP64; load average 3–5 |
| [`tiles_capacity.csv`](tiles_capacity.csv) | [`sme_probe tiles`](../../../benchmarks/experiments/sme_probe.cpp), load average 5–8 | `gemm_sme_tiles` FP32 (no cache blocking) at N = 1024 … 4096, powers of two and not, median of 5: the size of A against the 16 MiB L2 sets the slowdown, with one stride dip at N = 2048 |
| [`block_sweep.csv`](block_sweep.csv), [`block_sweep_load.txt`](block_sweep_load.txt) | [`sme_block_sweep`](../../../benchmarks/experiments/sme_block_sweep.cpp) | the complete kernel at the 1 × 4 shape with run-time Mc, Kc, Nc: every combination of Mc ∈ {256, 512, 1024, 2048}, Kc, Nc ∈ {1024, 2048, 4096}, plus the B-resident (128, 1024, 4096); N = 512 … 4096; median of 5 (3 at N = 4096) |
| [`block_stage2.json`](block_stage2.json), [`block_stage2_load.txt`](block_stage2_load.txt) | `bench_gemm --benchmark_filter=SmeBlock`, 5 interleaved repetitions | the six leading blocks of the sweep, compiled in, at N = 512, 1024, 4096 |
| [`head_to_head.json`](head_to_head.json), [`head_to_head_load.txt`](head_to_head_load.txt) | `VECLIB_MAXIMUM_THREADS=1 bench_gemm --benchmark_filter='^(Sme\|Amx\|KleidiAI)/'`, 3 interleaved repetitions | `gemm_sme`, Accelerate and KleidiAI on one thread; load average 2–3. Accelerate with all cores is in [sme-mt/default.json](../sme-mt/default.json) |

GFLOP/s in the Google Benchmark CSVs is the `GFLOP/s` column in FLOP/s
(divide by 10⁹).

Conditions: a desktop machine with other work running (load averages in the
`*_load.txt` files); medians of interleaved repetitions. Results of separate
runs differ by up to ~2% for the same kernel and size, which is why the
article notes which run each table comes from.
