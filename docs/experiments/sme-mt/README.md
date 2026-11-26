# `gemm_sme_mt`: several SME units

Raw data behind [kernels/sme.md § Several SME units](../../kernels/sme.md#several-sme-units)
and [benchmarks.md § Several SME units](../../benchmarks.md#several-sme-units-gemm_sme_mt).
Apple M4 Max (two P-clusters of six cores, `hw.perflevel0.cpusperl2` = 6),
Apple Clang 17, `-DHPC_ENABLE_SME=ON`, AC power, ZA shape 1 × 4. Wall-clock
time for every multi-threaded row (as for `Amx`); `Sme` is single-threaded.

| File | Command | Contents |
|---|---|---|
| [`threads.json`](threads.json), [`threads_load.txt`](threads_load.txt) | `bench_gemm --benchmark_filter='^(SmeMtT[0-9]+Mc1024\|Sme\|Amx)/f(32\|64)/N=(1024\|2048\|4096)(/\|$)' --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true --benchmark_report_aggregates_only=true` | threads {1, 2, 3, 4, 6, 7, 8, 10, 12} at TileM = 1024, with `gemm_sme` and Accelerate (all cores); load average 6–10 |
| [`default.json`](default.json), [`default_load.txt`](default_load.txt) | `bench_gemm --benchmark_filter='^(SmeMt\|Sme\|Amx)/' --benchmark_repetitions=5 --benchmark_enable_random_interleaving=true` | the measured configuration (8 threads, `SmeParams`) at N = 64 … 4096 and 4000, with `gemm_sme` and Accelerate; load average 2.7–6 |

Results: [docs/kernels/sme.md](../../kernels/sme.md#several-sme-units)
(thread sweep) and [docs/benchmarks.md](../../benchmarks.md#several-sme-units-gemm_sme_mt)
(the default against `gemm_sme` and Accelerate).
