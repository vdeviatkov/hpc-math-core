# `gemm_sme_mt`: several SME units

Raw data behind [kernels/sme.md § Several SME units](../../kernels/sme.md#several-sme-units)
and [benchmarks.md § Several SME units](../../benchmarks.md#several-sme-units-gemm_sme_mt).
Apple M4 Max (two P-clusters of six cores, `hw.perflevel0.cpusperl2` = 6),
Apple Clang 17, `-DHPC_ENABLE_SME=ON`, 2026-10-24, AC power. Wall-clock
time for every multi-threaded row (as for `Amx`); `Sme` is single-threaded.

| File | Command | Contents |
|---|---|---|
| [`sweep.json`](sweep.json), [`load.txt`](load.txt) | `bench_gemm --benchmark_filter='^(SmeMtT[0-9]+Mc[0-9]+\|Sme\|Amx)/f(32\|64)/N=(1024\|2048\|4096)(/\|$)' --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true --benchmark_report_aggregates_only=true` | threads {1, 2, 3, 4, 6, 8, 12} × A block height {256, 512, 1024}, with `gemm_sme` and Accelerate (all cores); load average 3.9 at start and end |
| [`default.json`](default.json), [`load_default.txt`](load_default.txt) | `bench_gemm --benchmark_filter='^(SmeMt\|Sme\|Amx)/' --benchmark_repetitions=5 …` | the measured configuration (8 threads, `SmeParams`) at N = 64 … 4096 and 4000, with `gemm_sme` and Accelerate; load average 2.6–6.2 |

Both files are from the final, dynamically scheduled kernel. A first
version that gave each thread an equal static band of rows was measured with
the same sweep; its numbers are quoted in kernels/sme.md (e.g. f32 N=1024:
1,470 GFLOP/s with 2 threads, 1,805 with 8; N=4096: 2,707 with 2, 2,361 with
3, 3,008 with 8), and its JSON was overwritten by the rerun.

## 8 threads against one unit and Accelerate (default.json)

GFLOP/s, median of 5:

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4000 | N=4096 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `gemm_sme` f32, 1 thread | 276 | 849 | 1,350 | 1,419 | 1,514 | 1,551 | 1,541 |
| `gemm_sme_mt` f32, 8 threads | 26 | 370 | 1,480 | 2,670 | 3,071 | 3,498 | 3,496 |
| Accelerate f32, all cores | 785 | 1,711 | 2,639 | 3,211 | 3,195 | — | 3,160 |
| `gemm_sme` f64, 1 thread | 86 | 326 | 402 | 421 | 424 | 440 | 429 |
| `gemm_sme_mt` f64, 8 threads | 14 | 270 | 618 | 758 | 894 | 915 | 914 |
| Accelerate f64, all cores | 325 | 462 | 797 | 827 | 785 | — | 783 |

At N=4096, f32 3,496 is 111% of Accelerate and 89% of the two units'
3.93 TFLOP/s; f64 914 is 117% and 91% of 1.00 TFLOP/s. Below N≈512 the
eight threads created per call cost more than they bring.
