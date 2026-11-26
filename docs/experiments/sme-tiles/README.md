# SME ZA shapes

Apple M4 Max, macOS, one calling thread, Apple Clang 17.0.0,
`-O3 -mcpu=apple-m4 -ffast-math -funroll-loops -fno-omit-frame-pointer`.
Streaming vector length 512 bits (16 float or 8 double elements). The
reported 24 MHz CPU field in the JSON is the platform timer frequency, not
a measured core clock, and is not used to derive a throughput peak.
Cache blocks held at the defaults (`SmeParams<T>` in
`benchmarks/sme_params.hpp`: 1024×2048×4096 FP32, 512×2048×4096 FP64).

| File | Program | What |
|---|---|---|
| [`tiles_m4.json`](tiles_m4.json), [`tiles_m4_load.txt`](tiles_m4_load.txt) | `bench_gemm --benchmark_filter='^SmeTile'` | `gemm_sme` with the FP32 shapes 1×4, 2×2, 4×1 and the FP64 shapes 1×4, 2×4, 2×2, 4×1 at N = 256, 512, 1024, 4096; seven randomly interleaved repetitions, minimum time 0.1 s; load average 12–13 |
| [`microkernel_m4.csv`](microkernel_m4.csv) | [`sme_microkernel`](../../../benchmarks/experiments/sme_microkernel.cpp) | the macro-kernel alone on cache-resident packed A (MC × K) and B (K × NC), FP32 1×4 / 2×2 / 4×1 and FP64 1×4 / 2×4 / 2×2 / 4×2, K = 256, 512, 2048; best of 7 runs of ≥ 0.2 s; GFLOP/s and share of the measured FMOPA peak (1,963 / 500). Every shape reaches 99–100% at K = 2048 |

Matrix allocation and packing inside each GEMM are timed; input allocation
and initialization in the benchmark harness are excluded. The desktop had
background work; medians and variability are in the JSON. These are
measurements of these implementations on one M4, not a proof of
optimality across SME CPUs. Reproduction commands and API:
[SME guide](../../kernels/sme.md). The cache-block sweep is in
[sme2-gemm](../sme2-gemm/README.md).

## Results (tiles_m4.json)

Median GFLOP/s of seven interleaved repetitions (coefficient of variation
≤ 2.7%). A shape's dimensions are counts of ZA tiles, not output elements.

| Type | ZA shape | N=256 | 512 | 1024 | 4096 |
|---|---|--:|--:|--:|--:|
| f32 | 1x4 (default) | 768.5 | 1322.7 | 1574.8 | 1602.9 |
| f32 | 2x2 | 758.4 | 1333.2 | 1467.2 | 1557.6 |
| f32 | 4x1 | 663.4 | 1143.7 | 1365.0 | 1422.6 |
| f64 | 1x4 (default) | 318.8 | 407.3 | 439.7 | 429.5 |
| f64 | 2x4 | 314.9 | 405.7 | 424.2 | 426.3 |
| f64 | 2x2 | 318.8 | 392.8 | 415.4 | 419.2 |
| f64 | 4x1 | 294.9 | 356.8 | 390.8 | 387.0 |

1×4 leads 2×2 FP32 by 7.3% at N=1024 and 2.9% at 4096 (−0.8% at 512) and
2×4 FP64 by 0.4–3.7% at every size, so `SmeParams` uses 1×4 for FP32 and
FP64. The widening types use 2×2; 1×4 has not been measured for them.
