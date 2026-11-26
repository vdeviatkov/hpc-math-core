# CUDA tile sweeps, RTX 5080

Data behind [docs/kernels/cuda.md](../../../kernels/cuda.md). NVIDIA GeForce
RTX 5080 (sm_120, 84 SMs), driver 615.71.09, CUDA 13.4, build
`CMAKE_CUDA_ARCHITECTURES=120a`, AMD Ryzen 9 9950X host, Ubuntu 24.04, host
idle (load average < 1.1). GPU event time of the GEMM alone;
median of 3 randomly interleaved repetitions; coefficient of variation
≤ 0.9% in every row.

| File | Program | What |
|---|---|---|
| [`cuda_sweep_fma.csv`](cuda_sweep_fma.csv) | `bench_gemm_cuda --benchmark_filter='Var/'` | Levels 2–7 FP32 variants (`cuda_variants.hpp`), N = 1024, 4096, 8192 |
| [`cuda_sweep_fma_check.json`](cuda_sweep_fma_check.json) | the same rows at N = 1024, one repetition | `max_rel_diff` of every variant against cuBLAS on the same inputs (the CSV writer drops counters absent from its first row, so the check is in this JSON) |
| [`cuda_sweep_extra.json`](cuda_sweep_extra.json) | `CudaNaiveVar`, `CudaBlockedVar` | Levels 0–1 with 8, 16, 32 threads square |
| [`cuda_sweep_f64.json`](cuda_sweep_f64.json) | `Cuda{RegTile,DoubleBuf,Vectorized}Var/…__db…/f64` | Levels 2–4 FP64 variants, N = 1024, 4096 |
| [`cuda_sweep_lowp.csv`](cuda_sweep_lowp.csv) | `bench_gemm_lowp_cuda --benchmark_filter='Var/'` | low-precision cp.async kernel and Level 8 variants, all five formats, N = 1024 … 8192, with the sampled `err` counter |
| [`cuda_sweep_info.txt`](cuda_sweep_info.txt) | `nvidia-smi`, `date` | GPU, driver, temperature at the start; start and end times |

The levels at the sizes the sweep chose are measured in
[`../bench_gemm_cuda.json`](../bench_gemm_cuda.json).

Checks: FP32 FMA variants within 3.1 × 10⁻⁵ relative of cuBLAS SGEMM,
FP64 variants identical to cuBLAS DGEMM, Level 7 identical to cuBLAS FP16,
WMMA levels (fp16 inputs) within 1.2% of SGEMM; low-precision `err` equal
across variants of a format (INT8 exact).
