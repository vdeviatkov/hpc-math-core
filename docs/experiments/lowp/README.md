# Low-precision GEMM: bf16, fp16, int8 and FP8

Data behind [part 5](../../articles/lowp-gemm.md). Measured on 2026-10-16.

## Machines and builds

| | Apple M4 Max (P-core, SME unit) | AMD Ryzen 9 9950X (Zen 5) | NVIDIA RTX 5080 (sm_120) |
|---|---|---|---|
| OS | macOS 26.3 | Ubuntu 24.04 | same host as Zen 5 |
| Compiler | Apple Clang 17.0.0, `-O3 -mcpu=apple-m4 -ffast-math` | GCC 13.3, `-O3 -march=native -ffast-math` | nvcc 13.2, driver 615.71, cuBLAS / cuBLASLt 13.4 |
| CMake | `-DHPC_ENABLE_SME=ON` | default | `-DCMAKE_CUDA_ARCHITECTURES=120a` (and `120` for the sm_89 FP8 row) |
| Clock during the peak runs | 4.47 GHz (core) | 5.70 GHz | not pinned; board power limit 360 W |

The M4 ran with background load (Spotlight indexing, an IDE): load average
4–11. Repetitions agreed to under 1% (coefficient of variation ≤ 0.9% on
every M4 row), but treat M4 figures as ±5%, as in parts 1–2.

## Files

| File | Program | What |
|---|---|---|
| `lowp_peak_m4.csv`, `lowp_peak_zen5.csv` | [`lowp_peak`](../../../benchmarks/experiments/lowp_peak.cpp) | Latency (1 accumulator) and throughput (8, 16 accumulators; 1 and 4 ZA tiles) of every multiply-accumulate instruction used, with FP32 on the same unit |
| `cuda_peak_sm120.csv`, `cuda_peak_sm120a.csv`, `cuda_peak_device.txt` | [`cuda_peak`](../../../benchmarks/experiments/cuda_peak.cu) | Tensor Core `mma.sync` throughput per format; the block-scaled FP8 row exists only in the sm_120a build |
| `bench_lowp_m4.csv`, `bench_lowp_zen5.csv` | [`bench_gemm_lowp`](../../../benchmarks/bench_gemm_lowp.cpp) | CPU GEMMs, square N = 256 … 4096, with the FP32 kernel of the same unit; 3 repetitions, mean / median / stddev / cv |
| `bench_lowp_cuda_sm120a.csv` | [`bench_gemm_lowp_cuda`](../../../benchmarks/bench_gemm_lowp_cuda.cpp) | `Mma/…` the cp.async kernel, `Tma/…` Level 8, `Cublas/…` cuBLAS (`cublasGemmEx`; `cublasLtMatmul` for FP8); five formats, N = 1024 … 16384, one session (2026-10-16) |
| `bench_lowp_cuda_fp8_sm120.csv` | same | Both FP8 kernels built for plain sm_120 (sm_89 `mma` form) |
| `ncu_lowp_N8192.csv` | Nsight Compute | Per implementation and format at N = 8192: kernel name, registers, warp and Tensor instructions, Tensor pipe activity, time, DRAM bytes |
| `lowp_accuracy.csv` | [`lowp_accuracy`](../../../benchmarks/experiments/lowp_accuracy.cpp) | Relative Frobenius error of each format against the exact product of the FP32 inputs, K = 256 … 16384 |

The Zen 5 CPU benchmark ran pinned to core 2 (`taskset -c 2`); the GPU
benchmarks report GPU-event time of the GEMM alone, as in part 4.

## Method notes

- **Operations.** Every rate counts a multiply-add as 2 operations, for
  INT8 too, so GFLOP/s and GOP/s compare directly.
- **Inputs.** Converted to the format before timing: a real workload stores
  weights and activations in the format. Packing (CPU) is timed, as for the
  FP32 kernels.
- **The `err` counter** (CUDA benchmark): 256 sampled C(i,j) recomputed in
  double from the same converted inputs; max |C − C_ref| / Σ_k |A(i,k)·B(k,j)|.
  It measures accumulation error only. Both hand-written kernels and cuBLAS
  give identical values for every format and size.
- **Kernel tuning (CUDA, cp.async kernel).** Swept on N = 4096, 8192,
  16384, all formats, one run each in an sm_120 build
  (`tuning_cp_async.csv`): pipeline stages 2 / 3 / 4, block order row-major
  / grouped by 8 rows of tiles; a 128 × 256 block tile was also tried and
  was not faster. Best everywhere: 2 stages (two 40 KiB blocks per SM, 8–18%
  faster than 3 or 4), grouped order. The grouped order matters most for
  the fast formats: INT8 at N = 16384 goes from 181 to 308 TOP/s, bf16 from
  97 to 110 TFLOP/s. (The final run, `bench_lowp_cuda_sm120a.csv`, is a
  later session; its values differ from the sweep by up to 2%.)
- **Kernel tuning (Level 8).** Swept at N = 2048 and 8192, all formats:
  128 × 128 tiles with 4, 5, 6 stages and 128 × 256 tiles with 3, 4
  stages (`tuning_level8.csv`). 128 × 256 makes the GEMM 3–11% faster; the stage count changes
  little from 3 on. With one block per SM the tile is chosen by how full
  the last wave of blocks is (see `launch_tma` in `gemm_lowp.cu`).
- **cuBLAS FP8.** Nsight Compute shows cuBLASLt's FP8 kernel executing
  `QMMA.SF.16832.F32.E4M3.E4M3.E8`, the block-scaled MMA, the same number
  of times as the hand-written kernels' `mma` (16.8 M at N = 4096). At
  N = 8192 cuBLAS picks `nvjet_sm120_qqsss_mma_256x128x128_…_tmaAB…` for
  FP8 and the matching `bii` kernel for INT8 (`ncu_lowp_N8192.csv`).

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON   # Apple M4
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a   # RTX 50
cmake --build build --parallel
./build/tests/hpc_tests_lowp && ./build/tests/hpc_tests_lowp_cuda
./build/benchmarks/lowp_peak
./build/benchmarks/cuda_peak
./build/benchmarks/lowp_accuracy
./build/benchmarks/bench_gemm_lowp --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
./build/benchmarks/bench_gemm_lowp_cuda --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
```
