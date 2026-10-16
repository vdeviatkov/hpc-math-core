# RTX 5080: measured limits and per-level profiles

Raw data behind the article
[*Matrix multiplication on an RTX 5080*](../../articles/cuda-gemm.md).
NVIDIA GeForce RTX 5080 (sm_120, 84 SMs), driver 615.71.09, VBIOS
98.03.3B.00.B4, CUDA 13.2, Ubuntu 24.04, host idle. 2026-10-15.

| File | Program | What it measures |
|---|---|---|
| [`peak.csv`](peak.csv) | [`cuda_peak`](../../../benchmarks/experiments/cuda_peak.cu) | FP32 / FP64 FMA, FP16 Tensor Core (`mma.sync.m16n8k16`, fp32 accumulate) and DRAM copy rates |
| [`clocks.csv`](clocks.csv) | `cuda_peak hold <test> 8` + `nvidia-smi` | SM and memory clock, power and temperature under each load |
| [`bench_gemm_cuda.txt`](bench_gemm_cuda.txt) | [`bench_gemm_cuda`](../../../benchmarks/bench_gemm_cuda.cpp) | every level and cuBLAS, GPU time of the GEMM alone, N = 256 … 16384 |
| [`profile.csv`](profile.csv) | Nsight Compute 2026 (`ncu`) | per-level counters at N = 4096, f32 |

## Peaks

`cuda_peak`: each test is a kernel that does nothing else, launched as one
full wave (84 SMs × 6 blocks of 256 threads), timed with CUDA events, best
of 5. FMA and MMA tests run 16 independent chains per thread or warp on
registers, each started from a different value so the compiler cannot merge
them. The Tensor Core instruction count was checked with `ncu`
(`smsp__inst_executed_pipe_tensor.sum` = the expected 1,056,964,608).

| Test | Result | Clock under that load | Power |
|---|--:|--:|--:|
| FP32 FMA | 59.7 TFLOP/s | 2,827 MHz | 300 W |
| FP64 FMA | 0.82 TFLOP/s | 2,895 MHz | 102 W |
| FP16 `mma.sync`, fp32 accumulate | 124.4 TFLOP/s | 2,887 MHz | 114 W |
| DRAM copy, read + write | 812 GB/s | 2,895 MHz | 160 W |

FP32: 59.7 TFLOP/s at 2.83 GHz is 126 flops per SM per cycle, 98% of
128 lanes × 2. FP16 Tensor Cores: 513 flops per SM per cycle. FP64: the
microbenchmark compiles to 128 DFMA and 640 NOP instructions per loop
body and reads 0.82 TFLOP/s with 4, 16 or 32 chains; the FP64 GEMM
kernels reach 0.96 TFLOP/s, so 0.82 understates the FP64 rate.

The clock is sampled by `nvidia-smi` while `cuda_peak hold` runs one test
for 8 s; `clock64()` inside the kernel is unusable for this, because the
compiler may move arithmetic across it.

## Profiles

```bash
ncu --clock-control none -k regex:'^kernel_<level>(<|$|\()' --launch-skip 1 --launch-count 1 \
    --csv --metrics <list> ./build/benchmarks/bench_gemm_cuda --benchmark_filter='^Cuda<Level>/f32/N=4096/'
```

One launch per level after the warm-up, at the GPU's normal clock
(`--clock-control none`; by default `ncu` locks it to the 2.70 GHz base
clock). The metrics:

| Metric | Meaning |
|---|---|
| `gpu__time_duration.sum` | kernel time, ns |
| `launch__registers_per_thread`, `launch__occupancy_limit_*` | registers, and blocks per SM allowed by registers / shared memory / warps |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy |
| `sm__pipe_fma_cycles_active…`, `sm__pipe_tensor_cycles_active…` | how busy the FMA and Tensor pipes are |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` / `…_wavefronts_mem_shared_op_ld.sum` | shared-memory load bank conflicts / load wavefronts |
| `dram__bytes.sum`, `lts__t_sector_hit_rate.pct`, `l1tex__t_sector_hit_rate.pct` | DRAM traffic, L2 and L1 hit rates |

Summary at N = 4096 (full data in `profile.csv`):

| Level | Time, ms | Regs | Occupancy | FMA pipe | Tensor pipe | Shared-load conflicts | DRAM, MB |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0 naive | 36.9 | 32 | 99.7% | 13.4% | — | — | 2,286 |
| 1 shared tiles | 40.4 | 40 | 99.7% | 6.8% | — | 0.0% | 2,708 |
| 2 register tile | 5.81 | 128 | 32.9% | 50.1% | — | 47.0% | 554 |
| 3 double buffer | 5.16 | 128 | 32.4% | 56.0% | — | 47.2% | 777 |
| 4 vectorized | 5.01 | 128 | 32.9% | 54.2% | — | 40.0% | 553 |
| 5 WMMA | 9.55 | 40 | 97.5% | 3.2% | 11.9% | 87.5% | 1,261 |
| 6 mma.sync | 7.99 | 40 | 97.3% | 0.9% | 14.2% | 80.0% | 1,263 |
| 7 pipelined | 1.41 | 118 | 32.2% | 3.2% | 85.8% | 1.1% | 116 |

The minimum DRAM traffic at N = 4096 is 201 MB in fp32 (A and B read, C
written) and 134 MB with fp16 inputs.

## Reproduce

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cuda_peak bench_gemm_cuda
./build/benchmarks/cuda_peak > peak.csv
./build/benchmarks/bench_gemm_cuda > bench_gemm_cuda.txt
```
