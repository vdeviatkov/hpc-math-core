---
title: HPC Math Core — engineering investigations
description: Measured, reproducible investigations into matrix-multiplication kernels and memory performance on modern CPUs and GPUs.
---
# HPC Math Core

<p class="lead">Measured, reproducible investigations into how matrix
multiplication meets real hardware: instruction latencies, caches, TLBs and
the memory layouts that decide whether a kernel runs at peak or at half of
it.</p>

Every number in these articles comes from code in the
[hpc-math-core](https://github.com/vdeviatkov/hpc-math-core) repository and is
recorded with its raw samples, machine details and the exact source it was
measured with.

## Matrix multiplication, from SIMD to Tensor Cores

A five-part series: the same multiplication on CPU vector units (Apple M4
and AMD Zen 5), on Apple's SME2 matrix unit, and on an NVIDIA GPU, then in
16- and 8-bit formats on all of them, each step measured.

1. **[Matrix multiplication with SIMD on Apple M4 and AMD Zen 5](articles/simd-gemm.md)**
   — loop order, cache blocks, register tiles and packed B take one Zen 5
   core from 0.4 to 328 GFLOP/s FP32, 91% of peak.
2. **[Matrix multiplication on Apple M4's SME2 unit](articles/sme2-gemm.md)**
   — outer products, all four ZA tiles, blocking and packing: 475 to
   1,645 GFLOP/s on one matrix unit, 91–100% of Accelerate.
3. **[A power-of-two stride cuts SME2 GEMM throughput by two thirds on Apple M4](articles/sme2-cache-associativity.md)**
   — B rows 16 KiB apart cause L2 set conflicts; padding or packing B fixes
   it. On Zen 5 the same effect costs 14–17%.
4. **[Matrix multiplication on an RTX 5080](articles/cuda-gemm.md)**
   — nine CUDA kernels, one technique apart, from 3.7 to 120 TFLOP/s; the
   last, fed by TMA with warp specialisation, matches cuBLAS.
5. **[Matrix multiplication in bf16, fp16, INT8 and FP8](articles/lowp-gemm.md)**
   — the speed-up belongs to the instruction, not the bit count: from 0.53×
   FP32 (bf16 on M4 NEON) to 8.3× (INT8 and FP8 on the RTX 5080).

## The project

[hpc-math-core](https://github.com/vdeviatkov/hpc-math-core) is a set of
GEMM implementations, each one optimisation step apart: scalar C++, AVX2,
AVX-512, NEON, ARM SME2, Apple's Accelerate, Metal, and CUDA from naive
kernels to pipelined Tensor Cores, every one tested against a reference and
benchmarked with Google Benchmark.

| Best kernel per family | Hardware | FP32 |
|---|---|--:|
| Scalar, cache-blocked | Apple M4 Max | 83 GFLOP/s |
| AVX-512, register-tiled | AMD Ryzen 9 9950X (Zen 5) | 328 GFLOP/s |
| ARM SME2, packed and blocked, one core | Apple M4 Max | 1.65 TFLOP/s |
| CUDA Tensor Cores, TMA + warp specialisation (fp16) | NVIDIA RTX 5080 | 120 TFLOP/s |

- [Benchmark results](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/benchmarks.md) for Apple M4 Max, AMD Zen 5 and RTX 5080
- [Kernel map](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/README.md): every kernel and the technique it adds
- [Build guide](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/build.md)
- [Cache background](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/cache-behavior.md)
