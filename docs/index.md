---
title: Engineering investigations
---
# HPC Math Core

Technical investigations into matrix kernels and memory performance by Vladyslav Deviatkov.

## Articles

[Matrix multiplication on Apple M4's SME2 unit: from outer products to 1.45 TFLOP/s]({{ '/articles/sme2-gemm.html' | relative_url }})

How an SME2 GEMM goes from 474 to 1,450 GFLOP/s on one core: the M4 Max measured down to its TLBs, outer products, all four ZA tiles, GotoBLAS blocking and packing, each step with data.

[A power-of-two stride halves SME2 GEMM throughput on Apple M4]({{ '/articles/sme2-cache-associativity.html' | relative_url }})

An SME2 matrix multiplication runs at half speed when B's rows are 16 KiB apart. Tracing it to L2 set conflicts, measuring where they start, and the cost of padding and packing.

[Implementation and project documentation](https://github.com/vdeviatkov/hpc-math-core)
