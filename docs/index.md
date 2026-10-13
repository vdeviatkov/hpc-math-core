---
title: Engineering investigations
---
# HPC Math Core

Technical investigations into matrix kernels and memory performance by Vladyslav Deviatkov.

## Articles

[A power-of-two stride halves SME2 GEMM throughput on Apple M4]({{ '/articles/sme2-cache-associativity.html' | relative_url }})

An SME2 matrix multiplication runs at half speed when B's rows are 16 KiB apart. Tracing it to L2 set conflicts, measuring where they start, and the cost of padding and packing.

[Implementation and project documentation](https://github.com/vdeviatkov/hpc-math-core)
