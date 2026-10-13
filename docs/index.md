---
title: Engineering investigations
---
# HPC Math Core

Technical investigations into matrix kernels and memory performance by Vladyslav Deviatkov.

## Articles

[ARM SME2 GEMM: a power-of-two performance cliff]({{ '/articles/sme2-cache-associativity.html' | relative_url }})

Why an SME2 matrix multiplication on Apple M4 Max runs at half speed when B's rows are 16 KiB apart, and what packing and padding recover.

[Implementation and project documentation](https://github.com/vdeviatkov/hpc-math-core)
