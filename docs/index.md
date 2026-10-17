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

{% assign series = site.pages | where_exp: "p", "p.series_part" | sort: "series_part" %}
<ul class="posts">
{% for post in series %}
  <li>
    <p class="meta">Part {{ post.series_part }} · {{ post.date | date: "%B %-d, %Y" }} · {{ post.content | number_of_words | divided_by: 230 | plus: 1 }} min read</p>
    <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
    <p>{{ post.description }}</p>
  </li>
{% endfor %}
</ul>

## The project

[hpc-math-core](https://github.com/vdeviatkov/hpc-math-core) is a set of
GEMM implementations, each one optimisation step apart: scalar C++, AVX2,
AVX-512, NEON, SVE, ARM SME2, Apple's Accelerate, and CUDA from naive
kernels to pipelined Tensor Cores, every one tested against a reference and
benchmarked with Google Benchmark.

| Best kernel per family | Hardware | FP32 |
|---|---|--:|
| Scalar, cache-blocked | Apple M4 Max | 83 GFLOP/s |
| AVX-512, register-tiled | AMD Ryzen 9 9950X (Zen 5) | 333 GFLOP/s |
| ARM SME2, packed and blocked, one core | Apple M4 Max | 1.53 TFLOP/s |
| CUDA Tensor Cores, TMA + warp specialisation (fp16) | NVIDIA RTX 5080 | 120 TFLOP/s |

- [Benchmark results](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/benchmarks.md) for Apple M4 Max, AMD Zen 5 and RTX 5080
- [Kernel map](https://github.com/vdeviatkov/hpc-math-core/blob/main/src/gemm/README.md): every kernel and the technique it adds
- [Build guide](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/build.md)
- [Cache background](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/cache-behavior.md)
