# hpc-math-core

[![CI — Build & Test](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml/badge.svg)](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml)

Progressive GEMM optimization and benchmarking across scalar C++, SIMD, ARM SME, Apple AMX and CUDA Tensor Cores.

Starting from readable scalar code, each step adds one layer of hardware exploitation — cache-friendly loop order, cache blocking, explicit SIMD (AVX2 · AVX-512 · NEON · SVE), software prefetch, CUDA (shared-memory tiling through Tensor Cores), and matrix-engine hardware (ARM SME2, Apple AMX). Every kernel is benchmarked with Google Benchmark, cross-validated against a scalar reference by a Google Test suite, and documented with memory-access diagrams and cache analysis.

---

## Headline results

Best kernel per family, single-threaded unless noted. Full output, speedup tables and analysis in [docs/benchmarks.md](docs/benchmarks.md).

| Family | Hardware | f32 peak | f64 peak |
|---|---|---|---|
| Scalar, cache-blocked (`gemm_blocked`) | Apple M4 Max | 83 GFLOP/s | 26 GFLOP/s |
| ARM NEON (`gemm_neon_blocked`) | Apple M4 Max | 95 GFLOP/s | 40 GFLOP/s |
| AVX2 (`gemm_avx2_blocked`) | AMD Zen 5, GCC | 152 GFLOP/s | 75 GFLOP/s |
| AVX-512 (`gemm_avx512_blocked`) | AMD Zen 5, GCC | **333 GFLOP/s** | **162 GFLOP/s** |
| ARM SME2 (`gemm_sme`) | Apple M4 Max | **1.53 TFLOP/s** | **436 GFLOP/s** |
| Apple AMX via Accelerate (`gemm_amx`) ¹ | Apple M4 Max | 3.3 TFLOP/s | 860 GFLOP/s |
| Accelerate / KleidiAI, 1 core (reference) | Apple M4 Max | 1.72 / 1.66 TFLOP/s | 456 GFLOP/s / — |
| CUDA, FMA (`gemm_cuda_vectorized`) ² | NVIDIA RTX 5080 | 27.5 TFLOP/s | 0.94 TFLOP/s |
| CUDA, Tensor Cores (`gemm_cuda_wmma_pipelined`) ² ³ | NVIDIA RTX 5080 | **101 TFLOP/s** | — |
| cuBLAS SGEMM / dense-FP16 reference ² | NVIDIA RTX 5080 | 39 / 120 TFLOP/s | 0.85 TFLOP/s |

¹ Vendor BLAS, may use multiple cores — not a single-core comparison.
² GPU time of the GEMM alone, data already on the GPU (as the CPU rows measure the multiplication alone); best over N ≤ 16384.
³ fp16 inputs, fp32 accumulate; 83 % of cuBLAS FP16 on the same GPU.

---

## Optimisation ladder

Each family steps through the same techniques, one kernel per step.

| Family | Kernels | Steps |
|---|---|---|
| Scalar | `gemm_naive` → `gemm_reordered` → `gemm_blocked` | i-j-k → cache-friendly i-k-j → cache tiling (tile 64) |
| SIMD: AVX2, AVX-512, NEON, SVE | `gemm_<isa>_naive` → `_reordered` → `_blocked` | the same three steps with intrinsics, plus a register-tiled micro-kernel; SVE is vector-length agnostic with predicated tails |
| Software prefetch | `gemm_<isa>_blocked_prefetch` | `__builtin_prefetch` for A, B and C, distance swept over {2, 4, 8, 16} |
| ARM SME2 | `gemm_sme_1tile` → `gemm_sme_tiles` → `gemm_sme_blocked` → `gemm_sme` | `FMOPA` outer products into one ZA tile → all ZA tiles → GotoBLAS cache blocking → packed B |
| CUDA | `gemm_cuda_*`, Levels 0–7 | shared-memory tiling → register tiling → `cp.async` double buffering → 128-bit loads + swizzle → Tensor Cores via WMMA → raw `ldmatrix`/`mma.sync` → pipelined WMMA with 128×128 tiles |
| References | `gemm_amx` (Accelerate), `gemm_kleidiai`, `gemm_cuda_cublas*` | vendor libraries the hand-written kernels are measured against |

A family is compiled only where its ISA exists; elsewhere its kernels are
declared `= delete`, and the benchmarks list them as `SKIPPED`. Every family
has run on real hardware except SVE, for which no machine was available.
Files, techniques and intrinsics: [src/gemm/README.md](src/gemm/README.md).

---

## Quick start

Requires CMake ≥ 3.25 and a C++20 compiler (GCC 12, Clang 16, Apple Clang 15 or MSVC 19.35+). CUDA 11.8+ is optional; without it the CUDA tests and benchmarks build against a stub and skip.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # -O3 -march=native -ffast-math; CUDA detected automatically
cmake --build build --parallel
ctest --test-dir build --output-on-failure

./build/benchmarks/bench_gemm      --benchmark_filter="f32"
./build/benchmarks/bench_gemm_cuda --benchmark_filter="N=4096"
```

Options (`HPC_ENABLE_SME`, `HPC_MARCH`, …), Windows, and what runs on which
machine: [docs/build.md](docs/build.md).

---

## Repository layout

```
include/hpc/                  Matrix<T> (64-byte aligned, row-major) and the ISA detection macros
src/gemm/                     One header per kernel family, plus README.md (map of kernels and techniques)
src/cuda/                     CUDA kernels; gemm_kernels_stub.cpp for builds without CUDA
tests/                        Correctness tests: every kernel against gemm_naive
benchmarks/                   bench_gemm (CPU) and bench_gemm_cuda (GPU)
docs/                         Benchmark results, build guide, cache background, articles
.github/workflows/build.yml   CI: Linux x86, Linux ARM, macOS
```

---

## Documentation

| Document | Contents |
|---|---|
| [Matrix multiplication with SIMD on Apple M4 and AMD Zen 5](docs/articles/simd-gemm.md) | Article 1: loop order, cache blocking and SIMD register tiles, with FMA latency and throughput measured on both CPUs |
| [Matrix multiplication on Apple M4's SME2 unit](docs/articles/sme2-gemm.md) | Article 2: the SME2 GEMM step by step, from outer products to 1.5 TFLOP/s, with the machine measured down to its TLBs |
| [A power-of-two stride halves SME2 GEMM throughput on Apple M4](docs/articles/sme2-cache-associativity.md) | Article 3: L2 set conflicts at a 16 KiB row stride, located and measured, and the two fixes |
| [Matrix multiplication on an RTX 5080](docs/articles/cuda-gemm.md) | Article 4: eight CUDA kernels from 3.7 to 100 TFLOP/s, with the GPU's peaks, occupancy and bank conflicts measured |
| [src/gemm/README.md](src/gemm/README.md) | Map of the kernels: files, techniques, registers and intrinsics per family. Each kernel's design is in its own header. |
| [docs/benchmarks.md](docs/benchmarks.md) | All measurements and their analysis: Apple M4 Max, AMD Zen 5, RTX 5080 |
| [docs/build.md](docs/build.md) | Build on Linux, macOS and Windows, CMake options, SME/AMX flags, what runs where, CI |
| [docs/cache-behavior.md](docs/cache-behavior.md) | Background: cache lines, reuse distance, working sets, memory-bound limits, GPU memory hierarchy |

---

## License

MIT. See [LICENSE](LICENSE).
