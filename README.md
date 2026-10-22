# hpc-math-core

[![CI — Build & Test](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml/badge.svg)](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml)

Progressive GEMM optimization and benchmarking across scalar C++, SIMD, ARM SME, Apple AMX and CUDA Tensor Cores, in FP32/FP64 and in bf16, fp16, INT8 and FP8.

Starting from readable scalar code, each step adds one layer of hardware exploitation — cache-friendly loop order, cache blocking, explicit SIMD (AVX2 · AVX-512 · NEON · SVE), software prefetch, CUDA (shared-memory tiling through Tensor Cores and TMA), and matrix-engine hardware (ARM SME2, Apple AMX). Low-precision versions of the fastest kernels show what each narrower format changes. Google Benchmark registers every kernel family; available kernels are checked against a reference by Google Test. SVE still lacks hardware validation.

---

## Headline results

Best kernel per family, single-threaded unless noted. Full output, speedup tables and analysis in [docs/benchmarks.md](docs/benchmarks.md).

| Family | Hardware | f32 peak | f64 peak |
|---|---|---|---|
| Scalar, cache-blocked (`gemm_blocked`) | Apple M4 Max | 83 GFLOP/s | 26 GFLOP/s |
| ARM NEON (`gemm_neon_blocked`) ⁴ | Apple M4 Max | 124 GFLOP/s | 62 GFLOP/s |
| AVX2 (`gemm_avx2_blocked`) ⁴ | AMD Zen 5, GCC | 172 GFLOP/s | 84 GFLOP/s |
| AVX-512 (`gemm_avx512_blocked`) ⁴ | AMD Zen 5, GCC | **331 GFLOP/s** | **162 GFLOP/s** |
| ARM SME2 (`gemm_sme`) | Apple M4 Max | **1.53 TFLOP/s** | **436 GFLOP/s** |
| Apple AMX via Accelerate (`gemm_amx`) ¹ | Apple M4 Max | 3.18 TFLOP/s | 804 GFLOP/s |
| Accelerate / KleidiAI, 1 core (reference) | Apple M4 Max | 1.72 / 1.66 TFLOP/s | 456 GFLOP/s / — |
| CUDA, FMA (`gemm_cuda_vectorized`) ² | NVIDIA RTX 5080 | 27.5 TFLOP/s | 0.94 TFLOP/s |
| CUDA, Tensor Cores (`gemm_cuda_tma`, Level 8) ² ³ | NVIDIA RTX 5080 | **120 TFLOP/s** | — |
| cuBLAS SGEMM / dense-FP16 reference ² | NVIDIA RTX 5080 | 39 / 120 TFLOP/s | 0.85 TFLOP/s |

¹ Vendor BLAS with its default threading (all cores) — not a single-core comparison.
² GPU time of the GEMM alone, data already on the GPU (as the CPU rows measure the multiplication alone); best over N ≤ 16384.
³ fp16 inputs, fp32 accumulate; 100 % of cuBLAS FP16 on the same GPU at N = 16384 (Level 7, `gemm_cuda_wmma_pipelined`: 101 TFLOP/s, 83 %).
⁴ Best over every register-tile shape and cache block measured ([docs/kernels](docs/kernels/README.md)). The defaults reach 124 / 62 (NEON 6×4), 172 / 83 (AVX2 6×2) and 328 / 162 (AVX-512 8×3); the AVX-512 f32 best is the 14×2 tile at N = 256, the AVX2 f64 best the 6×2 tile with the earlier 64-row block. At N = 4096 the defaults run at 102 / 51, 106 / 54 and 295 / 145.

Low precision, best size, GOP/s counting a multiply-add as 2 ([article 5](docs/articles/lowp-gemm.md)):

| Kernel | Hardware | bf16 → fp32 | INT8 → INT32 |
|---|---|--:|--:|
| `gemm_avx512_bf16` / `gemm_avx512_i8` | AMD Zen 5, 1 core | 609 | 922 |
| `gemm_neon_bf16` / `gemm_neon_i8` | Apple M4 Max, 1 P-core | 67 | 508 |
| `gemm_sme_widening` | Apple M4 Max, 1 SME unit | 1,774 | 3,525 |
| `gemm_cuda_lowp_tma_device` (Level 8) | NVIDIA RTX 5080 | 120,500 | 419,000 |

---

## Optimisation ladder

Each family steps through the same techniques, one kernel per step.

| Family | Kernels | Steps |
|---|---|---|
| Scalar | `gemm_naive` → `gemm_reordered` → `gemm_blocked` | i-j-k → cache-friendly i-k-j → cache tiling (tile 64) |
| SIMD: AVX2, AVX-512, NEON, SVE | `gemm_<isa>_naive` → `_reordered` → `_blocked` | the same three steps with intrinsics, plus a register-tiled micro-kernel; SVE is vector-length agnostic with predicated tails |
| Software prefetch | `gemm_<isa>_blocked_prefetch` | `__builtin_prefetch` for A, B and C, distance swept over {2, 4, 8, 16} |
| ARM SME2 | `gemm_sme_1tile` → `gemm_sme_tiles` → `gemm_sme_blocked` → `gemm_sme` | `FMOPA` outer products into one ZA tile → all ZA tiles → GotoBLAS cache blocking → packed B |
| CUDA | `gemm_cuda_*`, Levels 0–8 | shared-memory tiling → register tiling → `cp.async` double buffering → 128-bit loads + swizzle → Tensor Cores via WMMA → raw `ldmatrix`/`mma.sync` → pipelined WMMA with 128×128 tiles → TMA + warp-specialised `mma.sync` |
| Low precision, CPU | `gemm_avx512_{bf16,i8}`, `gemm_neon_{bf16,i8,f16,f16_acc}`, `gemm_sme_widening<…>` | the FP32 kernels with widening dot-product / outer-product instructions and k-group packing |
| Low precision, CUDA | `gemm_cuda_{bf16,fp16,fp16_acc16,i8,fp8}` | one `mma.sync` kernel template for five formats, as a cp.async kernel and as Level 8 |
| References | `gemm_amx` (Accelerate), `gemm_kleidiai`, `gemm_cuda_cublas*` | vendor libraries the hand-written kernels are measured against |

A family is compiled only where its ISA exists; elsewhere its kernels are
declared `= delete`, and the benchmarks list them as `SKIPPED`. Every family
has run on real hardware except SVE, for which no machine was available.
Files, techniques and intrinsics: [src/gemm/README.md](src/gemm/README.md).
Detailed reasoning, contracts and tradeoffs for each source file:
[docs/kernel-design.md](docs/kernel-design.md).

---

## Quick start

Requires CMake ≥ 3.25 and a C++20 compiler (GCC 12, Clang 16, Apple Clang 15 or MSVC 19.35+). CUDA 12.0+ is optional (Level 8 uses the CUDA 12 tensor-map API); without it the CUDA tests and benchmarks build against a stub and skip. Measured with GCC 13.3, Apple Clang 17, CUDA 13.2 and CMake 3.28 / 4.1.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # -O3 -march=native -ffast-math; CUDA detected automatically
cmake --build build --parallel
ctest --test-dir build --output-on-failure

./build/benchmarks/bench_gemm           --benchmark_filter="f32"
./build/benchmarks/bench_gemm_cuda      --benchmark_filter="N=4096"
./build/benchmarks/bench_gemm_lowp      # bf16 / fp16 / INT8 on this CPU
./build/benchmarks/bench_gemm_lowp_cuda # five formats on the GPU
```

On an RTX 50-series GPU, `-DCMAKE_CUDA_ARCHITECTURES=120a` enables the
full-rate block-scaled FP8 instruction; everything else is the same as the
default native build.

Options (`HPC_ENABLE_SME`, `HPC_MARCH`, …), Windows, and what runs on which
machine: [docs/build.md](docs/build.md).

---

## Repository layout

```
include/hpc/                  Matrix<T> (64-byte aligned, row-major), ISA detection macros, bf16/fp16/FP8 types
src/gemm/                     One header per kernel family, plus README.md (map of kernels and techniques)
src/cuda/                     CUDA kernels (Levels 0-7; low precision and Level 8), stubs for builds without CUDA
tests/                        Correctness tests: every kernel against a reference
benchmarks/                   bench_gemm, bench_gemm_lowp (CPU); bench_gemm_cuda, bench_gemm_lowp_cuda (GPU)
benchmarks/experiments/       Microbenchmarks behind the articles: instruction peaks, caches, strides, block sizes
docs/articles/                The five articles and their charts (published with GitHub Pages)
docs/experiments/             Raw data and method for every measured table in the articles
docs/                         Benchmark results, build guide, kernel design guide, cache background
.github/workflows/            CI (Linux x86, Linux ARM, macOS) and the Pages site
```

---

## Documentation

| Document | Contents |
|---|---|
| [Matrix multiplication with SIMD on Apple M4 and AMD Zen 5](docs/articles/simd-gemm.md) | Article 1: loop order, cache blocking and SIMD register tiles, with FMA latency and throughput measured on both CPUs |
| [Matrix multiplication on Apple M4's SME2 unit](docs/articles/sme2-gemm.md) | Article 2: the SME2 GEMM step by step, from outer products to 1.5 TFLOP/s, with the machine measured down to its TLBs |
| [A power-of-two stride cuts SME2 GEMM throughput by two thirds on Apple M4](docs/articles/sme2-cache-associativity.md) | Article 3: L2 set conflicts at a 16 KiB row stride, located and measured, and the two fixes |
| [Matrix multiplication on an RTX 5080](docs/articles/cuda-gemm.md) | Article 4: nine CUDA kernels from 3.7 to 120 TFLOP/s, with the GPU's peaks, occupancy and bank conflicts measured |
| [Matrix multiplication in bf16, fp16, INT8 and FP8](docs/articles/lowp-gemm.md) | Article 5: the same GEMM in five low-precision formats on M4 NEON and SME2, Zen 5 AVX-512 and the RTX 5080: peaks per format, the kernel changes, and the accuracy cost |
| [src/gemm/README.md](src/gemm/README.md) | Map of the kernels: files, techniques, registers and intrinsics per family. Each kernel's design is in its own header. |
| [docs/kernel-design.md](docs/kernel-design.md) | File-by-file design: algorithms, layouts, edges, precision, benchmark methodology and supporting files |
| [docs/kernels/](docs/kernels/README.md) | Register-tile and cache-block sweeps of the AVX2, AVX-512 and NEON kernels on Zen 5 and M4, with the reasons for each default |
| [docs/benchmarks.md](docs/benchmarks.md) | All measurements and their analysis: Apple M4 Max, AMD Zen 5, RTX 5080 |
| [docs/build.md](docs/build.md) | Build on Linux, macOS and Windows, CMake options, SME/AMX flags, what runs where, CI |
| [docs/cache-behavior.md](docs/cache-behavior.md) | Background: cache lines, reuse distance, working sets, memory-bound limits, GPU memory hierarchy |

---

## License

MIT. See [LICENSE](LICENSE).
