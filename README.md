# hpc-math-core

[![CI — Build & Test](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml/badge.svg)](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml)

Progressive GEMM optimization and benchmarking across scalar C++, SIMD, ARM SME, Apple AMX, CUDA Tensor Cores and Apple GPUs (Metal), in FP32/FP64 and in bf16, fp16, INT8 and FP8.

Starting from readable scalar code, each step adds one layer of hardware exploitation — cache-friendly loop order, cache blocking, explicit SIMD (AVX2 · AVX-512 · NEON · SVE), software prefetch, CUDA (shared-memory tiling through Tensor Cores and TMA), and matrix-engine hardware (ARM SME2, Apple AMX). Low-precision versions of the fastest kernels show what each narrower format changes. Google Benchmark registers every kernel family; available kernels are checked against a reference by Google Test. SVE still lacks hardware validation.

---

## Articles

A five-part series built on this code; every number in it comes from the kernels and experiments here. Read on the [web](https://vdeviatkov.github.io/hpc-math-core/) or as Markdown.

| Part | Article | Hardware | Key result |
|--:|---|---|---|
| 1 | [Matrix multiplication with SIMD on Apple M4 and AMD Zen 5](https://vdeviatkov.github.io/hpc-math-core/articles/simd-gemm.html) ([md](docs/articles/simd-gemm.md)) | M4 P-core (NEON), Zen 5 core (AVX2, AVX-512) | The textbook loop at 0.4 GFLOP/s to 328 GFLOP/s FP32 on one Zen 5 core, 91% of peak: loop order, cache blocks, register tiles and packed B, each chosen by measurement, and a compiler trap that held AVX-512 at 60% |
| 2 | [Matrix multiplication on Apple M4's SME2 unit](https://vdeviatkov.github.io/hpc-math-core/articles/sme2-gemm.html) ([md](docs/articles/sme2-gemm.md)) | M4 SME2 matrix unit | Outer products, all four accumulator tiles, blocking and packing: 472 → 1,532 GFLOP/s on one unit (78% of peak), 85–94% of Apple's Accelerate |
| 3 | [A power-of-two stride cuts SME2 GEMM throughput by two thirds on Apple M4](https://vdeviatkov.github.io/hpc-math-core/articles/sme2-cache-associativity.html) ([md](docs/articles/sme2-cache-associativity.md)) | M4 SME2, Zen 5 | B rows 16 KiB apart cost the M4 kernel two thirds of its speed through L2 set conflicts; padding or packing B fixes it. On Zen 5 the same effect costs 14–17% |
| 4 | [Matrix multiplication on an RTX 5080](https://vdeviatkov.github.io/hpc-math-core/articles/cuda-gemm.html) ([md](docs/articles/cuda-gemm.md)) | RTX 5080 (FMA cores, Tensor Cores) | Nine CUDA kernels, one technique apart, from 3.7 to 120 TFLOP/s; the last, fed by TMA with warp specialisation, matches cuBLAS |
| 5 | [Matrix multiplication in bf16, fp16, INT8 and FP8](https://vdeviatkov.github.io/hpc-math-core/articles/lowp-gemm.html) ([md](docs/articles/lowp-gemm.md)) | M4 NEON and SME2, Zen 5 AVX-512, RTX 5080 | The speed-up is a property of the instruction, not of the bit count: from 0.53× FP32 (bf16 on M4 NEON) to 8.3× (INT8 and FP8 on the RTX 5080), with the accuracy each format costs |

---

## Headline results

Best kernel per family, single-threaded unless noted. The percentage after each result is its **efficiency**: measured throughput ÷ the compute peak in the last column, i.e. how much of the hardware's arithmetic the kernel actually uses (100% would mean every FMA unit busy every cycle). Full output, speedup tables and analysis in [docs/benchmarks.md](docs/benchmarks.md).

| Family | Hardware | f32 (% of peak) | f64 (% of peak) | Compute peak, f32 / f64 ⁵ |
|---|---|---|---|---|
| Scalar, cache-blocked (`gemm_blocked`) | Apple M4 Max | 83 GFLOP/s (64%) | 26 GFLOP/s (40%) | ~130 / ~65 GFLOP/s: 4 FMA pipes × 128-bit, ~4.05 GHz |
| ARM NEON (`gemm_neon_blocked`) ⁴ | Apple M4 Max | 124 GFLOP/s (86%) | 62 GFLOP/s (87%) | 144 / 72 GFLOP/s: 4 FMA pipes × 128-bit, 4.49 GHz |
| AVX2 (`gemm_avx2_blocked`) ⁴ | AMD Zen 5, GCC | 172 GFLOP/s (94%) | 84 GFLOP/s (92%) | 183 / 91 GFLOP/s: 2 FMA pipes × 256-bit, 5.71 GHz |
| AVX-512 (`gemm_avx512_blocked`) ⁴ | AMD Zen 5, GCC | **331 GFLOP/s** (91%) | **162 GFLOP/s** (89%) | 365 / 183 GFLOP/s: 2 FMA pipes × 512-bit, 5.71 GHz |
| ARM SME2 (`gemm_sme`) | Apple M4 Max | **1.53 TFLOP/s** (78%) | **436 GFLOP/s** (87%) | 1.96 / 0.50 TFLOP/s: one SME unit, 1 `FMOPA` per cycle ⁶ |
| ARM SME2, both units (`gemm_sme_mt`, 8 threads) ⁷ | Apple M4 Max | **3.50 TFLOP/s** (89%) | **915 GFLOP/s** (92%) | 3.93 / 1.00 TFLOP/s: two SME units ⁶ |
| Apple AMX via Accelerate (`gemm_amx`) ¹ | Apple M4 Max | 3.18 TFLOP/s (81%) | 804 GFLOP/s (80%) | 3.93 / 1.00 TFLOP/s: two SME units ⁶ |
| Accelerate / KleidiAI, 1 core (reference) | Apple M4 Max | 1.72 / 1.66 TFLOP/s (88% / 85%) | 456 GFLOP/s / — (91%) | 1.96 / 0.50 TFLOP/s: one SME unit ⁶ |
| CUDA, FMA (`gemm_cuda_vectorized`) ² | NVIDIA RTX 5080 | 27.5 TFLOP/s (45%) | 0.94 TFLOP/s (97%) | 60.9 / 0.97 TFLOP/s: 84 SMs × 128 FP32 / 2 FP64 lanes, 2.83–2.90 GHz |
| CUDA, Tensor Cores (`gemm_cuda_tma`, Level 8) ² ³ | NVIDIA RTX 5080 | **120 TFLOP/s** (96%) | — | 124.7 TFLOP/s fp16 → fp32: 84 SMs × 512 flops per cycle, 2.90 GHz |
| cuBLAS SGEMM / dense-FP16 reference ² | NVIDIA RTX 5080 | 39 / 120 TFLOP/s (64% / 96%) | 0.85 TFLOP/s (88%) | as the two rows above |
| Metal, simdgroup matrices (`MetalSimdTg` / `MetalSimdTgF16`) ⁸ | Apple M4 Max GPU, 40 cores | 10.7 TFLOP/s (75%) / 11.6 with fp16 inputs (82%) | — (no FP64) | 14.2 TFLOP/s: measured FMA peak, same run; fp16 runs at the f32 rate |
| MPS reference (`MetalMps` / `MetalMpsF16`) ⁸ | Apple M4 Max GPU | 11.1 TFLOP/s (78%) / 13.7 with fp16 inputs (97%) | — | as the row above |

¹ Vendor BLAS with its default threading (all cores) — not a single-core comparison.
² GPU time of the GEMM alone, data already on the GPU (as the CPU rows measure the multiplication alone); best over N ≤ 16384.
³ fp16 inputs, fp32 accumulate; 100 % of cuBLAS FP16 on the same GPU at N = 16384 (Level 7, `gemm_cuda_wmma_pipelined`: 100.7 TFLOP/s at N = 8192, 85 %; 83 % at N = 16384).
⁴ Best over every register-tile shape and cache block measured ([docs/kernels](docs/kernels/README.md)). The defaults reach 124 / 62 (NEON 6×4), 172 / 83 (AVX2 6×2) and 328 / 162 (AVX-512 8×3); the AVX-512 f32 best is the 14×2 tile at N = 256, the AVX2 f64 best the 6×2 tile with the earlier 64-row block. At N = 4096 the defaults run at 102 / 51, 106 / 54 and 295 / 145; the packed-B kernels (`gemm_<isa>_packed`) hold 120 / 59, 167 / 81 and 319 / 152 there, within 3% of their best from N = 1024 to 4096, power of two or not, but run at 0.87–0.95× of the blocked kernels at N = 64, where the copy is not amortised ([packed-b](docs/experiments/packed-b/README.md)).
⁵ Vector units × lanes × 2 flops per FMA × the clock measured during that row's runs (the M4 P-core clock moves with background load: 3.96–4.13 GHz for the scalar row, 4.49 GHz for NEON; the GPU's with power). For the Tensor Cores, 512 flops per SM per cycle is the rate the `mma.sync` microbenchmark implies. The microbenchmarks reach 98–100% of these figures (GPU FP64: 85%, below what the FP64 GEMM kernels themselves reach) ([cpu-simd](docs/experiments/cpu-simd/README.md), [CUDA](docs/articles/cuda-gemm.md)).
⁶ Apple publishes no figures for its matrix unit. These are measured: `FMOPA` issues once per cycle with four accumulator tiles, 512 FP32 or 128 FP64 flops each ([part 2](docs/articles/sme2-gemm.md)); `gemm_amx` runs on both P-clusters' units.
⁷ Wall-clock time, 8 threads; 111% of Accelerate with all cores in f32 and 117% in f64 at N=4096, 83% / 92% at N=1024 ([§ Several SME units](docs/benchmarks.md#several-sme-units-gemm_sme_mt)).
⁸ GPU time of the GEMM alone, best over N = 256 … 8192, median of 5 in one run. The M4 Max GPU's sustained rate moved by up to ~3× between sessions on the recording laptop (power limiting is the likely cause); compare these rows with each other, not across sessions ([§ Metal](docs/benchmarks.md#apple-m4-max-gpu--metal)).

Low precision, best size, GOP/s counting a multiply-add as 2 ([article 5](docs/articles/lowp-gemm.md)):

| Kernel | Hardware | bf16 → fp32, measured / peak | INT8 → INT32, measured / peak |
|---|---|--:|--:|
| `gemm_avx512_bf16` / `gemm_avx512_i8` | AMD Zen 5, 1 core | 638 / 713 (90%) | 1,139 / 1,435 (79%) |
| `gemm_neon_bf16` / `gemm_neon_i8` | Apple M4 Max, 1 P-core | 66 / 71 (93%) | 503 / 511 (98%) |
| `gemm_sme_widening` | Apple M4 Max, 1 SME unit | 1,774 / 2,018 (88%) | 3,525 / 4,029 (87%) |
| `gemm_cuda_lowp_tma_device` (Level 8) | NVIDIA RTX 5080 | 120,500 / 124,000 (97%) | 419,000 / 497,000 (84%) |

The percentage is again measured ÷ peak. Peaks are the instructions' measured throughput (`VDPBF16PS` / `VPDPBUSD`, `BFMMLA` / `SMMLA`, widening `FMOPA` / `SMOPA`, `mma.sync`), from part 5's [peak table](docs/articles/lowp-gemm.md).

---

## Optimisation ladder

Each family steps through the same techniques, one kernel per step.

| Family | Kernels | Steps |
|---|---|---|
| Scalar | `gemm_naive` → `gemm_reordered` → `gemm_blocked` | i-j-k → cache-friendly i-k-j → cache tiling (tile 64) |
| SIMD: AVX2, AVX-512, NEON, SVE | `gemm_<isa>_naive` → `_reordered` → `_blocked` → `_packed` | the same three steps with intrinsics, plus a register-tiled micro-kernel; then GotoBLAS loop order with B packed into contiguous strips (AVX2, AVX-512, NEON); SVE is vector-length agnostic with predicated tails |
| Software prefetch | `gemm_<isa>_blocked_prefetch` | `__builtin_prefetch` for A, B and C, distance swept over {2, 4, 8, 16} |
| ARM SME2 | `gemm_sme_1tile` → `gemm_sme_tiles` → `gemm_sme_blocked` → `gemm_sme` → `gemm_sme_mt` | `FMOPA` outer products into one ZA tile → all ZA tiles → GotoBLAS cache blocking → packed B → both P-clusters' SME units (shared B panel, dynamic A chunks) |
| CUDA | `gemm_cuda_*`, Levels 0–8 | shared-memory tiling → register tiling → `cp.async` double buffering → 128-bit loads + swizzle → Tensor Cores via WMMA → raw `ldmatrix`/`mma.sync` → pipelined WMMA with 128×128 tiles → TMA + warp-specialised `mma.sync` |
| Metal (Apple GPU) | `MetalNaive` → `MetalTiled` → `MetalRegTile` → `MetalSimd` → `MetalSimdTg` → `MetalSimdTgF16` | threadgroup-memory tiles → 8×4 register tile → `simdgroup_matrix` 8×8 MMA → tiles staged in threadgroup memory → fp16 inputs; MPS as the reference |
| Low precision, CPU | `gemm_avx512_{bf16,i8}`, `gemm_neon_{bf16,i8,f16,f16_acc}`, `gemm_sme_widening<…>` | the FP32 kernels with widening dot-product / outer-product instructions and k-group packing |
| Low precision, CUDA | `gemm_cuda_{bf16,fp16,fp16_acc16,i8,fp8}` | one `mma.sync` kernel template for five formats, as a cp.async kernel and as Level 8 |
| References | `gemm_amx` (Accelerate), `gemm_kleidiai`, `gemm_cuda_cublas*`, MPS | vendor libraries the hand-written kernels are measured against |

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
./build/benchmarks/bench_gemm_metal     # Apple GPU: Metal ladder and MPS
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
src/metal/                    Metal kernels (gemm.metal, compiled at run time) and their Objective-C++ host, stub elsewhere
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
| [src/gemm/README.md](src/gemm/README.md) | Map of the kernels: files, techniques, registers and intrinsics per family. Each kernel's design is in its own header. |
| [docs/kernel-design.md](docs/kernel-design.md) | File-by-file design: algorithms, layouts, edges, precision, benchmark methodology and supporting files |
| [docs/kernels/](docs/kernels/README.md) | Register-tile, cache-block and packed-B block sweeps of the AVX2, AVX-512 and NEON kernels on Zen 5 and M4, with the reasons for each default |
| [docs/benchmarks.md](docs/benchmarks.md) | All measurements and their analysis: Apple M4 Max (CPU and GPU), AMD Zen 5 and RTX 5080 |
| [docs/build.md](docs/build.md) | Build on Linux, macOS and Windows, CMake options, SME/AMX flags, what runs where, CI |
| [docs/cache-behavior.md](docs/cache-behavior.md) | Background: cache lines, reuse distance, working sets, memory-bound limits, GPU memory hierarchy |

---

## License

MIT. See [LICENSE](LICENSE).
