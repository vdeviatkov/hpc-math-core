# hpc-math-core

[![CI — Build & Test](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml/badge.svg)](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml)

Matrix multiplication (GEMM) optimised step by step on CPUs and GPUs: scalar C++,
AVX2, AVX-512, NEON, ARM SME2, CUDA (SIMT and Tensor Cores) and Metal, in
FP32/FP64 and in bf16, fp16, INT8 and FP8.

Each kernel is one technique away from the previous one: loop order, cache
blocking, register tiling, packing, matrix engines, asynchronous copies. Every
kernel is tested against a reference with Google Test and measured with Google
Benchmark, and every published number links to its raw data.

## Results

Best FP32 kernel per family at a large size (N = 4096 on the CPU, 8192 on the
GPUs), as throughput and as a share of the measured hardware peak. Vendor
libraries measured on the same hardware are listed under the kernel they are
compared with, in *italics*:

| Family | Kernel | Hardware | Throughput | % of peak |
|---|---|---|--:|--:|
| ARM NEON, packed B | `gemm_simd_packed<Q>` | Apple M4 Max, 1 core | 121 GFLOP/s | 84% |
| AVX2, packed B | `gemm_simd_packed<Ymm>` | AMD Zen 5, 1 core | 167 GFLOP/s | 91% |
| AVX-512, packed B | `gemm_simd_packed<Zmm>` | AMD Zen 5, 1 core | 318 GFLOP/s | 87% |
| ARM SME2, one unit | `gemm_sme` | Apple M4 Max, 1 core | 1.60 TFLOP/s | 82% |
| *Reference* | *Apple Accelerate framework, BLAS (`cblas_sgemm`)* | *Apple M4 Max, 1 core* | *1.60 TFLOP/s* | *82%* |
| *Reference* | *Arm KleidiAI* | *Apple M4 Max, 1 core* | *1.10 TFLOP/s* | *56%* |
| ARM SME2, both units | `gemm_sme_mt` (8 threads) | Apple M4 Max | 3.49 TFLOP/s | 89% |
| *Reference* | *Apple Accelerate framework, BLAS (`cblas_sgemm`)* | *Apple M4 Max, all cores* | *3.16 TFLOP/s* | *80%* |
| CUDA Tensor Cores, fp16 inputs | `gemm_cuda_tma` | NVIDIA RTX 5080 | 117.6 TFLOP/s | 94% |
| *Reference* | *NVIDIA cuBLAS, FP16 GEMM* | *NVIDIA RTX 5080* | *118.3 TFLOP/s* | *95%* |
| Metal simdgroup matrices, fp16 inputs | `MetalSimdTgF16` | Apple M4 Max GPU | 13.5 TFLOP/s | 91% |
| *Reference* | *Apple Metal Performance Shaders (`MPSMatrixMultiplication`), FP16* | *Apple M4 Max GPU* | *13.8 TFLOP/s* | *93%* |

The SIMD rows have no reference: no BLAS library was measured on the NEON or
AVX units (Apple's Accelerate framework runs on the M4's matrix unit). In low precision,
INT8 reaches 1,038 GOP/s on Zen 5 (AVX-512), 3.73 TOP/s on the M4's SME2 unit
and 408 TOP/s on the RTX 5080.

Full tables, peak derivations, every size and FP64:
[docs/benchmarks.md](docs/benchmarks.md#summary-efficiency-against-peak).

## Articles

A five-part series built on this code, on the
[web](https://vdeviatkov.github.io/hpc-math-core/) or as Markdown:

| # | Article | Key result |
|--:|---|---|
| 1 | [SIMD on Apple M4 and AMD Zen 5](https://vdeviatkov.github.io/hpc-math-core/articles/simd-gemm.html) ([md](docs/articles/simd-gemm.md)) | 0.4 → 328 GFLOP/s FP32 on one Zen 5 core, 91% of peak |
| 2 | [Apple M4's SME2 unit](https://vdeviatkov.github.io/hpc-math-core/articles/sme2-gemm.html) ([md](docs/articles/sme2-gemm.md)) | 475 → 1,645 GFLOP/s on one matrix unit, 91–100% of Apple Accelerate |
| 3 | [A power-of-two stride and SME2 on M4](https://vdeviatkov.github.io/hpc-math-core/articles/sme2-cache-associativity.html) ([md](docs/articles/sme2-cache-associativity.md)) | L2 set conflicts cost two thirds of throughput; packing B fixes it |
| 4 | [An RTX 5080](https://vdeviatkov.github.io/hpc-math-core/articles/cuda-gemm.html) ([md](docs/articles/cuda-gemm.md)) | Nine CUDA kernels, 3.7 → 120 TFLOP/s, matching NVIDIA cuBLAS |
| 5 | [bf16, fp16, INT8 and FP8](https://vdeviatkov.github.io/hpc-math-core/articles/lowp-gemm.html) ([md](docs/articles/lowp-gemm.md)) | From 0.53× to 8.3× FP32 speed, with the accuracy each format costs |

## Optimisation ladder

| Family | Steps |
|---|---|
| Scalar | naive i-j-k → cache-friendly i-k-j → cache blocking |
| SIMD (AVX2, AVX-512, NEON) | intrinsics → blocking + register-tiled micro-kernel → GotoBLAS order with packed B; optional software prefetch |
| ARM SME2 | one ZA tile → all four ZA tiles → cache blocking → packed B → both P-cluster units |
| CUDA (Levels 0–8) | shared-memory tiles → register tiles → `cp.async` double buffering → vector loads + swizzle → WMMA → `ldmatrix`/`mma.sync` → pipelined Tensor Cores → TMA with warp specialisation |
| Metal | threadgroup tiles → register tiles → `simdgroup_matrix` → threadgroup staging → fp16 inputs |
| Low precision | widening dot-product and outer-product instructions on the CPU; one `mma.sync` template for five formats on CUDA |
| References | Apple Accelerate framework, Arm KleidiAI, NVIDIA cuBLAS, Apple Metal Performance Shaders (MPS) |

A family is compiled only where its ISA exists; elsewhere its kernels are
`= delete` and the benchmarks report them as `SKIPPED`. The kernel map is in
[src/gemm/README.md](src/gemm/README.md).

## Quick start

Requires CMake ≥ 3.25 and a C++20 compiler (tested with GCC 13.3 / 14 on Linux
and Apple Clang 17 on macOS). The CUDA toolkit (13.2 / 13.4) is optional;
without it the CUDA targets build against a stub and skip.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

./build/benchmarks/bench_gemm --benchmark_filter="f32"   # CPU
./build/benchmarks/bench_gemm_cuda                       # NVIDIA GPU
./build/benchmarks/bench_gemm_metal                      # Apple GPU
```

CMake options, per-machine notes and the other benchmarks:
[docs/build.md](docs/build.md).

## Repository layout

```
include/hpc/              Matrix<T>, ISA detection, bf16/fp16/FP8 types
src/gemm/                 One header per kernel family; README.md maps them
src/cuda/, src/metal/     GPU kernels and host code, with stubs for other platforms
tests/                    Every kernel against a reference
benchmarks/               Google Benchmark drivers; experiments/ holds the microbenchmarks
docs/                     Benchmarks, build guide, kernel design, articles, raw experiment data
```

## Documentation

| Document | Contents |
|---|---|
| [docs/benchmarks.md](docs/benchmarks.md) | All measurements and their analysis |
| [docs/kernel-design.md](docs/kernel-design.md) | File-by-file design: algorithms, layouts, edge handling, precision |
| [docs/kernels/](docs/kernels/README.md) | Tile and cache-block sweeps, and the reason for each default |
| [docs/cache-behavior.md](docs/cache-behavior.md) | Memory-hierarchy background for CPUs and GPUs |
| [docs/build.md](docs/build.md) | Build options, what runs where, CI |

## License

MIT. See [LICENSE](LICENSE).
