# hpc-math-core

[![CI — Build & Test](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml/badge.svg)](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml)

Progressive GEMM optimization and benchmarking across scalar C++, SIMD, CUDA, Tensor Cores, ARM SME and Apple AMX, focused on hardware-aware performance engineering for ML systems and low-latency compute.

Starting from readable scalar code, each step adds one layer of hardware exploitation — cache-friendly loop order, cache blocking, explicit SIMD (AVX2 · AVX-512 · NEON · SVE), software prefetch, CUDA (shared-memory tiling through Tensor Cores), and matrix-engine hardware (ARM SME2, Apple AMX). Every kernel is benchmarked with Google Benchmark, cross-validated against a scalar reference by a Google Test suite, and documented with memory-access diagrams and cache analysis.

---

## Headline results

Best kernel per family, single-threaded unless noted. Full output, speedup tables and analysis in [docs/benchmarks.md](docs/benchmarks.md).

| Family | Hardware | f32 peak | f64 peak |
|---|---|---|---|
| Scalar, cache-blocked (`gemm_blocked`) | Apple M4 Max | 86 GFLOP/s | 27 GFLOP/s |
| ARM NEON (`gemm_neon_blocked`) | Apple M4 Max | 97 GFLOP/s | 36 GFLOP/s |
| AVX2 (`gemm_avx2_blocked`) | Intel x86, MSVC | 146 GFLOP/s | 67 GFLOP/s |
| AVX-512 (`gemm_avx512_blocked`) | Intel x86, MSVC | 291 GFLOP/s | 139 GFLOP/s |
| ARM SME2 (`gemm_sme_*`) | Apple M4 Max | **386 GFLOP/s** | **116 GFLOP/s** |
| Apple AMX via Accelerate (`gemm_amx_*`) ¹ | Apple M4 Max | 3.3 TFLOP/s | 860 GFLOP/s |
| CUDA, FMA (`gemm_cuda_double_buf`) ² | NVIDIA RTX 5080 | 5.7 TFLOP/s | 0.7 TFLOP/s |
| CUDA, Tensor Cores (`gemm_cuda_wmma_pipelined`) ³ | NVIDIA RTX 5080 | **82 TFLOP/s** | — |
| cuBLAS dense-FP16 reference ³ | NVIDIA RTX 5080 | 118 TFLOP/s | — |

¹ Vendor BLAS, may use multiple cores — not a single-core comparison.
² End-to-end, including host↔device transfer, N=4096.
³ Compute-only (device-resident buffers), N=16384; fp16 inputs, fp32 accumulate. The hand-written kernel reaches ~70 % of cuBLAS on the same GPU.

---

## Optimisation ladder

| Level | Kernels | Technique |
|---|---|---|
| 0 | `gemm_naive`, `gemm_reordered` | i-j-k baseline → cache-friendly i-k-j loop order |
| 1 | `gemm_blocked` | L2 cache tiling (tile = 64) |
| 2 | `gemm_avx2_{naive,reordered,blocked}` | AVX2 FMA intrinsics, 4×16 f32 / 4×8 f64 register tile |
| 3 | `gemm_avx512_{naive,reordered,blocked}` | 512-bit ZMM register tile, embedded broadcast |
| 4 | `gemm_neon_*`, `gemm_sve_*` | ARM NEON Q-register tile; vector-length-agnostic SVE with predicated tails |
| 5 | `gemm_*_blocked_prefetch` | `__builtin_prefetch` on A rows, B k-tiles and C rows; distance sweep D ∈ {2, 4, 8, 16} |
| 6 | `gemm_cuda_{naive,reordered,blocked,reg_tile,double_buf,vectorized,wmma,mma_ldmatrix,hopper_wgmma}` | Shared-memory tiling → register tiling → `cp.async` double buffering → `float4` loads + swizzle → Tensor Cores via WMMA → raw `mma.sync`/`ldmatrix` → Hopper `wgmma` + TMA |
| 7 | `gemm_sme_{naive,reordered,blocked}` | ARM SME2 `FMOPA` outer-product accumulate into a ZA tile |
| 8 | `gemm_amx_*`, `gemm_cuda_wmma_pipelined` | Apple AMX through Accelerate BLAS; 128×128-tile, `cp.async`-pipelined WMMA kernel (~16× the Level 6 WMMA kernel) |

A family is compiled only where its ISA exists; elsewhere its kernels are declared `= delete`, so a wrong call is a compile-time error rather than a silently slower substitute. Benchmarks still list absent families as `SKIPPED`. Every family has been verified on real hardware except `gemm_sve_*` (no SVE machine was available) and `gemm_cuda_hopper_wgmma` (requires sm_90a).

---

## Quick start

Requires CMake ≥ 3.25 and a C++20 compiler (GCC 12 / Clang 16 / Apple Clang 15 / MSVC 19.35+). CUDA 11.8+ is optional; without it the CUDA targets build against a stub.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # -O3 -march=native -ffast-math; CUDA auto-detected
cmake --build build --parallel
ctest --test-dir build --output-on-failure  # 355 CPU + 54 CUDA tests

./build/benchmarks/bench_gemm                 --benchmark_filter="f32"
./build/benchmarks/cuda/bench_gemm_cuda       --benchmark_filter="N=4096"
```

Windows/MSVC uses a multi-config generator: drop `CMAKE_BUILD_TYPE` and pass `--config Release` to both `cmake --build` and `ctest`. See [docs/build.md](docs/build.md) for the full walkthrough.

### Build options

| Option | Default | Purpose |
|---|---|---|
| `HPC_MARCH` | `native` | `-march=` value; `x86-64-v3` for a portable AVX2 build |
| `HPC_ENABLE_AVX512` | `OFF` | AVX-512 kernels — enable only on AVX-512 hardware (SIGILL otherwise) |
| `HPC_ENABLE_SME` | `OFF` | ARM SME2 kernels; configure-time compile-and-run probe, Apple M4-class only |
| `HPC_ENABLE_AMX` | `ON` on Apple | Apple AMX via Accelerate.framework; no special flags, no SIGILL risk |
| `HPC_ENABLE_LTO` | `OFF` | Link-time optimisation |

Why SME is opt-in while AMX is on by default, which kernels run on which machine, and the CI matrix are covered in [docs/build.md](docs/build.md).

---

## Repository layout

```
include/hpc/matrix.hpp        Matrix<T>: 64-byte aligned, row-major
src/gemm/                     One header per kernel family (naive … amx), plus README.md
src/cuda/gemm_kernels.cu      CUDA kernels; gemm_kernels_stub.cpp for CPU-only builds
benchmarks/                   bench_gemm (CPU) and bench_gemm_cuda drivers
tests/                        Google Test suites for CPU and CUDA kernels
docs/                         Build guide, benchmark results, cache and approach notes
.github/workflows/build.yml   CI: Linux x86, Linux ARM, macOS, CUDA-stub
```

---

## Documentation

| Document | Contents |
|---|---|
| [docs/gemm-approaches.md](docs/gemm-approaches.md) | One-page cheat sheet: every family's cache technique, register width and key intrinsics |
| [src/gemm/README.md](src/gemm/README.md) | Per-kernel loop analysis with memory-access diagrams; CUDA level-by-level design notes |
| [docs/benchmarks.md](docs/benchmarks.md) | Full benchmark output, speedup tables and key observations for Apple M4 Max, Intel x86 and RTX 5080 |
| [docs/build.md](docs/build.md) | Platform build instructions, CMake options, SME/AMX flag rationale, ISA-skip matrix, CI |
| [docs/cache-behavior.md](docs/cache-behavior.md) | Cache lines, reuse distance, working-set analysis, roofline model, GPU memory hierarchy |

---

## License

MIT. See [LICENSE](LICENSE).
