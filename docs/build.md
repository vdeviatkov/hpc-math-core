# Build Guide

Everything needed to configure, build, test and benchmark the suite on
Linux, macOS and Windows, plus the details behind the ISA-specific build
options and what runs (or is skipped) on each class of machine.

Contents

- [Prerequisites](#prerequisites)
- [Build & run](#build--run)
- [CMake options](#cmake-options)
- [Optimisation flags](#optimisation-flags-applied-automatically-in-release-mode)
- [SME and AMX build flags](#sme-and-amx-build-flags)
- [ISA availability & skipping](#isa-availability--skipping)
- [Filtering benchmarks](#filtering-benchmarks)
- [Continuous Integration](#continuous-integration)

---

## Prerequisites

| Tool | Minimum version | Notes |
|---|---|---|
| CMake | 3.25 | |
| C++ compiler | GCC 12 / Clang 16 / Apple Clang 15 / MSVC 19.35+ | C++20 required |
| CUDA toolkit | 11.8+ (optional) | GPU kernels only; CPU-only build works without it |

---

## Build & run

### Linux / macOS

```bash
# 1. Configure — Release enables -O3 -march=native -ffast-math -funroll-loops
#    CUDA is detected automatically. Pass nothing extra — CMake finds it.
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Build everything (CPU + CUDA if available, stubs otherwise)
cmake --build build --parallel

# 3. Run all tests (the CUDA tests skip without a GPU)
ctest --test-dir build --output-on-failure

# 4. Run the CPU and CUDA benchmarks (CUDA rows report SKIPPED without a GPU)
./build/benchmarks/bench_gemm
./build/benchmarks/bench_gemm_cuda
```

### Windows (MSVC)

```powershell
# 1. Configure (MSVC uses a multi-config generator — no CMAKE_BUILD_TYPE needed)
cmake -B build

# 2. Build in Release mode (--config Release is essential for optimised numbers)
cmake --build build --config Release --parallel

# 3. Run all tests
ctest --test-dir build --build-config Release --output-on-failure

# 4. Run the CPU and CUDA benchmarks
.\build\benchmarks\Release\bench_gemm.exe
.\build\benchmarks\Release\bench_gemm_cuda.exe
```

---

## CMake options

| Option | Default | Effect |
|---|---|---|
| `CMAKE_BUILD_TYPE` | — | Use `Release` for benchmark numbers (single-config generators only). |
| `HPC_MARCH` | `native` | Value passed to `-march=`. Set to `x86-64-v3` for a portable AVX2 build (used by CI); empty string disables the flag. Ignored by MSVC. |
| `HPC_ENABLE_AVX512` | `OFF` | Adds AVX-512 compile flags (`-mavx512f…` / MSVC `/arch:AVX512`). Not needed with the default `-march=native` on an AVX-512 CPU, which already enables AVX-512; use it for MSVC or a non-native `HPC_MARCH`. ⚠ SIGILL on CPUs without AVX-512. |
| `HPC_ENABLE_SME` | `OFF` | ARM SME2 kernels. Runs a compile-*and-execute* probe at configure time; supersedes `HPC_MARCH` with `-mcpu=apple-m4`. See below. |
| `HPC_ENABLE_AMX` | `ON` on Apple, `OFF` elsewhere | Apple AMX via Accelerate.framework (`cblas_sgemm`/`cblas_dgemm`). See below. |
| `HPC_ENABLE_KLEIDIAI` | `ON` when the SME probe passed, else `OFF` | Fetches Arm KleidiAI v1.31.0 (FetchContent) for its SME2 f32 reference GEMM. |
| `HPC_ENABLE_LTO` | `OFF` | Link-time optimisation for Release builds. |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | The GPU in the build machine. CUDA is detected automatically; without nvcc a CPU stub is built. If the GPU is newer than the toolkit can target, the build falls back to the newest PTX that toolkit can emit and warns — see below. |

### CUDA architecture selection

The default is `native` — the GPU in the build machine. Two cases need care:

**The GPU is newer than the CUDA toolkit.** nvcc refuses any architecture newer than itself, so `native` fails outright (e.g. CUDA 12.0, which predates Blackwell, on an RTX 50-series card: `Unsupported gpu architecture 'compute_120'`). The configure step probes for this and falls back to the newest *virtual* architecture the toolkit can emit, which the driver JIT-compiles forward onto the actual GPU. The kernels then run correctly, at the cost of a JIT pause on first launch — but do not benchmark in this configuration; install a toolkit new enough to target the card.

Measured on an RTX 5080, CUDA 12.0 (`90-virtual` PTX) against CUDA 13.2 (native `sm_120`), compute-only at N=4096: cuBLAS dense-FP16 ran at 21.7 vs 114.0 TFLOP/s — a 5.2× difference, because CUDA 12.0's cuBLAS has no Blackwell kernels to dispatch to. Hand-written kernel codegen moved far less, and not always in the same direction: `gemm_cuda_reg_tile` gained 18% from native compilation, while `gemm_cuda_wmma_pipelined` was 4.7% *faster* under the JIT (80.6 vs 76.6 TFLOP/s, σ = 0.003 ms over 5 repetitions; measured before its shared-memory padding, which brought it to ~100 TFLOP/s). So the toolkit version matters enormously for anything touching cuBLAS, and is roughly a wash for this project's own kernels. The configure step prints a warning when this happens, and always reports the final choice:

```
-- CUDA architectures: 90-virtual
```

**Re-configuring an existing build directory.** `CMAKE_CUDA_ARCHITECTURES` is a cache variable, so a value from an earlier configure persists and overrides the default. Use a fresh build directory (or `-DCMAKE_CUDA_ARCHITECTURES=native`) when changing toolkits or GPUs.

### Optimisation flags (applied automatically in Release mode)

| Flag | Effect |
|---|---|
| `-O3` | Full optimisation: auto-vectorisation, aggressive inlining, loop transforms |
| `-march=native` | Emit instructions for the exact build CPU. **Binary is not portable.** |
| `-ffast-math` | Permits FMA contraction, reassociation, reciprocal approximations. Assumes no NaN/Inf. ⚠ Not safe where strict IEEE-754 is required. |
| `-funroll-loops` | Unroll loops with statically-known trip counts; reduces branch overhead in the inner j-loop |
| `-fno-omit-frame-pointer` | Preserves call-stack unwinding for `perf`/Instruments profiling |

---

## SME and AMX build flags

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON   # SME2, Apple M4-class; AMX is on by default on Apple
```

`HPC_ENABLE_SME` is off by default because a wrong flag combination builds
a binary that dies with `SIGILL`. `-march=armv9-a+sme2` compiles on Apple
Silicon, but Clang sizes the ZA save buffer in each streaming function's
prologue with `CNTD`, a non-streaming SVE instruction, and Apple Silicon has
no non-streaming SVE. `-mcpu=apple-m4` avoids it, and combining it with the
default `-march=native` silently drops the SME features, so the option clears
`HPC_MARCH`. Since only running the code shows either failure, configure
compiles and runs a small SME program with each candidate flag
(`-mcpu=apple-m4`, `-mcpu=native`, `-march=armv9-a+sme2`) and keeps the
first that works. KleidiAI (`HPC_ENABLE_KLEIDIAI`) follows SME by default.

`HPC_ENABLE_AMX` is on by default on Apple platforms: `gemm_amx` only links
Accelerate.framework, which ships with every macOS SDK and needs no flags.
Accelerate may use several cores, so its numbers are not single-core unless
`VECLIB_MAXIMUM_THREADS=1` is set.

---

## ISA availability & skipping

Which kernel families exist in a build is decided once, at compile time, by the `HPC_HAS_*` macros in [`include/hpc/isa.hpp`](../include/hpc/isa.hpp) (each always defined to 0 or 1). Where an ISA is absent that family's `gemm_*` functions are declared `= delete`, so calling one is a compile-time error — there is no silent fallback to a slower kernel under the same name.

- `bench_gemm` still registers the family's benchmarks, which report
  `SKIPPED`; the kernel call sits in an `if constexpr` branch that is only
  instantiated where the family exists. The full kernel list therefore
  always appears in the output.
- The family's entries in the test table (`kKernels` in
  `tests/test_gemm.cpp`) are compiled only under its `HPC_HAS_*`, so the test
  list on a machine is exactly the set of kernels that run there.
- CUDA is the one run-time check, since a GPU belongs to the machine rather
  than the build: without a device (always so with the stub),
  `cuda_device_count()` returns 0 and the CUDA tests and benchmarks skip.

| Machine | Runs | Skipped |
|---|---|---|
| **Apple M-series** (default build — `HPC_ENABLE_AMX` defaults ON) | Scalar · NEON · Scalar-Pf · NEON-Pf · **AMX** | AVX2 · AVX-512 · SVE · SME · CUDA |
| **Apple M4 Max, `-DHPC_ENABLE_SME=ON`** (the run recorded in [benchmarks.md](benchmarks.md)) | Scalar · NEON · Scalar-Pf · NEON-Pf · **SME** · **AMX** · **KleidiAI** | AVX2 · AVX-512 · SVE · CUDA |
| Intel Mac (Accelerate present, but no Apple AMX coprocessor — dispatches to AVX/AVX-512 internally instead) | Scalar · AVX2 · AVX-512 · all Pf · **AMX**\* | NEON · SVE · SME · CUDA |
| Intel Skylake (Linux/Windows, AVX2, no AVX-512, no GPU) | Scalar · AVX2 · Scalar-Pf · AVX2-Pf | AVX-512 · NEON · SVE · SME · AMX · CUDA |
| Intel Skylake + NVIDIA GPU | Scalar · AVX2 · Scalar-Pf · AVX2-Pf · **CUDA** | AVX-512 · NEON · SVE · SME · AMX |
| Intel Sapphire Rapids + NVIDIA GPU | Scalar · AVX2 · AVX-512 · all Pf · **CUDA** | NEON · SVE · SME · AMX |
| AWS Graviton3 / Neoverse V1 | Scalar · NEON · SVE (256-bit) · all Pf variants | AVX2 · AVX-512 · SME · AMX · CUDA |
| Fujitsu A64FX | Scalar · NEON · SVE (512-bit) · all Pf variants | AVX2 · AVX-512 · SME · AMX · CUDA |

\* Expected to work, since Accelerate ships with every macOS SDK, but not run: no Intel Mac was available. The two Apple Silicon rows are verified.

---

## Filtering benchmarks

Benchmarks are named `<Kernel>/<f64|f32>/N=<N>`, and `--benchmark_filter`
takes a regular expression:

```bash
./build/benchmarks/bench_gemm --benchmark_filter="f32"                 # float only
./build/benchmarks/bench_gemm --benchmark_filter="^Neon"               # NEON family
./build/benchmarks/bench_gemm --benchmark_filter="BlockedPf"           # prefetch distance sweep
./build/benchmarks/bench_gemm --benchmark_filter="N=1024$"             # one size, every kernel
./build/benchmarks/bench_gemm --benchmark_filter="Sme"                 # the four SME kernels
./build/benchmarks/bench_gemm --benchmark_filter="BStride"             # SME B-stride experiment
./build/benchmarks/bench_gemm_cuda --benchmark_filter="N=4096"         # GPU, every kernel at one size
```

A full `bench_gemm` run includes the naive kernels at N=4096, which alone
take minutes per entry; filter for anything else.

---

## Continuous Integration

[`.github/workflows/build.yml`](../.github/workflows/build.yml) runs one job
per runner on every push and pull request: configure with Ninja in Release,
build, `ctest`, and upload the JUnit results. ccache and the FetchContent
sources are cached between runs.

| Job | Runner | Compiler | Kernel families tested |
|---|---|---|---|
| `linux-x86` | `ubuntu-24.04` | GCC 14, `HPC_MARCH=x86-64-v3` | scalar, AVX2, prefetch |
| `linux-arm` | `ubuntu-24.04-arm` | GCC 14 | scalar, NEON, prefetch (SVE if the runner has it) |
| `macos` | `macos-14` | Apple Clang | scalar, NEON, prefetch, AMX via Accelerate |

No hosted runner has a GPU, a CUDA toolkit or an SME-capable CPU, so every
job builds the CUDA stub and none compiles the SME or KleidiAI kernels; the
SME, AMX and CUDA numbers in [benchmarks.md](benchmarks.md) come from local
runs. `linux-x86` also checks that every CUDA benchmark reports SKIPPED, and
pins `x86-64-v3` because ccache may hand its objects to a job on a CPU with
different extensions.
