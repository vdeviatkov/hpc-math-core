# Build Guide

Everything needed to configure, build, test and benchmark the suite on
Linux and macOS, plus the details behind the ISA-specific build options
and what runs (or is skipped) on the machines it has been built on.

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

CMake 3.25 or newer (`cmake_minimum_required`) and a C++20 compiler. The
toolchains the suite has been built and tested with:

| Machine | Compiler | CUDA |
|---|---|---|
| Apple M4 Max, macOS 26 | Apple Clang 17 | — |
| AMD Ryzen 9 9950X + RTX 5080, Ubuntu 24.04 | GCC 13.3; Clang 18.1 (CPU benchmarks only) | 13.2 and 13.4 |
| CI: `ubuntu-24.04` x86-64 and arm64, `macos-14` | GCC 14; Apple Clang | — (stub) |

The CUDA toolkit is optional: without it a CPU-only stub is built. Level 8
uses the CUDA 12 tensor-map API (`cuTensorMapEncodeTiled`).

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
./build/benchmarks/bench_gemm_lowp        # bf16 / fp16 / INT8 kernels of this CPU
./build/benchmarks/bench_gemm_lowp_cuda   # bf16 / fp16 / INT8 / FP8 on the GPU
./build/benchmarks/bench_gemm_metal       # Apple GPU (SKIPPED elsewhere)
```

The CMake files carry MSVC flags (`/arch:AVX2`, `/arch:AVX512`), but no
Windows build has been run or tested.

---

## CMake options

| Option | Default | Effect |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` for single-config generators | Use `Release` for benchmark numbers (single-config generators only). |
| `HPC_MARCH` | `native` | Value passed to `-march=`. Set to `x86-64-v3` for a portable AVX2 build (used by CI); empty string disables the flag. Ignored by MSVC. |
| `HPC_ENABLE_AVX512` | `OFF` | Adds AVX-512 compile flags (`-mavx512f…` / MSVC `/arch:AVX512`). Not needed with the default `-march=native` on an AVX-512 CPU, which already enables AVX-512; use it for MSVC or a non-native `HPC_MARCH`. ⚠ SIGILL on CPUs without AVX-512. |
| `HPC_ENABLE_SME` | `OFF` | ARM SME2 kernels. Runs a compile-*and-execute* probe for SME2 loads and FP32/FP64 FMOPA at configure time; when an `-mcpu` flag wins (`-mcpu=apple-m4` on Apple Silicon), it replaces `HPC_MARCH`. See below. |
| `HPC_ENABLE_AMX` | `ON` on Apple, `OFF` elsewhere | Apple AMX via Accelerate.framework (`cblas_sgemm`/`cblas_dgemm`). See below. |
| `HPC_ENABLE_KLEIDIAI` | `ON` when the SME probe passed, else `OFF` | Fetches Arm KleidiAI v1.31.0 (FetchContent) for its SME2 f32 reference GEMM. |
| `HPC_ENABLE_METAL` | `ON` on Apple, `OFF` elsewhere | Metal GEMM kernels on the Apple GPU (`bench_gemm_metal`, `hpc_tests_metal`). The shaders in `src/metal/gemm.metal` are embedded into the binary at configure time and compiled when first used, so no offline Metal toolchain is needed. Elsewhere a stub is built and the Metal rows skip. |
| `HPC_ENABLE_LTO` | `OFF` | Link-time optimisation for Release builds. |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | The GPU in the build machine. CUDA is detected automatically; without nvcc a CPU stub is built. If the GPU is newer than the toolkit can target, the build falls back to the newest PTX that toolkit can emit and warns — see below. |

The optional CPU experiments use GNU-style inline assembly and are omitted
under MSVC. SME needs a compiler with the required ACLE SME2 intrinsics
(the recorded build uses Apple Clang 17); the general C++20 minimum does
not guarantee support for every optional ISA.

### CUDA architecture selection

The default is `native` — the GPU in the build machine. A visible GPU is
needed for native detection. For a headless build, specify an architecture
explicitly, such as `-DCMAKE_CUDA_ARCHITECTURES=80`. This source uses WMMA
and requires targets at least sm_70; include an sm_80+ target for the raw
`mma.sync.m16n8k16` path and the low-precision kernels (FP8 needs sm_89+,
Level 8's TMA sm_90+). Runtime capability checks do not supply code
excluded by the compilation target.

**Architecture-specific targets.** On an RTX 50-series GPU,
`-DCMAKE_CUDA_ARCHITECTURES=120a` (tested with CMake 3.28) lets the FP8 kernels use
the block-scaled `mma` that runs at full rate with an FP32 accumulator;
with plain `120`/`native` they use the half-rate sm_89 instruction.
Everything else is identical. `cuda_lowp_fp8_block_scaled()` reports which
one a build has.

Two cases need care:

**The GPU is newer than the CUDA toolkit.** nvcc refuses any architecture newer than itself, so `native` fails outright (e.g. CUDA 12.0, which predates Blackwell, on an RTX 50-series card: `Unsupported gpu architecture 'compute_120'`). The configure step probes for this and falls back to the newest *virtual* architecture the toolkit can emit, which the driver JIT-compiles forward onto the actual GPU. The kernels then run correctly, at the cost of a JIT pause on first launch — but do not benchmark in this configuration; install a toolkit new enough to target the card.

Measured on an RTX 5080, CUDA 12.0 (`90-virtual` PTX) against CUDA 13.2 (native `sm_120`), compute-only at N=4096: cuBLAS dense-FP16 ran at 21.7 vs 114.0 TFLOP/s, because CUDA 12.0's cuBLAS has no Blackwell kernels to dispatch to. The configure step prints a warning when this happens, and always reports the final choice:

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
compiles and runs a small SME2/FP64 program with each candidate flag
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
- The GPUs are run-time checks, since a GPU belongs to the machine rather
  than the build: without a device (always so with the stubs),
  `cuda_device_count()` returns 0 / `metal_available()` false and the CUDA /
  Metal tests and benchmarks skip.

The builds that have been run:

| Build | Runs | Skipped |
|---|---|---|
| Apple M4 Max, default (`HPC_ENABLE_AMX` and `HPC_ENABLE_METAL` default ON) | Scalar · NEON · Scalar-Pf · NEON-Pf · **AMX** · **Metal** | AVX2 · AVX-512 · SME · CUDA |
| Apple M4 Max, `-DHPC_ENABLE_SME=ON` (the M4 runs in [benchmarks.md](benchmarks.md)) | Scalar · NEON · Scalar-Pf · NEON-Pf · **SME** (incl. `gemm_sme_mt`) · **AMX** · **KleidiAI** · **Metal** | AVX2 · AVX-512 · CUDA |
| AMD Ryzen 9 9950X + RTX 5080, `-DHPC_ENABLE_AVX512=ON` (native) | Scalar · AVX2 · AVX-512 · all Pf · **CUDA** | NEON · SME · AMX · Metal |
| AMD Ryzen 9 9950X, `-DHPC_MARCH=x86-64-v3` (16 YMM registers) | Scalar · AVX2 · Scalar-Pf · AVX2-Pf | AVX-512 · NEON · SME · AMX · Metal |
| CI `linux-x86` (`x86-64-v3`), `linux-arm`, `macos` | see [Continuous Integration](#continuous-integration) | CUDA (stub) |

---

## Filtering benchmarks

Benchmarks are named `<Kernel>/<f64|f32>/N=<N>`, and `--benchmark_filter`
takes a regular expression:

```bash
./build/benchmarks/bench_gemm --benchmark_filter="f32"                 # float only
./build/benchmarks/bench_gemm --benchmark_filter="^Neon"               # NEON family
./build/benchmarks/bench_gemm --benchmark_filter="BlockedPf"           # prefetch distance sweep
./build/benchmarks/bench_gemm --benchmark_filter="N=1024$"             # one size, every kernel
./build/benchmarks/bench_gemm --benchmark_filter="Sme"                 # the SME kernels (also SmeMt sweep)
./build/benchmarks/bench_gemm --benchmark_filter="^SmeMt/"             # several SME units
./build/benchmarks/bench_gemm --benchmark_filter="BStride"             # SME B-stride experiment
./build/benchmarks/bench_gemm_cuda --benchmark_filter="N=4096"         # GPU, every kernel at one size
./build/benchmarks/bench_gemm_metal --benchmark_filter="-Variant"      # Metal ladder, MPS and peaks, no tile variants
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
| `linux-arm` | `ubuntu-24.04-arm` | GCC 14 | scalar, NEON, prefetch |
| `macos` | `macos-14` | Apple Clang | scalar, NEON, prefetch, AMX via Accelerate |

These jobs do not enable SME. The configured hosted runners are expected
to have no CUDA toolkit/GPU, so every
job builds the CUDA stub and none compiles the SME or KleidiAI kernels; the
SME, AMX and CUDA numbers in [benchmarks.md](benchmarks.md) come from local
runs. `linux-x86` also checks that every CUDA benchmark reports SKIPPED, and
pins `x86-64-v3` because ccache may hand its objects to a job on a CPU with
different extensions.
