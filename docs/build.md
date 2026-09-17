# Build Guide

Everything needed to configure, build, test and benchmark the suite on
Linux, macOS and Windows, plus the details behind the ISA-specific build
options and what runs (or is skipped) on each class of machine.

**Contents**

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

# 3. Run all tests (355 CPU + 54 CUDA; CUDA tests skip if no GPU)
cd build && ctest --output-on-failure

# 4. Run CPU benchmarks
./build/benchmarks/bench_gemm --benchmark_format=console

# 5. Run CUDA benchmarks (skips gracefully on CPU-only machines)
./build/benchmarks/cuda/bench_gemm_cuda --benchmark_format=console

# Filter examples
./build/benchmarks/cuda/bench_gemm_cuda --benchmark_filter="CudaBlocked"
./build/benchmarks/cuda/bench_gemm_cuda --benchmark_filter="f32"
./build/benchmarks/cuda/bench_gemm_cuda --benchmark_filter="N=1024"
```

### Windows (MSVC)

```powershell
# 1. Configure (MSVC uses a multi-config generator — no CMAKE_BUILD_TYPE needed)
cmake -B build

# 2. Build in Release mode (--config Release is essential for optimised numbers)
cmake --build build --config Release --parallel

# 3. Run all tests
cd build; ctest --build-config Release --output-on-failure

# 4. Run CPU benchmarks
.\build\benchmarks\Release\bench_gemm.exe --benchmark_format=console

# 5. Run CUDA benchmarks (skips gracefully if no GPU)
.\build\benchmarks\cuda\Release\bench_gemm_cuda.exe --benchmark_format=console

# Filter examples
.\build\benchmarks\Release\bench_gemm.exe --benchmark_filter="f32"
.\build\benchmarks\Release\bench_gemm.exe --benchmark_filter="Blocked"
.\build\benchmarks\Release\bench_gemm.exe --benchmark_filter="N=1024"
```

---
---

## CMake options

| Option | Default | Effect |
|---|---|---|
| `CMAKE_BUILD_TYPE` | — | Use `Release` for benchmark numbers (single-config generators only). |
| `HPC_MARCH` | `native` | Value passed to `-march=`. Set to `x86-64-v3` for a portable AVX2 build (used by CI); empty string disables the flag. Ignored by MSVC. |
| `HPC_ENABLE_AVX512` | `OFF` | Adds AVX-512 compile flags (`-mavx512f…` / MSVC `/arch:AVX512`). ⚠ SIGILL on CPUs without AVX-512. |
| `HPC_ENABLE_SME` | `OFF` | ARM SME2 kernels. Runs a compile-*and-execute* probe at configure time; supersedes `HPC_MARCH` with `-mcpu=apple-m4`. See below. |
| `HPC_ENABLE_AMX` | `ON` on Apple, `OFF` elsewhere | Apple AMX via Accelerate.framework (`cblas_sgemm`/`cblas_dgemm`). See below. |
| `HPC_ENABLE_LTO` | `OFF` | Link-time optimisation for Release builds. |
| `CMAKE_CUDA_ARCHITECTURES` | CMake default | Pass `native` to target the GPU in the build machine. CUDA is detected automatically; without nvcc a CPU stub is built. |

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

`gemm_sme_*` and `gemm_amx_*` are both **matrix-engine** kernels (whole-tile
outer-product / vendor-BLAS-dispatched compute, not wider SIMD FMA — see
[src/gemm/README.md](../src/gemm/README.md) for the architectural explanation),
but they need very different build handling:

```bash
# ARM SME2 (Apple M4 / M4 Pro / M4 Max only, as of this writing) — opt-in,
# because a wrong flag combination here is a genuine SIGILL risk.
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON
cmake --build build --parallel
./build/benchmarks/bench_gemm --benchmark_filter="Sme"

# Apple AMX via Accelerate.framework — ON by default on Apple platforms,
# since it only links a standard system framework (no SIGILL risk at all).
cmake -B build -DCMAKE_BUILD_TYPE=Release   # HPC_ENABLE_AMX defaults to ON here
cmake --build build --parallel
./build/benchmarks/bench_gemm --benchmark_filter="Amx"
```

**`HPC_ENABLE_SME`** (default OFF) runs a *compile-and-execute* probe at
configure time (not just a compile check) because a compile-only check is
provably insufficient here: `-march=armv9-a+sme2` compiles cleanly on Apple
Silicon but the resulting binary `SIGILL`s at runtime on the very first
non-streaming-SVE instruction Clang emits in the function prologue — Apple
Silicon has no non-streaming SVE unit at all, only Streaming SVE via SME
(`-mcpu=apple-m4` avoids this). The probe also discovered that combining
the default `-march=native` with `-mcpu=apple-m4` silently drops the SME
target features altogether, so `HPC_ENABLE_SME=ON` clears `HPC_MARCH` in
favour of the verified `-mcpu=` flag. **Verified end-to-end** on an Apple
M4 Max — see [benchmarks.md](benchmarks.md). Full writeup:
[src/gemm/sme.hpp](../src/gemm/sme.hpp).

**`HPC_ENABLE_AMX`** (default ON on Apple platforms) targets Apple's own
AMX coprocessor — architecturally unrelated to Intel's AMX despite sharing
an acronym, and not something Apple exposes as a public instruction set the
way ARM SME is. The only Apple-sanctioned way to reach it is
**Accelerate.framework**'s BLAS (`cblas_sgemm`/`cblas_dgemm`), which Apple's
own performance guidance points to for matrix math and which is understood
to dispatch to AMX blocks internally. Because this only requires linking a
standard framework shipped in every macOS SDK — no special compiler flags,
no CPUID probing, no OS permission handshake, no SIGILL risk — it carries
none of the fragility that keeps `HPC_ENABLE_SME` opt-in, so it defaults ON
wherever `APPLE` is true. **Verified end-to-end** on an Apple M4 Max: up to
**3.3 TFLOP/s f32** — see [benchmarks.md](benchmarks.md). Full writeup:
[src/gemm/amx.hpp](../src/gemm/amx.hpp).

Note that `gemm_amx_naive`, `_reordered`, and `_blocked` are intentionally
identical wrappers around the same Accelerate call — the vendor BLAS
exposes no algorithm-staging knob to reorder or block from the caller's
side, so unlike every other family in this repo there is only one real
implementation here (see [src/gemm/amx.hpp](../src/gemm/amx.hpp) for why the
three names still exist: benchmark/test naming symmetry with the rest of
the suite). Accelerate's BLAS may also use multiple CPU cores internally
for large matrices — unlike every other, strictly single-threaded, CPU
kernel in this repo — so treat its numbers as "best vendor-library
throughput on this machine", not an apples-to-apples single-core
comparison against `gemm_sme_*`/`gemm_avx512_*`/`gemm_neon_*`.

---

## ISA availability & skipping

Which kernel families exist in a build is decided once, at compile time, by the `HPC_HAS_*` macros in [`include/hpc/isa.hpp`](../include/hpc/isa.hpp) (each always defined to 0 or 1). Where an ISA is absent that family's `gemm_*` functions are declared `= delete`, so calling one is a compile-time error — **there is no silent fallback** to a slower kernel under the same name.

- **Benchmarks** report the family as `SKIPPED` without instantiating it:

  ```cpp
  run_gemm<N, T, kHaveNeon>(state, kNoNeon,
      [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_neon_blocked(A, B, C); });
  ```

  The name still appears in the output, so you always see the full kernel catalogue and know exactly which paths ran.
- **Tests** for the family are not compiled (`#if HPC_HAS_NEON … #endif`), so the test count reported on a machine is exactly the set of kernels that ran there.
- **CUDA** is the one runtime check — GPU presence is a property of the machine, not the build:

  ```cpp
  if (hpc::gemm::cuda_device_count() == 0) {
      state.SkipWithMessage("No CUDA device available");
      return;
  }
  ```

| Machine | Runs | Skipped |
|---|---|---|
| **Apple M-series** (default build — `HPC_ENABLE_AMX` defaults ON) | Scalar · NEON · Scalar-Pf · NEON-Pf · **AMX** | AVX2 · AVX-512 · SVE · SME · CUDA |
| **Apple M4 Max, `-DHPC_ENABLE_SME=ON`** (the run recorded in [benchmarks.md](benchmarks.md)) | Scalar · NEON · Scalar-Pf · NEON-Pf · **SME** · **AMX** | AVX2 · AVX-512 · SVE · CUDA |
| Intel Mac (Accelerate present, but no Apple AMX coprocessor — dispatches to AVX/AVX-512 internally instead) | Scalar · AVX2 · AVX-512 · all Pf · **AMX**\* | NEON · SVE · SME · CUDA |
| Intel Skylake (Linux/Windows, AVX2, no AVX-512, no GPU) | Scalar · AVX2 · Scalar-Pf · AVX2-Pf | AVX-512 · NEON · SVE · SME · AMX · CUDA |
| Intel Skylake + NVIDIA GPU | Scalar · AVX2 · Scalar-Pf · AVX2-Pf · **CUDA** | AVX-512 · NEON · SVE · SME · AMX |
| Intel Sapphire Rapids + NVIDIA GPU | Scalar · AVX2 · AVX-512 · all Pf · **CUDA** | NEON · SVE · SME · AMX |
| AWS Graviton3 / Neoverse V1 | Scalar · NEON · SVE (256-bit) · all Pf variants | AVX2 · AVX-512 · SME · AMX · CUDA |
| Fujitsu A64FX | Scalar · NEON · SVE (512-bit) · all Pf variants | AVX2 · AVX-512 · SME · AMX · CUDA |

\* Not run — no Intel Mac was available to this project; Accelerate.framework itself is a normal part of every macOS SDK regardless of CPU vendor, so this is expected to work, just unverified here. Everything on Apple Silicon (both rows above it) **is** verified — see [§ SME and AMX build flags](#sme-and-amx-build-flags).

---

## Filtering benchmarks

```bash
./build/benchmarks/bench_gemm --benchmark_filter="f32"          # float only
./build/benchmarks/bench_gemm --benchmark_filter="Neon"         # NEON family
./build/benchmarks/bench_gemm --benchmark_filter="BlockedPf"    # all prefetch kernels
./build/benchmarks/bench_gemm --benchmark_filter="N=1024"       # one size across all kernels
./build/benchmarks/bench_gemm --benchmark_filter="NeonBlockedPf.*f32"  # NEON prefetch, float
```

---

## Continuous Integration

| Job | Runner | Compiler | What runs |
|---|---|---|---|
| **`build-linux-x86`** | `ubuntu-24.04` | GCC 14 | All CPU tests (scalar, AVX2, prefetch); CUDA/NEON/SVE/SME/AMX skipped |
| **`build-linux-arm`** | `ubuntu-24.04-arm` | GCC 14 | All CPU tests (scalar, NEON, prefetch); AVX2/AVX-512/CUDA/SME/AMX skipped |
| **`build-macos`** | `macos-14` (Apple M) | Apple Clang | All CPU tests (scalar, NEON, prefetch, **AMX**); AVX2/AVX-512/CUDA/SME skipped (`HPC_ENABLE_SME` defaults OFF; `HPC_ENABLE_AMX` defaults ON on Apple) |
| **`build-cuda-stub`** | `ubuntu-24.04` | GCC 14 (no nvcc) | CMake finds no nvcc → stub library; CUDA bench + tests built and run; every CUDA row prints `SKIPPED` |

None of the CI runners above pass `-DHPC_ENABLE_SME=ON` — GitHub-hosted runners have no Apple M4-class hardware, so it stays at its default OFF and `gemm_sme_*` is not compiled there. `build-macos` DOES exercise real `gemm_amx_*` (Accelerate.framework ships in the `macos-14` runner's SDK, and `HPC_ENABLE_AMX` defaults ON there), so the AMX dispatch — though not the specific numbers in [benchmarks.md](benchmarks.md), which come from a local M4 Max run — is continuously verified. The SME and AMX benchmark numbers in [benchmarks.md](benchmarks.md) both come from a manual local run on Apple M4 Max hardware.

Each job restores **ccache** and the **FetchContent cache** (`build/_deps`), configures with `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release`, builds in parallel, and uploads JUnit XML from `ctest --output-junit`.

The **`build-cuda-stub`** job has no CUDA toolkit installed — `check_language(CUDA)` falls back to `gemm_kernels_stub.cpp`. The CUDA binaries build and link cleanly; every entry prints `SKIPPED: 'No CUDA device available'` at runtime.

---
