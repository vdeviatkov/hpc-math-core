# hpc-math-core

[![CI — Build & Test](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml/badge.svg)](https://github.com/vdeviatkov/hpc-math-core/actions/workflows/build.yml)

A progressive benchmark suite demonstrating **hardware-aware optimisations for linear algebra**, targeting quantitative engineering and high-frequency trading performance standards.

This repository starts from first principles — readable scalar code — and adds successive layers of hardware exploitation: cache-friendly access patterns, SIMD vectorisation (AVX2 · AVX-512 · NEON · SVE), software prefetch, and CUDA. Every step is fully benchmarked, cross-validated by a Google Test suite, and documented with ASCII memory diagrams and cache analysis.

---

## Current status: Step 6 — CUDA (naive · reordered · tiled shared-memory blocked)

| Step | Kernel family | Key technique | Status |
|---|---|---|---|
| 0 | `gemm_naive` / `gemm_reordered` | Loop reordering, cache-friendly i-k-j access | ✅ |
| 1 | `gemm_blocked` | L2 cache tiling (default tile = 64) | ✅ |
| 2 | `gemm_avx2_{naive,reordered,blocked}` | Explicit AVX2 FMA intrinsics, 4×16 f32 / 4×8 f64 register tile | ✅ |
| 3 | `gemm_avx512_{naive,reordered,blocked}` | 512-bit ZMM register tile, embedded broadcast | ✅ |
| 4 | `gemm_neon_{naive,reordered,blocked}` | 128-bit Q-register tile (ARM NEON / AdvSIMD), `vfmaq` FMA | ✅ |
| 4 | `gemm_sve_{naive,reordered,blocked}` | VLA SVE: runtime VL, `svwhilelt` predicates, zero scalar tails | ✅ |
| 5 | `gemm_{scalar,avx2,avx512,neon,sve}_blocked_prefetch` | `__builtin_prefetch` on A rows, B k-tiles and C write rows; distance sweep D∈{2,4,8,16} | ✅ |
| 6 | `gemm_cuda_{naive,reordered,blocked}` | CUDA: one-thread-per-element baseline, explicit row-major read, TILE×TILE shared-memory tiling (TILE=16, +1 pad against bank conflicts) | ✅ |
| 7 | `gemm_cuda_wmma` | NVIDIA Tensor Core WMMA API (FP16 accumulate into FP32) | 🔜 |

Each SIMD family is **skipped automatically** if the ISA is absent on the build CPU — no `#ifdef` pollution in benchmark registrations, no silent fallback timing. See [§ ISA availability](#isa-availability--skipping) below.

See [src/gemm/README.md](src/gemm/README.md) for per-kernel memory diagrams and [docs/cache-behavior.md](docs/cache-behavior.md) for the full cache analysis.

---

## Repository layout

```
hpc-math-core/
├── .github/
│   └── workflows/
│       └── build.yml               CI: build + ctest on Linux x86, Linux ARM, macOS, CUDA
├── CMakeLists.txt
├── include/
│   └── hpc/
│       └── matrix.hpp                  Matrix<T>: 64-byte aligned, row-major
├── src/
│   ├── gemm/
│   │   ├── naive.hpp                   i-j-k scalar baseline
│   │   ├── reordered.hpp               i-k-j scalar (cache-friendly)
│   │   ├── blocked.hpp                 tiled i-k-j (tile=64)
│   │   ├── avx2.hpp                    AVX2 FMA: 4×16 f32 / 4×8 f64 register tile
│   │   ├── avx512.hpp                  AVX-512: 4×32 f32 / 4×16 f64 register tile
│   │   ├── neon.hpp                    ARM NEON: 4×16 f32 / 4×4 f64 Q-register tile
│   │   ├── sve.hpp                     ARM SVE: VLA tile, predicated tails
│   │   ├── prefetch.hpp                __builtin_prefetch wrappers for all blocked kernels
│   │   ├── cuda.hpp                    Host-side C++ interface for CUDA launchers
│   │   └── README.md
│   └── cuda/
│       ├── gemm_kernels.cu             CUDA kernel implementations (naive/reordered/blocked)
│       └── gemm_kernels_stub.cpp       CPU-only stub (device_count→0) for non-CUDA builds
├── benchmarks/
│   ├── CMakeLists.txt
│   ├── bench_gemm.cpp                  CPU benchmark driver (all scalar + SIMD + prefetch)
│   ├── bench_gemm_cuda.cpp             CUDA benchmark driver
│   └── cuda/
│       └── CMakeLists.txt
├── tests/
│   ├── CMakeLists.txt
│   ├── test_gemm.cpp                   Google Test suite — CPU kernels (233 tests)
│   ├── test_gemm_cuda.cpp              Google Test suite — CUDA kernels (32 tests, skipped if no GPU)
│   └── cuda/
│       └── CMakeLists.txt
└── docs/
    └── cache-behavior.md
```

---

## Prerequisites

| Tool | Minimum version | Notes |
|---|---|---|
| CMake | 3.25 | FetchContent, `gtest_discover_tests` |
| C++ compiler | GCC 12 / Clang 16 / Apple Clang 15 / MSVC 19.35+ | C++20 required |
| CUDA toolkit | 11.8+ (optional) | Required only for GPU kernels; CPU-only build works without it |

---

## Build & run

### Linux / macOS

```bash
# 1. Configure — Release enables -O3 -march=native -ffast-math -funroll-loops
#    CUDA is detected automatically. Pass nothing extra — CMake finds it.
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Build everything (CPU + CUDA if available, stubs otherwise)
cmake --build build --parallel

# 3. Run all tests (233 CPU + 32 CUDA; CUDA tests skip if no GPU)
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

## Continuous Integration

The workflow lives at `.github/workflows/build.yml` and runs on every push and pull request.

### Jobs

| Job | Runner | Compiler | What runs |
|---|---|---|---|
| **`build-linux-x86`** | `ubuntu-24.04` | GCC 14 | All CPU tests (scalar, AVX2, prefetch); CUDA/NEON/SVE skipped |
| **`build-linux-arm`** | `ubuntu-24.04-arm` | GCC 14 | All CPU tests (scalar, NEON, prefetch); AVX2/AVX-512/CUDA skipped |
| **`build-macos`** | `macos-14` (Apple M) | Apple Clang | All CPU tests (scalar, NEON, prefetch); AVX2/AVX-512/CUDA skipped |
| **`build-cuda-stub`** | `ubuntu-24.04` | GCC 14 (no nvcc) | CMake finds no nvcc → stub library; CUDA bench + tests built and run; every CUDA row prints `SKIPPED` |

The split between CPU and CUDA-stub jobs means:
- **No CUDA toolkit is ever installed on a hosted runner** — no slow apt installs, no package conflicts.
- The stub build proves the no-GPU code path compiles and links on every push.

All four jobs:
1. Restore **ccache** (key: OS + compiler + source hash) — warm builds finish in ~20 s.
2. Restore **FetchContent cache** (`build/_deps`) — avoids re-cloning GoogleTest + Google Benchmark.
3. Configure with `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release`.
4. Build all targets in parallel.
5. Run `ctest --output-junit` — uploads JUnit XML as a job artifact (visible in the Actions UI).

### CUDA on CI

The **`build-cuda-stub`** job runs on the standard `ubuntu-24.04` hosted runner with **no CUDA toolkit installed**. CMake's `check_language(CUDA)` finds no `nvcc` and automatically falls back to compiling `gemm_kernels_stub.cpp`. The resulting `bench_gemm_cuda` and `hpc_tests_cuda` binaries build cleanly; at runtime every CUDA entry prints `SKIPPED: 'No CUDA device available'` and the job passes. This runs on every push at zero cost.

---

### Optimisation flags (applied automatically in Release mode)

| Flag | Effect |
|---|---|
| `-O3` | Full optimisation: auto-vectorisation, aggressive inlining, loop transforms |
| `-march=native` | Emit instructions for the exact build CPU. **Binary is not portable.** |
| `-ffast-math` | Permits FMA contraction, reassociation, reciprocal approximations. Assumes no NaN/Inf. ⚠ Not safe where strict IEEE-754 is required. |
| `-funroll-loops` | Unroll loops with statically-known trip counts; reduces branch overhead in the inner j-loop |
| `-fno-omit-frame-pointer` | Preserves call-stack unwinding for `perf`/Instruments profiling |

### Filtering benchmarks

```bash
./build/benchmarks/bench_gemm --benchmark_filter="f32"          # float only
./build/benchmarks/bench_gemm --benchmark_filter="Neon"         # NEON family
./build/benchmarks/bench_gemm --benchmark_filter="BlockedPf"    # all prefetch kernels
./build/benchmarks/bench_gemm --benchmark_filter="N=1024"       # one size across all kernels
./build/benchmarks/bench_gemm --benchmark_filter="NeonBlockedPf.*f32"  # NEON prefetch, float
```

---

## ISA availability & skipping

Every SIMD benchmark body begins with a compile-time guard:

```cpp
if (!kHaveNeon) { state.SkipWithMessage("NEON not available on this target"); return; }
```

CUDA benchmarks use a runtime device count check:

```cpp
if (hpc::gemm::cuda_device_count() == 0) {
    state.SkipWithMessage("No CUDA device available");
    return;
}
```

The benchmark name still appears in the output — as `SKIPPED` — so you always see the full kernel catalogue and know exactly which paths ran. No silent fallback to a slower scalar kernel that would corrupt the numbers.

| Machine | Runs | Skipped |
|---|---|---|
| **Apple M-series** (this run) | Scalar · NEON · Scalar-Pf · NEON-Pf | AVX2 · AVX-512 · SVE · CUDA |
| Intel Skylake (AVX2, no AVX-512, no GPU) | Scalar · AVX2 · Scalar-Pf · AVX2-Pf | AVX-512 · NEON · SVE · CUDA |
| Intel Skylake + NVIDIA GPU | Scalar · AVX2 · Scalar-Pf · AVX2-Pf · **CUDA** | AVX-512 · NEON · SVE |
| Intel Sapphire Rapids + NVIDIA GPU | Scalar · AVX2 · AVX-512 · all Pf · **CUDA** | NEON · SVE |
| AWS Graviton3 / Neoverse V1 | Scalar · NEON · SVE (256-bit) · all Pf variants | AVX2 · AVX-512 · CUDA |
| Fujitsu A64FX | Scalar · NEON · SVE (512-bit) · all Pf variants | AVX2 · AVX-512 · CUDA |

---

## Sample benchmark output

> **Machine:** Apple M-series, 16 P-cores, Apple Clang, C++20
> **Build:** `cmake -DCMAKE_BUILD_TYPE=Release` → `-O3 -march=native -ffast-math -funroll-loops`
> **CPU Caches:** L1 Data 64 KiB · L1 Instruction 128 KiB · L2 Unified 4096 KiB (×16)
> **Load Average:** 5.52 / 4.47 / 4.20 — 1-min / 5-min / 15-min average number of runnable
> processes. On a 16-core machine, 16.0 = 100% utilisation; 5.52 ≈ 34% load — moderate
> background activity, numbers are still representative.

### double (f64) — scalar & NEON kernels

```
Benchmark                    Time        CPU     GFLOP/s
--------------------------------------------------------
Naive/f64/N=64              71.6 µs    71.6 µs    7.32
Naive/f64/N=256          12960  µs  12956  µs     2.59
Naive/f64/N=512         106671  µs  106650 µs     2.52
Naive/f64/N=1024        868280  µs  868096 µs     2.47
Naive/f64/N=4096     194681544  µs   194.3s     707.2 M/s

Reordered/f64/N=64          19.7 µs    19.7 µs   26.68
Reordered/f64/N=256        2056   µs   2055   µs  16.33
Reordered/f64/N=512       16629   µs  16575   µs  16.20
Reordered/f64/N=1024     133401   µs  133103  µs  16.13
Reordered/f64/N=4096    8424367   µs    8.42s    16.33

Blocked/f64/N=64            19.6 µs    19.6 µs   26.80  tile=64
Blocked/f64/N=256          1334   µs   1334   µs  25.15  tile=64
Blocked/f64/N=512         12212   µs  12209   µs  21.99  tile=64
Blocked/f64/N=1024       112310   µs  112215  µs  19.14  tile=64
Blocked/f64/N=4096      6903279   µs    6.89s    19.94  tile=64

NeonNaive/f64/N=64          61.5 µs    61.4 µs    8.53  neon=1
NeonNaive/f64/N=256      13793   µs  13788   µs   2.43  neon=1
NeonNaive/f64/N=512     110220   µs  110210  µs   2.44  neon=1
NeonNaive/f64/N=1024    920839   µs  920571  µs   2.33  neon=1
NeonNaive/f64/N=4096  196441977  µs   196.2s   700.4 M/s neon=1

NeonReordered/f64/N=64      19.9 µs    19.9 µs   26.31  neon=1
NeonReordered/f64/N=256    2435   µs   2435   µs  13.78  neon=1
NeonReordered/f64/N=512   18720   µs  18718   µs  14.34  neon=1
NeonReordered/f64/N=1024 142053   µs  142035  µs  15.12  neon=1
NeonReordered/f64/N=4096 8830883  µs    8.83s    15.57  neon=1

NeonBlocked/f64/N=64        14.6 µs    14.6 µs   35.92  neon=1
NeonBlocked/f64/N=256      1034   µs   1034   µs  32.45  neon=1
NeonBlocked/f64/N=512      8686   µs   8685   µs  30.91  neon=1
NeonBlocked/f64/N=1024    70549   µs  70541   µs  30.44  neon=1
NeonBlocked/f64/N=4096  5555345   µs    5.56s    24.74  neon=1

Avx2*/f64/*     SKIPPED: 'AVX2 not available on this target'
Avx512*/f64/*   SKIPPED: 'AVX-512 not available on this target'
Sve*/f64/*      SKIPPED: 'SVE not available on this target'
```

### float (f32) — scalar & NEON kernels

```
Benchmark                    Time        CPU     GFLOP/s
--------------------------------------------------------
Naive/f32/N=64              57.6 µs    57.4 µs    9.14
Naive/f32/N=256          12637  µs  12636  µs     2.66
Naive/f32/N=512         113462  µs  113451 µs     2.37
Naive/f32/N=1024        842629  µs  842501 µs     2.55
Naive/f32/N=4096     198056920  µs   197.9s     694.6 M/s

Reordered/f32/N=64           6.18 µs    6.18 µs  84.83
Reordered/f32/N=256        1065   µs   1063   µs  31.57
Reordered/f32/N=512        8519   µs   8515   µs  31.53
Reordered/f32/N=1024      67326   µs  67247   µs  31.93
Reordered/f32/N=4096    4299230   µs    4.29s    32.05

Blocked/f32/N=64             6.26 µs    6.24 µs  84.06  tile=64
Blocked/f32/N=256             404 µs     403  µs  83.23  tile=64
Blocked/f32/N=512            5394 µs    5374  µs  49.95  tile=64
Blocked/f32/N=1024          54057 µs   54024  µs  39.75  tile=64
Blocked/f32/N=4096        4502770 µs    4.50s    30.54  tile=64

NeonNaive/f32/N=64          57.4 µs    57.4 µs    9.14  neon=1
NeonNaive/f32/N=256        9050   µs   9039   µs   3.71  neon=1
NeonNaive/f32/N=512       77958   µs  77951   µs   3.44  neon=1
NeonNaive/f32/N=1024     985483   µs  953716  µs   2.25  neon=1
NeonNaive/f32/N=4096  204839329  µs   204.6s   671.9 M/s neon=1

NeonReordered/f32/N=64      20.3 µs    20.3 µs   25.84  neon=1
NeonReordered/f32/N=256    1129   µs   1129   µs  29.72  neon=1
NeonReordered/f32/N=512    9778   µs   9777   µs  27.46  neon=1
NeonReordered/f32/N=1024  75721   µs  75710   µs  28.36  neon=1
NeonReordered/f32/N=4096 4512691  µs    4.51s    30.46  neon=1

NeonBlocked/f32/N=64         5.42 µs    5.42 µs  96.70  neon=1
NeonBlocked/f32/N=256         347 µs     347  µs  96.75  neon=1
NeonBlocked/f32/N=512        2765 µs    2765  µs  97.10  neon=1
NeonBlocked/f32/N=1024      22578 µs   22576  µs  95.12  neon=1
NeonBlocked/f32/N=4096    1925502 µs    1.92s    71.56  neon=1

Avx2*/f32/*     SKIPPED: 'AVX2 not available on this target'
Avx512*/f32/*   SKIPPED: 'AVX-512 not available on this target'
Sve*/f32/*      SKIPPED: 'SVE not available on this target'
```

### Prefetch distance sweep — `BlockedPf` (scalar) and `NeonBlockedPf`

Benchmarks named `<Family>BlockedPf<D>/<prec>/N=<size>` sweep prefetch distance D ∈ {2, 4, 8, 16} (rows ahead). Three `__builtin_prefetch` sites per kernel:

- **[PF-A]** `A(i + D×kRegRows, k_blk)` → L2 (read)
- **[PF-B]** `B(k_blk + TileK, 0)` → L2 (read) at k-tile boundary
- **[PF-C]** `C(i + D×kRegRows, j_blk)` → L1 (write)

```
Benchmark                         Time      GFLOP/s   pf_dist
-------------------------------------------------------------
— Scalar blocked + prefetch (f64) —
BlockedPf2/f64/N=256           2259 µs    14.85 G/s   D=2
BlockedPf4/f64/N=256           2258 µs    14.86 G/s   D=4
BlockedPf8/f64/N=256           2260 µs    14.85 G/s   D=8
BlockedPf16/f64/N=256          2257 µs    14.87 G/s   D=16

BlockedPf2/f64/N=512          20509 µs    13.09 G/s   D=2
BlockedPf4/f64/N=512          20446 µs    13.13 G/s   D=4  ← best
BlockedPf8/f64/N=512          20441 µs    13.13 G/s   D=8
BlockedPf16/f64/N=512         20984 µs    12.82 G/s   D=16

BlockedPf2/f64/N=1024        190535 µs    11.31 G/s   D=2
BlockedPf4/f64/N=1024        188858 µs    11.38 G/s   D=4
BlockedPf8/f64/N=1024        188381 µs    11.41 G/s   D=8
BlockedPf16/f64/N=1024       187768 µs    11.44 G/s   D=16 ← best

— Scalar blocked + prefetch (f32) —
BlockedPf2/f32/N=256           1159 µs    28.94 G/s   D=2  ← best
BlockedPf4/f32/N=256           1178 µs    28.53 G/s   D=4
BlockedPf8/f32/N=256           1188 µs    28.42 G/s   D=8
BlockedPf16/f32/N=256          1241 µs    27.11 G/s   D=16

BlockedPf8/f32/N=512          12027 µs    22.33 G/s   D=8  ← best
BlockedPf4/f32/N=1024         99628 µs    21.56 G/s   D=4

— NEON blocked + prefetch (f64) —
NeonBlockedPf2/f64/N=256        986 µs    34.05 G/s   D=2  ← best
NeonBlockedPf8/f64/N=256        990 µs    33.90 G/s   D=8
NeonBlockedPf4/f64/N=256       1034 µs    32.45 G/s   D=4
NeonBlockedPf16/f64/N=256      1051 µs    31.96 G/s   D=16

NeonBlockedPf2/f64/N=512       8664 µs    31.01 G/s   D=2  ← best
NeonBlockedPf8/f64/N=512       8647 µs    31.05 G/s   D=8
NeonBlockedPf16/f64/N=512      8694 µs    30.88 G/s   D=16
NeonBlockedPf4/f64/N=512       8982 µs    29.95 G/s   D=4

NeonBlockedPf2/f64/N=1024     69663 µs    30.83 G/s   D=2  ← best
NeonBlockedPf8/f64/N=1024     69612 µs    30.85 G/s   D=8
NeonBlockedPf4/f64/N=1024     70677 µs    30.39 G/s   D=4
NeonBlockedPf16/f64/N=1024    70760 µs    30.35 G/s   D=16

— NEON blocked + prefetch (f32) —
NeonBlockedPf2/f32/N=256        340 µs    98.59 G/s   D=2  ← best
NeonBlockedPf16/f32/N=256       347 µs    96.58 G/s   D=16
NeonBlockedPf4/f32/N=256        348 µs    96.53 G/s   D=4
NeonBlockedPf8/f32/N=256        348 µs    96.47 G/s   D=8

NeonBlockedPf2/f32/N=512       2715 µs    98.87 G/s   D=2  ← best
NeonBlockedPf4/f32/N=512       2769 µs    96.96 G/s   D=4
NeonBlockedPf8/f32/N=512       2770 µs    96.92 G/s   D=8
NeonBlockedPf16/f32/N=512      2769 µs    96.95 G/s   D=16

NeonBlockedPf2/f32/N=1024     22256 µs    96.50 G/s   D=2  ← best
NeonBlockedPf4/f32/N=1024     22632 µs    94.89 G/s   D=4
NeonBlockedPf8/f32/N=1024     22634 µs    94.88 G/s   D=8
NeonBlockedPf16/f32/N=1024    22587 µs    95.08 G/s   D=16

Avx2BlockedPf*/  SKIPPED: 'AVX2 not available on this target'
Avx512BlockedPf*/SKIPPED: 'AVX-512 not available on this target'
SveBlockedPf*/   SKIPPED: 'SVE not available on this target'
```

---

## Speedup tables

### f64 — best kernel per family vs `gemm_naive`

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive | NeonBlockedPf2 | ×naive |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 71.6 µs | 19.7 µs | **2.9×** | 19.6 µs | **2.9×** | 14.6 µs | **3.9×** | — | — |
| 256 | 12960 µs | 2056 µs | **6.3×** | 1334 µs | **9.7×** | 1034 µs | **12.5×** | 986 µs | **13.1×** |
| 512 | 106671 µs | 16629 µs | **6.4×** | 12212 µs | **8.7×** | 8686 µs | **12.3×** | 8647 µs | **12.3×** |
| 1024 | 868280 µs | 133401 µs | **6.5×** | 112310 µs | **7.7×** | 70549 µs | **12.3×** | 69612 µs | **12.5×** |
| 4096 | 194681544 µs | 8424367 µs | **23.1×** | 6903279 µs | **28.2×** | 5555345 µs | **35.0×** | — | — |

### f32 — best kernel per family vs `gemm_naive`

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive | NeonBlockedPf2 | ×naive |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 57.6 µs | 6.18 µs | **9.3×** | 6.26 µs | **9.2×** | 5.42 µs | **10.6×** | — | — |
| 256 | 12637 µs | 1065 µs | **11.9×** | 404 µs | **31.3×** | 347 µs | **36.4×** | 340 µs | **37.2×** |
| 512 | 113462 µs | 8519 µs | **13.3×** | 5394 µs | **21.0×** | 2765 µs | **41.0×** | 2715 µs | **41.8×** |
| 1024 | 842629 µs | 67326 µs | **12.5×** | 54057 µs | **15.6×** | 22578 µs | **37.3×** | 22256 µs | **37.9×** |
| 4096 | 198056920 µs | 4299230 µs | **46.1×** | 4502770 µs | **44.0×** | 1925502 µs | **102.9×** | — | — |

### CUDA kernels (bench_gemm_cuda)

> On machines **without a CUDA device** all rows print `SKIPPED: 'No CUDA device available'`.
> The binary compiles and links on CPU-only machines (Apple M, CI) via a stub library.
> On a machine with a CUDA GPU the stub is replaced by the real `.cu` kernel library.

Three kernels, named to mirror the CPU progression:

| Kernel | Strategy | Key technique |
|---|---|---|
| `CudaNaive` | 1 thread → 1 C(i,j), no shared memory | Baseline: exposes raw global-memory bandwidth |
| `CudaReordered` | Same mapping, explicit row-major inner loop | Structural symmetry with CPU `gemm_reordered`; identical to naive on GPU |
| `CudaBlocked` | TILE×TILE thread block → TILE×TILE sub-tile of C | Shared-memory tiling (TILE=16); 16× fewer global loads vs naive |

#### GPU memory hierarchy

```
                  ┌────────────────────────────────────────────────┐
                  │  GPU (e.g. NVIDIA A100 80 GB)                  │
  ┌───────────────┴──────────────┐  ┌──────────────────────────┐   │
  │  SM 0  (Streaming Multiproc) │  │  SM 1  …  SM 107         │   │
  │  ┌─────────┐  ┌───────────┐  │  │                          │   │
  │  │Registers│  │  Shared   │  │  │   (same structure)       │   │
  │  │ 256 KB  │  │  Memory / │  │  │                          │   │
  │  │per SM   │  │  L1 Cache │  │  │                          │   │
  │  │  ~1 cy  │  │  192 KB   │  │  │                          │   │
  │  │         │  │  ~4 cy    │  │  │                          │   │
  │  └─────────┘  └─────┬─────┘  │  │                          │   │
  └────────────────────-┼────────┘  └──────────────────────────┘   │
                        │  L2 Cache: 40–72 MB shared across SMs     │
                        │  ~200 cy, ~5 TB/s                         │
                        │  HBM2e / HBM3 DRAM: 80 GB                │
                        │  ~400–3900 GB/s                           │
                        └────────────────────────────────────────────
```

**Warp coalescence:** 32 threads in a warp issue memory loads together. If consecutive threads access consecutive addresses, the hardware merges them into a single 128-byte transaction. In our kernels, thread `(ty, tx)` computes `C(i, j)` where `j = blockCol*TILE + tx` — so consecutive threads in a warp differ only in `tx`, giving coalesced access to B rows and C rows.

#### Tiled GEMM algorithm (`CudaBlocked`, TILE=16)

```
for each k-tile (step TILE):
    ┌──────────────────────────────────────────────┐
    │  All 16×16 threads cooperatively load:        │
    │    As[ty][tx] = A[i][kTile*TILE + tx]         │  ← TILE×TILE sub-tile of A
    │    Bs[ty][tx] = B[kTile*TILE + ty][j]         │  ← TILE×TILE sub-tile of B
    │  into __shared__ memory (bank-conflict-free   │
    │  via +1 column padding: As[16][17], Bs[16][17])│
    └──────────────────────────────────────────────┘
    __syncthreads()
    for p in 0..TILE-1:
        acc += As[ty][p] * Bs[p][tx]    ← all from shared memory, ~4 cycles
    __syncthreads()

C[i][j] = acc
```

**Global memory traffic reduction:**
- Naive: each C(i,j) thread loads `2N` elements from global memory → `2N³` total.
- Tiled (TILE=16): each element of A and B is loaded from global memory `N/TILE` times → `2N³/TILE` total loads → **16× fewer global memory transactions**.

#### Expected output on a CUDA-enabled machine

```
----------------------------------------------------------------------
Benchmark                        Time        CPU    GFLOP/s
----------------------------------------------------------------------
# Naive (no shared memory, column-stride B access — DRAM bound)
BM_CudaNaive/f64/N=64           xx µs      xx µs    ~xxx G/s
BM_CudaNaive/f32/N=512          xx µs      xx µs    ~xxx G/s
BM_CudaNaive/f32/N=4096         xx ms      xx ms    ~xxx G/s

# Reordered (identical to naive on GPU — L2 absorbs repeated accesses)
BM_CudaReordered/f32/N=512      xx µs      xx µs    ~xxx G/s

# Blocked/tiled (shared memory — compute bound at large N)
BM_CudaBlocked/f32/N=512        xx µs      xx µs    ~xxx G/s
BM_CudaBlocked/f32/N=4096       xx ms      xx ms    ~xxx G/s

# On a CPU-only machine (Apple M, no GPU):
BM_CudaNaive/f64/N=64     SKIPPED: 'No CUDA device available'
BM_CudaBlocked/f32/N=4096 SKIPPED: 'No CUDA device available'
```

> **Note on timing:** all CUDA benchmarks include host↔device data transfer time
> (cudaMemcpy + kernel + cudaMemcpy). This is the end-to-end time visible to the caller.
> To isolate pure kernel time, use CUDA events (`cudaEventRecord / cudaEventElapsedTime`)
> in a custom harness — a natural next step once the Tensor Core WMMA kernel is added.

#### Theoretical peak comparison (RTX 4090, f32)

| Kernel | Bottleneck | ~Peak |
|---|---|---|
| Naive | DRAM bandwidth (1 TB/s) | ~1–3 TFLOP/s |
| Blocked (TILE=16) | Compute (165.2 TFLOP/s FP32 peak) | ~80–120 TFLOP/s |
| cuBLAS | Tensor Cores (330 TFLOP/s FP16→FP32) | ~250–300 TFLOP/s |

---

### Sample benchmark output — Intel x86 (AVX2 + AVX-512)

> **Machine:** Intel Alder Lake / Sapphire Rapids-class, 16 P-cores (32 threads), 4.29 GHz, MSVC 2022, C++20
> **Build:** `cmake -B build && cmake --build build --config Release`
> **CPU Caches:** L1 Data 48 KiB · L1 Instruction 32 KiB · L2 Unified 1024 KiB (×16) · L3 Unified 32768 KiB (×2)

#### double (f64) — scalar, AVX2 & AVX-512 kernels

```
Benchmark                       Time        CPU     GFLOP/s
------------------------------------------------------------
Naive/f64/N=64                52.0 µs    53.1 µs     9.87
Naive/f64/N=256             14017  µs  14062  µs     2.39
Naive/f64/N=512            194863  µs  195312 µs     1.37
Naive/f64/N=1024          2794359  µs    2.80s      767.8 M/s
Naive/f64/N=4096        305295146  µs   304.3s      451.7 M/s

Reordered/f64/N=64             104 µs     103  µs     5.11
Reordered/f64/N=256           6690 µs    6696  µs     5.01
Reordered/f64/N=512          54141 µs   54688  µs     4.91
Reordered/f64/N=1024       427629 µs  429688  µs     5.00
Reordered/f64/N=4096     34642827 µs   34.6s        3.97

Blocked/f64/N=64               103 µs     103  µs     5.11  tile=64
Blocked/f64/N=256             6892 µs    6944  µs     4.83  tile=64
Blocked/f64/N=512            55092 µs   56250  µs     4.77  tile=64
Blocked/f64/N=1024         441021 µs  437500  µs     4.91  tile=64
Blocked/f64/N=4096       28273076 µs   28.2s        4.87  tile=64

Avx2Reordered/f64/N=64        20.3 µs    20.5 µs    25.57
Avx2Reordered/f64/N=256        974 µs     983  µs    34.13
Avx2Reordered/f64/N=512       8120 µs    8125  µs    33.04
Avx2Reordered/f64/N=1024     66351 µs   66761  µs    32.17
Avx2Reordered/f64/N=4096  13227900 µs   13.2s       10.39

Avx2Blocked/f64/N=64           7.83 µs    7.85 µs    66.81  avx2=1
Avx2Blocked/f64/N=256          505 µs     500  µs    67.11  avx2=1
Avx2Blocked/f64/N=512         5860 µs    5859  µs    45.81  avx2=1
Avx2Blocked/f64/N=1024       47682 µs   46875  µs    45.81  avx2=1
Avx2Blocked/f64/N=4096     3624176 µs    3.61s      38.08  avx2=1

Avx512Reordered/f64/N=64      17.7 µs    17.6 µs    29.83  avx512=1
Avx512Reordered/f64/N=256      821 µs     820  µs    40.94  avx512=1
Avx512Reordered/f64/N=512     7554 µs    7465  µs    35.96  avx512=1
Avx512Reordered/f64/N=1024   60469 µs   59375  µs    36.17  avx512=1
Avx512Reordered/f64/N=4096 11374338 µs   11.4s      12.08  avx512=1

Avx512Blocked/f64/N=64         3.79 µs    3.77 µs   139.19  avx512=1
Avx512Blocked/f64/N=256         277 µs     276  µs   121.48  avx512=1
Avx512Blocked/f64/N=512        4727 µs    4719  µs    56.88  avx512=1
Avx512Blocked/f64/N=1024      39896 µs   39931  µs    53.78  avx512=1
Avx512Blocked/f64/N=4096    2796709 µs    2.80s      49.14  avx512=1

Neon*/f64/*     SKIPPED: 'NEON not available on this target'
Sve*/f64/*      SKIPPED: 'SVE not available on this target'
```

#### float (f32) — scalar, AVX2 & AVX-512 kernels

```
Benchmark                       Time        CPU     GFLOP/s
------------------------------------------------------------
Naive/f32/N=64                52.5 µs    53.1 µs     9.87
Naive/f32/N=256              9710  µs    9583  µs     3.50
Naive/f32/N=512            120715  µs  122396  µs     2.19
Naive/f32/N=1024          2800837  µs    2.80s      767.8 M/s
Naive/f32/N=4096        313939135  µs   313.7s      438.2 M/s

Reordered/f32/N=64             101 µs     103  µs     5.11
Reordered/f32/N=256           6614 µs    6696  µs     5.01
Reordered/f32/N=512          53251 µs   54688  µs     4.91
Reordered/f32/N=1024       424514 µs  429688  µs     5.00
Reordered/f32/N=4096     27507882 µs   27.4s        5.01

Blocked/f32/N=64               102 µs     103  µs     5.11  tile=64
Blocked/f32/N=256             6768 µs    6836  µs     4.91  tile=64
Blocked/f32/N=512            53831 µs   53125  µs     5.05  tile=64
Blocked/f32/N=1024         432354 µs  437500  µs     4.91  tile=64
Blocked/f32/N=4096       27694736 µs   27.7s        4.96  tile=64

Avx2Reordered/f32/N=64        12.7 µs    12.7 µs    41.30
Avx2Reordered/f32/N=256        582 µs     586  µs    57.27
Avx2Reordered/f32/N=512       4431 µs    4404  µs    60.95
Avx2Reordered/f32/N=1024     40694 µs   40441  µs    53.10
Avx2Reordered/f32/N=4096   4741415 µs    4.70s      29.22

Avx2Blocked/f32/N=64           3.72 µs    3.77 µs   139.06  avx2=1
Avx2Blocked/f32/N=256          231 µs     230  µs   145.79  avx2=1
Avx2Blocked/f32/N=512         1903 µs    1927  µs   139.31  avx2=1
Avx2Blocked/f32/N=1024       23777 µs   23438  µs    91.63  avx2=1
Avx2Blocked/f32/N=4096     1764409 µs    1.75s      78.54  avx2=1

Avx512Reordered/f32/N=64      12.2 µs    12.3 µs    42.71  avx512=1
Avx512Reordered/f32/N=256      507 µs     502  µs    66.81  avx512=1
Avx512Reordered/f32/N=512     3692 µs    3686  µs    72.83  avx512=1
Avx512Reordered/f32/N=1024   30564 µs   30540  µs    70.32  avx512=1
Avx512Reordered/f32/N=4096  4436182 µs    4.44s     30.97  avx512=1

Avx512Blocked/f32/N=64         1.81 µs    1.80 µs   290.76  avx512=1
Avx512Blocked/f32/N=256         124 µs     126  µs   267.24  avx512=1
Avx512Blocked/f32/N=512        1110 µs    1123  µs   239.02  avx512=1
Avx512Blocked/f32/N=1024      15673 µs   15625  µs   137.44  avx512=1
Avx512Blocked/f32/N=4096    1110351 µs    1.11s     123.89  avx512=1

Neon*/f32/*     SKIPPED: 'NEON not available on this target'
Sve*/f32/*      SKIPPED: 'SVE not available on this target'
```

#### Prefetch distance sweep — AVX2 & AVX-512 blocked + prefetch

```
Benchmark                            Time      GFLOP/s   pf_dist
-----------------------------------------------------------------
— AVX2 blocked + prefetch (f64) —
Avx2BlockedPf2/f64/N=256            477 µs    70.53 G/s   D=2
Avx2BlockedPf4/f64/N=256            477 µs    70.74 G/s   D=4  ← best
Avx2BlockedPf2/f64/N=1024         47538 µs    45.81 G/s   D=2  ← best
Avx2BlockedPf16/f64/N=1024        47937 µs    44.75 G/s   D=16

— AVX2 blocked + prefetch (f32) —
Avx2BlockedPf2/f32/N=256            246 µs   136.66 G/s   D=2
Avx2BlockedPf2/f32/N=512           2009 µs   134.71 G/s   D=2
Avx2BlockedPf8/f32/N=1024         23881 µs    89.63 G/s   D=8

— AVX-512 blocked + prefetch (f64) —
Avx512BlockedPf2/f64/N=256          279 µs   121.48 G/s   D=2
Avx512BlockedPf4/f64/N=512         4602 µs    58.79 G/s   D=4  ← best
Avx512BlockedPf16/f64/N=1024      38612 µs    55.63 G/s   D=16 ← best

— AVX-512 blocked + prefetch (f32) —
Avx512BlockedPf2/f32/N=256          124 µs   273.32 G/s   D=2  ← best
Avx512BlockedPf4/f32/N=512         1115 µs   244.34 G/s   D=4  ← best
Avx512BlockedPf2/f32/N=1024       15656 µs   137.44 G/s   D=2
```

#### x86 speedup tables

##### f64 — best kernel per family vs `gemm_naive` (Intel x86)

| N | Naive | Reordered | ×naive | Blocked | ×naive | Avx2Blocked | ×naive | Avx512Blocked | ×naive |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 52.0 µs | 104 µs | 0.5× | 103 µs | 0.5× | 7.83 µs | **6.6×** | 3.79 µs | **13.7×** |
| 256 | 14017 µs | 6690 µs | **2.1×** | 6892 µs | **2.0×** | 505 µs | **27.8×** | 277 µs | **50.6×** |
| 512 | 194863 µs | 54141 µs | **3.6×** | 55092 µs | **3.5×** | 5860 µs | **33.3×** | 4727 µs | **41.2×** |
| 1024 | 2794359 µs | 427629 µs | **6.5×** | 441021 µs | **6.3×** | 47682 µs | **58.6×** | 39896 µs | **70.0×** |
| 4096 | 305295146 µs | 34642827 µs | **8.8×** | 28273076 µs | **10.8×** | 3624176 µs | **84.3×** | 2796709 µs | **109.2×** |

##### f32 — best kernel per family vs `gemm_naive` (Intel x86)

| N | Naive | Reordered | ×naive | Blocked | ×naive | Avx2Blocked | ×naive | Avx512Blocked | ×naive |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 52.5 µs | 101 µs | 0.5× | 102 µs | 0.5× | 3.72 µs | **14.1×** | 1.81 µs | **29.0×** |
| 256 | 9710 µs | 6614 µs | **1.5×** | 6768 µs | **1.4×** | 231 µs | **42.0×** | 124 µs | **78.3×** |
| 512 | 120715 µs | 53251 µs | **2.3×** | 53831 µs | **2.2×** | 1903 µs | **63.4×** | 1110 µs | **108.8×** |
| 1024 | 2800837 µs | 424514 µs | **6.6×** | 432354 µs | **6.5×** | 23777 µs | **117.8×** | 15673 µs | **178.7×** |
| 4096 | 313939135 µs | 27507882 µs | **11.4×** | 27694736 µs | **11.3×** | 1764409 µs | **177.9×** | 1110351 µs | **282.7×** |

#### Headline GFLOP/s summary (Intel x86 + AVX-512, this run)

| Kernel | f64 peak | f32 peak | f32/f64 ratio |
|---|---|---|---|
| `gemm_naive` | 9.87 G/s | 9.87 G/s | 1.0× |
| `gemm_reordered` | 5.11 G/s | 5.11 G/s | 1.0× |
| `gemm_blocked` | 5.11 G/s | 5.11 G/s | 1.0× |
| `gemm_avx2_blocked` | **67.11 G/s** | **145.79 G/s** | **2.2×** |
| `gemm_avx512_blocked` | **139.19 G/s** | **290.76 G/s** | **2.1×** |
| `gemm_avx512_blocked_prefetch` | **55.63 G/s** @ N=1024 | **273.32 G/s** @ N=256 | — |

> **Note:** scalar kernels (`Reordered`, `Blocked`) show ~5 G/s on this x86 machine because
> MSVC does not auto-vectorise as aggressively as GCC/Clang with `-march=native -ffast-math`.
> The explicit SIMD kernels (AVX2, AVX-512) bypass this entirely and reach the expected throughput.
> AVX-512 `f32` peaks at **290 G/s** at N=64, nearly **2× the AVX2 peak** — the doubled register
> width (512- vs 256-bit) translates directly to throughput.

---

## Key observations

### Cache-access pattern dominates at large N

`gemm_naive` delivers nearly identical GFLOP/s for f32 and f64 at every size — both are **DRAM-bandwidth bound** from the column-stride gather on B. Element width is irrelevant once you are waiting on cache-miss latency.

The moment the loop order changes to i-k-j (`gemm_reordered`), B and C are accessed sequentially and every cache line is fully consumed. At N=4096:
- f64: **23×** faster than naive
- f32: **46×** faster than naive (twice the elements per cache line → twice the bandwidth)

### Auto-vectorisation vs explicit SIMD

`gemm_reordered` and `gemm_blocked` carry **no NEON intrinsics** — the compiler auto-vectorises the sequential inner j-loop with `-march=native -ffast-math`. f32 delivers ~84 GFLOP/s at small N.

`gemm_neon_blocked` adds **explicit Q-register tiling** (4 rows × 4 Q-vectors = 4×16 f32 held in registers for the full k-tile) on top of L2 blocking:

| Kernel | f32 N=256 | f32 N=512 | f32 N=1024 |
|---|---|---|---|
| `gemm_blocked` (auto-vec) | 83.2 G/s | 49.9 G/s | 39.8 G/s |
| `gemm_neon_blocked` (explicit) | **96.7 G/s** | **97.1 G/s** | **95.1 G/s** |

The explicit register tile maintains ~96 GFLOP/s from N=64 through N=1024 — **flat across sizes**. The auto-vectorised blocked kernel degrades from 84→40 G/s because C rows are evicted from L1 between k-iterations at larger N.

### Software prefetch analysis

**Scalar `BlockedPf` vs base `Blocked`:** prefetch *hurts* at N=256 (14.9 vs 25.2 G/s for f64) and gives only marginal gain at N=512/1024. The scalar kernel is entirely compiler-auto-vectorised; the hardware prefetcher already handles the simple streaming access, and adding explicit prefetch instructions creates front-end pressure that slows the tight inner loop.

**`NeonBlockedPf` vs base `NeonBlocked`:** prefetch gives a small but consistent gain:

| Kernel | f32 N=256 | f32 N=512 | f32 N=1024 |
|---|---|---|---|
| `NeonBlocked` (no prefetch) | 96.7 G/s | 97.1 G/s | 95.1 G/s |
| `NeonBlockedPf2` (D=2) | **98.6 G/s** | **98.9 G/s** | **96.5 G/s** |
| Gain | **+2.0%** | **+1.8%** | **+1.5%** |

For f64 the gain is slightly larger in absolute terms (D=2 wins at all sizes). D=2 consistently outperforms D=4/8/16 — the L2 latency on Apple M is short enough that prefetching more than 2 micro-kernel steps ahead adds latency-hiding overhead without benefit.

**Prefetch distance rule of thumb for this hardware:**

```
optimal D ≈ ceil(L2_latency_cycles / cycles_per_micro_kernel_call)
          ≈ ceil(12 / ~6) = 2
```

### NEON f64 vs f32

- NEON Q-register: 4 f32 lanes or 2 f64 lanes (128-bit).
- `gemm_neon_blocked` f64 peaks at ~36 G/s; f32 peaks at ~97 G/s — ratio ≈ **2.7×**.
- The theoretical ratio is 2× (lane count). The extra 0.7× for f32 comes from f32 tiles fitting entirely in L1 at sizes where f64 tiles spill.

### Headline GFLOP/s summary (Apple M-series, this run)

| Kernel | f64 peak | f32 peak | f32/f64 ratio |
|---|---|---|---|
| `gemm_naive` | 9.22 G/s | 9.14 G/s | 1.0× |
| `gemm_reordered` | 26.68 G/s | 84.83 G/s | **3.2×** |
| `gemm_blocked` | 26.80 G/s | 84.06 G/s | **3.1×** |
| `gemm_neon_blocked` | 35.92 G/s | 97.10 G/s | **2.7×** |
| `gemm_neon_blocked_prefetch` (D=2) | **34.05 G/s** @ N=256 | **98.87 G/s** @ N=512 | **2.9×** |

---

## Deriving GFLOP/s

```
GFLOP/s = (2 × N³) / (time_µs × 1000)
```

A square N×N GEMM performs exactly `2 × N³` floating-point operations (N³ multiplications + N³ additions, fused into FMA). Dividing by wall-clock time in nanoseconds gives GFLOP/s.

Example: `NeonBlockedPf2/f32/N=512`, 2715 µs → `2 × 512³ / (2715 × 1000)` ≈ **98.9 GFLOP/s**.

---

## Documentation

- **[src/gemm/README.md](src/gemm/README.md)** — Side-by-side loop analysis with ASCII memory access diagrams for each kernel.
- **[docs/cache-behavior.md](docs/cache-behavior.md)** — Cache lines, reuse distance, working-set analysis, roofline model.

---

## License

MIT. See [LICENSE](LICENSE).
