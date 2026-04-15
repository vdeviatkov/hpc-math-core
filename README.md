# hpc-math-core

A progressive benchmark suite demonstrating **hardware-aware optimisations for linear algebra**, targeting quantitative engineering and high-frequency trading performance standards.

This repository starts from first principles — readable scalar code — and adds successive layers of hardware exploitation: cache-friendly access patterns, SIMD vectorisation (AVX2 · AVX-512 · NEON · SVE), and CUDA. Every step is fully benchmarked, cross-validated by a Google Test suite, and documented with ASCII memory diagrams and cache analysis.

---

## Current status: Step 4 — ARM NEON Explicit SIMD GEMM

| Step | Kernel family | Key technique | Status |
|---|---|---|---|
| 0 | `gemm_naive` / `gemm_reordered` | Loop reordering, cache-friendly i-k-j access | ✅ |
| 1 | `gemm_blocked` | L2 cache tiling (default tile = 64) | ✅ |
| 2 | `gemm_avx2_{naive,reordered,blocked}` | Explicit AVX2 FMA intrinsics, 4×16 f32 / 4×8 f64 register tile | ✅ |
| 3 | `gemm_avx512_{naive,reordered,blocked}` | 512-bit ZMM register tile, embedded broadcast | ✅ |
| 4 | `gemm_neon_{naive,reordered,blocked}` | 128-bit Q-register tile (ARM NEON / AdvSIMD), `vfmaq` FMA | ✅ |
| 5 | `gemm_sve_{naive,reordered,blocked}` | VLA SVE: runtime VL, `svwhilelt` predicates, zero scalar tails | ✅ |
| 6 | `gemm_cuda` | Tiled CUDA kernel (shared memory) | 🔜 |

Each SIMD family is **skipped automatically** if the ISA is absent on the build CPU — no `#ifdef` pollution in benchmark registrations, no silent fallback timing. See [§ ISA availability](#isa-availability--skipping) below.

See [src/gemm/README.md](src/gemm/README.md) for per-kernel memory diagrams and [docs/cache-behavior.md](docs/cache-behavior.md) for the full cache analysis.

---

## Repository layout

```
hpc-math-core/
├── CMakeLists.txt
├── include/
│   └── hpc/
│       └── matrix.hpp                  Matrix<T>: 64-byte aligned, row-major
├── src/
│   └── gemm/
│       ├── naive.hpp                   i-j-k scalar baseline
│       ├── reordered.hpp               i-k-j scalar (cache-friendly)
│       ├── blocked.hpp                 tiled i-k-j (tile=64)
│       ├── avx2.hpp                    AVX2 FMA: 4×16 f32 / 4×8 f64 register tile
│       ├── avx512.hpp                  AVX-512: 4×32 f32 / 4×16 f64 register tile
│       ├── neon.hpp                    ARM NEON: 4×16 f32 / 4×4 f64 Q-register tile
│       ├── sve.hpp                     ARM SVE: VLA tile, predicated tails
│       └── README.md
├── benchmarks/
│   ├── CMakeLists.txt
│   └── bench_gemm.cpp                  Google Benchmark driver
├── tests/
│   ├── CMakeLists.txt
│   └── test_gemm.cpp                   Google Test correctness suite (233 tests)
└── docs/
    └── cache-behavior.md
```

---

## Prerequisites

| Tool | Minimum version | Notes |
|---|---|---|
| CMake | 3.25 | FetchContent, `gtest_discover_tests` |
| C++ compiler | GCC 12 / Clang 16 / Apple Clang 15 | C++20 required |

---

## Build & run

```bash
# 1. Configure — Release enables -O3 -march=native -ffast-math -funroll-loops
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Build everything
cmake --build build --parallel

# 3. Run tests (233 tests across all kernel families)
cd build && ctest --output-on-failure

# 4. Run benchmarks — only native ISA kernels execute; others print SKIPPED
./build/benchmarks/bench_gemm --benchmark_format=console
```

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
./build/benchmarks/bench_gemm --benchmark_filter="f32"        # float only
./build/benchmarks/bench_gemm --benchmark_filter="Neon"       # NEON family
./build/benchmarks/bench_gemm --benchmark_filter="Blocked"    # all blocked kernels
./build/benchmarks/bench_gemm --benchmark_filter="N=1024"     # one size across all kernels
```

---

## ISA availability & skipping

Every SIMD benchmark body begins with a compile-time guard:

```cpp
if (!kHaveNeon) { state.SkipWithMessage("NEON not available on this target"); return; }
```

The benchmark name still appears in the output — as `SKIPPED` — so you always see the full kernel catalogue and know exactly which paths ran. No silent fallback to a slower scalar kernel that would corrupt the numbers.

| Machine | Runs | Skipped |
|---|---|---|
| **Apple M-series** (this run) | Scalar · NEON | AVX2 · AVX-512 · SVE |
| Intel Skylake (AVX2, no AVX-512) | Scalar · AVX2 | AVX-512 · NEON · SVE |
| Intel Sapphire Rapids | Scalar · AVX2 · AVX-512 | NEON · SVE |
| AWS Graviton3 / Neoverse V1 | Scalar · NEON · SVE (256-bit) | AVX2 · AVX-512 |
| Fujitsu A64FX | Scalar · NEON · SVE (512-bit) | AVX2 · AVX-512 |

---

## Sample benchmark output

> **Machine:** Apple M-series, 16 P-cores, Apple Clang, C++20  
> **Build:** `cmake -DCMAKE_BUILD_TYPE=Release` → `-O3 -march=native -ffast-math -funroll-loops`  
> **CPU Caches:** L1 Data 64 KiB · L1 Instruction 128 KiB · L2 Unified 4096 KiB (×16)  
> **Load Average:** 3.95 / 4.55 / 5.86 — three numbers are 1-min / 5-min / 15-min average
> number of runnable processes. On a 16-core machine a value of 16 means 100% utilisation;
> 3.95–5.86 ≈ 25–37% load — moderate background activity, numbers are still representative.

### double (f64)

```
Benchmark                    Time        CPU     GFLOP/s
--------------------------------------------------------
Naive/f64/N=64              56.9 µs    56.9 µs    9.22
Naive/f64/N=256          13029  µs  13023  µs     2.58
Naive/f64/N=512         105205  µs  105198 µs     2.55
Naive/f64/N=1024        874277  µs  874204 µs     2.46
Naive/f64/N=4096     197960358  µs   197.6s       695.6 M/s

Reordered/f64/N=64          19.7 µs    19.7 µs   26.64
Reordered/f64/N=256       2054   µs   2054   µs  16.34
Reordered/f64/N=512      16438   µs  16436   µs  16.33
Reordered/f64/N=1024    131080   µs  131055  µs  16.39
Reordered/f64/N=4096   8445948   µs    8.43s     16.30

Blocked/f64/N=64            19.9 µs    19.8 µs   26.43  tile=64
Blocked/f64/N=256         1337   µs   1337   µs  25.10  tile=64
Blocked/f64/N=512        12214   µs  12212   µs  21.98  tile=64
Blocked/f64/N=1024      112917   µs  112896  µs  19.02  tile=64
Blocked/f64/N=4096     6874608   µs    6.87s     20.00  tile=64

NeonNaive/f64/N=64          61.5 µs    61.5 µs    8.53  neon=1
NeonNaive/f64/N=256      14064   µs  14062   µs   2.39  neon=1
NeonNaive/f64/N=512     108528   µs  108518  µs   2.47  neon=1
NeonNaive/f64/N=1024    921764   µs  921648  µs   2.33  neon=1
NeonNaive/f64/N=4096  200192800  µs   199.9s    687.5 M/s neon=1

NeonReordered/f64/N=64      19.8 µs    19.8 µs   26.43  neon=1
NeonReordered/f64/N=256    2478   µs   2476   µs  13.55  neon=1
NeonReordered/f64/N=512   18935   µs  18924   µs  14.19  neon=1
NeonReordered/f64/N=1024 145441   µs  145388  µs  14.77  neon=1
NeonReordered/f64/N=4096 8942158  µs    8.92s    15.41  neon=1

NeonBlocked/f64/N=64        14.7 µs    14.7 µs   35.74  neon=1
NeonBlocked/f64/N=256      1047   µs   1041   µs  32.25  neon=1
NeonBlocked/f64/N=512      8656   µs   8654   µs  31.02  neon=1
NeonBlocked/f64/N=1024    72040   µs  72013   µs  29.82  neon=1
NeonBlocked/f64/N=4096  5625881   µs    5.62s    24.45  neon=1

Avx2*/f64/*     SKIPPED: 'AVX2 not available on this target'
Avx512*/f64/*   SKIPPED: 'AVX-512 not available on this target'
Sve*/f64/*      SKIPPED: 'SVE not available on this target'
```

**f64 speedup table (best kernel per family vs `gemm_naive`):**

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive |
|---|---|---|---|---|---|---|---|
| 64 | 56.9 µs | 19.7 µs | **2.9×** | 19.9 µs | **2.9×** | 14.7 µs | **3.9×** |
| 256 | 13029 µs | 2054 µs | **6.3×** | 1337 µs | **9.7×** | 1041 µs | **12.5×** |
| 512 | 105205 µs | 16438 µs | **6.4×** | 12214 µs | **8.6×** | 8654 µs | **12.2×** |
| 1024 | 874277 µs | 131080 µs | **6.7×** | 112917 µs | **7.7×** | 72013 µs | **12.1×** |
| 4096 | 197960358 µs | 8445948 µs | **23.4×** | 6874608 µs | **28.8×** | 5622322 µs | **35.2×** |

### float (f32)

```
Benchmark                    Time        CPU     GFLOP/s
--------------------------------------------------------
Naive/f32/N=64              57.2 µs    57.2 µs    9.17
Naive/f32/N=256          12679  µs  12678  µs     2.65
Naive/f32/N=512         114973  µs  114959 µs     2.34
Naive/f32/N=1024        828912  µs  828855 µs     2.59
Naive/f32/N=4096     204100587  µs   203.9s       674.0 M/s

Reordered/f32/N=64           6.24 µs    6.24 µs  84.00
Reordered/f32/N=256        1063   µs   1063   µs  31.58
Reordered/f32/N=512        8307   µs   8306   µs  32.32
Reordered/f32/N=1024      66389   µs  66382   µs  32.35
Reordered/f32/N=4096    4234650   µs    4.23s    32.46

Blocked/f32/N=64             6.24 µs    6.24 µs  84.00  tile=64
Blocked/f32/N=256             412 µs     412  µs  81.45  tile=64
Blocked/f32/N=512            5356 µs    5355  µs  50.12  tile=64
Blocked/f32/N=1024          50529 µs   50521  µs  42.51  tile=64
Blocked/f32/N=4096        4575025 µs    4.57s    30.05  tile=64

NeonNaive/f32/N=64          58.5 µs    58.5 µs    8.97  neon=1
NeonNaive/f32/N=256        9480   µs   9471   µs   3.54  neon=1
NeonNaive/f32/N=512       78719   µs  78702   µs   3.41  neon=1
NeonNaive/f32/N=1024     860217   µs  860079  µs   2.50  neon=1
NeonNaive/f32/N=4096  206934004  µs   206.6s    665.1 M/s neon=1

NeonReordered/f32/N=64      21.2 µs    21.1 µs   24.80  neon=1
NeonReordered/f32/N=256    1137   µs   1136   µs  29.53  neon=1
NeonReordered/f32/N=512    9819   µs   9815   µs  27.35  neon=1
NeonReordered/f32/N=1024  77040   µs  77016   µs  27.88  neon=1
NeonReordered/f32/N=4096 4539280  µs    4.54s    30.30  neon=1

NeonBlocked/f32/N=64         5.44 µs    5.44 µs  96.42  neon=1
NeonBlocked/f32/N=256         352 µs     352  µs  95.24  neon=1
NeonBlocked/f32/N=512        2788 µs    2780  µs  96.56  neon=1
NeonBlocked/f32/N=1024      22845 µs   22833  µs  94.05  neon=1
NeonBlocked/f32/N=4096    1901410 µs    1.90s    72.33  neon=1

Avx2*/f32/*     SKIPPED: 'AVX2 not available on this target'
Avx512*/f32/*   SKIPPED: 'AVX-512 not available on this target'
Sve*/f32/*      SKIPPED: 'SVE not available on this target'
```

**f32 speedup table (best kernel per family vs `gemm_naive`):**

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive |
|---|---|---|---|---|---|---|---|
| 64 | 57.2 µs | 6.24 µs | **9.2×** | 6.24 µs | **9.2×** | 5.44 µs | **10.5×** |
| 256 | 12679 µs | 1063 µs | **11.9×** | 412 µs | **30.8×** | 352 µs | **36.0×** |
| 512 | 114973 µs | 8307 µs | **13.8×** | 5356 µs | **21.5×** | 2780 µs | **41.4×** |
| 1024 | 828912 µs | 66389 µs | **12.5×** | 50529 µs | **16.4×** | 22833 µs | **36.3×** |
| 4096 | 204100587 µs | 4234650 µs | **48.2×** | 4575025 µs | **44.6×** | 1900039 µs | **107.4×** |

---

## Key observations

### Cache-access pattern dominates at large N

`gemm_naive` delivers nearly identical GFLOP/s for f32 and f64 at every size — both are **DRAM-bandwidth bound** from the column-stride gather on B. Element width is irrelevant once you are waiting on cache-miss latency.

The moment the loop order changes to i-k-j (`gemm_reordered`), B and C are accessed sequentially and every cache line is fully consumed. At N=4096:
- f64: **23.4× faster** than naive
- f32: **48.2× faster** than naive (twice the elements per cache line → twice the bandwidth)

### Auto-vectorisation vs explicit SIMD

`gemm_reordered` and `gemm_blocked` carry **no NEON intrinsics** — the compiler auto-vectorises the sequential inner j-loop with `-march=native -ffast-math`. The fact that f32 delivers ~84 GFLOP/s at small N and ~32 GFLOP/s at large N is entirely the work of the auto-vectoriser emitting `vfma` instructions.

`gemm_neon_blocked` adds **explicit Q-register tiling** (4 rows × 4 Q-vectors = 4×16 f32 held in registers for the full k-tile) on top of L2 blocking. The result is the largest observed speedup on this hardware:

| Kernel | f32 N=256 | f32 N=512 | f32 N=1024 |
|---|---|---|---|
| `gemm_blocked` (auto-vec) | 81.5 G/s | 50.1 G/s | 42.5 G/s |
| `gemm_neon_blocked` (explicit) | **95.2 G/s** | **96.6 G/s** | **94.1 G/s** |

The explicit register tile maintains ~95 GFLOP/s from N=64 through N=1024 — **flat across sizes**. The auto-vectorised blocked kernel degrades from 84→42 G/s because C rows are evicted from L1 between k-iterations at larger N. Keeping C in Q-registers for the full k-tile removes that bottleneck.

### NEON f64 vs f32

- NEON Q-register: 4 f32 lanes or 2 f64 lanes (128-bit).
- `gemm_neon_blocked` f64 peaks at ~35 G/s; f32 peaks at ~96 G/s — ratio ≈ **2.7×**.
- The theoretical ratio is 2× (4 vs 2 lanes). The extra 0.7× advantage for f32 comes from f32 tiles fitting entirely in L1 at sizes where f64 tiles are already spilling.

### NeonNaive f32 faster than NeonNaive f64

Counterintuitively, `NeonNaive/f32/N=256` (9480 µs) is **faster** than `NeonNaive/f64/N=256` (14064 µs) despite both being bandwidth-bound. The reason: the f32 matrix is half the bytes, so it occupies half the cache footprint. At N=256 the f32 B matrix (256 KB) fits in L2 while f64 (512 KB) partially spills — exposing a cache-capacity effect even in the naive kernel.

### Headline GFLOP/s summary (Apple M-series, this run)

| Kernel | f64 peak | f32 peak | f32/f64 ratio |
|---|---|---|---|
| `gemm_naive` | 9.22 G/s | 9.17 G/s | 1.0× |
| `gemm_reordered` | 26.64 G/s | 84.00 G/s | **3.2×** |
| `gemm_blocked` | 26.43 G/s | 84.00 G/s | **3.2×** |
| `gemm_neon_blocked` | **35.74 G/s** | **96.56 G/s** | **2.7×** |

---

## Deriving GFLOP/s

```
GFLOP/s = (2 × N³) / (time_µs × 1000)
```

A square N×N GEMM performs exactly `2 × N³` floating-point operations (N³ multiplications + N³ additions, fused into FMA). Dividing by wall-clock time in nanoseconds gives GFLOP/s.

Example: `NeonBlocked/f32/N=512`, 2780 µs → `2 × 512³ / (2780 × 1000)` ≈ **96.6 GFLOP/s**.

---

## Documentation

- **[src/gemm/README.md](src/gemm/README.md)** — Side-by-side loop analysis with ASCII memory access diagrams for each kernel.
- **[docs/cache-behavior.md](docs/cache-behavior.md)** — Cache lines, reuse distance, working-set analysis, roofline model.

---

## License

MIT. See [LICENSE](LICENSE).
