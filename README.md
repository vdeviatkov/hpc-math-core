# hpc-math-core

A progressive benchmark suite demonstrating **hardware-aware optimisations for linear algebra**, targeting quantitative engineering and high-frequency trading performance standards.

This repository starts from first principles — readable scalar code — and adds successive layers of hardware exploitation: cache-friendly access patterns, SIMD vectorisation, and CUDA. Every step is fully benchmarked and documented.

---

## Current status: Step 1 — Cache-Blocked GEMM

| Kernel | Loop order | Description |
|---|---|---|
| `gemm_naive` | i → j → k | Baseline. Column-stride B access thrashes cache. |
| `gemm_reordered` | i → k → j | Cache-friendly. Sequential access to both B and C. |
| `gemm_blocked` | tiled i → k → j | L2-resident tiles. Eliminates reuse-distance problem for large N. |

**Expected speedup** of `gemm_reordered` over `gemm_naive`: **4–8×** for N ≥ 256 on modern hardware. See [docs/cache-behavior.md](docs/cache-behavior.md) for the full analysis.

---

## Repository layout

```
hpc-math-core/
├── CMakeLists.txt              Root build configuration
├── include/
│   └── hpc/
│       └── matrix.hpp          Matrix<T>: 64-byte aligned, row-major
├── src/
│   └── gemm/
│       ├── naive.hpp           i-j-k implementation
│       ├── reordered.hpp       i-k-j implementation
│       ├── blocked.hpp         tiled i-k-j implementation (default tile=64)
│       └── README.md           Per-kernel documentation with ASCII diagrams
├── benchmarks/
│   ├── CMakeLists.txt
│   └── bench_gemm.cpp          Google Benchmark driver (N = 64/256/512/1024/4096)
├── tests/
│   ├── CMakeLists.txt
│   └── test_gemm.cpp           Google Test correctness suite
└── docs/
    └── cache-behavior.md       Cache theory: reuse distance, roofline model
```

---

## Prerequisites

| Tool | Minimum version | Notes |
|---|---|---|
| CMake | 3.25 | FetchContent, gtest_discover_tests |
| C++ compiler | GCC 12 / Clang 16 / Apple Clang 15 | C++20 required |

---

## Build & run

```bash
# 1. Configure (Release is essential for meaningful benchmark numbers)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Build everything
cmake --build build --parallel

# 3. Run tests
cd build && ctest --output-on-failure

# 4. Run benchmarks
./build/benchmarks/bench_gemm --benchmark_format=console
```

### Optional: enable AVX-512 (Intel Skylake-SP / Ice Lake / Sapphire Rapids only)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_AVX512=ON
```

> ⚠️ **Do not enable AVX-512 on CPUs that lack it** (pre-Skylake-SP Intel, all Apple Silicon, older AMD). The binary will crash with `SIGILL` at runtime.

---

## Sample benchmark output

> Run on 16 × 24 MHz CPU cores. Build: `cmake -DCMAKE_BUILD_TYPE=Release`, Apple Clang, C++20.
>
> CPU Caches: L1 Data 64 KiB · L1 Instruction 128 KiB · L2 Unified 4096 KiB (×16)  
> Load Average: 2.17, 3.11, 4.94

```
---------------------------------------------------------------------------
Benchmark                 Time             CPU   Iterations   GFLOP/s
---------------------------------------------------------------------------
Naive/N=64             85.1 µs          85.0 µs        8087    6.17 G/s
Naive/N=256            9531 µs          9525 µs          73    3.52 G/s
Naive/N=512          103568 µs        103561 µs           7    2.59 G/s
Naive/N=1024        1012672 µs       1012527 µs           1    2.12 G/s
Naive/N=4096      226560454 µs     195484645 µs           1  703.07 M/s

Reordered/N=64         19.1 µs          19.0 µs       36771   27.56 G/s
Reordered/N=256        2033 µs          2032 µs         349   16.52 G/s
Reordered/N=512       16342 µs         16313 µs          43   16.46 G/s
Reordered/N=1024     130420 µs        130329 µs           5   16.48 G/s
Reordered/N=4096    8356900 µs       8351151 µs           1   16.46 G/s

Blocked/N=64           19.2 µs          19.2 µs       36192   27.24 G/s  (tile=64)
Blocked/N=256          1484 µs          1469 µs         484   22.84 G/s  (tile=64)
Blocked/N=512         13082 µs         13038 µs          54   20.59 G/s  (tile=64)
Blocked/N=1024       115676 µs        115653 µs           6   18.57 G/s  (tile=64)
Blocked/N=4096      7377471 µs       7372975 µs           1   18.64 G/s  (tile=64)
```

**Speedup over `gemm_naive`:**

| N | Naive | Reordered | Speedup | Blocked | Speedup |
|---|---|---|---|---|---|
| 64 | 85.1 µs | 19.1 µs | **4.5×** | 19.2 µs | **4.4×** |
| 256 | 9531 µs | 2033 µs | **4.7×** | 1484 µs | **6.4×** |
| 512 | 103568 µs | 16342 µs | **6.3×** | 13082 µs | **7.9×** |
| 1024 | 1012672 µs | 130420 µs | **7.8×** | 115676 µs | **8.8×** |
| 4096 | 226560454 µs | 8356900 µs | **27.1×** | 7377471 µs | **30.7×** |

Notable observations:
- **`gemm_reordered`** sustains a flat **~16.5 GFLOP/s** across all sizes — hardware prefetching keeps row accesses to B and C fully pipelined regardless of working-set size.
- **`gemm_blocked`** consistently beats reordered for N ≥ 256, reaching **~19–23 GFLOP/s** — tiling reduces inter-tile reuse distance and keeps the active C sub-matrix hotter in L1, especially visible at N=256 where blocked is **1.4× faster than reordered**.
- **`gemm_naive`** collapses to **703 M/s at N=4096** (column-stride B access, every load is an L3/DRAM miss). The **30.7× gap** vs. blocked at N=4096 is purely a memory-access-pattern effect — zero algorithmic difference.
- The blocked kernel's GFLOP/s is still well below the scalar FMA peak (~24 GFLOP/s on this CPU), leaving clear headroom for AVX2/AVX-512 SIMD in the next steps.

---

## Deriving GFLOP/s

```
GFLOP/s = (2 × N³) / (time_µs × 1000)
```

Example: `gemm_blocked`, N=1024, 115653 µs → `2 × 1024³ / (115653 × 1000)` ≈ **18.57 GFLOP/s** (single-core, scalar, tile=64). With AVX-512 FMA the theoretical peak on a 3 GHz core is ≈ 192 GFLOP/s — there is still ~10× headroom to recover through SIMD vectorisation (Steps 2–3).

---

## Roadmap

| Step | Kernel | Key technique |
|---|---|---|
| ✅ 0 | `gemm_naive` / `gemm_reordered` | Loop reordering, cache-friendly access |
| ✅ 1 | `gemm_blocked` | Loop tiling (L2-resident tiles) |
| 🔜 2 | `gemm_avx2` | 256-bit AVX2 FMA intrinsics |
| 🔜 3 | `gemm_avx512` | 512-bit AVX-512 + software prefetch |
| 🔜 4 | `gemm_cuda` | Tiled CUDA kernel (shared memory) |

---

## Documentation

- **[src/gemm/README.md](src/gemm/README.md)** — Side-by-side loop analysis with ASCII memory diagrams.
- **[docs/cache-behavior.md](docs/cache-behavior.md)** — Deep dive: cache lines, reuse distance, working-set analysis, roofline model.

---

## License

MIT. See [LICENSE](LICENSE).
