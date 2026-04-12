# hpc-math-core

A progressive benchmark suite demonstrating **hardware-aware optimisations for linear algebra**, targeting quantitative engineering and high-frequency trading performance standards.

This repository starts from first principles — readable scalar code — and adds successive layers of hardware exploitation: cache-friendly access patterns, SIMD vectorisation, and CUDA. Every step is fully benchmarked and documented.

---

## Current status: Step 0 — Scalar CPU GEMM

| Kernel | Loop order | Description |
|---|---|---|
| `gemm_naive` | i → j → k | Baseline. Column-stride B access thrashes cache. |
| `gemm_reordered` | i → k → j | Cache-friendly. Sequential access to both B and C. |

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
│       └── README.md           Per-kernel documentation with ASCII diagrams
├── benchmarks/
│   ├── CMakeLists.txt
│   └── bench_gemm.cpp          Google Benchmark driver (N = 64/256/512/1024)
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
> Load Average: 3.43, 3.08, 3.57

```
---------------------------------------------------------------------------
Benchmark                 Time             CPU   Iterations   GFLOP/s
---------------------------------------------------------------------------
Naive/N=64             85.3 µs          85.3 µs        8196    6.15 G/s
Naive/N=256            9544 µs          9540 µs          73    3.52 G/s
Naive/N=512          103163 µs        103154 µs           7    2.60 G/s
Naive/N=1024         856731 µs        856668 µs           1    2.51 G/s
Naive/N=4096      197161840 µs     196766521 µs           1  698.49 M/s

Reordered/N=64         19.5 µs          19.5 µs       35664   26.92 G/s
Reordered/N=256        2050 µs          2049 µs         343   16.38 G/s
Reordered/N=512       16476 µs         16471 µs          43   16.30 G/s
Reordered/N=1024     131522 µs        131484 µs           5   16.33 G/s
Reordered/N=4096    8440910 µs       8437719 µs           1   16.29 G/s
```

**Speedup of `gemm_reordered` over `gemm_naive`:**

| N | Naive | Reordered | Speedup |
|---|---|---|---|
| 64 | 85.3 µs | 19.5 µs | **4.4×** |
| 256 | 9544 µs | 2050 µs | **4.7×** |
| 512 | 103163 µs | 16476 µs | **6.3×** |
| 1024 | 856731 µs | 131522 µs | **6.5×** |
| 4096 | 197161840 µs | 8440910 µs | **23.4×** |

Notable observations:
- The reordered kernel sustains a **flat ~16 GFLOP/s** across all sizes — hardware prefetching keeps the pipeline saturated even when the working set far exceeds L2/L3.
- The naïve kernel degrades continuously with N, collapsing to **698 M/s at N=4096** — a direct consequence of column-stride access to B thrashing every cache level and becoming fully DRAM-latency-bound.
- The **23× gap at N=4096** is the clearest possible argument for loop reordering as a zero-cost (same algorithmic complexity) transformation.

---

## Deriving GFLOP/s

```
GFLOP/s = (2 × N³) / (time_µs × 1000)
```

Example: `gemm_reordered`, N=1024, 131484 µs → `2 × 1024³ / (131484 × 1000)` ≈ **16.33 GFLOP/s** (single-core, scalar). With AVX-512 FMA the theoretical peak on a 3 GHz core is ≈ 192 GFLOP/s — there is ~12× headroom still to recover through SIMD vectorisation (Steps 2–3).

---

## Roadmap

| Step | Kernel | Key technique |
|---|---|---|
| ✅ 0 | `gemm_naive` / `gemm_reordered` | Loop reordering, cache-friendly access |
| 🔜 1 | `gemm_blocked` | Loop tiling (L1/L2 blocking) |
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
