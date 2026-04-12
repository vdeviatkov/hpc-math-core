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
| Internet access | — | FetchContent downloads GTest & Google Benchmark |

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

> Run on a hypothetical 4-core workstation. Replace with your actual output.

```
------------------------------------------------------------
Benchmark               Time           CPU   Iterations
------------------------------------------------------------
Naive/N=64              5 µs           5 µs       140000
Naive/N=256           370 µs         370 µs         1900
Naive/N=512          4200 µs        4200 µs          167
Naive/N=1024        52000 µs       52000 µs           13
Naive/N=4096     ~5000000 µs    ~5000000 µs            1
Reordered/N=64          4 µs           4 µs       180000
Reordered/N=256        75 µs          75 µs         9300
Reordered/N=512       610 µs         610 µs         1100
Reordered/N=1024     5100 µs        5100 µs          137
Reordered/N=4096   ~90000 µs      ~90000 µs            5
```

*Fill this table with your own numbers after running the benchmarks.*

---

## Deriving GFLOP/s

```
GFLOP/s = (2 × N³) / (time_µs × 1000)
```

Example: `gemm_reordered`, N=1024, 5100 µs → `2 × 1024³ / (5100 × 1000)` ≈ **0.42 GFLOP/s** (single-core, no SIMD). With AVX-512 FMA the theoretical peak on a 3 GHz core is ≈ 192 GFLOP/s — there is plenty of room for future steps.

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
