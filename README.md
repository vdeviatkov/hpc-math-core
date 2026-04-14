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

See [src/gemm/README.md](src/gemm/README.md) for per-kernel memory diagrams and [docs/cache-behavior.md](docs/cache-behavior.md) for the full cache analysis.

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
# 1. Configure — Release enables -O3 -march=native -ffast-math -funroll-loops
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Build everything
cmake --build build --parallel

# 3. Run tests
cd build && ctest --output-on-failure

# 4. Run benchmarks (both float and double, all kernels)
./build/benchmarks/bench_gemm --benchmark_format=console
```

### Optimisation flags applied automatically in Release mode

| Flag | Effect |
|---|---|
| `-O3` | Full optimisation: auto-vectorisation, aggressive inlining, loop transforms |
| `-march=native` | Emit instructions for the exact build CPU (AVX2, FMA, BMI2 …). **Binary is not portable.** |
| `-ffast-math` | Permits FMA contraction, reassociation and reciprocal approximations. Assumes no NaN/Inf. ⚠ Not safe for production code relying on strict IEEE-754. |
| `-funroll-loops` | Unroll loops with statically-known trip counts; reduces branch overhead in the inner j-loop |
| `-fno-omit-frame-pointer` | Preserves call-stack unwinding for `perf`/Instruments profiling even in optimised builds |

### Optional: Link-Time Optimisation (LTO)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_LTO=ON
```

Allows the compiler to inline and optimise across translation-unit boundaries. Increases link time but can improve performance when kernels are split into separate `.cpp` files in future steps.

### Optional: AVX-512 (Intel Skylake-SP / Ice Lake / Sapphire Rapids only)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_AVX512=ON
```

> ⚠️ **Do not enable AVX-512 on CPUs that lack it** (pre-Skylake-SP Intel, all Apple Silicon, older AMD). The binary will crash with `SIGILL` at runtime.

### Filtering benchmarks by precision or kernel

```bash
# Run only float benchmarks
./build/benchmarks/bench_gemm --benchmark_filter="f32"

# Run only blocked kernel
./build/benchmarks/bench_gemm --benchmark_filter="Blocked"

# Run only a specific size
./build/benchmarks/bench_gemm --benchmark_filter="N=1024"
```

---

## Sample benchmark output

> Run on 16 × 24 MHz CPU cores. Build: `cmake -DCMAKE_BUILD_TYPE=Release`, Apple Clang, C++20,
> `-O3 -march=native -ffast-math -funroll-loops`.
>
> CPU Caches: L1 Data 64 KiB · L1 Instruction 128 KiB · L2 Unified 4096 KiB (×16)  
> Load Average: 5.45, 5.13, 4.51
>
> **Load average** is a Unix metric reported as three numbers: the average number of processes
> that were either running or waiting for CPU time over the last **1 minute**, **5 minutes**,
> and **15 minutes** respectively. A value equal to the number of logical CPU cores (here 16)
> means the machine is exactly fully utilised. Values below that indicate idle capacity;
> values above indicate a queue of waiting processes. `5.45 / 5.13 / 4.51` on a 16-core machine
> means ≈28–34% utilisation — moderate background load, numbers are still representative.

### double (f64)

```
-------------------------------------------------------------------------------
Benchmark                     Time             CPU   Iterations   GFLOP/s
-------------------------------------------------------------------------------
Naive/f64/N=64             54.3 µs          54.3 µs       13058    9.66 G/s
Naive/f64/N=256           12944 µs         12940 µs          55    2.59 G/s
Naive/f64/N=512          103159 µs        103151 µs           7    2.60 G/s
Naive/f64/N=1024         887768 µs        887678 µs           1    2.42 G/s
Naive/f64/N=4096      195990001 µs     195763155 µs           1  702.07 M/s

Reordered/f64/N=64         19.7 µs          19.4 µs       36153   27.06 G/s
Reordered/f64/N=256        2060 µs          2042 µs         342   16.43 G/s
Reordered/f64/N=512       16171 µs         16163 µs          43   16.61 G/s
Reordered/f64/N=1024     129521 µs        129467 µs           5   16.59 G/s
Reordered/f64/N=4096    8355984 µs       8350599 µs           1   16.46 G/s

Blocked/f64/N=64           19.2 µs          19.2 µs       36216   27.33 G/s  (tile=64)
Blocked/f64/N=256          1313 µs          1312 µs         535   25.57 G/s  (tile=64)
Blocked/f64/N=512         12641 µs         12636 µs          55   21.24 G/s  (tile=64)
Blocked/f64/N=1024       110532 µs        110486 µs           6   19.44 G/s  (tile=64)
Blocked/f64/N=4096      6778724 µs       6775811 µs           1   20.28 G/s  (tile=64)
```

**f64 speedup over `gemm_naive`:**

| N | Naive | Reordered | Speedup | Blocked | Speedup |
|---|---|---|---|---|---|
| 64 | 54.3 µs | 19.7 µs | **2.8×** | 19.2 µs | **2.8×** |
| 256 | 12944 µs | 2060 µs | **6.3×** | 1313 µs | **9.9×** |
| 512 | 103159 µs | 16171 µs | **6.4×** | 12641 µs | **8.2×** |
| 1024 | 887768 µs | 129521 µs | **6.9×** | 110532 µs | **8.0×** |
| 4096 | 195990001 µs | 8355984 µs | **23.5×** | 6778724 µs | **28.9×** |

### float (f32)

```
-------------------------------------------------------------------------------
Benchmark                     Time             CPU   Iterations   GFLOP/s
-------------------------------------------------------------------------------
Naive/f32/N=64             55.4 µs          55.4 µs       12586    9.46 G/s
Naive/f32/N=256           12190 µs         12181 µs          58    2.75 G/s
Naive/f32/N=512          108410 µs        108352 µs           6    2.48 G/s
Naive/f32/N=1024         822792 µs        822737 µs           1    2.61 G/s
Naive/f32/N=4096      204547448 µs     204194893 µs           1  673.08 M/s

Reordered/f32/N=64          6.12 µs          6.11 µs      112443   85.75 G/s
Reordered/f32/N=256         1052 µs          1052 µs         670   31.89 G/s
Reordered/f32/N=512         8234 µs          8234 µs          85   32.60 G/s
Reordered/f32/N=1024       65465 µs         65450 µs          11   32.81 G/s
Reordered/f32/N=4096     4189995 µs       4189264 µs           1   32.81 G/s

Blocked/f32/N=64            6.12 µs          6.12 µs      111957   85.69 G/s  (tile=64)
Blocked/f32/N=256            403 µs           403 µs        1738   83.31 G/s  (tile=64)
Blocked/f32/N=512           5224 µs          5224 µs         128   51.39 G/s  (tile=64)
Blocked/f32/N=1024         51943 µs         51351 µs          14   41.82 G/s  (tile=64)
Blocked/f32/N=4096       4382110 µs       4380989 µs           1   31.37 G/s  (tile=64)
```

**f32 speedup over `gemm_naive`:**

| N | Naive | Reordered | Speedup | Blocked | Speedup |
|---|---|---|---|---|---|
| 64 | 55.4 µs | 6.12 µs | **9.1×** | 6.12 µs | **9.1×** |
| 256 | 12190 µs | 1052 µs | **11.6×** | 403 µs | **30.2×** |
| 512 | 108410 µs | 8234 µs | **13.2×** | 5224 µs | **20.7×** |
| 1024 | 822792 µs | 65465 µs | **12.6×** | 51943 µs | **15.8×** |
| 4096 | 204547448 µs | 4189995 µs | **48.8×** | 4382110 µs | **46.7×** |

### Key observations

**Float vs double:**
- `gemm_naive` performs nearly identically for f32 and f64 — both are cache-miss-bound on column-stride access to B; the element size doesn't matter when you're waiting on DRAM.
- `gemm_reordered` delivers **~2× higher GFLOP/s in f32** (32–85 G/s) vs f64 (16–27 G/s). The compiler auto-vectorises the sequential inner j-loop with AVX2: 8 f32 elements per 256-bit register vs 4 f64. This is exactly the predicted SIMD-width doubling.
- `gemm_blocked` f32 reaches **83 GFLOP/s at N=256** — the entire tile fits in L1, the compiler vectorises fully, and AVX2 FMA runs at near-peak throughput. This is 3× better than the f64 equivalent (25.6 G/s) at the same size.
- The blocked f32 kernel **drops off at N=4096** (31.4 G/s, below reordered's 32.8 G/s) — with large N the tile is evicted between outer-loop iterations and the tiling benefit is reduced. This is the clearest argument for adding a second level of tiling (L3-blocking) in a future step.

**Headline numbers (best scalar+auto-vectorised throughput):**

| Kernel | f64 peak | f32 peak | f32/f64 ratio |
|---|---|---|---|
| `gemm_naive` | 9.66 G/s | 9.46 G/s | 1.0× |
| `gemm_reordered` | 27.06 G/s | 85.75 G/s | **3.2×** |
| `gemm_blocked` | 27.33 G/s | 85.69 G/s | **3.1×** |

---

## Deriving GFLOP/s

```
GFLOP/s = (2 × N³) / (time_µs × 1000)
```

Example: `gemm_blocked` f32, N=256, 403 µs → `2 × 256³ / (403 × 1000)` ≈ **83.3 GFLOP/s** — the entire 3×64×64×4 B = 48 KB tile fits in L1, the compiler auto-vectorises with AVX2 (8 f32/register), and FMA runs at near-peak. With explicit AVX-512 intrinsics (16 f32/register) the theoretical headroom is still ~2×.

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
