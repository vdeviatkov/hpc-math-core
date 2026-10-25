# AVX2 register tiles (`gemm_simd_blocked<Ymm>`)

`gemm_simd_blocked<Ymm, R, V, …>` multiplies with an R × V register tile:
R rows of C, V vectors of 8 FP32 / 4 FP64 each, kept in R·V YMM
accumulators for a whole k block. The default is **6 × 2** (6 × 16 FP32,
6 × 8 FP64). This page gives the measurements behind that choice on AMD
Zen 5. Code: [`avx2.hpp`](../../src/gemm/avx2.hpp) (`vector_micro<Ymm<T>, R, V>`),
driver [`simd.hpp`](../../src/gemm/simd.hpp). The other
families: [AVX-512](avx512.md), [NEON](neon.md).

## What a tile shape decides

Per k step the micro-kernel loads V vectors of B row k, broadcasts R
elements of A and issues R·V FMAs. Three numbers follow from R and V:

- **Independent accumulators, R·V.** Zen 5 issues 2 FMAs per cycle with a
  4-cycle latency (`fma_peak`, [experiments/cpu-simd](../experiments/cpu-simd/README.md)),
  so 8 accumulators are the minimum and 12 reach the full rate.
- **Registers, R·V + V + 1.** The accumulators, the V vectors of B (live
  for the whole k step) and one broadcast of A (dead after its V FMAs).
  AVX2 has 16 YMM registers. On Zen 5, `-march=native` also enables
  AVX-512VL, which gives 256-bit code 32 YMM registers; a build for an
  AVX2-only CPU (`-DHPC_MARCH=x86-64-v3`) has 16.
- **B traffic, 32 / R bytes per FMA.** Within a cache block (256 k × 256 FP32 columns),
  the same R × 256 piece of A (6 KiB FP32 for R = 6) serves the 16 calls of
  the j loop and stays in L1; every call reads a different 256 × V-vector
  strip of B out of a 256 KiB block that L1 (48 KiB) cannot hold. B
  arrives from L2, and each row added to the tile reuses each B load once
  more.

## Tile sweep, Zen 5

`bench_gemm`'s `Avx2Tile<R>x<V>` benchmarks: the library kernel with each
shape, GFLOP/s, median of 3, pinned to one core
([data](../experiments/tile-sweeps/README.md)). Best per column in bold.

### 16 YMM registers (`-DHPC_MARCH=x86-64-v3`), the AVX2-only target

FP32:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 152.0 | 151.2 | 142.6 | 92.7 | 75.9 |
| 2 × 4 | 8 | 132.6 | 135.5 | 129.5 | 63.0 | 60.0 |
| 8 × 1 | 8 | 143.8 | 143.4 | 138.0 | 92.2 | 74.9 |
| 3 × 3 | 9 | 145.7 | 149.5 | 135.6 | 84.6 | 73.9 |
| 5 × 2 | 10 | 162.1 | 163.2 | 160.1 | 111.4 | 85.7 |
| **6 × 2** (default) | 12 | **168.1** | **169.0** | **166.6** | **126.7** | 94.1 |
| 4 × 3 | 12 | 155.4 | 154.2 | 150.9 | 108.2 | 87.9 |
| 3 × 4 | 12 | 151.1 | 152.1 | 147.5 | 108.0 | **94.3** |
| 7 × 2 (spills) | 14 | 108.7 | 97.1 | 95.8 | 90.7 | 83.8 |

FP64:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 74.9 | 70.7 | 46.2 | 41.4 | 37.2 |
| 2 × 4 | 8 | 66.3 | 64.7 | 31.9 | 31.9 | 30.8 |
| 8 × 1 | 8 | 71.5 | 69.5 | 45.9 | 40.7 | 37.5 |
| 3 × 3 | 9 | 72.9 | 68.5 | 43.2 | 40.2 | 37.3 |
| 5 × 2 | 10 | 79.8 | 79.5 | 55.5 | 47.7 | 42.8 |
| **6 × 2** (default) | 12 | **82.6** | **83.7** | **62.9** | **53.7** | 47.4 |
| 4 × 3 | 12 | 73.7 | 74.9 | 55.2 | 53.6 | **47.9** |
| 3 × 4 | 12 | 75.2 | 75.0 | 54.6 | 53.1 | 46.3 |
| 7 × 2 (spills) | 14 | 53.3 | 47.8 | 45.2 | 43.0 | 41.5 |

### 32 YMM registers (`-march=native` on Zen 5), the build of the main tables

FP32:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 152.1 | 151.1 | 142.3 | 93.6 | 76.3 |
| 2 × 4 | 8 | 133.3 | 136.0 | 127.0 | 66.7 | 61.9 |
| 8 × 1 | 8 | 145.8 | 148.8 | 141.0 | 94.7 | 79.2 |
| 3 × 3 | 9 | 146.6 | 150.4 | 136.5 | 84.5 | 74.6 |
| 5 × 2 | 10 | 165.9 | 166.5 | 162.0 | 111.9 | 86.0 |
| **6 × 2** (default) | 12 | 168.5 | **170.2** | **167.1** | 125.8 | 95.0 |
| 4 × 3 | 12 | 165.8 | 169.7 | 166.9 | 116.7 | 93.3 |
| 3 × 4 | 12 | **168.8** | 168.0 | 163.7 | 111.1 | 93.0 |
| 7 × 2 | 14 | 166.1 | 166.4 | 162.1 | **130.1** | **102.1** |

FP64:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 75.0 | 70.9 | 46.6 | 41.8 | 37.5 |
| 2 × 4 | 8 | 66.6 | 65.2 | 34.9 | 34.6 | 32.2 |
| 8 × 1 | 8 | 72.2 | 68.4 | 46.8 | 43.3 | 39.4 |
| 3 × 3 | 9 | 74.3 | 68.5 | 42.5 | 40.3 | 37.7 |
| 5 × 2 | 10 | 81.9 | 80.5 | 55.0 | 48.2 | 43.2 |
| **6 × 2** (default) | 12 | 82.2 | **83.8** | 61.6 | 54.3 | 48.1 |
| 4 × 3 | 12 | 81.1 | 83.0 | 58.1 | 52.4 | 47.6 |
| 3 × 4 | 12 | **83.6** | 81.7 | 55.2 | 52.6 | 47.3 |
| 7 × 2 | 14 | 82.5 | 82.0 | **64.8** | **56.9** | **51.2** |

Variation: coefficient of variation ≤ 1.1% in every cell.

The sweep ran with the cache block of the time, 64 × 256 × 256 (TileM × TileK
× TileN); the blocks were swept afterwards with the chosen tile
([blocks.md](blocks.md)). The tiles were not re-swept with the new
blocks.

### Registers and spills

GCC 13, from the disassembly of every `vector_micro<Ymm<float>, R, V>` shape
above. Only one shape spills:

| Build | Shape | Vector registers used | Stack accesses in the k loop |
|---|---|--:|--:|
| x86-64-v3 (16 YMM) | every shape except 7 × 2 | ≤ 16 | 0 |
| x86-64-v3 (16 YMM) | 7 × 2 | 16 | 14 stores + 14 reloads |
| native (32 YMM) | every shape | 14–23 | 0 |

6 × 2 needs 15 registers by the count above; GCC uses all 16 by
alternating two broadcast registers, still with no spill.

## Why 6 × 2

- **Accumulators first.** The 8-accumulator shapes (4 × 2, 2 × 4, 8 × 1)
  sit at exactly latency × pipes with no margin and stay 10–23% below
  6 × 2 at N = 256. On L1-resident data
  (`simd_microkernel`) the same rule shows without cache effects: 4, 5 and
  6 rows × 2 vectors reach 152.2, 167.6 and 175.0 GFLOP/s FP32, 83%, 92%
  and 96% of the 182 GFLOP/s AVX2 peak.
- **Then rows over vectors.** Among the 12-accumulator shapes, 6 × 2 reads
  5.3 bytes of B per FMA, 4 × 3 8 and 3 × 4 10.7. Where B stops fitting in
  L1, at N = 1024 FP32 (4 KiB row stride, see
  [Part 3](../articles/sme2-cache-associativity.md#the-same-experiment-on-amd-zen-5)),
  6 × 2 leads by 17% (126.7 vs 108.2 / 108.0). 2 × 4, with 16 bytes per
  FMA, is the slowest shape everywhere. The L1 micro-kernel with B rows
  4 KiB apart shows the same order: 66.2 / 83.3 / 115.3 GFLOP/s for 4, 5
  and 6 rows.
- **Then the register file.** One more row, 7 × 2, needs 17 registers. In
  the 16-register build it spills accumulators to the stack inside the k
  loop and falls to 96–109 GFLOP/s FP32 at N ≤ 512, −35 to −43% against 6 × 2.
- **The default targets 16 registers.** With 32 YMM registers (the native
  Zen 5 build) 7 × 2 does not spill and is the best at N ≥ 512 FP64 and
  N ≥ 1024 FP32 (+3–8%), since it reads less B per FMA. But an AVX2 kernel
  exists for CPUs without AVX-512, which have 16 registers; a CPU with 32
  is better served by `gemm_simd_blocked<Zmm>`. `gemm_simd_blocked<Ymm, 7, 2, …>`
  is there for anyone who wants it.

The 6 × 2 kernel reaches 94% (FP32) and 92% (FP64) of the AVX2 peak at
N = 256. Past N = 512 what remains is B's row stride, not the tile:
[Part 1](../articles/simd-gemm.md#what-remains-bs-row-stride).

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release                         # 32 YMM on Zen 5
cmake -B build-v3 -DCMAKE_BUILD_TYPE=Release -DHPC_MARCH=x86-64-v3 # 16 YMM
cmake --build build --target bench_gemm simd_microkernel
taskset -c 2 ./build/benchmarks/bench_gemm --benchmark_filter='^Avx2Tile' \
    --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
taskset -c 2 ./build/benchmarks/simd_microkernel
```

A shape not in the sweep: `gemm_simd_blocked<Ymm, R, V, TileM, TileK, TileN>(A, B, C)`, or a
line `HPC_AVX2_TILE(R, V);` in `add_tiles()` of
[`bench_gemm.cpp`](../../benchmarks/bench_gemm.cpp).
