# AVX2 register tiles (`gemm_simd_blocked<Ymm>`)

`gemm_simd_blocked<Ymm, R, V, …>` multiplies with an R × V register tile:
R rows of C, V vectors of 8 FP32 / 4 FP64 each, kept in R·V YMM
accumulators for a whole k block. The default is **6 × 2**, which covers a
6 × 16 block of C in FP32 and 6 × 8 in FP64. This page gives the measurements behind that choice on AMD
Zen 5. Code: [`avx2.hpp`](../../src/gemm/avx2.hpp) (`vector_micro<Ymm<T>, R, V>`),
driver [`simd.hpp`](../../src/gemm/simd.hpp). The other
families: [AVX-512](avx512.md), [NEON](neon.md).

## What a tile shape decides

Per k step the micro-kernel loads V vectors of B row k, broadcasts R
elements of A and issues R·V FMAs. Three numbers follow from R and V:

- **Independent accumulators, R·V.** Zen 5 issues 2 FMAs per cycle with a
  4-cycle latency (`fma_peak`, [experiments/cpu-simd](../experiments/cpu-simd/README.md)),
  so in theory 2 × 4 = 8 independent accumulators keep both FMA units busy.
  In practice 8 give 1.83 FMAs per cycle (92%). With 8, each accumulator
  becomes ready exactly when its next turn comes, so no other FMA is
  waiting: any one-cycle delay (loop overhead, issue order) leaves an FMA
  unit idle. With 12, ready FMAs are always waiting, so another one fills
  the gap; 12 give 1.99, the full rate.
- **Registers, R·V + V + 1.** The accumulators, the V vectors of B (live
  for the whole k step) and one broadcast of A (dead after its V FMAs).
  AVX2 has 16 YMM registers. On Zen 5, `-march=native` also enables
  AVX-512VL, which gives 256-bit code 32 YMM registers; a build for an
  AVX2-only CPU (`-DHPC_MARCH=x86-64-v3`) has 16.
- **B traffic, 32 / R bytes per FMA.** Each k step loads V vectors of B
  (32 bytes each) and uses each in R FMAs, one per row: 32·V bytes for
  R·V FMAs. Only the rows reduce it. This is the traffic that counts,
  because B comes from L2 and A from L1. The driver works on a block of B
  at a time: 256 rows (k) × 256 columns, 256 KiB in FP32, larger than L1
  (48 KiB). One R-row strip of A (R × 256 k, 6 KiB FP32 for R = 6) is
  multiplied by all 16 tiles across that block's columns and stays in L1.
  Each tile reads a different 256 × V-vector strip of B, so each strip
  of A reads the whole 256 KiB block of B, and the next strip of A reads
  it all again.

## Tile sweep, Zen 5

`bench_gemm`'s `Avx2Tile<R>x<V>` benchmarks: the library kernel with each
shape, GFLOP/s, median of 3, pinned to one core
([data](../experiments/tile-sweeps/README.md)). Best per column in bold.

### 16 YMM registers (`-DHPC_MARCH=x86-64-v3`), the AVX2-only target

FP32:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 152.0 | 152.1 | 144.7 | 93.5 | 81.2 |
| 2 × 4 | 8 | 132.8 | 135.3 | 128.3 | 61.9 | 61.2 |
| 8 × 1 | 8 | 144.6 | 146.1 | 140.5 | 92.2 | 78.1 |
| 3 × 3 | 9 | 146.0 | 152.1 | 142.1 | 87.9 | 80.5 |
| 5 × 2 | 10 | 162.4 | 162.7 | 161.5 | 112.5 | 93.2 |
| **6 × 2** (default) | 12 | **168.8** | **170.8** | **168.3** | **128.4** | 103.8 |
| 4 × 3 | 12 | 154.9 | 155.6 | 152.7 | 109.4 | 97.1 |
| 3 × 4 | 12 | 151.6 | 155.5 | 153.4 | 113.2 | **108.6** |
| 7 × 2 (spills) | 14 | 108.8 | 99.2 | 98.8 | 96.4 | 91.0 |

FP64:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 75.3 | 70.6 | 46.5 | 41.7 | 40.3 |
| 2 × 4 | 8 | 66.4 | 63.2 | 31.3 | 31.1 | 32.7 |
| 8 × 1 | 8 | 71.4 | 68.9 | 45.6 | 40.6 | 39.1 |
| 3 × 3 | 9 | 72.9 | 69.4 | 43.4 | 40.8 | 40.7 |
| 5 × 2 | 10 | 79.2 | 78.5 | 55.3 | 48.0 | 47.0 |
| **6 × 2** (default) | 12 | **82.6** | **84.6** | **63.5** | 54.0 | 52.8 |
| 4 × 3 | 12 | 73.7 | 75.8 | 53.8 | 49.3 | 48.5 |
| 3 × 4 | 12 | 75.3 | 75.4 | 56.5 | **54.3** | **53.5** |
| 7 × 2 (spills) | 14 | 54.3 | 49.1 | 48.0 | 45.7 | 45.6 |

### 32 YMM registers (`-march=native` on Zen 5), the build of the main tables

FP32:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 150.8 | 150.8 | 144.2 | 93.6 | 82.5 |
| 2 × 4 | 8 | 133.8 | 135.8 | 127.0 | 68.3 | 65.9 |
| 8 × 1 | 8 | 146.7 | 150.7 | 143.2 | 95.8 | 83.8 |
| 3 × 3 | 9 | 148.3 | 153.6 | 143.4 | 87.7 | 82.1 |
| 5 × 2 | 10 | 160.4 | 160.0 | 158.7 | 112.4 | 93.3 |
| **6 × 2** (default) | 12 | 167.3 | 172.1 | 168.5 | 126.2 | 105.4 |
| 4 × 3 | 12 | 165.5 | 169.4 | 167.2 | 117.1 | 102.3 |
| 3 × 4 | 12 | **170.0** | 170.1 | 168.5 | 115.6 | 103.9 |
| 7 × 2 | 14 | 166.7 | **172.7** | **172.8** | **139.8** | **117.5** |

FP64:

| Tile (rows × vectors) | Accumulators | N = 64 | N = 256 | N = 512 | N = 1024 | N = 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 4 × 2 | 8 | 75.2 | 71.0 | 46.9 | 42.2 | 40.6 |
| 2 × 4 | 8 | 66.9 | 64.9 | 33.8 | 34.2 | 34.5 |
| 8 × 1 | 8 | 72.5 | 68.8 | 47.0 | 43.2 | 41.3 |
| 3 × 3 | 9 | 74.3 | 70.1 | 43.8 | 41.1 | 40.9 |
| 5 × 2 | 10 | 82.0 | 81.1 | 56.6 | 49.0 | 47.8 |
| **6 × 2** (default) | 12 | 82.9 | 84.8 | 62.4 | 55.4 | 54.1 |
| 4 × 3 | 12 | 81.2 | 83.4 | 58.5 | 52.1 | 51.6 |
| 3 × 4 | 12 | **83.9** | 83.2 | 56.7 | 54.2 | 52.8 |
| 7 × 2 | 14 | 83.0 | **85.5** | **69.6** | **60.5** | **59.3** |

Variation: coefficient of variation ≤ 2.9% in every cell.

The sweep runs each shape at the default cache blocks (384 × 256 × 256
FP32, 384 × 256 × 128 FP64, TileM × TileK × TileN;
[blocks.md](blocks.md)).

**B stays in registers in every shape.** Unlike AVX-512 with GCC (see
[avx512.md](avx512.md)), no AVX2 shape reads B inside its FMAs: in the
running loop of 4 × 2 and 6 × 2 (`perf annotate`, both builds) every FMA
is register-only and B arrives through `vmovups`.

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
  sit at exactly latency × pipes with no margin and stay 11–21% below
  6 × 2 at N = 256. On L1-resident data
  (`simd_microkernel`) the same rule shows without cache effects: 4, 5 and
  6 rows × 2 vectors reach 152.2, 167.6 and 175.0 GFLOP/s FP32, 83%, 92%
  and 96% of the 182 GFLOP/s AVX2 peak.
- **Then rows over vectors.** Among the 12-accumulator shapes, 6 × 2 reads
  5.3 bytes of B per FMA, 4 × 3 8 and 3 × 4 10.7. Where B stops fitting in
  L1, at N = 1024 FP32 (4 KiB row stride, see
  [Part 3](../articles/sme2-cache-associativity.md#the-same-experiment-on-amd-zen-5)),
  6 × 2 leads by 13–17% (128.4 vs 109.4 / 113.2). 2 × 4, with 16 bytes per
  FMA, is the slowest shape everywhere. The L1 micro-kernel with B rows
  4 KiB apart shows the same order: 66.2 / 83.3 / 115.3 GFLOP/s for 4, 5
  and 6 rows.
- **Large N is a near-tie.** At N = 4096, 3 × 4 leads 6 × 2 by 4.6% FP32
  (and 1.3% FP64), but at N ≤ 1024 6 × 2 leads 3 × 4 by 10–13% FP32
  (170.8 against 155.5 at N = 256), so 6 × 2 is kept. Counted with `perf`
  at N = 4096, 3 × 4's lead does not come from fewer misses: it has 22%
  more L1 misses than 6 × 2, the same TLB misses and 2.6× the L2
  prefetches ([data](../experiments/zen5-counters/avx2_3x4.txt)).
- **Then the register file.** One more row, 7 × 2, needs 17 registers. In
  the 16-register build it spills accumulators to the stack inside the k
  loop and falls to 99–109 GFLOP/s FP32 at N ≤ 512, −36 to −42% against 6 × 2.
- **The default targets 16 registers.** With 32 YMM registers (the native
  Zen 5 build) 7 × 2 does not spill and is the best from N = 256 up in
  both types (+9–12% over 6 × 2 at N ≥ 1024). But an AVX2 kernel
  exists for CPUs without AVX-512, which have 16 registers; a CPU with 32
  is better served by `gemm_simd_blocked<Zmm>`. `gemm_simd_blocked<Ymm, 7, 2, …>`
  is there for anyone who wants it.

The 6 × 2 kernel reaches 94% (FP32) and 93% (FP64) of the AVX2 peak at
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
