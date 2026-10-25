# CPU SIMD: FMA limits, micro-kernels and the kernel ladder

Raw data behind the article
[*Matrix multiplication with SIMD on Apple M4 and AMD Zen 5*](../../articles/simd-gemm.md).
2026-10-15, one thread. These files record the kernels as they were then;
the current register tiles are measured in [tile-sweeps](../tile-sweeps/README.md).

| Machine | Compiler | Flags | Load |
|---|---|---|---|
| Apple M4 Max, P-core | Apple Clang 17 | `-O3 -mcpu=apple-m4 -ffast-math -funroll-loops` | load average 5–7, including two unrelated processes at 100% each |
| AMD Ryzen 9 9950X (Zen 5), Ubuntu 24.04 | GCC 13.3 | `-O3 -march=native -ffast-math -funroll-loops` | idle, `performance` governor |

| File | Program | What it measures |
|---|---|---|
| [`fma_m4.csv`](fma_m4.csv), [`fma_zen5.csv`](fma_zen5.csv) | [`fma_peak`](../../../benchmarks/experiments/fma_peak.cpp) | vector FMA latency and throughput against the number of independent accumulators |
| [`microkernel_m4.csv`](microkernel_m4.csv), [`microkernel_zen5.csv`](microkernel_zen5.csv) | [`simd_microkernel`](../../../benchmarks/experiments/simd_microkernel.cpp) | the GEMM register-tile micro-kernel on L1-resident data: rows × vectors, B packed or with rows N apart |
| [`bench_m4.csv`](bench_m4.csv), [`bench_zen5.csv`](bench_zen5.csv) | [`bench_gemm`](../../../benchmarks/bench_gemm.cpp) | every CPU kernel, N = 64 … 4096 (Google Benchmark CSV; GFLOP/s in the last column) |
| [`bench_m4_neon_f64_8x4.csv`](bench_m4_neon_f64_8x4.csv), [`bench_zen5_avx512_8x2.csv`](bench_zen5_avx512_8x2.csv) | `bench_gemm` | the two kernels whose micro-kernel changed (and their prefetch variants), re-measured after the change; the AVX-512 file is a re-run on 2026-10-16, within 1% of the same-night run quoted in the tables |

Cycles are at the core clock measured at the start of each program with a
chain of dependent integer adds ([`core_clock.hpp`](../../../benchmarks/experiments/core_clock.hpp)):
5.71 GHz on Zen 5 (identical before and after every run), 3.96–4.13 GHz on
the M4, whose clock moved with the background load between programs.

## FMA latency and throughput

`fma_peak`: a loop of `acc[c] = fma(acc[c], a, b)` on C named register
accumulators, no memory traffic, best of 5. The generated loops were
checked: C FMAs, one decrement and one branch (plus two `dup` on the M4
for C ≥ 8 in f32).

FMAs per cycle (f32; f64 within 0.1 of these):

| Accumulators | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 24 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Zen 5, AVX-512 | 0.25 | 0.50 | 1.00 | 1.83 | **1.98** | 1.98 | | 1.98 |
| Zen 5, AVX2 | 0.25 | 0.50 | 1.00 | 1.83 | **1.99** | | | |
| M4, NEON f32 | 0.30 | 0.58 | 1.22 | 2.01 | 2.81 | **3.46** | 2.97 | 3.05 |
| M4, NEON f64 | 0.29 | 0.57 | 1.24 | 2.03 | 2.84 | 3.77 | **3.97** | 3.29 |

- Zen 5: latency 4.0 cycles, 2 FMAs per cycle; 12 accumulators reach the
  full rate. Peak at 5.71 GHz: AVX-512 362 GFLOP/s f32 and 180 f64 in this
  file (365 / 182 in an identical earlier run the same day); AVX2 182 / 91.
- M4: one accumulator takes 3.3–3.5 cycles per FMA, and throughput rises to
  ~4 FMAs per cycle (3.97 measured, f64, 20 accumulators). Peak at 4.05 GHz:
  ~130 GFLOP/s f32, ~65 f64; the best measured were 111 and 63. With 20 or
  24 accumulators f32 falls back: the two `dup` per round compete for the
  same four pipes.

## Micro-kernel

`simd_microkernel`: the micro-kernel of `gemm_*_blocked` (R rows × V
vectors of C in registers; per k step V loads of B, R broadcasts of A,
R·V FMAs; k = 0 … 255), called 200,000 times on the same A, B and C so
everything is in L1. Best of 5. GFLOP/s:

| | Accumulators | B packed | B rows 256 apart | B rows 1024 apart |
|---|--:|--:|--:|--:|
| Zen 5 AVX-512 f32, 4 × 2 (old library tile) | 8 | 143.7 | 128.9 | 81.8 |
| Zen 5 AVX-512 f32, 6 × 2 | 12 | 348.5 | 312.4 | 210.8 |
| Zen 5 AVX-512 f32, 8 × 2 (library tile) | 16 | **352.4** | 335.6 | 275.5 |
| Zen 5 AVX-512 f64, 8 × 2 (library tile) | 16 | **174.1** | 166.0 | 125.6 |
| Zen 5 AVX2 f32, 4 × 2 (library tile) | 8 | 154.5 | 151.1 | 66.6 |
| Zen 5 AVX2 f32, 6 × 2 | 12 | 177.5 | 174.7 | 116.3 |
| M4 NEON f32, 4 × 4 (library tile) | 16 | 112.2 | 109.9 | 104.0 |
| M4 NEON f32, 5 × 4 | 20 | **118.0** | 117.2 | 116.5 |
| M4 NEON f64, 4 × 2 (old library tile) | 8 | 31.0 | 31.0 | 30.6 |
| M4 NEON f64, 8 × 2 (library tile) | 16 | **58.2** | 50.4 | 48.5 |

The 4 × 2 AVX-512 kernel is slower than its 8 accumulators allow because
GCC 13 compiles it with the B loads folded into the FMAs as memory operands
(`vfmadd231ps (mem), …`): its loop body, unrolled over four k steps, has
32 FMAs with memory operands and no separate B load, so B is re-loaded for
every FMA. With 6 or more rows it loads B once per k step (4 loads per two
k steps) and every FMA is
register-only. The library kernels `gemm_simd_blocked<Zmm>` and
`gemm_simd_blocked<Q>` (f64) were changed to the 8-row tiles on this evidence;
before and after, same session:

| N | 256 | 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|
| `gemm_simd_blocked<Zmm>` f32, 4 × 2 → 8 × 2 | 216.8 → 332.7 | 205.2 → 327.9 | 141.3 → 284.6 | 119.9 → 198.2 |
| `gemm_simd_blocked<Zmm>` f64, 4 × 2 → 8 × 2 | 106.2 → 161.5 | 55.8 → 105.2 | 55.1 → 90.9 | 51.1 → 77.2 |
| `gemm_simd_blocked<Q>` f64, 4 × 2 → 8 × 2 | 31.5 → 39.6 | 29.8 → 39.2 | 28.7 → 37.9 | 24.0 → 33.1 |

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release          # add -DHPC_ENABLE_AVX512=ON on x86
cmake --build build --target fma_peak simd_microkernel bench_gemm
./build/benchmarks/fma_peak > fma.csv
./build/benchmarks/simd_microkernel > microkernel.csv
./build/benchmarks/bench_gemm --benchmark_format=csv > bench.csv
```

## AVX2 register tile: 4 × 2 → 6 × 2 (2026-10-21)

Files in [`avx2-tiles/`](avx2-tiles/). Same machine and compiler, one core
pinned (`taskset -c 2`), idle. The comparison of all tile shapes is in
[tile-sweeps](../tile-sweeps/README.md) and [docs/kernels/avx2.md](../../kernels/avx2.md).

| File | Program | What |
|---|---|---|
| `library_{native,v3}_{old,new}_{1,2,3}.csv` | `bench_gemm`, filter `^(Avx2Blocked\|Avx512Blocked)(Pf4)?/` | the library before (git HEAD, 4×2 AVX2 tile) and after (6×2), three alternating runs per build; AVX-512 rows check that its rewrite to the `vector_micro<Zmm<T>, 8, 2>` template is neutral (±0.7%) |

Later tile changes (AVX-512 8 × 2 → 8 × 3, NEON → 6 × 4 with `LDR`) are
recorded in [tile-sweeps](../tile-sweeps/README.md); the tables above keep
the kernels as they were measured on 2026-10-15.
