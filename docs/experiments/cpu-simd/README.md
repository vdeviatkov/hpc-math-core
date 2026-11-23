# CPU SIMD: FMA limits and the scalar kernel ladder

Raw data behind the article
[*Matrix multiplication with SIMD on Apple M4 and AMD Zen 5*](../../articles/simd-gemm.md):
the FMA latency and throughput of both cores, and the scalar kernels
(`gemm_naive`, `gemm_reordered`, `gemm_blocked`, the SIMD reordered loops
and the scalar prefetch variants). One thread. The register-tiled SIMD
kernels and the micro-kernel on L1-resident data are in
[tile-sweeps](../tile-sweeps/README.md) and
[block-sweeps](../block-sweeps/README.md).

| Machine | Compiler | Flags | Load |
|---|---|---|---|
| Apple M4 Max, P-core | Apple Clang 17 | `-O3 -mcpu=apple-m4 -ffast-math -funroll-loops` | load average 5–7, including two unrelated processes at 100% each |
| AMD Ryzen 9 9950X (Zen 5), Ubuntu 24.04 | GCC 13.3 | `-O3 -march=native -ffast-math -funroll-loops` | idle, `performance` governor |

| File | Program | What it measures |
|---|---|---|
| [`fma_m4.csv`](fma_m4.csv), [`fma_zen5.csv`](fma_zen5.csv) | [`fma_peak`](../../../benchmarks/experiments/fma_peak.cpp) | vector FMA latency and throughput against the number of independent accumulators |
| [`bench_m4.csv`](bench_m4.csv), [`bench_zen5.csv`](bench_zen5.csv) | [`bench_gemm`](../../../benchmarks/bench_gemm.cpp) | the scalar kernels, the SIMD reordered loops and the scalar prefetch variants, N = 64 … 4096 (Google Benchmark CSV; GFLOP/s in the last column) |

Cycles are at the core clock measured at the start of each program with a
chain of dependent integer adds ([`core_clock.hpp`](../../../benchmarks/experiments/core_clock.hpp)):
5.71 GHz on Zen 5, 3.96–4.13 GHz on the M4, whose clock moved with the
background load between programs. (The register-tiled M4 kernels ran at
4.48–4.50 GHz; compare M4 numbers from different runs in cycles, not
GFLOP/s.)

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
  full rate. Peak at 5.71 GHz: AVX-512 362 GFLOP/s f32 and 180 f64; AVX2
  182 / 91.
- M4: one accumulator takes 3.3–3.5 cycles per FMA, and throughput rises to
  ~4 FMAs per cycle (3.97 measured, f64, 20 accumulators). Peak at 4.05 GHz:
  ~130 GFLOP/s f32, ~65 f64; at 4.49 GHz ~144 / ~72. With 20 or 24
  accumulators f32 falls back: the two `dup` per round compete for the
  same four pipes.

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release          # add -DHPC_ENABLE_AVX512=ON on x86
cmake --build build --target fma_peak bench_gemm
./build/benchmarks/fma_peak > fma.csv
./build/benchmarks/bench_gemm --benchmark_format=csv \
    --benchmark_filter='^(Naive|Reordered|Blocked|Avx2Reordered|Avx512Reordered|NeonReordered)(Pf[0-9]+)?/' > bench.csv
```
