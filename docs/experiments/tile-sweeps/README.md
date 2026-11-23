# Register-tile sweeps

Raw data behind [docs/kernels](../../kernels/README.md): the AVX2,
AVX-512 and NEON blocked kernels run with every register-tile shape in
`bench_gemm`'s `add_tiles()`, and the micro-kernel alone on L1-resident
data. One thread.

| Machine | Compiler | Flags | Load |
|---|---|---|---|
| AMD Ryzen 9 9950X (Zen 5), Ubuntu 24.04 | GCC 13.3 | `-O3 -march=native -ffast-math -funroll-loops` (native), or `-march=x86-64-v3` (16 YMM registers); pinned with `taskset -c 2` | idle, load average < 1.2 |
| Apple M4 Max, P-core | Apple Clang 17 | `-O3 -mcpu=apple-m4 -ffast-math -funroll-loops` | load average 5–14 on 16 cores (Spotlight indexing and an IDE, 2–3 of 12 P-cores busy) |

Core clock, measured by `simd_microkernel` at its start and end
([`core_clock.hpp`](../../../benchmarks/experiments/core_clock.hpp)):
5.70 GHz on Zen 5 and 4.48–4.50 GHz on the M4 in every run here.

The sweeps ran with a 64 × 256 × 256 cache block (AVX-512: 64 × 256 × 512);
the blocks were swept afterwards with the chosen tiles
([block-sweeps](../block-sweeps/README.md)).

Google Benchmark CSVs hold the mean, median, standard deviation and
coefficient of variation (`_cv`) of 3 repetitions; GFLOP/s is the last
column. The docs quote medians.

| File | Program | What |
|---|---|---|
| [`tiles_zen5_native.csv`](tiles_zen5_native.csv) | `bench_gemm --benchmark_filter='^(Avx2Tile\|Avx512Tile)'` | Zen 5, every AVX2 and AVX-512 shape, native build (32 vector registers) |
| [`tiles_zen5_x86-64-v3.csv`](tiles_zen5_x86-64-v3.csv) | `bench_gemm --benchmark_filter='^Avx2Tile'` | Zen 5, every AVX2 shape, 16-register build |
| [`microkernel_zen5.csv`](microkernel_zen5.csv), [`microkernel_zen5_clock.txt`](microkernel_zen5_clock.txt) | [`simd_microkernel`](../../../benchmarks/experiments/simd_microkernel.cpp) | the micro-kernels on L1-resident data, Zen 5 |
| [`tiles_m4.csv`](tiles_m4.csv) | `bench_gemm --benchmark_filter='^NeonTile'` | M4, every NEON shape |
| [`microkernel_m4.csv`](microkernel_m4.csv), [`microkernel_m4_clock.txt`](microkernel_m4_clock.txt), [`load_m4.txt`](load_m4.txt) | `simd_microkernel`, `uptime` | the same run, continued |
| [`m4_ld1r_ab/`](m4_ld1r_ab/) | `bench_gemm --benchmark_filter='^NeonTile(4x4\|6x3\|6x4)/(f32\|f64)/N=(256\|1024)$'` | A broadcast with `LD1R` against A loaded with `LDR` and multiplied by element ([docs/kernels/neon.md](../../kernels/neon.md#loading-a-ldr-not-ld1r)), the two builds alternated three times (`ld1r_<run>.csv`, `ldr_<run>.csv`), one repetition each |

## Micro-kernel on L1-resident data

`simd_microkernel`: the micro-kernel of `gemm_simd_blocked` (R rows × V
vectors of C in registers; per k step V loads of B, R broadcasts of A,
R·V FMAs; k = 0 … 255), called 200,000 times on the same A, B and C so
everything is in L1. Best of 5. GFLOP/s, selected rows:

| | Accumulators | B packed | B rows 1 KiB apart (f32) | B rows 4 KiB apart (f32) |
|---|--:|--:|--:|--:|
| Zen 5 AVX-512 f32, 4 × 2 | 8 | 142.5 | 127.8 | 83.8 |
| Zen 5 AVX-512 f32, 6 × 2 | 12 | 345.5 | 311.3 | 210.3 |
| Zen 5 AVX-512 f32, 8 × 2 | 16 | **349.5** | 334.3 | 274.6 |
| Zen 5 AVX-512 f32, 12 × 2 | 24 | 349.1 | 343.9 | 342.5 |
| Zen 5 AVX2 f32, 4 × 2 | 8 | 152.2 | 149.9 | 66.2 |
| Zen 5 AVX2 f32, 6 × 2 | 12 | **175.0** | 172.9 | 115.3 |
| M4 NEON f64, 4 × 2 | 8 | 33.6 | 32.6 | 33.0 |
| M4 NEON f64, 8 × 2 | 16 | **58.6** | 51.9 | 51.2 |

(The strides are `ldb` = 256 and 1024 elements; for f64 the byte strides
are twice those of the f32 column headers.)

GCC 13 compiles the AVX-512 shapes with 4 or fewer rows with the B loads
folded into the FMAs as memory operands (`vfmadd231ps (mem), …`), so B is
re-loaded for every FMA; with 6 or more rows it loads B once per k step
and every FMA is register-only. That, not the accumulator count, is why
4 × 2 runs at 40% of peak here.

## Notes

- At Zen 5 AVX-512 FP64 N = 512 (B rows 4 KiB apart) the sweep result
  depends on where the allocator places the three matrices: `Avx512Tile8x3`
  measured 150.9 GFLOP/s in the sweep and 126–128 run alone; 8 × 2
  measures 101–103 either way.
- One cell has a high coefficient of variation: `Avx512Tile12x2/f64/N=1024`,
  11.5%. Every other cell is below 2.5%.
