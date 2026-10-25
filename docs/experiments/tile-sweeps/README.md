# Register-tile sweeps

Raw data behind [docs/kernels](../../kernels/README.md): the AVX2,
AVX-512 and NEON blocked kernels run with every register-tile shape in
`bench_gemm`'s `add_tiles()`, and the re-measured library kernels after
their defaults changed. 2026-10-21, one thread.

| Machine | Compiler | Flags | Load |
|---|---|---|---|
| AMD Ryzen 9 9950X (Zen 5), Ubuntu 24.04 | GCC 13.3 | `-O3 -march=native -ffast-math -funroll-loops` (native), or `-march=x86-64-v3` (16 YMM registers); pinned with `taskset -c 2` | idle, load average < 1.2 |
| Apple M4 Max, P-core | Apple Clang 17 | `-O3 -mcpu=apple-m4 -ffast-math -funroll-loops` | load average 5–14 on 16 cores (Spotlight indexing and an IDE, 2–3 of 12 P-cores busy) |

Core clock, measured by `simd_microkernel` at its start and end
([`core_clock.hpp`](../../../benchmarks/experiments/core_clock.hpp)):
5.70 GHz on Zen 5 and 4.48–4.50 GHz on the M4 in every run here. (The
M4 measurements of 2026-10-15 in
[cpu-simd](../cpu-simd/README.md) ran at 3.96–4.13 GHz under a different
background load; compare M4 numbers across the two dates in cycles, not
GFLOP/s.)

Google Benchmark CSVs hold the mean, median, standard deviation and
coefficient of variation (`_cv`) of 3 repetitions; GFLOP/s is the last
column. The docs quote medians.

| File | Program | What |
|---|---|---|
| [`tiles_zen5_native.csv`](tiles_zen5_native.csv) | `bench_gemm --benchmark_filter='^(Avx2Tile\|Avx512Tile)'` | Zen 5, every AVX2 and AVX-512 shape, native build (32 vector registers) |
| [`tiles_zen5_x86-64-v3.csv`](tiles_zen5_x86-64-v3.csv) | `bench_gemm --benchmark_filter='^Avx2Tile'` | Zen 5, every AVX2 shape, 16-register build |
| [`microkernel_zen5.csv`](microkernel_zen5.csv), [`microkernel_zen5_clock.txt`](microkernel_zen5_clock.txt) | [`simd_microkernel`](../../../benchmarks/experiments/simd_microkernel.cpp) | the library micro-kernels on L1-resident data, Zen 5 |
| [`bench_zen5_avx512_8x3.csv`](bench_zen5_avx512_8x3.csv) | `bench_gemm --benchmark_filter='^Avx512Blocked'` | `gemm_simd_blocked<Zmm>` at its new 8 × 3 default and its prefetch variants |
| [`lowp_zen5_avx512_8x3.csv`](lowp_zen5_avx512_8x3.csv) | `bench_gemm_lowp --benchmark_filter='^Avx512Blocked'` | the same kernel at the sizes of the low-precision tables |
| [`load_zen5_avx512_8x3.txt`](load_zen5_avx512_8x3.txt) | `uptime` | load before and after the two runs above |
| [`tiles_m4.csv`](tiles_m4.csv) | `bench_gemm --benchmark_filter='^NeonTile'` | M4, every NEON shape, A loaded with `LDR` (the current kernel) |
| [`microkernel_m4.csv`](microkernel_m4.csv), [`microkernel_m4_clock.txt`](microkernel_m4_clock.txt), [`load_m4.txt`](load_m4.txt) | `simd_microkernel`, `uptime` | the same run, continued |
| [`tiles_m4_ld1r.csv`](tiles_m4_ld1r.csv), [`microkernel_m4_ld1r.csv`](microkernel_m4_ld1r.csv), [`microkernel_m4_ld1r_clock.txt`](microkernel_m4_ld1r_clock.txt), [`load_m4_ld1r.txt`](load_m4_ld1r.txt) | the same programs | the same sweep an hour earlier, with A broadcast by `LD1R` (the kernel before the change in [docs/kernels/neon.md](../../kernels/neon.md#loading-a-ldr-not-ld1r); no 5 × 4 shape) |
| [`m4_ld1r_ab/`](m4_ld1r_ab/) | `bench_gemm --benchmark_filter='^NeonTile(4x4\|6x3\|6x4)/(f32\|f64)/N=(256\|1024)$'` | the `LD1R` and `LDR` builds alternated three times (`ld1r_<run>.csv`, `ldr_<run>.csv`), one repetition each |
| [`bench_m4_neon_6x4.csv`](bench_m4_neon_6x4.csv), [`load_m4_neon_6x4.txt`](load_m4_neon_6x4.txt) | `bench_gemm --benchmark_filter='^NeonBlocked'` | `gemm_simd_blocked<Q>` at its new 6 × 4 default with `LDR`, and its prefetch variants |
| [`lowp_m4_neon_6x4.csv`](lowp_m4_neon_6x4.csv), [`load_lowp_m4_neon_6x4.txt`](load_lowp_m4_neon_6x4.txt) | `bench_gemm_lowp --benchmark_filter='^Neon'` | the FP32 kernel above with all NEON low-precision kernels, one session (replaces the M4 NEON rows of [lowp](../lowp/README.md) in the docs) |
| [`m4_driver_ab/`](m4_driver_ab/) | `bench_gemm --benchmark_filter='^(NeonBlocked\|NeonTile4x4\|NeonTile8x2)/(f32\|f64)/N=(256\|1024)$'` | the hand-written NEON micro-kernels of commit 3bd58c0 (`handwritten_<run>.csv`) against the template micro-kernel with the same shapes (`driver_<run>.csv`, `NeonBlocked` rows), alternated three times |

Notes:

- At Zen 5 AVX-512 FP64 N = 512 (B rows 4 KiB apart) the result depends
  on where the matrices land in memory: `Avx512Tile8x3` measured 150.9
  GFLOP/s in the sweep, while the same kernel run alone measures 126–128
  (also in `bench_zen5_avx512_8x3.csv`, 128.4). 8 × 2 measures 101–103
  either way.
- One cell has a high coefficient of variation: `Avx512Tile12x2/f64/N=1024`,
  11.5%. Every other cell is below 2.5%.
