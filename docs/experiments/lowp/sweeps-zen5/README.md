# AVX-512 BF16 / INT8 tile sweep, Zen 5

`bench_gemm_lowp --benchmark_filter='^(Avx512Dpbf16|Avx512Vnni)(Tile[0-9xKc]+)?/'`,
3 randomly interleaved repetitions, `taskset -c 2`, AMD Ryzen 9 9950X,
Ubuntu 24.04, GCC 13.3, `-O3 -march=native -ffast-math -funroll-loops`,
idle (load averages in `lowp_tiles_load.txt`). Coefficient of
variation ≤ 4.6%. `Avx512Dpbf16` / `Avx512Vnni` without a suffix is the
default 8 × 2 tile with a 1 KiB k block; `Kc512` rows use 512 bytes.

The default holds: 8 × 2 is the fastest at N = 256 and 1024 (BF16) and
1024 and 2048 (INT8). Elsewhere the leader is ahead by at most 3.7%
(BF16 8 × 3 at N = 2048, 8 × 3 Kc512 at 4096; INT8 4 × 4 at 256,
6 × 4 Kc512 at 4096), within the run's variation.
