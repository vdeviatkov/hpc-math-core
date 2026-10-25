# Packed B for the FP32 SIMD kernels

Data behind `gemm_simd_packed<Ymm>`, `gemm_simd_packed<Zmm>` and
`gemm_simd_packed<Q>` ([`simd.hpp`](../../../src/gemm/simd.hpp)),
[docs/kernels/blocks.md](../../kernels/blocks.md#packed-b) and the
"Step 5: pack B" section of
[part 1](../../articles/simd-gemm.md#step-5-pack-b). The packed kernels run
the blocked kernels' micro-kernels and register tiles (AVX2 6 × 2, AVX-512
8 × 3, NEON 6 × 4) in GotoBLAS order, with each Kc × Nc block of B copied
once into contiguous strips V vectors wide. Measured 2026-10-24, one
thread.

## Machines and builds

| | AMD Ryzen 9 9950X (Zen 5) | Apple M4 Max (P-core) |
|---|---|---|
| OS | Ubuntu 24.04, Linux 7.0 | macOS 26.3.1 |
| Compiler | GCC 13.3, `-O3 -march=native` | Apple Clang 17.0.0, `-O3 -mcpu=apple-m4` |
| Pinning | `taskset -c 2` | none (macOS) |
| Load | idle (load average ≤ 0.35) | see `load_m4.txt` |

## Files

| File | Run | What |
|---|---|---|
| [`zen5_avx512_packblock.csv`](zen5_avx512_packblock.csv), [`zen5_avx2_packblock.csv`](zen5_avx2_packblock.csv) | `bench_gemm --benchmark_filter='^Avx512PackBlock'` (and `Avx2`), one run | Mc ∈ {64, 128, 256, 512} × Kc ∈ {128, 256, 512} × Nc ∈ {512, 1024, 2048}, N = 1024, 2000, 4096, FP32 and FP64 |
| [`zen5_cmp.csv`](zen5_cmp.csv) | `--benchmark_filter='Cmp/' --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true` | `<Isa>BlockedCmp` (B in place), `<Isa>PackedCmp`, `<Isa>PackedACmp` (A packed too) at the chosen defaults, N = 1000, 1024, 2000, 2048, 4000, 4096 |
| [`zen5_small.csv`](zen5_small.csv) | `'^(Avx2\|Avx512)(Blocked\|Packed)/'`, 3 repetitions interleaved | the same two kernels at N = 64 … 4096 (the main table's sizes) |
| [`m4_neon_packblock_loaded.csv`](m4_neon_packblock_loaded.csv) | `'^NeonPackBlock'`, one run | Mc ∈ {64 … 1024} × Kc ∈ {256, 512, 1024} × Nc ∈ {512, 1024, 2048}. **Run under load average 7–23** (an IDE reindexing); kept for the record, not used for any number |
| [`m4_neon_slice.csv`](m4_neon_slice.csv) | `'^NeonPackBlock(256\|512\|1024)x512x(512\|1024)/'`, 3 repetitions interleaved | the candidates re-measured on a quieter machine; chose the NEON default |
| [`m4_cmp.csv`](m4_cmp.csv) | `'^Neon(Blocked\|Packed\|PackedA)(Cmp)?/'`, 5 repetitions interleaved | the NEON counterpart of `zen5_cmp.csv` and `zen5_small.csv` in one session |
| `load_m4.txt` | `uptime` | load before and after each M4 run |

Coefficient of variation of the time: ≤ 1.2% in every Zen 5 repeated cell;
≤ 1.9% in `m4_cmp.csv` and ≤ 1.6% in `m4_neon_slice.csv`. A first M4
comparison under load average 9–18 had CVs of 40–90%, and FP64 in-place
results a third of the usual (consistent with the process being moved to
efficiency cores, which macOS does not report); it was discarded.

The CSVs report time only (the GFLOP/s counter column is missing because
the first rows of a run are SKIPPED families); GFLOP/s = 2·N³ / `cpu_time`.

## Chosen defaults

| Family | Mc × Kc × Nc | Why |
|---|---|---|
| AVX2 | 256 × 256 × 512 | within 1% of the best block of each type (geometric mean over N) |
| AVX-512 | 128 × 512 × 512 | within 0.7% of the best per type (geometric mean over N); Kc = 512 beats 256 by ~2% although a 512 × 48-float strip (96 KiB) exceeds the 48 KiB L1 |
| NEON | 256 × 512 × 512 | every candidate within 3% on the quiet machine; best or tied in 4 of 6 cells |

## Results (medians, GFLOP/s)

| | N = 1000 | 1024 | 2000 | 2048 | 4000 | 4096 |
|---|--:|--:|--:|--:|--:|--:|
| AVX-512 FP32, in place | 250.6 | 311.5 | 308.2 | 301.6 | 309.8 | 294.2 |
| AVX-512 FP32, packed | 288.7 | 315.7 | 320.9 | 318.2 | 322.5 | 318.7 |
| AVX-512 FP64, in place | 152.2 | 149.8 | 150.5 | 144.7 | 149.9 | 142.4 |
| AVX-512 FP64, packed | 155.6 | 154.5 | 155.0 | 154.3 | 154.2 | 152.2 |
| AVX2 FP32, in place | 168.4 | 126.3 | 163.9 | 105.2 | 163.0 | 104.2 |
| AVX2 FP32, packed | 167.5 | 166.1 | 168.7 | 166.5 | 169.4 | 166.9 |
| AVX2 FP64, in place | 83.9 | 54.7 | 81.0 | 53.4 | 81.5 | 53.6 |
| AVX2 FP64, packed | 83.2 | 81.8 | 82.9 | 82.0 | 82.6 | 81.3 |
| NEON FP32, in place | 120.8 | 121.5 | 118.7 | 117.6 | 114.1 | 101.6 |
| NEON FP32, packed | 121.9 | 123.2 | 118.6 | 121.1 | 120.7 | 120.4 |
| NEON FP64, in place | 59.0 | 59.1 | 54.8 | 51.6 | 54.1 | 50.5 |
| NEON FP64, packed | 59.2 | 59.8 | 59.1 | 58.8 | 58.7 | 58.5 |

Small sizes, packed / in place: AVX-512 0.87 (FP32) and 0.89 (FP64) at
N = 64, 0.96 / 0.94 at 256, 0.98 / 1.00 at 512; AVX2 0.94 / 0.95 at 64,
0.98 / 0.97 at 256, 0.99 / 1.32 at 512 (FP64 N = 512 is already a 4 KiB
stride); NEON 0.90 / 0.89 at 64, 0.98 / 0.98 at 256, 1.01 / 1.01 at 512.
The packing copy and its buffer allocation are inside the timed call.

Packing A as well (`PackedACmp`) is 0.95–1.00× of packed B alone on
Zen 5 and 0.96–0.99× on M4 at the chosen defaults: with B packed, the
A rows K elements apart are not what limits these kernels.

## Limitations

No hardware counters (as in [zen5-b-stride](../zen5-b-stride/README.md)),
so cache-set conflicts are inferred from where the gains appear (power-of-
two row strides) rather than counted. On M4 the FP64 gain is as large at
N = 2000 and 4000 as at 2048 and 4096, so there the copy helps for some
reason beyond the power-of-two stride; it was not isolated. One compiler
per machine.
