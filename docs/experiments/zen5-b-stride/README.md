# B row stride on AMD Zen 5

Data behind the Zen 5 section of
[part 3](../../articles/sme2-cache-associativity.md#the-same-experiment-on-amd-zen-5):
part 3's B-stride experiment repeated on an x86 core with part 1's AVX-512
FP32 kernel. Measured 2026-10-17.

## Machine and build

AMD Ryzen 9 9950X (Zen 5), Ubuntu 24.04, Linux 7.0, GCC 13.3,
`-O3 -march=native -ffast-math -funroll-loops`, one thread pinned to core 2
(`taskset -c 2`), core clock 5.71 GHz (measured at the start of every run).
Idle machine (load average < 0.3).

Cache geometry as the kernel reports it (`/sys/devices/system/cpu/cpu2/cache`):

| Level | Size | Ways | Sets | Line | Shared by |
|---|--:|--:|--:|--:|---|
| L1 data | 48 KiB | 12 | 64 | 64 B | one core (2 SMT threads) |
| L2 | 1 MiB | 16 | 1,024 | 64 B | one core |
| L3 | 32 MiB | 16 | 32,768 | 64 B | 8 cores (one CCD) |

## Files

All from [`avx512_b_stride`](../../../benchmarks/experiments/avx512_b_stride.cpp).

| File | Mode | What |
|---|---|---|
| `chase.csv` | `chase` | Dependent pointer chase over `lines` lines placed `stride_bytes` apart, random order, on 2 MiB transparent huge pages (no TLB misses from the stride); best of 3 × 20 M loads. Strides 64 B (dense), 4, 8, 16, 64 KiB and each + 64 B |
| `ldb.csv` | `ldb` | FP32 GEMM M = N = K = 1024, 2048, 4096 with B's row stride `ldb` ≥ N; B in place (as `gemm_avx512_blocked` reads it) and packed |
| `size.csv` | `size` | The same at ldb = N for N around 1024, 2048 and 4096 |
| `height.csv` | `height` | M = 512, K = 2048, N = 1024; strip height `kc` (rows of the reused B block) from 16 to 1024, B rows 4, 8, 16 KiB apart, each + 64 B, and 4,352 B and 6 KiB apart |

GEMM columns: GFLOP/s of B in place and packed, medians of 5 interleaved
repetitions; every variant is checked against `gemm_avx512_blocked` before
timing. The GEMM uses the library kernel's blocking (64 rows of A × `kc`
rows of B × 512 columns, `kc` = 256 except in `height`) and its
micro-kernel with the 8 × 2 tile it had then (`avx512_micro<8, 2>`, 8 × 32
FP32; the library default is 8 × 3 since 2026-10-21). "Packed" copies each
`kc` × 512 block of B into contiguous 32-column strips before use; with
the library's loop order (64-row blocks of A outermost) that copy is
repeated for every block of A, so packed B still reads the strided B
N / 64 times. Packing time is included.

## Reading the chase

Latency levels: L1 4 cycles, L2 14, L3 about 45, DRAM about 200. Lines kept
at each level before the latency steps up:

| Stride | L1 | L2 | L3 |
|---|--:|--:|--:|
| 64 B (dense) | 768 | 16,384 | ≥ 32,768 |
| 4 KiB | 12 | 256 | 16,384 (not 32,768) |
| 8 KiB | 12 | 128 | 16,384 (not 32,768) |
| 16 KiB | 12 | 64 | about 16,384 |
| 64 KiB | 12 | 16 | about 4,096 (latency rises gradually beyond) |
| each + 64 B | 768 | fills L2 | — |

L1 and L2 match a modulo-indexed cache exactly: lines `s` bytes apart use
`sets / gcd(s / 64, sets)` sets, each `ways` deep (L1: 1 set × 12 at any
multiple of 4 KiB; L2: 16 × 16 = 256 at 4 KiB, 128, 64, 16). L3 keeps at
least twice what modulo indexing of its 32,768 sets would allow at a 4 KiB
stride (512 sets × 16 = 8,192): its set index is not the plain address
bits.

## Limitations

No hardware counters: `perf_event_paranoid` is 4 on this machine and no
root access was used, so cache misses are inferred from latency and
throughput thresholds, not counted. One core, one compiler, FP32 only.
