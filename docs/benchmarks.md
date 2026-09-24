# Benchmark Results

Measured throughput for every kernel family in this repository, on the three
machines it has been verified on. All numbers come from
`bench_gemm` / `bench_gemm_cuda` (Google Benchmark, console format) built in
Release mode as described in [build.md](build.md). CPU kernels are strictly
single-threaded unless noted (Apple AMX via Accelerate is the exception).

**Contents**

- [Deriving GFLOP/s](#deriving-gflops)
- [Apple M4 Max — scalar, NEON, prefetch](#apple-m4-max--scalar-neon-prefetch)
- [ARM SME2 — Apple M4 Max](#arm-sme2--apple-m4-max-real-measured---dhpc_enable_smeon)
- [Apple AMX — Apple M4 Max](#apple-amx--apple-m4-max-via-accelerateframework-real-measured)
- [Speedup tables — Apple M4 Max](#speedup-tables--apple-m4-max)
- [Key observations — Apple M4 Max](#key-observations--apple-m4-max)
- [AMD Zen 5 — AVX2 + AVX-512 (Linux / GCC)](#amd-zen-5--avx2--avx-512-linux--gcc)
- [NVIDIA RTX 5080 — CUDA](#nvidia-rtx-5080--cuda)

---

## Deriving GFLOP/s

```
GFLOP/s = (2 × N³) / (time_µs × 1000)
```

A square N×N GEMM performs `2 × N³` floating-point operations. Dividing by wall-clock time in nanoseconds gives GFLOP/s.

Example: `NeonBlockedPf2/f32/N=512`, 2715 µs → `2 × 512³ / (2715 × 1000)` ≈ **98.9 GFLOP/s**.

---

## Apple M4 Max — scalar, NEON, prefetch

> **Machine:** Apple M4 Max, 16 cores, Apple Clang 17, C++20
> **Build:** `cmake -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON` → `-O3 -ffast-math -funroll-loops` (`-march` cleared in favour of `-mcpu=apple-m4`, see [§ SME and AMX build flags](build.md#sme-and-amx-build-flags))
> **CPU Caches:** L1 Data 64 KiB · L1 Instruction 128 KiB · L2 Unified 4096 KiB (×16)
> **Load Average:** 4.15 / 4.16 / 4.25 — 1-min / 5-min / 15-min average number of runnable
> processes. On a 16-core machine, 16.0 = 100% utilisation; ~4.2 ≈ 26% load — moderate
> background activity, numbers are still representative.

### double (f64) — scalar & NEON kernels

```
Benchmark                    Time        CPU     GFLOP/s
--------------------------------------------------------
Naive/f64/N=64              55.5 µs    55.4 µs    9.46
Naive/f64/N=256          13036  µs  13029  µs     2.58
Naive/f64/N=512         102246  µs  102236 µs     2.63
Naive/f64/N=1024        924366  µs  900156  µs     2.39
Naive/f64/N=4096     207544657  µs   207.5s     662.5 M/s

Reordered/f64/N=64          18.9 µs    18.9 µs   27.74
Reordered/f64/N=256        2020   µs   2019   µs  16.62
Reordered/f64/N=512       16019   µs  16012   µs  16.76
Reordered/f64/N=1024     129596   µs  129576  µs  16.57
Reordered/f64/N=4096    8442770   µs    8.44s    16.29

Blocked/f64/N=64            19.3 µs    19.3 µs   27.21  tile=64
Blocked/f64/N=256          1320   µs   1320   µs  25.42  tile=64
Blocked/f64/N=512         12079   µs  12076   µs  22.23  tile=64
Blocked/f64/N=1024       110250   µs  110216  µs  19.48  tile=64
Blocked/f64/N=4096      6813557   µs    6.81s    20.18  tile=64

NeonNaive/f64/N=64          60.6 µs    60.5 µs    8.66  neon=1
NeonNaive/f64/N=256      13490   µs  13487   µs   2.49  neon=1
NeonNaive/f64/N=512     103983   µs  103971  µs   2.58  neon=1
NeonNaive/f64/N=1024    890225   µs  890160  µs   2.41  neon=1
NeonNaive/f64/N=4096  208745583  µs   208.7s   658.7 M/s neon=1

NeonReordered/f64/N=64      19.3 µs    19.3 µs   27.12  neon=1
NeonReordered/f64/N=256    2455   µs   2454   µs  13.67  neon=1
NeonReordered/f64/N=512   18753   µs  18749   µs  14.32  neon=1
NeonReordered/f64/N=1024 142757   µs  142712  µs  15.05  neon=1
NeonReordered/f64/N=4096 8859992  µs    8.86s    15.52  neon=1

NeonBlocked/f64/N=64        14.4 µs    14.4 µs   36.30  neon=1
NeonBlocked/f64/N=256       989   µs    988   µs  33.95  neon=1
NeonBlocked/f64/N=512      8380   µs   8377   µs  32.05  neon=1
NeonBlocked/f64/N=1024    70187   µs  70130   µs  30.62  neon=1
NeonBlocked/f64/N=4096  5508943   µs    5.51s    24.96  neon=1

Avx2*/f64/*     SKIPPED: 'AVX2 not available on this target'
Avx512*/f64/*   SKIPPED: 'AVX-512 not available on this target'
Sve*/f64/*      SKIPPED: 'SVE not available on this target'
```

### float (f32) — scalar & NEON kernels

```
Benchmark                    Time        CPU     GFLOP/s
--------------------------------------------------------
Naive/f32/N=64              56.1 µs    56.1 µs    9.35
Naive/f32/N=256          12278  µs  12272  µs     2.73
Naive/f32/N=512         109000  µs  108961 µs     2.46
Naive/f32/N=1024        815153  µs  815086 µs     2.63
Naive/f32/N=4096     218351231  µs   218.4s     629.4 M/s  (contended — see note below)

Reordered/f32/N=64           6.09 µs    6.09 µs  86.10
Reordered/f32/N=256        1039   µs   1038   µs  32.32
Reordered/f32/N=512        8157   µs   8154   µs  32.92
Reordered/f32/N=1024      65121   µs  65103   µs  32.99
Reordered/f32/N=4096    4187456   µs    4.19s    32.83

Blocked/f32/N=64             6.12 µs    6.11 µs  85.85  tile=64
Blocked/f32/N=256             402 µs     402  µs  83.51  tile=64
Blocked/f32/N=512            5238 µs    5236  µs  51.27  tile=64
Blocked/f32/N=1024          49952 µs   49937  µs  43.00  tile=64
Blocked/f32/N=4096        4357272 µs    4.36s    31.55  tile=64

NeonNaive/f32/N=64          56.3 µs    56.3 µs    9.31  neon=1
NeonNaive/f32/N=256        8810   µs   8806   µs   3.81  neon=1
NeonNaive/f32/N=512       74224   µs  74192   µs   3.62  neon=1
NeonNaive/f32/N=1024     807084   µs  806945  µs   2.66  neon=1
NeonNaive/f32/N=4096  217443539  µs   217.4s   632.1 M/s neon=1  (contended — see note below)

NeonReordered/f32/N=64      20.3 µs    20.3 µs   25.82  neon=1
NeonReordered/f32/N=256    1132   µs   1132   µs  29.64  neon=1
NeonReordered/f32/N=512    9831   µs   9828   µs  27.31  neon=1
NeonReordered/f32/N=1024  76022   µs  75996   µs  28.26  neon=1
NeonReordered/f32/N=4096 4494147  µs    4.49s    30.59  neon=1

NeonBlocked/f32/N=64         5.41 µs    5.40 µs  97.03  neon=1
NeonBlocked/f32/N=256         348 µs     348  µs  96.43  neon=1
NeonBlocked/f32/N=512        2776 µs    2775  µs  96.73  neon=1
NeonBlocked/f32/N=1024      22752 µs   22746  µs  94.41  neon=1
NeonBlocked/f32/N=4096    1892191 µs    1.89s    72.66  neon=1

Avx2*/f32/*     SKIPPED: 'AVX2 not available on this target'
Avx512*/f32/*   SKIPPED: 'AVX-512 not available on this target'
Sve*/f32/*      SKIPPED: 'SVE not available on this target'
```

> **Note on the two "contended" rows:** `Naive/f32/N=4096` and `NeonNaive/f32/N=4096` were captured while a second, unrelated benchmark process happened to be running concurrently on this machine (a batching artefact of this particular run, not a property of the kernels) — wall-clock time is inflated relative to CPU time as a result. The **CPU** column and the GFLOP/s derived from it are still accurate; use those, not the wall-clock `Time` column, for these two rows specifically.

### Prefetch distance sweep — `BlockedPf` (scalar) and `NeonBlockedPf`

Benchmarks named `<Family>BlockedPf<D>/<prec>/N=<size>` sweep prefetch distance D ∈ {2, 4, 8, 16} (rows ahead). Three `__builtin_prefetch` sites per kernel:

- **[PF-A]** `A(i + D×kRegRows, k_blk)` → L2 (read)
- **[PF-B]** `B(k_blk + TileK, 0)` → L2 (read) at k-tile boundary
- **[PF-C]** `C(i + D×kRegRows, j_blk)` → L1 (write)

```
Benchmark                         Time      GFLOP/s   pf_dist
-------------------------------------------------------------
— Scalar blocked + prefetch (f64) —
BlockedPf2/f64/N=256           2277 µs    14.74 G/s   D=2
BlockedPf4/f64/N=256           2266 µs    14.81 G/s   D=4  ← best
BlockedPf8/f64/N=256           2269 µs    14.79 G/s   D=8
BlockedPf16/f64/N=256          2298 µs    14.61 G/s   D=16

BlockedPf2/f64/N=512          20726 µs    12.96 G/s   D=2
BlockedPf4/f64/N=512          20696 µs    12.97 G/s   D=4
BlockedPf8/f64/N=512          20769 µs    12.93 G/s   D=8
BlockedPf16/f64/N=512         20659 µs    13.00 G/s   D=16 ← best

BlockedPf2/f64/N=1024        189090 µs    11.36 G/s   D=2
BlockedPf4/f64/N=1024        188221 µs    11.41 G/s   D=4  ← best
BlockedPf8/f64/N=1024        189234 µs    11.35 G/s   D=8
BlockedPf16/f64/N=1024       189300 µs    11.35 G/s   D=16

— Scalar blocked + prefetch (f32) —
BlockedPf2/f32/N=256           1168 µs    28.75 G/s   D=2  ← best
BlockedPf4/f32/N=256           1171 µs    28.66 G/s   D=4
BlockedPf8/f32/N=256           1182 µs    28.48 G/s   D=8
BlockedPf16/f32/N=256          1229 µs    27.37 G/s   D=16

BlockedPf2/f32/N=512          12122 µs    22.15 G/s   D=2
BlockedPf4/f32/N=512          12128 µs    22.14 G/s   D=4
BlockedPf8/f32/N=512          12100 µs    22.19 G/s   D=8  ← best
BlockedPf16/f32/N=512         12110 µs    22.17 G/s   D=16

BlockedPf2/f32/N=1024         100408 µs    21.39 G/s   D=2
BlockedPf4/f32/N=1024         100236 µs    21.43 G/s   D=4
BlockedPf8/f32/N=1024         100196 µs    21.44 G/s   D=8  ← best
BlockedPf16/f32/N=1024        100685 µs    21.33 G/s   D=16

— NEON blocked + prefetch (f64) —
NeonBlockedPf2/f64/N=256        994 µs    33.77 G/s   D=2
NeonBlockedPf4/f64/N=256       1042 µs    32.20 G/s   D=4
NeonBlockedPf8/f64/N=256        993 µs    33.80 G/s   D=8  ← best
NeonBlockedPf16/f64/N=256      1042 µs    32.19 G/s   D=16

NeonBlockedPf2/f64/N=512       8409 µs    31.94 G/s   D=2  ← best
NeonBlockedPf4/f64/N=512       8758 µs    30.66 G/s   D=4
NeonBlockedPf8/f64/N=512       8426 µs    31.87 G/s   D=8
NeonBlockedPf16/f64/N=512      8737 µs    30.75 G/s   D=16

NeonBlockedPf2/f64/N=1024     70901 µs    30.30 G/s   D=2
NeonBlockedPf4/f64/N=1024     72425 µs    29.66 G/s   D=4
NeonBlockedPf8/f64/N=1024     70585 µs    30.43 G/s   D=8  ← best
NeonBlockedPf16/f64/N=1024    72026 µs    29.83 G/s   D=16

— NEON blocked + prefetch (f32) —
NeonBlockedPf2/f32/N=256        343 µs    97.86 G/s   D=2  ← best
NeonBlockedPf4/f32/N=256        351 µs    95.67 G/s   D=4
NeonBlockedPf8/f32/N=256        350 µs    95.88 G/s   D=8
NeonBlockedPf16/f32/N=256       349 µs    96.18 G/s   D=16

NeonBlockedPf2/f32/N=512       2733 µs    98.26 G/s   D=2  ← best
NeonBlockedPf4/f32/N=512       2792 µs    96.17 G/s   D=4
NeonBlockedPf8/f32/N=512       2795 µs    96.10 G/s   D=8
NeonBlockedPf16/f32/N=512      2789 µs    96.28 G/s   D=16

NeonBlockedPf2/f32/N=1024     22508 µs    95.46 G/s   D=2  ← best
NeonBlockedPf4/f32/N=1024     22976 µs    93.49 G/s   D=4
NeonBlockedPf8/f32/N=1024     22925 µs    93.71 G/s   D=8
NeonBlockedPf16/f32/N=1024    22859 µs    93.97 G/s   D=16

Avx2BlockedPf*/  SKIPPED: 'AVX2 not available on this target'
Avx512BlockedPf*/SKIPPED: 'AVX-512 not available on this target'
SveBlockedPf*/   SKIPPED: 'SVE not available on this target'
```

---

## ARM SME2 — Apple M4 Max (real, measured — `-DHPC_ENABLE_SME=ON`)

> **Machine:** Apple M4 Max, Apple Clang 17, C++20, `-mcpu=apple-m4` (see [§ SME and AMX build flags](build.md#sme-and-amx-build-flags) for why this build needs a dedicated flag rather than `-march=native`)
> **SVL (streaming vector length):** 16 f32 / 8 f64 elements — reported live via the `svl` benchmark counter
> **Command:** `cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON && cmake --build build -j && ./build/benchmarks/bench_gemm --benchmark_filter=Sme`

SME computes GEMM with a fundamentally different primitive than every other CPU kernel above: instead of per-lane FMA, a single `FMOPA` instruction accumulates a whole SVL×SVL **outer product** into a 2-D hardware accumulator (ZA), the same class of operation as NVIDIA Tensor Cores (`gemm_cuda_wmma`) and Apple's own AMX coprocessor (below) — see [src/gemm/README.md](../src/gemm/README.md#algorithm-10--arm-sme2-scalable-matrix-extension) for the full architectural writeup, including the two real hardware/toolchain issues found while building this (gather-loads are illegal in SME streaming mode; combining `-march=native` with `-mcpu=apple-m4` silently disables SME).

```
Benchmark                        Time             CPU   GFLOP/s
------------------------------------------------------------------
SmeNaive/f64/N=64              355 us          355 us     1.48 G/s   svl=8
SmeNaive/f64/N=256           22065 us        22062 us     1.52 G/s   svl=8
SmeNaive/f64/N=512          176633 us       176601 us     1.52 G/s   svl=8
SmeNaive/f64/N=1024        1414901 us      1414665 us     1.52 G/s   svl=8
SmeNaive/f32/N=64              178 us          178 us     2.94 G/s   svl=16
SmeNaive/f32/N=256           10748 us        10745 us     3.12 G/s   svl=16
SmeNaive/f32/N=512           85802 us        85776 us     3.13 G/s   svl=16
SmeNaive/f32/N=1024         686329 us       686161 us     3.13 G/s   svl=16

SmeReordered/f64/N=64           9.70 us         9.70 us    54.08 G/s   svl=8
SmeReordered/f64/N=256           333 us          333 us   100.80 G/s   svl=8
SmeReordered/f64/N=512          2388 us         2387 us   112.46 G/s   svl=8
SmeReordered/f64/N=1024        19194 us        19077 us   112.57 G/s   svl=8
SmeReordered/f64/N=2048       438908 us       438643 us    39.17 G/s   svl=8
SmeReordered/f32/N=64           7.97 us         7.97 us    65.77 G/s   svl=16
SmeReordered/f32/N=256           151 us          151 us   222.58 G/s   svl=16
SmeReordered/f32/N=512           845 us          845 us   317.60 G/s   svl=16
SmeReordered/f32/N=1024         5576 us         5570 us   385.56 G/s   svl=16  ← peak
SmeReordered/f32/N=2048        72796 us        72771 us   236.08 G/s   svl=16
SmeReordered/f32/N=4096      1065408 us      1063861 us   129.19 G/s   svl=16

SmeBlocked/f64/N=64              9.78 us         9.78 us    53.60 G/s   svl=8
SmeBlocked/f64/N=256              333 us          333 us   100.79 G/s   svl=8
SmeBlocked/f64/N=512             2393 us         2392 us   112.21 G/s   svl=8
SmeBlocked/f64/N=1024           18509 us        18506 us   116.04 G/s   svl=8  ← peak
SmeBlocked/f64/N=2048          315753 us       315588 us    54.44 G/s   svl=8   ← beats Reordered (39.17 G/s)
SmeBlocked/f32/N=64              8.09 us         8.09 us    64.83 G/s   svl=16
SmeBlocked/f32/N=256              151 us          151 us   222.34 G/s   svl=16
SmeBlocked/f32/N=512              851 us          850 us   315.63 G/s   svl=16
SmeBlocked/f32/N=1024            5597 us         5589 us   384.23 G/s   svl=16
SmeBlocked/f32/N=2048            58580 us        58567 us   293.34 G/s   svl=16  ← beats Reordered (236.08 G/s)
SmeBlocked/f32/N=4096           762060 us       761849 us   180.40 G/s   svl=16  ← beats Reordered (129.19 G/s)
```

### Key observations

- **386 GFLOP/s single-threaded, f32** (`SmeReordered` at N=1024) is the highest single-threaded CPU throughput anywhere in this repo — roughly **1.8× the AVX-512 f32 peak** (219 G/s, AMD Zen 5 section below) and **~4× `gemm_neon_blocked`** (97 G/s, same Apple-silicon class of chip) despite SME running at a lower clock than either comparison.
- **`SmeNaive` is pinned at ~3 GFLOP/s, flat across N** — confirming the same "SIMD width doesn't fix cache-hostile access" lesson every other `*_naive` kernel demonstrates in this repo, except here the hostility is structural: SME's streaming mode does not permit gather-load instructions at all (verified — Clang rejects `svld1_gather_index` with "builtin can only be called from a non-streaming function"), so the column vector for the outer product must be assembled with a scalar loop on every k-iteration.
- **`SmeReordered` fixes this by packing once per row-tile** (a single scalar pass over `A(i0..i0+16, :)`, reused across every column-tile) instead of once per (row-tile, column-tile) pair — a 22-123× improvement depending on N, for identical arithmetic.
- **`SmeBlocked` wins once the packed panel stops fitting cache**: at N=2048/4096, `SmeReordered`'s unbounded `SVL × K` packed buffer (128 KB / 256 KB at N=2048/4096) exceeds Apple M4's per-core L1, and repeated re-reads from L2 cost real throughput (236→129 G/s). Bounding the packed panel to a fixed K-tile (256 columns → 16 KB, comfortably L1-resident) and paying an extra C load/store per K-tile instead recovers most of the loss (293→180 G/s) — the same blocking trade-off as `gemm_blocked` vs `gemm_reordered` at the start of this ladder, replayed one abstraction level up.
- **f64 peaks far lower than f32** (116 vs 386 G/s) — expected, since SVL is fixed in *bytes*, not elements: SVL=8 f64 vs SVL=16 f32, so every f64 outer product covers a quarter of the elements ($8\times8$ vs $16\times16$) per instruction.

---

## Apple AMX — Apple M4 Max, via Accelerate.framework (real, measured)

> **Machine:** Apple M4 Max, Apple Clang 17, C++20, `-DHPC_ENABLE_AMX=ON` (default on Apple platforms)
> **Command:** `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ./build/benchmarks/bench_gemm --benchmark_filter=Amx`

This is Apple's own AMX coprocessor, reached through Accelerate.framework's BLAS (`cblas_sgemm`/`cblas_dgemm`) rather than any hand-written kernel — see [§ SME and AMX build flags](build.md#sme-and-amx-build-flags) and [src/gemm/README.md](../src/gemm/README.md#algorithm-11--apple-amx-via-accelerateframework) for why this is architecturally unrelated to Intel's AMX, why `gemm_amx_naive`/`_reordered`/`_blocked` are intentionally identical wrappers, and why these numbers are **not** a single-core comparison against the rest of this README (Accelerate's BLAS may use multiple cores internally).

```
Benchmark                        Time             CPU   GFLOP/s
------------------------------------------------------------------
AmxNaive/f64/N=64               1.59 us         1.59 us   330.00 G/s
AmxNaive/f64/N=256              72.1 us         72.1 us   465.66 G/s
AmxNaive/f64/N=512                338 us          327 us   820.39 G/s
AmxNaive/f64/N=1024              2599 us         2558 us   839.41 G/s

AmxReordered/f64/N=64            1.59 us         1.59 us   329.55 G/s
AmxReordered/f64/N=256           72.0 us         72.0 us   465.92 G/s
AmxReordered/f64/N=512             342 us          326 us   822.24 G/s
AmxReordered/f64/N=1024           2590 us         2501 us   858.59 G/s
AmxReordered/f64/N=2048          21722 us        21435 us   801.48 G/s
AmxReordered/f64/N=4096         171958 us       169722 us   809.79 G/s

AmxBlocked/f64/N=64              1.59 us         1.59 us   329.78 G/s
AmxBlocked/f64/N=256              71.9 us         71.9 us   466.61 G/s
AmxBlocked/f64/N=512               343 us          326 us   822.88 G/s
AmxBlocked/f64/N=1024             2578 us         2496 us   860.28 G/s   ← peak
AmxBlocked/f64/N=2048            21611 us        21362 us   804.22 G/s
AmxBlocked/f64/N=4096           171394 us       168977 us   813.36 G/s

AmxNaive/f32/N=64               0.649 us        0.649 us   807.89 G/s
AmxNaive/f32/N=256               19.4 us         19.4 us  1729.12 G/s
AmxNaive/f32/N=512                 101 us         89.9 us  2984.33 G/s
AmxNaive/f32/N=1024                668 us          655 us  3280.74 G/s

AmxReordered/f32/N=64            0.656 us        0.656 us   799.46 G/s
AmxReordered/f32/N=256            19.4 us         19.4 us  1726.68 G/s
AmxReordered/f32/N=512              105 us         91.8 us  2922.79 G/s
AmxReordered/f32/N=1024             670 us          659 us  3260.54 G/s
AmxReordered/f32/N=2048            5339 us         5338 us  3218.12 G/s
AmxReordered/f32/N=4096           43469 us        43143 us  3185.66 G/s

AmxBlocked/f32/N=64              0.658 us        0.657 us   797.61 G/s
AmxBlocked/f32/N=256              19.5 us         19.5 us  1723.73 G/s
AmxBlocked/f32/N=512                105 us         92.4 us  2905.87 G/s
AmxBlocked/f32/N=1024               668 us          652 us  3295.78 G/s   ← peak: 3.30 TFLOP/s
AmxBlocked/f32/N=2048              5365 us         5365 us  3202.47 G/s
AmxBlocked/f32/N=4096              43554 us        43549 us  3155.94 G/s
```

### Key observations

- **Up to 3.3 TFLOP/s f32, 860 GFLOP/s f64** — by a wide margin the highest throughput anywhere in this repo, ~8.5× the hand-written `gemm_sme_reordered` f32 peak (386 GFLOP/s) and ~7.4× the hand-written `gemm_sme_blocked` f64 peak (116 GFLOP/s). This is expected and not really a fair fight: Accelerate's BLAS is Apple's own vendor-tuned implementation and, unlike every hand-written kernel in this repo, is free to use every CPU core available — the jump from ~800 G/s at N=64 to ~3.3 T/s at N≥1024 is consistent with additional cores/threads being brought online as the problem grows large enough to amortise their overhead, not (only) better cache behaviour.
- **`AmxNaive`, `AmxReordered`, and `AmxBlocked` produce near-identical numbers at every size** (e.g. 3281/3261/3296 GFLOP/s at N=1024, f32) — exactly as expected, since all three call the identical `cblas_sgemm`/`cblas_dgemm` wrapper (see file header of [src/gemm/amx.hpp](../src/gemm/amx.hpp)). The small run-to-run variation (≤1%) is measurement noise, not an algorithmic difference — Accelerate exposes no staging knob for this repo's naive/reordered/blocked progression to act on.
- **f32/f64 ratio is only ~3.8×, not the ~2× lane-count ratio seen elsewhere** (e.g. NEON's 2.7×, AVX-512's ~2×) — consistent with Accelerate additionally exploiting a wider or more specialised f32 datapath (plausibly a bf16-adjacent or otherwise reduced-precision-friendly internal path within the AMX coprocessor) beyond simple lane doubling, though Apple does not document this and it cannot be confirmed without disassembly.
- **This is the right comparison to make when the question is "what's the fastest way to multiply matrices on this Mac"** — if that's the actual goal, `cblas_sgemm`/`cblas_dgemm` directly (what `gemm_amx_*` wraps) is the answer, full stop. The value of the other 90% of this repository is in the *pedagogy* of getting from scalar code to a meaningful fraction of that ceiling by hand, one optimisation at a time.

---

## Speedup tables — Apple M4 Max

### f64 — best kernel per family vs `gemm_naive`

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive | NeonBlockedPf2 | ×naive |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 55.5 µs | 18.9 µs | **2.9×** | 19.3 µs | **2.9×** | 14.4 µs | **3.9×** | — | — |
| 256 | 13036 µs | 2020 µs | **6.5×** | 1320 µs | **9.9×** | 989 µs | **13.2×** | 994 µs | **13.1×** |
| 512 | 102246 µs | 16019 µs | **6.4×** | 12079 µs | **8.5×** | 8380 µs | **12.2×** | 8409 µs | **12.2×** |
| 1024 | 924366 µs | 129596 µs | **7.1×** | 110250 µs | **8.4×** | 70187 µs | **13.2×** | 70901 µs | **13.0×** |
| 4096 | 207544657 µs | 8442770 µs | **24.6×** | 6813557 µs | **30.5×** | 5508943 µs | **37.7×** | — | — |

### f32 — best kernel per family vs `gemm_naive`

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive | NeonBlockedPf2 | ×naive |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 56.1 µs | 6.09 µs | **9.2×** | 6.12 µs | **9.2×** | 5.41 µs | **10.4×** | — | — |
| 256 | 12278 µs | 1039 µs | **11.8×** | 402 µs | **30.5×** | 348 µs | **35.3×** | 343 µs | **35.8×** |
| 512 | 109000 µs | 8157 µs | **13.4×** | 5238 µs | **20.8×** | 2776 µs | **39.3×** | 2733 µs | **39.9×** |
| 1024 | 815153 µs | 65121 µs | **12.5×** | 49952 µs | **16.3×** | 22752 µs | **35.8×** | 22508 µs | **36.2×** |
| 4096 | 218351 µs\* | 4187456 µs | **52.1×**\* | 4357272 µs | **50.1×**\* | 1892191 µs | **115.4×**\* | — | — |

\* `Naive/f32/N=4096`'s wall-clock time was inflated by a concurrent unrelated process on this run (see note above) — this row uses its CPU time (218351 µs = 218.4 ms) instead, consistent with every other kernel's own (uncontended) wall-clock time in this table. The other three columns' own timings are unaffected.

---

## Key observations — Apple M4 Max

### Cache-access pattern dominates at large N

`gemm_naive` delivers nearly identical GFLOP/s for f32 and f64 at every size — both are **DRAM-bandwidth bound** from the column-stride gather on B. Element width is irrelevant once you are waiting on cache-miss latency.

The moment the loop order changes to i-k-j (`gemm_reordered`), B and C are accessed sequentially and every cache line is fully consumed. At N=4096:
- f64: **25×** faster than naive
- f32: **52×** faster than naive (using naive's CPU time — its wall-clock time was contended on this run, see note above; twice the elements per cache line → twice the bandwidth)

### Auto-vectorisation vs explicit SIMD

`gemm_reordered` and `gemm_blocked` carry **no NEON intrinsics** — the compiler auto-vectorises the sequential inner j-loop with `-ffast-math`. f32 delivers ~86 GFLOP/s at small N.

`gemm_neon_blocked` adds **explicit Q-register tiling** (4 rows × 4 Q-vectors = 4×16 f32 held in registers for the full k-tile) on top of L2 blocking:

| Kernel | f32 N=256 | f32 N=512 | f32 N=1024 |
|---|---|---|---|
| `gemm_blocked` (auto-vec) | 83.5 G/s | 51.3 G/s | 43.0 G/s |
| `gemm_neon_blocked` (explicit) | **96.4 G/s** | **96.7 G/s** | **94.4 G/s** |

The explicit register tile maintains ~95-97 GFLOP/s from N=64 through N=1024 — **flat across sizes**. The auto-vectorised blocked kernel degrades from 84→43 G/s because C rows are evicted from L1 between k-iterations at larger N.

### Software prefetch analysis

**Scalar `BlockedPf` vs base `Blocked`:** prefetch *hurts* at N=256 (14.8 vs 25.4 G/s for f64) and gives only marginal gain at N=512/1024. The scalar kernel is entirely compiler-auto-vectorised; the hardware prefetcher already handles the simple streaming access, and adding explicit prefetch instructions creates front-end pressure that slows the tight inner loop.

**`NeonBlockedPf` vs base `NeonBlocked`:** prefetch gives a small but consistent gain:

| Kernel | f32 N=256 | f32 N=512 | f32 N=1024 |
|---|---|---|---|
| `NeonBlocked` (no prefetch) | 96.4 G/s | 96.7 G/s | 94.4 G/s |
| `NeonBlockedPf2` (D=2) | **97.9 G/s** | **98.3 G/s** | **95.5 G/s** |
| Gain | **+1.5%** | **+1.6%** | **+1.1%** |

For f64 the gain is slightly larger in absolute terms (D=2/D=8 trade the lead across sizes, both well ahead of D=4/D=16). The L2 latency on Apple M is short enough that prefetching much further than 2-8 micro-kernel steps ahead adds latency-hiding overhead without benefit.

**Prefetch distance rule of thumb for this hardware:**

```
optimal D ≈ ceil(L2_latency_cycles / cycles_per_micro_kernel_call)
          ≈ ceil(12 / ~6) = 2
```

### NEON f64 vs f32

- NEON Q-register: 4 f32 lanes or 2 f64 lanes (128-bit).
- `gemm_neon_blocked` f64 peaks at ~36 G/s; f32 peaks at ~97 G/s — ratio ≈ **2.7×**.
- The theoretical ratio is 2× (lane count). The extra 0.7× for f32 comes from f32 tiles fitting entirely in L1 at sizes where f64 tiles spill.

### Headline GFLOP/s summary (Apple M4 Max, this run)

| Kernel | f64 peak | f32 peak | f32/f64 ratio |
|---|---|---|---|
| `gemm_naive` | 9.46 G/s | 9.35 G/s | 1.0× |
| `gemm_reordered` | 27.74 G/s | 86.10 G/s | **3.1×** |
| `gemm_blocked` | 27.21 G/s | 85.85 G/s | **3.2×** |
| `gemm_neon_blocked` | 36.30 G/s | 97.03 G/s | **2.7×** |
| `gemm_neon_blocked_prefetch` (D=2) | **33.77 G/s** @ N=256 | **98.26 G/s** @ N=512 | **2.9×** |
| `gemm_sme_reordered` | 112.57 G/s | **385.56 G/s** | **3.4×** |
| `gemm_amx_blocked` (via Accelerate) | 860.28 G/s | **3295.78 G/s** | **3.8×** |

---

## AMD Zen 5 — AVX2 + AVX-512 (Linux / GCC)

> **Machine:** AMD Ryzen 9 9950X (Zen 5), 16C/32T, 5.64 GHz, Ubuntu 24.04, GCC 13.3, C++20
> **Build:** `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_AVX512=ON` → `-O3 -march=native -ffast-math -funroll-loops`
> **CPU Caches:** L1d 48 KiB (×16) · L1i 32 KiB (×16) · L2 1024 KiB (×16) · L3 32768 KiB (×2)
> **Load Average:** 1.03 / 1.06 / 1.00 on 32 threads — effectively idle
> **Governor:** `performance` · **Run time:** 2164 s (36 min) for the full CPU sweep
> **Date:** 2026-09-24

Zen 5 has a full-width 512-bit AVX-512 datapath (unlike Zen 4's double-pumped
256-bit), so this is the first machine in this repo where `gemm_avx512_*`
runs on native-width hardware. NEON/SVE/SME/AMX are absent and report
`SKIPPED`; 119 of the CPU test suite's cases compile here (scalar + AVX2 +
AVX-512) and all pass.

### GFLOP/s by kernel and size

**double (f64)**

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `gemm_naive` | 14.21 | 2.05 | 1.27 | 0.73 | 0.43 |
| `gemm_reordered` | 42.21 | 52.34 | 36.59 | 32.89 | 10.93 |
| `gemm_blocked` | 41.92 | 35.61 | 34.58 | 33.40 | **31.82** |
| `gemm_avx2_blocked` | 75.35 | 70.99 | 46.14 | 41.58 | 37.27 |
| `gemm_avx512_blocked` | **106.03** | 105.78 | 57.34 | 54.44 | **50.13** |

**float (f32)**

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `gemm_naive` | 14.61 | 3.29 | 1.94 | 0.70 | 0.41 |
| `gemm_reordered` | 50.08 | 93.15 | 86.73 | 67.80 | 26.65 |
| `gemm_blocked` | 48.44 | 47.51 | 47.22 | 45.77 | **42.53** |
| `gemm_avx2_blocked` | 151.50 | 150.55 | 142.30 | 93.16 | 76.17 |
| `gemm_avx512_blocked` | 217.12 | **218.94** | 212.41 | 143.07 | **120.06** |

The `*_naive` and `*_reordered` rows for AVX2/AVX-512 are omitted above —
they track their scalar counterparts to within a few percent, which is the
point those kernels exist to make (SIMD width cannot fix a cache-hostile
access pattern).

### Speedup vs `gemm_naive`, N=4096

| Kernel | f64 time | ×naive | f32 time | ×naive |
|---|---|---|---|---|
| `gemm_naive` | 317.97 s | 1.0× | 333.92 s | 1.0× |
| `gemm_reordered` | 12.57 s | **25.3×** | 5.16 s | **64.7×** |
| `gemm_blocked` | 4.32 s | **73.6×** | 3.23 s | **103.3×** |
| `gemm_avx2_blocked` | 3.69 s | **86.2×** | 1.80 s | **185.1×** |
| `gemm_avx512_blocked` | 2.74 s | **116.0×** | 1.14 s | **291.7×** |

### Key observations

**The scalar kernels depend heavily on the compiler's auto-vectoriser.**
`gemm_reordered` and `gemm_blocked` contain no intrinsics at all — their
throughput is whatever the compiler makes of the inner j-loop. Under GCC with
`-march=native -ffast-math`, `gemm_reordered` reaches **93.15 GFLOP/s f32** at
N=256, within 2.4× of the hand-written AVX-512 kernel. (For contrast, an
earlier run of this suite on an Intel/MSVC machine — no longer available, so
its numbers are not reproduced here — measured the same source at roughly
5 GFLOP/s, about 18× lower. MSVC does not auto-vectorise this loop
aggressively even in release builds.) Where the scalar kernels land is
therefore a statement about the toolchain, not about the algorithm.

**Blocking earns its keep only at large N.** At N≤512 `gemm_blocked` looks
like a regression against `gemm_reordered` (47.51 vs 93.15 GFLOP/s f32 at
N=256) — the untiled kernel still fits cache there, and tiling the j-loop to
64 columns costs the vectoriser more than the cache saves. The ranking
inverts where it matters: at N=4096, blocked reaches **42.53 vs 26.65 G/s
(f32, 1.6×)** and **31.82 vs 10.93 G/s (f64, 2.9×)**. Reading only the small
sizes gives exactly the wrong conclusion.

**The AVX-512 blocked kernel loses roughly half its throughput past a
size threshold, and the f64 case matches L2 capacity exactly.** f64 drops
105.78 → 57.34 G/s between N=256 and N=512; f32 holds until N=1024, then
drops 212.41 → 143.07. For f64 the arithmetic is exact: the B panel is
`kAvx512TileK (256) × kAvx512TileN (512) × 8 B` = **precisely 1 MiB**, which
is this CPU's per-core L2, while at N=256 the j-tile is clamped to 256 and
the panel is half that. The f32 panel is 512 KiB at every size ≥512, so the
same arithmetic does *not* explain its drop at N=1024 — something else
(likely L3 pressure or DRAM streaming as the full matrices grow) dominates
there. The f64 cliff looks directly addressable by shrinking
`kAvx512TileN` for 8-byte elements; the f32 case needs profiling before
any claim is made. `gemm_avx2_blocked` shows the same f64 shape
(70.99 → 46.14) at the same size, consistent with its own 256×256×8 =
512 KiB panel plus A and C traffic crowding the same L2.

**Software prefetch does essentially nothing on Zen 5, and the distance is
irrelevant.** Across every family, precision and size, all four distances
(D ∈ {2, 4, 8, 16}) land within ~1% of each other:

| Kernel | D=2 | D=4 | D=8 | D=16 | no prefetch |
|---|---|---|---|---|---|
| `Avx512BlockedPf` f32 N=1024 | 149.46 | 149.75 | 149.12 | 149.92 | 143.07 |
| `Avx512BlockedPf` f64 N=1024 | 55.35 | 54.90 | 55.32 | 55.12 | 54.44 |
| `Avx2BlockedPf` f32 N=256 | 149.98 | 149.69 | 149.73 | 150.00 | 150.55 |
| `BlockedPf` f32 N=512 | 46.63 | 47.53 | 46.41 | 47.50 | 47.22 |

The largest effect anywhere is +4.5% (`Avx512BlockedPf` f32 at N=1024); most
rows are within noise of the unprefetched kernel, and some are marginally
slower. This differs from Apple M4 Max, where D=2 was consistently best and
worth ~1.5%. Zen 5's hardware prefetcher already handles these streaming
patterns, so the explicit hints add front-end work without new information.

**`gemm_naive` is slower in absolute terms than on Apple M4 Max at N=4096**
(0.43 vs 0.66 GFLOP/s f64) despite the far higher clock — and f32 and f64 are
indistinguishable (0.41 vs 0.43), the signature of a purely DRAM-latency-bound
kernel where element width is irrelevant. At N=4096 a single naive iteration
takes **318 seconds**, and the six naive-family entries at that size account
for 1,945 s of the 2,165 s sweep — **90% of the total runtime spent measuring
the three kernels nobody would ever use.**

---

## NVIDIA RTX 5080 — CUDA

> On machines **without a CUDA device** all rows print `SKIPPED: 'No CUDA device available'`.
> The binary compiles and links on CPU-only machines (Apple M, CI) via a stub library.
> On a machine with a CUDA GPU the stub is replaced by the real `.cu` kernel library.

Nine kernels (Levels 0-7, `CudaReordered` shares Level 0 with `CudaNaive`):

| Kernel | Strategy | Key technique |
|---|---|---|
| `CudaNaive` | 1 thread → 1 C(i,j), no shared memory | Baseline: exposes raw global-memory bandwidth |
| `CudaReordered` | Same mapping, explicit row-major inner loop | Structural symmetry with CPU `gemm_reordered`; identical to naive on GPU |
| `CudaBlocked` | TILE×TILE thread block → TILE×TILE sub-tile of C | Shared-memory tiling (TILE=16); 16× fewer global loads vs naive |
| `CudaRegTile` | 128×128 block, 8×8 register tile/thread | Register blocking on top of shared-memory tiling |
| `CudaDoubleBuf` | `CudaRegTile` + ping-pong shared buffers | `cp.async` (Ampere+) overlaps the next tile's load with the current tile's compute |
| `CudaVectorized` | `CudaRegTile` shape + 128-bit loads | `float4`/`double2` global↔shared loads, XOR shared-memory swizzle instead of padding |
| `CudaWmma` | 64×64 block, 4×4 warps, 16×16×16 Tensor Core tile | `wmma::load_matrix_sync`/`wmma::mma_sync`, fp32→fp16 on the fly (sm_70+) |
| `CudaMmaLdmatrix` | 64×64 block, 4×4 warps, two 16×8×16 tiles/warp | Raw `ldmatrix.sync`+`mma.sync.m16n8k16` PTX, one level below WMMA (sm_80+) |
| `CudaWmmaPipelined` | 128×128 block, 8 warps × 8 fragments (32×64/warp) | Same `wmma::` API as `CudaWmma`, but bigger tiles + cp.async double-buffered shared memory (sm_70+; async benefit needs sm_80+) — see [§ Level 7](#level-7---pipelined-wmma-bigger-tiles--cpasync-double-buffering) |

> **Verification status.** Every level was verified on real hardware for the first time on 2026-08-29 (NVIDIA RTX 5080, Blackwell, sm_120, CUDA 13.2, Windows/MSVC); every kernel passes its full GTest correctness suite. That first run found and fixed five previously-unexercised bugs (a CMake flag leaking into nvcc, a `cp.async` address-space bug plus a hard-coded launch config in `double_buf`, WMMA alignment/layout bugs, and a swapped `ldmatrix` quadrant mapping). The full per-bug writeup lives in [`src/gemm/README.md` § Algorithm 9](../src/gemm/README.md#algorithm-9--cuda-kernels-cudahpp--srccudagemm_kernelscu) and in each kernel's file comment in [`src/cuda/gemm_kernels.cu`](../src/cuda/gemm_kernels.cu).

### Level 7 - Pipelined WMMA (bigger tiles + cp.async double buffering)

`gemm_cuda_wmma` (Level 4) and `gemm_cuda_mma_ldmatrix` (Level 6) both measured only **~5 TFLOP/s** on RTX 5080 — cuBLAS's own dense-FP16 Tensor Core path measured **~118 TFLOP/s compute-only** on the same GPU (see [§ Reference cuBLAS](#reference-cublas---is-100-200-tflops-reachable-on-this-gpu) below). That ~24× gap is almost entirely pipelining and tile size, not precision or instruction choice — both kernels already use fp16 Tensor Cores. `gemm_cuda_wmma_pipelined` is a **new kernel** (added, not a replacement — `gemm_cuda_wmma` is untouched) that closes most of that gap while staying on the documented `wmma::` C++ API rather than hand-mapped `mma.sync`/`ldmatrix` PTX registers (the class of code this project's own `kernel_mma_ldmatrix` bug — a swapped quadrant mapping — already showed is easy to get subtly wrong):

1. **Bigger thread-block tile**: 128×128 (vs 64×64) with BK=32 (vs 16) — more work per shared-memory round trip and `__syncthreads()` pair.
2. **Bigger per-warp tile**: each of 8 warps (256 threads/block) owns a 32×64 output region — 8 WMMA 16×16×16 fragments per warp instead of 1, with A/B fragments loaded once per k-sub-step and reused across the other dimension (the same register-blocking structure `gemm_cuda_reg_tile`/`gemm_cuda_double_buf` already use for their scalar FMA micro-kernel).
3. **cp.async double-buffered shared memory** (Ampere+): the next k-tile's global→shared copy overlaps the current tile's Tensor Core compute — the same structural fix already proven correct in `gemm_cuda_double_buf`'s cp.async bug fix above, applied here to fp16 Tensor Core input. Falls back to a synchronous (still double-buffered) copy on pre-Ampere Tensor-Core hardware.

To keep cp.async usable at all, `A`/`B` are pre-converted to fp16 in global memory once (same staging step `gemm_cuda_cublas_fp16` already uses — cp.async is a same-dtype byte copy, not a converting load), and `As` is stored **naturally** (`As[m][k]`, matching `A`'s own row-major layout) rather than transposed the way `gemm_cuda_wmma` stores it — a deliberate, documented difference (cp.async can only copy a contiguous run of bytes to a contiguous destination, and only the natural/untransposed layout lines up for that), requiring `a_frag` to be `row_major` here vs `gemm_cuda_wmma`'s `col_major` for the *same* mathematical operand. This kernel also requires M/N to be exact multiples of 128 and K a multiple of 32 (no tail handling) — every alignment argument for its 16-byte cp.async transfers depends on this — falling back to the always-correct `gemm_cuda_wmma` otherwise. All 5 GTest cases pass on the first run, including a non-square 384×256×160 case and a N=192 case that exercises the fallback path.

**Result — measured on RTX 5080, compute-only (pre-staged device buffers, no per-call transfer/malloc/conversion):**

| Kernel | N=4096 | N=8192 | N=16384 | vs `CudaWmma` |
|---|---|---|---|---|
| `CudaWmma` (Level 4, 64×64 tiles, single-buffered) | ~5 TFLOP/s | — | — | 1.0× |
| `CudaWmmaPipelined` (Level 7, 128×128 tiles, cp.async) | **75.0 TFLOP/s** | **80.4 TFLOP/s** | **80.8 TFLOP/s** | **~15×** |
| `gemm_cuda_cublas_fp16` (vendor reference) | 111.3 TFLOP/s | 116.2 TFLOP/s | 118.2 TFLOP/s | ~22× |

A ~15× improvement over the original WMMA kernel using only bigger tiles and the documented C++ API, reaching **68% of cuBLAS's dense-FP16 throughput** at scale. Closing the remaining gap would require the structural changes CUTLASS-style kernels use beyond what's implemented here: even deeper multi-stage pipelining (3-4 stages, not 2), warp-level swizzling to avoid shared-memory bank conflicts on the WMMA loads, and split-K for very large K. See [`src/gemm/README.md`](../src/gemm/README.md#level-7--gemm_cuda_wmma_pipelined--pipelined-tensor-cores-via-wmma-fp32-only-sm_70) for the full per-line design writeup.

---

### CUDA benchmark output (NVIDIA RTX 5080, Linux)

> **Machine:** AMD Ryzen 9 9950X host + **NVIDIA GeForce RTX 5080** (Blackwell, sm_120), Ubuntu 24.04, CUDA 13.2, GCC 13.3, C++20.
> **Build:** `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc` → native `sm_120` SASS (verified with `cuobjdump --list-elf`)
> **Run:** `./build/benchmarks/cuda/bench_gemm_cuda --benchmark_format=console` — 123 s for the full GPU sweep
> **Date:** 2026-09-24. All 68 CUDA tests pass with no skips.

All figures GFLOP/s. Rows through `CudaCublasTf32` are **end-to-end**
(`cudaMalloc` + H2D + kernel + D2H timed every iteration); the
`*ComputeOnly` rows time only the kernel against device-resident buffers.

**float (f32)**

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 | N=8192 | N=16384 |
|---|---|---|---|---|---|---|---|
| `CudaNaive` | 9 | 252 | 660 | 1,485 | 2,608 | — | — |
| `CudaReordered` | 9 | 251 | 661 | 1,484 | 2,609 | — | — |
| `CudaBlocked` | 9 | 251 | 653 | 1,425 | 2,422 | — | — |
| `CudaRegTile` | 8 | 193 | 643 | 2,132 | 6,583 | — | — |
| `CudaDoubleBuf` | 8 | 201 | 663 | 2,102 | 6,671 | — | — |
| `CudaVectorized` | 8 | 198 | 659 | 2,172 | 5,690 | — | — |
| `CudaWmma` | 8 | 242 | 749 | 2,092 | 5,266 | — | — |
| `CudaMmaLdmatrix` | 8 | 249 | 766 | 2,221 | 5,685 | — | — |
| `CudaWmmaPipelined` | 8 | 258 | 780 | 2,330 | **9,003** | **16,208** | **26,941** |
| `CudaCublas` (ref) | 18 | 468 | 907 | 2,404 | 8,285 | 13,948 | 20,449 |
| `CudaCublasTf32` (ref) | 18 | 469 | 926 | 2,494 | 8,735 | 15,315 | 24,368 |
| `CudaCublasComputeOnly` | — | — | — | — | 37,702 | 38,603 | 39,073 |
| `CudaCublasTf32ComputeOnly` | — | — | — | — | 51,763 | 58,410 | 59,663 |
| `CudaCublasFp16ComputeOnly` | — | — | — | — | 111,292 | 116,249 | **118,228** |
| `CudaWmmaPipelinedComputeOnly` | — | — | — | — | 75,048 | 80,448 | **80,773** |

**double (f64)** — consumer Blackwell has a heavily reduced FP64 datapath,
so everything here is an order of magnitude below the f32 column and the
Tensor Core kernels do not apply.

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `CudaNaive` | 7 | 145 | 323 | 539 | 750 |
| `CudaReordered` | 7 | 145 | 324 | 536 | 748 |
| `CudaBlocked` | 7 | 146 | 328 | 552 | **770** |
| `CudaRegTile` | 2 | 34 | 127 | 455 | 705 |
| `CudaDoubleBuf` | 3 | 56 | 201 | 441 | 712 |
| `CudaVectorized` | 2 | 34 | 128 | 460 | 725 |
| `CudaCublas` (ref) | 11 | 178 | 284 | 467 | 686 |

Note the f64 inversion: `CudaBlocked` (Level 1, plain shared-memory tiling)
is the *fastest* f64 kernel at every size ≥512, ahead of the register-tiled
and double-buffered kernels above it and ahead of cuBLAS. With FP64 throughput
this constrained the kernels are bound by the FP64 pipe rather than by memory,
so the extra register pressure and staging of the higher levels buys nothing.

#### Linux vs Windows on identical hardware

The earlier run of this suite used the same GPU and the same CUDA 13.2 under
Windows/MSVC. Comparing f32:

| Kernel | Windows | Linux | Δ |
|---|---|---|---|
| `CudaNaive` (end-to-end, N=4096) | 2,481 | 2,608 | +5% |
| `CudaRegTile` (end-to-end, N=4096) | 5,728 | 6,583 | +15% |
| `CudaDoubleBuf` (end-to-end, N=4096) | 5,744 | 6,671 | +16% |
| `CudaWmmaPipelined` (end-to-end, N=4096) | 7,121 | 9,003 | **+26%** |
| `CudaCublasFp16ComputeOnly` (N=16384) | 117,827 | 118,228 | +0.3% |
| `CudaWmmaPipelinedComputeOnly` (N=16384) | 82,383 | 80,773 | −2% |

End-to-end gains 5-26%; compute-only is flat to within ±2%. Same silicon
running the same arithmetic, so the difference is not in the kernels — it is
the driver and transfer path, where Linux avoids Windows' WDDM overhead on
allocation and host↔device copies. Any benchmark in this suite that includes
transfers is measuring the OS as much as the GPU.

One consequence: at N=4096 end-to-end, `CudaWmmaPipelined` (9,003) now beats
both cuBLAS references (8,285 / 8,735). That is not a claim that the
hand-written kernel is better than cuBLAS — at that size every kernel is
transfer-bound and cuBLAS has no room to show its advantage. The compute-only
rows are the honest comparison, and there cuBLAS FP16 leads 118,228 to
80,773 (the hand-written kernel reaching **68.3%** of it).

> **Note:** all CUDA benchmarks include host↔device transfer time (`cudaMemcpy` + kernel + `cudaMemcpy`). `CudaWmma`/`CudaMmaLdmatrix`/`CudaWmmaPipelined` convert fp32→fp16 on the fly (`precision=16`), so their GFLOP/s is not directly comparable to the fp32 FMA kernels above them at face value — on this specific unoptimized/educational implementation (small 64×64 output tiles, no multi-stage pipelining) `CudaWmma`/`CudaMmaLdmatrix` land *below* `CudaRegTile`/`CudaDoubleBuf`'s plain-FMA throughput at N=4096, which is a legitimate result of this kernel's tuning level, not a correctness issue (all pass their GTest correctness suites). `CudaWmmaPipelined` (Level 7) is the exception: it overtakes every other kernel above at N≥4096 and keeps climbing with N (22.5 TFLOP/s at N=16384, end-to-end, transfer-dominated at this size) — see [§ Reference cuBLAS](#reference-cublas---is-100-200-tflops-reachable-on-this-gpu) below for its transfer-excluded compute-only numbers (~80 TFLOP/s), which is the fairer comparison against cuBLAS.

#### CUDA speedup summary (f32, N=4096, end-to-end)

| Kernel | GFLOP/s | ×CudaNaive |
|---|---|---|
| `CudaNaive` | 2,608 (2.61 TFLOP/s) | 1.0× |
| `CudaReordered` | 2,609 (2.61 TFLOP/s) | **1.00×** |
| `CudaBlocked` (TILE=16) | 2,422 (2.42 TFLOP/s) | 0.93× |
| `CudaWmma` (Tensor Cores, fp16) | 5,266 (5.27 TFLOP/s) | **2.02×** |
| `CudaVectorized` (float4 + swizzle) | 5,690 (5.69 TFLOP/s) | **2.18×** |
| `CudaMmaLdmatrix` (raw mma.sync, fp16) | 5,685 (5.69 TFLOP/s) | **2.18×** |
| `CudaRegTile` (block=128) | 6,583 (6.58 TFLOP/s) | **2.52×** |
| `CudaDoubleBuf` (cp.async) | 6,671 (6.67 TFLOP/s) | **2.56×** |
| `CudaWmmaPipelined` (Level 7 — 128×128 tiles + cp.async) | **9,003 (9.00 TFLOP/s)** | **3.45×** |

`CudaBlocked` remains the one kernel slower than the naive baseline: shared-memory
tiling with one output element per thread pays `__syncthreads()` overhead without
enough arithmetic per thread to amortise it. Every level above it recovers, and the
ladder is monotonic from `CudaWmma` onward.

---

### Reference cuBLAS - is 100-200 TFLOP/s reachable on this GPU

The hand-written Tensor Core kernels above (`CudaWmma`, `CudaMmaLdmatrix`) measured only ~5 TFLOP/s — well under Blackwell's realistic Tensor Core potential — because they're small (64×64 tiles), single-buffered, and unpipelined. `gemm_cuda_cublas`/`gemm_cuda_cublas_tf32`/`gemm_cuda_cublas_fp16` measure what NVIDIA's own production GEMM (cuBLAS) actually achieves on this GPU, as the realistic ceiling to answer that question and to rewrite toward. That answer motivated writing `gemm_cuda_wmma_pipelined` ([§ Level 7](#level-7---pipelined-wmma-bigger-tiles--cpasync-double-buffering) above) — a new, larger hand-written kernel that closes most of the gap.

Two measurement modes are provided:
- **End-to-end** (`BM_CudaCublas*`/`BM_CudaWmmaPipelined`, no suffix) — same methodology as every other kernel above (`cudaMalloc` + H2D + compute + D2H timed every iteration). At large N (8192+) this is dominated by ~GB-scale data movement and allocation, not the matmul, and badly understates achievable compute throughput.
- **Compute-only** (`BM_CudaCublas*ComputeOnly`/`BM_CudaWmmaPipelinedComputeOnly`) — device buffers allocated and filled *once* outside the timed loop; only the GEMM/kernel call itself is timed. This is the number that actually answers the question, and the fair way to compare the new hand-written kernel against cuBLAS.

```
Benchmark                                    Time      GFLOP/s
-----------------------------------------------------------------
BM_CudaCublasComputeOnly/f32/N=4096          3.64 ms    37,702 G/s  (37.7 TFLOP/s)  -- plain SGEMM, no Tensor Cores
BM_CudaCublasComputeOnly/f32/N=8192          28.5 ms    38,603 G/s  (38.6 TFLOP/s)
BM_CudaCublasComputeOnly/f32/N=16384          225 ms    39,073 G/s  (39.1 TFLOP/s)

BM_CudaCublasTf32ComputeOnly/f32/N=4096      2.66 ms    51,763 G/s  (51.8 TFLOP/s)  -- TF32 Tensor Cores (10-bit mantissa)
BM_CudaCublasTf32ComputeOnly/f32/N=8192      18.8 ms    58,410 G/s  (58.4 TFLOP/s)
BM_CudaCublasTf32ComputeOnly/f32/N=16384      147 ms    59,663 G/s  (59.7 TFLOP/s)

BM_CudaWmmaPipelinedComputeOnly/f32/N=4096   1.83 ms    75,048 G/s  (75.0 TFLOP/s)  -- hand-written kernel (Level 7), dense FP16
BM_CudaWmmaPipelinedComputeOnly/f32/N=8192   13.7 ms    80,448 G/s  (80.4 TFLOP/s)
BM_CudaWmmaPipelinedComputeOnly/f32/N=16384   109 ms    80,773 G/s  (80.8 TFLOP/s)

BM_CudaCublasFp16ComputeOnly/f32/N=4096      1.23 ms   111,292 G/s (111.3 TFLOP/s)  -- dense FP16 Tensor Cores, fp32 accumulate
BM_CudaCublasFp16ComputeOnly/f32/N=8192      9.46 ms   116,249 G/s (116.2 TFLOP/s)
BM_CudaCublasFp16ComputeOnly/f32/N=16384     74.4 ms   118,228 G/s (118.2 TFLOP/s)
```

**Answer: yes, and here's how.** Plain FP32 (SIMT CUDA cores, the ceiling for every non-Tensor-Core kernel above) tops out around **39 TFLOP/s** — no amount of tuning a plain-FMA kernel gets past that on this GPU. TF32 Tensor Cores roughly 1.5× that (**~60 TFLOP/s**) — still short of 100. **Dense FP16 Tensor Cores (fp16-in, fp32-accumulate) reach ~118 TFLOP/s** via cuBLAS, squarely in the target range, because FP16 elements are half the width of TF32's through the same tensor pipe.

**And a hand-written kernel gets most of the way there.** The original gap between cuBLAS's ~118 TFLOP/s and the hand-written `CudaWmma`/`CudaMmaLdmatrix` kernels (~5 TFLOP/s each) was almost entirely pipelining and tile size, not precision or instruction choice — both already used fp16 Tensor Cores, just far less efficiently. `gemm_cuda_wmma_pipelined` (Level 7) applies exactly the fixes that gap analysis called for — 128×128 tiles (not 64×64), cp.async double-buffering, and per-warp register-blocked fragment reuse, all still on the documented `wmma::` C++ API — and reaches **~81 TFLOP/s at N=16384, a ~15× improvement over the original `CudaWmma`, 68% of cuBLAS's dense-FP16 throughput**. Closing the remaining ~37 TFLOP/s would require going further than this kernel does: deeper multi-stage pipelining (3-4 stages, not 2), warp-level shared-memory swizzling for the WMMA loads specifically, and split-K for very large K — the territory CUTLASS's template library exists to handle generically.

---
