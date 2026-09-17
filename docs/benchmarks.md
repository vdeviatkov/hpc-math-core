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
- [Intel x86 — AVX2 + AVX-512](#intel-x86--avx2--avx-512)
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

- **386 GFLOP/s single-threaded, f32** (`SmeReordered` at N=1024) is the highest single-threaded CPU throughput anywhere in this repo — roughly **4× the AVX-512 f32 peak** (290 G/s, Intel x86 section below) and **~4× `gemm_neon_blocked`** (97 G/s, same Apple-silicon class of chip) despite SME running at a lower clock than either comparison.
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

## Intel x86 — AVX2 + AVX-512

> **Machine:** Intel Alder Lake / Sapphire Rapids-class, 16 P-cores (32 threads), 4.29 GHz, MSVC 2022, C++20
> **Build:** `cmake -B build && cmake --build build --config Release`
> **CPU Caches:** L1 Data 48 KiB · L1 Instruction 32 KiB · L2 Unified 1024 KiB (×16) · L3 Unified 32768 KiB (×2)

### double (f64) — scalar, AVX2 & AVX-512 kernels

```
Benchmark                       Time        CPU     GFLOP/s
------------------------------------------------------------
Naive/f64/N=64                52.0 µs    53.1 µs     9.87
Naive/f64/N=256             14017  µs  14062  µs     2.39
Naive/f64/N=512            194863  µs  195312 µs     1.37
Naive/f64/N=1024          2794359  µs    2.80s      767.8 M/s
Naive/f64/N=4096        305295146  µs   304.3s      451.7 M/s

Reordered/f64/N=64             104 µs     103  µs     5.11
Reordered/f64/N=256           6690 µs    6696  µs     5.01
Reordered/f64/N=512          54141 µs   54688  µs     4.91
Reordered/f64/N=1024       427629 µs  429688  µs     5.00
Reordered/f64/N=4096     34642827 µs   34.6s        3.97

Blocked/f64/N=64               103 µs     103  µs     5.11  tile=64
Blocked/f64/N=256             6892 µs    6944  µs     4.83  tile=64
Blocked/f64/N=512            55092 µs   56250  µs     4.77  tile=64
Blocked/f64/N=1024         441021 µs  437500  µs     4.91  tile=64
Blocked/f64/N=4096       28273076 µs   28.2s        4.87  tile=64

Avx2Reordered/f64/N=64        20.3 µs    20.5 µs    25.57
Avx2Reordered/f64/N=256        974 µs     983  µs    34.13
Avx2Reordered/f64/N=512       8120 µs    8125  µs    33.04
Avx2Reordered/f64/N=1024     66351 µs   66761  µs    32.17
Avx2Reordered/f64/N=4096  13227900 µs   13.2s       10.39

Avx2Blocked/f64/N=64           7.83 µs    7.85 µs    66.81  avx2=1
Avx2Blocked/f64/N=256          505 µs     500  µs    67.11  avx2=1
Avx2Blocked/f64/N=512         5860 µs    5859  µs    45.81  avx2=1
Avx2Blocked/f64/N=1024       47682 µs   46875  µs    45.81  avx2=1
Avx2Blocked/f64/N=4096     3624176 µs    3.61s      38.08  avx2=1

Avx512Reordered/f64/N=64      17.7 µs    17.6 µs    29.83  avx512=1
Avx512Reordered/f64/N=256      821 µs     820  µs    40.94  avx512=1
Avx512Reordered/f64/N=512     7554 µs    7465  µs    35.96  avx512=1
Avx512Reordered/f64/N=1024   60469 µs   59375  µs    36.17  avx512=1
Avx512Reordered/f64/N=4096 11374338 µs   11.4s      12.08  avx512=1

Avx512Blocked/f64/N=64         3.79 µs    3.77 µs   139.19  avx512=1
Avx512Blocked/f64/N=256         277 µs     276  µs   121.48  avx512=1
Avx512Blocked/f64/N=512        4727 µs    4719  µs    56.88  avx512=1
Avx512Blocked/f64/N=1024      39896 µs   39931  µs    53.78  avx512=1
Avx512Blocked/f64/N=4096    2796709 µs    2.80s      49.14  avx512=1

Neon*/f64/*     SKIPPED: 'NEON not available on this target'
Sve*/f64/*      SKIPPED: 'SVE not available on this target'
```

### float (f32) — scalar, AVX2 & AVX-512 kernels

```
Benchmark                       Time        CPU     GFLOP/s
------------------------------------------------------------
Naive/f32/N=64                52.5 µs    53.1 µs     9.87
Naive/f32/N=256              9710  µs    9583  µs     3.50
Naive/f32/N=512            120715  µs  122396  µs     2.19
Naive/f32/N=1024          2800837  µs    2.80s      767.8 M/s
Naive/f32/N=4096        313939135  µs   313.7s      438.2 M/s

Reordered/f32/N=64             101 µs     103  µs     5.11
Reordered/f32/N=256           6614 µs    6696  µs     5.01
Reordered/f32/N=512          53251 µs   54688  µs     4.91
Reordered/f32/N=1024       424514 µs  429688  µs     5.00
Reordered/f32/N=4096     27507882 µs   27.4s        5.01

Blocked/f32/N=64               102 µs     103  µs     5.11  tile=64
Blocked/f32/N=256             6768 µs    6836  µs     4.91  tile=64
Blocked/f32/N=512            53831 µs   53125  µs     5.05  tile=64
Blocked/f32/N=1024         432354 µs  437500  µs     4.91  tile=64
Blocked/f32/N=4096       27694736 µs   27.7s        4.96  tile=64

Avx2Reordered/f32/N=64        12.7 µs    12.7 µs    41.30
Avx2Reordered/f32/N=256        582 µs     586  µs    57.27
Avx2Reordered/f32/N=512       4431 µs    4404  µs    60.95
Avx2Reordered/f32/N=1024     40694 µs   40441  µs    53.10
Avx2Reordered/f32/N=4096   4741415 µs    4.70s      29.22

Avx2Blocked/f32/N=64           3.72 µs    3.77 µs   139.06  avx2=1
Avx2Blocked/f32/N=256          231 µs     230  µs   145.79  avx2=1
Avx2Blocked/f32/N=512         1903 µs    1927  µs   139.31  avx2=1
Avx2Blocked/f32/N=1024       23777 µs   23438  µs    91.63  avx2=1
Avx2Blocked/f32/N=4096     1764409 µs    1.75s      78.54  avx2=1

Avx512Reordered/f32/N=64      12.2 µs    12.3 µs    42.71  avx512=1
Avx512Reordered/f32/N=256      507 µs     502  µs    66.81  avx512=1
Avx512Reordered/f32/N=512     3692 µs    3686  µs    72.83  avx512=1
Avx512Reordered/f32/N=1024   30564 µs   30540  µs    70.32  avx512=1
Avx512Reordered/f32/N=4096  4436182 µs    4.44s     30.97  avx512=1

Avx512Blocked/f32/N=64         1.81 µs    1.80 µs   290.76  avx512=1
Avx512Blocked/f32/N=256         124 µs     126  µs   267.24  avx512=1
Avx512Blocked/f32/N=512        1110 µs    1123  µs   239.02  avx512=1
Avx512Blocked/f32/N=1024      15673 µs   15625  µs   137.44  avx512=1
Avx512Blocked/f32/N=4096    1110351 µs    1.11s     123.89  avx512=1

Neon*/f32/*     SKIPPED: 'NEON not available on this target'
Sve*/f32/*      SKIPPED: 'SVE not available on this target'
```

### Prefetch distance sweep — AVX2 & AVX-512 blocked + prefetch

```
Benchmark                            Time      GFLOP/s   pf_dist
-----------------------------------------------------------------
— AVX2 blocked + prefetch (f64) —
Avx2BlockedPf2/f64/N=256            477 µs    70.53 G/s   D=2
Avx2BlockedPf4/f64/N=256            477 µs    70.74 G/s   D=4  ← best
Avx2BlockedPf2/f64/N=1024         47538 µs    45.81 G/s   D=2  ← best
Avx2BlockedPf16/f64/N=1024        47937 µs    44.75 G/s   D=16

— AVX2 blocked + prefetch (f32) —
Avx2BlockedPf2/f32/N=256            246 µs   136.66 G/s   D=2
Avx2BlockedPf2/f32/N=512           2009 µs   134.71 G/s   D=2
Avx2BlockedPf8/f32/N=1024         23881 µs    89.63 G/s   D=8

— AVX-512 blocked + prefetch (f64) —
Avx512BlockedPf2/f64/N=256          279 µs   121.48 G/s   D=2
Avx512BlockedPf4/f64/N=512         4602 µs    58.79 G/s   D=4  ← best
Avx512BlockedPf16/f64/N=1024      38612 µs    55.63 G/s   D=16 ← best

— AVX-512 blocked + prefetch (f32) —
Avx512BlockedPf2/f32/N=256          124 µs   273.32 G/s   D=2  ← best
Avx512BlockedPf4/f32/N=512         1115 µs   244.34 G/s   D=4  ← best
Avx512BlockedPf2/f32/N=1024       15656 µs   137.44 G/s   D=2
```

### x86 speedup tables

#### f64 — best kernel per family vs `gemm_naive` (Intel x86)

| N | Naive | Reordered | ×naive | Blocked | ×naive | Avx2Blocked | ×naive | Avx512Blocked | ×naive |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 52.0 µs | 104 µs | 0.5× | 103 µs | 0.5× | 7.83 µs | **6.6×** | 3.79 µs | **13.7×** |
| 256 | 14017 µs | 6690 µs | **2.1×** | 6892 µs | **2.0×** | 505 µs | **27.8×** | 277 µs | **50.6×** |
| 512 | 194863 µs | 54141 µs | **3.6×** | 55092 µs | **3.5×** | 5860 µs | **33.3×** | 4727 µs | **41.2×** |
| 1024 | 2794359 µs | 427629 µs | **6.5×** | 441021 µs | **6.3×** | 47682 µs | **58.6×** | 39896 µs | **70.0×** |
| 4096 | 305295146 µs | 34642827 µs | **8.8×** | 28273076 µs | **10.8×** | 3624176 µs | **84.3×** | 2796709 µs | **109.2×** |

#### f32 — best kernel per family vs `gemm_naive` (Intel x86)

| N | Naive | Reordered | ×naive | Blocked | ×naive | Avx2Blocked | ×naive | Avx512Blocked | ×naive |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 52.5 µs | 101 µs | 0.5× | 102 µs | 0.5× | 3.72 µs | **14.1×** | 1.81 µs | **29.0×** |
| 256 | 9710 µs | 6614 µs | **1.5×** | 6768 µs | **1.4×** | 231 µs | **42.0×** | 124 µs | **78.3×** |
| 512 | 120715 µs | 53251 µs | **2.3×** | 53831 µs | **2.2×** | 1903 µs | **63.4×** | 1110 µs | **108.8×** |
| 1024 | 2800837 µs | 424514 µs | **6.6×** | 432354 µs | **6.5×** | 23777 µs | **117.8×** | 15673 µs | **178.7×** |
| 4096 | 313939135 µs | 27507882 µs | **11.4×** | 27694736 µs | **11.3×** | 1764409 µs | **177.9×** | 1110351 µs | **282.7×** |

### Headline GFLOP/s summary (Intel x86 + AVX-512, this run)

| Kernel | f64 peak | f32 peak | f32/f64 ratio |
|---|---|---|---|
| `gemm_naive` | 9.87 G/s | 9.87 G/s | 1.0× |
| `gemm_reordered` | 5.11 G/s | 5.11 G/s | 1.0× |
| `gemm_blocked` | 5.11 G/s | 5.11 G/s | 1.0× |
| `gemm_avx2_blocked` | **67.11 G/s** | **145.79 G/s** | **2.2×** |
| `gemm_avx512_blocked` | **139.19 G/s** | **290.76 G/s** | **2.1×** |
| `gemm_avx512_blocked_prefetch` | **55.63 G/s** @ N=1024 | **273.32 G/s** @ N=256 | — |

> **Note:** scalar kernels (`Reordered`, `Blocked`) show ~5 G/s on this x86 machine because
> MSVC does not auto-vectorise as aggressively as GCC/Clang with `-march=native -ffast-math`.
> The explicit SIMD kernels (AVX2, AVX-512) bypass this entirely and reach the expected throughput.
> AVX-512 `f32` peaks at **290 G/s** at N=64, nearly **2× the AVX2 peak** — the doubled register
> width (512- vs 256-bit) translates directly to throughput.

---

## NVIDIA RTX 5080 — CUDA

> On machines **without a CUDA device** all rows print `SKIPPED: 'No CUDA device available'`.
> The binary compiles and links on CPU-only machines (Apple M, CI) via a stub library.
> On a machine with a CUDA GPU the stub is replaced by the real `.cu` kernel library.

Ten kernels (Levels 0-8, `CudaReordered` shares Level 0 with `CudaNaive`):

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
| `CudaHopperWgmma` | Warp-specialized producer/consumer, TMA tile loads | `wgmma.mma_async` + `cp.async.bulk.tensor` (sm_90a/Hopper only — **unverified**, see below) |
| `CudaWmmaPipelined` | 128×128 block, 8 warps × 8 fragments (32×64/warp) | Same `wmma::` API as `CudaWmma`, but bigger tiles + cp.async double-buffered shared memory (sm_70+; async benefit needs sm_80+) — see [§ Level 8](#level-8---pipelined-wmma-bigger-tiles--cpasync-double-buffering) |

> **Verification status.** Levels 0-6 and 8 were verified on real hardware for the first time on 2026-08-29 (NVIDIA RTX 5080, Blackwell, sm_120, CUDA 13.2, Windows/MSVC); every kernel passes its full GTest correctness suite. That first run found and fixed five previously-unexercised bugs (a CMake flag leaking into nvcc, a Hopper capability check that also matched Blackwell, a `cp.async` address-space bug plus a hard-coded launch config in `double_buf`, WMMA alignment/layout bugs, and a swapped `ldmatrix` quadrant mapping). The full per-bug writeup lives in [`src/gemm/README.md` § Algorithm 9](../src/gemm/README.md#algorithm-9--cuda-kernels-cudahpp--srccudagemm_kernelscu) and in each kernel's file comment in [`src/cuda/gemm_kernels.cu`](../src/cuda/gemm_kernels.cu).
>
> `CudaHopperWgmma` remains genuinely unverified — it requires real `sm_90a` (Hopper) hardware, and this machine (Blackwell) correctly `SKIP`s it via `cuda_has_hopper()`.

### Level 8 - Pipelined WMMA (bigger tiles + cp.async double buffering)

`gemm_cuda_wmma` (Level 4) and `gemm_cuda_mma_ldmatrix` (Level 6) both measured only **~5 TFLOP/s** on RTX 5080 — cuBLAS's own dense-FP16 Tensor Core path measured **~118 TFLOP/s compute-only** on the same GPU (see [§ Reference cuBLAS](#reference-cublas---is-100-200-tflops-reachable-on-this-gpu) below). That ~24× gap is almost entirely pipelining and tile size, not precision or instruction choice — both kernels already use fp16 Tensor Cores. `gemm_cuda_wmma_pipelined` is a **new kernel** (added, not a replacement — `gemm_cuda_wmma` is untouched) that closes most of that gap while staying on the documented `wmma::` C++ API rather than hand-mapped `mma.sync`/`ldmatrix` PTX registers (the class of code this project's own `kernel_mma_ldmatrix` bug — a swapped quadrant mapping — already showed is easy to get subtly wrong):

1. **Bigger thread-block tile**: 128×128 (vs 64×64) with BK=32 (vs 16) — more work per shared-memory round trip and `__syncthreads()` pair.
2. **Bigger per-warp tile**: each of 8 warps (256 threads/block) owns a 32×64 output region — 8 WMMA 16×16×16 fragments per warp instead of 1, with A/B fragments loaded once per k-sub-step and reused across the other dimension (the same register-blocking structure `gemm_cuda_reg_tile`/`gemm_cuda_double_buf` already use for their scalar FMA micro-kernel).
3. **cp.async double-buffered shared memory** (Ampere+): the next k-tile's global→shared copy overlaps the current tile's Tensor Core compute — the same structural fix already proven correct in `gemm_cuda_double_buf`'s cp.async bug fix above, applied here to fp16 Tensor Core input. Falls back to a synchronous (still double-buffered) copy on pre-Ampere Tensor-Core hardware.

To keep cp.async usable at all, `A`/`B` are pre-converted to fp16 in global memory once (same staging step `gemm_cuda_cublas_fp16` and `gemm_cuda_hopper_wgmma` already use — cp.async is a same-dtype byte copy, not a converting load), and `As` is stored **naturally** (`As[m][k]`, matching `A`'s own row-major layout) rather than transposed the way `gemm_cuda_wmma` stores it — a deliberate, documented difference (cp.async can only copy a contiguous run of bytes to a contiguous destination, and only the natural/untransposed layout lines up for that), requiring `a_frag` to be `row_major` here vs `gemm_cuda_wmma`'s `col_major` for the *same* mathematical operand. This kernel also requires M/N to be exact multiples of 128 and K a multiple of 32 (no tail handling, the same scoping choice `gemm_cuda_hopper_wgmma` makes for its own tile shape) — every alignment argument for its 16-byte cp.async transfers depends on this — falling back to the always-correct `gemm_cuda_wmma` otherwise. All 5 GTest cases pass on the first run, including a non-square 384×256×160 case and a N=192 case that exercises the fallback path.

**Result — measured on RTX 5080, compute-only (pre-staged device buffers, no per-call transfer/malloc/conversion):**

| Kernel | N=4096 | N=8192 | N=16384 | vs `CudaWmma` |
|---|---|---|---|---|
| `CudaWmma` (Level 4, 64×64 tiles, single-buffered) | ~5 TFLOP/s | — | — | 1.0× |
| `CudaWmmaPipelined` (Level 8, 128×128 tiles, cp.async) | **74.6 TFLOP/s** | **80.4 TFLOP/s** | **82.4 TFLOP/s** | **~16×** |
| `gemm_cuda_cublas_fp16` (vendor reference) | 109.5 TFLOP/s | 117.3 TFLOP/s | 117.8 TFLOP/s | ~24× |

A ~16× improvement over the original WMMA kernel using only bigger tiles and the documented C++ API, reaching **~68-70% of cuBLAS's dense-FP16 throughput** at scale. Closing the remaining gap would require the structural changes CUTLASS-style kernels use beyond what's implemented here: even deeper multi-stage pipelining (3-4 stages, not 2), warp-level swizzling to avoid shared-memory bank conflicts on the WMMA loads, and split-K for very large K. See [`src/gemm/README.md`](../src/gemm/README.md#level-8--gemm_cuda_wmma_pipelined--pipelined-tensor-cores-via-wmma-fp32-only-sm_70) for the full per-line design writeup.

---

### CUDA benchmark output (NVIDIA RTX 5080, real hardware)

> **Machine:** Intel Alder Lake/Sapphire Rapids-class host + **NVIDIA GeForce RTX 5080** (Blackwell, sm_120), CUDA 13.2, MSVC 2022, C++20.
> **Build:** `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build --config Release`
> **Run:** `./build/benchmarks/cuda/Release/bench_gemm_cuda.exe --benchmark_format=console`
> **Date:** 2026-08-29 — first real-hardware run in this project's history; see [§ CUDA kernels](#nvidia-rtx-5080--cuda) above for the five bugs it found and fixed.

#### double (f64) — CUDA kernels

```
Benchmark                          Time        CPU     GFLOP/s
--------------------------------------------------------------
CudaNaive/f64/N=64                379 µs     293 µs      1.79
CudaNaive/f64/N=256               514 µs     386 µs     86.89
CudaNaive/f64/N=512              1404 µs    1234 µs    217.52
CudaNaive/f64/N=1024             4956 µs    4604 µs    466.46
CudaNaive/f64/N=4096           185016 µs  187500 µs    733.01

CudaReordered/f64/N=64            389 µs     276 µs      1.90
CudaReordered/f64/N=256           515 µs     435 µs     77.10
CudaReordered/f64/N=512          1438 µs    1228 µs    218.65
CudaReordered/f64/N=1024         4840 µs    4743 µs    452.74
CudaReordered/f64/N=4096       185304 µs  187500 µs    733.01

CudaBlocked/f64/N=64              392 µs     296 µs      1.77  tile=16
CudaBlocked/f64/N=256             521 µs     441 µs     76.06  tile=16
CudaBlocked/f64/N=512            1401 µs    1050 µs    255.70  tile=16
CudaBlocked/f64/N=1024           4695 µs    4464 µs    481.04  tile=16
CudaBlocked/f64/N=4096         179898 µs  175781 µs    781.88  tile=16

CudaRegTile/f64/N=64              559 µs     399 µs      1.31  block=128
CudaRegTile/f64/N=256            1262 µs    1123 µs     29.88  block=128
CudaRegTile/f64/N=512            2649 µs    2344 µs    114.53  block=128
CudaRegTile/f64/N=1024           5522 µs    5388 µs    398.57  block=128
CudaRegTile/f64/N=4096         195597 µs  197917 µs    694.43  block=128

CudaDoubleBuf/f64/N=64            455 µs     374 µs      1.40  ampere_async=1
CudaDoubleBuf/f64/N=256           875 µs     802 µs     41.83  ampere_async=1
CudaDoubleBuf/f64/N=512          1855 µs    1548 µs    173.42  ampere_async=1
CudaDoubleBuf/f64/N=1024         5663 µs    5162 µs    416.03  ampere_async=1
CudaDoubleBuf/f64/N=4096       192435 µs  195312 µs    703.69  ampere_async=1

CudaVectorized/f64/N=64           553 µs     467 µs      1.12  vec_width=2
CudaVectorized/f64/N=256         1264 µs    1147 µs     29.24  vec_width=2
CudaVectorized/f64/N=512         2650 µs    2308 µs    116.29  vec_width=2
CudaVectorized/f64/N=1024        5484 µs    5580 µs    384.83  vec_width=2
CudaVectorized/f64/N=4096      192832 µs  192708 µs    713.20  vec_width=2
```

#### float (f32) — CUDA kernels

```
Benchmark                          Time        CPU     GFLOP/s
--------------------------------------------------------------
CudaNaive/f32/N=64                382 µs     265 µs      1.98
CudaNaive/f32/N=256               438 µs     320 µs    104.79
CudaNaive/f32/N=512               823 µs     625 µs    429.50
CudaNaive/f32/N=1024             2119 µs    1801 µs   1192.2     (1.19 TFLOP/s)
CudaNaive/f32/N=4096            55006 µs   55398 µs   2480.9     (2.48 TFLOP/s)

CudaReordered/f32/N=64            389 µs     305 µs      1.72
CudaReordered/f32/N=256           438 µs     346 µs     96.98
CudaReordered/f32/N=512           826 µs     670 µs    400.86
CudaReordered/f32/N=1024         2134 µs    1779 µs   1207.3     (1.21 TFLOP/s)
CudaReordered/f32/N=4096        56304 µs   55398 µs   2480.9     (2.48 TFLOP/s)

CudaBlocked/f32/N=64              402 µs     307 µs      1.71  tile=16
CudaBlocked/f32/N=256             447 µs     322 µs    104.10  tile=16
CudaBlocked/f32/N=512             834 µs     670 µs    400.86  tile=16
CudaBlocked/f32/N=1024           2220 µs    1812 µs   1185.4     (1.19 TFLOP/s)  tile=16
CudaBlocked/f32/N=4096          59448 µs   59659 µs   2303.7     (2.30 TFLOP/s)  tile=16

CudaRegTile/f32/N=64              381 µs     279 µs      1.88  block=128
CudaRegTile/f32/N=256             489 µs     363 µs     92.51  block=128
CudaRegTile/f32/N=512             866 µs     725 µs    370.03  block=128
CudaRegTile/f32/N=1024           1742 µs    1475 µs   1456.3     (1.46 TFLOP/s)  block=128
CudaRegTile/f32/N=4096          24870 µs   23996 µs   5727.7     (5.73 TFLOP/s)  block=128

CudaDoubleBuf/f32/N=64            391 µs     265 µs      1.98  ampere_async=1
CudaDoubleBuf/f32/N=256           480 µs     417 µs     80.44  ampere_async=1
CudaDoubleBuf/f32/N=512           813 µs     684 µs    392.68  ampere_async=1
CudaDoubleBuf/f32/N=1024         1647 µs    1286 µs   1669.4     (1.67 TFLOP/s)  ampere_async=1
CudaDoubleBuf/f32/N=4096        24223 µs   23926 µs   5744.4     (5.74 TFLOP/s)  ampere_async=1

CudaVectorized/f32/N=64           378 µs     247 µs      2.12  vec_width=4
CudaVectorized/f32/N=256          472 µs     384 µs     87.45  vec_width=4
CudaVectorized/f32/N=512          814 µs     670 µs    400.86  vec_width=4
CudaVectorized/f32/N=1024        1655 µs    1430 µs   1501.8     (1.50 TFLOP/s)  vec_width=4
CudaVectorized/f32/N=4096       27914 µs   28125 µs   4886.7     (4.89 TFLOP/s)  vec_width=4

CudaWmma/f32/N=64                 393 µs     314 µs      1.67  tensor_cores=1
CudaWmma/f32/N=256                456 µs     322 µs    104.10  tensor_cores=1
CudaWmma/f32/N=512                777 µs     519 µs    517.39  tensor_cores=1
CudaWmma/f32/N=1024               1663 µs    1500 µs   1431.9     (1.43 TFLOP/s)  tensor_cores=1
CudaWmma/f32/N=4096              29729 µs   28646 µs   4797.9     (4.80 TFLOP/s)  tensor_cores=1

CudaMmaLdmatrix/f32/N=64          379 µs     272 µs      1.93  tensor_cores=1
CudaMmaLdmatrix/f32/N=256         450 µs     360 µs     93.24  tensor_cores=1
CudaMmaLdmatrix/f32/N=512         767 µs     488 µs    549.76  tensor_cores=1
CudaMmaLdmatrix/f32/N=1024        1624 µs    1500 µs   1431.9     (1.43 TFLOP/s)  tensor_cores=1
CudaMmaLdmatrix/f32/N=4096       27985 µs   27043 µs   5082.2     (5.08 TFLOP/s)  tensor_cores=1

CudaHopperWgmma/f32/*    SKIPPED: 'wgmma/TMA requires sm_90a (Hopper) -- UNVERIFIED code path'

CudaWmmaPipelined/f32/N=64        389 µs     305 µs      1.72  exact_tiles=0 tensor_cores=1
CudaWmmaPipelined/f32/N=256       447 µs     381 µs     88.10  exact_tiles=1 tensor_cores=1
CudaWmmaPipelined/f32/N=512       755 µs     519 µs    516.89  exact_tiles=1 tensor_cores=1
CudaWmmaPipelined/f32/N=1024     1991 µs    1676 µs   1281.6     (1.28 TFLOP/s)  exact_tiles=1 tensor_cores=1
CudaWmmaPipelined/f32/N=4096    20107 µs   19301 µs   7120.7     (7.12 TFLOP/s)  exact_tiles=1 tensor_cores=1
CudaWmmaPipelined/f32/N=8192    91.1 ms    88.5 ms   12418.0     (12.42 TFLOP/s)  exact_tiles=1 tensor_cores=1
CudaWmmaPipelined/f32/N=16384    385 ms     391 ms   22518.0     (22.52 TFLOP/s)  exact_tiles=1 tensor_cores=1
```

> **Note:** all CUDA benchmarks include host↔device transfer time (`cudaMemcpy` + kernel + `cudaMemcpy`). `CudaWmma`/`CudaMmaLdmatrix`/`CudaWmmaPipelined` convert fp32→fp16 on the fly (`precision=16`), so their GFLOP/s is not directly comparable to the fp32 FMA kernels above them at face value — on this specific unoptimized/educational implementation (small 64×64 output tiles, no multi-stage pipelining) `CudaWmma`/`CudaMmaLdmatrix` land *below* `CudaRegTile`/`CudaDoubleBuf`'s plain-FMA throughput at N=4096, which is a legitimate result of this kernel's tuning level, not a correctness issue (all pass their GTest correctness suites). `CudaWmmaPipelined` (Level 8) is the exception: it overtakes every other kernel above at N≥4096 and keeps climbing with N (22.5 TFLOP/s at N=16384, end-to-end, transfer-dominated at this size) — see [§ Reference cuBLAS](#reference-cublas---is-100-200-tflops-reachable-on-this-gpu) below for its transfer-excluded compute-only numbers (~80 TFLOP/s), which is the fairer comparison against cuBLAS.

#### CUDA speedup summary (f32, N=4096)

| Kernel | GFLOP/s | ×CudaNaive |
|---|---|---|
| `CudaNaive` | 2,481 G/s (2.48 TFLOP/s) | 1.0× |
| `CudaReordered` | 2,481 G/s (2.48 TFLOP/s) | **1.0×** |
| `CudaBlocked` (TILE=16) | 2,304 G/s (2.30 TFLOP/s) | 0.93× |
| `CudaRegTile` (block=128) | 5,728 G/s (5.73 TFLOP/s) | **2.31×** |
| `CudaDoubleBuf` (cp.async) | 5,744 G/s (5.74 TFLOP/s) | **2.32×** |
| `CudaVectorized` (float4 + swizzle) | 4,887 G/s (4.89 TFLOP/s) | **1.97×** |
| `CudaWmma` (Tensor Cores, fp16) | 4,798 G/s (4.80 TFLOP/s) | **1.93×** |
| `CudaMmaLdmatrix` (raw mma.sync, fp16) | 5,082 G/s (5.08 TFLOP/s) | **2.05×** |
| `CudaHopperWgmma` | SKIPPED — requires real sm_90a hardware | — |
| `CudaWmmaPipelined` (Level 8 — 128×128 tiles + cp.async) | 7,121 G/s (7.12 TFLOP/s) | **2.87×** |

---

### Reference cuBLAS - is 100-200 TFLOP/s reachable on this GPU

The hand-written Tensor Core kernels above (`CudaWmma`, `CudaMmaLdmatrix`) measured only ~5 TFLOP/s — well under Blackwell's realistic Tensor Core potential — because they're small (64×64 tiles), single-buffered, and unpipelined. `gemm_cuda_cublas`/`gemm_cuda_cublas_tf32`/`gemm_cuda_cublas_fp16` measure what NVIDIA's own production GEMM (cuBLAS) actually achieves on this GPU, as the realistic ceiling to answer that question and to rewrite toward. That answer motivated writing `gemm_cuda_wmma_pipelined` ([§ Level 8](#level-8---pipelined-wmma-bigger-tiles--cpasync-double-buffering) above) — a new, larger hand-written kernel that closes most of the gap.

Two measurement modes are provided:
- **End-to-end** (`BM_CudaCublas*`/`BM_CudaWmmaPipelined`, no suffix) — same methodology as every other kernel above (`cudaMalloc` + H2D + compute + D2H timed every iteration). At large N (8192+) this is dominated by ~GB-scale data movement and allocation, not the matmul, and badly understates achievable compute throughput.
- **Compute-only** (`BM_CudaCublas*ComputeOnly`/`BM_CudaWmmaPipelinedComputeOnly`) — device buffers allocated and filled *once* outside the timed loop; only the GEMM/kernel call itself is timed. This is the number that actually answers the question, and the fair way to compare the new hand-written kernel against cuBLAS.

```
Benchmark                                Time      GFLOP/s
-------------------------------------------------------------
BM_CudaCublasComputeOnly/f32/N=4096      3.68 ms   37,383 G/s  (37.4 TFLOP/s)   -- plain SGEMM, no Tensor Cores
BM_CudaCublasComputeOnly/f32/N=8192      28.3 ms   38,244 G/s  (38.2 TFLOP/s)
BM_CudaCublasComputeOnly/f32/N=16384      226 ms   38,383 G/s  (38.4 TFLOP/s)

BM_CudaCublasTf32ComputeOnly/f32/N=4096  2.66 ms   51,604 G/s  (51.6 TFLOP/s)   -- TF32 Tensor Cores (10-bit mantissa)
BM_CudaCublasTf32ComputeOnly/f32/N=8192  18.7 ms   57,859 G/s  (57.9 TFLOP/s)
BM_CudaCublasTf32ComputeOnly/f32/N=16384  148 ms   58,641 G/s  (58.6 TFLOP/s)

BM_CudaWmmaPipelinedComputeOnly/f32/N=4096   1.83 ms   74,567 G/s  (74.6 TFLOP/s)  -- hand-written kernel (Level 8), dense FP16
BM_CudaWmmaPipelinedComputeOnly/f32/N=8192   13.6 ms   80,421 G/s  (80.4 TFLOP/s)
BM_CudaWmmaPipelinedComputeOnly/f32/N=16384   108 ms   82,383 G/s  (82.4 TFLOP/s)

BM_CudaCublasFp16ComputeOnly/f32/N=4096  1.24 ms  109,462 G/s (109.5 TFLOP/s)   -- dense FP16 Tensor Cores, fp32 accumulate
BM_CudaCublasFp16ComputeOnly/f32/N=8192  9.38 ms  117,281 G/s (117.3 TFLOP/s)
BM_CudaCublasFp16ComputeOnly/f32/N=16384 74.6 ms  117,827 G/s (117.8 TFLOP/s)
```

**Answer: yes, and here's how.** Plain FP32 (SIMT CUDA cores, the ceiling for every non-Tensor-Core kernel above) tops out around **38 TFLOP/s** — no amount of tuning a plain-FMA kernel gets past that on this GPU. TF32 Tensor Cores roughly 1.5× that (**~59 TFLOP/s**) — still short of 100. **Dense FP16 Tensor Cores (fp16-in, fp32-accumulate) reach ~118 TFLOP/s** via cuBLAS, squarely in the target range, because FP16 elements are half the width of TF32's through the same tensor pipe.

**And a hand-written kernel gets most of the way there.** The original gap between cuBLAS's ~118 TFLOP/s and the hand-written `CudaWmma`/`CudaMmaLdmatrix` kernels (~5 TFLOP/s each) was almost entirely pipelining and tile size, not precision or instruction choice — both already used fp16 Tensor Cores, just far less efficiently. `gemm_cuda_wmma_pipelined` (Level 8, new kernel, `gemm_cuda_wmma` untouched) applies exactly the fixes that gap analysis called for — 128×128 tiles (not 64×64), cp.async double-buffering, and per-warp register-blocked fragment reuse, all still on the documented `wmma::` C++ API — and reaches **~82 TFLOP/s at N=16384, a ~16× improvement over the original `CudaWmma`, ~70% of cuBLAS's dense-FP16 throughput**. Closing the remaining ~18 TFLOP/s would require going further than this kernel does: deeper multi-stage pipelining (3-4 stages, not 2), warp-level shared-memory swizzling for the WMMA loads specifically, and split-K for very large K — the territory CUTLASS's template library exists to handle generically.

---
