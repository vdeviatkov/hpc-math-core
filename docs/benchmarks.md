# Benchmark Results

Measured throughput for every kernel family in this repository, on the three
machines it has been verified on. All numbers come from
`bench_gemm` / `bench_gemm_cuda` (Google Benchmark, console format) built in
Release mode as described in [build.md](build.md). CPU kernels are strictly
single-threaded unless noted (Apple AMX via Accelerate is the exception).

**Contents**

- [Deriving GFLOP/s](#deriving-gflops)
- [Apple M4 Max](#apple-m4-max) — scalar · NEON · SME2 · AMX
- [AMD Zen 5](#amd-zen-5--avx2--avx-512-linux--gcc) — scalar · AVX2 · AVX-512
- [NVIDIA RTX 5080](#nvidia-rtx-5080--cuda) — CUDA Levels 0-7 · cuBLAS reference

Each machine section has the same shape: build metadata, GFLOP/s by kernel
and size, speedup against the naive baseline, then key observations.

---

## Deriving GFLOP/s

```
GFLOP/s = (2 × N³) / (time_µs × 1000)
```

A square N×N GEMM performs `2 × N³` floating-point operations. Dividing by wall-clock time in nanoseconds gives GFLOP/s.

Example: `NeonBlockedPf2/f32/N=512`, 2715 µs → `2 × 512³ / (2715 × 1000)` ≈ **98.9 GFLOP/s**.

---

## Apple M4 Max

> **Machine:** Apple M4 Max, 16 cores, Apple Clang 17, C++20
> **Build:** `cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON` → `-O3 -ffast-math -funroll-loops` (`-march` cleared in favour of `-mcpu=apple-m4`, see [§ SME and AMX build flags](build.md#sme-and-amx-build-flags))
> **CPU Caches:** L1d 64 KiB · L1i 128 KiB · L2 4096 KiB (×16)
> **Load Average:** 4.15 / 4.16 / 4.25 on 16 cores ≈ 26% — moderate background activity, numbers still representative

Four families run here: scalar, NEON, SME2 (opt-in) and AMX via
Accelerate. AVX2, AVX-512 and SVE are absent and report `SKIPPED`.

### GFLOP/s by kernel and size

**double (f64)**

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `gemm_naive` | 9.46 | 2.58 | 2.63 | 2.39 | 0.66 |
| `gemm_reordered` | 27.74 | 16.62 | 16.76 | 16.57 | 16.29 |
| `gemm_blocked` | 27.21 | 25.42 | 22.23 | 19.48 | 20.18 |
| `gemm_neon_naive` | 8.66 | 2.49 | 2.58 | 2.41 | 0.66 |
| `gemm_neon_reordered` | 27.12 | 13.67 | 14.32 | 15.05 | 15.52 |
| `gemm_neon_blocked` | **36.30** | 33.95 | 32.05 | 30.62 | **24.96** |

**float (f32)**

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `gemm_naive` | 9.35 | 2.73 | 2.46 | 2.63 | 0.63 \* |
| `gemm_reordered` | 86.10 | 32.32 | 32.92 | 32.99 | 32.83 |
| `gemm_blocked` | 85.85 | 83.51 | 51.27 | 43.00 | 31.55 |
| `gemm_neon_naive` | 9.31 | 3.81 | 3.62 | 2.66 | 0.63 \* |
| `gemm_neon_reordered` | 25.82 | 29.64 | 27.31 | 28.26 | 30.59 |
| `gemm_neon_blocked` | **97.03** | 96.43 | 96.73 | 94.41 | **72.66** |

\* `Naive/f32/N=4096` and `NeonNaive/f32/N=4096` were captured while an
unrelated process ran concurrently — a batching artefact of this run, not a
property of the kernels. Their wall-clock time is inflated; these two cells
use CPU time instead, which is unaffected.

### ARM SME2 (`-DHPC_ENABLE_SME=ON`)

> **SVL (streaming vector length):** 16 f32 / 8 f64 elements — reported live via the `svl` benchmark counter
> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter=Sme`

SME computes GEMM with a fundamentally different primitive than every other CPU kernel above: instead of per-lane FMA, a single `FMOPA` instruction accumulates a whole SVL×SVL **outer product** into a 2-D hardware accumulator (ZA), the same class of operation as NVIDIA Tensor Cores (`gemm_cuda_wmma`) and Apple's own AMX coprocessor (below) — see [src/gemm/README.md](../src/gemm/README.md#algorithm-10--arm-sme2-scalable-matrix-extension) for the full architectural writeup, including the two real hardware/toolchain issues found while building this (gather-loads are illegal in SME streaming mode; combining `-march=native` with `-mcpu=apple-m4` silently disables SME).

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_sme_naive` f64 | 1.48 | 1.52 | 1.52 | 1.52 | — | — |
| `gemm_sme_reordered` f64 | 54.08 | 100.80 | 112.46 | 112.57 | 39.17 | — |
| `gemm_sme_blocked` f64 | 53.60 | 100.79 | 112.21 | **116.04** | 54.44 | — |
| `gemm_sme_naive` f32 | 2.94 | 3.12 | 3.13 | 3.13 | — | — |
| `gemm_sme_reordered` f32 | 65.77 | 222.58 | 317.60 | **385.56** | 236.08 | 129.19 |
| `gemm_sme_blocked` f32 | 64.83 | 222.34 | 315.63 | 384.23 | **293.34** | **180.40** |

- **386 GFLOP/s single-threaded f32** (`SmeReordered`, N=1024) is the highest single-threaded CPU throughput anywhere in this repo — roughly **1.8×** the AVX-512 f32 peak (219 G/s, AMD Zen 5 below) and **~4×** `gemm_neon_blocked` (97 G/s, same class of chip), despite SME running at a lower clock than either.
- **`SmeNaive` is pinned at ~3 GFLOP/s, flat across N.** The same "SIMD width doesn't fix cache-hostile access" lesson every `*_naive` kernel here demonstrates, except the hostility is structural: SME's streaming mode forbids gather-loads entirely (Clang rejects `svld1_gather_index` with "builtin can only be called from a non-streaming function"), so the outer product's column vector must be assembled by a scalar loop on every k-iteration.
- **`SmeReordered` fixes that by packing once per row-tile** — a single scalar pass over `A(i0..i0+16, :)` reused across every column-tile, instead of once per (row-tile, column-tile) pair. A 22–123× improvement depending on N, for identical arithmetic.
- **`SmeBlocked` wins once the packed panel stops fitting cache.** At N=2048/4096 `SmeReordered`'s unbounded `SVL × K` buffer (128 KB / 256 KB) exceeds per-core L1 and the re-reads cost real throughput (236→129 G/s). Bounding the panel to a fixed K-tile (256 columns → 16 KB, comfortably L1-resident) and paying an extra C load/store per K-tile recovers most of it (293→180 G/s) — the same trade-off as `gemm_blocked` vs `gemm_reordered` at the start of the ladder, one abstraction level up.
- **f64 peaks far lower than f32** (116 vs 386 G/s) because SVL is fixed in *bytes*: SVL=8 f64 vs 16 f32, so each f64 outer product covers a quarter of the elements (8×8 vs 16×16) per instruction.

### Apple AMX (via Accelerate.framework)

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter=Amx` (`HPC_ENABLE_AMX` defaults ON on Apple)

This is Apple's own AMX coprocessor, reached through Accelerate.framework's BLAS (`cblas_sgemm`/`cblas_dgemm`) rather than any hand-written kernel — see [§ SME and AMX build flags](build.md#sme-and-amx-build-flags) and [src/gemm/README.md](../src/gemm/README.md#algorithm-11--apple-amx-via-accelerateframework) for why this is architecturally unrelated to Intel's AMX, why `gemm_amx_naive`/`_reordered`/`_blocked` are intentionally identical wrappers, and why these numbers are **not** a single-core comparison against the rest of this document (Accelerate's BLAS may use multiple cores internally).

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_amx_naive` f64 | 330.00 | 465.66 | 820.39 | 839.41 | — | — |
| `gemm_amx_reordered` f64 | 329.55 | 465.92 | 822.24 | 858.59 | 801.48 | 809.79 |
| `gemm_amx_blocked` f64 | 329.78 | 466.61 | 822.88 | **860.28** | 804.22 | 813.36 |
| `gemm_amx_naive` f32 | 807.89 | 1,729 | 2,984 | 3,281 | — | — |
| `gemm_amx_reordered` f32 | 799.46 | 1,727 | 2,923 | 3,261 | 3,218 | 3,186 |
| `gemm_amx_blocked` f32 | 797.61 | 1,724 | 2,906 | **3,296** | 3,202 | 3,156 |

- **Up to 3.3 TFLOP/s f32 and 860 GFLOP/s f64** — by a wide margin the highest throughput in this repo, ~8.5× the hand-written `gemm_sme_reordered` f32 peak and ~7.4× `gemm_sme_blocked`'s f64 peak. Not a fair fight: Accelerate is Apple's own vendor-tuned BLAS and, unlike every hand-written kernel here, is free to use every core. The jump from ~800 G/s at N=64 to ~3.3 T/s at N≥1024 is consistent with more threads coming online as the problem grows, not only better cache behaviour.
- **All three variants produce near-identical numbers at every size** (3,281 / 3,261 / 3,296 G/s at N=1024 f32) — exactly as expected, since all three call the same `cblas_sgemm`/`cblas_dgemm` wrapper (see [src/gemm/amx.hpp](../src/gemm/amx.hpp)). The ≤1% spread is measurement noise; Accelerate exposes no staging knob for the naive/reordered/blocked progression to act on.
- **The f32/f64 ratio is ~3.8×, not the ~2× lane-count ratio seen elsewhere** (NEON 2.7×, AVX-512 ~2×) — consistent with Accelerate exploiting a wider or more specialised f32 path beyond simple lane doubling, though Apple does not document this and it cannot be confirmed without disassembly.
- **If the question is "what is the fastest way to multiply matrices on this Mac", this is the answer** — call `cblas_sgemm`/`cblas_dgemm` directly. The value of the rest of this repository is the pedagogy of reaching a meaningful fraction of that ceiling by hand, one optimisation at a time.

### Prefetch distance sweep

Benchmarks named `<Family>BlockedPf<D>/<prec>/N=<size>` sweep prefetch distance D ∈ {2, 4, 8, 16} (rows ahead), with three `__builtin_prefetch` sites per kernel:

- **[PF-A]** `A(i + D×kRegRows, k_blk)` → L2 (read)
- **[PF-B]** `B(k_blk + TileK, 0)` → L2 (read), at the k-tile boundary
- **[PF-C]** `C(i + D×kRegRows, j_blk)` → L1 (write)

GFLOP/s, best distance per row in bold, against the same kernel without prefetch:

| Kernel | N | D=2 | D=4 | D=8 | D=16 | no prefetch |
|---|---|---|---|---|---|---|
| `BlockedPf` f64 | 256 | 14.74 | **14.81** | 14.79 | 14.61 | 25.42 |
| `BlockedPf` f64 | 512 | 12.96 | 12.97 | 12.93 | **13.00** | 22.23 |
| `BlockedPf` f64 | 1024 | 11.36 | **11.41** | 11.35 | 11.35 | 19.48 |
| `BlockedPf` f32 | 256 | **28.75** | 28.66 | 28.48 | 27.37 | 83.51 |
| `BlockedPf` f32 | 512 | 22.15 | 22.14 | **22.19** | 22.17 | 51.27 |
| `BlockedPf` f32 | 1024 | 21.39 | 21.43 | **21.44** | 21.33 | 43.00 |
| `NeonBlockedPf` f64 | 256 | 33.77 | 32.20 | **33.80** | 32.19 | 33.95 |
| `NeonBlockedPf` f64 | 512 | **31.94** | 30.66 | 31.87 | 30.75 | 32.05 |
| `NeonBlockedPf` f64 | 1024 | 30.30 | 29.66 | **30.43** | 29.83 | 30.62 |
| `NeonBlockedPf` f32 | 256 | **97.86** | 95.67 | 95.88 | 96.18 | 96.43 |
| `NeonBlockedPf` f32 | 512 | **98.26** | 96.17 | 96.10 | 96.28 | 96.73 |
| `NeonBlockedPf` f32 | 1024 | **95.46** | 93.49 | 93.71 | 93.97 | 94.41 |

**Prefetch badly hurts the scalar kernel** — 14.81 vs 25.42 G/s at f64 N=256, and 28.75 vs 83.51 at f32 N=256, a 2.9× loss. That kernel is entirely compiler-auto-vectorised; the hardware prefetcher already handles its simple streaming access, so the explicit hints only add front-end pressure to a tight inner loop.

**On the NEON kernel it gives a small consistent gain** at D=2: +1.5% / +1.6% / +1.1% for f32 at N=256/512/1024. For f64, D=2 and D=8 trade the lead, both ahead of D=4 and D=16. A rule of thumb that fits this hardware:

```
optimal D ≈ ceil(L2_latency_cycles / cycles_per_micro_kernel_call)
          ≈ ceil(12 / ~6) = 2
```

### Speedup vs `gemm_naive`

**f64**

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive |
|---|---|---|---|---|---|---|---|
| 64 | 55.5 µs | 18.9 µs | **2.9×** | 19.3 µs | **2.9×** | 14.4 µs | **3.9×** |
| 256 | 13,036 µs | 2,020 µs | **6.5×** | 1,320 µs | **9.9×** | 989 µs | **13.2×** |
| 512 | 102,246 µs | 16,019 µs | **6.4×** | 12,079 µs | **8.5×** | 8,380 µs | **12.2×** |
| 1024 | 924,366 µs | 129,596 µs | **7.1×** | 110,250 µs | **8.4×** | 70,187 µs | **13.2×** |
| 4096 | 207.5 s | 8.44 s | **24.6×** | 6.81 s | **30.5×** | 5.51 s | **37.7×** |

**f32**

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive |
|---|---|---|---|---|---|---|---|
| 64 | 56.1 µs | 6.09 µs | **9.2×** | 6.12 µs | **9.2×** | 5.41 µs | **10.4×** |
| 256 | 12,278 µs | 1,039 µs | **11.8×** | 402 µs | **30.5×** | 348 µs | **35.3×** |
| 512 | 109,000 µs | 8,157 µs | **13.4×** | 5,238 µs | **20.8×** | 2,776 µs | **39.3×** |
| 1024 | 815,153 µs | 65,121 µs | **12.5×** | 49,952 µs | **16.3×** | 22,752 µs | **35.8×** |
| 4096 | 218.4 s \* | 4.19 s | **52.1×** | 4.36 s | **50.1×** | 1.89 s | **115.4×** |

\* Uses `Naive/f32/N=4096`'s CPU time, since its wall-clock was contended on this run (see the note under the f32 table above). The other columns' timings are unaffected.

### Key observations

**Cache-access pattern dominates at large N.** `gemm_naive` delivers nearly identical GFLOP/s for f32 and f64 at every size — both are DRAM-bandwidth bound on the column-stride gather of B, and element width is irrelevant once you are waiting on cache-miss latency. Switching to i-k-j (`gemm_reordered`) makes B and C sequential so every cache line is fully consumed: at N=4096 that is **25× faster than naive for f64** and **52× for f32** (twice the elements per cache line → twice the bandwidth).

**Explicit SIMD beats auto-vectorisation mainly by staying flat.** `gemm_reordered` and `gemm_blocked` carry no NEON intrinsics — the compiler vectorises the sequential inner j-loop under `-ffast-math`, reaching ~86 GFLOP/s f32 at small N. Adding an explicit 4×4 Q-register tile on top of L2 blocking changes the *shape* of the curve:

| Kernel | f32 N=256 | f32 N=512 | f32 N=1024 |
|---|---|---|---|
| `gemm_blocked` (auto-vectorised) | 83.5 | 51.3 | 43.0 |
| `gemm_neon_blocked` (explicit) | **96.4** | **96.7** | **94.4** |

The register tile holds ~95–97 GFLOP/s from N=64 through N=1024. The auto-vectorised kernel decays 84→43 because C rows are evicted from L1 between k-iterations as N grows.

**NEON f64 vs f32 is 2.7×, not the theoretical 2×.** A Q-register holds 4 f32 lanes or 2 f64 lanes, so lane count alone predicts 2×. `gemm_neon_blocked` peaks at ~36 G/s f64 and ~97 G/s f32. The extra 0.7× comes from f32 tiles fitting entirely in L1 at sizes where f64 tiles spill.

**Peak per family on this machine:**

| Kernel | f64 peak | f32 peak | f32/f64 |
|---|---|---|---|
| `gemm_naive` | 9.46 | 9.35 | 1.0× |
| `gemm_reordered` | 27.74 | 86.10 | 3.1× |
| `gemm_blocked` | 27.21 | 85.85 | 3.2× |
| `gemm_neon_blocked` | 36.30 | 97.03 | 2.7× |
| `gemm_neon_blocked_prefetch` (D=2) | 33.77 | 98.26 | 2.9× |
| `gemm_sme_reordered` | 112.57 | **385.56** | 3.4× |
| `gemm_amx_blocked` (Accelerate) | **860.28** | **3,295.78** | 3.8× |

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

### Speedup vs `gemm_naive`

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
| `CudaWmmaPipelined` | 128×128 block, 8 warps × 8 fragments (32×64/warp) | Same `wmma::` API as `CudaWmma`, but bigger tiles + cp.async double-buffered shared memory (sm_70+; async benefit needs sm_80+) — see [§ Level 7](#level-7--pipelined-wmma-bigger-tiles--cpasync-double-buffering) |

> **Verification status.** Every level was verified on real hardware for the first time on 2026-08-29 (NVIDIA RTX 5080, Blackwell, sm_120, CUDA 13.2, Windows/MSVC); every kernel passes its full GTest correctness suite. That first run found and fixed five previously-unexercised bugs (a CMake flag leaking into nvcc, a `cp.async` address-space bug plus a hard-coded launch config in `double_buf`, WMMA alignment/layout bugs, and a swapped `ldmatrix` quadrant mapping). The full per-bug writeup lives in [`src/gemm/README.md` § Algorithm 9](../src/gemm/README.md#algorithm-9--cuda-kernels-cudahpp--srccudagemm_kernelscu) and in each kernel's file comment in [`src/cuda/gemm_kernels.cu`](../src/cuda/gemm_kernels.cu).

### Level 7 — Pipelined WMMA (bigger tiles + cp.async double buffering)

`gemm_cuda_wmma` (Level 4) and `gemm_cuda_mma_ldmatrix` (Level 6) both measured only **~5 TFLOP/s** on RTX 5080 — cuBLAS's own dense-FP16 Tensor Core path measured **~118 TFLOP/s compute-only** on the same GPU (see [§ Reference cuBLAS](#reference-cublas--the-achievable-ceiling) below). That ~24× gap is almost entirely pipelining and tile size, not precision or instruction choice — both kernels already use fp16 Tensor Cores. `gemm_cuda_wmma_pipelined` is a **new kernel** (added, not a replacement — `gemm_cuda_wmma` is untouched) that closes most of that gap while staying on the documented `wmma::` C++ API rather than hand-mapped `mma.sync`/`ldmatrix` PTX registers (the class of code this project's own `kernel_mma_ldmatrix` bug — a swapped quadrant mapping — already showed is easy to get subtly wrong):

1. **Bigger thread-block tile**: 128×128 (vs 64×64) with BK=32 (vs 16) — more work per shared-memory round trip and `__syncthreads()` pair.
2. **Bigger per-warp tile**: each of 8 warps (256 threads/block) owns a 32×64 output region — 8 WMMA 16×16×16 fragments per warp instead of 1, with A/B fragments loaded once per k-sub-step and reused across the other dimension (the same register-blocking structure `gemm_cuda_reg_tile`/`gemm_cuda_double_buf` already use for their scalar FMA micro-kernel).
3. **cp.async double-buffered shared memory** (Ampere+): the next k-tile's global→shared copy overlaps the current tile's Tensor Core compute — the same structural fix already proven correct in `gemm_cuda_double_buf`'s cp.async bug fix above, applied here to fp16 Tensor Core input. Falls back to a synchronous (still double-buffered) copy on pre-Ampere Tensor-Core hardware.

To keep cp.async usable at all, `A`/`B` are pre-converted to fp16 in global memory once (same staging step `gemm_cuda_cublas_fp16` already uses — cp.async is a same-dtype byte copy, not a converting load), and `As` is stored **naturally** (`As[m][k]`, matching `A`'s own row-major layout) rather than transposed the way `gemm_cuda_wmma` stores it — a deliberate, documented difference (cp.async can only copy a contiguous run of bytes to a contiguous destination, and only the natural/untransposed layout lines up for that), requiring `a_frag` to be `row_major` here vs `gemm_cuda_wmma`'s `col_major` for the *same* mathematical operand. This kernel also requires M/N to be exact multiples of 128 and K a multiple of 32 (no tail handling) — every alignment argument for its 16-byte cp.async transfers depends on this — falling back to the always-correct `gemm_cuda_wmma` otherwise. All 5 GTest cases pass on the first run, including a non-square 384×256×160 case and a N=192 case that exercises the fallback path.

A fourth change — padding the shared-memory leading dimensions — came later, from profiling; see [§ Removing the bank conflicts](#removing-the-bank-conflicts) below.

**Result — measured on RTX 5080, compute-only (pre-staged device buffers, no per-call transfer/malloc/conversion):**

| Kernel | N=4096 | N=8192 | N=16384 | vs `CudaWmma` |
|---|---|---|---|---|
| `CudaWmma` (Level 4, 64×64 tiles, single-buffered) | ~5 TFLOP/s | — | — | 1.0× |
| `CudaWmmaPipelined` (Level 7, 128×128 tiles, cp.async) | **97.2 TFLOP/s** | **101.5 TFLOP/s** | **100.5 TFLOP/s** | **~19×** |
| `gemm_cuda_cublas_fp16` (vendor reference) | 108.7 TFLOP/s | 117.0 TFLOP/s | 120.5 TFLOP/s | ~23× |

A ~19× improvement over the original WMMA kernel using only bigger tiles, register-blocked fragment reuse, cp.async double buffering and a padded shared-memory layout — all on the documented C++ API — reaching **83% of cuBLAS's dense-FP16 throughput** at scale. See [`src/gemm/README.md`](../src/gemm/README.md#level-7--gemm_cuda_wmma_pipelined--pipelined-tensor-cores-via-wmma-fp32-only-sm_70) for the full per-line design writeup.

### Removing the bank conflicts

The first version of this kernel measured 80.8 TFLOP/s, 68% of cuBLAS. Nsight Compute identified why, and the fix was three constants.

Unpadded, both shared tiles are pathological for banking. A `wmma::load_matrix_sync` fragment reads 16 rows of 16 halves, and the bank a row starts in is `(row × ld × 2 / 4) % 32`:

| Tile | `ld` | Row stride | Distinct bank-starts over 16 rows | Conflict |
|---|---|---|---|---|
| `As` unpadded | 32 halves | 64 B | 2 | 8-way |
| `Bs` unpadded | 128 halves | 256 B — exactly 2 bank cycles, so *every* row starts in the same bank | 1 | 16-way |
| `As` **+8 pad** | 40 halves | 80 B | 8 | **2-way** |
| `Bs` **+8 pad** | 136 halves | 272 B | 8 | **2-way** |

Padding by **8** halves, not the usual 1: `wmma::load_matrix_sync` requires the leading dimension to be a multiple of 8 `__half` elements, and `cp.async` requires a 16-byte-aligned destination. A `+1` pad violates both — that is the `cudaErrorMisalignedAddress` bug recorded against `gemm_cuda_wmma`. 8 halves = 16 bytes satisfies both. `+16` would be *worse* (4-way); `+24` is equal but costs more memory.

An XOR swizzle — the usual zero-memory-cost alternative, and what `gemm_cuda_vectorized` uses — is **not applicable here**: `load_matrix_sync` takes a plain `(pointer, ld)` pair and cannot express a permuted layout. Swizzling requires hand-mapped `mma.sync`/`ldmatrix` addressing, which is `gemm_cuda_mma_ldmatrix`'s job.

**Measured, N=4096 compute-only:**

| Metric | Before | After |
|---|---|---|
| Shared-load bank conflicts | 285,879,068 | **519,545** |
| …as a share of load wavefronts | 85% | **1.0%** |
| Shared-load wavefronts | 336,210,716 | **50,851,193** |
| L1/TEX throughput | 87.9% | **31.8%** |
| Tensor pipe utilisation | 68.0% | **86.1%** |
| Warp stalls — MIO throttle | 26.3% | **3.9%** |
| Warp stalls — short scoreboard | 13.6% | **3.2%** |
| Registers per thread | 126 | 126 |
| **Throughput** | **80.8 TFLOP/s** | **100.5 TFLOP/s** |

The kernel had been moving 6.7× more shared-memory load traffic than the algorithm requires; afterwards it is within 1% of the theoretical minimum wavefront count. Shared memory per block rises 32 → 37 KB, dropping the shared-memory occupancy limit from 3 blocks/SM to 2 — free here, because 126 registers/thread already capped it at 2.

**What now limits this kernel:** register pressure. Occupancy is 33.3% (`Block Limit Registers: 2`), and Nsight estimates ~67% headroom from occupancy alone. Beyond that, closing the last 17% to cuBLAS would need deeper multi-stage pipelining (3-4 stages, not 2) and split-K for very large K — the territory CUTLASS exists to handle generically.

---

### GFLOP/s by kernel and size

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
| `CudaWmmaPipelined` | 8 | 262 | 786 | 2,346 | **9,000** | **16,709** | **29,115** |
| `CudaCublas` (ref) | 18 | 468 | 907 | 2,404 | 8,285 | 13,948 | 20,449 |
| `CudaCublasTf32` (ref) | 18 | 469 | 926 | 2,494 | 8,735 | 15,315 | 24,368 |
| `CudaCublasComputeOnly` | — | — | — | — | 37,702 | 38,603 | 39,073 |
| `CudaCublasTf32ComputeOnly` | — | — | — | — | 51,763 | 58,410 | 59,663 |
| `CudaCublasFp16ComputeOnly` | — | — | — | — | 108,658 | 116,991 | **120,499** |
| `CudaWmmaPipelinedComputeOnly` | — | — | — | — | 97,161 | 101,474 | **100,481** |

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

### Linux vs Windows on identical hardware

The earlier run of this suite used the same GPU and the same CUDA 13.2 under
Windows/MSVC. Comparing f32, **against the pre-padding version of
`gemm_cuda_wmma_pipelined`** so both columns run identical source:

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

The current kernel is faster than both columns — the shared-memory padding
described above lifted the Linux compute-only figure from 80,773 to 100,481.
Those numbers are not in this table because no Windows run exists for the
padded kernel, and comparing different source across two operating systems
would measure nothing.

One consequence of the transfer-bound regime: at N=4096 end-to-end,
`CudaWmmaPipelined` (9,000) beats both cuBLAS references (8,285 / 8,735).
That is not a claim that the hand-written kernel is better than cuBLAS — at
that size every kernel is transfer-bound and cuBLAS has no room to show its
advantage. The compute-only rows are the honest comparison, and there cuBLAS
FP16 leads 120,499 to 100,481 (the hand-written kernel reaching **83.4%**
of it).

> **Note:** all CUDA benchmarks include host↔device transfer time (`cudaMemcpy` + kernel + `cudaMemcpy`). `CudaWmma`/`CudaMmaLdmatrix`/`CudaWmmaPipelined` convert fp32→fp16 on the fly (`precision=16`), so their GFLOP/s is not directly comparable to the fp32 FMA kernels above them at face value — on this specific unoptimized/educational implementation (small 64×64 output tiles, no multi-stage pipelining) `CudaWmma`/`CudaMmaLdmatrix` land *below* `CudaRegTile`/`CudaDoubleBuf`'s plain-FMA throughput at N=4096, which is a legitimate result of this kernel's tuning level, not a correctness issue (all pass their GTest correctness suites). `CudaWmmaPipelined` (Level 7) is the exception: it overtakes every other kernel above at N≥4096 and keeps climbing with N (29.1 TFLOP/s at N=16384, end-to-end, transfer-dominated at this size) — see [§ Reference cuBLAS](#reference-cublas--the-achievable-ceiling) below for its transfer-excluded compute-only numbers (~100 TFLOP/s), which is the fairer comparison against cuBLAS.

### Speedup vs `CudaNaive`, N=4096

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
| `CudaWmmaPipelined` (Level 7 — 128×128 tiles + cp.async) | **9,000 (9.00 TFLOP/s)** | **3.45×** |

`CudaBlocked` remains the one kernel slower than the naive baseline: shared-memory
tiling with one output element per thread pays `__syncthreads()` overhead without
enough arithmetic per thread to amortise it. Every level above it recovers, and the
ladder is monotonic from `CudaWmma` onward.

---

### Reference cuBLAS — the achievable ceiling

The hand-written Tensor Core kernels above (`CudaWmma`, `CudaMmaLdmatrix`) measured only ~5 TFLOP/s — well under Blackwell's realistic Tensor Core potential — because they're small (64×64 tiles), single-buffered, and unpipelined. `gemm_cuda_cublas`/`gemm_cuda_cublas_tf32`/`gemm_cuda_cublas_fp16` measure what NVIDIA's own production GEMM (cuBLAS) actually achieves on this GPU, as the realistic ceiling to answer that question and to rewrite toward. That answer motivated writing `gemm_cuda_wmma_pipelined` ([§ Level 7](#level-7--pipelined-wmma-bigger-tiles--cpasync-double-buffering) above) — a new, larger hand-written kernel that closes most of the gap.

Two measurement modes are provided:
- **End-to-end** (`BM_CudaCublas*`/`BM_CudaWmmaPipelined`, no suffix) — same methodology as every other kernel above (`cudaMalloc` + H2D + compute + D2H timed every iteration). At large N (8192+) this is dominated by ~GB-scale data movement and allocation, not the matmul, and badly understates achievable compute throughput.
- **Compute-only** (`BM_CudaCublas*ComputeOnly`/`BM_CudaWmmaPipelinedComputeOnly`) — device buffers allocated and filled *once* outside the timed loop; only the GEMM/kernel call itself is timed. This is the number that actually answers the question, and the fair way to compare the new hand-written kernel against cuBLAS.

Compute-only, TFLOP/s (ascending):

| Path | Precision | N=4096 | N=8192 | N=16384 |
|---|---|---|---|---|
| `cublasSgemm` | FP32, SIMT cores — no Tensor Cores | 37.7 | 38.6 | 39.1 |
| `cublasGemmEx` TF32 | TF32 Tensor Cores, 10-bit mantissa | 51.8 | 58.4 | 59.7 |
| **`gemm_cuda_wmma_pipelined`** | **dense FP16 Tensor Cores, hand-written** | **97.2** | **101.5** | **100.5** |
| `cublasGemmEx` FP16 | dense FP16 Tensor Cores, fp32 accumulate | 108.7 | 117.0 | **120.5** |

**Yes — via dense FP16 Tensor Cores.** Plain FP32 (SIMT CUDA cores, the ceiling for every non-Tensor-Core kernel above) tops out around **39 TFLOP/s** — no amount of tuning a plain-FMA kernel gets past that on this GPU. TF32 Tensor Cores roughly 1.5× that (**~60 TFLOP/s**) — still short of 100. **Dense FP16 Tensor Cores (fp16-in, fp32-accumulate) reach ~118 TFLOP/s** via cuBLAS, squarely in the target range, because FP16 elements are half the width of TF32's through the same tensor pipe.

**And a hand-written kernel gets most of the way there.** The original gap between cuBLAS's ~118 TFLOP/s and the hand-written `CudaWmma`/`CudaMmaLdmatrix` kernels (~5 TFLOP/s each) was almost entirely pipelining and tile size, not precision or instruction choice — both already used fp16 Tensor Cores, just far less efficiently. `gemm_cuda_wmma_pipelined` (Level 7) applies exactly the fixes that gap analysis called for — 128×128 tiles (not 64×64), cp.async double-buffering, and per-warp register-blocked fragment reuse, all still on the documented `wmma::` C++ API — and — after a further fix, padding the shared-memory leading dimensions to eliminate bank conflicts ([§ Removing the bank conflicts](#removing-the-bank-conflicts)) — reaches **~100 TFLOP/s at N=16384, a ~19× improvement over the original `CudaWmma`, 83% of cuBLAS's dense-FP16 throughput**. The remaining ~20 TFLOP/s is now bounded by register pressure (126 registers/thread caps occupancy at 33%), then by the structural changes CUTLASS exists to handle generically: deeper multi-stage pipelining and split-K for very large K.

---
