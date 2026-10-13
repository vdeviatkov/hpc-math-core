# Benchmark Results

Measured throughput for every kernel family in this repository, on the three
machines it has been verified on. All numbers come from
`bench_gemm` / `bench_gemm_cuda` (Google Benchmark, console format) built in
Release mode as described in [build.md](build.md). CPU kernels are strictly
single-threaded unless noted (Apple AMX via Accelerate is the exception).

Contents

- [Deriving GFLOP/s](#deriving-gflops)
- [Apple M4 Max](#apple-m4-max) — scalar · NEON · SME2 · AMX · KleidiAI
- [AMD Zen 5](#amd-zen-5--avx2--avx-512-linux--gcc) — scalar · AVX2 · AVX-512
- [NVIDIA RTX 5080](#nvidia-rtx-5080--cuda) — CUDA Levels 0-7 · cuBLAS reference

Each machine section has the same shape: build metadata, GFLOP/s by kernel
and size, speedup against the naive baseline, then key observations.

---

## Deriving GFLOP/s

```
GFLOP/s = (2 × N³) / (time_µs × 1000)
```

A square N×N GEMM performs `2 × N³` floating-point operations; dividing by the time in nanoseconds gives GFLOP/s. Example: `Sme/f32/N=1024`, 1481 µs → `2 × 1024³ / (1481 × 1000)` ≈ 1,450 GFLOP/s.

In `bench_gemm`, N is a compile-time constant, as at a fixed-size call site. The kernels are header templates inlined into the benchmark, so the compiler specialises their loops for that N; with N passed at run time, `gemm_blocked` f32 at N=256 measured 28 instead of 80 GFLOP/s on M4 Max. The CUDA kernels are compiled separately and are unaffected.

---

## Apple M4 Max

> **Machine:** Apple M4 Max, 16 cores, Apple Clang 17, C++20
> **Build:** `cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON` → `-O3 -ffast-math -funroll-loops` (`-march` cleared in favour of `-mcpu=apple-m4`, see [§ SME and AMX build flags](build.md#sme-and-amx-build-flags))
> **CPU Caches (P-cores, which run the benchmarks):** L1d 128 KiB · L1i 192 KiB · L2 16 MiB shared per cluster (12 P-cores; the 4 E-cores have 64 KiB L1d and a 4 MiB L2 — the values Google Benchmark prints, since it reports the first core type)
> **Load Average:** 4.15 / 4.16 / 4.25 on 16 cores ≈ 26% — moderate background activity, numbers still representative

Five families run here: scalar, NEON, SME2 (opt-in), AMX via Accelerate,
and the KleidiAI reference (built when SME is on). AVX2, AVX-512 and SVE are
absent and report `SKIPPED`.

### GFLOP/s by kernel and size

double (f64)

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `gemm_naive` | 9.46 | 2.58 | 2.63 | 2.39 | 0.66 |
| `gemm_reordered` | 27.74 | 16.62 | 16.76 | 16.57 | 16.29 |
| `gemm_blocked` | 27.21 | 25.42 | 22.23 | 19.48 | 20.18 |
| `gemm_neon_naive` | 8.66 | 2.49 | 2.58 | 2.41 | 0.66 |
| `gemm_neon_reordered` | 27.12 | 13.67 | 14.32 | 15.05 | 15.52 |
| `gemm_neon_blocked` | **36.30** | 33.95 | 32.05 | 30.62 | **24.96** |

float (f32)

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

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter=Sme` (all four SME kernels; the `svl` counter reports the streaming vector length, 16 f32 / 8 f64 elements)

`gemm_sme` adds SME's outer products into the ZA tiles, with packed A and B,
GotoBLAS cache blocking and SME2 multi-vector loads; `gemm_sme_1tile`,
`gemm_sme_tiles` and `gemm_sme_blocked` are the steps before it. Design:
[src/gemm/sme.hpp](../src/gemm/sme.hpp).

> **Load Average during this run:** 10–16 on 16 cores. Treat ±5% as noise.

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_sme` f64 | 84 | 315 | 397 | **410** | 407 | 404 |
| `gemm_sme` f32 | 278 | 753 | 1,283 | **1,450** | 1,417 | 1,345 |

- f64 is ~¼ of f32, which matches the hardware: an f64 FMOPA covers 8×8 elements against 16×16 for f32.
- One compiler pitfall costs ~5×. A streaming helper without a ZA attribute (`__arm_inout("za")` etc.) is "private-ZA": Clang won't inline it into a ZA-owning caller and wraps every call in a lazy ZA save (`TPIDR2` + `smstart za`). Inside the k loop that dropped throughput from ~1,290 to ~270 G/s f32.

#### The four SME steps

Separate run, load average ~5, median of 3:

| Kernel | N=256 | N=512 | N=1024 | N=2048 | N=4000 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_sme_1tile` f32 | 420 | 472 | 474 | 225 | 125 | 131 |
| `gemm_sme_tiles` f32 | 976 | 1,461 | 1,381 | 624 | 429 | 445 |
| `gemm_sme_blocked` f32 | 911 | 1,400 | 1,401 | 1,249 | 1,373 | 597 |
| `gemm_sme` f32 | 752 | 1,283 | 1,421 | 1,438 | 1,419 | 1,372 |
| `gemm_sme_1tile` f64 | 121 | 121 | 110 | 40 | 42 | 37 |
| `gemm_sme_tiles` f64 | 350 | 420 | 403 | 232 | 210 | 208 |
| `gemm_sme_blocked` f64 | 342 | 422 | 442 | 284 | 373 | 271 |
| `gemm_sme` f64 | 315 | 397 | 414 | 406 | 410 | 403 |

- One tile → all tiles: 2.3–3.4× f32, 2.9–5.8× f64. With one tile every FMOPA waits for the previous one's accumulator, and at large N each loaded vector feeds a single FMOPA.
- All tiles → blocked: 2.0× f32 at N=2048, 3.2× at N=4000. Without blocking all of A is re-read for every B strip. N=4000 is as slow as N=4096, so this is not a stride effect; capacity is the likely cause.
- Blocked → `gemm_sme`: within 3% at N=4000, but 2.3× f32 / 1.5× f64 at N=4096 and 1.15× f32 at N=2048: the B row stride effect below.
- At N ≤ 512 the kernels without B packing are faster (+30% f32 at N=256 for `gemm_sme_tiles`): everything fits in cache, and `gemm_sme` pays for the copy.

#### SME: B row stride

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter=BStride --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true --benchmark_report_aggregates_only=true`

The same blocked kernel with B read in place (`gemm_sme_blocked`) and packed
(`gemm_sme`); GFLOP/s, median of 3 interleaved runs, load average ~5. Why a
power-of-two row stride is expected to hurt: [sme.hpp](../src/gemm/sme.hpp),
"Packing". First, N around powers of two (all matrices square, B row stride
= N·sizeof(T); the f64 strides are twice the f32 ones):

| N | B row stride (f32) | f32 in place | f32 packed | diff | f64 in place | f64 packed | diff |
|---|---|---|---|---|---|---|---|
| 1984 | 7,936 B | 1,565 | 1,517 | +3% | 407 | 418 | −3% |
| 2016 | 8,064 B | 1,556 | 1,541 | +1% | 410 | 419 | −2% |
| **2048** | **2¹³ B** | 1,196 | 1,432 | **−17%** | 281 | 407 | **−31%** |
| 2080 | 8,320 B | 1,493 | 1,483 | +1% | 398 | 404 | −2% |
| 4000 | 16,000 B | 1,365 | 1,406 | −3% | 380 | 409 | −7% |
| 4032 | 16,128 B | 1,354 | 1,399 | −3% | 380 | 408 | −7% |
| 4064 | 16,256 B | 1,303 | 1,399 | −7% | 364 | 411 | −11% |
| **4096** | **2¹⁴ B** | 608 | 1,344 | **−55%** | 268 | 398 | **−33%** |
| 4128 | 16,512 B | 1,260 | 1,304 | −3% | 362 | 395 | −8% |

Then the 4096×4096 problem with B in a wider buffer, so only its row stride
(ldb elements) changes and the work stays the same:

| ldb | B row stride (f32) | f32 in place | f32 packed | diff | f64 in place | f64 packed | diff |
|---|---|---|---|---|---|---|---|
| **4096** | **2¹⁴ B** | 608 | 1,344 | **−55%** | 268 | 398 | **−33%** |
| 4097 | 16,388 B | 1,145 | 1,365 | −16% | 367 | 407 | −10% |
| 4128 (+128 B) | 16,512 B | 1,339 | 1,386 | −3% | 376 | 407 | −8% |
| 4160 (+256 B) | 16,640 B | 1,332 | 1,390 | −4% | 376 | 406 | −7% |

- The collapse is confined to the exact powers of two; 32 columns either side, in place is within 11% of packed.
- Padding each row by one cache line (ldb = 4128) recovers most of the loss with the same N, work and pages touched, so the loss follows the byte stride. That is what cache-set conflicts predict. It does not reach packed strips: a repeat run (5 repetitions, interleaved) measured ldb = 4128 at −8% f32 / −6% f64 and ldb = 4160 at −7% / −9%, so padded B stays 3–9% behind packed B.
- ldb = 4097 is a control, not a fix: its 32-float row segments are no longer 128 B aligned, so each spans two cache lines, and the −16% mixes that cost with the stride effect.
- Where the conflict happens, and the cost of each fix: [experiments/sme-b-stride](experiments/sme-b-stride/README.md). With rows 16 KiB apart the macro-kernel matches a padded stride up to 384 rows per strip and breaks between 384 and 448, and the core's pointer chase at that stride holds ~512 lines in L2 against 8 in L1: the conflicts are in L2. With a padded stride, packing B costs ~2 ms and saves 11–14 ms of compute at N=4096; the remaining in-place cost grows with N (0% at 1024, 9–10% at 4096).

### Matrix engines, single core

> **Command:** `VECLIB_MAXIMUM_THREADS=1 ./build/benchmarks/bench_gemm --benchmark_filter='Sme|Amx|KleidiAI'`

Same machine and run as above. Every row uses one core, so this is the like-for-like comparison. Accelerate was limited to one thread with `VECLIB_MAXIMUM_THREADS=1`; KleidiAI is always single-threaded.

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_sme` f64 | 84 | 315 | 397 | 410 | 407 | 404 |
| Accelerate f64 | 326 | 460 | 421 | **434** | 394 | 396 |
| `gemm_sme` f32 | 278 | 753 | 1,283 | 1,450 | 1,417 | 1,345 |
| Accelerate f32 | 795 | 1,705 | **1,719** | 1,649 | 1,687 | 1,590 |
| KleidiAI v1.31 f32 | 327 | 1,060 | 1,511 | 1,643 | 1,408 | 1,023 |

- f64: `gemm_sme` is within ~6% of Accelerate from N=512 up, and ahead at N=2048–4096. Those gaps are close to the noise level of this run. Below N=512 Accelerate is far ahead (4× at N=64), because `gemm_sme` pays fixed per-call costs: allocating and packing A and B, plus a streaming-mode switch per block.
- f32: `gemm_sme` reaches 84–88% of Accelerate for N ≥ 1024. It ties KleidiAI at N=2048 and beats it at N=4096 (1,345 vs 1,023); KleidiAI uses the same 2×2-tile micro-kernel but no cache blocking, so it drops at large N.
- With all cores (default threading, wall-clock time): Accelerate 3,124 G/s f32 / 785 G/s f64 at N=4096. Roughly double the single-core figure, consistent with M4 Max's two performance clusters each having their own matrix unit.

### Apple AMX (via Accelerate.framework)

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter=Amx` (`HPC_ENABLE_AMX` defaults ON on Apple)

`gemm_amx` calls Accelerate's BLAS (`cblas_sgemm`/`cblas_dgemm`), Apple's route to its AMX coprocessor (unrelated to Intel AMX; see [src/gemm/amx.hpp](../src/gemm/amx.hpp)). Accelerate may use several cores, so these numbers are not single-core.

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_amx` f64 | 329.78 | 466.61 | 822.88 | **860.28** | 804.22 | 813.36 |
| `gemm_amx` f32 | 797.61 | 1,724 | 2,906 | **3,296** | 3,202 | 3,156 |

(Recorded when the family still had three entry points — naive/reordered/blocked — that all called the same `cblas_*gemm`; they measured within ~1% of each other, which is why they were merged into `gemm_amx`. The row above is the former `gemm_amx_blocked`.)

- Up to 3.3 TFLOP/s f32 and 860 GFLOP/s f64, the highest CPU throughput in this repo and ~2.3× / ~2.1× the single-threaded `gemm_sme` — but with every core available. Limited to one thread it reaches ~1.7 TFLOP/s f32 from N=256 up ([§ Matrix engines, single core](#matrix-engines-single-core)), so the climb from ~0.8 to ~3.3 TFLOP/s is about half problem size and half the second performance cluster.
- The f32/f64 ratio is ~3.8×, not the ~2× lane-count ratio seen elsewhere (NEON 2.7×, AVX-512 ~2×). It is close to the 4× of the M4's matrix unit, where an f64 outer product covers 8×8 elements against 16×16 for f32 (`gemm_sme` measures 3.5×) — consistent with Accelerate running on that unit, though Apple does not document it.
- For production code on a Mac, call Accelerate. The hand-written kernels show how close one can get to it, and what each step contributes.

### Prefetch distance sweep

Benchmarks named `<Family>BlockedPf<D>/<prec>/N=<size>` sweep prefetch distance D ∈ {2, 4, 8, 16} (rows ahead), with three `__builtin_prefetch` sites per kernel:

- [PF-A] `A(i + D×kRegRows, k_blk)` → L2 (read)
- [PF-B] `B(k_blk + TileK, 0)` → L2 (read), at the k-tile boundary
- [PF-C] `C(i + D×kRegRows, j_blk)` → L1 (write)

GFLOP/s, best distance per row in bold, against the same kernel without prefetch. Measured in a separate run from the tables above (2026-10-01, current build, load average ~4 — treat ±2% as noise):

| Kernel | N | D=2 | D=4 | D=8 | D=16 | no prefetch |
|---|---|---|---|---|---|---|
| `BlockedPf` f64 | 256 | 24.42 | 24.48 | 24.39 | **24.62** | 24.54 |
| `BlockedPf` f64 | 512 | 21.00 | **21.03** | **21.03** | 20.96 | 21.39 |
| `BlockedPf` f64 | 1024 | 18.51 | 18.50 | **18.54** | 18.53 | 18.55 |
| `BlockedPf` f32 | 256 | **80.67** | 80.51 | 80.57 | 80.59 | 80.69 |
| `BlockedPf` f32 | 512 | 46.60 | 46.65 | 46.75 | **46.80** | 49.38 |
| `BlockedPf` f32 | 1024 | 41.18 | 41.32 | **41.34** | 41.27 | 41.31 |
| `NeonBlockedPf` f64 | 256 | 31.54 | **33.06** | 31.50 | 32.90 | 31.92 |
| `NeonBlockedPf` f64 | 512 | 29.36 | 30.63 | 29.67 | **30.70** | 30.04 |
| `NeonBlockedPf` f64 | 1024 | 28.79 | 28.89 | 28.65 | **29.06** | 29.12 |
| `NeonBlockedPf` f32 | 256 | 93.67 | **96.03** | 94.17 | 93.90 | 94.31 |
| `NeonBlockedPf` f32 | 512 | 94.17 | **96.35** | 94.48 | 94.20 | 94.64 |
| `NeonBlockedPf` f32 | 1024 | 91.36 | **93.16** | 92.31 | 91.65 | 91.71 |

Software prefetch is close to neutral on M4. On the scalar kernel every distance is within ±1% of no prefetch, apart from one −5% outlier (f32, N=512); M4's hardware prefetcher already follows these streaming accesses. On the NEON kernel D=4 is consistently best, but the gain is small: +1.6–1.8% for f32 and up to +3.6% for f64 at N=256 — close to the noise level of this run.

### Speedup vs `gemm_naive`

f64

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive |
|---|---|---|---|---|---|---|---|
| 64 | 55.5 µs | 18.9 µs | 2.9× | 19.3 µs | 2.9× | 14.4 µs | 3.9× |
| 256 | 13,036 µs | 2,020 µs | 6.5× | 1,320 µs | 9.9× | 989 µs | 13.2× |
| 512 | 102,246 µs | 16,019 µs | 6.4× | 12,079 µs | 8.5× | 8,380 µs | 12.2× |
| 1024 | 924,366 µs | 129,596 µs | 7.1× | 110,250 µs | 8.4× | 70,187 µs | 13.2× |
| 4096 | 207.5 s | 8.44 s | 24.6× | 6.81 s | 30.5× | 5.51 s | 37.7× |

f32

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive |
|---|---|---|---|---|---|---|---|
| 64 | 56.1 µs | 6.09 µs | 9.2× | 6.12 µs | 9.2× | 5.41 µs | 10.4× |
| 256 | 12,278 µs | 1,039 µs | 11.8× | 402 µs | 30.5× | 348 µs | 35.3× |
| 512 | 109,000 µs | 8,157 µs | 13.4× | 5,238 µs | 20.8× | 2,776 µs | 39.3× |
| 1024 | 815,153 µs | 65,121 µs | 12.5× | 49,952 µs | 16.3× | 22,752 µs | 35.8× |
| 4096 | 218.4 s \* | 4.19 s | 52.1× | 4.36 s | 50.1× | 1.89 s | 115.4× |

\* Uses `Naive/f32/N=4096`'s CPU time, since its wall-clock was contended on this run (see the note under the f32 table above). The other columns' timings are unaffected.

### Key observations

Cache-access pattern dominates at large N. `gemm_naive` runs at nearly the same GFLOP/s for f32 and f64 at every size: it waits on cache misses from the column-stride walk of B, where element width doesn't matter. Switching to i-k-j (`gemm_reordered`) makes B and C sequential so every cache line is fully consumed: at N=4096 that is 25× faster than naive for f64 and 52× for f32 (twice the elements per cache line → twice the bandwidth).

Explicit SIMD beats auto-vectorisation mainly by staying flat. `gemm_reordered` and `gemm_blocked` carry no NEON intrinsics — the compiler vectorises the sequential inner j-loop under `-ffast-math`, reaching ~86 GFLOP/s f32 at small N. That figure relies on N being a compile-time constant ([Deriving GFLOP/s](#deriving-gflops)); with N passed at run time `gemm_blocked` f32 runs at 28 GFLOP/s at N=256. Adding an explicit 4×16 f32 register tile (4 rows × 4 Q registers) on top of L2 blocking changes the *shape* of the curve:

| Kernel | f32 N=256 | f32 N=512 | f32 N=1024 |
|---|---|---|---|
| `gemm_blocked` (auto-vectorised) | 83.5 | 51.3 | 43.0 |
| `gemm_neon_blocked` (explicit) | **96.4** | **96.7** | **94.4** |

The register tile holds ~95–97 GFLOP/s from N=64 through N=1024, while the auto-vectorised kernel decays 84→43 — most likely as C rows fall out of L1 between k-iterations (not profiled).

NEON f64 vs f32 is 2.7×, not the theoretical 2×. A Q-register holds 4 f32 lanes or 2 f64 lanes, so lane count alone predicts 2×. `gemm_neon_blocked` peaks at ~36 G/s f64 and ~97 G/s f32. A likely cause of the extra 0.7× (not profiled): the f32 micro-kernel keeps 16 accumulators (4 rows × 4 Q-registers), the f64 one only 8 (4 × 2), so f64 has half as many independent FMA chains to hide FMA latency.

Peak per family on this machine:

| Kernel | f64 peak | f32 peak | f32/f64 |
|---|---|---|---|
| `gemm_naive` | 9.46 | 9.35 | 1.0× |
| `gemm_reordered` | 27.74 | 86.10 | 3.1× |
| `gemm_blocked` | 27.21 | 85.85 | 3.2× |
| `gemm_neon_blocked` | 36.30 | 97.03 | 2.7× |
| `gemm_neon_blocked_prefetch` (D=4, prefetch-sweep run) | 33.06 | 96.35 | 2.9× |
| `gemm_sme` | 410 | **1,450** | 3.5× |
| `gemm_amx` (Accelerate) | **860.28** | **3,295.78** | 3.8× |

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
runs on native-width hardware. NEON, SVE, SME and AMX are absent and report
`SKIPPED`; the scalar, AVX2 and AVX-512 tests all pass.

### GFLOP/s by kernel and size

double (f64)

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `gemm_naive` | 14.21 | 2.05 | 1.27 | 0.73 | 0.43 |
| `gemm_reordered` | 42.21 | 52.34 | 36.59 | 32.89 | 10.93 |
| `gemm_blocked` | 41.92 | 35.61 | 34.58 | 33.40 | **31.82** |
| `gemm_avx2_blocked` | 75.35 | 70.99 | 46.14 | 41.58 | 37.27 |
| `gemm_avx512_blocked` | **106.03** | 105.78 | 57.34 | 54.44 | **50.13** |

float (f32)

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
| `gemm_reordered` | 12.57 s | 25.3× | 5.16 s | 64.7× |
| `gemm_blocked` | 4.32 s | 73.6× | 3.23 s | 103.3× |
| `gemm_avx2_blocked` | 3.69 s | 86.2× | 1.80 s | 185.1× |
| `gemm_avx512_blocked` | 2.74 s | 116.0× | 1.14 s | 291.7× |

### Key observations

The scalar kernels depend heavily on the compiler's auto-vectoriser.
`gemm_reordered` and `gemm_blocked` contain no intrinsics at all — their
throughput is whatever the compiler makes of the inner j-loop. Under GCC with
`-march=native -ffast-math`, `gemm_reordered` reaches 93.15 GFLOP/s f32 at
N=256, within 2.4× of the hand-written AVX-512 kernel. Where the
scalar kernels land is therefore as much about the compiler as the algorithm.

Blocking earns its keep only at large N. At N≤512 `gemm_blocked` looks
like a regression against `gemm_reordered` (47.51 vs 93.15 GFLOP/s f32 at
N=256) — the untiled kernel still fits cache there, and tiling the j-loop to
64 columns costs the vectoriser more than the cache saves. The ranking
inverts where it matters: at N=4096, blocked reaches 42.53 vs 26.65 G/s
(f32, 1.6×) and 31.82 vs 10.93 G/s (f64, 2.9×). Small sizes alone would
suggest the opposite.

The AVX-512 blocked kernel loses roughly half its throughput past a
size threshold, and the f64 case matches L2 capacity exactly. f64 drops
105.78 → 57.34 G/s between N=256 and N=512; f32 holds until N=1024, then
drops 212.41 → 143.07. For f64 the arithmetic is exact: the B panel is
`kAvx512TileK (256) × kAvx512TileN (512) × 8 B` = precisely 1 MiB, which
is this CPU's per-core L2, while at N=256 the j-tile is clamped to 256 and
the panel is half that. The f32 panel is 512 KiB at every size ≥512, so the
same arithmetic does *not* explain its drop at N=1024 — something else
(likely L3 pressure or DRAM streaming as the full matrices grow) dominates
there. The f64 cliff looks directly addressable by shrinking
`kAvx512TileN` for 8-byte elements; the f32 case needs profiling before
any claim is made. `gemm_avx2_blocked` shows the same f64 shape
(70.99 → 46.14) at the same size, consistent with its own 256×256×8 =
512 KiB panel plus A and C traffic crowding the same L2.

Software prefetch has almost no effect on Zen 5, and the distance is
irrelevant. Across every family, precision and size, all four distances
(D ∈ {2, 4, 8, 16}) land within ~1% of each other:

| Kernel | D=2 | D=4 | D=8 | D=16 | no prefetch |
|---|---|---|---|---|---|
| `Avx512BlockedPf` f32 N=1024 | 149.46 | 149.75 | 149.12 | 149.92 | 143.07 |
| `Avx512BlockedPf` f64 N=1024 | 55.35 | 54.90 | 55.32 | 55.12 | 54.44 |
| `Avx2BlockedPf` f32 N=256 | 149.98 | 149.69 | 149.73 | 150.00 | 150.55 |
| `BlockedPf` f32 N=512 | 46.63 | 47.53 | 46.41 | 47.50 | 47.22 |

The largest effect anywhere is +4.5% (`Avx512BlockedPf` f32 at N=1024); most
rows are within noise of the unprefetched kernel, and some are marginally
slower. Apple M4 Max shows the same picture (within ±1% for the scalar
kernel, at most a few percent for NEON at D=4): both hardware prefetchers
already follow these streaming access patterns, so explicit hints add little.

`gemm_naive` is slower than on Apple M4 Max at N=4096 (0.43 vs 0.66 GFLOP/s
f64) despite the higher clock, and f32 and f64 are indistinguishable (0.41 vs
0.43) — it is bound by DRAM latency, where element width doesn't matter. One
naive iteration at N=4096 takes 318 s; the six naive-family entries at that
size take 1,945 s of the 2,165 s sweep (90%).

---

## NVIDIA RTX 5080 — CUDA

> **Machine:** AMD Ryzen 9 9950X host + **NVIDIA GeForce RTX 5080** (Blackwell, sm_120), Ubuntu 24.04, CUDA 13.2, GCC 13.3, C++20
> **Build:** `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc` → native `sm_120` SASS (verified with `cuobjdump --list-elf`)
> **Run:** `./build/benchmarks/bench_gemm_cuda` — 123 s for the full GPU sweep
> **Date:** 2026-09-24. All CUDA tests passed with no skips.

What each level adds: [src/gemm/cuda.hpp](../src/gemm/cuda.hpp). Rows
without a suffix are end to end (`cudaMalloc`, host-to-device copies, kernel
and copy back on every iteration); `*ComputeOnly` rows time only the GEMM
call on device-resident buffers. `CudaWmma`, `CudaMmaLdmatrix` and
`CudaWmmaPipelined` multiply in fp16 (counter `precision=16`), so their
GFLOP/s is not like-for-like with the fp32 rows.

### GFLOP/s by kernel and size

float (f32)

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 | N=8192 | N=16384 |
|---|---|---|---|---|---|---|---|
| `CudaNaive` | 9 | 252 | 660 | 1,485 | 2,608 | — | — |
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

`CudaWmma` was measured before its edge-tile store fix (commit `13add97`).
In interleaved before/after runs it reaches ~6.0 TFLOP/s at N=4096 since
(5,988–6,025 vs 5,279–5,307 GFLOP/s); N=1024 is unchanged.

double (f64). Consumer Blackwell has a heavily reduced FP64 datapath, so
everything is an order of magnitude below f32, and the Tensor Core kernels
do not apply.

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `CudaNaive` | 7 | 145 | 323 | 539 | 750 |
| `CudaBlocked` | 7 | 146 | 328 | 552 | **770** |
| `CudaRegTile` | 2 | 34 | 127 | 455 | 705 |
| `CudaDoubleBuf` | 3 | 56 | 201 | 441 | 712 |
| `CudaVectorized` | 2 | 34 | 128 | 460 | 725 |
| `CudaCublas` (ref) | 11 | 178 | 284 | 467 | 686 |

### Level by level, f32, N=4096 end to end

| Level | Kernel | GFLOP/s | ×Naive | Note |
|---|---|---|---|---|
| 0 | `CudaNaive` | 2,608 | 1.0× | Coalesced B loads and L1/L2 reuse make this baseline strong. |
| 1 | `CudaBlocked` | 2,422 | 0.93× | One output per thread: two `__syncthreads()` per k-tile for 16 FMAs. In f64 it is the fastest kernel from N=512, every kernel there being bound by the FP64 pipe (not profiled). |
| 2 | `CudaRegTile` | 6,583 | 2.52× | 64 outputs per thread; the largest single step. |
| 3 | `CudaDoubleBuf` | 6,671 | 2.56× | The fastest FMA kernel. |
| 4 | `CudaWmma` | 5,266 | 2.02× | Below Levels 2–3 at this size: 64×64 tiles, one fragment per warp and a single buffer leave too little work per synchronisation. |
| 5 | `CudaVectorized` | 5,690 | 2.18× | Slower than Level 2. Not profiled; the swizzle's index arithmetic on every shared-memory read is the likely cost, while A's transposed scatter-store stays scalar. |
| 6 | `CudaMmaLdmatrix` | 5,685 | 2.18× | 8% above Level 4 with the same tile sizes and single buffering: dropping to PTX alone does not remove the bottleneck. |
| 7 | `CudaWmmaPipelined` | 9,000 | 3.45× | Ahead of both end-to-end cuBLAS rows (8,285 / 8,735) only because every kernel is transfer-bound at this size; compute-only, cuBLAS FP16 leads. |

### Compute-only, against cuBLAS

TFLOP/s, ascending:

| Path | Precision | N=4096 | N=8192 | N=16384 |
|---|---|---|---|---|
| `cublasSgemm` | FP32 on the SIMT cores | 37.7 | 38.6 | 39.1 |
| `cublasGemmEx` TF32 | TF32 Tensor Cores, 10-bit mantissa | 51.8 | 58.4 | 59.7 |
| **`gemm_cuda_wmma_pipelined`** | **FP16 Tensor Cores, hand-written** | **97.2** | **101.5** | **100.5** |
| `cublasGemmEx` FP16 | FP16 Tensor Cores, fp32 accumulate | 108.7 | 117.0 | **120.5** |

- 100+ TFLOP/s on this GPU needs dense FP16 Tensor Cores: plain FP32, the ceiling for every non-Tensor-Core kernel, tops out at ~39 TFLOP/s, and TF32 at ~60.
- `gemm_cuda_wmma_pipelined` reaches 83% of cuBLAS FP16 at N=16384, about 19× `gemm_cuda_wmma` (~5 TFLOP/s, end to end at N=4096; not measured compute-only).

### Level 7: shared-memory bank conflicts

Without padding its shared-memory tiles, `gemm_cuda_wmma_pipelined` measured
80.8 TFLOP/s, 68% of cuBLAS. Padding each row by 8 halves (why 8, and why
not a swizzle: the `kPipeAsLd` comment in
[gemm_kernels.cu](../src/cuda/gemm_kernels.cu)) changed, per Nsight Compute
at N=4096 compute-only:

| Metric | Unpadded | Padded (+8) |
|---|---|---|
| Shared-load bank conflicts | 285,879,068 | **519,545** |
| …as a share of load wavefronts | 85% | **1.0%** |
| Shared-load wavefronts | 336,210,716 | **50,851,193** |
| L1/TEX throughput | 87.9% | **31.8%** |
| Tensor pipe utilisation | 68.0% | **86.1%** |
| Warp stalls — MIO throttle | 26.3% | **3.9%** |
| Warp stalls — short scoreboard | 13.6% | **3.2%** |
| Registers per thread | 126 | 126 |
| **Throughput** | 80.8 TFLOP/s | 100.5 TFLOP/s |

The kernel had been moving 6.7× more shared-memory traffic than it needs;
afterwards it is within 1% of the minimum wavefront count. Shared memory
per block grows from 32 to 37 KB, lowering the shared-memory occupancy
limit from 3 to 2 blocks per SM, which costs nothing: 126 registers per
thread already cap it at 2. That register pressure (occupancy 33%, ~67%
headroom per Nsight) is what limits the kernel now; beyond it, the last
17% to cuBLAS would take a deeper pipeline (3–4 stages) and split-K.

### Linux vs Windows on identical hardware

An earlier run used the same GPU and CUDA 13.2 under Windows/MSVC. f32, with
the version of `gemm_cuda_wmma_pipelined` from before the padding above, so
both columns run the same source:

| Kernel | Windows | Linux | Δ |
|---|---|---|---|
| `CudaNaive` (end to end, N=4096) | 2,481 | 2,608 | +5% |
| `CudaRegTile` (end to end, N=4096) | 5,728 | 6,583 | +15% |
| `CudaDoubleBuf` (end to end, N=4096) | 5,744 | 6,671 | +16% |
| `CudaWmmaPipelined` (end to end, N=4096) | 7,121 | 9,003 | **+26%** |
| `CudaCublasFp16ComputeOnly` (N=16384) | 117,827 | 118,228 | +0.3% |
| `CudaWmmaPipelinedComputeOnly` (N=16384) | 82,383 | 80,773 | −2% |

End to end gains 5–26%, compute-only is flat within ±2%: the difference is
the driver and transfer path, where Linux avoids Windows' WDDM overhead on
allocation and host↔device copies. Any benchmark here that includes
transfers measures the OS as much as the GPU.

---
