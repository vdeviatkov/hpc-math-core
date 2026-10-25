# Benchmark Results

Measured throughput for every kernel family in this repository, on the three
machines it has been verified on. All numbers come from
`bench_gemm` / `bench_gemm_cuda` / `bench_gemm_metal` and, for low
precision, `bench_gemm_lowp` / `bench_gemm_lowp_cuda` (Google Benchmark)
built in Release mode as described in [build.md](build.md). CPU kernels
are strictly single-threaded unless noted (Apple AMX via Accelerate and
`gemm_sme_mt` are the exceptions).

Contents

- [Deriving GFLOP/s](#deriving-gflops)
- [Apple M4 Max](#apple-m4-max) — scalar · NEON · SME2 (one and several units) · AMX · KleidiAI
- [Apple M4 Max GPU — Metal](#apple-m4-max-gpu--metal) — Metal Levels 0-5 · MPS reference
- [AMD Zen 5](#amd-zen-5--avx2--avx-512-linux--gcc) — scalar · AVX2 · AVX-512
- [NVIDIA RTX 5080](#nvidia-rtx-5080--cuda) — CUDA Levels 0-8 · cuBLAS reference
- [Low precision](#low-precision-bf16-fp16-int8-fp8) — bf16 · fp16 · INT8 · FP8 on all three machines

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
> **Load Average:** 5–7 on 16 cores during the scalar and NEON runs (2026-10-15), including two unrelated processes at 100% each — treat ±5% as noise

Five families run here: scalar, NEON, SME2 (opt-in), AMX via Accelerate,
and the KleidiAI reference (built when SME is on). AVX2, AVX-512 and SVE are
absent and report `SKIPPED`.

### GFLOP/s by kernel and size

double (f64)

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `gemm_naive` | 8.69 | 2.40 | 2.12 | 2.03 | 0.62 |
| `gemm_reordered` | 25.62 | 15.90 | 15.84 | 15.98 | 15.62 |
| `gemm_blocked` | 25.79 | 24.59 | 21.25 | 18.39 | 19.45 |
| `gemm_simd_reordered<Q>` | 25.64 | 13.40 | 14.06 | 7.66 | 14.98 |
| `gemm_simd_blocked<Q>` | 62.13 | 62.25 | 60.99 | 59.52 | 51.03 |

float (f32)

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `gemm_naive` | 8.91 | 2.48 | 2.29 | 2.06 | 0.59 |
| `gemm_reordered` | 82.63 | 30.96 | 31.64 | 31.56 | 30.73 |
| `gemm_blocked` | 82.97 | 80.47 | 48.54 | 41.05 | 28.85 |
| `gemm_simd_reordered<Q>` | 82.74 | 28.71 | 26.60 | 27.48 | 29.42 |
| `gemm_simd_blocked<Q>` | 122.41 | 122.86 | 123.78 | 121.19 | 102.10 |

`gemm_simd_blocked<Q>` uses a 6×16 f32 / 6×8 f64 register tile (6 rows × 4
vectors, 24 accumulators) and loads A with `LDR` instead of `LD1R` since
2026-10-21, and a 384×256×256 cache block since 2026-10-22
([kernels/blocks.md](kernels/blocks.md)). Its rows are the median of three
runs with both
([`bench_m4_defaults.csv`](experiments/block-sweeps/bench_m4_defaults.csv)),
at a core clock of about 4.49 GHz; the other rows are from 2026-10-15, when the
clock was 3.96–4.13 GHz, so compare across them with ~10% allowance. In the
2026-10-15 run the previous kernel (4×16 f32, 8×4 f64, `LD1R`) measured
93.51 / 94.39 / 94.96 / 91.20 / 69.81 f32 and 37.27 / 39.64 / 39.18 / 37.90
/ 33.10 f64; at the same clock as the new one, the tile sweep has it at
115.2 f32 and 41.5 f64 for N=256. With the old 64-row block the 6×4
kernel measured 121.88 / 121.78 / 122.74 / 120.98 / 91.91 f32 and 61.35 /
61.29 / 60.32 / 58.25 / 45.80 f64. Every tile shape and why 6 × 4:
[kernels/neon.md](kernels/neon.md).

### ARM SME2 (`-DHPC_ENABLE_SME=ON`)

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter=Sme` (all four SME kernels; the `svl` counter reports the streaming vector length, 16 f32 / 8 f64 elements)

`gemm_sme` adds SME's outer products into the ZA tiles, with packed A and B,
GotoBLAS cache blocking and SME2 multi-vector loads; `gemm_sme_1tile`,
`gemm_sme_tiles` and `gemm_sme_blocked` are the steps before it. Design:
[src/gemm/sme.hpp](../src/gemm/sme.hpp).

> **Load Average during this run:** 5–10 on 16 cores. Median of 5 interleaved repetitions; treat ±5% as noise.

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_sme` f64 | 83 | 323 | 398 | 408 | 413 | **424** |
| `gemm_sme` f32 | 267 | 839 | 1,332 | 1,385 | 1,465 | **1,504** |

- f64 is ~¼ of f32, which matches the hardware: an f64 FMOPA covers 8×8 elements against 16×16 for f32.
- One compiler pitfall costs ~5×. A streaming helper without a ZA attribute (`__arm_inout("za")` etc.) is "private-ZA": Clang won't inline it into a ZA-owning caller and wraps every call in a lazy ZA save (`TPIDR2` + `smstart za`). Inside the k loop that dropped throughput from ~1,290 to ~270 G/s f32.

#### The four SME steps

Block sizes Mc = 1024 (f32) / 512 (f64), Kc = 2048, Nc = 4096 (the current
ones; see `sme.hpp`). Separate run, load average ~6–7, median of 3:

| Kernel | N=256 | N=512 | N=1024 | N=2048 | N=4000 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_sme_1tile` f32 | 420 | 472 | 464 | 220 | 125 | 131 |
| `gemm_sme_tiles` f32 | 975 | 1,447 | 1,319 | 617 | 427 | 446 |
| `gemm_sme_blocked` f32 | 989 | 1,460 | 1,333 | 633 | 1,468 | 489 |
| `gemm_sme` f32 | 835 | 1,322 | 1,365 | 1,452 | 1,532 | 1,519 |
| `gemm_sme_1tile` f64 | 121 | 121 | 105 | 40 | 42 | 38 |
| `gemm_sme_tiles` f64 | 352 | 418 | 401 | 227 | 204 | 205 |
| `gemm_sme_blocked` f64 | 351 | 420 | 423 | 243 | 422 | 225 |
| `gemm_sme` f64 | 322 | 396 | 404 | 413 | 436 | 421 |

- One tile → all tiles: 2.3–3.4× f32, 2.9–5.7× f64. With one tile every FMOPA waits for the previous one's accumulator, and at large N each loaded vector feeds a single FMOPA.
- All tiles → blocked: 3.4× f32 / 2.1× f64 at N=4000. Without blocking all of A is re-read for every B strip. N=4000 is as slow as N=4096 without blocking, so this is not a stride effect; capacity is the likely cause.
- Blocked → `gemm_sme`: +4% at N=4000, but 2.3× f32 at N=2048 and 3.1× at N=4096 (1.7× / 1.9× f64): the B row stride effect below. With 2,048-row strips an 8 KiB stride (N=2048) hurts as much as a 16 KiB one.
- At N ≤ 512 the kernels without B packing are faster (+18% f32 at N=256): everything fits in cache, and `gemm_sme` pays for the copy.
- Against the previous block sizes (Mc = 128, Kc = 1024), measured interleaved: `gemm_sme` f32 +4% at N=2048, +10% at N=4000, +12% at N=4096, +3% at N=512, unchanged within noise at N=1024; f64 +2–7% at N ≥ 2048. All tables in this section use the current sizes.

#### SME: B row stride

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter=BStride --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true --benchmark_report_aggregates_only=true`

The same blocked kernel with B read in place (`gemm_sme_blocked`) and packed
(`gemm_sme`), current block sizes (2,048-row B strips); GFLOP/s, median of 3
interleaved runs, load average 5–10. Why a power-of-two row stride is
expected to hurt: [sme.hpp](../src/gemm/sme.hpp), "Packing". First, N around
powers of two (all matrices square, B row stride = N·sizeof(T); the f64
strides are twice the f32 ones):

| N | B row stride (f32) | f32 in place | f32 packed | diff | f64 in place | f64 packed | diff |
|---|---|---|---|---|---|---|---|
| 1984 | 7,936 B | 1,570 | 1,561 | +1% | 427 | 432 | −1% |
| 2016 | 8,064 B | 1,572 | 1,555 | +1% | 421 | 425 | −1% |
| **2048** | **2¹³ B** | 636 | 1,485 | **−57%** | 242 | 416 | **−42%** |
| 2080 | 8,320 B | 1,462 | 1,440 | +2% | 404 | 414 | −2% |
| 4000 | 16,000 B | 1,494 | 1,515 | −1% | 423 | 438 | −3% |
| 4032 | 16,128 B | 1,489 | 1,529 | −3% | 320 | 439 | −27% |
| 4064 | 16,256 B | 1,476 | 1,525 | −3% | 235 | 435 | −46% |
| **4096** | **2¹⁴ B** | 480 | 1,506 | **−68%** | 223 | 428 | **−48%** |
| 4128 | 16,512 B | 1,356 | 1,433 | −5% | 226 | 419 | −46% |

Then the 4096×4096 problem with B in a wider buffer, so only its row stride
(ldb elements) changes and the work stays the same:

| ldb | B row stride (f32) | f32 in place | f32 packed | diff | f64 in place | f64 packed | diff |
|---|---|---|---|---|---|---|---|
| **4096** | **2¹⁴ B** | 480 | 1,506 | **−68%** | 223 | 428 | **−48%** |
| 4097 | 16,388 B | 1,283 | 1,527 | −16% | 302 | 432 | −30% |
| 4128 (+128 B) | 16,512 B | 1,465 | 1,553 | −6% | 232 | 429 | −46% |
| 4160 (+256 B) | 16,640 B | 1,498 | 1,532 | −2% | 319 | 434 | −26% |

- f32: the collapse is confined to the exact powers of two; 32 columns either side, in place is within 5% of packed. Padding each row by one cache line (ldb = 4128) recovers 3.1× with the same N and work, and only about 0.8% more storage, so the loss follows the byte stride. That is what cache-set conflicts predict.
- f64: in place now loses 26–48% at almost every N around 4096, padded or not; only N = 4000 escapes. Its strips are 2,048 rows of 256 B, twice the f32 footprint, at strides twice as large; with the previous 1,024-row strips padded f64 lost only 7–8%. Why padding no longer helps f64 was not investigated further. `gemm_sme` packs B, so none of this reaches it.
- ldb = 4097 is a control, not a fix: its row segments are no longer 128 B aligned, so each spans an extra cache line, and the loss mixes that cost with the stride effect.
- Where the conflict happens, and the cost of each fix (f32): [experiments/sme-b-stride](experiments/sme-b-stride/README.md). With rows 16 KiB apart the macro-kernel loses 5–26% up to 576 rows per strip and breaks between 576 and 640, and the core's pointer chase at that stride holds roughly 450–600 lines in L2 against 8 in L1: the conflicts are in L2. With a padded stride, packing B costs ~2.5 ms and saves 7–8 ms of compute at N=4096; the remaining in-place cost is 4–8% across N = 1024–4224.

### Matrix engines, single core

> **Command:** `VECLIB_MAXIMUM_THREADS=1 ./build/benchmarks/bench_gemm --benchmark_filter='Sme|Amx|KleidiAI'`

Same machine and run as above. Every row uses one core, so this is the like-for-like comparison. Accelerate was limited to one thread with `VECLIB_MAXIMUM_THREADS=1`; KleidiAI is always single-threaded.

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_sme` f64 | 83 | 323 | 398 | 408 | 413 | 424 |
| Accelerate f64 | 322 | **456** | 421 | 338 | 404 | 400 |
| `gemm_sme` f32 | 267 | 839 | 1,332 | 1,385 | 1,465 | 1,504 |
| Accelerate f32 | 777 | 1,692 | **1,718** | 1,629 | 1,681 | 1,599 |
| KleidiAI v1.31 f32 | 327 | 1,055 | 1,511 | 1,659 | 1,454 | 1,019 |

- f64: `gemm_sme` is within ~6% of Accelerate at N=512 and ahead from N=1024 up. Those gaps are close to the noise level of this run; Accelerate's dip at N=1024 appeared in two runs. Below N=512 Accelerate is far ahead (4× at N=64), because `gemm_sme` pays fixed per-call costs: allocating and packing A and B, plus a streaming-mode switch per block.
- f32: `gemm_sme` reaches 85–94% of Accelerate for N ≥ 1024. It ties KleidiAI at N=2048 and beats it at N=4096 (1,504 vs 1,019); KleidiAI uses the same 2×2-tile micro-kernel but no cache blocking, so it drops at large N.
- With all cores (default threading, wall-clock time): Accelerate 3,124 G/s f32 / 785 G/s f64 at N=4096 in this run (3,115 / 787 in the run part 2 quotes). Roughly double the single-core figure, consistent with M4 Max's two performance clusters each having their own matrix unit.

### Apple AMX (via Accelerate.framework)

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter=Amx` (`HPC_ENABLE_AMX` defaults ON on Apple)

`gemm_amx` calls Accelerate's BLAS (`cblas_sgemm`/`cblas_dgemm`), Apple's route to its AMX coprocessor (unrelated to Intel AMX; see [src/gemm/amx.hpp](../src/gemm/amx.hpp)). Accelerate may use several cores, so these numbers are not single-core.

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_amx` f64 | 329.78 | 466.61 | 822.88 | **860.28** | 804.22 | 813.36 |
| `gemm_amx` f32 | 797.61 | 1,724 | 2,906 | **3,296** | 3,202 | 3,156 |

(Recorded when the family still had three entry points — naive/reordered/blocked — that all called the same `cblas_*gemm`; they measured within ~1% of each other, which is why they were merged into `gemm_amx`. The row above is the former `gemm_amx_blocked`.)

- Up to 3.3 TFLOP/s f32 and 860 GFLOP/s f64 in this run (best 3,178 / 804 in the later run that the README and part 2 quote), the highest CPU throughput in this repo and ~2.3× / ~2.1× the single-threaded `gemm_sme` — but with every core available. Limited to one thread it reaches ~1.7 TFLOP/s f32 from N=256 up ([§ Matrix engines, single core](#matrix-engines-single-core)), so the climb from ~0.8 to ~3.3 TFLOP/s is about half problem size and half the second performance cluster.
- The f32/f64 ratio is ~3.8×, not the ~2× lane-count ratio seen elsewhere (NEON 2.0×, AVX-512 ~2×). It is close to the 4× of the M4's matrix unit, where an f64 outer product covers 8×8 elements against 16×16 for f32 (`gemm_sme` measures 3.5×) — consistent with Accelerate running on that unit, though Apple does not document it.
- For production code on a Mac, call Accelerate. The hand-written kernels show how close one can get to it, and what each step contributes.

### Several SME units (`gemm_sme_mt`)

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter='^(SmeMt|Sme|Amx)/' --benchmark_repetitions=5 --benchmark_enable_random_interleaving=true` (wall-clock time for `SmeMt` and `Amx`)
> **Date:** 2026-10-24, load average 2.6–6.2. Median of 5 interleaved repetitions. Data: [experiments/sme-mt](experiments/sme-mt/README.md)

`gemm_sme` is one thread, so one SME unit; the M4 Max has one per P-cluster,
two in all. `gemm_sme_mt` runs `gemm_sme`'s blocking on 8 threads: the B
panel is packed once and shared, and threads take chunks of A rows from an
atomic counter, so the work balances whichever cluster each thread lands
on (macOS cannot pin threads). Design and the thread sweep:
[kernels/sme.md § Several SME units](kernels/sme.md#several-sme-units).

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4000 | N=4096 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `gemm_sme` f32, 1 thread | 276 | 849 | 1,350 | 1,419 | 1,514 | 1,551 | 1,541 |
| `gemm_sme_mt` f32, 8 threads | 26 | 370 | 1,480 | 2,670 | 3,071 | **3,498** | **3,496** |
| Accelerate f32, all cores | 785 | 1,711 | 2,639 | **3,211** | 3,195 | — | 3,160 |
| `gemm_sme` f64, 1 thread | 86 | 326 | 402 | 421 | 424 | 440 | 429 |
| `gemm_sme_mt` f64, 8 threads | 14 | 270 | 618 | 758 | 894 | **915** | **914** |
| Accelerate f64, all cores | 325 | 462 | 797 | 827 | 785 | — | 783 |

- From N=2048 up, `gemm_sme_mt` matches or beats Accelerate with all cores:
  at N=4096, 111% in f32 and 117% in f64. Against the two units' measured
  peak (3.93 / 1.00 TFLOP/s) that is 89% / 91%, against 79% / 86% for
  `gemm_sme` on one unit.
- 2.3× `gemm_sme` in f32 and 2.1× in f64 at N=4096, a little more than two
  units. One candidate is packing, which `gemm_sme` does on its one core and
  `gemm_sme_mt` spreads over eight; it was not measured separately.
- At N=1024 it reaches 83% / 92% of Accelerate. Below N≈512 creating 8
  threads per call costs more than it saves (26 GFLOP/s at N=64), so small
  products should call `gemm_sme`. A first version with equal static row
  bands, and the evidence that macOS keeps a few threads on one cluster, are
  in [kernels/sme.md](kernels/sme.md#several-sme-units).

### Prefetch distance sweep

Benchmarks named `<Family>BlockedPf<D>/<prec>/N=<size>` sweep prefetch distance D ∈ {2, 4, 8, 16} (rows ahead), with three `__builtin_prefetch` sites per kernel:

- [PF-A] `A(i + D×kRegRows, k_blk)` → L2 (read)
- [PF-B] `B(k_blk + TileK, 0)` → L2 (read), at the k-tile boundary
- [PF-C] `C(i + D×kRegRows, j_blk)` → L1 (write)

GFLOP/s against the same kernel without prefetch, same run as the tables above (the `NeonBlockedPf` rows from the run with the current tile and block, [`bench_m4_defaults.csv`](experiments/block-sweeps/bench_m4_defaults.csv)):

| Kernel | N | D=2 | D=4 | D=8 | D=16 | no prefetch |
|---|---|---|---|---|---|---|
| `BlockedPf` f64 | 256 | 24.54 | 24.63 | 24.66 | 24.49 | 24.59 |
| `BlockedPf` f64 | 512 | 21.22 | 21.64 | 21.63 | 21.37 | 21.25 |
| `BlockedPf` f64 | 1024 | 18.43 | 18.59 | 18.58 | 18.56 | 18.39 |
| `BlockedPf` f32 | 256 | 80.83 | 80.88 | 81.08 | 80.14 | 80.47 |
| `BlockedPf` f32 | 512 | 48.54 | 48.88 | 49.15 | 48.26 | 48.54 |
| `BlockedPf` f32 | 1024 | 41.49 | 41.45 | 41.47 | 40.41 | 41.05 |
| `NeonBlockedPf` f64 | 256 | 61.39 | 61.37 | 61.22 | 61.18 | 62.25 |
| `NeonBlockedPf` f64 | 512 | 60.36 | 59.17 | 60.70 | 60.59 | 60.99 |
| `NeonBlockedPf` f64 | 1024 | 58.77 | 59.11 | 59.18 | 59.12 | 59.52 |
| `NeonBlockedPf` f32 | 256 | 122.61 | 122.27 | 121.98 | 121.95 | 122.86 |
| `NeonBlockedPf` f32 | 512 | 123.23 | 122.94 | 122.47 | 122.12 | 123.78 |
| `NeonBlockedPf` f32 | 1024 | 121.35 | 121.45 | 121.36 | 118.73 | 121.19 |

Software prefetch is neutral on M4: every distance is within ±2% of no prefetch for both kernels (the NEON rows: −3.0% to +0.2%). M4's hardware prefetcher already follows these streaming accesses.

### Speedup vs `gemm_naive`

f64

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive |
|---|---|---|---|---|---|---|---|
| 64 | 60.4 µs | 20.5 µs | 3.0× | 20.3 µs | 3.0× | 8.4 µs | 7.2× |
| 256 | 13,990 µs | 2,111 µs | 6.6× | 1,365 µs | 10.3× | 539 µs | 26.0× |
| 512 | 126,777 µs | 16,952 µs | 7.5× | 12,636 µs | 10.0× | 4,403 µs | 28.8× |
| 1024 | 1.07 s | 134,421 µs | 8.0× | 116,778 µs | 9.2× | 36,096 µs | 29.6× |
| 4096 | 222.38 s | 8.80 s | 25.3× | 7.07 s | 31.5× | 2.69 s | 82.5× |

f32

| N | Naive | Reordered | ×naive | Blocked | ×naive | NeonBlocked | ×naive |
|---|---|---|---|---|---|---|---|
| 64 | 58.9 µs | 6.3 µs | 9.3× | 6.3 µs | 9.3× | 4.3 µs | 13.8× |
| 256 | 13,526 µs | 1,085 µs | 12.5× | 417 µs | 32.4× | 273 µs | 49.5× |
| 512 | 117,710 µs | 8,496 µs | 13.9× | 5,531 µs | 21.3× | 2,170 µs | 54.3× |
| 1024 | 1.06 s | 68,128 µs | 15.5× | 52,317 µs | 20.2× | 17,725 µs | 59.8× |
| 4096 | 232.33 s | 4.47 s | 51.9× | 4.77 s | 48.7× | 1.35 s | 172.5× |

### Key observations

Cache-access pattern dominates at large N. `gemm_naive` runs at nearly the same GFLOP/s for f32 and f64 at every size: it waits on cache misses from the column-stride walk of B, where element width doesn't matter. Switching to i-k-j (`gemm_reordered`) makes B and C sequential so every cache line is fully consumed: at N=4096 that is 25× faster than naive for f64 and 52× for f32 (twice the elements per cache line → twice the bandwidth).

Explicit SIMD beats auto-vectorisation mainly by staying flat. `gemm_reordered` and `gemm_blocked` carry no NEON intrinsics — the compiler vectorises the sequential inner j-loop under `-ffast-math`, reaching ~83 GFLOP/s f32 at small N. That figure relies on N being a compile-time constant ([Deriving GFLOP/s](#deriving-gflops)). Adding an explicit 6×16 f32 register tile on top of L2 blocking changes the *shape* of the curve: `gemm_simd_blocked<Q>` holds 121–124 GFLOP/s f32 from N=64 through N=1024 (at 4.49 GHz, see above), while the auto-vectorised `gemm_blocked` decays 83 → 41.

The accumulator count sets the ceiling. The P-core issues up to 4 FMAs per cycle with a ~3-cycle latency (`fma_peak`, [experiments/cpu-simd](experiments/cpu-simd/README.md)), so a micro-kernel needs 12–16 independent accumulators; the 6×4 tile has 24. The other limit was a single instruction: broadcasting A with `LD1R` held every tile with many rows back, and loading it with `LDR` and multiplying by element made the 6×4 tile 23–30% faster. f32 and f64 now run at the same 85% of their peaks. Every shape measured: [kernels/neon.md](kernels/neon.md).

Peak per family on this machine:

| Kernel | f64 peak | f32 peak | f32/f64 |
|---|---|---|---|
| `gemm_naive` | 8.69 | 8.91 | 1.0× |
| `gemm_reordered` | 25.62 | 82.63 | 3.2× |
| `gemm_blocked` | 25.79 | 82.97 | 3.2× |
| `gemm_simd_blocked<Q>` | 62.25 | 123.78 | 2.0× |
| `gemm_sme` | 424 | **1,504** | 3.5× |
| `gemm_amx` (Accelerate) | **860.28** | **3,295.78** | 3.8× |

---
## AMD Zen 5 — AVX2 + AVX-512 (Linux / GCC)

> **Machine:** AMD Ryzen 9 9950X (Zen 5), 16C/32T, 5.71 GHz measured on one core under load, Ubuntu 24.04, GCC 13.3, C++20
> **Build:** `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_AVX512=ON` → `-O3 -march=native -ffast-math -funroll-loops`
> **CPU Caches:** L1d 48 KiB (×16) · L1i 32 KiB (×16) · L2 1024 KiB (×16) · L3 32768 KiB (×2)
> **Load Average:** 0.00 / 0.11 / 0.08 on 32 threads — idle
> **Governor:** `performance` · **Run time:** 2147 s (36 min) for the full CPU sweep
> **Date:** 2026-10-15; `gemm_simd_blocked<Zmm>` re-measured the same night after its micro-kernel change (below)

Zen 5 has a full-width 512-bit AVX-512 datapath (unlike Zen 4's double-pumped
256-bit), so this is the first machine in this repo where `gemm_avx512_*`
runs on native-width hardware. NEON, SVE, SME and AMX are absent and report
`SKIPPED`; the scalar, AVX2 and AVX-512 tests all pass.

### GFLOP/s by kernel and size

double (f64)

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `gemm_naive` | 14.55 | 2.19 | 1.25 | 0.76 | 0.44 |
| `gemm_reordered` | 41.91 | 52.50 | 36.76 | 33.10 | 13.04 |
| `gemm_blocked` | 41.85 | 36.86 | 35.76 | 35.56 | 33.02 |
| `gemm_simd_blocked<Ymm>` | 82.79 | 81.86 | 62.32 | 54.83 | 53.76 |
| `gemm_simd_blocked<Zmm>` | 153.03 | 162.03 | 157.66 | 150.99 | 145.03 |

float (f32)

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `gemm_naive` | 15.78 | 3.37 | 1.86 | 0.57 | 0.42 |
| `gemm_reordered` | 50.44 | 93.89 | 82.40 | 70.19 | 30.48 |
| `gemm_blocked` | 48.94 | 47.82 | 47.52 | 46.03 | 42.48 |
| `gemm_simd_blocked<Ymm>` | 166.65 | 171.72 | 169.65 | 126.89 | 105.60 |
| `gemm_simd_blocked<Zmm>` | 297.69 | 324.43 | 327.88 | 312.81 | 294.74 |

The `gemm_simd_reordered` rows for AVX2/AVX-512 are omitted above — they
trail scalar reordered, which GCC auto-vectorises at least as well.

The `gemm_simd_blocked<Ymm>` and `gemm_simd_blocked<Zmm>` rows are the median of
three runs on 2026-10-22 with their current register tiles and cache blocks
([`bench_zen5_defaults.csv`](experiments/block-sweeps/bench_zen5_defaults.csv);
blocks: [kernels/blocks.md](kernels/blocks.md)): AVX2 384×256 × 1 KiB of
B's rows, AVX-512 512×256 × 2 KiB. With the old 64-row blocks the same
tiles measured 297.22 / 325.52 / 328.96 / 312.15 / 230.25 f32 and 149.90 /
162.09 / 128.41 / 123.44 / 95.86 f64 (AVX-512), 169.01 / 171.16 / 167.72 /
124.01 / 95.40 f32 and 82.80 / 83.84 / 62.35 / 54.35 / 48.13 f64 (AVX2).

`gemm_simd_blocked<Zmm>` uses an 8×48 f32 / 8×24 f64 register tile (8 rows
× 3 vectors, 24 accumulators) since 2026-10-21. The 8×2 tile it replaced (16 accumulators, since 2026-10-15) measured
320.29 / 332.68 / 327.92 / 284.59 / 198.23 f32 and 154.62 / 161.54 /
105.21 / 90.87 / 77.16 f64; the 4×2 tile before that 216.8 / 205.2 / 141.3
/ 119.9 f32 and 106.2 / 55.8 / 55.1 / 51.1 f64 at N=256 / 512 / 1024 /
4096. Every tile shape and why 8 × 3: [kernels/avx512.md](kernels/avx512.md).

`gemm_simd_blocked<Ymm>` uses a 6×16 f32 / 6×8 f64 register tile (12
accumulators) since 2026-10-21, measured alternated with the previous
4-row tile
([experiments/cpu-simd/avx2-tiles](experiments/cpu-simd/avx2-tiles/)). The
4-row tile ran at 151.2 / 150.2 / 143.4 / 93.5 / 75.9 f32 and 75.1 / 71.3 /
46.0 / 41.7 / 37.3 f64 in the original sweep.

### Speedup vs `gemm_naive`

N=4096:

| Kernel | f64 time | ×naive | f32 time | ×naive |
|---|---|---|---|---|
| `gemm_naive` | 311.84 s | 1.0× | 328.38 s | 1.0× |
| `gemm_reordered` | 10.54 s | 29.6× | 4.51 s | 72.8× |
| `gemm_blocked` | 4.16 s | 74.9× | 3.24 s | 101.5× |
| `gemm_simd_blocked<Ymm>` | 2.56 s | 122.0× | 1.30 s | 252.3× |
| `gemm_simd_blocked<Zmm>` | 0.95 s | 329.0× | 0.47 s | 704.2× |

### Key observations

The scalar kernels depend heavily on the compiler's auto-vectoriser.
`gemm_reordered` and `gemm_blocked` contain no intrinsics at all — their
throughput is whatever the compiler makes of the inner j-loop. Under GCC with
`-march=native -ffast-math`, `gemm_reordered` reaches 93.89 GFLOP/s f32 at
N=256. Where the scalar kernels land is therefore as much about the compiler
as the algorithm.

Blocking earns its keep only at large N. At N≤512 `gemm_blocked` looks
like a regression against `gemm_reordered` (47.82 vs 93.89 GFLOP/s f32 at
N=256) — the untiled kernel still fits cache there, and tiling the j-loop to
64 columns costs the vectoriser more than the cache saves. The ranking
inverts where it matters: at N=4096, blocked reaches 42.48 vs 30.48 G/s
(f32, 1.4×) and 33.02 vs 13.04 G/s (f64, 2.5×).

The register tile needs enough accumulators and register-only FMAs. Zen 5
retires 2 FMAs per cycle with a 4-cycle latency (`fma_peak`,
[experiments/cpu-simd](experiments/cpu-simd/README.md)): 362 GFLOP/s f32 and
180 f64 at 5.71 GHz with AVX-512, and 12 independent accumulators for the
full rate. With a 4×2-ZMM tile (8 accumulators) GCC also folded the B loads
into the FMAs as memory operands, 12 loads per k step for 8 FMAs, and the
kernel stopped at 60% of peak. The 8×2 tile loads B once per k step and has
16 accumulators: 332.7 GFLOP/s f32 (92% of peak) and 161.5 f64 (90%) at
N=256. The 8×3 tile (24 accumulators) is as fast at N≤512 and holds up
better where B misses L1: +10% f32 at N=1024 and +16% at 4096, +22–36% f64
from N=512 up, at the same cache block
([kernels/avx512.md](kernels/avx512.md)). The current cache blocks add
+28% f32 and +51% f64 at N=4096 and +22–23% f64 at N=512–1024
([kernels/blocks.md](kernels/blocks.md)): the B block now fits in half of
L2 in either type, and each B block serves 512 rows of A instead of 64. `gemm_simd_blocked<Ymm>` uses a 6×2-YMM tile (12 accumulators, 15–16 of
AVX2's 16 registers without spilling): 171.7 f32 and 81.9 f64 at N=256
(83.8 with the old block), 94% and 90–92% of the AVX2 peak (182 / 91 GFLOP/s); the 4×2 tile it replaced
reached 83% ([kernels/avx2.md](kernels/avx2.md)).

The drops with N follow the byte stride of B's rows, which the kernels read
in place. The micro-kernel runs over 256 rows of B; at a 4 KiB row stride
(f32 N=1024, f64 N=512) those rows fall into a few cache sets of L1 and L2
and stop staying cached. On L1-resident data, `simd_microkernel` measures a
1 KiB stride costing 5% and a 4 KiB stride 22% against packed B (AVX-512
f32, 16 accumulators), and the 8×2 kernel dropped by the same order at
the same sizes: 327.9 → 284.6 f32 at N=1024, 161.5 → 105.2 f64 at N=512.
The current AVX-512 kernel drops less: 327.9 → 312.8 f32, and its f64
rows lost the drop with the smaller B block (157.7 at N=512); AVX2 still
drops (169.7 → 126.9 f32, 81.9 → 62.3 f64).
Packing B into contiguous strips, as `gemm_sme` does, would remove it.

Software prefetch has almost no effect on Zen 5, and the distance is
irrelevant. Across every family, precision and size, all four distances
(D ∈ {2, 4, 8, 16}) land within ~3% of each other and of no prefetch (the
AVX-512 and AVX2 rows from the run with the current tiles and blocks,
[`bench_zen5_defaults.csv`](experiments/block-sweeps/bench_zen5_defaults.csv)):

| Kernel | D=2 | D=4 | D=8 | D=16 | no prefetch |
|---|---|---|---|---|---|
| `Avx512BlockedPf` f32 N=1024 | 313.2 | 311.9 | 312.4 | 313.2 | 312.8 |
| `Avx512BlockedPf` f64 N=1024 | 150.9 | 151.1 | 151.5 | 151.4 | 151.0 |
| `Avx2BlockedPf` f32 N=256 | 171.62 | 171.55 | 171.49 | 171.49 | 171.72 |
| `BlockedPf` f32 N=512 | 47.16 | 47.29 | 47.01 | 47.96 | 47.52 |

Apple M4 Max shows the same picture: both hardware prefetchers already
follow these streaming access patterns, so explicit hints add little.

`gemm_naive` is slower than on Apple M4 Max at N=4096 (0.44 vs 0.62 GFLOP/s
f64) despite the higher clock, and f32 and f64 are indistinguishable (0.42 vs
0.44) — it is bound by memory latency, where element width doesn't matter.
One naive iteration at N=4096 takes 312 s, which dominates the sweep's
run time.

---

## NVIDIA RTX 5080 — CUDA

> **Machine:** AMD Ryzen 9 9950X host + **NVIDIA GeForce RTX 5080** (Blackwell, sm_120), Ubuntu 24.04, CUDA 13.2, GCC 13.3, C++20
> **Build:** `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc` → native `sm_120` SASS (verified with `cuobjdump --list-elf`)
> **Run:** `./build/benchmarks/bench_gemm_cuda` — 134 s for the full GPU sweep
> **Date:** 2026-10-13, driver 615.71.09, host idle. All CUDA tests passed with no skips.

What each level adds: [src/gemm/cuda.hpp](../src/gemm/cuda.hpp).

**What is timed.** The GEMM alone, as in the CPU tables: A, B and C are
allocated and filled on the GPU before timing, and each iteration is one
call of the kernel's device entry point (`gemm_cuda_*_device`). The time is
GPU time, from CUDA events recorded on the GPU just before and after the
launch. CPU elapsed time is not the reported metric, but GPU idle gaps
while the host submits work can lie between the events, especially for
small kernels; one untimed call
precedes the loop. Copying the matrices between host and GPU is not
included: at N=4096 it takes ~12 ms per call over PCIe (192 MiB), several
times the GEMM itself, and in real use the data stays on the GPU between
operations.

`CudaWmma`, `CudaMmaLdmatrix` and `CudaWmmaPipelined` multiply in fp16
(counter `precision=16`), so their GFLOP/s is not like-for-like with the
fp32 rows. `CudaWmma` and `CudaMmaLdmatrix` take fp32 inputs and convert
each tile inside the kernel; `CudaWmmaPipelined` and `CudaCublasFp16` take
fp16 inputs, converted once before timing.

### GFLOP/s by kernel and size

float (f32)

| Kernel | N=256 | N=1024 | N=4096 | N=8192 | N=16384 |
|---|--:|--:|--:|--:|--:|
| `CudaNaive` | 1,966 | 3,608 | 3,680 | 3,231 | 3,395 |
| `CudaBlocked` | 1,936 | 3,320 | 3,384 | 3,383 | 3,389 |
| `CudaRegTile` | 720 | 14,126 | 23,499 | 24,636 | 23,454 |
| `CudaDoubleBuf` | 838 | 17,051 | 25,636 | 26,486 | 26,125 |
| `CudaVectorized` | 793 | 15,970 | 26,854 | **27,536** | 27,403 |
| `CudaWmma` | 1,542 | 10,740 | 14,627 | 14,147 | 14,572 |
| `CudaMmaLdmatrix` | 1,739 | 15,018 | 16,956 | 17,174 | 16,983 |
| `CudaWmmaPipelined` | 2,865 | 58,935 | 96,688 | 100,727 | 99,790 |
| `CudaTma` ⁸ | 3,008 | 72,602 | 106,551 | **117,342** | **119,937** |
| `CudaCublas` (ref) | 3,599 | 26,498 | 37,906 | 38,860 | 38,962 |
| `CudaCublasTf32` (ref) | 4,399 | 41,574 | 52,864 | 59,174 | 60,858 |
| `CudaCublasFp16` (ref) | 6,117 | 74,165 | 112,981 | 119,088 | 119,985 |

double (f64)

| Kernel | N=256 | N=1024 | N=4096 | N=8192 |
|---|--:|--:|--:|--:|
| `CudaNaive` | 665 | 919 | 932 | 933 |
| `CudaBlocked` | 671 | 950 | **956** | **956** |
| `CudaRegTile` | 44 | 717 | 873 | 927 |
| `CudaDoubleBuf` | 84 | 680 | 874 | 891 |
| `CudaVectorized` | 45 | 723 | 885 | 940 |
| `CudaCublas` (ref) | 601 | 649 | 848 | 848 |

- ⁸ Level 8, measured in a later session (2026-10-16, build
  `CMAKE_CUDA_ARCHITECTURES=120a`) in which `CudaWmmaPipelined` and
  `CudaCublasFp16` reproduced within 0.5%:
  [`bench_gemm_cuda_level8.txt`](experiments/cuda-machine/bench_gemm_cuda_level8.txt).
- f64: every kernel, cuBLAS included, converges on 0.85–0.96 TFLOP/s. That
  is the FP64 rate of consumer Blackwell, a small fraction of its FP32 rate,
  so the f64 ladder shows the FP64 units, not the kernels.
- N=256 is too small for the tiled kernels: a 128×128 block tile gives 4
  thread blocks for the GPU's 84 SMs, so `CudaNaive` (16×16 tiles, 256
  blocks) beats them there. From N=1024 the ranking settles.

### Level by level, f32, N=4096

Levels 0–4 run on the FMA cores, 5–8 on the Tensor Cores.

| Level | Kernel | GFLOP/s | ×Naive | Note |
|---|---|--:|--:|---|
| 0 | `CudaNaive` | 3,680 | 1.0× | Coalesced B loads and L1/L2 reuse make this baseline strong. |
| 1 | `CudaBlocked` | 3,384 | 0.92× | One output per thread: two `__syncthreads()` per k-tile pay for only 16 FMAs. |
| 2 | `CudaRegTile` | 23,499 | 6.4× | 64 outputs per thread from 16 shared-memory reads per k step; the largest single step. |
| 3 | `CudaDoubleBuf` | 25,636 | 7.0× | The copy of the next k-tile overlaps the FMAs. |
| 4 | `CudaVectorized` | 26,854 | 7.3× | 128-bit loads into shared memory and 128-bit reads out of it: the fastest FMA kernel, 71% of cuBLAS SGEMM. |
| 5 | `CudaWmma` | 14,627 | 4.0× | Tensor Cores, but below Levels 2–4: 64×64 tiles, one fragment per warp and a single buffer leave too little work per synchronisation. |
| 6 | `CudaMmaLdmatrix` | 16,956 | 4.6× | 16% above Level 5 with the same tile sizes and single buffering: dropping to PTX alone does not remove the bottleneck. |
| 7 | `CudaWmmaPipelined` | 96,688 | 26.3× | 128×128 tiles, 8 fragments per warp, `cp.async` double buffering: 6.6× Level 5. |
| 8 | `CudaTma` | 106,551 | 29.0× | TMA copies into swizzled shared memory, a producer warp and `mbarrier` pipeline, 64×64 warp tiles: a third of Level 7's instructions; 99–100% of cuBLAS FP16 from N=8192. |

### Against cuBLAS

TFLOP/s, ascending:

| Path | Precision | N=4096 | N=8192 | N=16384 |
|---|---|--:|--:|--:|
| `gemm_cuda_vectorized` | FP32 on the FMA cores, hand-written | 26.9 | 27.5 | 27.4 |
| `cublasSgemm` | FP32 on the FMA cores | 37.9 | 38.9 | 39.0 |
| `cublasGemmEx` TF32 | TF32 Tensor Cores, 10-bit mantissa | 52.9 | 59.2 | 60.9 |
| `gemm_cuda_wmma_pipelined` | FP16 Tensor Cores, hand-written | 96.7 | 100.7 | 99.8 |
| **`gemm_cuda_tma`** | **FP16 Tensor Cores, hand-written (Level 8)** | **106.6** | **117.3** | **119.9** |
| `cublasGemmEx` FP16 | FP16 Tensor Cores, fp32 accumulate | 113.0 | 119.1 | **120.0** |

- 100+ TFLOP/s on this GPU needs dense FP16 Tensor Cores: plain FP32, the
  ceiling for every non-Tensor-Core kernel, tops out at ~39 TFLOP/s, and
  TF32 at ~61.
- The hand-written kernels reach 71% of cuBLAS on the FMA cores and, with
  Level 8, 100% on the Tensor Cores (N=16384; Level 7: 83%).

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

The kernel had required about 6.6× as many shared-load wavefronts;
these count bank-service work, not 6.6× as many requested data bytes;
afterwards it is within 1% of the minimum wavefront count. Shared memory
per block grows from 32 to 37 KB, lowering the shared-memory occupancy
limit from 3 to 2 blocks per SM, which costs nothing: its registers (126
per thread then, 118 now) already cap it at 2. That register pressure (occupancy 33%, ~67%
unused warp slots per Nsight) constrains occupancy; it does not by itself
prove the remaining performance bottleneck. Deeper pipelines and split-K
are possible experiments, not measured fixes for the remaining gap.

---

## Apple M4 Max GPU — Metal

> **Machine:** Apple M4 Max, 40-core GPU, macOS 26.3.1, Apple Clang 17, Metal shaders compiled at run time with fast math
> **Build:** default (`HPC_ENABLE_METAL` is on on Apple); `cmake -B build -DCMAKE_BUILD_TYPE=Release`
> **Command:** `./build/benchmarks/bench_gemm_metal --benchmark_repetitions=5 --benchmark_min_time=1s --benchmark_enable_random_interleaving=true`
> **Date:** 2026-10-24, AC power, load average 1.5–3.6. Median of 5 interleaved repetitions; cv up to 17% per row. Data and method: [experiments/metal-m4](experiments/metal-m4/README.md)

What each level adds: [src/gemm/metal.hpp](../src/gemm/metal.hpp). Apple
GPUs have no FP64, so every row is f32 in and out; the F16 rows multiply
fp16 copies of A and B, converted once before timing, like the CUDA fp16
rows. **What is timed** is the GEMM alone, as for CUDA: inputs sit in
GPU-visible memory before timing, and the time is GPU time from the command
buffer's start and end timestamps. Each command buffer holds enough GEMMs
to last ≥ ~5 ms (256 at N=256), since at small N one GEMM is shorter than the
command buffer's fixed cost.

**Read across rows, not across sessions.** On this laptop the absolute GPU
rate moved by up to ~3× between sessions (`MetalMps` f32 at N=4096: 13.5
TFLOP/s in a first quick pass, 11.1 in this run, 4.5–7.3 later), while the
register-only FMA peak stayed near 14.7. That fits sustained power limiting
of memory-heavy work; [the data README](experiments/metal-m4/README.md#two-things-found-on-the-way)
has the details. The rows below come from one run, interleaved, so they
compare fairly with each other.

### GFLOP/s by kernel and size

| Level | Kernel | N=256 | N=1024 | N=2048 | N=4096 | N=8192 |
|--:|---|--:|--:|--:|--:|--:|
| 0 | `MetalNaive` | 944 | 1,132 | 1,226 | 1,027 | 814 |
| 1 | `MetalTiled` | 1,327 | 1,605 | 1,615 | 1,598 | 1,860 |
| 2 | `MetalRegTile` | 1,248 | 5,923 | 6,494 | 6,298 | 6,469 |
| 3 | `MetalSimd` | 2,131 | 9,549 | 10,531 | 10,205 | 5,560 |
| 4 | `MetalSimdTg` | 1,364 | 8,846 | 10,717 | 9,104 | 9,373 |
| 5 | `MetalSimdTgF16` (fp16 in) | 1,492 | 9,450 | **11,617** | 11,380 | 11,211 |
| ref | `MetalMps` (MPSMatrixMultiplication) | 3,475 | **11,123** | 10,704 | 11,087 | 10,855 |
| ref | `MetalMpsF16` (fp16 in, fp32 out) | 4,974 | 10,868 | 11,479 | **13,732** | 12,671 |

Measured FMA peak in the same run: **14.2 TFLOP/s f32** (14.9 run alone,
three times out of three: 40 cores × 128 lanes × 2 at ~1.45 GHz) and 12.0
f16 (15.5–15.8 alone). fp16 FMA runs at the f32 rate.

- Level by level at N=2048: tiling in threadgroup memory 1.3×, the register
  tile 4.0× on top (8×4 outputs per thread; 4×4 measured 14–20% slower), and
  simdgroup matrices another 1.6×, to 74% of the f32 peak (`MetalSimd` at
  N=2048, 10.5 TFLOP/s). The register tile is the largest single step, as
  Level 2 is on the RTX 5080.
- Simdgroup matrices give 1.6× over the best SIMT kernel although on M1–M4
  they run on the same FP32 ALUs: there is no separate matrix unit (fp16 FMA
  runs at the f32 rate). Each 8×8 fragment load feeds 8 multiply-adds per
  element instead of one, so fewer instructions are spent on operands.
- Staging A and B in threadgroup memory (Level 4) gains nothing over loading
  fragments straight from device memory (Level 3) up to N=4096 (−11% to +2%,
  within noise at N=2048); the GPU's
  caches already serve the reuse between simdgroups. It wins at N=8192
  (9.4 against 5.6 TFLOP/s), where the operands no longer stay cached.
- fp16 inputs (Level 5) gain 8–25% over Level 4 from N=2048, from halving
  the bytes moved, not from faster arithmetic.
- Against MPS: the best f32-input kernel reaches 100% / 92% / 86% of
  `MetalMps` at N = 2048 / 4096 / 8192; Level 5 (fp16 inputs) 103–109% of
  `MetalMps` and 83–101% of `MetalMpsF16`. MPS with fp16 inputs is the
  fastest row, 13.7 TFLOP/s at N=4096, 97% of the peak measured in the same
  run.
- At N=256, MPS is 1.6–3.7× faster than every hand-written level: a 64×64
  tile gives 16 threadgroups for 40 GPU cores.

---

## Low precision: bf16, fp16, INT8, FP8

Analysis in [part 5](articles/lowp-gemm.md); raw data, machines and method
in [experiments/lowp](experiments/lowp/README.md). Inputs are converted to
the format before timing; every rate counts a multiply-add as 2 operations,
INT8 included. Medians of 3.

Apple M4 Max, one P-core / one SME unit, GFLOP/s or GOP/s
(`bench_gemm_lowp`; SME rows 2026-10-16, NEON rows one session on
2026-10-22 with the current FP32 kernel,
[`lowp_m4_defaults.csv`](experiments/block-sweeps/lowp_m4_defaults.csv)):

| Kernel | Format | N=256 | N=1024 | N=2048 | N=4096 |
|---|---|--:|--:|--:|--:|
| `Sme` | f32 | 779 | 1,440 | 1,540 | 1,535 |
| `SmeWidening` | fp16 | 967 | 1,689 | 1,732 | 1,756 |
| `SmeWidening` | bf16 | 967 | 1,691 | 1,737 | 1,774 |
| `SmeWidening` | int8 | 1,579 | 3,148 | 3,525 | 3,413 |
| `NeonBlocked` | f32 | 122 | 121 | 118 | 101 |
| `NeonBfmmla` | bf16 | 66 | 66 | 66 | 65 |
| `NeonFmlal` | fp16 | 109 | 114 | 113 | 112 |
| `NeonFp16Acc` | fp16 | 204 | 216 | 214 | 209 |
| `NeonSmmla` | int8 | 449 | 493 | 503 | 491 |

AMD Zen 5, one core, GFLOP/s or GOP/s (`bench_gemm_lowp`, pinned to core 2;
all rows one session on 2026-10-23, after the rewrite of the low-precision
kernels,
[`bench_lowp_zen5_rewrite.csv`](experiments/lowp/avx512-rewrite/bench_lowp_zen5_rewrite.csv)):

| Kernel | Format | N=256 | N=1024 | N=2048 | N=4096 |
|---|---|--:|--:|--:|--:|
| `Avx512Blocked` | f32 | 325 | 311 | 302 | 295 |
| `Avx512Dpbf16` | bf16 | 619 | 638 | 606 | 574 |
| `Avx512Vnni` | int8 | 824 | 1,139 | 1,138 | 1,038 |

RTX 5080, TFLOP/s or TOP/s, GPU time of the GEMM alone
(`bench_gemm_lowp_cuda`, `CMAKE_CUDA_ARCHITECTURES=120a`; fp16_acc16 sums in
fp16, FP8 uses the block-scaled instruction):

| Format | Implementation | N=1024 | N=2048 | N=4096 | N=8192 | N=16384 |
|---|---|--:|--:|--:|--:|--:|
| bf16 | cp.async kernel | 63.2 | 79.6 | 101.2 | 108.7 | 109.1 |
| bf16 | Level 8 | 72.6 | 87.9 | 106.9 | 117.8 | 120.5 |
| bf16 | cuBLAS | 74.3 | 96.9 | 113.1 | 120.1 | 120.9 |
| fp16 | cp.async kernel | 62.9 | 79.1 | 100.7 | 107.7 | 107.7 |
| fp16 | Level 8 | 72.7 | 87.9 | 106.6 | 117.3 | 120.0 |
| fp16 | cuBLAS | 74.1 | 96.4 | 112.6 | 118.8 | 119.2 |
| fp16_acc16 | cp.async kernel | 90.9 | 117.7 | 142.0 | 150.3 | 152.0 |
| fp16_acc16 | Level 8 | 123.3 | 162.0 | 182.5 | 205.8 | 209.6 |
| fp16_acc16 | cuBLAS | 116.4 | 155.4 | 196.1 | 202.0 | 203.4 |
| int8 | cp.async kernel | 153.5 | 230.1 | 278.1 | 298.3 | 304.4 |
| int8 | Level 8 | 192.1 | 291.5 | 357.9 | 408.2 | 419.0 |
| int8 | cuBLAS | 161.9 | 287.5 | 305.0 | 419.0 | 427.6 |
| fp8_e4m3 | cp.async kernel | 150.4 | 229.8 | 271.6 | 291.4 | 298.4 |
| fp8_e4m3 | Level 8 | 191.7 | 286.1 | 344.0 | 387.3 | 398.0 |
| fp8_e4m3 | cuBLAS | 170.9 | 269.0 | 359.7 | 399.9 | 408.3 |
