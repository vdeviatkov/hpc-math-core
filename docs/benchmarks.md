# Benchmark Results

Measured throughput for every kernel family in this repository, on the three
machines it has been verified on. All numbers come from
`bench_gemm` / `bench_gemm_cuda` / `bench_gemm_metal` and, for low
precision, `bench_gemm_lowp` / `bench_gemm_lowp_cuda` (Google Benchmark)
built in Release mode as described in [build.md](build.md). CPU kernels
are strictly single-threaded unless noted (Apple AMX via Accelerate and
`gemm_sme_mt` are the exceptions).

Contents

- [Summary: efficiency against peak](#summary-efficiency-against-peak) — best kernels per family as a share of measured peak
- [Deriving GFLOP/s](#deriving-gflops)
- [Apple M4 Max](#apple-m4-max) — scalar · NEON · SME2 (one and several units) · AMX · KleidiAI
- [Apple M4 Max GPU — Metal](#apple-m4-max-gpu--metal) — Metal Levels 0-5 · MPS reference
- [AMD Zen 5](#amd-zen-5--avx2--avx-512-linux--gcc) — scalar · AVX2 · AVX-512
- [NVIDIA RTX 5080](#nvidia-rtx-5080--cuda) — CUDA Levels 0-8 · cuBLAS reference
- [Low precision](#low-precision-bf16-fp16-int8-fp8) — bf16 · fp16 · INT8 · FP8 on all three machines

Each CPU section has the same shape: build metadata, GFLOP/s by kernel
and size, speedup against the naive baseline, then key observations.

---

## Summary: efficiency against peak

FP32 throughput at a small and a large size: N = 256 and 4096 on the CPU, N = 1024 and 8192 on the GPUs. CPU rows are single-threaded unless noted. Throughput is in GFLOP/s on the CPU and TFLOP/s on the GPUs, as each column header says; the percentage is its **efficiency**: measured throughput ÷ that peak, i.e. how much of the hardware's arithmetic the kernel uses (100% would mean every FMA unit busy every cycle). FP64, more sizes and analysis are in the per-machine sections below.

**CPU**

| Family | Hardware | Register tile, C rows × cols | Blocks, M × K × N | Compute peak, f32, GFLOP/s ⁵ | N = 256, GFLOP/s (% of peak) | N = 4096, GFLOP/s (% of peak) |
|---|---|---|---|--:|--:|--:|
| Scalar, cache-blocked (`gemm_blocked`) | Apple M4 Max | — | 64 × 64 × 64 | ~130 | 80 (62%) | 29 (22%) |
| ARM NEON, blocked (`gemm_simd_blocked<Q>`) ⁴ | Apple M4 Max | 6 × 16 | 384 × 256 × 256 | 144 | 123 (85%) | 101 (70%) |
| ARM NEON, packed B (`gemm_simd_packed<Q>`) ⁴ | Apple M4 Max | 6 × 16 | 256 × 512 × 512 | 144 | 121 (84%) | **121 (84%)** |
| AVX2, blocked (`gemm_simd_blocked<Ymm>`) ⁴ | AMD Zen 5, GCC | 6 × 16 | 384 × 256 × 256 | 183 | 170 (93%) | 104 (57%) |
| AVX2, packed B (`gemm_simd_packed<Ymm>`) ⁴ | AMD Zen 5, GCC | 6 × 16 | 256 × 256 × 512 | 183 | 166 (91%) | **167 (91%)** |
| AVX-512, blocked (`gemm_simd_blocked<Zmm>`) ⁴ | AMD Zen 5, GCC | 8 × 48 | 512 × 256 × 512 | 365 | 325 (89%) | 294 (81%) |
| AVX-512, packed B (`gemm_simd_packed<Zmm>`) ⁴ | AMD Zen 5, GCC | 8 × 48 | 128 × 512 × 512 | 365 | 311 (85%) | **318 (87%)** |
| ARM SME2 (`gemm_sme`) | Apple M4 Max | 16 × 64, 1 × 4 ZA tiles | 1024 × 2048 × 4096 | 1,960 ⁶ | 770 (39%) | **1,600 (82%)** |
| ARM SME2, both units (`gemm_sme_mt`, 8 threads) ⁷ | Apple M4 Max | 16 × 64, 1 × 4 ZA tiles | 1024 × 2048 × 4096 | 3,930 ⁶ | 392 (10%) | **3,487 (89%)** |
| Apple AMX via Accelerate (`gemm_amx`) ¹ | Apple M4 Max | — | — | 3,930 ⁶ | 1,717 (44%) | 3,162 (80%) |
| Accelerate, 1 core (reference) | Apple M4 Max | — | — | 1,960 ⁶ | 1,707 (87%) | 1,599 (82%) |
| KleidiAI, 1 core (reference) | Apple M4 Max | — | — | 1,960 ⁶ | 1,072 (55%) | 1,099 (56%) |

**GPU** (larger sizes: N = 256 is too small to occupy these GPUs)

| Family | Hardware | Compute peak, f32, TFLOP/s ⁵ | N = 1024, TFLOP/s (% of peak) | N = 8192, TFLOP/s (% of peak) |
|---|---|--:|--:|--:|
| CUDA, FMA (`gemm_cuda_vectorized`) ² | NVIDIA RTX 5080 | 60.9 | 16.3 (27%) | 29.8 (49%) |
| cuBLAS SGEMM (reference) ² | NVIDIA RTX 5080 | 60.9 | 26.5 (43%) | 38.7 (64%) |
| CUDA, Tensor Cores (`gemm_cuda_tma`, Level 8) ² ³ | NVIDIA RTX 5080 | 124.7 | 72.7 (58%) | **117.6 (94%)** |
| cuBLAS dense FP16 (reference) ² ³ | NVIDIA RTX 5080 | 124.7 | 83.7 (67%) | 118.3 (95%) |
| Metal, simdgroup matrices (`MetalSimdTg`) ⁸ | Apple M4 Max GPU | 14.9 | 9.9 (66%) | 11.4 (76%) |
| Metal, fp16 inputs (`MetalSimdTgF16`) ⁸ | Apple M4 Max GPU | 14.9 | 10.7 (72%) | 13.5 (91%) |
| MPS reference (`MetalMps` / `MetalMpsF16`) ⁸ | Apple M4 Max GPU | 14.9 | 12.4 / 13.8 (83% / 93%) | 12.6 / 13.8 (84% / 93%) |

Small matrices favour the CPU's vector units: at N = 256 one core's SIMD kernels already run at 84–93% of peak, while the work is too little to fill a matrix unit. At N = 4096 the matrix units lead, and packing B keeps the SIMD kernels at 84–91%. The GPUs need N = 8192 to approach their peaks.

¹ Vendor BLAS with its default threading (all cores) — not a single-core comparison.
² GPU time of the GEMM alone, data already on the GPU (as the CPU rows measure the multiplication alone). At N = 16384, the largest size measured: 28.5 (`gemm_cuda_vectorized`), 38.9 (cuBLAS SGEMM), 120.3 (`gemm_cuda_tma`, 96%) and 121.2 TFLOP/s (cuBLAS FP16); tile sizes from a sweep of every level ([docs/kernels/cuda.md](kernels/cuda.md); [§ CUDA](#nvidia-rtx-5080--cuda)).
³ fp16 inputs, fp32 accumulate; `gemm_cuda_tma` reaches 99% of cuBLAS FP16 on the same GPU at N = 8192 and 16384 (Level 7, `gemm_cuda_wmma_pipelined`: 107.9 TFLOP/s at N = 8192, 91%; 87% at N = 16384).
⁴ Default register tiles (NEON 6×4, AVX2 6×2, AVX-512 8×3) and cache blocks; blocked and packed of each ISA measured in the same session (medians of 3 on Zen 5 and 5 on M4, [packed-b](experiments/packed-b/README.md)). Blocked reads B in place and is fastest at small N; packed copies each block of B into contiguous strips (GotoBLAS order) and stays flat from N = 1024 up, where B's power-of-two row strides cost the blocked kernels up to 38% (AVX2). The copy costs 5–13% at N = 64. The best shape in the full sweeps reaches 331 GFLOP/s (91%) on AVX-512, the 14×2 tile at N = 256 ([docs/kernels](kernels/README.md)).
⁵ Peak = execution units × lanes × 2 flops per FMA × the clock measured during that row's runs. M4 Max P-core: 4 FMA pipes × 128-bit; ~4.05 GHz (3.96–4.13, moving with background load) in the scalar run → ~130 GFLOP/s, 4.49 GHz in the NEON run → 144. Zen 5: 2 FMA pipes × 256-bit (AVX2) or × 512-bit (AVX-512) at 5.71 GHz → 183 / 365. M4 Max SME2: 1 `FMOPA` per cycle, 512 flops, per unit → 1.96 TFLOP/s, 3.93 for both units ⁶. RTX 5080: 84 SMs × 128 FP32 lanes at 2.83–2.90 GHz → 60.9 TFLOP/s; Tensor Cores 84 SMs × 512 fp16 → fp32 flops per cycle at 2.90 GHz → 124.7, the rate the `mma.sync` microbenchmark implies. M4 Max GPU (40 cores): 14.2 TFLOP/s measured FMA peak in the same run; fp16 inputs run at the f32 rate. The microbenchmarks reach 98–100% of these figures ([cpu-simd](experiments/cpu-simd/README.md), [CUDA](articles/cuda-gemm.md)).
⁶ Apple publishes no figures for its matrix unit. These are measured: `FMOPA` issues once per cycle with four accumulator tiles, 512 FP32 flops each ([part 2](articles/sme2-gemm.md)); `gemm_amx` runs on both P-clusters' units. The `gemm_sme` row peaks at 1.63 TFLOP/s at N = 4000.
⁷ Wall-clock time, 8 threads; 110% of Accelerate with all cores at N=4096, 87% at N=1024 ([§ Several SME units](#several-sme-units-gemm_sme_mt)). Below N ≈ 512 creating 8 threads per call costs more than it saves.
⁸ GPU time of the GEMM alone, median of 5 in one run; best figures 12.1 (`MetalSimdTg`, N = 2048), 13.6 (`MetalSimdTgF16`, N = 4096) and 13.1 / 14.4 TFLOP/s (MPS / MPS fp16). The M4 Max GPU's sustained rate varies by up to ~3× between runs on the recording laptop while its register-only FMA peak stayed at 14.2–14.9 TFLOP/s; compare these rows with each other, not across runs ([§ Metal](#apple-m4-max-gpu--metal)).

Low precision at the same sizes as above: N = 256 and 4096 on the CPU, N = 1024 and 8192 on the GPU. A multiply-add counts as 2 operations, so for bf16 GOP/s equal GFLOP/s ([article 5](articles/lowp-gemm.md)):

**CPU**

| Family | Hardware | Register tile, C rows × cols | Blocks, M × K × N | Peak, GOP/s | N = 256, GOP/s (% of peak) | N = 4096, GOP/s (% of peak) |
|---|---|---|---|--:|--:|--:|
| AVX-512, bf16 → fp32 (`gemm_simd_lowp<ZmmDot>`) | AMD Zen 5, GCC | 8 × 32 | 128 × 512 × 512 | 713 | 619 (87%) | 574 (81%) |
| AVX-512, INT8 → INT32 (`gemm_simd_lowp<ZmmDot>`) | AMD Zen 5, GCC | 8 × 32 | 128 × 1024 × 512 | 1,435 | 824 (57%) | 1,038 (72%) |
| ARM NEON, bf16 → fp32 (`gemm_simd_lowp<QMmla>`) | Apple M4 Max | 8 × 8 | 256 × 1024 × 512 | 71 | 65 (92%) | 66 (93%) |
| ARM NEON, INT8 → INT32 (`gemm_simd_lowp<QMmla>`) | Apple M4 Max | 8 × 8 | 256 × 2048 × 512 | 511 | 444 (87%) | 484 (95%) |
| ARM SME2, widening, bf16 → fp32 (`gemm_sme<bf16>`) | Apple M4 Max | 16 × 64, 1 × 4 ZA tiles | 1024 × 4096 × 4096 | 2,018 | 972 (48%) | 1,853 (92%) |
| ARM SME2, widening, INT8 → INT32 (`gemm_sme<int8_t>`) | Apple M4 Max | 16 × 64, 1 × 4 ZA tiles | 1024 × 8192 × 4096 | 4,029 | 1,604 (40%) | 3,726 (92%) |

**GPU**

| Family | Hardware | Peak, TOP/s | N = 1024, TOP/s (% of peak) | N = 8192, TOP/s (% of peak) |
|---|---|--:|--:|--:|
| CUDA, Tensor Cores, bf16 → fp32 (`gemm_cuda_lowp_tma_device`, Level 8) | NVIDIA RTX 5080 | 124 | 72.6 (59%) | 117.8 (95%) |
| cuBLAS, bf16 → fp32 (reference) | NVIDIA RTX 5080 | 124 | 74.3 (60%) | 120.1 (97%) |
| CUDA, Tensor Cores, INT8 → INT32 (`gemm_cuda_lowp_tma_device`, Level 8) | NVIDIA RTX 5080 | 497 | 192.1 (39%) | 408.2 (82%) |
| cuBLAS, INT8 → INT32 (reference) | NVIDIA RTX 5080 | 497 | 161.9 (33%) | 419.0 (84%) |

The percentage is again measured ÷ peak. Peaks here are the instructions' measured throughput (`VDPBF16PS` / `VPDPBUSD`, `BFMMLA` / `SMMLA`, widening `FMOPA` / `SMOPA`, `mma.sync`), from part 5's [peak table](articles/lowp-gemm.md). Best sizes, where higher: `gemm_simd_lowp<ZmmDot>` bf16 / INT8 638 / 1,139 at N = 1024, `gemm_sme<int8_t>` INT8 3,752 at N = 4000, Level 8 120.5 / 419 at N = 16384.

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
> **Load Average:** 5–7 on 16 cores during the scalar runs, including two unrelated processes at 100% each; 2–10 during the NEON runs — treat ±5% as noise

Five families run here: scalar, NEON, SME2 (opt-in), AMX via Accelerate,
and the KleidiAI reference (built when SME is on). AVX2 and AVX-512 are
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
vectors, 24 accumulators), loads A with `LDR` and multiplies by element, and
uses a 384×256×256 cache block ([kernels/blocks.md](kernels/blocks.md)). Its
rows are the median of three runs
([`bench_m4_defaults.csv`](experiments/block-sweeps/bench_m4_defaults.csv)),
at a core clock of about 4.49 GHz; the other rows ran at 3.96–4.13 GHz, so
compare across them with ~10% allowance. Every tile shape and why 6 × 4:
[kernels/neon.md](kernels/neon.md).

### ARM SME2 (`-DHPC_ENABLE_SME=ON`)

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter=Sme` (all four SME kernels; the `svl` counter reports the streaming vector length, 16 f32 / 8 f64 elements)

`gemm_sme` adds SME's outer products into the ZA tiles, with packed A and B,
GotoBLAS cache blocking and SME2 multi-vector loads; `gemm_sme_1tile`,
`gemm_sme_tiles` and `gemm_sme_blocked` are the steps before it. Design:
[src/gemm/sme.hpp](../src/gemm/sme.hpp).

> **Load:** load average 3–5 on 16 cores. Median of 3 interleaved repetitions; treat ±5% as noise. ZA shape 1×4 in both types (`SmeParams`; [kernels/sme.md](kernels/sme.md#shape-sweep-m4-max)). Data: [experiments/sme2-gemm/ladder.json](experiments/sme2-gemm/ladder.json)

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_sme` f64 | 81 | 320 | 409 | **442** | 428 | 425 |
| `gemm_sme` f32 | 268 | 768 | 1,332 | 1,580 | 1,582 | **1,616** |

- f64 is ~¼ of f32, which matches the hardware: an f64 FMOPA covers 8×8 elements against 16×16 for f32.
- One compiler pitfall costs ~5×. A streaming helper without a ZA attribute (`__arm_inout("za")` etc.) is "private-ZA": Clang won't inline it into a ZA-owning caller and wraps every call in a lazy ZA save (`TPIDR2` + `smstart za`). Inside the k loop that dropped throughput from ~1,290 to ~270 GFLOP/s f32.

#### The four SME steps

Block sizes Mc = 1024 (f32) / 512 (f64), Kc = 2048, Nc = 4096, ZA shape
1×4 (`SmeParams` in [`benchmarks/sme_params.hpp`](../benchmarks/sme_params.hpp)).
Same run, median of 3:

| Kernel | N=256 | N=512 | N=1024 | N=2048 | N=4000 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_sme_1tile` f32 | 393 | 471 | 475 | 208 | 126 | 131 |
| `gemm_sme_tiles` f32 | 917 | 1,299 | 1,528 | 701 | 367 | 389 |
| `gemm_sme_blocked` f32 | 938 | 1,298 | 1,541 | 665 | 1,562 | 516 |
| `gemm_sme` f32 | 768 | 1,332 | 1,580 | 1,582 | 1,645 | 1,616 |
| `gemm_sme_1tile` f64 | 119 | 122 | 115 | 40 | 43 | 38 |
| `gemm_sme_tiles` f64 | 309 | 399 | 312 | 129 | 128 | 122 |
| `gemm_sme_blocked` f64 | 307 | 401 | 316 | 131 | 413 | 124 |
| `gemm_sme` f64 | 320 | 409 | 442 | 428 | 433 | 425 |

- One tile → all tiles: 2.3–3.4× f32, 2.6–3.3× f64. With one tile every FMOPA waits for the previous one's accumulator, and at large N each loaded vector feeds a single FMOPA.
- All tiles → blocked: 4.3× f32 / 3.2× f64 at N=4000. Without blocking all of A is re-read for every B strip. N=4000 is as slow as N=4096 without blocking, so this is not a stride effect. It is capacity: `gemm_sme_tiles` f32 holds 1,550–1,620 GFLOP/s up to N = 2304 (A 20 MiB, about the 16 MiB L2) and then falls as A grows (1,409 at N = 2560, 954 at 3072, 368 at 4000); apart from that, only the exact power of two N = 2048 dips (697, against 1,553 at 2000 and 1,528 at 2100) ([data](experiments/sme2-gemm/tiles_capacity.csv), `sme_probe tiles`).
- Blocked → `gemm_sme`: +5% at N=4000, but 2.4× f32 at N=2048 and 3.1× at N=4096 (3.3× / 3.4× f64): the B row stride effect below. With 2,048-row strips an 8 KiB stride (N=2048) hurts as much as a 16 KiB one.
- At N = 256 the kernels without B packing are faster (+22% f32): everything fits in cache, and `gemm_sme` pays for the copy; from N = 512 the copy has paid for itself.
- Against B-resident blocks (Mc = 128, Kc = 1024), in the run-time block sweep: `gemm_sme` f32 +13% at N=512, +1% at N=1024, +4% at N=2048, +11% at N=4096; f64 +11% / +0.3% / +3% / +7% ([data](experiments/sme2-gemm/README.md)).

#### SME: B row stride

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter=BStride --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true --benchmark_report_aggregates_only=true`

The same blocked kernel with B read in place (`gemm_sme_blocked`) and packed
(`gemm_sme`), default block sizes (2,048-row B strips), with the 2 × 2 tile
arrangement (32-column strips; the 1 × 4 kernel shows the same cliff, see
[§ The four SME steps](#the-four-sme-steps)); GFLOP/s, median of 3
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
- f64: in place loses 26–48% at almost every N around 4096, padded or not; only N = 4000 escapes. Its strips are 2,048 rows at strides twice as large, and the paddings above (256 and 512 B) leave them colliding: measured with the 1 × 4 kernel (`sme_probe f64`, [data](experiments/sme-b-stride/f64_kc.csv)), in-place f64 at N = 4096 with 2,048-row strips runs at 123 GFLOP/s unpadded, 145 and 184 padded by 256 and 512 B, but 405 padded by 128 B (packed: ~430). With strips of ≤ 1,024 rows every padding recovers (383–393). A 256-byte pad offsets 2,048 rows over only 64 positions within 16 KiB, a 512-byte pad over 32; 128 B gives twice the positions of 256 B. `gemm_sme` packs B, so none of this reaches it.
- ldb = 4097 is a control, not a fix: its row segments are no longer 128 B aligned, so each spans an extra cache line, and the loss mixes that cost with the stride effect.
- Where the conflict happens, and the cost of each fix (f32): [experiments/sme-b-stride](experiments/sme-b-stride/README.md). With rows 16 KiB apart the macro-kernel loses 5–26% up to 576 rows per strip and breaks between 576 and 640, and the core's pointer chase at that stride holds roughly 450–600 lines in L2 against 8 in L1: the conflicts are in L2. With a padded stride, packing B costs ~2.5 ms and saves 7–8 ms of compute at N=4096; the remaining in-place cost is 4–8% across N = 1024–4224.

### Matrix engines, single core

> **Command:** `VECLIB_MAXIMUM_THREADS=1 ./build/benchmarks/bench_gemm --benchmark_filter='Sme|Amx|KleidiAI'`

One run, load average 2–3, median of 3 interleaved repetitions ([data](experiments/sme2-gemm/head_to_head.json)). Every row uses one core, so this is the like-for-like comparison. Accelerate was limited to one thread with `VECLIB_MAXIMUM_THREADS=1`; KleidiAI is always single-threaded.

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_sme` f64 | 80 | 320 | 407 | **433** | 422 | 431 |
| Accelerate f64 | 325 | **461** | 423 | 366 | 394 | 394 |
| `gemm_sme` f32 | 266 | 770 | 1,323 | 1,556 | 1,546 | 1,600 |
| Accelerate f32 | 789 | **1,707** | 1,697 | 1,653 | **1,704** | 1,599 |
| KleidiAI v1.31 f32 | 349 | 1,072 | 1,509 | **1,694** | 1,560 | 1,099 |

- f64: `gemm_sme` is within 4% of Accelerate at N=512 and ahead from N=1024 up (+18% at N = 1024, +9% at 4096). Accelerate's dip at N=1024 appears in every run. Below N=512 Accelerate is far ahead (4× at N=64), because `gemm_sme` pays fixed per-call costs: allocating and packing A and B, plus a streaming-mode switch per block.
- f32: `gemm_sme` reaches 91–100% of Accelerate for N ≥ 1024 (100% at N = 4096). It ties KleidiAI at N=2048 and beats it at N=4096 (1,600 vs 1,099); KleidiAI uses a 2×2-tile micro-kernel and no cache blocking, so it drops at large N.
- With all cores (default threading, wall-clock time): Accelerate 3,162 GFLOP/s f32 / 794 GFLOP/s f64 at N=4096 ([§ Several SME units](#several-sme-units-gemm_sme_mt)), about double the single-core figure: the two performance clusters each have a matrix unit, which Accelerate's work shares with SME ([amx_sme_share](experiments/sme2-machine/README.md)).

### Apple AMX (via Accelerate.framework)

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter=Amx` (`HPC_ENABLE_AMX` defaults ON on Apple)

`gemm_amx` calls Accelerate's BLAS (`cblas_sgemm`/`cblas_dgemm`), Apple's route to its AMX coprocessor (unrelated to Intel AMX; see [src/gemm/amx.hpp](../src/gemm/amx.hpp)). Accelerate may use several cores, so these numbers are not single-core.

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `gemm_amx` f64 | 325 | 463 | 784 | **830** | 788 | 794 |
| `gemm_amx` f32 | 782 | 1,717 | 2,581 | 3,207 | **3,231** | 3,162 |

(The `default.json` run of [§ Several SME units](#several-sme-units-gemm_sme_mt).)

- Up to 3.2 TFLOP/s f32 and 830 GFLOP/s f64, ~2.0× / ~1.9× the single-threaded `gemm_sme` — with every core available. Limited to one thread it reaches ~1.7 TFLOP/s f32 from N=256 up ([§ Matrix engines, single core](#matrix-engines-single-core)), so the climb from ~0.8 to ~3.2 TFLOP/s is about half problem size and half the second performance cluster.
- The f32/f64 ratio is ~3.9×, not the ~2× lane-count ratio seen elsewhere (NEON 2.0×, AVX-512 ~2×): the 4× of the M4's matrix unit, where an f64 outer product covers 8×8 elements against 16×16 for f32. Accelerate does run on that unit: with both clusters busy, six Accelerate threads and six `gemm_sme` threads together get no more than either group alone ([amx_sme_share](experiments/sme2-machine/README.md)).
- For production code on a Mac, call Accelerate. The hand-written kernels show how close one can get to it, and what each step contributes.

### Several SME units (`gemm_sme_mt`)

> **Command:** `./build/benchmarks/bench_gemm --benchmark_filter='^(SmeMt|Sme|Amx)/' --benchmark_repetitions=5 --benchmark_enable_random_interleaving=true` (wall-clock time for `SmeMt` and `Amx`)
> **Load:** load average 2.7–6, ZA shape 1×4. Median of 5 interleaved repetitions. Data: [sme-mt/default.json](experiments/sme-mt/default.json); thread sweep: [experiments/sme-mt](experiments/sme-mt/README.md)

`gemm_sme` is one thread, so one SME unit; the M4 Max has one per P-cluster,
two in all. `gemm_sme_mt` runs `gemm_sme`'s blocking on 8 threads: the B
panel is packed once and shared, and threads take chunks of A rows from an
atomic counter, so the work balances whichever cluster each thread lands
on (macOS cannot pin threads). Design and the thread sweep:
[kernels/sme.md § Several SME units](kernels/sme.md#several-sme-units).

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4000 | N=4096 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `gemm_sme` f32, 1 thread | 250 | 838 | 1,324 | 1,552 | 1,539 | 1,628 | 1,584 |
| `gemm_sme_mt` f32, 8 threads | 16 | 392 | 1,659 | 2,781 | 3,116 | **3,429** | **3,487** |
| Accelerate f32, all cores | 782 | 1,717 | 2,581 | **3,207** | 3,231 | — | 3,162 |
| `gemm_sme` f64, 1 thread | 82 | 323 | 405 | 435 | 423 | 431 | 427 |
| `gemm_sme_mt` f64, 8 threads | 7 | 291 | 626 | 761 | 896 | **903** | **910** |
| Accelerate f64, all cores | 325 | 463 | 784 | 830 | 788 | — | 794 |

- From N=2048 up, `gemm_sme_mt` matches or beats Accelerate with all cores:
  at N=4096, 110% in f32 and 115% in f64. Against the two units' measured
  peak (3.93 / 1.00 TFLOP/s) that is 89% / 91%, against 81% / 85% for
  `gemm_sme` on one unit.
- 2.2× `gemm_sme` in f32 and 2.1× in f64 at N=4096, a little more than two
  P-cluster units; the efficiency cluster has an SME unit too (180–265
  GFLOP/s FP32, [sme2-machine](experiments/sme2-machine/README.md)).
- At N=1024 it reaches 87% / 92% of Accelerate. Below N≈512 creating 8
  threads per call costs more than it saves (16 GFLOP/s at N=64), so small
  products should call `gemm_sme`. Why rows are handed out dynamically:
  [kernels/sme.md](kernels/sme.md#several-sme-units).

### Prefetch distance sweep

Benchmarks named `<Family>BlockedPf<D>/<prec>/N=<size>` sweep prefetch distance D ∈ {2, 4, 8, 16} (rows ahead for the scalar `BlockedPf`, R-row register-tile strips ahead for the SIMD families), with three `__builtin_prefetch` sites per kernel:

- [PF-A] `A(i + D×R, k_blk)` → L2 (read)
- [PF-B] `B(k_blk + TileK, 0)` → L2 (read), at the k-tile boundary
- [PF-C] `C(i + D×R, j_blk)` → L1 (write)

R is the register-tile height (1 for the scalar kernel).

GFLOP/s against the same kernel without prefetch, same run as the tables above (the `NeonBlockedPf` rows from the run of the NEON kernel, [`bench_m4_defaults.csv`](experiments/block-sweeps/bench_m4_defaults.csv)):

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

The accumulator count sets the ceiling. The P-core issues up to 4 FMAs per cycle with a ~3-cycle latency (`fma_peak`, [experiments/cpu-simd](experiments/cpu-simd/README.md)), so a micro-kernel needs 12–16 independent accumulators; the 6×4 tile has 24. The other limit is a single instruction: broadcasting A with `LD1R` holds tiles with many rows back, and loading it with `LDR` and multiplying by element makes the 6×4 tile 23–30% faster. f32 and f64 run at the same 85% of their peaks. Every shape measured: [kernels/neon.md](kernels/neon.md).

Peak per family on this machine:

| Kernel | f64 peak | f32 peak | f32/f64 |
|---|---|---|---|
| `gemm_naive` | 8.69 | 8.91 | 1.0× |
| `gemm_reordered` | 25.62 | 82.63 | 3.2× |
| `gemm_blocked` | 25.79 | 82.97 | 3.2× |
| `gemm_simd_blocked<Q>` | 62.25 | 123.78 | 2.0× |
| `gemm_sme` | 442 | **1,645** | 3.7× |
| `gemm_amx` (Accelerate) | **830** | **3,231** | 3.9× |

---
## AMD Zen 5 — AVX2 + AVX-512 (Linux / GCC)

> **Machine:** AMD Ryzen 9 9950X (Zen 5), 16C/32T, 5.71 GHz measured on one core under load, Ubuntu 24.04, GCC 13.3, C++20
> **Build:** `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_AVX512=ON` → `-O3 -march=native -ffast-math -funroll-loops`
> **CPU Caches:** L1d 48 KiB (×16) · L1i 32 KiB (×16) · L2 1024 KiB (×16) · L3 32768 KiB (×2)
> **Load Average:** 0.00 / 0.11 / 0.08 on 32 threads — idle
> **Governor:** `performance` · **Run time:** 2147 s (36 min) for the full CPU sweep

Zen 5 has a full-width 512-bit AVX-512 datapath (unlike Zen 4's double-pumped
256-bit), so this is the first machine in this repo where the AVX-512 kernels
run on native-width hardware. NEON, SME and AMX are absent and report
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
three runs with their default register tiles and cache blocks
([`bench_zen5_defaults.csv`](experiments/block-sweeps/bench_zen5_defaults.csv);
blocks: [kernels/blocks.md](kernels/blocks.md)): AVX2 384×256 × 1 KiB of
B's rows, AVX-512 512×256 × 2 KiB. `gemm_simd_blocked<Zmm>` uses an 8×48
f32 / 8×24 f64 register tile (8 rows × 3 vectors, 24 accumulators); every
tile shape and why 8 × 3: [kernels/avx512.md](kernels/avx512.md).
`gemm_simd_blocked<Ymm>` uses a 6×16 f32 / 6×8 f64 register tile (12
accumulators): [kernels/avx2.md](kernels/avx2.md).

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
inverts where it matters: at N=4096, blocked reaches 42.48 vs 30.48 GFLOP/s
(f32, 1.4×) and 33.02 vs 13.04 GFLOP/s (f64, 2.5×).

The register tile needs enough accumulators and register-only FMAs. Zen 5
retires 2 FMAs per cycle with a 4-cycle latency (`fma_peak`,
[experiments/cpu-simd](experiments/cpu-simd/README.md)): 362 GFLOP/s f32 and
180 f64 at 5.71 GHz with AVX-512, and 12 independent accumulators for the
full rate. With a 4×2-ZMM tile (8 accumulators) GCC also folds the B loads
into the FMAs as memory operands, 12 loads per k step for 8 FMAs, and the
kernel stops at 60% of peak. The 8×2 tile loads B once per k step and has
16 accumulators: 328.1 GFLOP/s f32 (91% of peak) and 160.5 f64 (89%) at
N=256. The 8×3 tile (24 accumulators) is as fast at N≤512 and holds up
better where B misses L1: +12% f32 at N=1024 and +29% at 4096, +13–31% f64
from N=512 up, at the same cache block
([kernels/avx512.md](kernels/avx512.md)). The default cache blocks add
+28% f32 and +50% f64 at N=4096 against 64-row blocks
([kernels/blocks.md](kernels/blocks.md)): the B block fits in half of L2
in either type, and each B block serves 512 rows of A. `gemm_simd_blocked<Ymm>`
uses a 6×2-YMM tile (12 accumulators, 15–16 of AVX2's 16 registers without
spilling): 171.7 f32 and 81.9 f64 at N=256, 94% and 90% of the AVX2 peak
(182 / 91 GFLOP/s); a 4×2 tile reaches 83% ([kernels/avx2.md](kernels/avx2.md)).

The drops with N follow the byte stride of B's rows, which the kernels read
in place. The micro-kernel runs over 256 rows of B; at a 4 KiB row stride
(f32 N=1024, f64 N=512) those rows fall into a few cache sets of L1 and L2
and stop staying cached. On L1-resident data, `simd_microkernel` measures a
1 KiB stride costing 4% and a 4 KiB stride 21% against packed B (AVX-512
f32, 16 accumulators). The AVX-512 kernel drops little at those sizes:
327.9 → 312.8 f32 at N=1024, and f64 holds 157.7 at N=512 with its smaller
B block; AVX2 drops by a quarter (169.7 → 126.9 f32, 81.9 → 62.3 f64).
Packing B into contiguous strips (`gemm_simd_packed`) removes it:
[experiments/packed-b](experiments/packed-b/README.md).

Software prefetch has almost no effect on Zen 5, and the distance is
irrelevant. Across every family, precision and size, all four distances
(D ∈ {2, 4, 8, 16}) land within ~3% of each other and of no prefetch (the
AVX-512 and AVX2 rows from the run of the default kernels,
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

> **Machine:** AMD Ryzen 9 9950X host + **NVIDIA GeForce RTX 5080** (Blackwell, sm_120), Ubuntu 24.04, CUDA 13.4, GCC 13.3, C++20
> **Build:** `cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a`
> **Run:** `./build/benchmarks/bench_gemm_cuda --benchmark_filter=-Var/ --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true`, median of 3, coefficient of variation ≤ 2.4%
> **Driver:** 615.71.09, host idle (load average 0.2). All CUDA tests passed with no skips. Tile sizes: `CudaParams` ([benchmarks/cuda_params.hpp](../benchmarks/cuda_params.hpp), sweeps in [docs/kernels/cuda.md](kernels/cuda.md)). Data: [experiments/cuda-machine/bench_gemm_cuda.json](experiments/cuda-machine/bench_gemm_cuda.json)

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
| `CudaNaive` | 1,931 | 3,623 | 3,704 | 3,314 | 3,388 |
| `CudaBlocked` | 1,930 | 3,317 | 3,385 | 3,385 | 3,389 |
| `CudaRegTile` | 730 | 14,928 | 24,315 | 24,943 | 24,239 |
| `CudaDoubleBuf` | 882 | 17,912 | 26,555 | 27,201 | 26,726 |
| `CudaVectorized` | 799 | 16,287 | 28,387 | **29,782** | 28,495 |
| `CudaWmma` | 1,930 | 14,812 | 22,201 | 21,184 | 17,610 |
| `CudaMmaLdmatrix` | 2,173 | 16,331 | 18,599 | 17,871 | 14,193 |
| `CudaWmmaPipelined` | 2,982 | 63,304 | 104,674 | 107,876 | 106,003 |
| `CudaTma` | 3,022 | 72,742 | 106,662 | **117,601** | **120,322** |
| `CudaCublas` (ref) | 3,867 | 26,484 | 38,177 | 38,734 | 38,892 |
| `CudaCublasTf32` (ref) | 4,002 | 44,647 | 56,916 | 60,110 | 60,470 |
| `CudaCublasFp16` (ref) | 5,192 | 83,668 | 114,979 | 118,275 | 121,231 |

double (f64)

| Kernel | N=256 | N=1024 | N=4096 | N=8192 |
|---|--:|--:|--:|--:|
| `CudaNaive` | 643 | 919 | 931 | 932 |
| `CudaBlocked` | 669 | 954 | 956 | 956 |
| `CudaRegTile` | 171 | 729 | 961 | 963 |
| `CudaDoubleBuf` | 179 | 736 | **965** | **966** |
| `CudaVectorized` | 169 | 728 | 951 | 953 |
| `CudaCublas` (ref) | 595 | 648 | 848 | 848 |

- f64: every kernel, cuBLAS included, converges on 0.85–0.97 TFLOP/s. That
  is the FP64 rate of consumer Blackwell, a small fraction of its FP32 rate,
  so the f64 ladder shows the FP64 units, not the kernels.
- N=256 is too small for the tiled kernels: a 128×128 block tile gives 4
  thread blocks for the GPU's 84 SMs, so `CudaNaive` (16×16 tiles, 256
  blocks) beats them there. From N=1024 the ranking settles.
- Every level's tile sizes come from a sweep ([docs/kernels/cuda.md](kernels/cuda.md)).

### Level by level, f32, N=4096

Levels 0–4 run on the FMA cores, 5–8 on the Tensor Cores.

| Level | Kernel | GFLOP/s | ×Naive | Note |
|---|---|--:|--:|---|
| 0 | `CudaNaive` | 3,704 | 1.0× | Coalesced B loads and L1/L2 reuse make this baseline strong. |
| 1 | `CudaBlocked` | 3,385 | 0.91× | One output per thread: two `__syncthreads()` per k-tile pay for only 16 FMAs. |
| 2 | `CudaRegTile` | 24,315 | 6.6× | 64 outputs per thread from 16 shared-memory reads per k step; the largest single step. |
| 3 | `CudaDoubleBuf` | 26,555 | 7.2× | The copy of the next k-tile overlaps the FMAs. |
| 4 | `CudaVectorized` | 28,387 | 7.7× | 128-bit loads into shared memory and 128-bit reads out of it, 16 × 8 outputs per thread: the fastest FMA kernel, 73–77% of cuBLAS SGEMM from N = 4096. |
| 5 | `CudaWmma` | 22,201 | 6.0× | Tensor Cores, but below Levels 3–4: one fragment per warp and a single buffer leave too little work per synchronisation. |
| 6 | `CudaMmaLdmatrix` | 18,599 | 5.0× | The same tiling through PTX; below Level 5 at this size: dropping to PTX alone does not remove the bottleneck. |
| 7 | `CudaWmmaPipelined` | 104,674 | 28.3× | 128×128 tiles, 64×32 per warp, `cp.async` double buffering: 4.7× Level 5. |
| 8 | `CudaTma` | 106,662 | 28.8× | TMA copies into swizzled shared memory, a producer warp and `mbarrier` pipeline, 64×64 warp tiles: a third of Level 7's instructions; 99% of cuBLAS FP16 from N=8192. |

### Against cuBLAS

TFLOP/s, ascending:

| Path | Precision | N=4096 | N=8192 | N=16384 |
|---|---|--:|--:|--:|
| `gemm_cuda_vectorized` | FP32 on the FMA cores, hand-written | 28.4 | 29.8 | 28.5 |
| `cublasSgemm` | FP32 on the FMA cores | 38.2 | 38.7 | 38.9 |
| `cublasGemmEx` TF32 | TF32 Tensor Cores, 10-bit mantissa | 56.9 | 60.1 | 60.5 |
| `gemm_cuda_wmma_pipelined` | FP16 Tensor Cores, hand-written | 104.7 | 107.9 | 106.0 |
| **`gemm_cuda_tma`** | **FP16 Tensor Cores, hand-written (Level 8)** | **106.7** | **117.6** | **120.3** |
| `cublasGemmEx` FP16 | FP16 Tensor Cores, fp32 accumulate | 115.0 | 118.3 | **121.2** |

- 100+ TFLOP/s on this GPU needs dense FP16 Tensor Cores: plain FP32, the
  ceiling for every non-Tensor-Core kernel, tops out at ~39 TFLOP/s, and
  TF32 at ~61.
- The hand-written kernels reach 77% of cuBLAS on the FMA cores (N=8192)
  and, with Level 8, 99% on the Tensor Cores (N=8192 and 16384; Level 7:
  87–91%).

### Level 7: shared-memory bank conflicts

Without padding its shared-memory tiles, `gemm_cuda_wmma_pipelined` runs at
80.8 TFLOP/s, 68% of cuBLAS FP16 (profiled with 32 × 64 warp tiles, at 126
registers per thread). Padding each row by 8 halves (why 8, and why
not a swizzle: the Level 7 comment in
[gemm_kernels.cuh](../src/cuda/gemm_kernels.cuh)) changes, per Nsight Compute
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

Unpadded, the kernel needs about 6.6× as many shared-load wavefronts;
these count bank-service work, not 6.6× as many requested data bytes;
padded, it is within 1% of the minimum wavefront count. Shared memory
per block grows from 32 to 37 KB, lowering the shared-memory occupancy
limit from 3 to 2 blocks per SM, which costs nothing: its registers
already cap it at 2. That register pressure (occupancy 33%, ~67%
unused warp slots per Nsight) constrains occupancy; it does not by itself
prove the remaining performance bottleneck. Deeper pipelines and split-K
are possible experiments, not measured fixes for the remaining gap.

---

## Apple M4 Max GPU — Metal

> **Machine:** Apple M4 Max, 40-core GPU, macOS 26.3.1, Apple Clang 17, Metal shaders compiled at run time with fast math
> **Build:** default (`HPC_ENABLE_METAL` is on on Apple); `cmake -B build -DCMAKE_BUILD_TYPE=Release`
> **Command:** `./build/benchmarks/bench_gemm_metal --benchmark_repetitions=5 --benchmark_min_time=1s --benchmark_enable_random_interleaving=true --benchmark_report_aggregates_only=true`
> **Conditions:** AC power, load average 0.8–4. Median of 5 interleaved repetitions; cv up to 9% per row. Data: [metal-m4/bench_gemm_metal.json](experiments/metal-m4/bench_gemm_metal.json); method: [experiments/metal-m4](experiments/metal-m4/README.md)

What each level adds: [src/gemm/metal.hpp](../src/gemm/metal.hpp). Apple
GPUs have no FP64, so every row is f32 in and out; the F16 rows multiply
fp16 copies of A and B, converted once before timing, like the CUDA fp16
rows. **What is timed** is the GEMM alone, as for CUDA: inputs sit in
GPU-visible memory before timing, and the time is GPU time from the command
buffer's start and end timestamps. Each command buffer holds enough GEMMs
to last ≥ ~5 ms (93–256 at N=256), since at small N one GEMM is shorter than the
command buffer's fixed cost.

**Read across rows, not across runs.** On this laptop the absolute GPU
rate varies by up to ~3× between runs (`MetalMps` f32 at N=4096: 13.1
TFLOP/s in this run, as low as 4.5 in others), while the register-only FMA
peak stays at 14.2–14.9
([the data README](experiments/metal-m4/README.md#clock-and-power)).
The rows below come from one run, interleaved, so they compare fairly with
each other.

### GFLOP/s by kernel and size

| Level | Kernel | N=256 | N=1024 | N=2048 | N=4096 | N=8192 |
|--:|---|--:|--:|--:|--:|--:|
| 0 | `MetalNaive` | 961 | 1,337 | 1,350 | 1,242 | 866 |
| 1 | `MetalTiled` (32 × 32) | 1,494 | 2,353 | 2,429 | 2,418 | 2,341 |
| 2 | `MetalRegTile` | 1,241 | 6,369 | 7,349 | 7,547 | 7,446 |
| 3 | `MetalSimd` | 2,058 | 10,977 | 12,347 | 12,354 | 6,065 |
| 4 | `MetalSimdTg` | 1,497 | 9,875 | 12,129 | 11,842 | 11,381 |
| 5 | `MetalSimdTgF16` (fp16 in) | 1,614 | 10,717 | 13,376 | 13,577 | 13,503 |
| ref | `MetalMps` (MPSMatrixMultiplication) | 3,540 | 12,373 | 13,042 | 13,066 | 12,558 |
| ref | `MetalMpsF16` (fp16 in, fp32 out) | 5,113 | **13,766** | **14,131** | **14,369** | **13,780** |

Measured FMA peak in the same run: **14.9 TFLOP/s f32** (40 cores × 128
lanes × 2 at ~1.45 GHz) and 15.8 f16: fp16 FMA runs at the f32 rate.

- Level by level at N=2048: tiling in threadgroup memory 1.8×, the register
  tile 3.0× on top (8×4 outputs per thread), and simdgroup matrices another
  1.7×, to 83% of the f32 peak (`MetalSimd`, 12.3 TFLOP/s). The register
  tile is the largest single step, as Level 2 is on the RTX 5080.
- Simdgroup matrices give 1.7× over the best SIMT kernel although on M1–M4
  they run on the same FP32 ALUs: there is no separate matrix unit (fp16 FMA
  runs at the f32 rate). Each 8×8 fragment load feeds 8 multiply-adds per
  element instead of one, so fewer instructions are spent on operands.
- Staging A and B in threadgroup memory (Level 4) is 2–10% slower than
  loading fragments straight from device memory (Level 3) from N=1024 to
  4096; the GPU's caches already serve the reuse between simdgroups. It
  wins at N=8192 (11.4 against 6.1 TFLOP/s), where the operands no longer
  stay cached.
- fp16 inputs (Level 5) gain 10–19% over Level 4 from N=2048, from halving
  the bytes moved, not from faster arithmetic.
- Against MPS: the best f32-input kernel reaches 95% / 95% / 91% of
  `MetalMps` at N = 2048 / 4096 / 8192; Level 5 (fp16 inputs) 103–108% of
  `MetalMps` and 94–98% of `MetalMpsF16`. MPS with fp16 inputs is the
  fastest row, 14.4 TFLOP/s at N=4096, 97% of the peak measured in the same
  run.
- At N=256, MPS is 1.7–3.7× faster than every hand-written level: a 64×64
  tile gives 16 threadgroups for 40 GPU cores.

---

## Low precision: bf16, fp16, INT8, FP8

Analysis in [part 5](articles/lowp-gemm.md); raw data, machines and method
in [experiments/lowp](experiments/lowp/README.md). Inputs are converted to
the format before timing; every rate counts a multiply-add as 2 operations,
INT8 included. Medians of 3.

Apple M4 Max, one P-core / one SME unit, GFLOP/s or GOP/s
(`bench_gemm_lowp`, all rows one session, load average 3–4,
median of 3 interleaved repetitions,
[`bench_lowp_m4.json`](experiments/lowp/bench_lowp_m4.json); tiles:
SME 1 × 4 ZA tiles, NEON FMLAL 4 × 6, FP16-accumulate 4 × 5,
[sizing](kernels/sizing.md)):

| Kernel | Format | N=256 | N=1024 | N=2048 | N=4096 |
|---|---|--:|--:|--:|--:|
| `Sme` | f32 | 767 | 1,575 | 1,584 | 1,614 |
| `SmeWidening` | fp16 | 969 | 1,751 | 1,842 | 1,854 |
| `SmeWidening` | bf16 | 972 | 1,752 | 1,848 | 1,853 |
| `SmeWidening` | int8 | 1,604 | 3,377 | 3,676 | 3,726 |
| `NeonBlocked` | f32 | 122 | 121 | 118 | 100 |
| `NeonBfmmla` | bf16 | 65 | 65 | 66 | 66 |
| `NeonFmlal` | fp16 | 117 | 122 | 123 | 123 |
| `NeonFp16Acc` | fp16 | 213 | 235 | 236 | 233 |
| `NeonSmmla` | int8 | 444 | 490 | 501 | 484 |

Both SME units (`SmeMt`, `SmeWideningMt`, 8 threads, wall-clock time, same
run): f32 2,747 / 3,487, fp16 2,838 / 3,858, bf16 2,833 / 3,839, INT8
4,995 / 7,214 at N = 1024 / 4096.

AMD Zen 5, one core, GFLOP/s or GOP/s (`bench_gemm_lowp`, pinned to core 2;
all rows one session,
[`bench_lowp_zen5.csv`](experiments/lowp/bench_lowp_zen5.csv)):

| Kernel | Format | N=256 | N=1024 | N=2048 | N=4096 |
|---|---|--:|--:|--:|--:|
| `Avx512Blocked` | f32 | 325 | 311 | 302 | 295 |
| `Avx512Dpbf16` | bf16 | 619 | 638 | 606 | 574 |
| `Avx512Vnni` | int8 | 824 | 1,138 | 1,138 | 1,038 |

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
