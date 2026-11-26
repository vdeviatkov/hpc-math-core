# Zen 5 hardware counters

Cache, TLB and load-port measurements behind the Zen 5 explanations in
[part 1](../../articles/simd-gemm.md), [part 3](../../articles/sme2-cache-associativity.md#the-same-experiment-on-amd-zen-5),
[docs/kernels/avx512.md](../../kernels/avx512.md) and [avx2.md](../../kernels/avx2.md).

AMD Ryzen 9 9950X (Zen 5), Ubuntu 24.04, Linux 7.0, GCC 13.3 (Clang 18.1.3
for `fold/`), `-O3 -march=native -ffast-math -funroll-loops` unless noted,
one thread pinned to core 2 (`taskset -c 2`), idle machine, perf 7.0.14 with
`kernel.perf_event_paranoid = 1`. Counts are per 1,000 retired
FP operations (`fp_ret_sse_avx_ops.all`), so runs of different lengths
compare; five or six events per run (the core has six counters; perf
scales any multiplexed event).

| File | Program | What |
|---|---|---|
| [`b_stride.txt`](b_stride.txt) | `avx512_b_stride run N ldb 0 reps` | B rows at a 4 KiB-multiple stride against + 64 B, N = 1024, 2048, 4096 (8 × 2 tile, 64 × 256 × 512 blocks): L1D fills by source, L2 requests and misses, store-forwarding and TLB candidates at N = 1024, and throughput |
| [`height.csv`](height.csv) | `avx512_b_stride run 512 2048 1024 ldb kc 0 10` | the strip-height experiment's configurations: L1 misses to L2, L2 demand misses, fills from L3 and TLB misses for strips of 16, 64, 256 and 512 rows at 4, 8, 16 KiB and each + 64 B |
| [`third_vector.txt`](third_vector.txt) | `bench_gemm` `Avx512Tile8x2` / `8x3` | L1 misses to L2, L2 demand misses, L1 TLB misses, load-queue dispatch stalls; FP32 N = 1024, 4096, FP64 N = 512, 4096 |
| [`packed_b.txt`](packed_b.txt) | `bench_gemm` `*BlockedCmp` / `*PackedCmp` | blocked against packed B, AVX-512 at N = 2000, 2048, 4096 and AVX2 at 2048 |
| [`avx2_3x4.txt`](avx2_3x4.txt) | `bench_gemm` 16-register build, `Avx2Tile6x2` / `3x4` / `4x3` | why 3 × 4 leads at N = 4096: misses, TLB, load queue, L2 prefetches |
| [`load_ports.csv`](load_ports.csv) | [`x86_load_ports`](../../../benchmarks/experiments/x86_load_ports.cpp) | independent L1 vector loads per cycle at 128, 256, 512 bits, and with FMAs |
| [`fold/`](fold/README.txt) | GCC 13 and Clang 18 builds | which compilers read B inside the FMAs, per AVX-512 shape, and the throughput of each build |
| [`pstat.sh`](pstat.sh) | helper | one `perf stat` line per configuration |

## Results

- **B stride.** At the 256-row strips the kernels use, a 4 KiB-multiple
  stride raises L2 demand misses 1.6–1.7× (N = 1024–4096), served from L3;
  L1 misses rise only from N = 2048 (+9–14%). Strips of ≤ 64 rows lose to
  L1 conflicts instead (L1 misses +44–79%). Store-forwarding, TLB and
  load-queue counts do not differ between the strides.
- **Third vector (8 × 3 against 8 × 2).** L1 TLB misses −35 to −39%,
  load-queue stall cycles −55 to −69%; L1 misses unchanged at FP32
  N = 1024 and FP64 N = 512, −14 to −18% at N = 4096.
- **Packed B.** AVX2: L1 misses 2.4× and TLB misses 10× lower packed.
  AVX-512: L1 and L2 misses about equal, TLB misses 10× lower, at N = 2000
  as at the powers of two.
- **AVX2 3 × 4 at N = 4096.** 4.7% faster than 6 × 2 with 22% more L1
  misses, the same TLB misses and 2.6× the L2 prefetches: these counters
  do not show where its lead comes from.
- **Load ports.** 2.0 vector loads per cycle at every width; 1.92 loads
  and 1.92 FMAs per cycle mixed.
- **Folding.** GCC 13 reads B inside every FMA for 4 × 2, 2 × 4, 4 × 4;
  Clang 18 never does. Clang's 4 × 2 is 31% faster than GCC's at N = 256.

## Later additions

- **`gemm_blocked` (part 1, Step 3) at N = 256** (`blocked_vs_reordered.txt`):
  0.442 instructions per FLOP against the reordered loop's 0.157 (2.8×) at
  IPC 3.90 against 2.78, with the same L1 misses (32.9 against 31.6 per
  1,000 FLOP): the blocked loop is bound by the instructions of its short
  64-element inner loops, not by memory.
- **AVX-512 INT8 against bf16** (`lowp_mix.txt`, N = 1024, normalised by
  the benchmark's operation count): INT8 runs 1.56 other instructions per
  VPDPBUSD, bf16 0.95 per VDPBF16PS; at IPC 4.11 and 3.59 that is 1.60 and
  1.84 multiply-accumulates per cycle, 80% and 92% of the two-per-cycle
  peak.
- **bf16 / INT8 at N = 4000 against 4096**
  ([`../lowp/sweeps-zen5/stride_4000_vs_4096.txt`](../lowp/sweeps-zen5/stride_4000_vs_4096.txt)):
  bf16 627.6 / 619.7 / 596.5 GFLOP/s at N = 2048 / 4000 / 4096, INT8 1,186 /
  1,137 / 1,073 GOP/s, FP32 303.2 / 309.9 / 294.3.
