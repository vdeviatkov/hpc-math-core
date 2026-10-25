# Cache blocks of the SIMD kernels

Around the register tile ([AVX2](avx2.md), [AVX-512](avx512.md),
[NEON](neon.md)), `gemm_<isa>_blocked` blocks the multiplication for the
caches. All three block sizes are template parameters:

```cpp
gemm_avx512_blocked<T, R, V, TileM, TileK, TileN>(A, B, C);
```

| Family | Default block, TileM × TileK × TileN | B block (TileK × TileN) | Machine |
|---|---|---|---|
| AVX2 | 384 × 256 × 256 FP32, 384 × 256 × 128 FP64 | 256 KiB | AMD Zen 5 |
| AVX-512 | 512 × 256 × 512 FP32, 512 × 256 × 256 FP64 | 512 KiB | AMD Zen 5 |
| NEON | 384 × 256 × 256 | 256 KiB FP32, 512 KiB FP64 | Apple M4 Max |

Until 2026-10-22 every family used 64 × 256 × 256 (AVX-512: × 512), sizes
no measurement had chosen. The sweep below replaced them: at N = 4096 the
new blocks are 10–12% faster for AVX2 and NEON, 28% for AVX-512 FP32 and
51% for AVX-512 FP64; at N ≤ 1024 they change little, except AVX-512 FP64
(+22–23%).

## What each size decides

The driver ([`blocked_driver.hpp`](../../src/gemm/blocked_driver.hpp)) runs

```text
for each block of TileM rows of A            (i-block)
  for each block of TileK steps of k          (k-block)
    for each block of TileN columns of B      (j-block: the B block)
      for each strip of R rows in the i-block
        for each tile of V vectors in the j-block
          micro-kernel: C(R × V·W) += A(R × TileK) · B(TileK × V·W)
```

- **TileK × TileN, the B block,** is read by every R-row strip of the
  i-block, so it should stay in L2 for the whole i-block.
- **TileM** is how many strips share one trip of the B block into L2.
  Every B block is fetched again for each i-block: B travels N / TileM
  times from beyond L2. With 64 rows that is 64 times at N = 4096, where B
  (64 MiB FP32) fits no cache.
- **TileK** is also the length of the A strip, R × TileK, which stays in
  L1 for all the j-calls of a strip, and of the B strip each call streams.

## Sweep

`bench_gemm`'s `<Isa>Block<M>x<K>x<N>` benchmarks: the library kernel at
its default register tile with every block in a grid of 7 TileM values
(AVX2, NEON: 48, 64, 96, 192, 384, 768, 1536; AVX-512: 32 … 2048) × TileK
and TileN ∈ {128, 256, 512, 1024}, at N = 512, 1024 and 4096, both types:
1,344 measurements per family. Stage 1 ran the grid once; stage 2 re-ran
the best candidates and the old default three times, in interleaved order
([data](../experiments/block-sweeps/README.md)). GFLOP/s.

### TileM: the larger, the less B traffic, up to a plateau

TileK = 256 and the default TileN; Zen 5 from stage 1 (one run;
repetitions on this machine vary by < 1%), M4 three runs:

AVX-512, Zen 5 (TileN 512 FP32, 256 FP64):

| TileM | FP32 N = 512 | 1024 | 4096 | FP64 N = 512 | 1024 | 4096 |
|--:|--:|--:|--:|--:|--:|--:|
| 32 | 326.0 | 306.5 | 182.4 | 155.6 | 150.7 | 90.0 |
| 64 | 327.5 | 310.7 | 228.2 | 157.3 | 151.7 | 113.9 |
| 128 | 327.9 | 310.6 | 262.4 | 157.4 | 151.5 | 130.7 |
| 256 | 327.4 | 311.1 | 283.2 | 157.6 | 151.7 | 140.4 |
| **512** | 328.2 | 311.2 | 293.8 | 157.9 | 150.6 | 143.8 |
| 1024 | 327.4 | 311.5 | 297.1 | 156.4 | 150.3 | 143.2 |
| 2048 | 327.6 | 311.8 | 286.0 | 156.7 | 145.9 | 141.5 |

AVX2, Zen 5 (TileN 256 FP32, 128 FP64):

| TileM | FP32 N = 512 | 1024 | 4096 | FP64 N = 512 | 1024 | 4096 |
|--:|--:|--:|--:|--:|--:|--:|
| 48 | 167.9 | 128.1 | 92.5 | 62.9 | 55.5 | 46.9 |
| 64 | 166.3 | 125.3 | 94.2 | 61.8 | 54.2 | 47.8 |
| 96 | 168.0 | 127.6 | 99.0 | 62.4 | 54.9 | 50.3 |
| 192 | 168.0 | 126.9 | 102.4 | 62.0 | 54.6 | 52.1 |
| **384** | 167.9 | 126.5 | 104.3 | 62.2 | 54.7 | 53.5 |
| 768 | 170.2 | 127.5 | 106.5 | 62.3 | 54.6 | 54.2 |
| 1536 | 169.7 | 127.0 | 106.5 | 62.2 | 54.4 | 54.3 |

NEON, M4 (TileN 256):

| TileM | FP32 N = 512 | 1024 | 4096 | FP64 N = 512 | 1024 | 4096 |
|--:|--:|--:|--:|--:|--:|--:|
| 48 | 118.8 | 117.0 | 86.2 | 59.2 | 56.5 | 44.1 |
| 64 | 119.4 | 114.5 | 89.1 | 59.0 | 56.1 | 44.0 |
| 96 | 118.4 | 117.2 | 94.0 | 58.6 | 55.4 | 47.2 |
| 192 | 118.3 | 116.5 | 96.9 | 59.6 | 56.9 | 48.5 |
| **384** | 118.9 | 117.3 | 98.5 | 58.9 | 57.9 | 49.3 |
| 768 | 118.7 | 118.4 | 99.4 | 58.4 | 56.4 | 49.0 |
| 1536 | 119.5 | 117.9 | 98.9 | 59.5 | 56.5 | 49.4 |

TileM changes nothing at N ≤ 1024: B (4 MiB FP32 at N = 1024) fits Zen 5's
32 MiB L3 and the M4's caches, so fetching it again costs little. At
N = 4096 it does not fit, and every doubling of TileM halves the B
traffic from memory, until the gain flattens: from 384–512 rows on, within
2–3% of the best. The defaults take the smallest TileM on the plateau
(384 is 64 strips of 6 rows; 512 is 64 strips of 8). A taller block also
costs nothing at small N: a block larger than the matrix is clipped to it.

### TileK × TileN: the B block must fit in half of Zen 5's L2

Each (TileK, TileN) at TileM = 512 (AVX-512) or 384 (AVX2), stage 1,
grouped by the size of the B block, GFLOP/s at N = 1024 / 4096:

| B block | AVX-512 FP32 | AVX-512 FP64 | AVX2 FP32 | AVX2 FP64 |
|--:|---|---|---|---|
| 128 KiB | 292–296 / 274–275 | 140 / 132 | 125 / 105 | 54 / 53 |
| 256 KiB | 298–307 / 279–289 | 146–147 / 138–139 | 125–126 / 104–105 | 54–55 / 53 |
| 512 KiB | 303–311 / 285–294 ¹ | 146–151 / 139–144 | 125–126 / 104 ¹ | 54 / 53 |
| 1 MiB | 253–272 / 229–247 | 118–134 / 110–124 | 97–101 / 77–82 | 40–44 / 39–43 |

¹ Without TileK = 1024, below.

Up to 512 KiB the B block stays in Zen 5's 1 MiB L2 next to A's strips and
C's rows; at 1 MiB it does not, and the kernel loses 11–27% at every
TileK × TileN that makes 1 MiB. So the best TileN depends on the element
size: 512 FP32 columns but 256 FP64 columns for AVX-512 (2 KiB of each B
row), 256 FP32 and 128 FP64 for AVX2 (1 KiB). The old AVX-512 FP64 block,
256 × 512 doubles, was exactly 1 MiB; that alone cost FP64 22–23% at
N = 512–1024 (157.7 / 151.0 now, against 128.4 / 123.4).

TileK = 1024 is slow even with a small B block (AVX-512 FP32, 1024 × 128:
254 / 228): the A strip, 8 × 1024 × 4 = 32 KiB, no longer leaves room in
the 48 KiB L1 for the B lines each call streams. TileK = 256 is the best
or within 2% of it in every family.

A few blocks far larger than L2 do well at N = 4096 alone (AVX2 FP32
512 × 1024, 2 MiB: 116 GFLOP/s at N = 4096, but 72 at N = 1024); the sweep
does not explain them, and they lose at every other size.

On the M4, whose L2 is 16 MiB per cluster, the B block size makes no
difference up to 1 MiB (NEON, TileM = 384, three runs):

| TileN | FP32 N = 512 | 1024 | 4096 | FP64 N = 512 | 1024 | 4096 |
|--:|--:|--:|--:|--:|--:|--:|
| 128 | 120.0 | 114.2 | 95.9 | 58.4 | 56.7 | 49.0 |
| **256** | 118.9 | 117.3 | 98.5 | 58.9 | 57.9 | 49.3 |
| 512 | 120.3 | 118.4 | 98.9 | 59.4 | 56.9 | 48.8 |
| 1024 | 120.6 | 115.7 | 98.3 | 58.3 | 56.3 | 48.7 |

| TileK | FP32 N = 512 | 1024 | 4096 | FP64 N = 512 | 1024 | 4096 |
|--:|--:|--:|--:|--:|--:|--:|
| 128 | 117.7 | 112.2 | 94.3 | 56.8 | 56.6 | 46.5 |
| **256** | 118.9 | 117.3 | 98.5 | 58.9 | 57.9 | 49.3 |
| 512 | 120.8 | 117.8 | 96.8 | 59.7 | 55.8 | 45.8 |
| 1024 | 121.7 | 116.9 | 55.8 | 59.2 | 52.0 | 27.6 |

What fails on the M4 is TileK = 1024 at N = 4096 only: a micro-kernel call
then walks 1,024 rows of B 16 KiB apart, the strip height at which
[Part 3](../articles/sme2-cache-associativity.md) measured the M4's L2
dropping B rows to set conflicts. This is consistent with that
measurement, not separately verified.

### Stage 2: the candidates, three runs each

Interleaved, so every candidate sees the same machine state:

AVX2, Zen 5:

| Block M × K × N | FP32 N = 512 | 1024 | 4096 | FP64 N = 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 64 × 256 × 256 (old) | 166.9 | 125.4 | 94.7 | 61.4 | 54.1 | 47.8 |
| 384 × 256 × 256 (FP32 default) | 169.0 | 127.0 | 105.1 | 61.7 | 54.5 | 53.4 |
| 768 × 256 × 256 | 169.4 | 126.8 | 105.9 | 61.7 | 54.0 | 53.7 |
| 384 × 256 × 128 (FP64 default) | 168.5 | 125.2 | 105.2 | 62.1 | 54.7 | 53.5 |
| 768 × 256 × 128 | 168.4 | 125.0 | 106.4 | 62.1 | 54.4 | 53.9 |

AVX-512, Zen 5:

| Block M × K × N | FP32 N = 512 | 1024 | 4096 | FP64 N = 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 64 × 256 × 512 (old) | 326.5 | 307.8 | 229.1 | 151.5 ² | 123.5 | 95.9 |
| 512 × 256 × 512 (FP32 default) | 327.0 | 297.8 ² | 293.7 | 136.2 | 131.5 | 120.9 |
| 1024 × 256 × 512 | 327.3 | 298.6 | 296.0 | 135.2 | 131.2 | 121.2 |
| 512 × 256 × 256 (FP64 default) | 319.1 | 299.2 | 281.8 | 157.4 | 150.3 | 143.5 |
| 1024 × 256 × 256 | 319.9 | 299.8 | 283.3 | 157.3 | 150.1 | 144.0 |

² The 4 KiB B-row stride sizes (FP32 N = 1024, FP64 N = 512) depend on
where the allocator puts the matrices, and the interleaved order changes
that: the same old block measured 128 FP64 at N = 512 in the earlier
runs, and the new FP32 default 312.8 at N = 1024 in its own run (below).

NEON, M4 (load average 3–7):

| Block M × K × N | FP32 N = 512 | 1024 | 4096 | FP64 N = 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|--:|--:|
| 64 × 256 × 256 (old) | 124.4 | 122.2 | 91.5 | 60.3 | 57.6 | 45.9 |
| **384 × 256 × 256** (default) | 122.6 | 121.5 | 100.8 | 60.6 | 57.6 | 49.9 |
| 384 × 256 × 128 | 121.4 | 121.3 | 99.9 | 60.7 | 58.2 | 50.2 |
| 384 × 256 × 512 | 122.6 | 117.5 | 99.9 | 61.0 | 58.9 | 50.6 |
| 384 × 128 × 128 | 119.6 | 117.6 | 98.0 | 59.1 | 58.0 | 47.3 |
| 192 × 512 × 1024 | 123.7 | 121.4 | 93.4 | 61.1 | 58.3 | 45.6 |

Where candidates are within 1–2% of each other the default is the smaller
block. For NEON one TileN serves both types: on the M4 the B block size
does not matter.

## The kernels with the new blocks

The library kernels at their defaults, three runs each
([`bench_zen5_defaults.csv`](../experiments/block-sweeps/bench_zen5_defaults.csv),
[`bench_m4_defaults.csv`](../experiments/block-sweeps/bench_m4_defaults.csv)),
against the same kernels with the old blocks in their last runs:

| GFLOP/s | N = 64 | 256 | 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|--:|
| AVX-512 FP32, old blocks | 297.2 | 325.5 | 329.0 | 312.2 | 230.2 |
| AVX-512 FP32, new | 297.7 | 324.4 | 327.9 | 312.8 | **294.7** |
| AVX-512 FP64, old blocks | 149.9 | 162.1 | 128.4 | 123.4 | 95.9 |
| AVX-512 FP64, new | 153.0 | 162.0 | **157.7** | **151.0** | **145.0** |
| AVX2 FP32, old blocks | 169.0 | 171.2 | 167.7 | 124.0 | 95.4 |
| AVX2 FP32, new | 166.7 | 171.7 | 169.7 | 126.9 | **105.6** |
| AVX2 FP64, old blocks | 82.8 | 83.8 | 62.3 | 54.3 | 48.1 |
| AVX2 FP64, new | 82.8 | 81.9 | 62.3 | 54.8 | **53.8** |
| NEON FP32, old blocks | 121.9 | 121.8 | 122.7 | 121.0 | 91.9 |
| NEON FP32, new | 122.4 | 122.9 | 123.8 | 121.2 | **102.1** |
| NEON FP64, old blocks | 61.3 | 61.3 | 60.3 | 58.2 | 45.8 |
| NEON FP64, new | 62.1 | 62.3 | 61.0 | 59.5 | **51.0** |

What remains at large N is mostly B's row stride
([Part 3](../articles/sme2-cache-associativity.md)): AVX2 FP32 and FP64
still drop at the 4 KiB stride (N = 1024 FP32, 512 FP64). Packing B into
contiguous strips, as the SME2 kernel does, is the next step: below.

## Packed B

`gemm_<isa>_packed<T, R, V, Mc, Kc, Nc, PackA = false>`
([`packed_driver.hpp`](../../src/gemm/packed_driver.hpp)) runs the same
micro-kernel and register tile in GotoBLAS order:

```text
for each block of Nc columns of B                (jc; rounded down to whole strips)
  for each block of Kc steps of k                 (pc)
    pack B(pc.., jc..) into strips V·W columns wide, kc rows each, contiguous
    for each block of Mc rows of A                (ic)
      for each strip                              (jr)
        for each tile of R rows                   (ir)
          micro-kernel: C(R × V·W) += A(R × kc) · strip(kc × V·W), ldb = V·W
```

The B block is copied once and serves all of A, so B is read from memory
once per pc pass instead of once per block of A; within a strip, row k
and row k + 1 are V·W elements apart instead of N, so a power-of-two N no
longer folds the strip onto a few cache sets. The order of the inner two
loops is swapped against the blocked driver: one strip is reused by every
R-row tile of the A block.

### Blocks

Swept like the blocked kernels, Mc ∈ {64, 128, 256, 512} (NEON: up to
1024) × Kc ∈ {128, 256, 512} (NEON: 256, 512, 1024) × Nc ∈ {512, 1024,
2048}, N = 1024, 2000, 4096
([data](../experiments/packed-b/README.md)):

| Family | Default Mc × Kc × Nc | Notes |
|---|---|---|
| AVX2 | 256 × 256 × 512 | the B strip, 256 × 16 floats, is 16 KiB: in L1 |
| AVX-512 | 128 × 512 × 512 | a 512 × 48-float strip is 96 KiB, twice the L1, yet Kc = 512 beats 256 by ~2%: contiguous rows stream from L2 |
| NEON | 256 × 512 × 512 | Mc 256–1024, Nc 512–1024 all within 3% on a quiet M4 |

The Zen 5 grids are flat near the top: the best ten blocks of each family
and type lie within 2.5% (geometric mean over N). One outlier: NEON FP64 with 1024 × 512 × 2048
(a 4 MiB A block and an 8 MiB B block, 12 of the 16 MiB L2) collapsed to
17 GFLOP/s at N = 4096 in the one-run M4 grid, which ran under load.

### Packed against in place

Medians, GFLOP/s (Zen 5: 3 runs; M4: 5 runs, interleaved):

| | N = 1000 | 1024 | 2000 | 2048 | 4000 | 4096 |
|---|--:|--:|--:|--:|--:|--:|
| AVX-512 FP32, in place | 250.6 | 311.5 | 308.2 | 301.6 | 309.8 | 294.2 |
| AVX-512 FP32, packed | **288.7** | 315.7 | 320.9 | 318.2 | 322.5 | **318.7** |
| AVX-512 FP64, in place | 152.2 | 149.8 | 150.5 | 144.7 | 149.9 | 142.4 |
| AVX-512 FP64, packed | 155.6 | 154.5 | 155.0 | 154.3 | 154.2 | **152.2** |
| AVX2 FP32, in place | 168.4 | 126.3 | 163.9 | 105.2 | 163.0 | 104.2 |
| AVX2 FP32, packed | 167.5 | **166.1** | 168.7 | **166.5** | 169.4 | **166.9** |
| AVX2 FP64, in place | 83.9 | 54.7 | 81.0 | 53.4 | 81.5 | 53.6 |
| AVX2 FP64, packed | 83.2 | **81.8** | 82.9 | **82.0** | 82.6 | **81.3** |
| NEON FP32, in place | 120.8 | 121.5 | 118.7 | 117.6 | 114.1 | 101.6 |
| NEON FP32, packed | 121.9 | 123.2 | 118.6 | 121.1 | 120.7 | **120.4** |
| NEON FP64, in place | 59.0 | 59.1 | 54.8 | 51.6 | 54.1 | 50.5 |
| NEON FP64, packed | 59.2 | 59.8 | 59.1 | **58.8** | 58.7 | **58.5** |

- **AVX2** loses a quarter to a third of its speed at every 4 KiB-multiple
  stride in place; packed, it is flat at 166–169 FP32 and 81–83 FP64 from N = 1000
  to 4096: **+32% to +60%**.
- **AVX-512** gains 3–8% at N ≥ 2000, and 15% at N = 1000, where the
  in-place kernel is unusually slow (rows 4,000 B apart).
- **NEON** gains 16–19% at N = 4096, and in FP64 8–16% from N = 2000.
- **Small N pays for the copy:** packed / in place is 0.87–0.95 at N = 64
  and 0.94–0.98 at 256; at 512 it is break-even (or +32% for AVX2 FP64,
  whose rows are already 4 KiB apart). Packing is inside the timed call.
- **Packing A too** (`PackA = true`) is 0.95–1.00× of packing B alone on
  both machines: once B is contiguous, A's row stride does not limit these
  kernels.

The blocked kernels stay the defaults of the main tables, which include
N = 64 … 512; the packed ones are separate entry points for large N.

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_AVX512=ON   # x86; plain on Apple
cmake --build build --target bench_gemm
taskset -c 2 ./build/benchmarks/bench_gemm --benchmark_filter='^Avx512Block[0-9]'   # stage 1
taskset -c 2 ./build/benchmarks/bench_gemm \
    --benchmark_filter='^Avx512Block(64x256x512|512x256x512|512x256x256)/' \
    --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=true                                   # stage 2
# packed B: block grid, then packed vs in place (Zen 5 pinned as above)
./build/benchmarks/bench_gemm --benchmark_filter='^Avx512PackBlock'       # Avx2PackBlock, NeonPackBlock likewise
./build/benchmarks/bench_gemm --benchmark_filter='Cmp/' \
    --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=true
```

(`Avx2Block`, `NeonBlock` likewise.) A block outside the grid:
`gemm_avx2_blocked<float, 6, 2, M, K, N>(A, B, C)`.
