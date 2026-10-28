# SME2 GEMM and B's row stride: experiment data

Raw data behind the article
[*A power-of-two stride cuts SME2 GEMM throughput by two thirds on Apple M4*](../../articles/sme2-cache-associativity.md).
Apple M4 Max, single thread, FP32. Machine, compiler, commands and source
hashes: [`metadata.json`](metadata.json). The size and padding sweeps of
`bench_gemm --benchmark_filter=BStride` are in
[`docs/benchmarks.md`](../../benchmarks.md#sme-b-row-stride).

| File | Program | What it measures |
|---|---|---|
| [`phases.csv`](phases.csv) | [`sme_b_layout`](../../../benchmarks/experiments/sme_b_layout.cpp) | full driver, end to end and by phase, B in place vs packed, ldb = 4096 / 4128 / 4160 |
| [`height.csv`](height.csv) | [`sme_b_stride_probe height`](../../../benchmarks/experiments/sme_b_stride_probe.cpp) | macro-kernel only, strip height Kc = 8 … 2048 rows, B rows 16 KiB vs 16,512 B apart |
| [`width.csv`](width.csv) | [`sme_b_stride_probe width`](../../../benchmarks/experiments/sme_b_stride_probe.cpp) | macro-kernel only, N = 1024 … 4224, B in place with one line of padding vs packed |
| [`chase.csv`](chase.csv) | [`cache_chase`](../../../benchmarks/experiments/cache_chase.cpp) | CPU core load latency over L lines 16 KiB / 16,512 B apart |

The probes only time. Correctness of every B layout is covered by the test
`GemmSme.PaddedBMatchesNaive`, and `sme_b_layout` checks each timed path
against the packed driver before timing.

## Phases: where the time goes

`sme_b_layout 4096 15 3`, with M4's measured block sizes
(`SmeParams<float>`: Mc = 1024, Kc = 2048, Nc = 4096). Each of 15 repetitions runs six cases (three `ldb`
× B in place or packed) end to end (the driver, mean of 3 calls) and by
phase (the same loop nest with a clock around each phase), in shuffled
order. Medians, ms; interquartile range in brackets.

| ldb | B | End to end | GFLOP/s | Pack A | Pack B | Compute |
|---:|:--|--:|--:|--:|--:|--:|
| 4096 | in place | 281.2 [280.5–286.8] | 489 | 3.75 | — | 278.9 [275.2–282.2] |
| 4096 | packed | 91.1 [90.9–92.0] | 1,508 | 3.84 | 4.94 | 82.4 [82.1–84.0] |
| 4128 | in place | 92.9 [92.5–93.7] | 1,479 | 3.74 | — | 89.0 [88.6–89.7] |
| 4128 | packed | 88.5 [87.8–88.8] | 1,553 | 3.76 | 2.54 | 82.0 [81.7–83.1] |
| 4160 | in place | 93.2 [92.9–94.6] | 1,474 | 3.77 | — | 89.6 [89.4–91.3] |
| 4160 | packed | 88.4 [88.0–89.0] | 1,555 | 3.81 | 2.49 | 81.7 [81.3–85.0] |

Packed B won all 15 repetitions at every `ldb`. The phase clocks wrap
every `pack_a`, `pack_b` and macro-kernel call (8 macro-kernel calls per
driver call), so phase figures carry some timing overhead.

## Height: how many rows the conflicting cache holds

`sme_b_stride_probe height`: N = 1024, M = 4096 in blocks of Mc = 1024,
one Kc block; A packed and B laid out once, only the macro-kernel timed,
9 shuffled repetitions. B in place with rows 16 KiB apart (`ldb` = 4096)
or 16,512 B apart (`ldb` = 4128); packed for reference. Median GFLOP/s
[IQR].

| Kc (rows) | ldb 4096 | ldb 4128 | packed | 4096 / 4128 |
|---:|--:|--:|--:|--:|
| 8 | 159 [142–168] | 168 [159–176] | 176 | 0.95 |
| 32 | 396 [369–465] | 532 [518–538] | 533 | 0.74 |
| 128 | 878 [806–919] | 1,150 [1,113–1,170] | 1,250 | 0.76 |
| 256 | 1,200 [1,168–1,280] | 1,434 [1,418–1,452] | 1,505 | 0.84 |
| 384 | 1,343 [1,334–1,372] | 1,558 [1,532–1,563] | 1,665 | 0.86 |
| 448 | 1,335 [1,268–1,351] | 1,614 [1,573–1,638] | 1,694 | 0.83 |
| 512 | 1,267 [1,237–1,335] | 1,652 [1,594–1,665] | 1,693 | 0.77 |
| 576 | 1,236 [1,214–1,249] | 1,677 [1,671–1,687] | 1,696 | 0.74 |
| **640** | **946 [941–971]** | 1,668 [1,647–1,698] | 1,737 | **0.57** |
| 768 | 924 [907–937] | 1,688 [1,571–1,734] | 1,724 | 0.55 |
| 896 | 768 [749–774] | 1,703 [1,673–1,730] | 1,771 | 0.45 |
| 1024 | 626 [618–628] | 1,698 [1,667–1,724] | 1,734 | 0.37 |
| 1536 | 510 [502–512] | 1,666 [1,655–1,683] | 1,757 | 0.31 |
| 2048 | 500 [499–506] | 1,584 [1,563–1,610] | 1,700 | 0.32 |

`height.csv` also has Kc = 16 and 64. Up to 576 rows the power-of-two
stride costs 5–26%; between 576 and 640 rows it breaks. An earlier run
with Mc = 128 (4 A strips per B strip instead of 32) broke between 384
and 448 rows; both are in the range where the chase below loses L2.

## Width: what padding leaves

`sme_b_stride_probe width`: Kc = 2048, M = 4096, macro-kernel only, 9
shuffled repetitions. B in place with `ldb = N + 32` against packed.

| N | in place, ldb = N + 32 | packed | in place / packed |
|---:|--:|--:|--:|
| 1024 | 1,676 [1,651–1,698] | 1,740 [1,723–1,777] | 0.96 |
| 2048 | 1,654 [1,623–1,698] | 1,749 [1,734–1,774] | 0.95 |
| 3072 | 1,589 [1,587–1,616] | 1,725 [1,715–1,734] | 0.92 |
| 3584 | 1,572 [1,562–1,588] | 1,714 [1,695–1,722] | 0.92 |
| 3968 | 1,560 [1,546–1,569] | 1,686 [1,681–1,690] | 0.93 |
| 4000 | 1,559 [1,556–1,574] | 1,697 [1,696–1,709] | 0.92 |
| 4096 | 1,554 [1,533–1,556] | 1,686 [1,675–1,695] | 0.92 |
| 4128 | 1,572 [1,552–1,578] | 1,680 [1,674–1,692] | 0.94 |
| 4224 | 1,549 [1,546–1,557] | 1,682 [1,675–1,694] | 0.92 |

## Chase: the CPU core's view

`cache_chase stride`: a dependent pointer chase over L lines in random
order, best of 5, in cycles of the core clock measured in the same run
(3.99 GHz):

| L lines | 16,384 B apart | 16,512 B apart |
|---:|--:|--:|
| 8 | 3.0 | 3.0 |
| 16 | 24.2 | 3.0 |
| 256 | 34.4 | 9.1 |
| 448 | 32.7 | 11.4 |
| 512 | 41.0 | 11.1 |
| 640 | 41.7 | 11.5 |
| 768 | 52.0 | 11.8 |
| 1024 | 100.0 | 12.1 |
| 2048 | 144.1 | 30.0 |

16 KiB apart, only 8 lines stay in L1 (3 cycles) and roughly 450–600 in
L2 (~32 cycles, including a TLB miss per load); the step's position moved
between 448 and 640 lines across three runs on this busy machine. 128 bytes
more of stride, all 1,024 lines stay in L1; the 9–12 cycles from 256 lines
on, against 3, are first-level TLB misses, as every line sits on its own
16 KiB page and that TLB holds 128 pages
([machine data](../sme2-machine/README.md)).

## Conditions

A desktop machine with other work running (load average 4–6 during the
chase, 5–10 during the other runs). Results are medians of interleaved,
shuffled repetitions; the interquartile ranges show the noise. Inputs stay
warm in cache between calls; nothing is flushed.

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON
cmake --build build --target sme_b_layout sme_b_stride_probe cache_chase
./build/benchmarks/sme_b_layout 4096 15 3 > phases.csv
./build/benchmarks/sme_b_stride_probe height > height.csv
./build/benchmarks/sme_b_stride_probe width > width.csv
./build/benchmarks/cache_chase > chase.csv
```
