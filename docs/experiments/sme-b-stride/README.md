# SME2 GEMM and B's row stride: experiment data

Raw data behind the article
[*A power-of-two stride halves SME2 GEMM throughput on Apple M4*](../../articles/sme2-cache-associativity.md).
Apple M4 Max, single thread, FP32. Machine, compiler, commands and source
hashes: [`metadata.json`](metadata.json). The size and padding sweeps of
`bench_gemm --benchmark_filter=BStride` are in
[`docs/benchmarks.md`](../../benchmarks.md#sme-b-row-stride).

| File | Program | What it measures |
|---|---|---|
| [`phases.csv`](phases.csv) | [`sme_b_layout`](../../../benchmarks/experiments/sme_b_layout.cpp) | full driver, end to end and by phase, B in place vs packed, ldb = 4096 / 4128 / 4160 |
| [`height.csv`](height.csv) | [`sme_b_stride_probe height`](../../../benchmarks/experiments/sme_b_stride_probe.cpp) | macro-kernel only, strip height Kc = 8 … 1024 rows, B rows 16 KiB vs 16,512 B apart |
| [`width.csv`](width.csv) | [`sme_b_stride_probe width`](../../../benchmarks/experiments/sme_b_stride_probe.cpp) | macro-kernel only, N = 1024 … 4224, B in place with one line of padding vs packed |
| [`chase.csv`](chase.csv) | [`cache_chase`](../../../benchmarks/experiments/cache_chase.cpp) | CPU core load latency over L lines 16 KiB / 16,512 B apart |

The probes only time. Correctness of every B layout is covered by the test
`GemmSme.PaddedBMatchesNaive`, and `sme_b_layout` checks each timed path
against the packed driver before timing.

## Phases: where the time goes

`sme_b_layout 4096 15 3`. Each of 15 repetitions runs six cases (three
`ldb` × B in place or packed) end to end (the driver, mean of 3 calls) and
by phase (the same loop nest with a clock around each phase), in shuffled
order. Medians, ms; interquartile range in brackets.

| ldb | B | End to end | GFLOP/s | Pack A | Pack B | Compute |
|---:|:--|--:|--:|--:|--:|--:|
| 4096 | in place | 238.1 [232.3–240.9] | 577 | 2.95 | — | 230.8 [228.4–236.6] |
| 4096 | packed | 104.7 [103.5–105.5] | 1,313 | 2.96 | 4.81 | 96.9 [95.9–98.7] |
| 4128 | in place | 114.1 [109.6–117.4] | 1,204 | 2.94 | — | 110.0 [105.3–112.5] |
| 4128 | packed | 102.0 [100.8–103.6] | 1,348 | 2.96 | 2.07 | 96.0 [95.2–97.7] |
| 4160 | in place | 116.6 [109.9–117.6] | 1,179 | 2.93 | — | 107.2 [105.7–110.9] |
| 4160 | packed | 101.4 [100.8–102.5] | 1,355 | 2.98 | 2.03 | 97.0 [95.7–98.8] |

Packed B won all 15 repetitions at `ldb` = 4128 and 4160. The phase clocks
wrap every `pack_a`, `pack_b` and macro-kernel call (128 macro-kernel calls
per driver call), so phase figures carry some timing overhead.

## Height: how many rows the conflicting cache holds

`sme_b_stride_probe height`: N = 1024, M = 4096, one Kc block; A packed
and B laid out once, only the macro-kernel timed, 9 shuffled repetitions.
B in place with rows 16 KiB apart (`ldb` = 4096) or 16,512 B apart
(`ldb` = 4128); packed for reference. Median GFLOP/s [IQR].

| Kc (rows) | ldb 4096 | ldb 4128 | packed | 4096 / 4128 |
|---:|--:|--:|--:|--:|
| 8 | 191 [190–193] | 199 [196–207] | 224 | 0.96 |
| 32 | 552 [491–563] | 615 [603–625] | 638 | 0.90 |
| 128 | 1,048 [1,008–1,104] | 1,170 [1,138–1,208] | 1,297 | 0.90 |
| 256 | 1,321 [1,202–1,331] | 1,436 [1,397–1,487] | 1,564 | 0.92 |
| 384 | 1,380 [1,339–1,398] | 1,606 [1,569–1,616] | 1,670 | 0.86 |
| **448** | **1,034 [978–1,035]** | 1,624 [1,539–1,661] | 1,709 | **0.64** |
| 512 | 1,104 [1,098–1,146] | 1,634 [1,592–1,644] | 1,736 | 0.68 |
| 640 | 1,171 [961–1,196] | 1,676 [1,574–1,692] | 1,733 | 0.70 |
| 768 | 960 [882–980] | 1,710 [1,699–1,737] | 1,753 | 0.56 |
| 896 | 791 [691–799] | 1,710 [1,658–1,751] | 1,771 | 0.46 |
| 1024 | 652 [647–657] | 1,728 [1,712–1,732] | 1,763 | 0.38 |

`height.csv` also has Kc = 16, 64, 576. Up to 384 rows the power-of-two
stride costs 4–14%; between 384 and 448 rows it breaks.

## Width: what padding leaves

`sme_b_stride_probe width`: Kc = 1024, M = 4096, macro-kernel only, 9
shuffled repetitions. B in place with `ldb = N + 32` against packed.

| N | in place, ldb = N + 32 | packed | in place / packed |
|---:|--:|--:|--:|
| 1024 | 1,790 [1,788–1,806] | 1,790 [1,761–1,819] | 1.00 |
| 2048 | 1,730 [1,705–1,738] | 1,783 [1,776–1,802] | 0.97 |
| 3072 | 1,573 [1,548–1,626] | 1,637 [1,626–1,646] | 0.96 |
| 3584 | 1,545 [1,515–1,560] | 1,622 [1,614–1,631] | 0.95 |
| 3968 | 1,456 [1,429–1,476] | 1,561 [1,530–1,563] | 0.93 |
| 4000 | 1,426 [1,353–1,447] | 1,568 [1,564–1,583] | 0.91 |
| 4096 | 1,413 [1,318–1,424] | 1,546 [1,537–1,551] | 0.91 |
| 4128 | 1,394 [1,369–1,426] | 1,563 [1,550–1,571] | 0.89 |
| 4224 | 1,394 [1,312–1,428] | 1,544 [1,541–1,551] | 0.90 |

## Chase: the CPU core's view

`cache_chase`: a dependent pointer chase over L lines in random order,
best of 5 runs, ns per load.

| L lines | 16,384 B apart | 16,512 B apart |
|---:|--:|--:|
| 8 | 0.76 | 0.74 |
| 16 | 5.00 | 0.76 |
| 256 | 7.86 | 2.28 |
| 512 | 8.35 | 2.78 |
| 576 | 8.09 | 2.87 |
| 640 | 9.20 | 2.87 |
| 768 | 10.77 | 2.96 |
| 1024 | 23.69 | 3.06 |
| 2048 | 36.27 | 7.59 |

16 KiB apart, only 8 lines stay in L1 (0.76 ns) and about 576 in L2
(~8 ns). 128 bytes more of stride, all 1,024 lines stay in L1; the 2–3 ns
from 256 lines on, against 0.76 ns, is consistent with first-level TLB
misses, as every line sits on its own 16 KiB page.

## Conditions

A desktop machine with other work running (load average 4–6). Results are
medians of interleaved, shuffled repetitions; the interquartile ranges show
the noise. Inputs stay warm in cache between calls; nothing is flushed.

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON
cmake --build build --target sme_b_layout sme_b_stride_probe cache_chase
./build/benchmarks/sme_b_layout 4096 15 3 > phases.csv
./build/benchmarks/sme_b_stride_probe height > height.csv
./build/benchmarks/sme_b_stride_probe width > width.csv
./build/benchmarks/cache_chase > chase.csv
```
