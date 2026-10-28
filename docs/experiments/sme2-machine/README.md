# Apple M4 Max: caches, TLBs and the SME unit

Measurements behind the article
[*Matrix multiplication on Apple M4's SME2 unit*](../../articles/sme2-gemm.md).
Machine, sysctl values, compiler, commands and source hashes:
[`metadata.json`](metadata.json).

| File | Program | What it measures |
|---|---|---|
| [`latency.csv`](latency.csv) | [`cache_chase sizes`](../../../benchmarks/experiments/cache_chase.cpp) | load latency over a random chase in a working set of 8 KiB … 1 GiB |
| [`tlb.csv`](tlb.csv) | [`cache_chase tlb`](../../../benchmarks/experiments/cache_chase.cpp) | one line per 16 KiB page against the same lines packed densely |
| [`fmopa.csv`](fmopa.csv) | [`sme_fmopa`](../../../benchmarks/experiments/sme_fmopa.cpp) | FMOPA latency and throughput, 1 … 12 threads |

## What macOS reports

`sysctl hw` on a P-core: 128-byte cache lines, 128 KiB L1 data and 192 KiB
L1 instruction cache per core, a 16 MiB L2 shared by the 6 cores of a
cluster, 16 KiB pages, and a 64-byte (512-bit) maximum streaming vector
length. macOS reports no TLB sizes and nothing about the SME unit beyond
its features.

## Core clock

macOS does not report the clock either. Each program measures it first: a
chain of dependent integer adds runs at exactly one add per cycle, so its
time per add is one cycle (`core_clock.hpp`). The P-core ran at
**3.94–3.99 GHz** in these runs; cycle counts below use the clock measured
in the same run.

## Load latency by working set

A dependent pointer chase in random order, one load per 128-byte line,
best of 5:

| Working set | 8 KiB – 128 KiB | 256 KiB – 2 MiB | 4 MiB | 8 MiB | 16 MiB | 32 MiB | 64 MiB | 1 GiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| cycles | 3 | 26–27 | 32 | 36 | 66 | 151 | 377 | 437 |
| ns | 0.76 | 6.7–6.9 | 8.1 | 9.0 | 16.9 | 38.3 | 95.8 | 110.9 |

L1 up to 128 KiB (3 cycles), L2 from there to a few MiB (26–27 cycles).
From 4 MiB the working set exceeds the first-level TLB's 2 MiB reach
(below), adding translation to each L2 hit. At 16 MiB the data no longer
fits in the L2 it shares with the cluster, at 32 MiB it is served by the
system-level cache, and from 64 MiB by DRAM at 380–440 cycles.

## TLBs

`cache_chase tlb` touches P lines, one per 16 KiB page, with the line inside
each page rotating so that the lines spread over all cache sets, and
compares that with the same P lines packed densely. Cycles per load:

| Pages | 128 | 136 | 160 | 192 | 256 – 1,024 | 2,048 | 3,072 | 3,328 | 8,192 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| one line per page | 3.0 | 3.9 | 6.2 | 14.3 | 17.0 | 32.1 | 34.6 | 65.1 | 66.7 |
| same lines, dense | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 26.2 | 26.6 | 26.2 | 27.2 |

- **First-level data TLB: 128 entries**, 2 MiB of 16 KiB pages. Latency
  starts rising at 136 pages; from about 256, every load misses it and
  costs **14 cycles** more (17 against 3, data still in L1).
- **Second-level TLB: 3,072 entries**, 48 MiB. Up to 3,072 pages the
  spread chase costs 6–8 cycles over the same lines dense (both in L2); at
  3,328 it costs **39 cycles** more: a page-table walk.

## The SME unit

`sme_fmopa`: a streaming loop of FMOPAs on constant vectors into T ZA
tiles in turn, no memory traffic; best of 5.

| | tiles | cycles per FMOPA | GFLOP/s |
|---|--:|--:|--:|
| FP32 (16 × 16 tiles, 512 flops) | 1 | 4.09 | 494 |
| | 2 | 2.04 | 990 |
| | 4 | 1.03 | 1,963 |
| FP64 (8 × 8 tiles, 128 flops) | 1 | 4.09 | 124 |
| | 4 | 1.01 | 500 |
| | 8 | 1.02 | 497 |

An FMOPA's result reaches the next FMOPA on the same tile after **4
cycles**, and the unit issues **one FMOPA per cycle**: four independent
tiles are needed to reach peak, 512 FP32 flops per cycle, **1.96 TFLOP/s
FP32 and 0.50 TFLOP/s FP64** at this clock.

Same 4-tile FP32 loop on several threads at once, total GFLOP/s:

| Threads | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|--:|--:|--:|--:|--:|--:|--:|
| GFLOP/s | 1,963 | 1,954 | 2,119 | 2,830 | 3,912 | 4,007 | 4,203 |

Two threads together run no faster than one: the scheduler placed them on
one cluster, and they shared its SME unit. Spread over both P-clusters the
total reaches about twice the single-thread peak: **one SME unit per
P-cluster**, not per core.

## Conditions

A desktop machine with other work running (load average 5–6). Latencies
are best of 5 runs, so they reflect the hardware rather than the noise;
the clock is measured in the same process right before each run.

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON
cmake --build build --target cache_chase sme_fmopa
./build/benchmarks/cache_chase sizes > latency.csv
./build/benchmarks/cache_chase tlb > tlb.csv
./build/benchmarks/sme_fmopa > fmopa.csv
```
