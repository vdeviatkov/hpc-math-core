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

## Load latency by working set

A dependent pointer chase in random order, one load per 128-byte line,
best of 5, ns per load:

| Working set | 8 KiB – 128 KiB | 256 KiB – 2 MiB | 4 MiB | 8 MiB | 16 MiB | 32 MiB | 64 MiB | 1 GiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ns per load | 0.76 | 6.7–6.9 | 8.2 | 9.0 | 17.1 | 43.3 | 95.8 | 110.6 |

L1 up to 128 KiB, L2 from there to a few MiB. From 4 MiB the working set
exceeds the first-level TLB's 2 MiB reach (below), adding translation to
each L2 hit. At 16 MiB the data no longer fits in the L2 it shares with the
cluster, at 32 MiB it is served by the system-level cache, and from 64 MiB
by DRAM at ~100–110 ns.

## TLBs

`cache_chase tlb` touches P lines, one per 16 KiB page, with the line inside
each page rotating so that the lines spread over all cache sets, and
compares that with the same P lines packed densely. ns per load:

| Pages | 128 | 136 | 160 | 192 | 256 – 1,024 | 2,048 | 3,072 | 3,328 | 8,192 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| one line per page | 0.76 | 1.00 | 1.58 | 3.65 | 4.32 | 8.16 | 8.72 | 16.51 | 17.13 |
| same lines, dense | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 6.76 | 6.79 | 6.62 | 6.79 |

- **First-level data TLB: 128 entries**, 2 MiB of 16 KiB pages. Latency
  starts rising at 136 pages; past about 192, every load misses it and
  costs ~3.6 ns more (4.32 against 0.76 ns, data still in L1).
- **Second-level TLB: 3,072 entries**, 48 MiB. Up to 3,072 pages the
  spread chase costs 1.4–1.9 ns over the same lines dense (both in L2); at
  3,328 it costs 9.9 ns more: a page-table walk.

## The SME unit

`sme_fmopa`: a streaming loop of FMOPAs on constant vectors into T ZA
tiles in turn, no memory traffic; best of 5.

| | tiles | ns per FMOPA | GFLOP/s |
|---|--:|--:|--:|
| FP32 (16 × 16 tiles, 512 flops) | 1 | 1.036 | 494 |
| | 2 | 0.518 | 989 |
| | 4 | 0.259 | 1,977 |
| FP64 (8 × 8 tiles, 128 flops) | 1 | 1.035 | 124 |
| | 4 | 0.259 | 495 |
| | 8 | 0.258 | 495 |

An FMOPA takes ~1.04 ns to feed its result into the next FMOPA on the same
tile, and the unit issues one every 0.26 ns: four independent tiles are
needed to reach peak, **1.98 TFLOP/s FP32 and 0.49 TFLOP/s FP64**.

Same 4-tile FP32 loop on several threads at once, total GFLOP/s:

| Threads | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|--:|--:|--:|--:|--:|--:|--:|
| GFLOP/s | 1,977 | 1,954 | 2,121 | 2,448 | 3,670 | 3,999 | 4,205 |

Two threads together run no faster than one: the scheduler placed them on
one cluster, and they shared its SME unit. Spread over both P-clusters the
total approaches twice the single-thread peak: **one SME unit per
P-cluster**, not per core.

## Conditions

A desktop machine with other work running (load average 5–6). Latencies
are best of 5 runs, so they reflect the hardware rather than the noise.

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON
cmake --build build --target cache_chase sme_fmopa
./build/benchmarks/cache_chase sizes > latency.csv
./build/benchmarks/cache_chase tlb > tlb.csv
./build/benchmarks/sme_fmopa > fmopa.csv
```
