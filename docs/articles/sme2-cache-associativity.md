---
title: "ARM SME2 GEMM: Investigating Cache Associativity"
description: "Investigating a GEMM performance cliff on Apple M4 Max through B packing and fixed-size row-stride experiments."
permalink: /articles/sme2-cache-associativity.html
---
# ARM SME2 GEMM: Investigating Cache Associativity

*How changing the distance between matrix rows recovered a performance cliff on Apple M4 Max.*

While implementing matrix multiplication with ARM SME2, I found a surprising result: a slightly larger matrix ran at less than half the throughput.

For FP32 square matrices, reading B directly from its row-major layout gave **1,365 GFLOP/s at N=4000**, but only **608 GFLOP/s at N=4096**. Packing B into contiguous strips brought the latter result to **1,344 GFLOP/s**, a **2.21× speedup**, including the packing cost.

Packing changes several aspects of memory access. The more revealing experiment kept the logical multiplication at 4096×4096 and changed only B's physical row stride. Adding 32 unused floats per row increased the unpacked kernel's throughput from **608 to 1,339 GFLOP/s**.

That is strong evidence of a stride-sensitive memory bottleneck, consistent with cache-set conflicts. It does not establish the exact cache level or its indexing function. This article follows the evidence from the GEMM access pattern to the padding experiment.

## The problem: feeding an SME2 GEMM kernel

The implementation computes `C = A × B`, with row-major matrices. For a square problem:

```text
C(i,j) = sum over k of A(i,k) × B(k,j)
```

SME provides a two-dimensional accumulator called ZA. Its floating-point outer-product instruction, FMOPA, takes two vectors and adds their outer product to a ZA tile. At one reduction step, the inputs are a segment of a column of A and a segment of a row of B. Arm describes this decomposition in its [NEON, SVE, and SME matrix-multiplication comparison](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/matrix-matrix-multiplication-neon-sve-and-sme-compared).

FMOPA belongs to SME; this implementation also uses SME2 multi-vector loads. On the tested M4 Max, the streaming vector length is 512 bits: 16 FP32 elements. Four 16×16 ZA tiles together accumulate a **32×32 output block**.

Conceptually, one reduction step is:

```text
a = A[i .. i+31, k]       // 32 values from a column
b = B[k, j .. j+31]       // 32 values from a row
C[i .. i+31, j .. j+31] += outer_product(a, b)
```

The 32 A values are separated by a row stride in the original matrix. The driver packs A into a layout where each such column segment is contiguous. Both versions compared here pack A in the same way.

B looks easier: its 32 values are already adjacent. A vector load can read them directly. The issue emerges when `k` advances: the next useful segment is in the next matrix row, potentially thousands of bytes away.

The implementation is in [sme.hpp](https://github.com/vdeviatkov/hpc-math-core/blob/d71255d2dcff2f4909aead5d9a4b0d4334166465/src/gemm/sme.hpp); the [kernel documentation](https://github.com/vdeviatkov/hpc-math-core/blob/d71255d2dcff2f4909aead5d9a4b0d4334166465/src/gemm/README.md#why-a-and-b-are-packed) contains the recorded stride experiments.

## What the comparison holds constant

This investigation compares two paths through the same blocked driver:

| Property | B read in place | B packed |
|---|---|---|
| Public kernel | `gemm_sme_blocked` | `gemm_sme` |
| A packing | Included | Included |
| Cache blocking | Same | Same |
| ZA tile arrangement and FMOPA operations | Same | Same |
| B source | Original matrix rows | Contiguous packed strips |
| B copying | None | Included in timing |

The driver is `sme_detail::blocked_driver<T, PackB>`. For the padded experiments, the benchmark calls it directly so it can supply a physical leading dimension, `ldb`, different from the logical width.

The block sizes are `Mc=128`, `Kc=1024`, and `Nc=4096` elements. For FP32, one B strip is `1024 × 32 × 4 = 131,072` bytes, or **128 KiB**. The driver reuses that strip across multiple A strips and output row blocks. Keeping it available for reuse matters more than making a single load convenient.

These are single-threaded kernel measurements on Apple M4 Max. The repository reports Apple Clang 17 and a Release build with `-O3`, `-ffast-math`, `-funroll-loops`, and `-mcpu=apple-m4`. The stride series reports medians of three randomly interleaved repetitions with background load around 5. The exact OS version, run date, and per-repetition timing samples are not recorded alongside this table.

The numbers below all come from that stride series. Earlier exploratory measurements differ slightly; combining them into one curve would obscure the experiment.

## The performance cliff

The first experiment varied square matrix dimensions around 2048 and 4096. Here `ldb=N`, so increasing N also increases the distance between B rows.

![FP32 throughput near 2048 and 4096, comparing original and packed B](assets/sme2-size-sweep.svg)

| N | FP32 row stride | Original B, GFLOP/s | Packed B, GFLOP/s | Packed / original |
|---:|---:|---:|---:|---:|
| 1984 | 7,936 bytes | 1,565 | 1,517 | 0.97× |
| 2016 | 8,064 bytes | 1,556 | 1,541 | 0.99× |
| 2048 | 8,192 bytes | 1,196 | 1,432 | 1.20× |
| 2080 | 8,320 bytes | 1,493 | 1,483 | 0.99× |
| 4000 | 16,000 bytes | 1,365 | 1,406 | 1.03× |
| 4032 | 16,128 bytes | 1,354 | 1,399 | 1.03× |
| 4064 | 16,256 bytes | 1,303 | 1,399 | 1.07× |
| 4096 | 16,384 bytes | 608 | 1,344 | 2.21× |
| 4128 | 16,512 bytes | 1,260 | 1,304 | 1.03× |

From 4000 to 4096, the matrix dimension increases by 2.4%, and the arithmetic count increases by about 7.4%. Yet unpacked throughput falls by **55.5%**. GFLOP/s already accounts for the larger operation count: doing more arithmetic alone does not explain that throughput collapse.

The recovery at 4128 is equally useful. A still larger matrix performs much better than 4096. At 2048 there is a smaller, similarly narrow dip, with recovery on either side. That pattern motivates looking at the row stride rather than assuming a smooth capacity limit.

## Why contiguous rows can still be a difficult layout

For row-major B with physical leading dimension `ldb`, element addresses are:

```cpp
// ldb is measured in elements, not bytes.
const float* row_segment = B + k * ldb + j;
```

At `ldb=4096`, successive FP32 segments start 16 KiB apart:

```text
Original B, fixed output-column segment:
k=0: base +      0 bytes    [32 useful floats]
k=1: base + 16,384 bytes    [32 useful floats]
k=2: base + 32,768 bytes    [32 useful floats]
k=3: base + 49,152 bytes    [32 useful floats]

Packed strip:
k=0: strip +   0 bytes     [32 useful floats]
k=1: strip + 128 bytes     [32 useful floats]
k=2: strip + 256 bytes     [32 useful floats]
k=3: strip + 384 bytes     [32 useful floats]
```

Each individual load accesses adjacent values in both layouts. The difference is the sequence of addresses across reduction steps, and whether those addresses remain cached until the next output row block reuses them.

A 1024-row in-place strip spans almost 16 MiB of address space while containing only 128 KiB of useful FP32 data. Its packed counterpart occupies 128 KiB contiguously. The wider address span can affect cache mapping, translation, and prefetching, even if both read the same useful values.

## Cache associativity, explained with a small model

A cache stores memory in cache lines. In a set-associative cache, each line maps to one set, and each set has a limited number of slots, called ways.

Suppose a cache has eight ways. Nine frequently reused lines that all map to one set cannot remain resident together. They can evict each other while most other sets are empty. This is a **conflict miss**: total cache capacity may be sufficient, but placement restrictions prevent the working set from fitting.

For an illustrative cache with simple modulo indexing:

```text
set(address) = floor(address / line_bytes) modulo number_of_sets
```

Assume 128-byte lines and 128 sets. This is a teaching model, not a claim about M4's cache design.

| Row stride | Stride in lines | Set-start sequence |
|---|---:|---|
| 16,384 bytes | 128 | 0, 0, 0, 0, ... |
| 16,512 bytes | 129 | 0, 1, 2, 3, ... |
| 128 bytes | 1 | 0, 1, 2, 3, ... |

In this model, a 16 KiB stride preserves the set index. Adding one line to the stride rotates the row starts through the sets. More generally, an aligned stride of `q` lines visits `S / gcd(S,q)` row-start sets in a cache with `S` sets.

This explains why powers of two can be troublesome for some cache organizations. Actual caches may use different indexing functions, physical-address bits, or hashing. Preserving the low 14 address bits with a 16 KiB stride does **not**, by itself, prove that all accesses map to one real M4 cache set.

The measurements support the conflict hypothesis through the response to stride changes. They do not reveal Apple's exact set mapping.

## Packing B changes the stride used by the compute loop

Packing copies each narrow B strip into a separate contiguous region. This is the relevant implementation, with FP32 substituted for the template type:

```cpp
for (std::size_t j0 = 0; j0 < nc; j0 += nr) {
    const std::size_t cols = std::min(nr, nc - j0);
    float* strip = dst + j0 * kc;
    for (std::size_t k = 0; k < kc; ++k) {
        std::memcpy(strip + k * nr,
                    B + k * ldb + j0,
                    cols * sizeof(float));
        std::fill(strip + k * nr + cols,
                  strip + (k + 1) * nr, 0.0f);
    }
}
```

For this FP32 microkernel, `nr=32`. Within a packed strip, the distance between reduction rows becomes 32 floats, or 128 bytes, independent of the original matrix width. Zero filling handles partial strips.

The shared macro-kernel still loads B with the same address expression:

```cpp
const auto b = svld1_x2(pb0, b_strip + k * ldb);
```

The packed driver supplies `ldb=nr`; the original-layout driver supplies the physical matrix row stride. The surrounding FMOPA operations are unchanged. The precise SME intrinsic semantics and streaming attributes are specified in [Arm's C Language Extensions](https://arm-software.github.io/acle/main/acle.html).

The number 32 comes from the microkernel's output shape. It is not a universal cache-conflict remedy. The useful transformation is from widely separated segments to a compact stream that is reused after packing.

Packing still has to read the original strided layout once. Its benefit is amortizing that read and copy across repeated compute accesses. It does not make the source layout disappear for free.

## The stronger test: same multiplication, different leading dimension

Changing N changes arithmetic, storage size, and traversal counts simultaneously. To separate those effects, the next experiment fixed `M=N=K=4096` and stored B in a buffer with wider physical rows.

```text
Logical B: 4096 rows × 4096 useful columns
Physical B: 4096 rows × ldb stored elements

B(k,j) = buffer[k * ldb + j], where j < 4096
```

Padding columns are not multiplied. A, C, and the number of logical multiply-adds retain the same dimensions.

![FP32 throughput at fixed logical N=4096 for four physical B leading dimensions](assets/sme2-padding.svg)

| Physical ldb | FP32 row stride | Original B, GFLOP/s | Packed B, GFLOP/s | Original vs ldb=4096 |
|---:|---:|---:|---:|---:|
| 4096 | 16,384 bytes | 608 | 1,344 | 1.00× |
| 4097 | 16,388 bytes | 1,145 | 1,365 | 1.88× |
| 4128 | 16,512 bytes | 1,339 | 1,386 | 2.20× |
| 4160 | 16,640 bytes | 1,332 | 1,390 | 2.19× |

Adding 32 floats per row increases B's allocation by just 512 KiB, from 64 MiB to 64.5 MiB. The larger allocation runs much faster. At `ldb=4128`, the unpacked path is within **3.4%** of the packed path measured with the same source stride.

This result makes a pure arithmetic-size explanation untenable and weakens a simple total-capacity explanation. It is consistent with changing cache-set placement while preserving the useful working set.

Padding also changes alignment, page offsets, and translation behavior. It does not hold every property of the memory system constant. We should therefore describe it as a controlled stride experiment, rather than claiming that it proves cache conflicts are the sole cause.

### Why adding one float helps less

At `ldb=4097`, the row stride changes by four bytes. Successive row segments move through different offsets within a cache line. Assuming 128-byte lines, a 128-byte segment occupies two lines whenever its start is not line-aligned.

The 4128 and 4160 cases preserve a constant offset modulo 128 while changing the larger stride. They avoid that rotating alignment effect. Matrix storage is guaranteed to be 64-byte aligned in this repository, so a 128-byte segment is not guaranteed to occupy exactly one 128-byte line even in those cases. The distinction is constant versus changing line offsets, not a guarantee that every segment is aligned.

That makes 4097 a useful additional control, but not the cleanest padding choice for comparing these full-width loads.

## FP64 provides a second view

The same benchmark includes double precision. The FP64 microkernel uses a different output height and ZA arrangement, so its absolute throughput should not be interpreted as a direct architectural comparison with FP32. Its B strip still contains 32 elements per reduction row, now 256 bytes.

| N, with ldb=N | FP64 row stride | Original B, GFLOP/s | Packed B, GFLOP/s |
|---:|---:|---:|---:|
| 1984 | 15,872 bytes | 407 | 418 |
| 2016 | 16,128 bytes | 410 | 419 |
| 2048 | 16,384 bytes | 281 | 407 |
| 2080 | 16,640 bytes | 398 | 404 |
| 4000 | 32,000 bytes | 380 | 409 |
| 4032 | 32,256 bytes | 380 | 408 |
| 4064 | 32,512 bytes | 364 | 411 |
| 4096 | 32,768 bytes | 268 | 398 |
| 4128 | 33,024 bytes | 362 | 395 |

At fixed logical N=4096:

| Physical ldb | FP64 row stride | Original B, GFLOP/s | Packed B, GFLOP/s |
|---:|---:|---:|---:|
| 4096 | 32,768 bytes | 268 | 398 |
| 4097 | 32,776 bytes | 367 | 407 |
| 4128 | 33,024 bytes | 376 | 407 |
| 4160 | 33,280 bytes | 376 | 406 |

The power-of-two dips and padding recovery appear in both precisions. FP64 reaches a 16 KiB stride already at N=2048; its larger dips there reinforce the value of reasoning in bytes rather than dimensions alone. These are supporting observations, not a complete cache-index reconstruction.

## What the evidence establishes—and what remains open

The recorded results establish three practical facts:

1. With B read in place, throughput drops sharply at the tested power-of-two strides and recovers at nearby dimensions.
2. At fixed logical N=4096, padding B's leading dimension recovers most of the FP32 loss without packing B or changing the arithmetic.
3. Packing B removes the large cliff in these tests even when the timed call includes packing and scratch-buffer allocation.

Cache-set conflicts are a plausible explanation for the narrow stride sensitivity. There are no hardware-counter measurements in this series identifying cache misses or attributing them to a particular cache level. A CPU load-path experiment would also need care before being used to infer the behavior of SME's memory path.

Translation and prefetching remain candidates for part of the effect. A strided strip touches widely separated addresses, while its packed version has a compact footprint. Padding preserves a broadly similar sparse access pattern but changes page offsets and the precise pages touched. It cannot independently exclude all TLB effects.

Packing is not uniformly faster. Near N=1984 and 2016, FP32 in-place throughput is slightly higher. Those small differences should be treated cautiously with three repetitions and background activity. The important result is the large cliff, not a claim that a one-percent difference is statistically established.

Nor does packing guarantee freedom from evictions. Other operands, competing threads, larger panels, and different hardware can still limit cache residency. Away from the exact powers of two, the remaining packed advantage may have several causes that this experiment does not separate.

## Reproducing the investigation

The benchmark registrations and timed loop are in [bench_gemm.cpp](https://github.com/vdeviatkov/hpc-math-core/blob/d71255d2dcff2f4909aead5d9a4b0d4334166465/benchmarks/bench_gemm.cpp). They cover nine nearby dimensions and three additional padded configurations, for both precisions and both B paths: **48 benchmark cases**.

Follow the repository's [SME build instructions](https://github.com/vdeviatkov/hpc-math-core/blob/main/docs/build.md#sme-and-amx-build-flags) on supported hardware:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

./build/benchmarks/bench_gemm \
  --benchmark_filter=BStride \
  --benchmark_repetitions=3 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=bstride-results.json \
  --benchmark_out_format=json
```

The padded-layout correctness checks are in [test_gemm.cpp](https://github.com/vdeviatkov/hpc-math-core/blob/d71255d2dcff2f4909aead5d9a4b0d4334166465/tests/test_gemm.cpp): both B paths are compared against a reference multiplication with junk in the padding columns.

Input matrices are allocated and initialized before the benchmark iteration loop. Inside it, the complete blocked driver runs: scratch allocation, A packing, optional B packing, and computation. This measures repeated calls against existing inputs, not an explicitly flushed cold-cache workload or a prepacked-weight compute-only workload.

The `GFLOP/s` counter uses `2 × N³` operations per iteration. It is an iteration-invariant rate counter; the BStride registrations do not request `UseRealTime()`, so the rate follows Google Benchmark's default CPU-time basis. Preserve both CPU and real-time fields in a new result archive rather than silently treating them as interchangeable.

For a publication-quality rerun, record the commit, working-tree diff, OS and compiler versions, benchmark context, and individual repetitions. Prefer a quiet machine and more repetitions to quantify spread. The existing documentation reports rounded medians, so the charts here intentionally have no invented error bars.

The companion [CSV](assets/sme2-bstride-documented.csv) transcribes those documented medians; the [chart script](assets/render_charts.py) regenerates the figures. It is derived data, not the original benchmark output. Code and evidence links are pinned to commit `d71255d2dcff2f4909aead5d9a4b0d4334166465`, which includes the BStride experiment. General project and build-guide links follow `main`. Additional working-tree source snapshots are included with the article assets for archival context.

## The engineering lesson

The B values needed by one FMOPA step were already contiguous. That made the original row-major layout look suitable. The problem appeared across steps and across reuse: a narrow strip was spread over a large address range with a particularly sensitive row stride.

The useful experiment was to keep the multiplication fixed and change the distance between rows. Padding by 32 FP32 elements recovered a **2.20×** throughput loss without changing the compute operations. Packing recovered a similar amount while giving the kernel control over B's compute-time layout for arbitrary caller strides.

The lesson is to inspect both the bytes consumed by each instruction and the sequence of addresses across the loop. A small useful working set can still be difficult to cache. In this implementation, the arithmetic kernel was capable of much more throughput once its input layout stopped exposing the pathological stride.
