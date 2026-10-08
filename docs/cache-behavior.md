# Cache Behaviour of Matrix Multiplication

This document is the theoretical companion to the kernel implementations in `src/gemm/`. It explains why loop order matters for performance, building up from first principles.

---

## 1. The Memory Hierarchy

Modern CPUs do not read from DRAM directly. Data travels through a hierarchy of ever-faster, ever-smaller caches:

```
Registers     ~0 cycles    a few hundred bytes to a few KB
   ↕
L1 cache      ~4 cycles    32–128 KB per core  (Zen 5: 48 KB, M4 P-core: 128 KB)
   ↕
L2 cache     ~12–20 cycles 1–16 MB            (Zen 5: 1 MB/core, M4: 16 MB per P-cluster)
   ↕
L3 cache     ~40–50 cycles tens of MB, shared  (Zen 5: 32 MB per CCD; M4: none)
   ↕
DRAM         ~80–120 ns    GBs
```

An algorithm is compute-bound when the CPU's arithmetic units are the bottleneck. It is memory-bound when the CPU stalls waiting for data from a lower level of the hierarchy. Naïve GEMM is memory-bound at all but the smallest sizes.

---

## 2. Cache Lines

The unit of transfer between any two adjacent levels of the hierarchy is the cache line — 64 bytes on x86 and most ARM cores, 128 bytes on Apple M-series (`sysctl hw.cachelinesize`). The examples below use 64-byte lines: when you read a single `double` (8 bytes), the CPU loads the surrounding 64 bytes — 8 doubles — into the cache.

```
DRAM layout:
offset  0  8 16 24 32 40 48 56 64 72 80 …
       [d0 d1 d2 d3 d4 d5 d6 d7|d8 d9 …]
        ←── 1 cache line (64 B) ──→

Accessing d0 loads {d0…d7} into L1; d1…d7 then hit in L1.
```

This is spatial locality: data near a recently-used address is likely to be reused soon. Algorithms that exploit spatial locality use every byte of every loaded cache line.

---

## 3. Row-Major Layout and Matrix Element Addresses

`hpc::Matrix<T>` stores elements in row-major order. For a matrix with `cols` columns:

```
element (i, j)  →  data[ i * cols + j ]
```

Consecutive elements in the same row have adjacent memory addresses (stride 1). A full row fits in `cols * sizeof(T)` bytes = `cols * 8` bytes for `double`.

Consecutive elements in the same column are separated by `cols * sizeof(T)` bytes — for N=1024 that is 8 KB, spanning 128 cache lines.

This asymmetry is the root cause of naive GEMM's poor performance.

---

## 4. Reuse Distance

Reuse distance is the number of distinct memory addresses accessed between two accesses to the same address. If the reuse distance exceeds the number of cache lines in a cache level, that level will not hold the data from the first access when the second access occurs — a cache miss.

### Naive GEMM (i-j-k): reuse distance of B

For a fixed `j`, the inner k-loop reads `B(0,j), B(1,j), …, B(K-1,j)`. These
are `N` elements apart, so each one is on a different cache line, and only
one of that line's 8 doubles is used.

The other 7 are used for `B(k,j+1)`, `B(k,j+2)`, … — but only in the next
iterations of the j-loop, after the k-loop has touched about `K` other lines
of B. That is the reuse distance: ~`K` cache lines. At K=1024 with 64-byte
lines that is 64 KB, more than a typical 32–48 KB L1, so the line has been
evicted from L1 by the time it is reused and every B access misses L1.

### Reordered GEMM (i-k-j): reuse distance of B

The inner j-loop accesses `B(k,0), B(k,1), …, B(k,N-1)` — a sequential walk across row k.

Consecutive iterations read `B(k, j)` and `B(k, j+1)`, 8 bytes apart: all 8 doubles of a line are used back to back (reuse distance 0), and the hardware prefetcher can run ahead of the stream.

---

## 5. Working Set Analysis

The working set of a loop nest is the set of cache lines touched in one execution of the inner loop.

### Naive inner loop (fixed i, fixed j)

| Array | Elements accessed | Cache lines |
|---|---|---|
| A row i | K elements | K/8 |
| B column j | K elements (stride N) | K (one per element!) |
| C(i,j) | 1 element | 1 |
| **Total** | | **K + K/8 + 1 ≈ 1.125 K** |

For K=1024: ~1152 cache lines = ~72 KB > L1 (32 KB). B constantly thrashes L1.

### Reordered inner loop (fixed i, fixed k)

| Array | Elements accessed | Cache lines |
|---|---|---|
| A(i,k) | 1 (register) | 0 |
| B row k | N elements | N/8 |
| C row i | N elements | N/8 |
| **Total per j-tile of 8 elements** | | **2** |

The inner loop processes 8 j-elements per iteration (one cache line of B, one of C). Working set during any 8-element tile = 2 cache lines = 128 bytes, well within L1.

---

## 6. Hardware Prefetching

Hardware prefetchers detect sequential and constant-stride access patterns and fetch lines before they are needed, hiding much of the memory latency.

The reordered kernel's stride-1 walk over B row k is the easiest case. The naïve kernel's column walk has a constant stride too (8 KB at N=1024), which stride prefetchers can follow, but it still needs a new cache line for every element and uses only 8 of its 64 bytes — prefetching cannot fix the wasted bandwidth.

---

## 7. What Comes Next: Loop Tiling

Even the reordered kernel has an issue for very large matrices: the outer k-loop causes row `i` of C to be evicted from L1 between k-iterations if N is large. Loop tiling (blocking) addresses this by processing a small tile (e.g. 64×64 elements) that stays cache-resident before moving on. This is Level 1, `gemm_blocked`.

```
Tiled access pattern (tile size T_r × T_c):

  for i_block in [0, M, T_r):
    for k_block in [0, K, T_k):
      for j_block in [0, N, T_c):
        // This 3D tile of A, B, C fits in L1:
        for i in [i_block, min(i_block+T_r, M)):
          for k in [k_block, min(k_block+T_k, K)):
            for j in [j_block, min(j_block+T_c, N)):
              C(i,j) += A(i,k) * B(k,j)
```

With a tile size of 64×64 doubles, the working set is `3 * 64 * 64 * 8 = 98 KB`. This fits comfortably in L2 (256 KB) and significantly reduces L3 traffic compared to the reordered kernel.

---

## 8. DRAM Bandwidth Ceiling

The maximum achievable GFLOP/s for a memory-bound kernel is bounded by:

```
Peak GFLOP/s ≤ (DRAM bandwidth GB/s) × (Arithmetic intensity FLOP/byte)
```

For naïve GEMM in the worst case, where every access to B misses all the way to DRAM:
- Each multiply-add (2 FLOPs) pulls a 64-byte line to use one 8-byte double → arithmetic intensity ≈ 2 / 64 ≈ 0.03 FLOP/byte
- DRAM bandwidth ≈ 50 GB/s (dual-channel DDR4-3200; 25.6 GB/s per channel)
- Peak ≈ 50 × 0.03 = ~1.6 GFLOP/s

Caches soften this at moderate N (part of B stays resident), but the trend shows in the measurements: naïve f64 on M4 Max falls from 9.5 GFLOP/s at N=64 to 0.66 at N=4096 ([benchmarks.md](benchmarks.md#apple-m4-max)).

For the reordered kernel, effective bandwidth is much higher (from caches), but tiling is needed to reach the compute roofline of:
```
Peak compute = cores × SIMD width × FMA throughput × frequency
```
This motivates the SIMD implementations in Levels 2–4 (AVX2, AVX-512, NEON/SVE).


---

## 9. GPU Memory Hierarchy

The same latency ladder exists on a GPU, with one extra tier — per-SM shared memory — that the CUDA kernels in [src/gemm/README.md](../src/gemm/README.md#algorithm-9--cuda-kernels-cudahpp--srccudagemm_kernelscu) exploit explicitly.

```
                  ┌──────────────────────────────────────────────────┐
                  │  GPU (this repo's: NVIDIA RTX 5080, sm_120)      │
  ┌───────────────┴──────────────┐  ┌──────────────────────────┐     │
  │  SM 0  (Streaming Multiproc) │  │  SM 1  …  SM 83          │     │
  │  ┌─────────┐  ┌───────────┐  │  │                          │     │
  │  │Registers│  │  Shared   │  │  │   (same structure)       │     │
  │  │ 256 KB  │  │  Memory / │  │  │                          │     │
  │  │ per SM  │  │  L1 Cache │  │  │                          │     │
  │  │  ~1 cy  │  │ ≤100 KB   │  │  │                          │     │
  │  │         │  │  shared   │  │  │                          │     │
  │  └─────────┘  └─────┬─────┘  │  │                          │     │
  └─────────────────────┼────────┘  └──────────────────────────┘     │
                        │  L2 Cache: 64 MB shared across SMs         │
                        │  GDDR7 DRAM: 16 GB, 960 GB/s               │
                        └────────────────────────────────────────────┘
```

(Values from `cudaGetDeviceProperties` on the RTX 5080 used for
[benchmarks.md](benchmarks.md#nvidia-rtx-5080--cuda). Shared memory and L1
share one on-chip array per SM; up to 100 KB of it can be shared memory.)

Warp coalescence: 32 threads in a warp issue memory loads together. If consecutive threads access consecutive addresses, the hardware merges them into a single 128-byte transaction. In our kernels, thread `(ty, tx)` computes `C(i, j)` where `j = blockCol*TILE + tx` — so consecutive threads in a warp differ only in `tx`, giving coalesced access to B rows and C rows.
