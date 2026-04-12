# Cache Behaviour of Matrix Multiplication

This document is the theoretical companion to the kernel implementations in `src/gemm/`. It explains **why** loop order matters for performance, building up from first principles.

---

## 1. The Memory Hierarchy

Modern CPUs do not read from DRAM directly. Data travels through a hierarchy of ever-faster, ever-smaller caches:

```
Registers     ~0 cycles    handful of values
   ↕
L1 cache      ~4 cycles    32 KB   (per core, typically)
   ↕
L2 cache     ~12 cycles   256 KB  (per core, typically)
   ↕
L3 cache     ~40 cycles     8–64 MB (shared across cores)
   ↕
DRAM        ~80 ns       GB–TB
```

An algorithm is **compute-bound** when the CPU's arithmetic units are the bottleneck. It is **memory-bound** when the CPU stalls waiting for data from a lower level of the hierarchy. Naïve GEMM is almost always severely memory-bound.

---

## 2. Cache Lines

The unit of transfer between any two adjacent levels of the hierarchy is the **cache line** — always 64 bytes on x86 and ARM CPUs. When you read a single `double` (8 bytes), the CPU loads the surrounding 64 bytes — 8 doubles — into the cache.

```
DRAM layout:
offset  0  8 16 24 32 40 48 56 64 72 80 …
       [d0 d1 d2 d3 d4 d5 d6 d7|d8 d9 …]
        ←── 1 cache line (64 B) ──→

Accessing d0 loads {d0…d7} into L1. Accessing d1…d7 immediately is FREE.
```

This is **spatial locality**: data near a recently-used address is likely to be reused soon. Algorithms that exploit spatial locality use every byte of every loaded cache line.

---

## 3. Row-Major Layout and Matrix Element Addresses

`hpc::Matrix<T>` stores elements in row-major order. For a matrix with `cols` columns:

```
element (i, j)  →  data[ i * cols + j ]
```

**Consecutive elements in the same row** have adjacent memory addresses (stride 1). A full row fits in `cols * sizeof(T)` bytes = `cols * 8` bytes for `double`.

**Consecutive elements in the same column** are separated by `cols * sizeof(T)` bytes — for N=1024 that is **8 KB**, spanning 128 cache lines.

This asymmetry is the root cause of naive GEMM's poor performance.

---

## 4. Reuse Distance

**Reuse distance** is the number of distinct memory addresses accessed between two accesses to the same address. If the reuse distance exceeds the number of cache lines in a cache level, that level will not hold the data from the first access when the second access occurs — a **cache miss**.

### Naive GEMM (i-j-k): reuse distance of B

For a fixed `j`, the inner k-loop accesses `B(0,j), B(1,j), …, B(K-1,j)`. Between `B(0,j)` and `B(1,j)` the loop also touches:
- `A(i, 0)` through `A(i, K-1)` → K addresses in A

Between `B(k,j)` and `B(k+1,j)`:
- distance = N (because `B(k,j)` is at `data[k*N+j]` and `B(k+1,j)` is at `data[(k+1)*N+j]`)
- In terms of cache lines: reuse distance = N/8 cache lines of B row k

For N=1024: after loading one element of column j of B, the next access to the same cache line will not occur until `j` advances past 7 — but by then the outer loop has moved on. Effectively **every access to B is a cache miss**.

### Reordered GEMM (i-k-j): reuse distance of B

The inner j-loop accesses `B(k,0), B(k,1), …, B(k,N-1)` — a sequential walk across row k.

Between `B(k, j)` and `B(k, j+1)`: stride = 1 element = 8 bytes. The hardware prefetcher trivially predicts this and issues prefetch requests speculatively. Reuse distance within a single cache line = 0 additional accesses between the 8 elements of that line.

---

## 5. Working Set Analysis

The **working set** of a loop nest is the set of cache lines touched in one execution of the inner loop.

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

The inner loop processes 8 j-elements per iteration (one cache line of B, one of C). Working set during any 8-element tile = **2 cache lines = 128 bytes**, trivially in L1.

---

## 6. Hardware Prefetching

Modern CPUs include **hardware stream prefetchers** that detect sequential (stride-1) access patterns and automatically issue load requests before the data is needed. This hides the DRAM latency entirely — when the CPU needs a cache line, it is already in L1.

The naïve kernel's column-stride access to B **defeats** the hardware prefetcher: the stride of 8 KB (N=1024) looks like random access, so no prefetch is issued.

The reordered kernel's stride-1 access to B row k **activates** the prefetcher on every row. On a typical 4-wide prefetcher, the effective DRAM latency is reduced from ~80 ns to ~5–10 ns.

---

## 7. What Comes Next: Loop Tiling

Even the reordered kernel has an issue for very large matrices: the outer k-loop causes row `i` of C to be evicted from L1 between k-iterations if N is large. **Loop tiling** (blocking) addresses this by processing a small tile (e.g. 64×64 elements) that fits entirely in L1 before moving on. This is the subject of Step 1.

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

For naïve GEMM with N=1024:
- Arithmetic intensity ≈ N³ multiplies / (N² cache-line loads × 64 B/line) ≈ 0.125 FLOP/byte
- DRAM bandwidth ≈ 50 GB/s (DDR4-3200 per channel)
- Peak ≈ 50 × 0.125 = **6.25 GFLOP/s**

For the reordered kernel, effective bandwidth is much higher (from caches), but tiling is needed to reach the **compute roofline** of:
```
Peak compute = cores × SIMD width × FMA throughput × frequency
```
This motivates the SIMD implementations in Steps 2 and 3.

