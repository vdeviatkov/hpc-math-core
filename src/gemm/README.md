# GEMM Kernel Implementations

This directory contains the CPU scalar General Matrix Multiply (GEMM) implementations that form **Step 0** of the `hpc-math-core` benchmark suite. All kernels compute:

```
C = A × B
```

where A is M×K, B is K×N, and C is M×N (row-major, double-precision).

---

## Files

| File | Loop order | Status |
|---|---|---|
| `naive.hpp` | i → j → k | ✅ Baseline |
| `reordered.hpp` | i → k → j | ✅ Cache-friendly |
| `blocked.hpp` | tiled i → k → j | ✅ Cache-blocked (L2-resident tiles) |

---

## Algorithm 1 — Naïve `gemm_naive` (i-j-k)

### Loop structure

```
for i in [0, M):
  for j in [0, N):
    acc = 0
    for k in [0, K):
      acc += A(i,k) * B(k,j)   ← inner loop
    C(i,j) = acc
```

### Memory access pattern (inner loop, fixed i and j)

```
Variable  │ Index expression  │ Stride as k advances │ Cache behaviour
──────────┼───────────────────┼──────────────────────┼────────────────────
A(i, k)   │ data[i*K + k]     │ +1 element (8 B)     │ ✅ Sequential
B(k, j)   │ data[k*N + j]     │ +N elements (8N B)   │ ❌ Column stride
C(i, j)   │ data[i*N + j]     │ 0 (invariant)        │ ✅ Register
```

### Cache analysis

Reading `B(k, j)` as `k` increases steps through memory in strides of `N * sizeof(double)` bytes — for N=1024 that is **8 KB per step**, nearly 128× the cache-line width of 64 bytes.

```
B memory (N=8, row-major):

 k=0 → │B(0,0)│B(0,1)│B(0,2)│B(0,3)│B(0,4)│B(0,5)│B(0,6)│B(0,7)│  ← cache line 0
 k=1 → │B(1,0)│B(1,1)│B(1,2)│B(1,3)│B(1,4)│B(1,5)│B(1,6)│B(1,7)│  ← cache line 1
 k=2 → │B(2,0)│...
         ↑
         Only this element (column j) is used from each cache line!
         Cache-line utilisation = 1/8 = 12.5 %
```

**Consequence:** For an N=1024 matrix, `B` alone causes ~1 M cache-line misses per output row of C. Most of those fall in DRAM (working set > L3).

---

## Algorithm 2 — Cache-Friendly `gemm_reordered` (i-k-j)

### Loop structure

```
for i in [0, M):
  for k in [0, K):
    a_ik = A(i, k)        ← hoist scalar into register
    for j in [0, N):
      C(i,j) += a_ik * B(k,j)   ← inner loop
```

### Memory access pattern (inner loop, fixed i and k)

```
Variable  │ Index expression  │ Stride as j advances │ Cache behaviour
──────────┼───────────────────┼──────────────────────┼────────────────────
A(i, k)   │ scalar register   │ —                    │ ✅ Free (register)
B(k, j)   │ data[k*N + j]     │ +1 element (8 B)     │ ✅✅ Sequential
C(i, j)   │ data[i*N + j]     │ +1 element (8 B)     │ ✅✅ Sequential
```

### Cache analysis

```
B memory (N=8, row-major):

 k=0 → │B(0,0)│B(0,1)│B(0,2)│B(0,3)│B(0,4)│B(0,5)│B(0,6)│B(0,7)│  ← entire row consumed
 k=1 → │B(1,0)│B(1,1)│...
         ↑↑↑↑↑↑↑↑↑↑↑
         ALL 8 elements per cache line are used!
         Cache-line utilisation = 8/8 = 100 %
```

The inner j-loop reads row `k` of `B` and row `i` of `C` — both sequentially. The hardware prefetcher can predict the access pattern exactly and issue prefetch requests before the data is needed, keeping the pipeline fully occupied.

**L1 working set** during the inner j-loop: ≈ 2 cache lines (one from B row k, one from C row i) = 128 bytes, always resident in L1.

### Why the hoist matters

Without the explicit `a_ik` hoist, the compiler must prove that writing to `C(i,j)` does not alias `A` or `B` to eliminate the re-read. With `const T a_ik = A(i,k)` the programmer makes this explicit, guaranteeing register allocation even without `__restrict__` or LTO.

---

## Algorithm 3 — Cache-Blocked `gemm_blocked` (tiled i-k-j)

### Loop structure

```
for i_blk in [0, M, TILE):
  for k_blk in [0, K, TILE):
    for j_blk in [0, N, TILE):
      // --- micro-kernel: TILE × TILE × TILE sub-problem ---
      for i in [i_blk, min(i_blk+TILE, M)):
        for k in [k_blk, min(k_blk+TILE, K)):
          a_ik = A(i, k)                 ← scalar register
          for j in [j_blk, min(j_blk+TILE, N)):
            C(i,j) += a_ik * B(k,j)     ← inner loop
```

Default tile: `TILE = 64` (compile-time constant; override with `-DHPC_GEMM_TILE=<N>`).

### Why the reordered kernel still leaves performance on the table

The i-k-j reordered kernel streams B and C sequentially in the inner j-loop. However for large N, row-i of C and row-k of B are each **N × 8 = 8 KB** (at N=1024). Together they nearly fill a 32 KB L1:

```
L1 pressure (reordered, N=1024):
  C row i  = 8 KB
  B row k  = 8 KB
  ──────────────
  Total   = 16 KB  ← tight; any other activity causes evictions
```

When the outer k-loop increments, row-i of C may be partially evicted and must be reloaded on the next k-iteration. For large N this produces a steady drip of L1 misses through the entire computation.

### How tiling fixes it

By capping the inner loops to `TILE` elements, the entire working set shrinks to a predictable size:

```
Working set of the micro-kernel (TILE=64, double):
  A tile: TILE × TILE × 8 B =  32 KB
  B tile: TILE × TILE × 8 B =  32 KB
  C tile: TILE × TILE × 8 B =  32 KB
  ─────────────────────────────────────
  Total:                        96 KB  ← fits comfortably in L2 (4096 KB)
```

The B tile is reused `TILE` times (once per i-row in the tile) before the j_blk loop advances — so each cache line of B is loaded once and used `TILE` times, compared to once per i-iteration in the reordered kernel.

```
B reuse factor comparison (per cache line):

  naïve:      1 use  / load  (column walk, always cold)
  reordered:  8 uses / load  (sequential row, but large working set)
  blocked:   TILE uses / load (entire tile stays in L2 for all T_r passes)
```

### Memory access diagram (tile level, TILE=4 for illustration)

```
B layout (N=8):
┌──────────────────────────────────────┐
│ B(0,0) B(0,1) B(0,2) B(0,3) │ B(0,4)…│  ← j_blk=0 tile  (kept in L2 for
│ B(1,0) B(1,1) B(1,2) B(1,3) │ B(1,4)…│    all TILE i-rows)
│ B(2,0) B(2,1) B(2,2) B(2,3) │ B(2,4)…│
│ B(3,0) B(3,1) B(3,2) B(3,3) │ B(3,4)…│
└──────────────────────────────────────┘
         ↑ inner j-loop reads this
           contiguous tile repeatedly
```

---

## Expected Performance Comparison

> Measured on: 16 × 24 MHz cores · L1 Data 64 KiB · L2 Unified 4096 KiB (×16).
> Release build, Apple Clang, C++20. Default tile = 64.

| Kernel | N=64 | N=256 | N=512 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| `gemm_naive` | 85.3 µs | 9544 µs | 103163 µs | 856731 µs | 197161840 µs |
| `gemm_reordered` | 19.5 µs | 2050 µs | 16476 µs | 131522 µs | 8440910 µs |
| `gemm_blocked` | — | — | — | — | — |

*Run `./build/benchmarks/bench_gemm` to fill in the blocked row with your hardware's numbers.*

---

## Upcoming Steps

| Step | Kernel | Description |
|---|---|---|
| ✅ 0 | `gemm_naive` / `gemm_reordered` | Loop reordering, cache-friendly access |
| ✅ 1 | `gemm_blocked` | Loop tiling (L2-resident tiles) |
| 🔜 2 | `gemm_avx2` | 256-bit AVX2 FMA intrinsics |
| 🔜 3 | `gemm_avx512` | 512-bit AVX-512 + software prefetch |
| 🔜 4 | `gemm_cuda` | Tiled CUDA kernel (shared memory) |

