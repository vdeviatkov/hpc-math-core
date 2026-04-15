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
| `avx2.hpp` | tiled i → k → j + explicit FMA | ✅ AVX2 micro-kernel (4×16 f32 / 4×8 f64) |

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
| `gemm_naive` (f64) | 54.3 µs | 12944 µs | 103159 µs | 887768 µs | 195990001 µs |
| `gemm_reordered` (f64) | 19.7 µs | 2060 µs | 16171 µs | 129521 µs | 8355984 µs |
| `gemm_blocked` (f64) | 19.2 µs | 1313 µs | 12641 µs | 110532 µs | 6778724 µs |
| `gemm_avx2` (f64) | — | — | — | — | — |
| `gemm_naive` (f32) | 55.4 µs | 12190 µs | 108410 µs | 822792 µs | 204547448 µs |
| `gemm_reordered` (f32) | 6.12 µs | 1052 µs | 8234 µs | 65465 µs | 4189995 µs |
| `gemm_blocked` (f32) | 6.12 µs | 403 µs | 5224 µs | 51943 µs | 4382110 µs |
| `gemm_avx2` (f32) | — | — | — | — | — |

*Run `./build/benchmarks/bench_gemm --benchmark_filter="Avx2"` to fill in the AVX2 rows.*

---

## Algorithm 4 — Explicit AVX2 FMA `gemm_avx2`

### Why explicit intrinsics after auto-vectorisation?

The blocked kernel already achieves ~83 GFLOP/s for f32 at small N via compiler
auto-vectorisation. However the compiler faces four obstacles:

1. **Alias analysis** — without `__restrict__` it may insert runtime checks or fall back to scalar.
2. **Register pressure** — the compiler may spill accumulators to the stack across a tile.
3. **Unroll decisions** — the compiler's cost model may not unroll by exactly the factor needed to keep both FMA ports busy simultaneously.
4. **Load/store scheduling** — explicit code controls exactly when prefetch hints appear relative to compute.

### AVX2 register file

```
256-bit YMM register holds:
  f32: [f0][f1][f2][f3][f4][f5][f6][f7]   ← 8 floats,  4 B each
  f64: [d0][d1][d2][d3]                   ← 4 doubles, 8 B each

FMA: vfmadd231ps  acc = acc + a * b   (2 FLOP, 0.5 cycle throughput on Skylake)
Peak single-core:
  f32: 2 ports × 8 lanes × 2 FLOP/FMA = 32 FLOP/cycle → 96 GFLOP/s @ 3 GHz
  f64: 2 ports × 4 lanes × 2 FLOP/FMA = 16 FLOP/cycle → 48 GFLOP/s @ 3 GHz
```

### Micro-kernel register tile (f32, 4×16)

```
C tile in YMM registers (8 accumulators):

        j+0..7     j+8..15
  i+0: [c00 YMM] [c01 YMM]
  i+1: [c10 YMM] [c11 YMM]
  i+2: [c20 YMM] [c21 YMM]
  i+3: [c30 YMM] [c31 YMM]

Per k-iteration:
  broadcast A(i+r, k)  → a0..a3   (4 YMM, each holds 8 copies of one scalar)
  load B(k, j..j+7)    → b0       (1 YMM)
  load B(k, j+8..j+15) → b1       (1 YMM)
  8× vfmadd231ps       → c_rq += a_r * b_q

Total YMM in use: 8 (acc) + 4 (broadcast) + 2 (B) = 14 of 16 available.
```

### Portability

| Target | `__AVX2__` defined? | Behaviour |
|---|---|---|
| Intel Haswell+ / AMD Zen1+ | ✅ (with `-march=native`) | Full AVX2 FMA micro-kernel |
| Apple Silicon (M1/M2/M3/M4) | ❌ (ARM NEON, not x86) | Falls back to `gemm_blocked` — correct, no SIGILL |
| Pre-Haswell x86 | ❌ | Falls back to `gemm_blocked` |

---

## Upcoming Steps

| Step | Kernel | Description |
|---|---|---|
| ✅ 0 | `gemm_naive` / `gemm_reordered` | Loop reordering, cache-friendly access |
| ✅ 1 | `gemm_blocked` | Loop tiling (L2-resident tiles) |
| ✅ 2 | `gemm_avx2` | Explicit AVX2 FMA intrinsics, 4×16 f32 / 4×8 f64 register tile |
| 🔜 3 | `gemm_avx512` | 512-bit AVX-512 + software prefetch |
| 🔜 4 | `gemm_cuda` | Tiled CUDA kernel (shared memory) |

