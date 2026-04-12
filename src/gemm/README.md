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

## Expected Performance Comparison

> Measured on a single core, Release build (`-O3`). Actual numbers depend on CPU model, cache sizes, and thermal state.

| Kernel | N=64 | N=256 | N=512 | N=1024 |
|---|---|---|---|---|
| `gemm_naive` | ~fast (fits L1) | slower | slow | very slow |
| `gemm_reordered` | ~same | 3–5× faster | 4–8× faster | 4–8× faster |

Run `./build/benchmarks/bench_gemm` to populate this table with your hardware's actual numbers.

---

## Upcoming Steps

| Step | Kernel | Description |
|---|---|---|
| 1 | `gemm_blocked` | Loop tiling / blocking to exploit L1/L2 capacity |
| 2 | `gemm_avx2` | 256-bit AVX2 SIMD inner loop |
| 3 | `gemm_avx512` | 512-bit AVX-512 SIMD + prefetch intrinsics |
| 4 | `gemm_cuda` | cuBLAS-comparable CUDA tiled kernel |

