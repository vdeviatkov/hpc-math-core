# GEMM Kernel Implementations

This directory contains all CPU GEMM implementations for the `hpc-math-core` benchmark suite.
All kernels compute **C = A × B** where A is M×K, B is K×N, C is M×N (row-major, `float` or `double`).

---

## Files

| File | Kernels | ISA guard |
|---|---|---|
| `naive.hpp` | `gemm_naive` | — (scalar, always) |
| `reordered.hpp` | `gemm_reordered` | — (scalar, always) |
| `blocked.hpp` | `gemm_blocked` | — (scalar, always) |
| `avx2.hpp` | `gemm_avx2_naive` · `gemm_avx2_reordered` · `gemm_avx2_blocked` | `__AVX2__` |
| `avx512.hpp` | `gemm_avx512_naive` · `gemm_avx512_reordered` · `gemm_avx512_blocked` | `__AVX512F__` |
| `neon.hpp` | `gemm_neon_naive` · `gemm_neon_reordered` · `gemm_neon_blocked` | `__ARM_NEON` |
| `sve.hpp` | `gemm_sve_naive` · `gemm_sve_reordered` · `gemm_sve_blocked` | `__ARM_FEATURE_SVE` |
| `prefetch.hpp` | `gemm_blocked_prefetch` · `gemm_avx2_blocked_prefetch` · `gemm_avx512_blocked_prefetch` · `gemm_neon_blocked_prefetch` · `gemm_sve_blocked_prefetch` | per ISA |
| `cuda.hpp` | `gemm_cuda_naive` (L0) · `gemm_cuda_reordered` (L0b) · `gemm_cuda_blocked` (L1) · `gemm_cuda_reg_tile` (L2) · `gemm_cuda_double_buf` (L3) · `gemm_cuda_wmma` (L4, fp32) | `HPC_HAVE_CUDA` |

Each kernel gracefully degrades at runtime: if the required ISA is not present the benchmark prints
`SKIPPED` and the test calls `GTEST_SKIP()` — no SIGILL, no silent wrong answer.

---

## Fallback chain

```
gemm_sve_*
  └─ falls back to gemm_neon_*       (if __ARM_FEATURE_SVE not defined)
       └─ falls back to gemm_avx2_*  (if __ARM_NEON not defined)
            └─ falls back to gemm_*  (scalar, always compiles)

gemm_avx512_*
  └─ falls back to gemm_avx2_*       (if __AVX512F__ not defined)
       └─ falls back to gemm_*

gemm_cuda_*
  └─ stub library: cuda_device_count() == 0 → SKIPPED
```

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

Reading `B(k, j)` steps through memory in strides of `N × sizeof(T)` bytes —
for N=1024 f64 that is **8 KB per step**, 128× a 64-byte cache line.

```
B memory (N=8, row-major):

 k=0 → │B(0,0)│B(0,1)│B(0,2)│B(0,3)│B(0,4)│B(0,5)│B(0,6)│B(0,7)│  ← cache line 0
 k=1 → │B(1,0)│B(1,1)│...
         ↑
         Only column j used per cache line → 12.5% utilisation
```

---

## Algorithm 2 — Cache-Friendly `gemm_reordered` (i-k-j)

### Loop structure

```
for i in [0, M):
  for k in [0, K):
    a_ik = A(i, k)              ← hoist scalar into register
    for j in [0, N):
      C(i,j) += a_ik * B(k,j)  ← inner loop
```

### Memory access pattern (inner loop, fixed i and k)

```
Variable  │ Stride as j advances │ Cache behaviour
──────────┼──────────────────────┼────────────────────
A(i, k)   │ — (register)         │ ✅ Free
B(k, j)   │ +1 element           │ ✅✅ Sequential
C(i, j)   │ +1 element           │ ✅✅ Sequential
```

Both B row k and C row i are accessed sequentially — 100% cache-line utilisation.
The hardware prefetcher predicts the stride exactly and keeps the pipeline full.

### Why the hoist matters

Without the explicit `a_ik` hoist the compiler must prove `C` does not alias `A`
to eliminate the re-read. The explicit scalar guarantees register allocation
even without `__restrict__`.

---

## Algorithm 3 — Cache-Blocked `gemm_blocked` (tiled i-k-j)

### Loop structure

```
for i_blk in [0, M, TILE):
  for k_blk in [0, K, TILE):
    for j_blk in [0, N, TILE):
      for i in [i_blk, min(i_blk+TILE, M)):
        for k in [k_blk, min(k_blk+TILE, K)):
          a_ik = A(i, k)
          for j in [j_blk, min(j_blk+TILE, N)):
            C(i,j) += a_ik * B(k,j)
```

Default `TILE = 64`. Override: `-DHPC_GEMM_TILE=<N>`.

### Why blocking improves on reordered

For large N the reordered working set is `2 × N × sizeof(T)` — 16 KB at N=1024 f64,
tight for a 32–64 KB L1. Blocking caps it to `TILE × TILE × sizeof(T) = 32 KB` per tile,
which fits in L2 and is reused `TILE` times per load.

```
B cache-line reuse factor:
  naive:     1 use / load   (column stride — always cold)
  reordered: 8 uses / load  (full row — large working set at big N)
  blocked:  TILE uses / load (tile stays in L2 for all TILE i-rows)
```

---

## Algorithm 4 — AVX2 Explicit FMA (`avx2.hpp`)

Three kernels mirroring the scalar progression with 256-bit YMM registers (`__AVX2__`).

### `gemm_avx2_naive` — i→j→k, SIMD on k-loop

Gathers B column j via a stack buffer, FMAs with sequential A row load.
**Pedagogical purpose:** SIMD width alone cannot overcome a cache-hostile access pattern.

### `gemm_avx2_reordered` — i→k→j, SIMD on j-loop

`_mm256_broadcast_ss/sd` broadcasts A scalar · `_mm256_loadu` sequential B/C ·
`_mm256_fmadd_ps/pd` FMA. Scalar tail for `N % 8`.

```
f32 peak (Skylake):  8 FLOP/FMA × 2 ports = 16 FLOP/cycle ≈ 8× scalar
f64 peak (Skylake):  4 FLOP/FMA × 2 ports =  8 FLOP/cycle ≈ 4× scalar
```

### `gemm_avx2_blocked` — tiled, 4×16 f32 / 4×8 f64 register tile

Keeps a 4-row × 2-vector C tile in YMM registers across the entire k-tile:

```
        j+0..7     j+8..15
  i+0: [c00 YMM] [c01 YMM]
  i+1: [c10 YMM] [c11 YMM]
  i+2: [c20 YMM] [c21 YMM]
  i+3: [c30 YMM] [c31 YMM]

YMM in use: 8 acc + 4 broadcast + 2 B = 14 of 16
```

Falls back to `gemm_blocked` on non-AVX2 targets.

---

## Algorithm 5 — AVX-512 Explicit FMA (`avx512.hpp`)

Same three-kernel structure as AVX2, extended to 512-bit ZMM registers (`__AVX512F__`).

| | AVX2 | AVX-512 |
|---|---|---|
| Register width | 256 bit | 512 bit |
| f32 lanes | 8 | 16 |
| f64 lanes | 4 | 8 |
| C tile (f32) | 4×16 = 64 elem | 4×32 = 128 elem |
| C tile (f64) | 4×8  = 32 elem | 4×16 = 64 elem |
| Key FMA | `_mm256_fmadd_ps` | `_mm512_fmadd_ps` |

AVX-512 also gains 32 ZMM registers (vs 16 YMM) — more room for accumulators.
Falls back to `gemm_avx2_blocked` on non-AVX-512 targets.

---

## Algorithm 6 — ARM NEON (`neon.hpp`)

NEON Q-registers are fixed 128-bit (unlike SVE). Available on all AArch64 CPUs including Apple Silicon.

| | f32 | f64 |
|---|---|---|
| Lanes | 4 (`float32x4_t`) | 2 (`float64x2_t`) |
| Broadcast | `vdupq_n_f32/f64` | same |
| FMA | `vfmaq_f32/f64` | same |

### `gemm_neon_naive` — i→j→k, NEON on k-loop

Manual gather into Q-register buffer. Cache-hostile — identical lesson to AVX2 naive.

### `gemm_neon_reordered` — i→k→j, NEON on j-loop

`vdupq_n` broadcast · `vld1q` sequential load · `vfmaq` FMA · `vst1q` store.

### `gemm_neon_blocked` — tiled, 4-row × 2-vector register tile

C tile: 4 rows × 2 NEON vectors = **4×8 f32** or **4×4 f64** in Q-registers.
On Apple Silicon this is the highest-throughput CPU kernel (no SVE available).

Falls back to `gemm_avx2_blocked` on non-NEON targets, then to scalar.

---

## Algorithm 7 — ARM SVE / SVE2 (`sve.hpp`)

SVE is **vector-length agnostic (VLA)**: register width VL is implementation-defined
(128–2048 bit) and queried at runtime — the same binary runs correctly on all implementations.

```
svcntw()  — f32 elements per vector (runtime, e.g. 4 on 128-bit, 8 on 256-bit, 16 on 512-bit)
svcntd()  — f64 elements per vector

Predicate: svbool_t pg = svwhilelt_b32(j, N)
           → active lanes only where (j + lane) < N — handles any N with no scalar tail
```

### `gemm_sve_naive` — i→j→k, VLA SIMD on k-loop

Fills a `std::vector<T>(vl)` buffer (scalar loop) then loads as SVE vector.
`svaddv` reduces accumulator to scalar. Same gather penalty as other naive kernels.

### `gemm_sve_reordered` — i→k→j, VLA SIMD on j-loop

`svdup_n` broadcast · `svld1` sequential · `svmla_x` FMA · `svst1` store.
j-loop step `= svcntw/d()` — automatically wider on higher-VL hardware.

```
Expected vs NEON (same clock):
  128-bit SVE (VL=4 f32):  ≈ 1× NEON
  256-bit SVE (VL=8 f32):  ≈ 2× NEON  ← Graviton3
  512-bit SVE (VL=16 f32): ≈ 4× NEON  ← A64FX (Fugaku)
```

### `gemm_sve_blocked` — tiled, VLA register tile (4 rows × 2 SVE vectors)

Tile width `kJStep = 2 × svcntw/d()` scales automatically with VL:

```
128-bit SVE:  kJStep =  8 f32 / call
256-bit SVE:  kJStep = 16 f32 / call  ← Graviton3
512-bit SVE:  kJStep = 32 f32 / call  ← A64FX
```

Predicates `pg0` / `pg1` handle the j-tail inside the micro-kernel — no scalar tail loop.
Falls back to `gemm_neon_blocked` on non-SVE targets.

**Available on:** Graviton3/4, Neoverse V1/V2, A64FX. **Not on** Apple Silicon (M-series).

---

## Algorithm 8 — Software Prefetch (`prefetch.hpp`)

Wraps each ISA family's blocked kernel with `__builtin_prefetch` hints at tunable
distance `PfDist` elements ahead (template parameter, default = 8).

Three streams per kernel:
1. A rows ahead: `__builtin_prefetch(A + (i+PfDist)*lda + k_blk, 0, 1)`
2. B k-rows ahead: `__builtin_prefetch(B + (k+PfDist)*ldb + j_blk, 0, 1)`
3. C write rows ahead: `__builtin_prefetch(C + (i+PfDist)*ldc + j_blk, 1, 1)`

Benchmarks sweep `PfDist ∈ {2, 4, 8, 16}` to find the optimal distance for each machine.
On Apple M the hardware prefetcher is aggressive enough that explicit hints provide no
consistent gain. Measurable improvement expected on Graviton3 and Intel Xeon.

Five variants: `gemm_blocked_prefetch` · `gemm_avx2_blocked_prefetch` ·
`gemm_avx512_blocked_prefetch` · `gemm_neon_blocked_prefetch` · `gemm_sve_blocked_prefetch`

---

## Algorithm 9 — CUDA Kernels (`cuda.hpp` + `src/cuda/gemm_kernels.cu`)

Six GPU kernels across five optimization levels, compiled by nvcc.
On CPU-only machines a stub is compiled; all CUDA benchmarks/tests print
`SKIPPED: 'No CUDA device available'` at runtime.

---

### Level 0 — `gemm_cuda_naive` — global memory, 1 thread per C(i,j)

One thread computes one output element. All A and B data fetched from HBM
on every access. Non-coalesced B column access.

```
Thread (ty, tx): acc = 0
  for k in [0, K): acc += A[i][k] * B[k][j]
C[i][j] = acc
```

**Bottleneck:** HBM bandwidth (~1 TB/s on A100). DRAM-bound at all sizes.
**Typical:** 500 GFLOP/s on RTX 4090 (0.3% of peak).

---

### Level 0b — `gemm_cuda_reordered` — same mapping, explicit row-major inner loop

Mirrors `gemm_reordered` on CPU for naming symmetry. Structurally identical
to naive on GPU. L2 cache absorbs repeated warp accesses at small N; still
DRAM-bound at large N. Included as a CPU-comparison baseline.

---

### Level 1 — `gemm_cuda_blocked` — TILE=16 shared-memory tiling

```
For each k-tile (step TILE=16):
  Block of 16x16 threads cooperatively loads:
    As[ty][tx] = A[i][kTile*16 + tx]   __shared__ As[16][17]  (+1 col padding)
    Bs[ty][tx] = B[kTile*16 + ty][j]   __shared__ Bs[16][17]
  __syncthreads()
  for p in 0..15: acc += As[ty][p] * Bs[p][tx]   <- shared memory (~4 cycle latency)
  __syncthreads()
C[i][j] = acc
```

**+1 column padding** eliminates 16-way shared-memory bank conflicts.
**Global memory reduction:** 2*N^3 / TILE loads vs 2*N^3 for naive = **16x fewer HBM transactions**.
**Bottleneck:** `__syncthreads` overhead + low arithmetic intensity (~2 FLOP/byte).
Each thread owns only 1 output — sync cost amortised over 16 FMAs.
**Typical:** 5-15 TFLOP/s (3-9% of peak).

---

### Level 2 — `gemm_cuda_reg_tile` — 128x128 thread block, 8x8 register tile

**Key insight:** each thread should own many output elements, not just one.
This amortises `__syncthreads` and shared-memory bandwidth over many FMAs.

```
Thread block: 256 threads -> 128x128 output tile of C
Each thread owns: TM=8 rows x TN=8 cols = 64 register accumulators

Shared memory:
  As[BK=16][BM=128]  (A sub-tile, transposed for column access)
  Bs[BK=16][BN=128]  (B sub-tile, row-major)

Per k-step (BK=16):
  Load 128x16 of A into As (all 256 threads participate)
  Load 16x128 of B into Bs (all 256 threads participate)
  __syncthreads()
  for k in 0..15:               <- inner loop over k-step
    reg_A[0..7] = As[k][threadRow*8 .. +8]   <- load 8 A values to registers
    reg_B[0..7] = Bs[k][threadCol*8 .. +8]   <- load 8 B values to registers
    for m in 0..7:
      for n in 0..7:
        reg_C[m][n] += reg_A[m] * reg_B[n]   <- 64 FMAs from registers only
  __syncthreads()
```

**Arithmetic intensity:** ~32 FLOP/byte (vs ~2 for Level 1) -> compute-bound.
**Typical:** 80-120 TFLOP/s (50-75% of peak).

---

### Level 3 — `gemm_cuda_double_buf` — double-buffered register tile

Same register tiling as Level 2, but eliminates the `__syncthreads` bubble
by using two ping-pong shared-memory buffers:

```
As[2][BK][BM], Bs[2][BK][BN]   <- ping-pong buffers

While computing tile k from buffer[cur]:
    Prefetch tile k+1 into buffer[1-cur]
Swap buffers, repeat.
```

**On Ampere+ (sm_80+):** `__pipeline_memcpy_async` / `cp.async` performs
asynchronous global->shared DMA — the copy executes in parallel with FMA
computation, completely hiding memory latency.

**On older GPUs (sm_70..79):** falls back to synchronous loads with
`__syncthreads`; the double-buffer structure is preserved but the overlap
benefit requires hardware async copy support.

**Typical:** 120-140 TFLOP/s (75-85% of peak).

---

### Level 4 — `gemm_cuda_wmma` — Tensor Cores via WMMA (fp32 only, sm_70+)

NVIDIA Tensor Cores (Volta+, SM70+) perform a 16x16x16 matrix-multiply in
a single warp-synchronous instruction — ~8x the throughput of SIMT FP32.

```
// WMMA fragment API (fp16 input, fp32 accumulate):
wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;

wmma::fill_fragment(c_frag, 0.0f);
for each k-tile:
    // Load fp32 A/B tiles into shared memory as fp16 (convert on-the-fly)
    wmma::load_matrix_sync(a_frag, As_ptr, stride);
    wmma::load_matrix_sync(b_frag, Bs_ptr, stride);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // 16x16x16 = 4096 ops
wmma::store_matrix_sync(C_ptr, c_frag, N, wmma::mem_row_major);
```

**Thread block:** 4x4 warps = 512 threads, 64x64 output tile.
**fp16 conversion:** introduces ~1e-3 relative error (test uses relaxed tolerance).
**Falls back** to `gemm_cuda_double_buf` on pre-Volta hardware at runtime.

---

### Performance ladder (RTX 4090, f32, N=4096)

| Kernel | Level | Bottleneck | ~TFLOP/s | % of SIMT peak |
|---|---|---|---|---|
| `gemm_cuda_naive` | 0 | HBM bandwidth | 0.0005 | 0.3% |
| `gemm_cuda_blocked` TILE=16 | 1 | `__syncthreads` + low AI | 5-15 | 3-9% |
| `gemm_cuda_reg_tile` 128x128 | 2 | Compute-bound | 80-120 | 50-75% |
| `gemm_cuda_double_buf` | 3 | Latency hidden | 120-140 | 75-85% |
| `gemm_cuda_wmma` (fp16 TC) | 4 | Tensor Core bound | 250-300 | — (TC peak) |
| cuBLAS SGEMM | ref | All of the above | 140-165 | 85-100% |
| cuBLAS TF32 TC | ref | Tensor Core bound | 300-330 | — |

> **Summary:** Going from Level 1 to Level 2 (register tiling) is the biggest
> single jump — from 3-9% to 50-75% of peak. Level 3 (double buffering + cp.async)
> closes the gap to cuBLAS SIMT. Level 4 (Tensor Cores) requires accepting fp16
> precision in the matrix multiply and achieves 2-3x beyond any SIMT kernel.

---

### Runtime guards

```cpp
// All CUDA kernels:
if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device"); }

// WMMA only (additionally):
if (!hpc::gemm::cuda_has_tensor_cores()) { state.SkipWithMessage("sm_70+ required"); }

// Double-buf reports whether cp.async is active:
state.counters["ampere_async"] = hpc::gemm::cuda_has_ampere() ? 1.0 : 0.0;
```

---

## Benchmark Results

See [../../README.md](../../README.md) for full benchmark tables, speedup analysis, and key observations.
