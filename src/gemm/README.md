# GEMM Kernel Implementations

This directory contains all CPU GEMM implementations for the `hpc-math-core` benchmark suite.
All kernels compute **C = A × B** where A is M×K, B is K×N, C is M×N (row-major, `float` or `double`).

Looking for a quick refresher rather than the full derivation below? See
**[docs/gemm-approaches.md](../../docs/gemm-approaches.md)** — a one-page
summary of every family's cache technique and key intrinsics side by side.

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
| `sme.hpp` | `gemm_sme` (packed, cache-blocked, all ZA tiles, SME2 loads) — **verified, Apple M4 Max** | `__ARM_FEATURE_SME` (+ `-DHPC_ENABLE_SME=ON`) |
| `amx.hpp` | `gemm_amx_naive` · `gemm_amx_reordered` · `gemm_amx_blocked` — **verified, Apple M4 Max, via Accelerate.framework** | `HPC_HAS_AMX` (Apple + Accelerate.framework; on by default) |
| `kleidiai.hpp` | `gemm_kleidiai` (f32 only) — reference, Arm KleidiAI SME2 `FMOPA` micro-kernel | `HPC_HAS_KLEIDIAI` (`HPC_ENABLE_KLEIDIAI=ON`, default when SME works; needs SME2) |
| `prefetch.hpp` | `gemm_blocked_prefetch` · `gemm_avx2_blocked_prefetch` · `gemm_avx512_blocked_prefetch` · `gemm_neon_blocked_prefetch` · `gemm_sve_blocked_prefetch` | per ISA |
| `cuda.hpp` | `gemm_cuda_naive` (L0) · `gemm_cuda_blocked` (L1) · `gemm_cuda_reg_tile` (L2) · `gemm_cuda_double_buf` (L3) · `gemm_cuda_wmma` (L4, fp32) · `gemm_cuda_vectorized` (L5) · `gemm_cuda_mma_ldmatrix` (L6, fp32) · `gemm_cuda_wmma_pipelined` (L7, fp32) · `gemm_cuda_cublas{,_tf32,_fp16}` (reference, not part of the ladder) — **all verified, RTX 5080 (Blackwell sm_120)** | `HPC_HAVE_CUDA` |

The ISA guards are the `HPC_HAS_*` macros from `include/hpc/isa.hpp`, each always defined to 0 or 1.

---

## ISA availability — no silent fallback

A kernel family is compiled only where its ISA is present. Elsewhere the
same names are declared `= delete`:

```cpp
#if !HPC_HAS_AVX2
template <typename T>
void gemm_avx2_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;
#else
template <typename T>
void gemm_avx2_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) { … }
#endif
```

So `gemm_avx2_blocked(A, B, C)` on an ARM build is a compile-time error
(`call to deleted function 'gemm_avx2_blocked'`), never a scalar kernel
quietly timed under an AVX2 name. Consequences:

- **Benchmarks** (`bench_gemm.cpp`) pass the flag as a template parameter
  to `run_gemm<N, T, kHaveAvx2>(…)`; when it is false the kernel lambda sits
  in a discarded `if constexpr` branch and is never instantiated, and the
  row is reported as `SKIPPED` — the full catalogue stays visible.
- **Tests** (`test_gemm.cpp`) wrap each family in `#if HPC_HAS_*`; the test
  count on a machine is exactly the set of kernels that ran on it.
- **`gemm_cuda_*`** is the one runtime case: GPU presence is a property of
  the machine the binary runs on, not of the build, so the CPU-only stub
  reports `cuda_device_count() == 0` and callers `SKIP`. It never computes
  a CPU result under a CUDA name.
- **`gemm_cuda_wmma_pipelined`** has a documented *shape* precondition
  (M, N multiples of 128, K of 32) and falls back to `gemm_cuda_wmma`
  otherwise — see Level 7 below.

All flags are compile-time because every build uses `-march=native` /
`-mcpu=`: the build CPU is the run CPU. Runtime dispatch (cpuid → best
kernel) would be a separate explicit facility, not something hidden inside
each kernel.

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

Eight GPU kernels, one per optimization level (0–7), plus cuBLAS reference
entry points. Compiled by nvcc; on CPU-only machines a stub is compiled and
every CUDA benchmark/test prints `SKIPPED: 'No CUDA device available'` at
runtime. All levels pass `hpc_tests_cuda` on an NVIDIA RTX 5080 (Blackwell,
sm_120, CUDA 13.2); measured throughput is in
[§ NVIDIA RTX 5080 — CUDA](../../docs/benchmarks.md#nvidia-rtx-5080--cuda).

| Level | Kernel | Technique | f32 GFLOP/s, N=4096 ¹ |
|---|---|---|---|
| 0 | `gemm_cuda_naive` | 1 thread → 1 C(i,j), global memory only | 2,608 |
| 1 | `gemm_cuda_blocked` | 16×16 shared-memory tiles | 2,422 |
| 2 | `gemm_cuda_reg_tile` | 128×128 block, 8×8 register tile per thread | 6,583 |
| 3 | `gemm_cuda_double_buf` | Level 2 + `cp.async` double-buffered shared memory | 6,671 |
| 4 | `gemm_cuda_wmma` | Tensor Cores via `wmma::`, 64×64 tiles (fp32 only) | 5,266 |
| 5 | `gemm_cuda_vectorized` | Level 2 + `float4`/`double2` loads + XOR-swizzled shared memory | 5,690 |
| 6 | `gemm_cuda_mma_ldmatrix` | Tensor Cores via raw `ldmatrix` + `mma.sync` PTX (fp32 only) | 5,685 |
| 7 | `gemm_cuda_wmma_pipelined` | `wmma::` with 128×128 tiles, 8 fragments/warp, `cp.async` double buffering (fp32 only) | **9,000** |
| ref | `gemm_cuda_cublas{,_tf32,_fp16}` | cuBLAS SGEMM/DGEMM, TF32, dense FP16 | 8,285 / 8,735 / — |

¹ End-to-end (`cudaMalloc` + H2D + kernel + D2H every call), RTX 5080,
Linux. Compute-only, Level 7 reaches **100.5 TFLOP/s** at N=16384 against
cuBLAS FP16's 120.5 (83%).

---

### Level 0 — `gemm_cuda_naive` — global memory, 1 thread per C(i,j)

One thread computes one output element. All A and B data fetched from
global memory on every access.

```
Thread (ty, tx): acc = 0
  for k in [0, K): acc += A[i][k] * B[k][j]
C[i][j] = acc
```

**Bottleneck:** global-memory bandwidth. Adjacent threads in a warp read
adjacent `B[k][j]`, so B loads coalesce and L1/L2 absorb much of the reuse —
which is why this baseline is harder to beat than it looks.
**Measured:** 2.6 TFLOP/s (f32, N=4096).

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
**Global memory reduction:** 16x fewer global loads than naive.
**Bottleneck:** each thread still owns only 1 output, so two
`__syncthreads()` per k-tile are amortised over just 16 FMAs.
**Measured:** 2.4 TFLOP/s f32 — slightly *below* naive: the sync overhead
isn't paid back with one output per thread. In f64 it is the fastest kernel
at N ≥ 512: the
reduced FP64 pipe of a consumer GPU is the bottleneck there, so the extra
staging of the higher levels buys nothing.

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
  Load 128x16 of A into As (256 threads, strided loop, 8 elements each)
  Load 16x128 of B into Bs (256 threads, strided loop, 8 elements each)
  __syncthreads()
  for k in 0..15:               <- inner loop over k-step
    reg_A[0..7] = As[k][threadRow*8 .. +8]   <- load 8 A values to registers
    reg_B[0..7] = Bs[k][threadCol*8 .. +8]   <- load 8 B values to registers
    for m in 0..7:
      for n in 0..7:
        reg_C[m][n] += reg_A[m] * reg_B[n]   <- 64 FMAs from registers only
  __syncthreads()
```

**Arithmetic intensity:** ~32 FLOP/byte (vs ~2 for Level 1).
**Measured:** 6.6 TFLOP/s f32 — the biggest single step on the ladder
(2.7× Level 1).

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

**On Ampere+ (sm_80+):** `__pipeline_memcpy_async` / `cp.async` copies
global → shared asynchronously, overlapping the copy with FMA compute. The
source must be a real global address; out-of-bounds elements use the
zfill form (copy 0 bytes, zero the destination).

**On older GPUs (sm_70..79):** falls back to synchronous loads with
`__syncthreads`; the double-buffer structure is preserved but the overlap
needs hardware async copy.

For `double` the block tile shrinks to 64×64 to stay within 48 KB of shared
memory, so the launch uses `(BM/8)·(BN/8)` threads: 256 for `float`, 64 for
`double`.
**Measured:** 6.7 TFLOP/s f32 — the fastest plain-FMA kernel.

---

### Level 4 — `gemm_cuda_wmma` — Tensor Cores via WMMA (fp32 only, sm_70+)

NVIDIA Tensor Cores (Volta+, SM70+) perform a 16x16x16 matrix-multiply in
a single warp-synchronous instruction.

```
// WMMA fragment API (fp16 input, fp32 accumulate):
wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::col_major> a_frag;
wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> b_frag;
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
**Two layout rules** the code depends on:
- **No +1 padding** on `As`/`Bs`: `load_matrix_sync` needs the leading
  dimension to be a multiple of 8 `__half` elements (64 is; 65 would fail
  with `cudaErrorMisalignedAddress`).
- **Fragment tags must match the physical layout.** `As` is stored
  transposed (`As[k][m]`), so `a_frag` is `col_major`; `Bs` is stored
  naturally (`Bs[k][n]`), so `b_frag` is `row_major`. A mismatch silently
  transposes the operand — wrong values, no error.
- **Edge tiles are staged.** `store_matrix_sync` always writes a full 16×16
  tile and needs a 32-byte-aligned destination. Tiles wholly inside C with
  `N % 8 == 0` store directly; any other tile is stored to a per-warp
  shared-memory tile and copied out with bounds checks. Without this, sizes
  that aren't multiples of 16 write past the end of C or wrap into the next
  row (`compute-sanitizer` flags it; covered by `CudaWmmaFloat.EdgeTiles`).

**Falls back** to `gemm_cuda_double_buf` on pre-Volta hardware at runtime.
**Measured:** 5.3 TFLOP/s — below the FMA kernels at this size: with 64×64
tiles, one fragment per warp and a single buffer, there is too little work
per synchronization to keep the Tensor Cores busy. Level 7 addresses that.

---

### Level 5 — `gemm_cuda_vectorized` — float4/double2 loads + shared-memory XOR swizzle

Same register-tile shape as Level 2 (128x128 block, 8x8 per thread), with
two changes: global→shared loads use 128-bit vector instructions instead
of one scalar per thread per element, and shared memory uses an XOR
"swizzle" instead of +1 padding.

```
B's fast dimension (N) matches Bs's fast dimension -> load AND store vectorized:
  Vec v = *reinterpret_cast<const Vec*>(&B[row][colBase]);
  Bs[row][swizzle_slot(row, colBase/W, slots) * W] = v;   // vectorized store

A's fast dimension (K) does NOT match As's fast (transposed to M) -> load
vectorized, scatter-store scalar:
  Vec v = *reinterpret_cast<const Vec*>(&A[row][kBase]);  // 4 (or 2) K-values
  for e in 0..W-1: As[kBase+e][swizzle_slot(kBase+e, row/W, slots)*W + row%W] = v[e];

swizzle_slot(row, slot, slots) = slot ^ (row & (slots-1))   // self-inverse XOR
```

**Correctness does not depend on bank-conflict elimination**: the same
`swizzle_slot()` call is used at every write site and every read site, so
whatever permutation it computes is applied and undone consistently.
**Requires K and N to be multiples of the vector width** (4 for float, 2 for
double) for the vectorized loads to stay 16-byte aligned; the host dispatch
falls back to `gemm_cuda_reg_tile` otherwise.
**Measured:** 5.7 TFLOP/s f32 — *slower* than Level 2 (6.6). Not profiled;
the likely cost is the swizzle's index arithmetic on every shared-memory
read in the k-loop, while A's transposed scatter-store still stays scalar.

---

### Level 6 — `gemm_cuda_mma_ldmatrix` — raw Tensor Cores via mma.sync + ldmatrix (fp32 only, sm_80+)

Computes the same thing as Level 4 (fp16 in, fp32 accumulate) one level
below the WMMA C++ API. Here the per-lane register mapping is written by
hand, so a mistake produces wrong values rather than a build or launch
error — the trade-off this level exists to show.

```
// ldmatrix.x4: hardware distributes an 8x8x4-quadrant tile across the warp's
// 32 threads into the EXACT registers mma.sync expects -- no manual
// per-element fragment placement (unlike a from-scratch tensor-core kernel):
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {a0,a1,a2,a3}, [a_addr];       // A, .row -- no .trans (A stored natural row-major here)
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {b0,b1}, [b_addr];      // B, .col -- .trans (B stored natural row-major, needs transposing load)
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {d0..d3}, {a0..a3}, {b0,b1}, {d0..d3};

// A operand's per-lane address (chunk = lane/8 selects which of ldmatrix.x4's
// 4 registers; the row-quadrant bit must be the FAST-varying one -- quadIdx%2,
// not quadIdx/2 -- to land each 8x8 chunk in the register mma.sync expects):
quadIdx = lane / 8; quadRow = lane % 8;
aM = warpRow*16 + quadRow + (quadIdx % 2) * 8;   // M offset
aK =              (quadIdx / 2) * 8;              // K offset
```

Native tile is **16x8x16** (not WMMA's 16x16x16 — mma.sync's f16 shape is
narrower in N), so each warp issues two side-by-side MMAs to cover the
same 16x16 area WMMA computes in one call. Falls back to `gemm_cuda_wmma`
on sm_70-75 (Volta/Turing, which lack the m16n8k16 shape).
**Measured:** 5.7 TFLOP/s — 8% above Level 4, but the same order: same tile
sizes and single buffering, so dropping to PTX alone doesn't remove the
bottleneck.

---

### Level 7 — `gemm_cuda_wmma_pipelined` — pipelined Tensor Cores via WMMA (fp32 only, sm_70+)

Levels 4 and 6 reach ~5 TFLOP/s, while cuBLAS's dense-FP16 Tensor Core path
reaches ~120 TFLOP/s compute-only on the same GPU
([§ Reference cuBLAS](../../docs/benchmarks.md#reference-cublas--the-achievable-ceiling)).
All three use fp16 Tensor Cores; the gap is pipelining and tile size. This
kernel closes most of it the way CUTLASS-style kernels do, while staying on
the documented `wmma::` C++ API, where the compiler manages the fragment
register mapping.

Changes relative to Level 4:

```
Level 4 (gemm_cuda_wmma)              Level 7 (gemm_cuda_wmma_pipelined)
-------------------------------       -------------------------------------
64x64 block tile, BK=16               128x128 block tile, BK=32
4x4 warps, 1 fragment/warp (16x16)    4x2 warps, 8 fragments/warp (32x64)
Single-buffered (load, sync, MMA)     Double-buffered via cp.async
fp32->fp16 converted per-tile,        fp32->fp16 converted ONCE up front
  synchronous scalar load               (kernel_f32_to_f16, global mem),
                                         then 16-byte cp.async chunk loads
As stored TRANSPOSED (As[k][m])       As stored NATURAL (As[m][k]) --
  -> a_frag must be col_major           -> a_frag must be row_major
                                         (opposite tag, correct for the
                                         opposite physical layout -- see
                                         below)
```

**1. Bigger thread-block tile (128x128, BK=32 vs 64x64, BK=16).** More
work per shared-memory round trip and `__syncthreads()` pair.

**2. Bigger per-warp tile (32x64 = 8 fragments/warp vs 16x16 = 1
fragment/warp).** 8 warps (256 threads/block), each issuing 8
`wmma::mma_sync` calls per k-sub-step instead of Level 4's 1, amortizing
load/sync overhead across more compute. A/B fragments are register-
blocked and reused the same way `gemm_cuda_reg_tile`'s scalar FMA
micro-kernel reuses `reg_A`/`reg_B`: for each k-sub-step, all 4 `b_frag`
values are loaded once and reused across both `fm` iterations; each
`a_frag` is loaded once and reused across all 4 `fn` iterations.

```cpp
// Per k-sub-step (kSub in {0,1}, since BK=32 = 2 x WMMA's native K=16):
wmma::fragment<matrix_b, ...> b_frag[4];
for fn in 0..3: wmma::load_matrix_sync(b_frag[fn], &Bs[cur][kSub*16][warpCol*64+fn*16], 128);
for fm in 0..1:
    wmma::fragment<matrix_a, ...> a_frag;
    wmma::load_matrix_sync(a_frag, &As[cur][warpRow*32+fm*16][kSub*16], 32);
    for fn in 0..3:
        wmma::mma_sync(c_frag[fm][fn], a_frag, b_frag[fn], c_frag[fm][fn]);
```

**3. cp.async double-buffered shared memory (Ampere+).** The next
k-tile's global->shared copy overlaps the current tile's Tensor Core
compute, with the same control flow as `gemm_cuda_double_buf` (prefetch
tile 0, then each iteration issues the next tile's load before computing on
the current one, and waits for it after). On pre-Ampere Tensor-Core
hardware (sm_70-75) it compiles to a synchronous `float4`-sized copy, still
double-buffered, via the same `#ifdef HPC_HAVE_CP_ASYNC` pattern.

**4. Padded shared-memory leading dimensions (+8 halves).** Unpadded, `As`
(ld = 32 halves = 64 B/row) gives only 2 distinct bank-starts across a
fragment's 16 rows and `Bs` (ld = 128 halves = 256 B/row, exactly two
32-bank cycles) gives 1 — an 8-way and a 16-way conflict. Padding both by 8
halves (`kPipeAsLd`, `kPipeBsLd`) gives 8 distinct bank-starts each, a
2-way conflict. The pad must be 8, not the usual 1: `load_matrix_sync`
needs ld to be a multiple of 8 halves and cp.async needs a 16-byte-aligned
destination. An XOR swizzle — the zero-memory-cost alternative Level 5
uses — can't be expressed through `load_matrix_sync`'s `(pointer, ld)`
interface. Nsight Compute: shared-load conflicts drop from 85% to 1.0% of
wavefronts, and throughput rises from 80.8 to 100.5 TFLOP/s (details in
[§ Removing the bank conflicts](../../docs/benchmarks.md#removing-the-bank-conflicts)).

**Why `As` is untransposed here (unlike Level 4).** cp.async can only copy
a *contiguous* run of bytes to a *contiguous* destination — it cannot
transpose during the copy the way Level 4's per-element scalar load can.
A16 (the pre-converted fp16 copy of A) is row-major (K-contiguous), so `As`
must also be stored K-contiguous (`As[m][k]`), the opposite of Level 4's
`As[k][m]`. Consequently `a_frag` is `wmma::row_major` here, the opposite
tag from Level 4's `col_major` for the same mathematical operand — the tag
follows the physical layout. `Bs`/`b_frag` are unchanged from Level 4
(`Bs[k][n]`, `row_major`).

**No boundary/zfill logic, by design.** This kernel requires M, N to be
**exact multiples of 128** and K an **exact multiple of 32** (no tail
handling); the host dispatch falls back to `gemm_cuda_wmma` otherwise.
Every cp.async transfer moves a full 16-byte (8 x `__half`) chunk — the
largest `__pipeline_memcpy_async` supports — and the exact-multiple
requirement is what makes every source/destination address provably
16-byte-aligned (`cudaMalloc` buffers are >=256-byte aligned; with K/N
multiples of 32/128, every chunk starts at an element offset that's a
multiple of 8, i.e. a byte offset that's a multiple of 16).

**Measured (RTX 5080, compute-only — pre-staged device buffers, no
per-call transfer/malloc/conversion):**

| N | Level 4 (`gemm_cuda_wmma`) | Level 7 (`gemm_cuda_wmma_pipelined`) | cuBLAS dense FP16 | Level 7 vs Level 4 |
|---|---|---|---|---|
| 4096 | ~5 TFLOP/s (end-to-end; not measured compute-only) | 97.2 TFLOP/s | 108.7 TFLOP/s | ~19x |
| 8192 | — | 101.5 TFLOP/s | 117.0 TFLOP/s | ~20x |
| 16384 | — | 100.5 TFLOP/s | 120.5 TFLOP/s | ~20x |

~83% of cuBLAS's dense-FP16 throughput, all on the documented `wmma::` API.
What limits it now is register pressure: 126 registers/thread caps
occupancy at 33% (`Block Limit Registers: 2`). Closing the last ~17% would
need deeper multi-stage pipelining (3-4 stages, not 2) and split-K for very
large K — the territory CUTLASS's template library handles generically.

---

### Reference — cuBLAS

Not part of the ladder: NVIDIA's production GEMM, the realistic ceiling for
the hand-written kernels.

- `gemm_cuda_cublas<T>` — plain SGEMM/DGEMM on the SIMT cores; the ceiling
  for Levels 0–3 and 5 (~39 TFLOP/s f32 compute-only).
- `gemm_cuda_cublas_tf32` — TF32 Tensor Cores via `cublasGemmEx`, fp32 in/out
  (~60 TFLOP/s, sm_80+).
- `gemm_cuda_cublas_fp16` — dense FP16 Tensor Cores, fp32 accumulate; the
  ceiling for Levels 4, 6 and 7 (~120 TFLOP/s, sm_70+).

Each has a raw-device-pointer `*_device` variant used by the compute-only
benchmarks, which time only the GEMM call against pre-staged buffers.
`hpc::Matrix` is row-major and cuBLAS column-major, so every call computes
`Cᵀ = Bᵀ·Aᵀ` over the same memory — no transposes, no copies.

---

### Runtime guards

```cpp
// All CUDA kernels:
if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device"); }

// WMMA / mma_ldmatrix / wmma_pipelined (additionally):
if (!hpc::gemm::cuda_has_tensor_cores()) { state.SkipWithMessage("sm_70+ required"); }  // WMMA, wmma_pipelined
if (!hpc::gemm::cuda_has_ampere())       { state.SkipWithMessage("sm_80+ required"); }  // mma_ldmatrix
// wmma_pipelined ALSO requires M/N exact multiples of 128, K a multiple
// of 32 -- checked in the host dispatch, not a runtime capability guard;
// falls back to gemm_cuda_wmma (correct for any shape) otherwise.

// Double-buf reports whether cp.async is active:
state.counters["ampere_async"] = hpc::gemm::cuda_has_ampere() ? 1.0 : 0.0;
```

---

## Algorithm 10 — ARM SME2 (Scalable Matrix Extension)

**Verified end-to-end on Apple M4 Max** — see [docs/benchmarks.md](../../docs/benchmarks.md) for
measured GFLOP/s. Opt-in via `-DHPC_ENABLE_SME=ON` (see
[§ SME and AMX build flags](../../docs/build.md#sme-and-amx-build-flags)).

### Why this is a different primitive, not "wider NEON/SVE"

Every kernel above — AVX2, AVX-512, NEON, SVE — computes GEMM with vector
**FMA**: broadcast one scalar, multiply against a vector, accumulate into
another vector. Adding lanes makes the vector wider; the operation stays
the same shape.

SME instead computes GEMM with a hardware **outer product**. A single
`FMOPA` instruction takes a column vector `a` (SVL elements) and a row
vector `b` (SVL elements) and accumulates the full SVL×SVL outer product
into a dedicated 2-D accumulator register array called **ZA** — not a
vector register, a whole tile of them:

```
ZA[r][c] += a[r] * b[c]     for r, c in [0, SVL)
```

Looping this over `k = 0..K-1` with `a[k] = A(i0+r, k)` and
`b[k] = B(k, j0+c)` computes an entire SVL×SVL tile of `C` in `K`
instructions instead of `K × SVL` FMAs. This is the same class of
primitive as NVIDIA Tensor Cores (`gemm_cuda_wmma`, Algorithm 9 above) and
Apple's own AMX coprocessor (Algorithm 11, below) — trade a wider, more
specialised instruction for dramatically higher FLOPs/instruction.

### Hardware finding: gather-loads are illegal in SME streaming mode

SME instructions only execute in "Streaming SVE mode" (entered via
`SMSTART`, exited via `SMSTOP` — Clang generates both automatically for a
function marked `__arm_locally_streaming`). The obvious way to build the
`a` column vector — `svld1_gather_index` with a stride-`lda` index vector,
since `A` is row-major and a column is strided — is **not legal** there:
Clang rejects it with *"builtin can only be called from a non-streaming
function"*. Gather/scatter addressing modes are excluded from the
Streaming SVE instruction subset by the architecture itself, not a
NEON/SVE-style limitation. The column vector must instead be assembled
with ordinary scalar loads into a buffer, then loaded contiguously with
`svld1`. In practice that means packing A (see below).

### Hardware finding: `-march=native` silently disables SME on Apple Silicon

`-march=armv9-a+sme2` compiles cleanly on Apple Silicon but the resulting
binary `SIGILL`s at runtime on the very first instruction inside the
streaming region. Clang emits a `CNTD` instruction *outside* streaming
mode to size the ZA-save prologue buffer; `CNTD` is an ordinary
(non-streaming) SVE instruction, and Apple Silicon implements **no
non-streaming SVE unit at all** — only Streaming SVE via SME.
`-mcpu=apple-m4` avoids this by generating a prologue that doesn't need an
outside-streaming SVE instruction. Worse: combining `-march=native` with
`-mcpu=apple-m4` — the repo's default Release flag plus the SME flag —
silently drops the SME/SVE target features altogether rather than
erroring, so `HPC_ENABLE_SME=ON` clears `HPC_MARCH` in CMakeLists.txt in
favour of the verified `-mcpu=` flag. Because these are *runtime* SIGILL
failure modes that a compile-only check cannot catch, this repo's CMake
SME detection actually **compiles and runs** a probe program at configure
time (`check_cxx_source_runs`, not `check_cxx_compiler_flag`) — see
CMakeLists.txt's `HPC_ENABLE_SME` block.

### The kernel: `gemm_sme`

A straightforward SME kernel — one ZA tile, A packed per row-panel, B read
in place — peaks at 386 GFLOP/s f32 / 116 GFLOP/s f64 on one M4 Max core
and falls to 180 GFLOP/s f32 at N=4096. Accelerate runs at ~1,650 / ~410 on
the same core. `gemm_sme` closes most of that gap with four design choices:

**1. All ZA tiles in use.** ZA holds 4 f32 tiles (16×16 each at 512-bit SVL)
or 8 f64 tiles (8×8). With one tile, every `FMOPA` has to wait for the
previous one to finish updating the same accumulator, so the loop runs at
FMOPA *latency*. The micro-kernel keeps every tile busy with independent
outer products:

```
f32: 2×2 tiles → 32×32 C block      f64: 2×4 tiles → 16×32 C block
per k:  a0,a1 = A column (2 vectors)  per k:  a0,a1 = A column (2 vectors)
        b0,b1 = B row    (2 vectors)          b0..b3 = B row (4 vectors)
        za0 += a0⊗b0   za1 += a0⊗b1           za(4i+j) += ai⊗bj   (8 FMOPA)
        za2 += a1⊗b0   za3 += a1⊗b1
```

Each loaded vector now feeds 2 (f32) or 2–4 (f64) FMOPAs instead of one.

**2. A and B both packed, with GotoBLAS cache blocking.**
`jc (Nc) → pc (Kc) → pack B panel → ic (Mc) → pack A block → jr (nr) → ir (mr) → k`.
Inside the k loop both operands are now unit-stride streams. Before, each k
read one B row `ldb` elements away from the last, which is why the old
kernels fell apart at N ≥ 2048. Partial C sums across `pc` blocks are
carried by loading C into ZA (`svld1_hor_za32/64`) before the k loop.
The block sizes (`kSmeMc=128`, `kSmeKc=1024`, `kSmeNc=4096`) came from a
sweep on M4 Max. Large Kc and Nc won because they amortise both the C
reloads and the A repacking. On M4 the SME unit is shared by a P-core
cluster and reads from L2, so blocks are sized for L2, not L1.

**3. Packing outside streaming mode.** Scalar and NEON code is slow in
streaming mode, so packing runs in the ordinary (non-streaming) driver. A
is transposed into column strips with NEON 4×4 (f32) / 2×2 (f64)
in-register transposes, which doubled packing bandwidth over a scalar loop
(12 → 26 GB/s). Only the macro-kernel runs streaming: one
`SMSTART`/`SMSTOP` per Mc×Nc×Kc block. `svcntsw()`/`svcntsd()` compile to
`RDSVL`, an SME instruction that is legal outside streaming mode, so the
driver can size buffers with them.

**4. SME2 multi-vector loads.** `svld1_x2` fetches two vectors in one
`LD1W {z0.s-z1.s}` / `LD1D {z0.d-z1.d}`. The f32 inner loop is 2 loads +
4 FMOPAs; f64 is 3 loads + 8 FMOPAs. SME2 is required: `HPC_HAS_SME` checks
`__ARM_FEATURE_SME2` as well as `__ARM_FEATURE_SME`.

**One template for both precisions.** `macro_kernel<T>` covers f32 and
f64. The only shape difference is `kCols<T>` (2 or 4 tile columns); one
`if constexpr` adds f64's extra four FMOPAs. Small `__arm_inout("za")`
helpers (`mopa`, `move_tile`, `move_column`, `move_c`) hide the
`za32`/`za64` intrinsic split. A 2×2 layout for f64 as well (4 of 8 tiles)
would remove even that branch, but measured 9% slower at N=4096.

Edges: packing zero-pads partial strips, so FMOPAs always run with an
all-true predicate. Only the C transfers into and out of ZA are predicated.

**Pitfall: every streaming helper needs a ZA attribute.** A
`__arm_streaming` helper without `__arm_preserves("za")` /
`__arm_inout("za")` is "private-ZA". Clang refuses to inline it into a
ZA-owning caller and instead wraps every call in a lazy ZA save
(`TPIDR2_EL0` + `smstart za`). With the load helper in the k loop, that
cuts throughput from ~1,290 to ~270 GFLOP/s f32.

**Measured (single core, M4 Max):** 1,450 GFLOP/s f32 / 410 GFLOP/s f64 at
N=1024 (84–88% of single-threaded Accelerate f32 and on par in f64 for
N ≥ 512), holding 1,345 / 404 at N=4096. Comparison against Accelerate,
and KleidiAI:
[docs/benchmarks.md § Matrix engines, single core](../../docs/benchmarks.md#matrix-engines-single-core).

### Hardware availability

Apple M4 / M4 Pro / M4 Max (SME2, 512-bit SVL) is, as of this writing,
essentially the only shipping SME2 hardware widely available to individual
developers. Not on Apple M1/M2/M3, AWS Graviton3/4, Fujitsu A64FX, or
x86 — there `gemm_sme` is declared `= delete` (`HPC_HAS_SME == 0`).

---

## Algorithm 11 — Apple AMX (via Accelerate.framework)

**Verified on Apple M4 Max** — see [docs/benchmarks.md](../../docs/benchmarks.md) for measured
GFLOP/s (up to 3.3 TFLOP/s f32). On by default on Apple platforms
(`HPC_ENABLE_AMX`, see
[§ SME and AMX build flags](../../docs/build.md#sme-and-amx-build-flags)).

### Which "AMX" this is

"AMX" names two, architecturally unrelated, matrix-multiply accelerators
that happen to share an acronym. Intel AMX is a public x86 ISA extension
(tile registers + TMUL, programmed via `<immintrin.h>` intrinsics,
Sapphire Rapids+ only). **Apple AMX** — the Apple Matrix coprocessor
present in every Apple Silicon SoC since the M1 — is what this file
targets, and it works completely differently from a build/programming
perspective: Apple has never published instruction-level documentation or
an ACLE-style intrinsic header for it (unlike ARM SME, which is a public,
documented ISA — see Algorithm 10 above). The instruction encodings are
known only through third-party reverse engineering and are not something
this repository emits directly. The one Apple-sanctioned, stable way to
benefit from the AMX coprocessor's throughput is **Accelerate.framework**
— its BLAS (`cblas_sgemm`/`cblas_dgemm`) is Apple's own implementation,
and Apple's own performance guidance points to Accelerate for matrix math
on Apple Silicon; the reverse-engineering community has identified that it
dispatches to AMX blocks internally. `gemm_amx_*` in this file is
therefore a thin, verified wrapper around Accelerate's BLAS — not a
hand-written tile-multiply kernel — and it answers a different question
than every other family in this repo: not "how fast can a hand-written
GEMM in this style go", but "what does Apple's own vendor-tuned
implementation achieve, as a ceiling to compare everything else against".

### No precision trade-off, and no algorithm-staging knob

Unlike Intel AMX (bf16-in/fp32-accumulate only) and `gemm_cuda_wmma`
(fp16-in/fp32-accumulate), Accelerate's BLAS computes at full fp32/fp64
precision throughout, so `gemm_amx_*` supports both `float` and `double`
with no reduced-precision caveat. It also exposes no algorithm-staging
knob: there is no tile size, blocking factor, or packing strategy for a
caller to select. Consequently `gemm_amx_naive`, `gemm_amx_reordered`, and
`gemm_amx_blocked` are **intentionally identical** — all three call the
same `cblas_sgemm`/`cblas_dgemm` wrapper. They exist as three separate,
identically-named entry points purely so this family's benchmarks and
tests slot into the same naming convention as every other family in this
repo, not because there are three different implementations here. The
measured benchmark numbers confirm this: all three report GFLOP/s within
~1% of each other at every matrix size (see README.md).

### Threading

Accelerate's BLAS may use multiple CPU cores internally for large
matrices (an undocumented, size-dependent heuristic) — unlike every other
CPU kernel in this repo, which is strictly single-threaded by design. This
is almost certainly why measured throughput jumps from ~820 GFLOP/s at
N=64 to ~3.3 TFLOP/s at N≥1024 (see README.md): more cores coming online
as the problem grows large enough to amortise their coordination
overhead, not (only) improving cache behaviour. Treat these numbers as
"the fastest way to multiply matrices on this machine" rather than an
apples-to-apples comparison against the single-threaded `gemm_sme`,
`gemm_avx512_*`, or `gemm_neon_*` results elsewhere in this document.

### Hardware / platform availability

Accelerate.framework: macOS and iOS only. On Apple Silicon (M1 and later)
it is understood to dispatch to the AMX coprocessor; on Intel Macs it
dispatches to AVX/AVX-512 instead — still a fast, correct BLAS, just not
exercising the AMX coprocessor this file is about. Not on Linux or
Windows — there `gemm_amx_*` is declared `= delete` (`HPC_HAS_AMX == 0`).

---

## Benchmark Results

See [docs/benchmarks.md](../../docs/benchmarks.md) for full benchmark tables, speedup analysis, and key observations.
