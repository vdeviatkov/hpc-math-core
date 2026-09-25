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
| `sme.hpp` | `gemm_sme_naive` · `gemm_sme_reordered` · `gemm_sme_blocked` — **verified, Apple M4 Max** | `__ARM_FEATURE_SME` (+ `-DHPC_ENABLE_SME=ON`) |
| `amx.hpp` | `gemm_amx_naive` · `gemm_amx_reordered` · `gemm_amx_blocked` — **verified, Apple M4 Max, via Accelerate.framework** | `HPC_HAS_AMX` (Apple + Accelerate.framework; on by default) |
| `prefetch.hpp` | `gemm_blocked_prefetch` · `gemm_avx2_blocked_prefetch` · `gemm_avx512_blocked_prefetch` · `gemm_neon_blocked_prefetch` · `gemm_sve_blocked_prefetch` | per ISA |
| `cuda.hpp` | `gemm_cuda_naive` (L0) · `gemm_cuda_reordered` (L0b) · `gemm_cuda_blocked` (L1) · `gemm_cuda_reg_tile` (L2) · `gemm_cuda_double_buf` (L3) · `gemm_cuda_wmma` (L4, fp32) · `gemm_cuda_vectorized` (L5) · `gemm_cuda_mma_ldmatrix` (L6, fp32) · `gemm_cuda_wmma_pipelined` (L7, fp32) · `gemm_cuda_cublas{,_tf32,_fp16}` (reference, not part of the ladder) — **all verified, RTX 5080 (Blackwell sm_120)** | `HPC_HAVE_CUDA` |

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

Nine GPU kernels across eight optimization levels (0 through 7), compiled
by nvcc. On CPU-only machines a stub is compiled; all CUDA benchmarks/tests
print `SKIPPED: 'No CUDA device available'` at runtime.

**Verification status (updated 2026-08-29).** Every level is verified on
real hardware — an NVIDIA RTX 5080 (Blackwell, sm_120, CUDA 13.2) — the
first CUDA-capable machine this project has ever had access to. All 68
cases in `hpc_tests_cuda` pass. That first real run found and fixed four
genuine, previously-unexercised bugs:
  1. `CMakeLists.txt`'s MSVC `/O2 /fp:fast /Oy-` flags leaked into nvcc's
     own command line (missing a `COMPILE_LANGUAGE:CXX` guard) and broke
     its argument parser.
  2. `kernel_double_buf`'s `cp.async` calls copied from a local register
     instead of the real global address (`cudaErrorNotSupported`,
     poisoning the CUDA context for the rest of the process), and its
     launch hardcoded 256 threads/block regardless of element type —
     correct only by coincidence for `float`; for `double` it silently
     computed with 4× too many threads and corrupted neighboring blocks'
     output at N > 64.
  3. `kernel_wmma`'s shared-memory padding broke `load_matrix_sync`'s
     8-element alignment requirement, and its A/B fragment major-order
     tags were swapped relative to the physical layout (silently
     transposed operands).
  4. `kernel_mma_ldmatrix`'s A-fragment `ldmatrix.x4` quadrant-to-register
     mapping had its row/col bits swapped (silently wrong output, no
     crash).

See each kernel's section below and its file comment in `gemm_kernels.cu`
(search "found running on real hardware") for the full per-bug writeup,
and
[§ CUDA kernels](../../docs/benchmarks.md#nvidia-rtx-5080--cuda) for
measured throughput.

That verification pass also added a cuBLAS reference (see "Reference —
cuBLAS" further down) to measure this GPU's realistic achievable peak,
which led to a follow-up: **Level 7 (`gemm_cuda_wmma_pipelined`)** (added,
not a replacement — Level 4 is untouched) that closes most of the ~24x gap
between Level 4/6's ~5 TFLOP/s and cuBLAS's ~118 TFLOP/s using bigger tiles
and cp.async pipelining, while staying on the documented `wmma::` API.
Verified: ~100 TFLOP/s (compute-only, N=16384), ~19x Level 4's throughput,
83% of cuBLAS's. See Level 7's section below for the full design writeup.

**Correctness fix (Levels 2-3, found before real hardware was available).**
`kernel_reg_tile`'s and `kernel_double_buf`'s shared-memory load previously
computed `row = threadIdx.x / kBM` directly as an index into the 16-row
tile — with 256 threads and kBM=128, that expression can only ever produce
0 or 1, silently leaving 14 of the tile's 16 rows uninitialized before the
k-loop read them. Found via arithmetic inspection while extending this
file, fixed with a strided load loop matching the pattern `kernel_wmma`
already used correctly, and since confirmed correct by the real-hardware
test run above.

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
**Falls back** to `gemm_cuda_double_buf` on pre-Volta hardware at runtime.

**Verified on real hardware (RTX 5080, Blackwell sm_120).** The first real
run found two bugs here, both now fixed: (1) the `As`/`Bs` shared-memory
arrays used the usual "+1" bank-conflict padding trick, which broke
`wmma::load_matrix_sync`'s requirement that the leading dimension be a
multiple of 8 `__half` elements (`cudaErrorMisalignedAddress` on every
call) — fixed by dropping the padding, since `kBlockM`/`kBlockN` (64) are
already multiples of 8; (2) `a_frag`/`b_frag`'s `row_major`/`col_major`
tags (shown correctly above) were originally swapped relative to how `As`
(stored transposed, `As[k][m]`) and `Bs` (stored natural, `Bs[k][n]`) are
actually laid out, silently transposing both operands — the kernel
compiled and ran without error but computed numerically wrong output until
fixed. See `kernel_wmma`'s file comment in `gemm_kernels.cu` for the full
derivation of which major-order tag matches which physical layout.

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
**Verified on real hardware** (RTX 5080, Blackwell sm_120) — `Vectorized`
passes its full GTest correctness suite for both `float` and `double`,
including the non-multiple-of-vector-width fallback path; the
bank-conflict-avoidance *performance* claim specifically (fewer conflicts
than padding) has not been checked with a profiler against the padding
alternative. **Requires K and N to be multiples of the vector width** (4
for float, 2 for double) for the vectorized loads to stay 16-byte aligned;
the host dispatch falls back to `gemm_cuda_reg_tile` otherwise, which is
always correct for any shape.

---

### Level 6 — `gemm_cuda_mma_ldmatrix` — raw Tensor Cores via mma.sync + ldmatrix (fp32 only, sm_80+)

**Verified on real hardware (RTX 5080, Blackwell sm_120)** — all three
GTest cases (N=64/128/256) pass against the reference GEMM; see
`gemm_kernels.cu`'s file comment for `kernel_mma_ldmatrix` for the full
writeup. That first real run found the A-fragment's `ldmatrix.x4`
quadrant-to-register address mapping had its row/col bits swapped
(`aM`/`aK` below shown already corrected) — it compiled and ran without
error but computed numerically wrong output (not a crash) until fixed and
cross-checked against a reference implementation. Computes the same thing
as Level 4 (fp16 in, fp32 accumulate) one level below the WMMA C++ API:

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

---

### Level 7 — `gemm_cuda_wmma_pipelined` — pipelined Tensor Cores via WMMA (fp32 only, sm_70+)

**NEW kernel** (added, `gemm_cuda_wmma` untouched), **verified on real
hardware (RTX 5080, Blackwell sm_120)**. Motivated by a direct comparison
against cuBLAS: Levels 4/6 (`gemm_cuda_wmma`/`gemm_cuda_mma_ldmatrix`)
measured only ~5 TFLOP/s each on this GPU, while cuBLAS's own dense-FP16
Tensor Core path measured ~118 TFLOP/s compute-only on the same hardware
(see
[§ Reference cuBLAS](../../docs/benchmarks.md#reference-cublas--the-achievable-ceiling)).
That ~24x gap is almost entirely pipelining and tile size, not precision
or instruction choice — every kernel above already uses fp16 Tensor
Cores. This kernel closes most of that gap the way CUTLASS-style kernels
do, while staying on the documented `wmma::` C++ API rather than hand-
mapped `mma.sync`/`ldmatrix` PTX registers (Level 6's own bug — a swapped
`ldmatrix.x4` quadrant mapping that silently computed wrong VALUES, no
crash — is exactly the failure mode that approach risks).

Three changes relative to Level 4:

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
compute — the exact same double-buffer control flow already proven
correct in `gemm_cuda_double_buf`'s cp.async fix above (prefetch tile 0,
then each loop iteration issues the next tile's load before computing on
the current one, and waits for it after), applied here to fp16 Tensor
Core input instead of scalar FMA input. On pre-Ampere Tensor-Core
hardware (sm_70-75), falls back at compile time to a synchronous
vectorized copy (`float4`-sized, still double-buffered structurally,
just without the async overlap) via the same `#ifdef HPC_HAVE_CP_ASYNC`
pattern `gemm_cuda_double_buf` uses.

**Why As is untransposed here (unlike Level 4), and why that's not a
bug.** cp.async can only copy a *contiguous* run of bytes to a
*contiguous* destination — it cannot transpose during the copy the way
Level 4's per-element scalar load can. A16 (the pre-converted fp16 copy
of A) is naturally row-major (M-major, K-contiguous), so for a
contiguous cp.async copy, `As` must ALSO be stored M-major/K-contiguous
(`As[m][k]`) — the opposite of Level 4's `As[k][m]` (chosen there for
per-thread scalar-load convenience, not for cp.async). Consequently
`a_frag` must be `wmma::row_major` here, the *opposite* tag from Level
4's `col_major` for the mathematically identical operand — this is a
direct consequence of the different physical layout, not a
reintroduction of Level 4's original layout/tag-mismatch bug (see Level
4's section above); `Bs`/`b_frag` are unchanged from Level 4 (`Bs[k][n]`,
`row_major`) since B's natural layout already matches what cp.async needs
with no transpose either way.

**No boundary/zfill logic needed, by design.** Unlike `gemm_cuda_double_
buf`, this kernel requires M, N to be **exact multiples of 128** and K an
**exact multiple of 32** (no tail handling — the same scoping choice
no tail handling); the host dispatch
falls back to the always-correct `gemm_cuda_wmma` otherwise. Every
cp.async transfer moves a full 16-byte (8 x `__half`) chunk — the largest
`__pipeline_memcpy_async` supports — and this exact-multiple requirement
is what makes every source/destination address for those chunks provably
16-byte-aligned (`cudaMalloc` buffers are >=256-byte aligned; with K/N
multiples of 32/128, every row this kernel reads a chunk from starts at
an element offset that's a multiple of 8, i.e. a byte offset that's a
multiple of 16) without needing a single per-element bounds check in the
load loop.

**Verified**: all 5 GTest cases pass on the first run, including a
non-square 384x256x160 case (M/N/K all different) and an N=192 case
(not a multiple of 128) that exercises the `gemm_cuda_wmma` fallback path.

**Measured (RTX 5080, compute-only — pre-staged device buffers, no
per-call transfer/malloc/conversion):**

| N | Level 4 (`gemm_cuda_wmma`) | Level 7 (`gemm_cuda_wmma_pipelined`) | cuBLAS dense FP16 | Level 7 vs Level 4 |
|---|---|---|---|---|
| 4096 | ~5 TFLOP/s (end-to-end; not measured compute-only) | 97.2 TFLOP/s | 108.7 TFLOP/s | ~19x |
| 8192 | — | 101.5 TFLOP/s | 117.0 TFLOP/s | ~20x |
| 16384 | — | 100.5 TFLOP/s | 120.5 TFLOP/s | ~20x |

A ~19x improvement using bigger tiles, register-blocked fragment reuse,
cp.async double buffering, and padded shared-memory leading dimensions —
all still on the documented `wmma::` API — reaching ~83% of cuBLAS's
dense-FP16 throughput.

**The padding is what took this kernel from 80.8 to 100.5 TFLOP/s**, and
it was found by profiling rather than by inspection. Unpadded, `As`
(ld = 32 halves = 64 B/row) gave only 2 distinct bank-starts across a
fragment's 16 rows, and `Bs` (ld = 128 halves = 256 B/row, exactly two
32-bank cycles) gave just 1 -- an 8-way and a 16-way conflict
respectively. Nsight Compute measured 285.9M of 336.2M shared-load
wavefronts as conflicts (85%), with warps stalled on MIO throttle 26% of
the time and the tensor pipe consequently idle a third of the time.
Padding both leading dimensions by 8 halves (`kPipeAsLd`, `kPipeBsLd`)
gives 8 distinct bank-starts each -- a 2-way conflict -- and drops the
conflict count to 519K (1.0%), within 1% of the theoretical minimum
wavefront count. See those constants' comment in `gemm_kernels.cu` for
the full derivation, including why the pad must be 8 rather than the
usual 1 (`load_matrix_sync` needs ld to be a multiple of 8 halves;
cp.async needs a 16-byte-aligned destination) and why an XOR swizzle --
the zero-memory-cost alternative `kernel_vectorized` uses -- cannot be
applied to a `wmma::` kernel at all.

What limits it now is register pressure: 126 registers/thread caps
occupancy at 33% (`Block Limit Registers: 2`). Beyond that, closing the
last ~17% to cuBLAS would require deeper multi-stage pipelining (3-4
stages, not 2) and split-K for very large K -- the territory CUTLASS's
template library exists to handle generically.

---

### Performance ladder (measured on NVIDIA RTX 5080, Blackwell sm_120, f32, N=4096)

Real measured numbers from `bench_gemm_cuda` (2026-08-29) — see the
measured throughput in
[§ CUDA speedup summary](../../docs/benchmarks.md#speedup-vs-cudanaive-n4096)
for the full benchmark table this is drawn from. All CUDA benchmarks
include host↔device transfer time.

| Kernel | Level | Bottleneck | GFLOP/s | TFLOP/s |
|---|---|---|---|---|
| `gemm_cuda_naive` | 0 | HBM bandwidth | 2,481 | 2.48 |
| `gemm_cuda_reordered` | 0b | HBM bandwidth | 2,481 | 2.48 |
| `gemm_cuda_blocked` TILE=16 | 1 | `__syncthreads` + low AI | 2,304 | 2.30 |
| `gemm_cuda_reg_tile` 128x128 | 2 | Compute-bound | 5,728 | 5.73 |
| `gemm_cuda_double_buf` | 3 | Latency hidden (cp.async) | **5,744** | **5.74** |
| `gemm_cuda_wmma` (fp16 TC) | 4 | Tensor Core bound | 4,798 | 4.80 |
| `gemm_cuda_vectorized` | 5 | Compute-bound, fewer load instrs | 4,887 | 4.89 |
| `gemm_cuda_mma_ldmatrix` (fp16 TC) | 6 | Tensor Core bound | 5,082 | 5.08 |
| `gemm_cuda_wmma_pipelined` (fp16 TC) | 7 | Tensor Core bound | **7,121** | **7.12** |

> **Summary (all levels, measured on real hardware):**
> Level 1 (shared-memory tiling alone, no register tiling) is barely
> worth it over naive at this problem size. Level 2 (register tiling) is
> the biggest single jump — 2.5× over Level 1. Level 3 (double buffering
> + cp.async) edges out Level 2 slightly. The original Tensor Core
> kernels (4 and 6) and the vectorized-load kernel (5) all land *below*
> Levels 2-3's plain-FMA throughput here at N=4096 — a real, measured
> result of Levels 4/6 being small (64×64-tile), untuned kernels rather
> than a tuned production Tensor Core pipeline. **Level 7 already
> overtakes every kernel above it at this size** (7.12 TFLOP/s vs Level
> 3's 5.74) and the gap widens sharply at larger N: at N=16384,
> compute-only (excluding transfer), Level 7 reaches ~100 TFLOP/s — ~19x
> Level 4's throughput and 83% of cuBLAS's own dense-FP16 ceiling (~120
> TFLOP/s) — by applying exactly the multi-stage-pipelining and
> bigger-tile fixes that production libraries like cuBLAS/CUTLASS use.
> See Level 7's section above and
> [§ Reference cuBLAS](../../docs/benchmarks.md#reference-cublas--the-achievable-ceiling)
> for the full compute-only comparison and what closing the remaining gap
> to cuBLAS would still require (deeper pipelining, WMMA-specific
> shared-memory swizzling, split-K).

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
`svld1` — which reframes this repo's usual naive-vs-reordered lesson one
level up (see the three kernels below).

### Hardware finding: `-march=native` silently disables SME on Apple Silicon

`-march=armv9-a+sme2` compiles cleanly on Apple Silicon but the resulting
binary `SIGILL`s at runtime on the very first instruction inside the
streaming region. Clang emits a `CNTD` instruction *outside* streaming
mode to size the ZA-save prologue buffer; `CNTD` is an ordinary
(non-streaming) SVE instruction, and Apple Silicon implements **no
non-streaming SVE unit at all** — only Streaming SVE via SME.
`-mcpu=apple-m4` avoids this by generating a prologue that doesn't need an
outside-streaming SVE instruction. Worse: combining `-march=native` with
`-mcpu=apple-m4` — the repo's default Release flag plus the SME fix —
silently drops the SME/SVE target features altogether rather than
erroring, so `HPC_ENABLE_SME=ON` clears `HPC_MARCH` in CMakeLists.txt in
favour of the verified `-mcpu=` flag. Because these are *runtime* SIGILL
failure modes that a compile-only check cannot catch, this repo's CMake
SME detection actually **compiles and runs** a probe program at configure
time (`check_cxx_source_runs`, not `check_cxx_compiler_flag`) — see
CMakeLists.txt's `HPC_ENABLE_SME` block.

### Three kernels

**`gemm_sme_naive`** — for each SVL×SVL output tile `(i0, j0)`: zero the ZA
tile, then for each `k` re-gather `A(i0..i0+SVL, k)` via a scalar loop into
a stack buffer, load `B(k, j0..j0+SVL)` contiguously, and accumulate one
outer product. The A-column gather is redone for every `(i0, j0, k)`
triple — `O(N/SVL)` more scalar work than necessary. Measured: ~3 GFLOP/s,
flat across N — the same "SIMD width can't fix cache-hostile access"
lesson as every other `*_naive` kernel, except here the hostility is a
structural consequence of streaming mode rather than a memory-layout
choice.

**`gemm_sme_reordered`** — packs the entire `A(i0..i0+SVL, :)` row-panel
into a contiguous buffer **once** per i0-tile (a single scalar pass), then
reuses it with pure contiguous loads across every j0-tile. Removes the
`O(N/SVL)` redundant gathering. Measured: 218–380 GFLOP/s single-threaded
f32 — the highest CPU throughput anywhere in this repo. Degrades once the
packed panel (`SVL × K × sizeof(T)` bytes) exceeds L1/L2.

**`gemm_sme_blocked`** — adds K-tiling (`kSmeTileK = 256`) on top of the
panel-packing scheme: the packed A buffer is bounded to `SVL × kSmeTileK`
regardless of K, keeping it cache-resident. Partial C sums are carried
across k-tiles by reloading them directly into ZA via `svld1_hor_za`
(rather than re-deriving them from scratch) — using the hardware's ability
to load an existing accumulator state, not just zero it — at the cost of
extra C traffic. Wins over `gemm_sme_reordered` once K is large enough
that the unbounded packed panel would spill L2: measured 254 vs 210
GFLOP/s at N=2048, 172 vs 127 GFLOP/s at N=4096 (Apple M4 Max, f32).

### Hardware availability

Apple M4 / M4 Pro / M4 Max (SME2, 512-bit SVL) is, as of this writing,
essentially the only shipping SME2 hardware widely available to individual
developers. Not on Apple M1/M2/M3, AWS Graviton3/4, Fujitsu A64FX, or
x86 — there `gemm_sme_*` is declared `= delete` (`HPC_HAS_SME == 0`).

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
apples-to-apples comparison against the single-threaded `gemm_sme_*`,
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
