# GEMM Approaches — Quick Reference

A one-page cheat sheet of every optimisation technique and instruction
family used in this repo. For the full derivation (loop diagrams, register
tiling math, per-kernel memory-access tables) see
[src/gemm/README.md](../src/gemm/README.md). For the cache theory behind
*why* loop order matters, see [cache-behavior.md](cache-behavior.md). For
measured GFLOP/s, see the top-level [README.md](../README.md).

---

## Measured peak GFLOP/s, best kernel per family

Single-threaded unless noted. f32/f64 = 32-/64-bit float. Sources: this
README's benchmark sections, machine noted per row.

| Family | Machine | f32 peak | f64 peak | Notes |
|---|---|---|---|---|
| Scalar (`gemm_blocked`) | Apple M4 Max | 85.9 G/s | 27.2 G/s | N=64; degrades at large N (no register tiling) |
| AVX2 (`gemm_avx2_blocked`) | Intel x86 (MSVC) | 145.8 G/s | 67.1 G/s | |
| AVX-512 (`gemm_avx512_blocked`) | Intel x86 (MSVC) | 290.8 G/s | 139.2 G/s | |
| NEON (`gemm_neon_blocked`) | Apple M4 Max | 97.0 G/s | 36.3 G/s | flat across N — best-behaved CPU SIMD kernel here |
| SVE | Not available | — | — | Apple Silicon has no non-streaming SVE unit |
| **SME2** (`gemm_sme_reordered`/`_blocked`) | Apple M4 Max | **386 G/s** | **116 G/s** | outer-product engine, not FMA — see below |
| **Apple AMX** (via Accelerate) | Apple M4 Max | **3,296 G/s** | **860 G/s** | multi-threaded vendor BLAS — not a single-core comparison |
| CUDA (`gemm_cuda_reg_tile`) | NVIDIA RTX-class GPU | 6,255 G/s | — | register-tiled shared-memory kernel, single GPU |
| CUDA (`gemm_cuda_wmma`) | NVIDIA RTX-class GPU | not benchmarked in this repo | — | Tensor Cores, fp16-in/fp32-accumulate; published Tensor Core peaks for this GPU class are ~250–300 TFLOP/s, but that is not a measurement from this codebase |

---

## The three cache/access techniques, in order

Every kernel family in this repo — scalar or SIMD — is built from the same
three techniques, layered on top of each other:

| # | Technique | What it fixes | Where |
|---|---|---|---|
| 1 | **Loop reordering** (i-j-k → i-k-j) | Column-stride access to `B` (1 element per cache line used) becomes row access (all 8/16 elements per line used) | `reordered.hpp`, and the `*_reordered` variant of every SIMD family |
| 2 | **Cache blocking / tiling** | Working set of `B`+`C` no longer fits L1/L2 at large N, causing repeated cache misses | `blocked.hpp` (`TILE=64`), and the `*_blocked` variant of every family |
| 3 | **Register tiling** | Even a blocked kernel reloads `C` from L1 every k-iteration; holding a small `C` tile in registers across the whole k-loop eliminates that traffic | The register-tile micro-kernels inside every `*_blocked` SIMD kernel |

A fourth, orthogonal technique — **software prefetch** (`prefetch.hpp`,
`__builtin_prefetch`) — issues an explicit load hint some distance `D`
ahead of where the loop currently is, to hide memory latency the hardware
prefetcher doesn't predict on its own. It wraps the `*_blocked` kernel of
every family; see the "Prefetch distance sweep" results in the top README
for when it does (and doesn't) help.

---

## Instruction-family summary

| Family | File | Register / tile width | Key intrinsics or API | Cache technique(s) used |
|---|---|---|---|---|
| **Scalar** | `naive.hpp`, `reordered.hpp`, `blocked.hpp` | 1 element | — (plain C++, relies on `-O3` auto-vectorisation) | reorder, block |
| **AVX2** | `avx2.hpp` | 256-bit YMM: 8×f32 / 4×f64 | `_mm256_fmadd_ps/pd`, `_mm256_broadcast_ss/sd`, `_mm256_load(u)_ps/pd` | reorder, block, register-tile |
| **AVX-512** | `avx512.hpp` | 512-bit ZMM: 16×f32 / 8×f64 | `_mm512_fmadd_ps/pd`, `_mm512_set1_ps/pd`, `_mm512_reduce_add_ps/pd` | reorder, block, register-tile |
| **NEON** | `neon.hpp` | 128-bit Q: 4×f32 / 2×f64 | `vfmaq_f32/f64`, `vfmaq_laneq_f32` (fused broadcast+FMA), `vld1q_f32/f64`, `vaddvq_f32` (horizontal reduce) | reorder, block, register-tile |
| **SVE / SVE2** | `sve.hpp` | Scalable (VLA), 128–2048-bit, width read at runtime | `svld1_f32/f64`, `svmla_f32/f64_x`, `svdup_n_f32/f64`, `svwhilelt_b32/b64` (predicated tail — no scalar remainder loop), `svcntw()/svcntd()` | reorder, block, register-tile, predication |
| **SME2** (ARM) | `sme.hpp` | ZA tile: SVL×SVL 2-D accumulator (16×16 f32 / 8×8 f64 on Apple M4) | `svmopa_za32/za64_f32/f64_m` (outer-product-accumulate, **not** FMA), `svzero_za`, `svld1_hor_za32/64`, `svst1_hor_za32/64`, `svcntsw()/svcntsd()` | panel-packing (repacked "reorder"), K-tile "block" — see note below |
| **Apple AMX** (via Accelerate) | `amx.hpp` | Opaque — vendor-controlled | `cblas_sgemm`, `cblas_dgemm` (standard BLAS call, `<Accelerate/Accelerate.h>`) | none exposed — Apple's implementation, not ours (see note below) |
| **CUDA (GPU)** | `cuda.hpp` + `src/cuda/gemm_kernels.cu` | Thread → 1 elem (naive) up to 8×8 register tile/thread (`reg_tile`) | `__shared__` tile buffers, `__syncthreads()`, `__pipeline_memcpy_async`/`cp.async` (double-buffer), `wmma::fragment`/`wmma::mma_sync` (Tensor Cores), `float4`/`double2` vectorized loads + XOR smem swizzle, `ldmatrix.sync`+`mma.sync` PTX (raw Tensor Cores), `wgmma.mma_async`+TMA (Hopper, best-effort) | shared-memory tiling, register tiling, double buffering, vectorized loads, Tensor Core tile-multiply (3 abstraction levels), warp specialization |

---

## Implementation notes worth remembering

**AVX2/AVX-512 broadcast+FMA pattern.** The `*_reordered` kernels broadcast
one scalar `A(i,k)` to a full vector (`_mm256_broadcast_ss`) and FMA it
against a contiguous `B` row (`_mm256_fmadd_ps`). The `*_blocked` kernels
additionally keep a small grid of `C` accumulators (e.g. 4 rows × 2
vectors) resident in registers for the entire k-tile — that's the
"register tiling" layer.

**NEON's `vfmaq_laneq_f32`.** Instead of a separate broadcast instruction
per scalar, this fuses "take lane N of this vector, broadcast it, and FMA"
into one instruction — saves a `vdup` per row in the blocked micro-kernel.

**SVE has no fixed width.** `svcntw()`/`svcntd()` query the hardware vector
length *at runtime*; the same compiled binary adapts its tile width to
128-bit, 256-bit (Graviton3), or 512-bit (A64FX) hardware. The predicated
tail (`svwhilelt_b32(j, N)` — active only where `j+lane < N`) means SVE
kernels never need a separate scalar remainder loop, unlike AVX2/NEON.

**SME is not "wider SVE" — it's a different primitive.** Every family above
computes `C(i,j)` with vector **FMA**: one scalar broadcast, one vector
multiply-add. SME's `svmopa_za32_f32_m` instead computes a whole SVL×SVL
**outer product** in one instruction (`ZA[r][c] += a[r]*b[c]`), the same
class of operation as CUDA's `wmma::mma_sync` or Apple AMX below. A real
hardware quirk fell out of building this: gather-load intrinsics are
illegal inside SME's required "streaming mode," so the `A` column vector
must be assembled with a scalar loop and packed once per row-tile (see
`gemm_sme_reordered`) rather than gathered directly — full writeup in
`sme.hpp`'s file header.

**Apple AMX has no public intrinsics at all.** Unlike every other family
here, there's no ACLE-style header to include — Apple's AMX coprocessor is
reached only by calling into `Accelerate.framework`'s BLAS
(`cblas_sgemm`/`cblas_dgemm`), Apple's own vendor-tuned implementation.
Because that's a single opaque call with no tiling/blocking parameter,
`gemm_amx_naive`/`_reordered`/`_blocked` are intentionally identical
wrappers — see `amx.hpp`'s file header for the full reasoning.

**CUDA's tile progression is the clearest illustration of all three
techniques stacked, and then some.** `gemm_cuda_naive` (no shared memory) →
`gemm_cuda_blocked` (shared-memory tiling, the GPU analogue of L2
blocking) → `gemm_cuda_reg_tile` (each thread owns an 8×8 register tile of
`C`, not just one element) → `gemm_cuda_double_buf` (prefetches the next
k-tile into a second shared-memory buffer while computing on the current
one, hiding load latency) → `gemm_cuda_wmma` (Tensor Core tile-multiply,
the GPU's own outer-product-style hardware, analogous to SME/AMX on CPU)
→ `gemm_cuda_vectorized` (128-bit `float4`/`double2` loads + a
self-consistent XOR shared-memory swizzle instead of padding) →
`gemm_cuda_mma_ldmatrix` (the *same* Tensor Core computation as WMMA, one
level lower: hand-issued `ldmatrix.sync` + `mma.sync` PTX instead of the
C++ `wmma::` API) → `gemm_cuda_hopper_wgmma` (producer/consumer warp
specialization: one warpgroup issues TMA bulk-tensor loads while another
runs `wgmma.mma_async` directly against shared memory). **Verification
drops sharply toward the end of this list**: Levels 0-4 have historical
benchmark data from a separate machine; Levels 5-7 were written with zero
CUDA toolkit or GPU access, and the last one (Hopper wgmma+TMA) is
explicitly a best-effort sketch the user asked for with that understanding
— see `src/gemm/README.md`'s CUDA section for the full, per-level caveat.

---

## Precision notes

All CPU families above compute at native `float`/`double` precision except
where noted:

- **`gemm_cuda_wmma`**: converts inputs to fp16, accumulates in fp32 (~1e-3
  relative error) — Tensor Cores require reduced-precision input.
- **Apple AMX (`amx.hpp`)**: full fp32/fp64 throughout — Accelerate's BLAS
  does *not* force a reduced-precision format, unlike the GPU/Intel-AMX
  designs above.
