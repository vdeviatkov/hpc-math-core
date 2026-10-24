# GEMM kernels

Every kernel computes C = A × B, with A M×K, B K×N and C M×N, all row-major
`hpc::Matrix<float>` or `hpc::Matrix<double>`; C is overwritten. Each file's
header comment explains its design; [the detailed file guide](../../docs/kernel-design.md)
traces data flow, edge handling and tradeoffs; this page maps the files and the
techniques. Measurements: [docs/benchmarks.md](../../docs/benchmarks.md).
Cache background: [docs/cache-behavior.md](../../docs/cache-behavior.md).

## Files

| File | Kernels | Compiled when |
|---|---|---|
| [`naive.hpp`](naive.hpp) | `gemm_naive` (i-j-k, the reference every test compares against) | always |
| [`reordered.hpp`](reordered.hpp) | `gemm_reordered` (i-k-j) | always |
| [`blocked.hpp`](blocked.hpp) | `gemm_blocked` (cache-tiled i-k-j) | always |
| [`avx2.hpp`](avx2.hpp) | `gemm_avx2_{naive,reordered,blocked}` | `__AVX2__` + FMA |
| [`avx512.hpp`](avx512.hpp) | `gemm_avx512_{naive,reordered,blocked}` | `__AVX512F__` |
| [`neon.hpp`](neon.hpp) | `gemm_neon_{naive,reordered,blocked}` | AArch64 NEON |
| [`sve.hpp`](sve.hpp) | `gemm_sve_{naive,reordered,blocked}`; blocked register/cache tiles are template parameters ([sweeps](../../docs/kernels/sve.md)) | `__ARM_FEATURE_SVE` |
| [`blocked_driver.hpp`](blocked_driver.hpp) | `detail::blocked_gemm<Micro, R, V, …>`: the cache-blocked, register-tiled driver of the AVX2, AVX-512 and NEON blocked kernels and their prefetch variants; any R × V tile, edges by smaller tiles of the same micro-kernel | always (the micro-kernels need their ISA) |
| [`prefetch.hpp`](prefetch.hpp) | `gemm_{,avx2_,avx512_,neon_,sve_}blocked_prefetch` | per family |
| [`sme.hpp`](sme.hpp) | `gemm_sme_1tile`, `gemm_sme_tiles`, `gemm_sme_blocked`, `gemm_sme` | SME2, `HPC_ENABLE_SME=ON` |
| [`amx.hpp`](amx.hpp) | `gemm_amx` (Accelerate BLAS, Apple's route to its AMX unit) | Apple, `HPC_ENABLE_AMX` (default on) |
| [`kleidiai.hpp`](kleidiai.hpp) | `gemm_kleidiai` (Arm's SME2 micro-kernel, f32 only) | SME2, `HPC_ENABLE_KLEIDIAI` |
| [`cuda.hpp`](cuda.hpp) | `gemm_cuda_*`, Levels 0–8, and the cuBLAS reference; kernels in [`src/cuda/gemm_kernels.cu`](../cuda/gemm_kernels.cu) (Level 8: [`src/cuda/gemm_lowp.cu`](../cuda/gemm_lowp.cu)) | CUDA toolkit found (else a stub that reports no device) |
| [`avx512_lowp.hpp`](avx512_lowp.hpp) | `gemm_avx512_bf16` (`VDPBF16PS`), `gemm_avx512_i8` (`VPDPBUSD`) | `__AVX512BF16__` + `__AVX512VNNI__` |
| [`neon_lowp.hpp`](neon_lowp.hpp) | `gemm_neon_bf16` (`BFMMLA`), `gemm_neon_i8` (`SMMLA`), `gemm_neon_f16` (`FMLAL`), `gemm_neon_f16_acc` (`FMLA .8h`) | NEON with BF16, I8MM, FP16, FHM |
| [`sme_lowp.hpp`](sme_lowp.hpp) | `gemm_sme_widening<fp16 \| bf16 \| int8_t>` (widening `FMOPA` / `SMOPA`) | SME2, `HPC_ENABLE_SME=ON` |
| [`cuda_lowp.hpp`](cuda_lowp.hpp) | `gemm_cuda_lowp_device` (cp.async), `gemm_cuda_lowp_tma_device` (Level 8), `gemm_cuda_{bf16,fp16,fp16_acc16,i8,fp8}`, cuBLAS / cuBLASLt reference; kernels in [`src/cuda/gemm_lowp.cu`](../cuda/gemm_lowp.cu) | CUDA, sm_80+ (FP8: sm_89+; full-rate FP8 with `CMAKE_CUDA_ARCHITECTURES=120a`) |
| [`lowp_reference.hpp`](lowp_reference.hpp) | `gemm_lowp_reference`, the low-precision tests' reference (double / int64 sums) | always |

Where an ISA is absent, its kernels are declared `= delete`, so calling one is
a compile-time error rather than a slower kernel under the same name
([`include/hpc/isa.hpp`](../../include/hpc/isa.hpp); what runs on which
machine: [docs/build.md](../../docs/build.md#isa-availability--skipping)).

## Techniques

The ladders isolate techniques; CUDA levels 3 and 4 are separate extensions
of level 2, and an added technique need not improve every shape.

| Technique | What it fixes | Where |
|---|---|---|
| Loop order i-k-j | In i-j-k, B is read down a column: one element per cache line. i-k-j reads B and C rows with stride 1. | `reordered.hpp`, `*_reordered` |
| Cache blocking | For large N a C row falls out of cache between k iterations; tiles bound the working set. | `blocked.hpp`, `*_blocked` |
| Register tiling | A small C tile stays in registers for a whole k-tile instead of being reloaded every k. | the micro-kernels of `*_blocked` |
| Software prefetch | Hints for streams the hardware prefetcher might miss; close to neutral on both measured CPUs. | `prefetch.hpp` |
| Outer products into a 2-D accumulator | One SME `FMOPA` adds an SVL×SVL outer product into ZA instead of one vector FMA. | `sme.hpp` |
| Packing (GotoBLAS) | Copies blocks of A and B into contiguous strips in the order the micro-kernel reads them. | `sme.hpp` |
| GPU shared-memory and register tiling, double buffering, Tensor Cores | The GPU counterparts, one per CUDA level. | `cuda.hpp`, `gemm_kernels.cu` |
| TMA, swizzled shared memory, warp specialisation | A producer warp issues one TMA copy per tile into hardware-swizzled shared memory; consumer warps only run `ldmatrix` + `mma.sync`, synchronised by mbarriers. | `gemm_lowp.cu`, `kernel_tma` |
| k-group packing for widening instructions | bf16 / fp16 / INT8 instructions add a dot product of G = 2 or 4 consecutive k into each 32-bit lane, so B is packed with G rows interleaved and K padded to a multiple of G. | `*_lowp.hpp` |

## Families

The AVX2, AVX-512 and NEON register tiles and cache blocks are template parameters (`gemm_<isa>_blocked<T, R, V, TileM, TileK, TileN>`); every shape and block measured, and why each default: [docs/kernels](../../docs/kernels/README.md).

| Family | Register / tile | Key intrinsics or API |
|---|---|---|
| Scalar | — (auto-vectorised by the compiler at `-O3 -ffast-math`) | — |
| AVX2 | 16 × 256-bit YMM: 8 f32 / 4 f64; C tile 6×16 f32, 6×8 f64 by default (`avx2_micro<6, 2>`) | `_mm256_fmadd_ps/pd`, `_mm256_broadcast_ss/sd` |
| AVX-512 | 32 × 512-bit ZMM: 16 f32 / 8 f64; C tile 8×48 f32, 8×24 f64 by default (`avx512_micro<8, 3>`) | `_mm512_fmadd_ps/pd`, `_mm512_set1_ps/pd` |
| NEON | 32 × 128-bit Q: 4 f32 / 2 f64; C tile 6×16 f32, 6×8 f64 by default (`neon_micro<6, 4>`) | `vfmaq_lane_f32/f64` (A by element, loaded with `LDR`) |
| SVE | vector length read at run time (128–2048 bit); C tile R rows × V vectors, default 4×2 | `svmla_f32/f64_x`, `svwhilelt_b32/b64` (predicated tails) |
| SME2 | ZA: 4 f32 / 8 f64 tiles of SVL×SVL; all used; at M4's 512-bit SVL, 32×32 f32 / 16×32 f64 C block | `svmopa_za32/za64_m`, `svld1_x2`, `svld1/st1_hor_za32/64` |
| Accelerate | opaque | `cblas_sgemm`, `cblas_dgemm` |
| KleidiAI | 2VL×2VL (all four f32 ZA tiles) | `kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa` |
| CUDA | 1 output per thread (Level 0) → 8×8 per thread (Level 2) → 8 WMMA fragments per warp (Level 7) → 64×64 per warp, 32 `mma.sync` per k-step (Level 8) | `__shared__`, `cp.async`, `wmma::mma_sync`, `ldmatrix` + `mma.sync` PTX, `cp.async.bulk.tensor` + `mbarrier` (Level 8) |

## Precision

Every CPU kernel computes in its element type. The Tensor Core kernels
(`gemm_cuda_wmma`, `gemm_cuda_mma_ldmatrix`, `gemm_cuda_wmma_pipelined`) and
`gemm_cuda_cublas_fp16` take fp32 matrices but multiply in fp16 and
accumulate in fp32 (roughly 1e-3 input rounding scale, not an output error bound); `gemm_cuda_cublas_tf32` multiplies
in TF32.
