# Kernel and file design guide

This guide explains the purpose, data flow, edge handling and tradeoffs of
all project-owned source files. Start with [the kernel map](../src/gemm/README.md)
for function names, [build.md](build.md) for availability and
[benchmarks.md](benchmarks.md) for recorded results. A progression is a
controlled experiment, not a promise that each successive kernel is faster.

## Shared contract and storage

All Matrix-based GEMMs compute `C[M,N] = A[M,K] * B[K,N]` in row-major
storage. C is already allocated and is overwritten. Dimensions must agree;
assertions diagnose mismatches in debug builds. C must be a different matrix
from both inputs: zeroing or storing C can otherwise destroy an input before
it is consumed. A and B may be the same matrix. Rectangular shapes
and zero dimensions are supported within backend indexing and launch limits: K=0 writes zeros, M=0 or N=0 does no work.
These are educational kernels, not a general BLAS interface: no alpha/beta,
transpose flags, submatrix views, runtime CPU dispatch or CPU thread pool.

CPU arithmetic uses the element type, float or double. Reduction order and
FMA contraction differ between families. Release builds allow fast math;
NaN/Inf propagation, signed zero and strict IEEE rounding are not guaranteed.
Reduced-precision CUDA multiplication has additional range and rounding
limits; an approximate error scale is not a bound for cancellation-heavy
products. Tests compare finite, bounded random inputs with mixed tolerances.

### `include/hpc/matrix.hpp`

`Matrix<T>` owns one contiguous allocation with a 64-byte-aligned base and
no row padding. Row i begins at `data() + i*cols()`, so row alignment depends
on the row width. Construction zeros the buffer; fill writes each element.
Copies own independent storage; copy assignment uses a temporary for strong
exception safety. Moving transfers ownership and empties the source.

Allocation checks both the rows-times-columns product and the byte count
including alignment rounding before allocating. Empty matrices retain their
shape but have a null buffer. The matching allocator/deallocator pair is
`aligned_alloc/free`, or `_aligned_malloc/_aligned_free` on MSVC. T must be
trivially copyable; the kernels are intended for float and double.

### `include/hpc/isa.hpp`

Each `HPC_HAS_*` macro is always 0 or 1, with a constexpr mirror for template
registration. Detection describes this translation unit's compiler target,
not the CPU currently executing it. AVX2 kernels also require FMA. Missing
families have deleted entry points; tests omit them and benchmarks register
skip-only rows. An ISA-enabled binary must run on compatible hardware.
Accelerate's availability macro describes a library, not proof of Apple AMX
hardware or a particular internal dispatch choice.

### `include/hpc/lowp.hpp`

Storage types for the low-precision GEMMs: `bf16`, `fp16` and `fp8_e4m3` are
bit patterns in a struct, so `Matrix<bf16>` and friends work on every
compiler; kernels reinterpret them as the hardware types. INT8 is plain
`std::int8_t`. Conversions from float round to nearest, ties to even, with
subnormals, overflow (fp16: to infinity; E4M3: saturating to ±448) and NaN.
The tests check every bit pattern round-trips (NaNs are recognised by their
bits, because `-ffast-math` may fold `std::isnan`) and the rounding of ties.

## Scalar kernels

### `src/gemm/naive.hpp`: `gemm_naive`

The i-j-k loop owns one C element at a time, accumulating a dot product in
a scalar variable. A's row is contiguous but B's column has stride N.
For sufficiently wide B, successive k steps touch different cache lines;
reuse of neighboring columns is delayed by a complete dot product. This
makes the code a readable correctness reference and a cache-hostile baseline.
It uses no scratch storage and naturally handles tails and K=0. The initial
C zeroing is redundant with the final assignments, but remains part of the
measured baseline.

### `src/gemm/reordered.hpp`: `gemm_reordered`

Swapping j and k makes the inner loop stream B and C rows. Each A(i,k) is
reused across N columns. C must first be zeroed because each k contributes
a partial sum. Contiguous accesses enable automatic vectorization and
hardware prefetch, but C is loaded/stored every k and all of B is traversed
again for each i. This is the simplest demonstration that data reuse and
access order matter more than spelling arithmetic with intrinsics.

### `src/gemm/blocked.hpp`: `gemm_blocked<T,TILE>`

The outer i-block, k-block, j-block loops restrict the inner i-k-j loops to
three tiles. A B tile serves several rows before leaving cache; the working
set is roughly `3*TILE*TILE*sizeof(T)`. The default TILE=64 is configurable
through the compiler definition `HPC_GEMM_TILE`, and must be positive.
`min` clips every boundary. C is zeroed once and accumulates across k-blocks.
Blocking has overhead and can inhibit vectorization at small N, so it need
not beat the reordered kernel on every shape.

## SIMD families

AVX2, AVX-512 and NEON share one set of kernels (`simd.hpp`), generic over
the ISA: each ISA header defines only a primitive struct (`Ymm<T>`,
`Zmm<T>`, `Q<T>`), and the kernels take it as their first template
argument. Reordered kernels
broadcast A and vectorize the contiguous j loop. Blocked kernels keep a
small rectangular C tile in registers for a whole k-block, amortizing C
traffic and reusing both A broadcasts and B loads; they read B in place.
Packed kernels use the same micro-kernel with B copied into contiguous
strips (below). Full SIMD loads are unaligned where needed, because the
Matrix base alignment does not align every row.

No kernel in `src/` holds a tuned size: register tiles and cache blocks
are template parameters without defaults, and the values the sweeps chose
live with the benchmarks (`benchmarks/simd_params.hpp`).

### `src/gemm/simd.hpp`

`gemm_simd_reordered<Ops>(A, B, C)`,
`gemm_simd_blocked<Ops, R, V, TileM, TileK, TileN, PfDist = 0>(A, B, C)` and
`gemm_simd_packed<Ops, R, V, Mc, Kc, Nc, PackA = false>(A, B, C)`, with T
deduced from the matrices. The reordered loop is written once against
`Ops::load`, `store`, `load_a` and `fma`, with a scalar loop for the
columns that do not fill a vector. The blocked kernel uses the
micro-kernel and the edge helpers below; `PfDist > 0` is the prefetch
variant. The packed kernel is the k-group kernel that `gemm_simd_lowp` also
runs (below), with the FMA as an instruction of k group G = 1. `Ops<T>` is declared on every target and defined only where its
ISA is compiled, so instantiating a kernel for an absent ISA is a
compile-time error, and the benchmarks and tests guard each family with
`HPC_HAS_*` (or `if constexpr` on `kHave*`).

#### `simd.hpp`: `gemm_simd_blocked`

The register tile, R rows × V vectors, and the cache block,
TileM × TileK × TileN, are template parameters of
`gemm_simd_blocked<Ops, R, V, TileM, TileK, TileN>`; `bench_gemm` sweeps both (`<Isa>Tile<R>x<V>`,
`<Isa>Block<M>x<K>x<N>`, [docs/kernels](kernels/README.md)) and the tests
run several, including blocks that are not multiples of the tile. Remainders use the same micro-kernel (`detail::row_strip`): a strip of
fewer than R rows runs `vector_micro<Ops<T>, rows, V>`, the last whole vectors of a block run a
narrower tile, and only columns that do not fill one vector take a scalar
loop. `PfDist > 0` adds the prefetch hints of `prefetch.hpp`. Every loop in
a micro-kernel is unrolled with `#pragma GCC unroll 16`, so R and V are
limited to 16 (a `static_assert`); a partially unrolled loop would move the
accumulators to memory.

#### `simd.hpp`: `gemm_simd_packed`

`gemm_simd_packed<Ops, R, V, Mc, Kc, Nc, PackA>` is `detail::gemm_kgroup`,
the kernel of `gemm_simd_lowp`, run with `detail::Fma<Ops, PackA>`: the
ISA's `load_a` / `load` / `fma` as an instruction of k group G = 1, one
accumulator per vector of a row of C. GotoBLAS order: for each Kc × Nc
block of B (Nc rounded down to whole strips), copy it once into strips
Nr = V·W columns wide (kc rows of Nr contiguous elements, zero-padded),
then for each Mc-row block of A, for each strip, for each R-row tile, run
the micro-kernel. With G = 1, packing B by k group is a copy of rows, so
the per-k work is the blocked kernel's outer product. A is read in place
unless `PackA`, which packs the Mc × Kc block into R-row strips (the R
values of one k step contiguous). A strip's last columns run a tile with
only as many accumulator columns as they need, and a tile cut short by
the matrix or a block goes through a buffer. B's row-major layout already suits the FP32
micro-kernel; the copy only removes the cache-set conflicts of a
power-of-two row stride. Packing costs 2–13% at N ≤ 256 on Zen 5 and
gains up to 60% at N ≥ 1024 ([experiments/packed-b](experiments/packed-b/README.md)),
so the blocked kernels stay as they are and the packed ones are a separate
entry point, `gemm_simd_packed<Ops, R, V, Mc, Kc, Nc, PackA = false>`.

#### `simd.hpp`: the micro-kernel

The micro-kernel is one template, `vector_micro<Ops<T>, R, V>`: load the
R × V-vector tile of C, then per k load V vectors of B, load A once per
row and do R·V FMAs, then store the tile. `load_a` is where the ISAs
differ (a broadcast from memory, a broadcast of a scalar, a plain `LDR`
for an FMLA by element). The blocked and packed kernels compile to the
same machine code as the three per-ISA kernels did before (Clang, M4 and Zen 5 targets).

### `src/gemm/avx2.hpp`, `avx512.hpp`, `neon.hpp`

Each defines its ISA's primitive struct for float and double and nothing
else: `load`, `store`, `load_a` (A(i, k) as the first FMA operand) and
`fma`. `Ymm<T>` (AVX2: 16 YMM registers, 8 floats or 4 doubles; FMA is
required too) and `Zmm<T>` (AVX-512: 32 ZMM registers, 16 floats or 8
doubles) broadcast A into a full vector. `Q<T>` (NEON: 32 Q registers, 4
floats or 2 doubles) loads A with a plain `LDR` into lane 0 of a D
register and multiplies by element; the `vdupq` + `vfmaq` form compiles to
`LD1R` (load and replicate), which held tiles with many rows 23-30% back on
M4. For float, Clang turns every intrinsic spelling back into `LD1R`, so
`Q<float>::load_a` is a one-instruction inline `asm` on GCC/Clang. AArch64
is required because `Q<double>` uses double-precision FMA.

The measured parameters (`benchmarks/simd_params.hpp`): AVX2 a 6 × 2 tile
(6x16 float, 12 accumulators; with two B vectors and one or two A
broadcasts that is 15-16 of the 16 YMM registers, and 7 rows spill) with
384 × 256 × 1 KiB-of-B blocks; AVX-512 an 8 × 3 tile (24 accumulators in
28 of 32 ZMM registers; with 4 or fewer rows GCC folds the B loads into
every FMA and the kernel becomes load-bound) with 512 × 256 × 2 KiB-of-B
blocks; NEON a 6 × 4 tile (24 accumulators in 29 of 32 registers) with
384 × 256 × 256 blocks. Each was chosen by sweeping nine or ten shapes
([kernels/avx2.md](kernels/avx2.md), [kernels/avx512.md](kernels/avx512.md),
[kernels/neon.md](kernels/neon.md), [kernels/blocks.md](kernels/blocks.md))
and is not a universal optimum for other CPUs or compiler versions.

### `src/gemm/prefetch.hpp`

Each variant retains its base blocked traversal and micro-kernel, adding
hints for a future A row, the start of the next B k-block, and a future C
row. PfDist counts micro-kernel row groups (individual rows for scalar).
The hints cover particular lines, not whole tiles. `pf<Rw,Loc>` maps to
`__builtin_prefetch`, or x86 `_mm_prefetch` under MSVC; the latter ignores
the write intent. Locality is a compiler/architecture-dependent hint,
not a guaranteed cache destination. Distances 2/4/8/16 are benchmarked;
the recorded runs show little benefit. The SIMD kernels share a driver with
their ordinary blocked entry points; only the scalar variant keeps a
separate traversal.

## Matrix engines and vendor references

### `src/gemm/sme.hpp`

SME FMOPA adds an outer product into a two-dimensional ZA tile. Streaming
vector length is independent of non-streaming SVE. On the measured M4,
SVL is 16 floats or 8 doubles. `pack_a` transposes row-major A into
column-contiguous strips using NEON 4x4 or 2x2 transposes plus scalar edges;
`pack_b` copies B row segments into contiguous strips. Both pad edges.

- `gemm_sme_1tile` packs all A into SVL-row strips and streams B in place,
  using only ZA tile 0. One dependency chain exposes FMOPA latency.
- `gemm_sme_tiles<T,R,V>` packs A into R*SVL-row strips and uses R×V ZA tiles:
  measured 2x2 for float, 2x4 for double. Independent accumulators hide latency,
  and each loaded vector feeds multiple outer products.
- `gemm_sme_blocked` keeps that micro-kernel and adds jc/pc/ic blocking.
  B remains in the caller's layout, exposing power-of-two stride conflicts.
- `gemm_sme` additionally packs B once per jc/pc panel. Contiguous strips
  control the stride regardless of the input N and amortize copying over
  all ic blocks.

`gemm_sme` and `gemm_sme_blocked` expose R/V and M/K/N cache sizes as
template parameters with no defaults; the measured values are in
`benchmarks/sme_params.hpp`. Every legal rectangular ZA shape has a benchmark
registration and correctness tests. Every shape loads its A and B vectors in
SME2 pairs. See [SME tuning](kernels/sme.md)
and [the sizing rationale for all CPU families](kernels/sizing.md).

The measured Mc=1024 float / 512 double, Kc=2048 and Nc=4096 aim for an 8 MiB packed A
block in M4 Max's shared 16 MiB L2. The streaming macro-kernel visits B
strips then A strips; C stays in ZA through kc, and is reloaded only when
pc>0. Predicated C transfers and row bounds handle edges; full tiles use a
separate inlined transfer path so edge address checks stay out of that path. ZA helpers declare
shared state so the compiler does not insert private-ZA saves in the hot
loop. Explicit zeroing initializes each new accumulation. Allocation and
packing happen outside streaming mode, but inside each timed GEMM call.
The L2 explanation is an inference from controlled measurements; Apple's
physical cache indexing and matrix-engine internals are undocumented.

`gemm_sme_mt<T,R,V,TileM,TileK,TileN>(A, B, C, threads)` (measured: 8 threads) runs the same jc/pc loops on a
team of threads, to use the SME unit of each P-cluster. Per pass, every
thread packs every team-th B strip into one shared panel; after a barrier,
threads claim row chunks (at most TileM, about four per thread, at least
one 2·SVL micro-tile) from an atomic counter, pack them into a private A
buffer and run `macro_kernel`; a second barrier protects the panel. Chunks
write disjoint rows of C, and pc>0 passes reload them, so there is no
reduction. Dynamic chunks, not equal bands, because macOS cannot pin a
thread to a cluster and tends to keep a few threads on one, so equal bands
finish at the pace of the busier unit. Workers raise themselves to
user-interactive QoS to stay on P-cores. Threads are created per call, which
dominates below N≈512. The default of 8 threads comes from a sweep
([Several SME units](kernels/sme.md#several-sme-units)); cluster placement
is inferred from throughput, not observed.

### `src/gemm/amx.hpp`

This is an Accelerate BLAS adapter, not an Apple AMX intrinsic kernel.
Row-major `cblas_sgemm/dgemm` uses alpha=1 and beta=0 to overwrite C.
Empty dimensions bypass BLAS because zero leading dimensions are invalid.
Accelerate chooses its internal implementation and may use multiple threads;
`VECLIB_MAXIMUM_THREADS=1` constrains the reference comparison. On Intel
Macs the same wrapper uses the available CPU implementation. Dimensions
must fit the BLAS integer interface. Apple AMX is unrelated to Intel AMX.

### `src/gemm/kleidiai.hpp`

The float-only adapter calls Arm's packed SME2 2VLx2VL micro-kernel. It
queries packing granularity and byte counts from KleidiAI, allocates aligned
scratch, packs both operands and supplies a zero bias. The clamp limits are
±FLT_MAX, effectively inactive for the finite test range; they do not promise
unclamped infinity behavior. Both packing steps are timed. The wrapper calls
the micro-kernel once over the whole matrix and supplies no cache blocking,
which explains why this particular integration loses speed at large N.

## Low-precision kernels

All compute `C = A · B` with bf16 / fp16 / FP8 inputs and FP32 output, or
INT8 inputs and INT32 output. The narrow instructions add a dot product of
G consecutive k values (G = 2 for 16-bit, 4 for 8-bit) into each 32-bit
lane, so every CPU kernel packs B with G rows interleaved and pads K to a
multiple of G. Article: [part 5](articles/lowp-gemm.md).

| File | Design |
|---|---|
| `src/gemm/lowp_naive.hpp` | `gemm_lowp_naive`: products summed in double (exactly in int64 for INT8), rounded once. The tests' reference; `lowp_out_t<In>` names the output type. |
| `src/gemm/simd.hpp`: `gemm_simd_lowp` | `gemm_simd_lowp<Ops, R, V, Mc, Kc, Nc>`: the low-precision SIMD kernel, generic over an instruction-shape struct (`ZmmDot<In>` in `avx512.hpp`; `QMmla<In>`, `QFmlal<In>`, `QFmlaF16<In>` in `neon.hpp`). It runs `detail::gemm_kgroup`, the kernel `gemm_simd_packed` runs with G = 1. GotoBLAS order; B is always packed by k group (word w of column c holds G consecutive k), A only where the instruction needs it. One micro-kernel: per k group, V B operands, one A operand per row unit, R·V instructions; rows and vectors unrolled with `#pragma GCC unroll`, so FMLAL's lane is a constant. Kept out of line: inlined, the AVX-512 bf16 k loop reloaded its row pointers (56 → 420 instructions with Clang). Tile and blocks are template parameters; the measured values are in `benchmarks/simd_params.hpp` (`LowpParams`). |
| `ZmmDot<In>` (`avx512.hpp`) | `VDPBF16PS` (bf16), `VPDPBUSD` (INT8). One accumulator is 1 × 16 of C; measured tile 8 × 2 (8 × 32 of C), swept by `bench_gemm_lowp`'s `Avx512{Dpbf16,Vnni}Tile*` rows. Only B is packed (G rows interleaved, as the instructions need); A is broadcast in place. `VPDPBUSD` is unsigned × signed, so B is packed + 128 and each row's accumulators start at −128 · Σ_k A(i, k): exact. `VPDPBUSD` is one inline `asm` instruction because GCC 13 wraps the intrinsic in register copies that cost 16–26%. Edges: masked C loads and stores. Measured blocks: Mc = 128, Kc = 1024 bytes of k, Nc = 512. |
| `QMmla<In>`, `QFmlal<In>`, `QFmlaF16<In>` (`neon.hpp`) | `BFMMLA` (bf16) and `SMMLA` (INT8): 2 × 2-tile matrix instructions, one accumulator per 2 × 2 of C, A and B both packed; measured as an 8 × 8 C block in 16 registers (R = V = 4). `FMLAL` (fp16, FP32 accumulate) and `FMLA .8h` (fp16 accumulate, converted on load and store) multiply by element: A is packed 4 rows per k and row r uses lane r, so R = 4; measured with 16 accumulators. Edges through a buffer. |
| `src/gemm/sme.hpp`: `gemm_sme<fp16 \| bf16 \| int8_t>` | `gemm_sme` and `gemm_sme_mt` with widening `FMOPA` / `SMOPA` into the four 32-bit ZA tiles: the FP32 kernels with a 32-bit word of G = 2 or 4 elements in place of an element (G = 1 for FP32 / FP64), so one `pack_a` (NEON transposes of 32-bit words), one `macro_kernel` and one driver serve all five types; `pack_b` zips G rows. Any R × V ZA shape and cache block, as for FP32 (TileK in elements, a multiple of G). Default blocks keep FP32's sizes in k groups (8 MiB A block). |
| `src/gemm/cuda.hpp`, `src/cuda/gemm.cu` | One `mma.sync` kernel template per format struct (`Bf16`, `Fp16`, `Fp16Acc16`, `Int8`, `Fp8`), in two kernels: `kernel_lowp` (cp.async, 2 stages, padded rows, 128 × 128) and `kernel_tma` (Level 8). Both take A and B K-major (B transposed), so one non-transposing `ldmatrix` serves 8- and 16-bit formats. FP8 uses the block-scaled `kind::mxf8f6f4` instruction with unit scales in an `sm_120a` build (full rate), the sm_89 form otherwise. cuBLAS / cuBLASLt references on the same buffers. Matrix wrappers pad, transpose B and select the implementation (`CudaLowpImpl`). |
| `src/cuda/gemm_stub.cpp` | The same symbols without CUDA; every format reports unsupported and kernels throw. |

## CUDA interface and implementation

### `src/gemm/cuda.hpp`

The header exposes Matrix wrappers, device-buffer launchers, capability
queries, copies, allocation and event timing without including CUDA headers.
Only float/double template instantiations are supplied. Matrix wrappers copy
inputs to the selected GPU and return a completed CPU result. Device
launchers enqueue on the default stream; the caller owns buffer lifetimes,
alignment, non-aliasing and synchronization. Dimensions and linear element
indices must fit signed int; Matrix wrappers reject larger problems. The
fixed 2-D launch grids must also fit the device limits (in particular, the
number of output row tiles cannot exceed gridDim.y's limit).

K=0 clears nonempty C, and M=0 or N=0 returns without launching. The raw
pipelined device path takes half buffers and rejects non-tile shapes rather
than silently truncating them. For nonempty work it needs Tensor Cores.
FP16 buffers are void* so CPU-only builds can compile this header. Use
16-byte-aligned inputs for vectorized loads and 32-byte-aligned C for WMMA;
cudaMalloc supplies sufficient alignment. Compile architectures appropriate
to the GPU; runtime detection cannot add instructions omitted at compile time.

### `src/cuda/gemm.cu`

One file holds every CUDA kernel (Levels 0-8 and the low-precision
formats), so their layouts and shared building blocks can be compared
directly. Device helpers come first: `load_tile` coalesces global reads,
optionally transposes stores and zero-fills edges (through
`cp_async_or_zero` on sm_80+ when source and destination types agree);
`copy_tile_16B` copies whole tiles in 16-byte `cp_async_16` chunks;
`cp_async_pipeline` runs the STAGES-deep copy/compute loop that Level 3,
Level 7 and `kernel_lowp` share. `tile_fma` computes an outer product in
registers; `store_tile` clips output edges. `kernel_lowp` and `kernel_tma`
share `warp_mma_k32` (ldmatrix + mma for one 32-byte k step, over a
`PaddedRows` or `Swizzle64` shared-memory layout), `store_acc` and
`grouped_tile`; Level 6 reuses the `Fp16` format's `mma`.

| Level / kernel | Idea and data flow | Limits and tradeoffs |
|---|---|---|
| 0 `kernel_naive` | 16x16 threads; each owns one dot product using global A/B | A warp covers two adjacent 16-column row segments, not one 32-column row. Caches absorb reuse, but each FMA needs two source loads. |
| 1 `kernel_blocked` | Cooperative 16x16 shared tiles, one C value per thread | Two barriers per 16 FMAs; synchronization can outweigh saved global loads. The padding separates the two double-precision A broadcasts across banks. |
| 2 `kernel_reg_tile` | 128x128 outputs, 256 threads, 8x8 register tile each; shared A transposed and B natural | More reuse per shared load, but register pressure limits occupancy. +1 padding addresses A's transposed stores; B reads still conflict. |
| 3 `kernel_double_buf` | Prefetch the next shared tile while accumulating the current tile | cp.async overlap requires sm_80+; older targets copy synchronously. Double uses 64x64 outputs and 64 threads to stay within static shared-memory limits. |
| 4 `kernel_vectorized` | 128-bit global/shared loads, XOR-swizzled vector slots, same register tile as level 2 | K/N must be multiples of 4 float / 2 double, otherwise level 2 runs. It builds on level 2, not the double-buffered kernel. |
| 5 `kernel_wmma` | 64x64 outputs, 16 warps; one 16x16 accumulator per warp; float converted to half in shared memory | sm_70+; else level 3. Uniform warp participation is required. Partial output tiles stage through per-warp shared memory before clipped stores. |
| 6 `kernel_mma_ldmatrix` | Same output decomposition as level 5, with explicit lane mapping and two m16n8k16 MMAs per warp | sm_80+; else level 5. A uses row-major shared layout; B's ldmatrix is transposed. When compiled below sm_80, the kernel body uses the shared WMMA implementation, including when older PTX is JIT-compiled on a newer GPU. |
| 7 `kernel_wmma_pipelined` | 128x128 output, BK=32; eight warps own 2x4 WMMA fragments each; two shared stages with padded strides | Half inputs are converted once before timing. Matrix wrapper falls back to level 5 unless M/N are multiples of 128 and K of 32. Raw device path has no shape fallback. |
| 8 `kernel_tma` | 128x256 (8 consumer warps, 4 stages) or 128x128 (4 consumer warps, 6 stages) output, chosen by wave fill; 64x64 per consumer warp; one producer warp issues TMA copies with 64-byte swizzle; full/empty mbarriers per stage | sm_90+; the Matrix wrapper runs level 7 otherwise. Takes A (MxK) and B transposed (NxK); the Matrix wrapper converts to fp16, transposes and pads. Tensor maps are encoded on the host and cached per buffer. Shared with the low-precision formats. |

The buffered kernels go through `cp_async_pipeline`: commit copies, wait
for each thread's copies, then one block barrier per tile makes the data
visible and protects buffer reuse. Padding
by eight halves keeps the WMMA leading dimension a multiple of eight while
changing shared-bank mapping. Shared array bases explicitly meet WMMA's
32-byte alignment requirement. Padded row pitches must also satisfy the
WMMA stride contract; only the fragment origin needs 32-byte alignment.

cuBLAS references reinterpret row-major C=A*B as column-major C^T=B^T*A^T,
swapping operands and M/N without a physical transpose. SGEMM/DGEMM use full
input precision; TF32 requests a faster reduced-mantissa compute mode;
FP16 supplies preconverted half buffers with float accumulation. The cuBLAS handle is
created lazily on first use and retained until process exit, matching the
single-host-thread, fixed-device benchmark contract. Do not call these
cuBLAS wrappers concurrently or switch/reset the device after first use.

DeviceBuffer releases temporary allocations during exceptions. Event objects
also clean up if a callback or synchronization fails. GPU event timing
measures the interval between stream markers; for short launches it can
include GPU idle gaps while the host submits work. It is not a direct
hardware-counter measurement of only the kernel body.

### `src/cuda/gemm_stub.cpp`

This supplies the same linkable symbols when no toolkit is found. Device
count is zero, capability queries are false, free is harmless, and compute,
copies, synchronization, allocation and timing throw. This preserves build
coverage and explicit skip behavior without pretending CPU work is CUDA.

## Metal interface and implementation

### `src/gemm/metal.hpp`

The C++ face of the Metal kernels, free of Objective-C so the tests and
benchmarks stay plain C++. `MetalGemm` owns A, B and C (plus fp16 copies of
A and B) in shared-storage buffers, which the CPU and GPU address directly
on Apple silicon, so uploads are memcpys and dispatch copies nothing.
`run_ms(kernel, reps)` encodes `reps` dispatches of one kernel into one
command buffer and returns GPU time per GEMM from `GPUStartTime` /
`GPUEndTime`; `run_variant_ms` does the same for a tile variant by name.
`effective()` reports the fallback: Levels 2–5 need M, N multiples of 64 and
K of 16, otherwise Level 1 runs. K=0 clears C, M=0 or N=0 returns.

### `src/metal/gemm.metal`

Six levels, f32 in and out. Level 0 reads every operand from device memory.
Level 1 stages 16×16 tiles in threadgroup memory and bounds-checks every
load, which makes it the any-shape fallback. Level 2 is a template
(BM, BN, BK, TM, TN; default 64×64×16 with 8×4 outputs per thread): A is
stored transposed so both operands of the inner product are contiguous.
Levels 3–5 use `simdgroup_float8x8` multiply-accumulate: Level 3 loads
fragments straight from device memory, Level 4 stages padded tiles in
threadgroup memory first, Level 5 is Level 4 on half inputs with float
accumulators. Fixed-trip loops carry `clang loop unroll(full)`: left rolled,
the accumulator arrays are indexed dynamically and spill, which measured
5–10× slower. Shared helpers keep the levels short: `load_tile` copies a
tile into threadgroup memory as 4-wide vectors (Level 2's B, Level 4's A
and B); `zero_acc`, `simd_mma` (load FM×FN fragments from device or
threadgroup memory and multiply-accumulate) and `store_acc` are the whole
simdgroup inner loop of Levels 3–5. `fma_peak` runs eight independent
4-wide FMA chains per thread to measure the arithmetic peak.

### `src/metal/gemm_metal.mm`, `src/metal/gemm_metal_stub.cpp`

The Objective-C++ host (ARC). One process-lifetime context creates the
device and queue and compiles the shader source CMake embedded
(`hpc_metal_source.inc`) with fast math; pipelines are created on first
use and cached. A table gives each tiled function's threadgroup geometry
and tile multiples, so tile variants dispatch without code changes. The MPS
reference wraps the same buffers in `MPSMatrix` descriptors, fp16 inputs
with an fp32 result for `kMpsF16`. The stub reports no device and throws
otherwise, as the CUDA stub does.

## Correctness and benchmark files

| File | Purpose and reasoning |
|---|---|
| `tests/test_gemm.cpp` | Hand-checked reference products, then a table of available CPU kernels against that reference. Boundary shapes exercise vector/register/cache tails, rectangular matrices, split K accumulation and empty dimensions. C starts as junk to distinguish overwrite from accumulation. SME additionally checks bit-preserving A packing and padded B strides. |
| `tests/test_matrix.cpp` | Checks alignment, deep copies, shape-preserving empty copies, safe moved-from objects and overflow rejection before allocation. These are storage invariants all kernels depend on. |
| `tests/test_gemm_cuda.cpp` | Per-kernel shape tables exercise CUDA bounds, vector and pipelined fallbacks, reduced precision, empty products and rectangular vendor calls. Direct device checks cover zero-K clearing and rejection of invalid launch shapes. Missing GPUs skip explicitly. |
| `tests/test_gemm_metal.cpp` | Every Metal level, MPS and every tile variant against `gemm_naive`, on tile multiples with M, K, N all different, off-tile shapes (fallback) and empty products. Each GEMM runs twice into the same C, so accumulation instead of overwrite fails. 1e-4 relative tolerance, 1e-2 for fp16 inputs. Skips without Metal. |
| `tests/test_gemm_lowp.cpp`, `tests/test_gemm_lowp_cuda.cpp` | Low-precision kernels against `gemm_lowp_naive` on the same converted inputs: INT8 exact, FP32 accumulation within 2·K·2⁻²⁴·Σ\|ab\|, fp16 accumulation with 2⁻¹¹. Exhaustive conversion round-trips. The CUDA test runs both kernels and cuBLAS, with shapes that select both Level 8 tiles. |
| `tests/CMakeLists.txt` | Links CPU tests and CUDA-or-stub tests separately. PRE_TEST discovery delays executing binaries until ctest. A binary built for another ISA still needs compatible execution hardware. |
| `benchmarks/bench_gemm.cpp` | Fixed-size square template registrations allow inlining and constant-size optimization. Allocation/random initialization are outside timing, but each kernel's zeroing and packing are inside. Uses thread CPU time except Accelerate's wall time. Prefetch distances and padded B experiments are separate registrations. |
| `benchmarks/bench_gemm_cuda.cpp` | Allocates device problems once, warms up, then reports event durations as manual time. Host transfers and preconversion for levels 7-8/cuBLAS FP16 are excluded; level 5/6 tile conversion remains included. Precision counters distinguish storage/compute paths but do not specify error bounds. |
| `benchmarks/bench_gemm_metal.cpp` | Inputs resident in GPU-visible memory before timing; one warm-up call compiles the pipeline. Each iteration is one command buffer of `reps` GEMMs (≥ ~5 ms, so the command buffer's fixed cost does not dominate at N=256), reported as manual time with the flop counter scaled by reps. Also tile variants and the FMA peak. |
| `benchmarks/bench_gemm_lowp.cpp`, `benchmarks/bench_gemm_lowp_cuda.cpp` | Low-precision kernels with the FP32 kernel of the same unit (CPU), or `Mma` / `Tma` / `Cublas` per format (GPU, event time). Inputs converted before timing. The GPU benchmark also reports `err`: the worst of 256 sampled elements against a double-precision recomputation. |
| `benchmarks/CMakeLists.txt` | Registers experiments only for compatible architectures/toolchains, GPU experiments only with CUDA, and SME experiments only after the runtime probe passes. GNU inline-assembly CPU probes are omitted on MSVC. |

## Experiment files

These measure mechanisms rather than provide production kernels. Their
assumptions belong with their results; running on a different architecture
does not automatically preserve a cycle or cache interpretation.

| File under `benchmarks/experiments/` | Design and interpretation |
|---|---|
| `core_clock.hpp` | Times a long chain of dependent integer adds and estimates GHz assuming one add per cycle. Best of 20 reduces interruptions; scheduler migration, frequency changes and a different add latency can bias the estimate. This is a calibration for the measured machines. |
| `simd_isa.hpp` | Small vector-operation adapters let FMA and micro-kernel experiments share one loop while changing ISA/precision. It is experiment infrastructure, not runtime dispatch. |
| `fma_peak.cpp` | Named independent accumulators plus compiler barriers prevent dead-code elimination and artificial merging. Sweeping chains separates latency from throughput; excess chains can spill and then measure register pressure. Reports best of five. |
| `simd_microkernel.cpp` | Runs the library micro-kernel (`vector_micro` with `Q`, `Ymm`, `Zmm`) alone for several row counts on repeatedly reused data, comparing packed B with ldb=256/1024. Strides are compile-time constants here, so the code generated can differ from the library kernel's. The accessed data is small, but stride conflicts can keep it from remaining in L1 despite fitting by capacity. Reports best of five. |
| `cache_chase.cpp` | A random cyclic pointer chain defeats independent-load overlap and sequential prefetch. Size, stride and one-line-per-page sweeps probe cache/TLB latency. Its 128-byte lines and 16 KiB pages model Apple Silicon; they are not queried portably. |
| `sme_fmopa.cpp` | Explicitly zeros ZA, then repeats constant-vector outer products on 1/2/4/8 tile chains. Multiple threads probe shared matrix-unit capacity. Threads are scheduled by the OS, not pinned; cluster placement is inferred. The printed flop counts assume M4's 512-bit SVL. |
| `sme_b_layout.cpp` | Compares identical logical products with padded or packed B. Both full-driver timing and intrusive phase clocks are recorded in shuffled order; phase medians need not sum to end-to-end medians. Validates every layout before timing. |
| `sme_b_stride_probe.cpp` | Packs A and constructs B before timing only the macro-kernel. Height sweeps expose reused-strip capacity; width sweeps measure remaining costs after padding. kNr=32 encodes the measured M4 SVL. Nine shuffled repetitions. |
| `avx512_b_stride.cpp` | Part 3's B-stride experiment on x86 (Linux, AVX-512): a pointer chase on 2 MiB huge pages, and part 1's blocked FP32 GEMM with B at any row stride, in place or packed, over stride, size and strip height. Uses the library micro-kernel and checks every variant against `gemm_simd_blocked<Zmm>`. Builds a stub without AVX-512. |
| `sme_block_sweep.cpp` | Repeats the packed driver with runtime Mc/Kc/Nc combinations and reports medians. Includes allocation and packing, allowing end-to-end block-size comparisons. Unlike the shuffled stride probes, configurations run in a fixed order; account for clock/load drift when interpreting small gains. |
| `lowp_peak.cpp` | `fma_peak`'s method for the low-precision instructions (NEON `FMLAL`, `FMLA .8h`, `BFDOT`, `BFMMLA`, `SDOT`, `SMMLA`; AVX-512 `VDPBF16PS`, `VPDPBUSD`; SME widening `FMOPA` / `SMOPA`), next to FP32 on the same unit. |
| `lowp_accuracy.cpp` | Relative Frobenius error of each format, from FP32 inputs, against the exact product; INT8 with per-row / per-column scales; fp16 accumulation emulated by rounding every partial sum. |
| `cuda_peak.cu` | Separate FMA chains, raw MMA chains and a large copy estimate rates for the tested compute/copy instruction paths, not universal hardware ceilings. Best-of-five events; hold mode permits external clock/power sampling. Also measures every low-precision `mma.sync` form; the block-scaled FP8 row appears only in an `sm_120a` build. The launch count is based on the thread limit; registers can prevent all nominal blocks residing at once. The FP64 probe understates the measured GEMM rate. |

## Build, publishing and data files

| Files | Idea and maintenance contract |
|---|---|
| `CMakeLists.txt` | Selects optimization flags by language, probes SME2/fp64 by compiling and executing real instructions, finds Accelerate, pins FetchContent releases and chooses CUDA or stub. hpc_math_core carries header paths/link requirements; hpc_math_cuda supplies compiled launchers; hpc_math_metal (Apple) embeds `gemm.metal` as a string at configure time and compiles the Objective-C++ host. Native flags target the build host; this is not an installed package or cross-compilation workflow. |
| `.github/workflows/build.yml` | Hosted Linux x86/ARM and macOS correctness builds with cached dependencies and compiler objects. x86-64-v3 bounds the ISA for cached x86 binaries. These jobs do not establish CUDA/SME correctness or performance. |
| `.github/workflows/pages.yml` | Builds Jekyll from docs and deploys the resulting article site through GitHub Pages on main. It does not run kernel benchmarks. |
| `.clang-format`, `.clang-tidy` | Formatting and static-analysis policy; their presence does not mean a lint job runs in CI. |
| `.gitignore` | Excludes build products, editor state and root-level benchmark JSON while keeping recorded experiment data. |
| `LICENSE` | MIT license for project code; fetched dependencies retain their own licenses. |
| `README.md`, `src/gemm/README.md` | Entry point and compact kernel map. Headline results are measured best-size values, not hardware specifications or universal speed guarantees. |
| `docs/build.md`, `docs/benchmarks.md`, `docs/cache-behavior.md` | Build instructions, historical measured tables and explanatory cache models. Timings from distinct sessions must remain labeled rather than being combined into a synthetic experiment. |
| `docs/articles/*.md` | Five connected investigations: CPU SIMD, SME2 design, SME2 stride conflicts, CUDA, and low-precision formats. Front matter supplies publishing metadata; explanatory pseudocode can omit guards present in the source. |
| `docs/index.md`, `docs/_layouts/article.html`, `docs/_config.yml` | Article index, shared HTML/CSS template and Jekyll settings. Only index/articles publish; project guides and experiments remain linked on GitHub. |
| `docs/articles/assets/render_charts.py` | Standard-library SVG generator. CSVs beside it hold curated chart values; strip-height medians are calculated from raw experiment samples. Labels distinguish categorical x axes from numerical spacing. |
| `docs/articles/assets/*.csv`, `*.svg` | Curated numeric inputs and generated charts. Keep SVGs reproducible from those inputs; do not edit generated numbers independently. |
| `docs/experiments/cpu-simd/*` | Raw FMA, micro-kernel and GEMM runs, plus focused post-tile-change measurements. README identifies compiler, load and which rows replace the older tile. |
| `docs/experiments/tile-sweeps/*`, `docs/kernels/*` | `bench_gemm` register-tile sweeps on Zen 5 and M4, the re-measured default kernels, and the pages that compare the shapes. |
| `docs/experiments/sme2-gemm/*` | Raw SME ladder, block-sweep and vendor-comparison samples behind part 2; the README records their provenance. |
| `docs/experiments/zen5-b-stride/*` | Pointer chase and stride / size / height sweeps behind part 3's Zen 5 section; the README lists the cache geometry and method. |
| `docs/experiments/lowp/*` | Instruction peaks, CPU and GPU GEMM runs, accuracy and Nsight counters behind part 5; the README lists machines, builds and method. |
| `docs/experiments/sme2-machine/*` | Cache/TLB/FMOPA samples and metadata for the M4 machine model. TLB capacities and cluster sharing are experimental inferences. |
| `docs/experiments/sme-b-stride/*` | Phase timings and stride/height/width/chase data; metadata includes commands and source hashes for historical reproduction. |
| `docs/experiments/metal-m4/*` | Metal ladder, MPS and FMA peak in one bracketed GPU session, plus the tile-variant sweep; the README records the clock and power caveats. |
| `docs/experiments/frameworks-m4/*` | NumPy / PyTorch / JAX CSVs and metadata (versions, BLAS) on the M4 Max. |
| `docs/experiments/sme-mt/*` | `gemm_sme_mt` thread × A-block sweep and the default against `gemm_sme` and Accelerate. |
| `docs/experiments/cuda-machine/*` | GPU peaks, clock samples, full GEMM output and Nsight counters. Profiling perturbs timings; profile rates and benchmark rates need not coincide. |

Primary specifications for the architectural contracts:
[Arm ACLE](https://arm-software.github.io/acle/main/acle.html),
[NVIDIA WMMA requirements](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#wmma).
