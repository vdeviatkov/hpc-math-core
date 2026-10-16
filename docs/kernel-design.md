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

### Concurrent addition: `include/hpc/lowp.hpp`

This header appeared while the review was running. It introduces two-byte
bf16/fp16 storage structs, software float conversions and elementwise Matrix
conversion; int8 uses the standard integer type. Conversion uses integer bit
manipulation to implement ties-to-even rounding, with explicit subnormal,
overflow and NaN handling. It is not integrated into the GEMM API reviewed
here. Its numerical edge cases and bitwise agreement with hardware conversion
were not validated by this review; in particular, NaN payload/canonicalization
and fast-math behavior need dedicated tests before making a bit-for-bit claim.

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

All four families retain naive, reordered and blocked measurement points.
Their naive kernels build vectors from strided B-column elements and reduce
vector accumulators to a scalar dot product. Reordered kernels broadcast A
and vectorize the contiguous j loop. Blocked kernels keep a small rectangular
C tile in registers for a whole k-block, amortizing C traffic and reusing
both A broadcasts and B loads. None packs B. Full SIMD loads are unaligned
where needed, because the Matrix base alignment does not align every row.

### `src/gemm/avx2.hpp`

256-bit vectors contain 8 floats or 4 doubles. The blocked kernel uses
64x256x256 outer M/K/N tiles and a 4-row by 2-vector register tile: 4x16
float or 4x8 double. Eight accumulators, two B vectors and up to four A
broadcasts fit the 16 YMM registers. Explicit micro-kernels expose this
register budget clearly. Scalar j tails accumulate dot products for the
remaining columns; scalar i tails use i-k-j. The prefetch family reuses
these micro-kernels. AVX2 alone is insufficient: the code emits FMA too.

### `src/gemm/avx512.hpp`

512-bit vectors double the lane count. Outer tiles are 64x256x512; the
micro-kernel holds eight rows by two vectors, or 8x32 float / 8x16 double,
in 16 of 32 ZMM registers. The larger tile supplies more independent FMA
chains and encourages GCC to load B once instead of folding a repeated
memory operand into every FMA. One templated micro-kernel handles both
types with `if constexpr`; scalar row/column tails preserve arbitrary shapes.
The register shape is supported by the recorded Zen 5 experiment, not a
universal optimum for all AVX-512 CPUs or compiler versions.

### `src/gemm/neon.hpp`

AArch64 NEON provides 32 128-bit registers, containing 4 floats or 2 doubles.
Outer tiles are 64x256x256. Float uses a 4x16 tile (4x4 vectors); double
uses 8x4 (8x2 vectors), both with 16 accumulator chains. The double tile's
extra rows hide FMA latency better than the former 4-row tile on M4.
Broadcasts may become by-element FMLA instructions; that is a compiler
choice, not an intrinsic guarantee. Scalar tails handle incomplete tiles.
AArch64 is required because these kernels include double-precision FMA.

### `src/gemm/sve.hpp`

SVE reads the calling thread's vector length at runtime. The blocked tile
is four rows by two vectors, inside 64x256x512 outer tiles. Predicates mask
incomplete column vectors, so there is no scalar column cleanup; leftover
rows use scalar loops. The naive kernel uses a reusable vector buffer for
its manual column gather. Inactive predicates prevent memory accesses,
but C++ still requires valid pointer arithmetic: an entirely inactive second
vector reuses the first vector's address. Callers must keep VL stable for
the duration of a call. This family has not been measured on physical SVE
hardware in this project; NEON or SME results do not validate its speed.

### `src/gemm/prefetch.hpp`

Each variant retains its base blocked traversal and micro-kernel, adding
hints for a future A row, the start of the next B k-block, and a future C
row. PfDist counts micro-kernel row groups (individual rows for scalar).
The hints cover particular lines, not whole tiles. `pf<Rw,Loc>` maps to
`__builtin_prefetch`, or x86 `_mm_prefetch` under MSVC; the latter ignores
the write intent. Locality is a compiler/architecture-dependent hint,
not a guaranteed cache destination. Distances 2/4/8/16 are benchmarked;
the recorded runs show little benefit. Repeated outer loops are deliberate:
they keep the prefetch placement visible without changing the base kernel.

## Matrix engines and vendor references

### `src/gemm/sme.hpp`

SME FMOPA adds an outer product into a two-dimensional ZA tile. Streaming
vector length is independent of non-streaming SVE. On the measured M4,
SVL is 16 floats or 8 doubles. `pack_a` transposes row-major A into
column-contiguous strips using NEON 4x4 or 2x2 transposes plus scalar edges;
`pack_b` copies B row segments into contiguous strips. Both pad edges.

- `gemm_sme_1tile` packs all A into SVL-row strips and streams B in place,
  using only ZA tile 0. One dependency chain exposes FMOPA latency.
- `gemm_sme_tiles` packs A into 2*SVL-row strips and uses all ZA tiles:
  2x2 for float, 2x4 for double. Independent accumulators hide latency,
  and each loaded vector feeds multiple outer products.
- `gemm_sme_blocked` keeps that micro-kernel and adds jc/pc/ic blocking.
  B remains in the caller's layout, exposing power-of-two stride conflicts.
- `gemm_sme` additionally packs B once per jc/pc panel. Contiguous strips
  control the stride regardless of the input N and amortize copying over
  all ic blocks.

Mc=1024 float / 512 double, Kc=2048 and Nc=4096 aim for an 8 MiB packed A
block in M4 Max's shared 16 MiB L2. The streaming macro-kernel visits B
strips then A strips; C stays in ZA through kc, and is reloaded only when
pc>0. Predicated C transfers and row bounds handle edges; full tiles use a
separate inlined transfer path so edge address checks stay out of that path. ZA helpers declare
shared state so the compiler does not insert private-ZA saves in the hot
loop. Explicit zeroing initializes each new accumulation. Allocation and
packing happen outside streaming mode, but inside each timed GEMM call.
The L2 explanation is an inference from controlled measurements; Apple's
physical cache indexing and matrix-engine internals are undocumented.

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

### `src/cuda/gemm_kernels.cu`

The file keeps the eight kernels together so their layouts and reusable
primitives can be compared directly. `load_tile` coalesces global reads,
optionally transposes stores and zero-fills edges. `tile_fma` computes an
outer product in registers; `store_tile` clips output edges. `copy_or_zero`
uses cp.async on sm_80+ when source and destination types agree.

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

The two buffered kernels commit copies, wait for each thread's copies, then
use a block barrier to make data visible and protect buffer reuse. Padding
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

### `src/cuda/gemm_kernels_stub.cpp`

This supplies the same linkable symbols when no toolkit is found. Device
count is zero, capability queries are false, free is harmless, and compute,
copies, synchronization, allocation and timing throw. This preserves build
coverage and explicit skip behavior without pretending CPU work is CUDA.

## Correctness and benchmark files

| File | Purpose and reasoning |
|---|---|
| `tests/test_gemm.cpp` | Hand-checked reference products, then a table of available CPU kernels against that reference. Boundary shapes exercise vector/register/cache tails, rectangular matrices, split K accumulation and empty dimensions. C starts as junk to distinguish overwrite from accumulation. SME additionally checks bit-preserving A packing and padded B strides. |
| `tests/test_matrix.cpp` | Checks alignment, deep copies, shape-preserving empty copies, safe moved-from objects and overflow rejection before allocation. These are storage invariants all kernels depend on. |
| `tests/test_gemm_cuda.cpp` | Per-kernel shape tables exercise CUDA bounds, vector and pipelined fallbacks, reduced precision, empty products and rectangular vendor calls. Direct device checks cover zero-K clearing and rejection of invalid launch shapes. Missing GPUs skip explicitly. |
| `tests/CMakeLists.txt` | Links CPU tests and CUDA-or-stub tests separately. PRE_TEST discovery delays executing binaries until ctest. A binary built for another ISA still needs compatible execution hardware. |
| `benchmarks/bench_gemm.cpp` | Fixed-size square template registrations allow inlining and constant-size optimization. Allocation/random initialization are outside timing, but each kernel's zeroing and packing are inside. Uses thread CPU time except Accelerate's wall time. Prefetch distances and padded B experiments are separate registrations. |
| `benchmarks/bench_gemm_cuda.cpp` | Allocates device problems once, warms up, then reports event durations as manual time. Host transfers and preconversion for level 7/cuBLAS FP16 are excluded; level 5/6 tile conversion remains included. Precision counters distinguish storage/compute paths but do not specify error bounds. |
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
| `simd_microkernel.cpp` | Sweeps register-tile rows and vectors on repeatedly reused data, comparing packed B with ldb=256/1024. The accessed data is small, but stride conflicts can keep it from remaining in L1 despite fitting by capacity. Reports best of five. |
| `cache_chase.cpp` | A random cyclic pointer chain defeats independent-load overlap and sequential prefetch. Size, stride and one-line-per-page sweeps probe cache/TLB latency. Its 128-byte lines and 16 KiB pages model Apple Silicon; they are not queried portably. |
| `sme_fmopa.cpp` | Explicitly zeros ZA, then repeats constant-vector outer products on 1/2/4/8 tile chains. Multiple threads probe shared matrix-unit capacity. Threads are scheduled by the OS, not pinned; cluster placement is inferred. The printed flop counts assume M4's 512-bit SVL. |
| `sme_b_layout.cpp` | Compares identical logical products with padded or packed B. Both full-driver timing and intrusive phase clocks are recorded in shuffled order; phase medians need not sum to end-to-end medians. Validates every layout before timing. |
| `sme_b_stride_probe.cpp` | Packs A and constructs B before timing only the macro-kernel. Height sweeps expose reused-strip capacity; width sweeps measure remaining costs after padding. kNr=32 encodes the measured M4 SVL. Nine shuffled repetitions. |
| `sme_block_sweep.cpp` | Repeats the packed driver with runtime Mc/Kc/Nc combinations and reports medians. Includes allocation and packing, allowing end-to-end block-size comparisons. Unlike the shuffled stride probes, configurations run in a fixed order; account for clock/load drift when interpreting small gains. |
| `cuda_peak.cu` | Separate FMA chains, raw MMA chains and a large copy estimate rates for the tested compute/copy instruction paths, not universal hardware ceilings. Best-of-five events; hold mode permits external clock/power sampling. The launch count is based on the thread limit; registers can prevent all nominal blocks residing at once. The FP64 probe understates the measured GEMM rate. |

## Build, publishing and data files

| Files | Idea and maintenance contract |
|---|---|
| `CMakeLists.txt` | Selects optimization flags by language, probes SME2/fp64 by compiling and executing real instructions, finds Accelerate, pins FetchContent releases and chooses CUDA or stub. hpc_math_core carries header paths/link requirements; hpc_math_cuda supplies compiled launchers. Native flags target the build host; this is not an installed package or cross-compilation workflow. |
| `.github/workflows/build.yml` | Hosted Linux x86/ARM and macOS correctness builds with cached dependencies and compiler objects. x86-64-v3 bounds the ISA for cached x86 binaries. These jobs do not establish CUDA/SME correctness or performance. |
| `.github/workflows/pages.yml` | Builds Jekyll from docs and deploys the resulting article site through GitHub Pages on main. It does not run kernel benchmarks. |
| `.clang-format`, `.clang-tidy` | Formatting and static-analysis policy; their presence does not mean a lint job runs in CI. |
| `.gitignore` | Excludes build products, editor state and root-level benchmark JSON while keeping recorded experiment data. |
| `LICENSE` | MIT license for project code; fetched dependencies retain their own licenses. |
| `README.md`, `src/gemm/README.md` | Entry point and compact kernel map. Headline results are measured best-size values, not hardware specifications or universal speed guarantees. |
| `docs/build.md`, `docs/benchmarks.md`, `docs/cache-behavior.md` | Build instructions, historical measured tables and explanatory cache models. Timings from distinct sessions must remain labeled rather than being combined into a synthetic experiment. |
| `docs/articles/*.md` | Four connected investigations: CPU SIMD, SME2 design, SME2 stride conflicts, and CUDA. Front matter supplies publishing metadata; explanatory pseudocode can omit guards present in the source. |
| `docs/index.md`, `docs/_layouts/article.html`, `docs/_config.yml` | Article index, shared HTML/CSS template and Jekyll settings. Only index/articles publish; project guides and experiments remain linked on GitHub. |
| `docs/articles/assets/render_charts.py` | Standard-library SVG generator. CSVs beside it hold curated chart values; strip-height medians are calculated from raw experiment samples. Labels distinguish categorical x axes from numerical spacing. |
| `docs/articles/assets/*.csv`, `*.svg` | Curated numeric inputs and generated charts. Keep SVGs reproducible from those inputs; do not edit generated numbers independently. |
| `docs/experiments/cpu-simd/*` | Raw FMA, micro-kernel and GEMM runs, plus focused post-tile-change measurements. README identifies compiler, load and which rows replace the older tile. |
| `docs/experiments/sme2-gemm/*` | Concurrently added raw SME ladder, block-sweep and vendor-comparison samples. The accompanying README records their provenance; the review leaves those historical samples intact. |
| `docs/experiments/sme2-machine/*` | Cache/TLB/FMOPA samples and metadata for the M4 machine model. TLB capacities and cluster sharing are experimental inferences. |
| `docs/experiments/sme-b-stride/*` | Phase timings and stride/height/width/chase data; metadata includes commands and source hashes for historical reproduction. |
| `docs/experiments/review-20261016/*` | Before/after duration tables, source hashes and review-only Mac timing harnesses. The harness loads namespace-isolated before/after libraries, randomizes paired order and offers a same-buffer SME probe. Machine load limits the conclusions; see the accompanying README. |
| `docs/review-20261016.md` | Audit scope, fixes, test evidence and explicit performance/hardware limitations. |
| `docs/experiments/cuda-machine/*` | GPU peaks, clock samples, full GEMM output and Nsight counters. Profiling perturbs timings; profile rates and benchmark rates need not coincide. |

Primary specifications for the architectural contracts:
[Arm ACLE](https://arm-software.github.io/acle/main/acle.html),
[Linux SVE thread vector lengths](https://www.kernel.org/doc/html/latest/arch/arm64/sve.html),
[NVIDIA WMMA requirements](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#wmma).
