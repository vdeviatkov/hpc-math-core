# Why these cache blocks and register tiles?

A register tile holds partial sums across K in registers (SIMD) or ZA (SME).
A cache block controls how many micro-kernel calls reuse loaded operands.
These are separate choices. Neither follows from asking whether one `float`
fits in L1: the relevant quantities are operand working sets, reuse distance,
load bandwidth, instruction latency, and register pressure.

## FP32 / FP64 defaults and their evidence

SIMD R×V means R scalar rows by V vectors of columns. SME R×V means R by V
**square ZA tiles**; with S streaming lanes per vector the output is RS×VS.
Cache dimensions below are M×K×N in elements, not bytes.

| Family | Register tile | Cache block FP32 / FP64 | Why this choice; evidence |
|---|---|---|---|
| AVX2 | 6×2 | 384×256×256 / 384×256×128 | 12 independent accumulators hide FMA latency; 12+2+1=15 vector registers fit the 16-register AVX2 target. Six rows reuse B without spilling. The 256 KiB B block is a quarter of Zen 5's 1 MiB L2; M=384 amortizes fetching it across 64 row strips. [Tile measurements](avx2.md), [block measurements](blocks.md). |
| AVX-512 | 8×3 | 512×256×512 / 512×256×256 | 24 accumulators plus 3 B registers fit 32 ZMMs. GCC's emitted code reuses B loads for this shape; smaller shapes sometimes repeat them per FMA. The 512 KiB B block leaves half the L2 for other data; M=512 reaches the measured plateau without the largest blocks' small-size cost. [Tiles](avx512.md), [blocks](blocks.md). |
| NEON | 6×4 | 384×256×256 for both | 24 accumulators+4 B+1 A=29 of 32 vector registers. Six rows reuse B; lane FMAs with scalar LDR avoid the compiler's slower LD1R schedule on M4. M=384 was chosen near the block sweep's plateau; B is 256/512 KiB, well below the P-cluster L2, rather than sized to fill it. [Tiles](neon.md), [blocks](blocks.md). |
| SVE | 4×2 | 64×256×512 for both | Unmeasured starting values, retained for compatibility, **not tuned defaults**. Eight accumulators use a modest part of the 32 Z registers. B is 512 KiB FP32 / 1 MiB FP64, which need not fit a particular SVE CPU's L2. Physical VL and cache sizes vary. [Sweep instructions and correctness evidence](sve.md). |
| SME2 | 2×2 FP32 / 2×4 FP64 | 1024×2048×4096 / 512×2048×4096 | All four/eight ZA tiles give independent FMOPA chains and reuse A/B vectors. At M4's 512-bit streaming length, C is 32×32 / 16×32 elements. Cache sizes came from an M4 sweep: A is 8 MiB, half that machine's 16 MiB P-cluster L2; a B strip is 256/512 KiB and serves 32 A strips. The 32/64 MiB packed B panel is streamed, not expected to fit L2. [SME guide](sme.md), [original block data](../experiments/sme2-gemm/README.md). |

For fixed-width SIMD, K=256 balances C reload/store overhead against operand
footprint. The reused A strip R×K occupies only 6/12 KiB (AVX2 and NEON)
or 8/16 KiB (AVX-512), so L1 reuse of A is useful even though the whole
B cache block is intended for L2 reuse. A larger K is not automatically
better: it increases each strip's working set and can amplify stride/cache
conflicts. [The block sweep](blocks.md) varies K and N independently.

The SIMD register-count formula R·V+V+1 is a starting budget, not a proof
of optimal performance: compiler scheduling, address temporaries, spills,
and actual instruction selection matter. More independent accumulators
than latency×issue-rate can still help by reducing operand traffic.

SME's M4 L2 reuse explanation is specific to the measured implementation
and machine. Capacity arithmetic alone does not prove residency, nor does
it establish an ISA-wide requirement to bypass L1. Other SME CPUs need
their own measurements. New shape results are recorded separately from the
historical default measurements.

## Other SIMD / SME variants

Naive/reordered SIMD kernels have no R×V register-blocked micro-kernel or
three-dimensional cache-block policy. Their vector width comes from the
instruction type (or runtime VL for SVE). Prefetch variants inherit the
blocked kernel's shape and blocks; distance is a separate experiment.
The scalar blocked kernel's single tile dimension is also independent of
SIMD register tiling.

The low-precision implementations have distinct instruction and packing
constraints. Their throughput has been measured, but there is no exhaustive
register/cache-shape sweep establishing all their sizes as optimal:

| Kernel | Tile / blocks | Rationale and limitation |
|---|---|---|
| NEON BF16 / INT8 matrix instructions | 8×8 C, 16 vector accumulators; Mc=256, Kc=1024 BF16 / 2048 INT8, Nc=512 | Each instruction accumulates a 2×2 output, so four row pairs × four column pairs reuse four vectors of A and four of B. Kc keeps input bytes equal across types. These are design choices, not the FP32 sweep's selected shapes. |
| NEON FP16 widening / FP16 accumulate | 4×16 / 4×32 C, 16 accumulators; Mc=256, Kc=512, Nc=512 | Inherits the earlier four-row FMA structure. Wider FP16 lanes double columns in the FP16-accumulate form. The later FP32 6×4 tuning does not establish this choice as optimal. |
| AVX-512 BF16 / INT8 | 8×32 C, 16 accumulators; Mc=128, Kc=512 BF16 / 1024 INT8, Nc=512 | Eight rows reuse each of two B vectors; a 32-column packed B strip is 32 KiB in either type, leaving headroom in Zen 5's 48 KiB L1. A is read in place. This uses a different packed layout and loop order from FP32. |
| SME widening FP16 / BF16 / INT8 | 2×2 ZA32 tiles; Mc=1024, Kc=4096 for 16-bit / 8192 INT8, Nc=4096 | All four 32-bit tiles; K groups of 2 or 4 inputs per accumulated element. Sizes inherit FP32's 8 MiB A-block budget in bytes. No separate shape/block optimum has been demonstrated. |

See [low-precision measurements](../experiments/lowp/README.md) and the
headers `simd.hpp` (with the structs in `neon.hpp` and `avx512.hpp`) and `sme.hpp`. The SME
shape/cache API covers the widening kernels too (`gemm_sme<fp16, R, V, TileM, TileK, TileN>`),
with TileK in input elements, a multiple of the k group G; the values used
(`SmeParams` in `benchmarks/sme_params.hpp`) are FP32's sizes in k groups,
not yet swept.
