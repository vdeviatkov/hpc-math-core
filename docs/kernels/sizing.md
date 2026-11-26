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
| SME2 | 1×4 FP32 / 1×4 FP64 | 1024×2048×4096 / 512×2048×4096 | Four ZA tiles give the four independent FMOPA chains the 4-cycle latency needs. 1×4 measured fastest of the 4-tile shapes in both types (FP32 +7.3% over 2×2 at N=1024, FP64 +0.4–3.7% over 2×4), although it loads one more vector per FMOPA group. At M4's 512-bit streaming length, C is 16×64 / 8×32 elements. Cache sizes came from an M4 sweep at the 1×4 shape: A is 8 MiB, half that machine's 16 MiB P-cluster L2; a B strip serves 64 A strips. The 32/64 MiB packed B panel is streamed, not expected to fit L2. [SME guide](sme.md), [block data](../experiments/sme2-gemm/README.md). |

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
their own measurements.

## Other SIMD / SME variants

Naive/reordered SIMD kernels have no R×V register-blocked micro-kernel or
three-dimensional cache-block policy. Their vector width comes from the
instruction type. Prefetch variants inherit the
blocked kernel's shape and blocks; distance is a separate experiment.
The scalar blocked kernel's single tile dimension is also independent of
SIMD register tiling.

The low-precision implementations have distinct instruction and packing
constraints. Their register tiles were swept with `bench_gemm_lowp`'s
`*Tile` rows (every shape `kTileFits` allows) and, at the chosen tiles,
their blocks with the `*Block` rows (Mc ∈ {128, 256, 512} × k block of
512–4096 bytes × Nc ∈ {256, 512, 1024} for NEON; nine Mc × Kc × Nc
combinations for SME). Every block default is the best or within the
confirmation run's variation of it (≤ 3.7%, cv ≤ 3.5%; where several tie,
the smaller block is kept). M4 data in
[lowp/sweeps-m4](../experiments/lowp/sweeps-m4/), Zen 5 in
[lowp/sweeps-zen5](../experiments/lowp/sweeps-zen5/):

| Kernel | Tile / blocks | Rationale and limitation |
|---|---|---|
| NEON BF16 / INT8 matrix instructions | 4×4 tiles of 2×2 (8×8 C), 16 accumulators; Mc=256, Kc=1024 BF16 / 2048 INT8, Nc=512 | Each instruction accumulates a 2×2 output. Swept over 11 shapes (2×4 … 8×2): BF16 runs at 64–68 GFLOP/s in every shape, within the run's variation (BFMMLA's rate sets it, not the shape); INT8 4×4 and 2×8 tie as the fastest (497–508 GOP/s from N = 1024), 2×4 and 4×2 trail by 30%. |
| NEON FP16 widening (FMLAL) | 4×6 (4×24 C), 24 accumulators; Mc=256, Kc=512, Nc=512 | Measured fastest of 4×2, 4×4, 4×6: 8–10% ahead of 4×4 at N = 256 … 4096 (7 interleaved repetitions). |
| NEON FP16 accumulate (FMLA .8h) | 4×5 (4×40 C), 20 accumulators; Mc=256, Kc=512, Nc=512 | Measured fastest of 4×2 … 4×6: 4–10% ahead of 4×4, within 1% of 4×6. |
| AVX-512 BF16 / INT8 | 8×2 (8×32 C), 16 accumulators; Mc=128, Kc=512 BF16 / 1024 INT8, Nc=512 | Swept over 10 shapes and two k blocks on Zen 5 ([data](../experiments/lowp/sweeps-zen5/)): 8×2 is the fastest at N = 256 and 1024 (BF16) and 1024 and 2048 (INT8); where another shape leads (8×3 BF16 at N ≥ 2048, 4×4 and 6×4 INT8 at 256 and 4096) it is by ≤ 3.7%, within the run's variation (cv ≤ 4.6%). A 32-column packed B strip is 32 KiB in either type, inside Zen 5's 48 KiB L1. A is read in place (the k-group kernel of `gemm_simd_packed`). |
| SME widening FP16 / BF16 / INT8 | 1×4 ZA32 tiles; Mc=1024, Kc=4096 for 16-bit / 8192 INT8, Nc=4096 | All four 32-bit tiles; K groups of 2 or 4 inputs per accumulated element. 1×4 measured 0.8–6.1% ahead of 2×2 at every size and type, 4×1 behind both. |

## GPU kernels

The Metal and CUDA kernels choose a block tile per thread block and a tile
per thread, simdgroup or warp. Their sizes and how each was chosen:
[metal.md](metal.md) (every level swept at N = 256 … 4096: Level 1's
32 × 32 and Level 2's 8 × 4 per thread lead from N = 1024 up; Levels 3–5
are the best or within the run's variation where the GPU is full) and [cuda.md](cuda.md)
(Level 8's tile and the low-precision kernel's stages were swept; Levels
2–7 sizes are design choices, apart from FP64's 64 × 64 forced by shared
memory).

See [low-precision measurements](../experiments/lowp/README.md) and the
headers `simd.hpp` (with the structs in `neon.hpp` and `avx512.hpp`) and `sme.hpp`. The SME
shape/cache API covers the widening kernels too (`gemm_sme<fp16, R, V, TileM, TileK, TileN>`),
with TileK in input elements, a multiple of the k group G; the values used
(`SmeParams` in `benchmarks/sme_params.hpp`) are FP32's sizes in k groups.
