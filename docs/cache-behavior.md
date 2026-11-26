# Cache Behaviour of Matrix Multiplication

The theory behind the GEMM kernels in [`src/gemm/`](../src/gemm/README.md):
how data moves through the memory hierarchy, why loop order and tiling decide
performance, and where the GPU hierarchy differs from the CPU one. Numbers
come from the two CPUs and two GPUs benchmarked in this repository, unless
marked as a vendor specification.

| Machine | Role in this repository | Measurements |
|---|---|---|
| AMD Zen 5 (Ryzen, one core) | x86 SIMD kernels (AVX2, AVX-512) | [zen5-b-stride](experiments/zen5-b-stride/README.md) |
| Apple M4 Max (one P-core) | NEON and SME2 kernels | [sme2-machine](experiments/sme2-machine/README.md) |
| NVIDIA GeForce RTX 5080 (sm_120) | CUDA kernels | [cuda-machine](experiments/cuda-machine/README.md) |
| Apple M4 Max, 40-core GPU | Metal kernels | [metal-m4](experiments/metal-m4/README.md) |

## Contents

1. [CPU memory hierarchy](#1-cpu-memory-hierarchy)
2. [Cache lines and spatial locality](#2-cache-lines-and-spatial-locality)
3. [Row-major layout](#3-row-major-layout)
4. [Reuse distance](#4-reuse-distance)
5. [Working set of the inner loop](#5-working-set-of-the-inner-loop)
6. [Hardware prefetching](#6-hardware-prefetching)
7. [Loop tiling](#7-loop-tiling)
8. [Bandwidth and the roofline](#8-bandwidth-and-the-roofline)
9. [GPU memory hierarchy](#9-gpu-memory-hierarchy)
10. [Summary](#10-summary)

---

## 1. CPU memory hierarchy

A load is served by the nearest level that holds the line. Each level is
larger and slower than the one above it:

| Level | AMD Zen 5 | Apple M4 Max (P-core) |
|---|---|---|
| L1 data | 48 KiB per core, 4 cycles | 128 KiB per core, 3 cycles |
| L2 | 1 MiB per core (private), 14 cycles | 16 MiB per 6-core P-cluster (shared), 26–27 cycles |
| Last-level cache | L3: 32 MiB per 8-core CCD, ~45 cycles | SLC: ~48 MB per chip (unofficial; 32–64 MiB measured), ~151 cycles |
| DRAM | ~200 cycles | 377–437 cycles (96–111 ns) |

Latencies are from a dependent pointer chase in random order
([Zen 5](experiments/zen5-b-stride/README.md#reading-the-chase),
[M4](experiments/sme2-machine/README.md#load-latency-by-working-set)).

### The last level is a different kind of cache on each machine

- **Zen 5 L3** is a victim cache, mostly exclusive of the L2s. It is filled
  with lines the L2s of its CCD evict, not on the way in from memory, and it
  belongs to the CPU cores alone.
- **Apple SLC** is a memory-side cache in front of the DRAM controllers,
  shared by both CPU clusters, the GPU, the Neural Engine and the other
  agents on the SoC. Apple documents neither its size nor its allocation
  policy; the commonly cited figure for the Max chips is about 48 MB, from
  third-party estimates rather than an Apple specification. In the pointer chase a 32 MiB working set is served from the SLC
  (151 cycles) and a 64 MiB working set from DRAM (377 cycles), so the
  capacity one core can use is **between 32 and 64 MiB**. That figure is a
  bound under GPU-idle conditions, not a reserved allocation: other agents
  compete for the same capacity. The ~48 MB estimate is consistent with that
  bound.

The consequence for kernels: on Zen 5 a line evicted from L2 returns from L3
in ~45 cycles; on the M4 it returns from the SLC in ~150, about three times
the cost ([SME2 cache associativity](articles/sme2-cache-associativity.md)).
M4 block sizes are therefore chosen to stay inside the 16 MiB L2 rather than
to rely on the SLC.

### Compute-bound and memory-bound

A kernel is *compute-bound* when the arithmetic units are the bottleneck and
*memory-bound* when they stall waiting for data from a lower level. Naive
GEMM is memory-bound at all but the smallest sizes.

---

## 2. Cache lines and spatial locality

The unit of transfer between levels is the **cache line**: 64 bytes on Zen 5,
128 bytes on Apple M-series (`sysctl hw.cachelinesize`). Reading one 8-byte
`double` brings in its whole line:

```
DRAM layout (64-byte lines):
offset  0  8 16 24 32 40 48 56 64 72 80 …
       [d0 d1 d2 d3 d4 d5 d6 d7|d8 d9 …]
        ←──── one cache line ────→

Loading d0 brings {d0 … d7} into L1; d1 … d7 then hit in L1.
```

This is **spatial locality**: an access pattern that uses every byte of every
line it loads gets the full bandwidth of each level. The examples below use
64-byte lines (8 doubles); on the M4 every count of lines halves.

---

## 3. Row-major layout

`hpc::Matrix<T>` stores elements in row-major order:

```
element (i, j)  →  data[i * cols + j]
```

- Elements of one row are adjacent (stride 1).
- Elements of one column are `cols * sizeof(T)` bytes apart: 8 KiB for an
  N = 1024 matrix of doubles, so every element of a column is on its own line.

This asymmetry is the root cause of naive GEMM's poor performance.

---

## 4. Reuse distance

The **reuse distance** of an access is the number of distinct cache lines
touched since the previous access to the same line. In a fully associative
LRU cache, a reuse distance larger than the capacity guarantees a miss. Real
caches are set-associative and can evict earlier through conflicts;
replacement policy and prefetching also change the result.

### Naive order (i-j-k)

```cpp
for i: for j: for k:  C(i,j) += A(i,k) * B(k,j);
```

For fixed `j`, the inner loop reads `B(0,j), B(1,j), …, B(K-1,j)`, one line
per element, using 8 of each line's 64 bytes. The other 7 doubles are needed
for `j+1 … j+7`, but only after the k-loop has touched about `K` other lines
of B. At K = 1024 that reuse distance is 64 KiB, larger than Zen 5's 48 KiB
L1: every line is evicted before its reuse and every B access misses L1.

### Reordered (i-k-j)

```cpp
for i: for k: for j:  C(i,j) += A(i,k) * B(k,j);
```

The inner loop walks `B(k,0), B(k,1), …` and `C(i,0), C(i,1), …` with
stride 1. All 8 doubles of a line are used back to back (reuse distance 0),
`A(i,k)` stays in a register, and the prefetcher can run ahead of both
streams.

---

## 5. Working set of the inner loop

Lines touched by one execution of the innermost loop:

| Array | Naive (fixed i, j) | Reordered (fixed i, k) |
|---|---|---|
| A | row i: K elements, K/8 lines | `A(i,k)`: 1 element, held in a register |
| B | column j: K elements, **K lines** | row k: N elements, N/8 lines |
| C | `C(i,j)`: 1 element, 1 line | row i: N elements, N/8 lines |
| **Total** | **≈ 1.125 K lines** | **N/4 lines, streamed** |

At K = 1024 the naive loop touches ~1,152 lines (72 KiB), more than Zen 5's
L1, so B thrashes. The reordered loop touches more lines in total at large N,
but consumes them sequentially: at any instant only the current line of B
and of C is live, and the prefetcher hides the rest.

---

## 6. Hardware prefetching

Prefetchers detect sequential and constant-stride streams and fetch lines
before they are requested.

- The reordered kernel's stride-1 walks are the easiest case.
- The naive kernel's column walk has a constant stride too (8 KiB at
  N = 1024), which stride prefetchers can follow. Prefetching hides latency,
  not waste: each fetched line still delivers 8 useful bytes out of 64.

---

## 7. Loop tiling

The reordered kernel still re-reads all of B, `K × N` elements, for every
row of C. Once B exceeds the last-level cache that traffic goes to DRAM,
M times over. **Tiling** (blocking) restructures the loops so a tile of each
operand is reused many times while it is cache-resident (`gemm_blocked`):

```
for i0 in [0, M, Tr):
  for k0 in [0, K, Tk):
    for j0 in [0, N, Tc):
      // Tr×Tk of A, Tk×Tc of B and Tr×Tc of C stay in cache
      for i in [i0, min(i0+Tr, M)):
        for k in [k0, min(k0+Tk, K)):
          for j in [j0, min(j0+Tc, N)):
            C(i,j) += A(i,k) * B(k,j)
```

Three 64×64 tiles of doubles occupy `3 × 64 × 64 × 8 B = 96 KiB`: more than
Zen 5's L1, within M4's 128 KiB L1, and well inside the L2 of both machines.
The production kernels tile at several levels at once (registers, L1, L2);
their block sizes are measured, not derived, and are documented in
[kernels/blocks.md](kernels/blocks.md) and [kernels/sizing.md](kernels/sizing.md).

---

## 8. Bandwidth and the roofline

A kernel's attainable rate is bounded by both compute and memory:

```
Attainable FLOP/s = min( Peak FLOP/s,  Bandwidth (B/s) × Arithmetic intensity (FLOP/B) )
```

**Naive GEMM, worst case.** Each multiply-add (2 FLOPs) fetches a 64-byte
line of B to use 8 bytes: an intensity of 2 / 64 ≈ 0.03 FLOP/B. Whatever the
core's peak, the kernel cannot exceed ~0.03 × the DRAM bandwidth. Caches
soften this at moderate N, but the trend is visible: naive f64 on the M4 Max
falls from 8.7 GFLOP/s at N = 64 to 0.62 GFLOP/s at N = 4096
([benchmarks.md](benchmarks.md#apple-m4-max)).

**Tiled GEMM.** A T×T tile scheme moves about `2 × N³ / T` elements for
`2 × N³` FLOPs, so intensity grows linearly with the tile size until the
kernel reaches the compute roof:

```
Peak compute = cores × SIMD lanes × FMAs per cycle × 2 FLOP per FMA × frequency
```

Reaching that roof needs the explicit SIMD kernels (AVX2, AVX-512, NEON) and
the matrix engines (SME2).

**Ridge points measured here.** The intensity at which a kernel stops being
memory-bound:

| Device | Peak | DRAM bandwidth | Ridge point |
|---|---|---|---|
| RTX 5080 | 59.7 TFLOP/s FP32 FMA (measured) | 812 GB/s copy (measured) | ~74 FLOP/B |
| M4 Max GPU | 14.9 TFLOP/s FP32 FMA (measured) | 546 GB/s (Apple specification) | ~27 FLOP/B |

---

## 9. GPU memory hierarchy

A GPU runs tens of thousands of threads and hides latency by switching
between them rather than with large per-thread caches. Two consequences
shape every GPU kernel in this repository:

- **The fastest storage is software-managed.** Registers and shared
  (threadgroup) memory are allocated by the kernel, and their per-thread
  budget limits how many threads can be resident (occupancy).
- **Accesses are issued per warp, not per thread.** 32 threads load
  together, and the hardware merges their addresses into as few memory
  transactions as possible.

### 9.1 NVIDIA RTX 5080 (Blackwell, sm_120)

84 SMs; each SM holds up to 48 resident warps (1,536 threads).

| Tier | Scope | Capacity (RTX 5080) | Managed by | Used in this repository |
|---|---|---|---|---|
| Registers | thread | 64 K × 32-bit (256 KiB) per SM; ≤ 255 per thread | compiler | register tiles from Level 2 on |
| Uniform registers | warp | 63 per warp | compiler | implicitly (addresses, loop counters) |
| Shared memory | thread block | ≤ 100 KiB per SM, ≤ 99 KiB per block | kernel | operand tiles from Level 1 on |
| L1 / texture cache | SM | 128 KiB array, shared with shared memory | hardware | all global loads |
| Constant cache | SM | 8 KiB working set over 64 KiB constant memory | hardware | kernel parameters |
| Distributed shared memory | thread block cluster | shared memory of the cluster's SMs | kernel | not used |
| L2 cache | device | 64 MiB | hardware | all device traffic |
| Device memory (GDDR7) | device | 16 GiB, 960 GB/s spec, 812 GB/s measured copy | — | A, B, C |

Capacities are the compute capability 12.0 limits from the CUDA Programming
Guide and `cudaGetDeviceProperties`; bandwidth measured with
[`cuda_peak`](experiments/cuda-machine/README.md#peaks).

**Registers.** The largest and fastest on-chip store. Registers per thread
trade against occupancy: at 128 registers per thread an SM holds 512 threads
(16 warps), a third of its maximum, which is the 32% occupancy measured for
Levels 2–4 ([profiles](experiments/cuda-machine/README.md#profiles)). Those
levels are still 7–8× faster than Level 1 at 99.7% occupancy: reuse from
registers matters more than thread count. Values a kernel cannot keep in
registers spill to *local memory*, which is per-thread but physically in
device memory, cached by L1 and L2.

**Uniform registers.** Since Turing each warp also has a small uniform
register file and datapath for values that are identical across its 32
threads: base addresses, loop bounds, tile indices. The compiler assigns
them (`UR` operands in SASS); moving warp-invariant arithmetic there frees
regular registers and issue slots. There is no source-level control over
them.

**Shared memory and L1.** One 128 KiB SRAM array per SM is split between
shared memory and the L1 / texture cache; whatever shared memory a launch
does not claim serves as L1. Shared memory is divided into 32 banks of 4
bytes; a warp access whose threads hit different addresses in the same bank
is serialised into several *wavefronts*. Level 4 removes most of these with
an XOR swizzle, and Level 7 brings them to 1.1%
([profiles](experiments/cuda-machine/README.md#profiles)). The texture path
shares the same cache and adds hardware filtering and addressing modes; it
is not used here. Read-only loads (`const __restrict__`, `__ldg`) go
through the same L1 with relaxed coherence.

**Constant cache.** `__constant__` data and kernel parameters live in a
64 KiB constant bank, read through a small per-SM cache. It broadcasts
efficiently when all threads of a warp read the same address and serialises
when they read different ones. Matrix dimensions and pointers reach every
kernel this way.

**Distributed shared memory.** From compute capability 9.0, blocks can be
grouped into a *thread block cluster* scheduled together on neighbouring
SMs; each block can then read and write the shared memory of the others
(DSMEM) and synchronise with them. The kernels here launch one block per
cluster, so they do not use it. Level 8's TMA copies target the
`shared::cluster` state space only because that is how the instruction is
defined.

**Asynchronous copies into shared memory.** Two mechanisms skip the
round-trip through registers:

- `cp.async` (sm_80+, Levels 3 and 7): each thread starts a 4–16-byte
  global-to-shared copy and continues working. The `.cg` variant bypasses
  L1, since a block reads each tile once.
- TMA, the Tensor Memory Accelerator (sm_90+, Level 8): one thread issues a
  bulk copy of a whole 2-D tile described by a tensor map, and an `mbarrier`
  signals its completion.

Consumer Blackwell (sm_120) has no Tensor Memory (TMEM) and no
`tcgen05`/`wgmma` instructions; Tensor Core operands come from registers via
`mma.sync`, loaded from shared memory with `ldmatrix`.

**L2 and coalescing.** All SMs share the 64 MiB L2. Lines are 128 bytes,
divided into four 32-byte *sectors*, the unit of transfer between L2 and
DRAM. A warp's request is served by the set of sectors it touches: 32
consecutive, aligned 4-byte loads take 4 sectors, while a strided access can
take 32. In Levels 0 and 1 a 16×16 block places one warp across two 16-thread
rows; each half-warp reads a contiguous segment of a B row (both halves read
the same segment) and writes a contiguous segment of its own C row.

### 9.2 Apple M4 Max GPU

40 GPU cores, each with 128 FP32 lanes in SIMD-groups of 32 threads. Apple
publishes far less about its GPU hierarchy than NVIDIA does; undocumented
values are marked as such.

| Tier | Scope | Capacity | Managed by | Used in this repository |
|---|---|---|---|---|
| Registers | thread | allocated dynamically (see below); size not published | compiler | register tiles from Level 2 on |
| Threadgroup memory | threadgroup | ≤ 32 KiB per threadgroup | kernel | Levels 1, 2, 4, 5 |
| `constant` address space | dispatch | read-only buffers, preloaded when small and statically sized | compiler / driver | dimensions (`constant Dims&`) |
| Texture cache | GPU core | not published | hardware | not used |
| GPU L1 / L2 caches | core / GPU | not published | hardware | all device loads |
| System-level cache | SoC, shared with CPU | ~48 MB (unofficial; 32–64 MiB measured from one CPU core) | hardware | all DRAM traffic |
| Unified memory (LPDDR5X) | SoC | 64 GB in the test machine, 546 GB/s spec | — | A, B, C |

**Dynamic Caching.** From the M3 generation on, each GPU core's on-chip SRAM
is allocated at run time between registers, threadgroup memory, tile memory
and cache, instead of being split into fixed partitions. A kernel's register
use is therefore not a fixed per-SM quotient as on NVIDIA, and occupancy
depends on what the kernel actually uses while it runs.

**Threadgroup memory.** The counterpart of CUDA shared memory, limited to
32 KiB per threadgroup (`maxThreadgroupMemoryLength`). Its bank structure is
not documented. On this GPU, staging operands in it is not automatically a
win: Level 4 (simdgroup matrices fed from threadgroup memory) is 2–10%
slower than Level 3 (fed straight from device memory) for N = 1024–4096,
because the caches already serve the reuse between simdgroups. It wins at
N = 8192 (11.4 vs 6.1 TFLOP/s), where the operands no longer stay cached
([benchmarks.md](benchmarks.md#apple-m4-max-gpu--metal)).

**`constant` address space.** Metal's equivalent of CUDA constant memory:
read-only data that every thread reads at the same address. Apple's
guidance is to place small, uniformly read arguments there so the compiler
can preload them; the kernels pass the matrix dimensions this way.

**System-level cache and unified memory.** The GPU has no dedicated memory:
it shares LPDDR5X and the SLC with the CPU clusters. Buffers here use
shared storage, so no copy separates the CPU and GPU views. The GPU and CPU
also compete for SLC capacity and DRAM bandwidth. There is no counterpart of
NVIDIA's thread block clusters or distributed shared memory: threadgroups
cannot access each other's threadgroup memory.

### 9.3 CPU and GPU terms

| Concept | CPU | CUDA | Metal |
|---|---|---|---|
| Fastest storage | vector registers | registers | registers |
| Software-managed on-chip memory | — | shared memory | threadgroup memory |
| First hardware cache | L1d | L1 / texture | GPU L1 (undocumented) |
| Shared hardware cache | L2 / L3 / SLC | L2 | GPU L2, SLC |
| Transfer unit | 64 B / 128 B line | 32 B sector of a 128 B line | not documented |
| Unit of access | one core | warp (32 threads) | SIMD-group (32 threads) |

---

## 10. Summary

- Performance is decided by how many bytes each level delivers per useful
  FLOP. Loop order fixes spatial locality; tiling fixes temporal locality.
- Use whole cache lines: stride-1 inner loops on CPUs, coalesced warp
  accesses on GPUs.
- Size tiles to a level that is actually close. On the M4 that is the 16 MiB
  L2: the SLC is three times farther than Zen 5's L3 and shared with the
  rest of the SoC.
- On GPUs, reuse from registers outweighs occupancy, and software staging
  in shared or threadgroup memory pays off only when the hardware caches
  cannot hold the reuse.

Kernel-level detail: [kernel-design.md](kernel-design.md),
[kernels/](kernels/README.md); measured results:
[benchmarks.md](benchmarks.md).
