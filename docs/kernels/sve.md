# SVE: configurable register and cache tiles

The blocked SVE kernel exposes the same R-row by V-vector register shape as
the fixed-width SIMD kernels. R and V are compile-time parameters; each
vector still has the executing thread's runtime SVE length. Both float and
double use the same algorithm.

```cpp
// Defaults: R=4, V=2, TileM=64, TileK=256, TileN=512.
gemm_sve_blocked<float>(A, B, C);
gemm_sve_blocked<float, 6, 4>(A, B, C);
gemm_sve_blocked<double, 8, 3, 128, 256, 512>(A64, B64, C64);

// Existing prefetch argument order is preserved: T, PfDist, then R/V/cache.
gemm_sve_blocked_prefetch<float, 4, 6, 4>(A, B, C);
gemm_sve_blocked_prefetch<float, 0, 6, 4>(A, B, C); // same shape without hints
```

Changing a tile does not require editing the micro-kernel. `sve_micro<R,V>`
is also available for isolated full-tile experiments; its arguments are
`a, b, c, lda, ldb, ldc, k_len`, plus an optional actual column count for a
predicated edge tile. The output is accumulated into C. Matrix entry points
instead overwrite C by zeroing it before the blocked traversal.

The 4×2 / 64×256×512 defaults are starting values retained for compatibility,
not sizes selected by a hardware performance sweep. See the
[cross-family sizing rationale](sizing.md).

## How it works

A register tile owns R*V scalable accumulators. Each K step loads V vectors
of B once, broadcasts one A element per row and performs R*V predicated
FMAs. All accumulator loads and stores are outside the K loop. More rows
reuse each B load more; more vectors reuse each A broadcast more. Both
increase register pressure. SVE has 32 Z registers, so a starting budget is
R*V accumulators + V B vectors + one A broadcast; codegen may need extra
registers, and exceeding this budget can spill. A larger tile is not
necessarily faster.

SVE ACLE types are sizeless and cannot be ordinary array elements or class
members ([Arm ACLE](https://arm-software.github.io/acle/main/acle.html)).
The implementation uses compile-time index/parameter packs instead of an
array of vectors. These expand into individual local accumulators and
unrolled row/column operations; they do not introduce runtime indexing or
an allocated accumulator buffer. Both GCC and Clang code generation are
checked during validation.

Column predicates handle partial vectors and entirely inactive vectors in
a wide tile. Their pointers are clamped to a valid row-end address. Partial
row tiles use smaller R specializations, including when TileM is smaller
than R or R does not divide TileM. Non-vector-aligned TileN and split K blocks
are supported. Template register/cache dimensions must be positive.

## Sweep and publish measurements

Run on an AArch64 machine with non-streaming SVE enabled by the compiler
(e.g. a native Release build on SVE hardware). Apple M4 SME does not supply
non-streaming SVE; x86 and Apple builds register the rows as skipped.

```bash
cmake -S . -B build-sve -DCMAKE_BUILD_TYPE=Release
cmake --build build-sve --target hpc_tests bench_gemm --parallel
./build-sve/tests/hpc_tests --gtest_filter='*Sve*'

# Register tiles, holding the 64x256x512 cache block fixed.
./build-sve/benchmarks/bench_gemm \
  --benchmark_filter='^SveTile[0-9]+x[0-9]+/' \
  --benchmark_repetitions=7 --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=sve-tiles.json --benchmark_out_format=json

# Prefetch distances on 4x2, 6x4 and 2x8 register tiles.
./build-sve/benchmarks/bench_gemm --benchmark_filter='^SveTile.*Pf' \
  --benchmark_repetitions=7 --benchmark_enable_random_interleaving=true \
  --benchmark_out=sve-prefetch.json --benchmark_out_format=json

# Cache blocks, holding the 4x2 register tile fixed.
./build-sve/benchmarks/bench_gemm --benchmark_filter='^SveBlock[0-9]+x[0-9]+x[0-9]+/' \
  --benchmark_repetitions=7 --benchmark_enable_random_interleaving=true \
  --benchmark_out=sve-blocks.json --benchmark_out_format=json
```

Registered register shapes: 1x1, 4x2, 2x4, 4x4, 6x3, 8x2, 8x3, 6x4,
12x2, 2x8 and 16x1, at N=64/256/512/1024/4096 in both precisions.
The cache grid is M={16,32,64,128,256}, K/N={128,256,512,1024}, at
N=512/1024/4096. Prefetch sweeps use distances 2/4/8/16 and N=256/512/1024.

Each new sweep row records `mr` (R), `vectors` (V), `accumulators` (R*V),
`vl` (elements per vector), and `nr` (V*vl elements per register-tile row),
as well as GFLOP/s. Multiply vl by the element width to recover vector bits.
Record CPU, compiler, flags, selected vector length, load conditions and
raw JSON with a published table. Compare medians and variability within
the same session; do not select a new default from emulator speed.

## Results status

The default remains 4x2 because no physical SVE machine is available for
performance measurements. The sweep infrastructure is ready; no SVE
GFLOP/s table or best-tile claim is fabricated from compilation or emulation.
Correctness validation includes configurable shapes, prefetch distances,
non-aligned cache blocks, and runtime-vector-length-dependent tails.


GCC/QEMU execution passes 20 tests at each of 128, 256, 384, 512, 1024 and
2048 bits. Clang also generated the SVE-enabled benchmark object and all
register shapes. [Raw correctness results and reproduction](../experiments/sve-tiles/README.md)
are archived separately from any future physical performance measurements.
