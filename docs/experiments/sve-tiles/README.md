# SVE register-tile validation, 2026-10-23

The [SVE kernel and sweep guide](../../kernels/sve.md) exposes register rows R,
vectors per row V, cache blocks and prefetch distance as template parameters.
This directory records **correctness validation**, not hardware throughput.

| SVE length | Compiler / runtime | Tests | Result |
|---|---|---:|---|
| 128 bits | GCC 13.3.0 / QEMU 8.2.2 | 20 | pass |
| 256 bits | same | 20 | pass |
| 384 bits | same | 20 | pass |
| 512 bits | same | 20 | pass |
| 1024 bits | same | 20 | pass |
| 2048 bits | same | 20 | pass |

Raw output: `qemu-vl<bits>.txt`. The single scalable binary was run at all
six lengths, including the non-power-of-two 384-bit case. The main function
sets VL with Linux `PR_SVE_SET_VL` and checks `svcntb()` before starting the
tests. The 20 tests cover both precisions, all 11 registered register shapes,
normal/empty/rectangular products, partial row tiles, split K accumulation,
a non-aligned 5x7x13 cache block with R=6/V=4, and prefetch distances
0/2/4/8/16 at vector-width-dependent column boundaries.

The tests compare against `gemm_naive` on deterministic finite random data,
with C prefilled to check overwrite semantics. They run the same kernel
implementation used by the benchmarks. The emulator checks execution and
numerical results; it does not predict a physical CPU's pipeline, cache
behavior, throughput, tile optimum or performance regression.

Additional checks on the Apple host:

- Apple Clang 17.0.0 generated SVE code for both precisions and every sweep
  shape, as well as the complete SVE-enabled `bench_gemm.cpp` object.
- The native CPU suite passed 28 tests; SVE cases are absent on this host.
- All 696 SVE blocked/tile/cache/prefetch benchmark rows reported SKIPPED
  on the non-SVE host, with zero measured iterations.

## Reproduce the emulator check

The cross-compiler and emulator came from Ubuntu 24.04 packages.
`validation.json` records compiler
flags, emulator version, test filter and hashes of the tested source files.
`sve_test_main.cpp` is the VL-setting test entry point; use it instead of
Google Test's usual main.

With an installed AArch64 Linux cross-toolchain, QEMU user emulator and
Google Test source tree:

```bash
gtest_src=build/_deps/googletest-src/googletest
aarch64-linux-gnu-g++ -std=c++20 -O3 -ffast-math -funroll-loops \
  -fno-omit-frame-pointer -march=armv8-a+sve -static -pthread \
  -Iinclude -Isrc -I"$gtest_src/include" -I"$gtest_src" \
  tests/test_gemm.cpp "$gtest_src/src/gtest-all.cc" \
  docs/experiments/sve-tiles/sve_test_main.cpp -o sve-tests

for bits in 128 256 384 512 1024 2048; do
  SVE_TEST_BITS=$bits qemu-aarch64 -cpu max,sve-max-vq=16 ./sve-tests \
    --gtest_filter='*SveBlocked*:*SvePrefetch*:*GemmSve*' --gtest_brief=1
done
```

Set `gtest_src` to the actual Google Test source directory if it differs
from the default FetchContent location above. Physical performance results remain pending;
use the sweep commands in the kernel guide on an SVE CPU and retain its
actual VL, compiler flags, machine metadata and raw benchmark JSON.
