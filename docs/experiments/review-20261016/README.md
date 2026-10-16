# Review validation data, 2026-10-16

These are before/after comparisons for the [repository review](../../review-20261016.md),
not replacements for the historical article measurements. A speedup is
`before duration / after duration`; 1 is unchanged. No benchmark data was
edited to remove unfavorable samples.

## Linux CPU and CUDA

`linux-cuda-comparison.csv`: medians and coefficients of variation from
Google Benchmark. 48 CPU cases at N=256/1024, three repetitions; 51 GPU cases
at N=256/4096/8192, five repetitions; minimum time 0.05 seconds per repetition.
CPU uses calling-thread CPU time, GPU uses manual CUDA-event time. CV is a
fraction, not a percentage. Remote machine: AMD Ryzen 9 9950X, GCC 13.3,
RTX 5080, driver 615.71.09, CUDA 13.2, native sm_120, Release/native CPU flags.

Before source is commit `7a9adcee60bf0639410936b3d6ff17c9d65099e5`.
Candidate is the reviewed source; per-file hashes are in `source-hashes.json`.
Saved binaries and full Google Benchmark JSON/logs are under
`/tmp/hpc-review-20261016/{before,after}` on cuda.0. Its existing checkout
was not changed. Native tests pass all 36 cases. The pre-sm_80 compatibility
check used a separate `build-ptx75` directory and `75-virtual` code.

Representative commands, run on saved before and candidate binaries:

```bash
./bench_gemm_cuda --benchmark_filter='/N=(256|4096|8192)/' \
  --benchmark_min_time=0.05s --benchmark_repetitions=5 --benchmark_out=cuda.json
./bench_gemm --benchmark_filter='^(Naive|Reordered|Blocked|Avx2Naive|Avx2Reordered|Avx2Blocked|Avx512Naive|Avx512Reordered|Avx512Blocked|BlockedPf4|Avx2BlockedPf4|Avx512BlockedPf4)/(f32|f64)/N=(256|1024)$' \
  --benchmark_min_time=0.05s --benchmark_repetitions=3 --benchmark_out=cpu.json
compute-sanitizer --tool memcheck --error-exitcode=1 ./hpc_tests_cuda
compute-sanitizer --tool racecheck --error-exitcode=1 ./hpc_tests_cuda \
  --gtest_filter='*DoubleBuf:*WmmaPipelined'
```

The 26 custom CUDA rows at N=4096/8192 have speedup ratios 0.9964–1.0045
(median 0.9999). The tiny N=256 cuBLAS FP16 case showed a larger apparent
change between the distant full sweeps: 5.248 → 5.629 microseconds. A proposed
per-device handle cache was removed to preserve the original lightweight,
single-host-thread/fixed-device contract.

`vendor-abba.csv` checks the final code immediately before/after/after/before,
15 repetitions per run. Before medians are 5.608 and 5.507 microseconds;
after medians are 5.646 and 5.592. Neighboring comparisons differ by roughly
0.04–0.08 microseconds (0.7–1.5%), within the variability of the before runs
(CV 2.9–3.4%). The larger distant-sweep gap is not reproduced. An idle gap
from host submission can fall inside the event interval, so these timings
cannot establish a strict zero-overhead result for tiny launches. All raw
JSONs remain under the remote `after` directory, including the superseded
cache experiment. Large-kernel conclusions use the final custom-kernel run;
the vendor rows in the comparison CSV use `vendor-final.json` after removing
the cache.

## Apple CPU comparisons

Initial separate-process Google Benchmark sweeps included 69 cases from
scalar, NEON, all four SME steps, Accelerate and KleidiAI, generally
N=256/1024/4096, with naive limited to 256. These runs showed large changes
in unchanged vendor paths as the desktop load changed; they cannot support
precise improvement claims. Their JSON remains in
`/tmp/hpc-review-local-before/bench.json` and `/tmp/hpc-review-local-after.json`.

`mac-paired.csv` is a follow-up using the same process and randomly ordered
before/after pairs, nine repetitions. `paired_harness.cpp` is compiled twice
from header snapshots, with `-Dhpc=hpc_before` / `-Dhpc=hpc_after` to isolate
symbols. Matrices are allocated and filled outside the measured calls.
Unlike the main fixed-N benchmark, shapes here are passed at runtime.
`paired_bench.py` requests user-interactive QoS to reduce efficiency-core
placement, sets `VECLIB_MAXIMUM_THREADS=1`, warms each case, and targets
roughly 20 ms batches. This is a scheduling request, not CPU affinity.
Timing uses process CPU time except Accelerate's wall time. It records both
the ratio of medians and the median of per-pair ratios; those statistics
need not agree on a busy machine.

`mac-same-buffer.csv` isolates FP64 SME transfers at N=1024: both versions
receive the very same 128-byte-aligned input/output buffers. It excludes
allocation/packing and reports 31 randomly ordered paired samples.
The inputs are zero; it is a layout/control-flow probe, not a numerical
correctness test. Test-suite random data remains the correctness check.

To reproduce, copy the three harness files to a temporary directory, place
before/after source snapshots in its `before` and `after` subdirectories,
then compile and run from that directory:

```bash
for phase in before after; do
  clang++ -std=c++20 -O3 -mcpu=apple-m4 -ffast-math -funroll-loops \
    -fno-omit-frame-pointer -DHPC_ACCELERATE_AVAILABLE -DACCELERATE_NEW_LAPACK \
    -Dhpc=hpc_$phase -I$phase/include -I$phase/src \
    -dynamiclib paired_harness.cpp -framework Accelerate -o $phase.dylib
done
VECLIB_MAXIMUM_THREADS=1 python3 paired_bench.py
python3 same_buffer.py
```

A loaded desktop and a matrix unit shared with other threads remain sources
of noise. Results here establish behavior on this machine under these
conditions; they do not prove zero performance loss for all shapes. SVE
has no physical hardware validation; MSVC was reviewed but not executed.
