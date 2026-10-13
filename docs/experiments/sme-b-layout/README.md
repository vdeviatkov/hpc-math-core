# Packed B vs B in place: where the time goes

FP32, M = N = K = 4096, single thread, Apple M4 Max. The SME blocked driver
(`sme_detail::blocked_driver`) with B read in place (the path of
`gemm_sme_blocked`) and packed (the path of `gemm_sme`), for B stored with
three leading dimensions. The power-of-two `ldb = 4096` is the case with the
large slowdown; `4128` and `4160` pad each row by 128 and 256 bytes.

Driver: [`benchmarks/experiments/sme_b_layout.cpp`](../../../benchmarks/experiments/sme_b_layout.cpp).
Raw samples: [`timings.csv`](timings.csv). Machine, compiler and source
hashes: [`metadata.json`](metadata.json).

## Method

Each of 15 repetitions runs all six cases twice, in an order shuffled per
repetition:

- **End to end:** the driver itself, the mean of 3 calls: scratch
  allocation, packing A (and B) and the macro-kernel, as the kernels run.
- **By phase:** a copy of the driver's loop nest with a clock around each
  phase, one call. Diagnostic only; the clocks add overhead.

All 12 paths are checked against the packed driver before timing. Every B
buffer started on a 128-byte boundary. The inputs are created once and stay
warm in cache between calls; no prepacked B is reused across calls.

## Results

Medians over 15 repetitions, milliseconds; interquartile range in brackets.

| ldb | B | End to end | GFLOP/s | Pack A | Pack B | Compute |
|---:|:--|--:|--:|--:|--:|--:|
| 4096 | in place | 238.1 [232.3–240.9] | 577 | 2.95 | — | 230.8 [228.4–236.6] |
| 4096 | packed | 104.7 [103.5–105.5] | 1,313 | 2.96 | 4.81 | 96.9 [95.9–98.7] |
| 4128 | in place | 114.1 [109.6–117.4] | 1,204 | 2.94 | — | 110.0 [105.3–112.5] |
| 4128 | packed | 102.0 [100.8–103.6] | 1,348 | 2.96 | 2.07 | 96.0 [95.2–97.7] |
| 4160 | in place | 116.6 [109.9–117.6] | 1,179 | 2.93 | — | 107.2 [105.7–110.9] |
| 4160 | packed | 101.4 [100.8–102.5] | 1,355 | 2.98 | 2.03 | 97.0 [95.7–98.8] |

Scratch allocation is under 0.1 ms in every case.

- **Packing B pays for itself even with a padded stride.** At `ldb = 4128`,
  packing costs 2.1 ms and makes the macro-kernel 14 ms faster (110.0 →
  96.0 ms); end to end, packed B takes 10.6% less time. It won all 15
  repetitions, as it did at `ldb = 4160`.
- **The compute time with packed B does not depend on `ldb`** (96–97 ms):
  the macro-kernel only ever sees the 128-byte packed row stride.
- **At `ldb = 4096` the loss is almost all compute** (231 vs 97 ms). Packing
  from that stride is itself slower (4.8 vs 2.1 ms) but still small.

## Caveats

One run on one machine with a load average of 5.5–6.2 from other work. The
phase clocks run around every `pack_a`, `pack_b` and macro-kernel call (128
macro-kernel calls per driver call), so the phase figures carry some timing
overhead and are best compared with each other. The data says where the time goes, not
why the in-place stream is slower; see the article for the candidates.

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_SME=ON
cmake --build build --target sme_b_layout
./build/benchmarks/sme_b_layout 4096 15 3 > timings.csv   # summary table on stderr
```
