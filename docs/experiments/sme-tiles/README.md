# SME ZA shape and cache-block sweeps

Apple M4 Max, macOS, one calling thread, Apple Clang 17.0.0,
`-O3 -mcpu=apple-m4 -ffast-math -funroll-loops -fno-omit-frame-pointer`.
Measured 2026-10-23 local time. Streaming vector length 512 bits (16 float
or 8 double elements). Each JSON includes date and system-load context;
the reported 24 MHz CPU field is the platform timer frequency, not a
measured core clock, and is not used to derive a throughput peak.

"Default" below means the shapes and cache blocks that were the template
defaults when these runs were made (2×2 FP32 / 2×4 FP64; 1024×2048×4096 /
512×2048×4096). The kernels no longer have defaults; the same values are
now `SmeParams<T>` in `benchmarks/sme_params.hpp`.

- `tiles_m4.json`: all eight FP32 and twenty FP64 legal positive rectangular
  ZA shapes at N=64/256/512/1024/4096, three randomly interleaved repetitions,
  minimum measured time 0.02 s per repetition. Default cache blocks held fixed.
- `tiles_m4_stage2.json`: 1×4, 2×2 and (FP64) 2×4 at
  N=256/1024/4096, seven randomly interleaved repetitions, minimum time
  0.1 s; confirmation of the leading candidate against defaults.
- `blocks_m4.json`: default ZA shapes, M={256,512,1024},
  K/N={1024,2048,4096}, N=512/1024/4096, both precisions; three randomly
  interleaved repetitions, minimum time 0.02 s.
- `before_1.json` … `before_3.json` and `after_1.json` … `after_3.json`:
  pre-change versus configurable implementation at unchanged defaults,
  N=256/1024/4096, alternated in three rounds. Each round has one randomly
  interleaved repetition, minimum time 0.1 s. The baseline binary was built
  before modifying `sme.hpp`, with the same compiler/build settings.
- `tests.txt`, `lowp-tests.txt`: all 32 regular and 12 low-precision tests
  pass on actual SME2 hardware. New shape tests exercise every legal shape
  through packed-B, unpacked-B and unblocked entry points, plus 5×7×13
  cache blocks, partial tiles, empty dimensions and K accumulation.
- `validation.json`: source hashes, compiler and benchmark flags.

The binaries run sequentially; no concurrent benchmark or build is part of
these retained runs. Matrix allocation/packing inside each GEMM is timed;
input allocation/initialization in the benchmark harness is excluded.
The desktop has background work; medians and variability are included in
the JSON. These are measurements of these implementations, not a proof of
optimality across all SME CPUs. In these runs, nondefault shapes used
expanded single-vector loads and only the defaults the paired SME2 loads;
every shape now uses paired loads ([SME guide](../../kernels/sme.md)).

Reproduction commands and API: [SME guide](../../kernels/sme.md).
Historical default-selection data remain in
[sme2-gemm](../sme2-gemm/README.md); this experiment does not overwrite them.

## Default performance preservation

Median GFLOP/s across the three alternated rounds (whole GEMM, including
packing). The changes range from −1.03% to +1.40%, consistent with the
run-to-run variation on this desktop; this check finds no material default
throughput regression. It is not a guarantee for every shape or machine.

| Type | N | Before | After | Change |
|---|--:|--:|--:|--:|
| f32 | 256 | 768.5 | 771.9 | +0.44% |
| f32 | 1024 | 1444.8 | 1430.8 | -0.97% |
| f32 | 4096 | 1540.3 | 1524.5 | -1.03% |
| f64 | 256 | 318.9 | 318.6 | -0.10% |
| f64 | 1024 | 421.0 | 426.9 | +1.40% |
| f64 | 4096 | 426.4 | 424.3 | -0.49% |

## Cache sweep

At N=4096, the existing defaults are the best measured cache configurations
in this grid for both types: FP32 1024×2048×4096 at 1538.2 GFLOP/s, FP64
512×2048×4096 at 435.2 GFLOP/s. This supports retaining those defaults on
this M4; the leading configurations differ at smaller sizes.

## Full register-shape sweep

Median GFLOP/s. Each table holds that type's default cache blocks fixed.
A tile's dimensions are counts of ZA tiles, not output elements.

### f32

| ZA shape | N=64 | 256 | 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|--:|
| 1x1 | 199.1 | 367.6 | 447.4 | 467.3 | 470.8 |
| 1x2 | 253.6 | 577.2 | 822.0 | 886.0 | 887.8 |
| 1x3 | 231.6 | 648.4 | 1095.3 | 1236.0 | 1231.0 |
| 1x4 | 298.7 | 784.6 | 1344.3 | 1603.6 | 1560.0 |
| 2x1 | 240.3 | 549.4 | 782.0 | 831.2 | 834.3 |
| 2x2 (default) | 291.1 | 772.2 | 1345.9 | 1468.0 | 1535.9 |
| 3x1 | 201.8 | 612.0 | 1020.3 | 1164.7 | 1108.2 |
| 4x1 | 238.5 | 642.2 | 1085.4 | 1265.7 | 1274.2 |

### f64

| ZA shape | N=64 | 256 | 512 | 1024 | 4096 |
|---|--:|--:|--:|--:|--:|
| 1x1 | 48.7 | 113.0 | 120.7 | 120.5 | 119.9 |
| 1x2 | 78.0 | 199.7 | 225.8 | 234.6 | 229.2 |
| 1x3 | 68.0 | 262.3 | 316.1 | 344.3 | 329.7 |
| 1x4 | 89.0 | 323.6 | 409.7 | 443.5 | 421.5 |
| 1x5 | 67.4 | 303.8 | 403.3 | 426.2 | 424.0 |
| 1x6 | 65.0 | 296.1 | 395.9 | 429.6 | 425.1 |
| 1x7 | 63.4 | 299.5 | 377.2 | 424.6 | 424.5 |
| 1x8 | 68.7 | 183.0 | 209.3 | 216.2 | 215.0 |
| 2x1 | 54.4 | 190.0 | 220.4 | 229.6 | 222.0 |
| 2x2 | 83.1 | 319.4 | 393.8 | 422.6 | 405.4 |
| 2x3 | 70.9 | 314.5 | 393.4 | 426.3 | 415.0 |
| 2x4 (default) | 88.8 | 318.3 | 404.0 | 430.9 | 429.9 |
| 3x1 | 53.2 | 245.9 | 297.2 | 316.1 | 293.8 |
| 3x2 | 72.0 | 311.7 | 383.5 | 417.5 | 389.4 |
| 4x1 | 74.2 | 272.1 | 336.7 | 343.8 | 349.3 |
| 4x2 | 85.7 | 310.3 | 381.3 | 412.5 | 402.9 |
| 5x1 | 51.2 | 260.6 | 348.4 | 364.1 | 352.0 |
| 6x1 | 47.5 | 257.3 | 349.2 | 361.6 | 350.5 |
| 7x1 | 44.6 | 261.1 | 328.3 | 340.7 | 341.2 |
| 8x1 | 56.7 | 270.8 | 334.4 | 361.7 | 362.6 |


## Repeated candidate comparison

Seven interleaved repetitions in one session. Median GFLOP/s; percentage
compares 1×4 to the unchanged default (2×2 FP32, 2×4 FP64).

| Type | N | Default | 1×4 | Change |
|---|--:|--:|--:|--:|
| f32 | 256 | 770.3 | 785.1 | +1.92% |
| f32 | 1024 | 1452.2 | 1589.9 | +9.48% |
| f32 | 4096 | 1547.3 | 1578.4 | +2.01% |
| f64 | 256 | 320.1 | 324.6 | +1.42% |
| f64 | 1024 | 422.7 | 441.0 | +4.33% |
| f64 | 4096 | 430.9 | 425.1 | -1.36% |

The FP32 1×4 shape is a promising alternative, particularly at N=1024
(+9.48%). FP64 1×4 helps at N=1024 (+4.33%) but loses at N=4096 (−1.36%).
`SmeParams` keeps the 2×2 / 2×4 shapes; these square-size results do not
establish a universal shape for rectangular workloads or other machines.
Selecting `gemm_sme<float, 1, 4, Mc, Kc, Nc>` requires no micro-kernel
rewrite.
The old balanced/all-tile layout was a sensible starting design, not an
exhaustively established shape optimum. The shape sweep makes that
limitation explicit.
