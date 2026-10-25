# NumPy, PyTorch and JAX on the Apple M4 Max

Recorded data from a NumPy / PyTorch / JAX GEMM comparison.
Apple M4 Max (12 P-cores in two clusters, 4 E-cores, 40-core GPU), macOS
26.3.1, Python 3.13.8, 2026-10-24, AC power. The benchmark script
(`bench_frameworks.py`) is not part of this repository. Triton and every
CUDA row report SKIPPED here: no NVIDIA GPU.

| File | Command | Contents |
|---|---|---|
| [`bench_frameworks.csv`](bench_frameworks.csv), [`.txt`](bench_frameworks.txt), [`_metadata.json`](bench_frameworks_metadata.json) | `bench_frameworks.py` (all parts) | CPU rows: NumPy 2.5.4, PyTorch 2.14.1, JAX 0.12.0.1; f32 / f64, N = 64 … 4096, 1 thread and all cores |
| [`session_torch_mps.csv`](session_torch_mps.csv), [`.txt`](session_torch_mps.txt) | `--only torch --parts gpu` | PyTorch MPS, f32 / fp16 / bf16, N = 256 … 8192 |
| [`session_jax_metal.csv`](session_jax_metal.csv), [`.txt`](session_jax_metal.txt) | `--only jax --parts gpu`, separate venv: jax / jaxlib 0.4.35 + jax-metal 0.1.1 | JAX on the Apple GPU, f32 |

## Edits to the recorded files

- `bench_frameworks.csv` keeps the CPU rows only. Its GPU rows (also in the
  `.txt` log) predate a fix to the accuracy check and are superseded by the
  session files. Removed with them: JAX's `threads=1` rows. XLA:CPU ignores
  every thread limit tried (`VECLIB_MAXIMUM_THREADS`, `OMP_NUM_THREADS`,
  `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`): with any of them a 4096
  f32 product keeps ~14 of 16 cores busy (process CPU time ÷ wall time) and
  runs at ~3.2 TFLOP/s. Those rows were all-core runs under a 1-thread label.
  The harness now reports them as SKIPPED.
- The session files are the PyTorch and JAX steps of one bracketed GPU
  session (Metal ladder → PyTorch MPS → JAX on Metal → Metal ladder again;
  [metal-m4/session_log.txt](../metal-m4/session_log.txt), 01:56–02:02).
  The harness's CSVs for those two steps were later overwritten by a rerun,
  stopped because the GPU was down to ~1 TFLOP/s (PyTorch bf16 at N=8192:
  1,081; JAX f32: 916–947). The session `.txt` logs were restored from the
  console output of the session, and the CSVs rebuilt from them, so they
  have no `ms_min` / `gflops_best` columns.
- JAX's fp16 and bf16 rows failed in that session: jax-metal cannot lower a
  dot with fp16 inputs and an fp32 result (`preferred_element_type`). The
  harness now keeps the input type on Metal, and a short check ran both, but
  they have no measurement from the session.

## CPU results

GFLOP/s, median of 5 measurements of ≥ 0.2 s. The load average was 26 at
the end of the run (an IDE was indexing), so the all-core rows in
particular are lower bounds.

| Library call | dtype | threads | N=64 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|--:|--:|--:|--:|--:|--:|
| NumPy `matmul` (Accelerate) | f32 | 1 | 354 | 1,432 | 1,686 | 1,673 | 1,738 | 1,656 |
| PyTorch `mm` (Accelerate) | f32 | 1 | 396 | 1,423 | 1,683 | 1,670 | 1,732 | 1,625 |
| NumPy `matmul` | f32 | all | 346 | 1,435 | 2,608 | 2,982 | 2,907 | 2,724 |
| PyTorch `mm` | f32 | all | 337 | 1,279 | 2,544 | 2,844 | 2,667 | 2,584 |
| JAX `jit(matmul)` (XLA:CPU) | f32 | all | 156 | 534 | 1,417 | 1,879 | 1,863 | 2,010 |
| NumPy `matmul` | f64 | 1 | 194 | 409 | 424 | 390 | 411 | 408 |
| PyTorch `mm` | f64 | 1 | 212 | 410 | 422 | 444 | 410 | 403 |
| NumPy `matmul` | f64 | all | 166 | 404 | 727 | 714 | 694 | 718 |
| PyTorch `mm` | f64 | all | 166 | 398 | 728 | 749 | 699 | 726 |
| JAX `jit(matmul)` | f64 | all | 22 | 84 | 192 | 223 | 212 | 177 |

- NumPy and PyTorch both call Accelerate (`numpy.show_config()`:
  `blas: accelerate`; `torch.__config__.show()`: `BLAS_INFO=accelerate`), and
  match each other and `gemm_amx` within the noise: 1-thread f32 ~1.7
  TFLOP/s, the single-core Accelerate figure in benchmarks.md.
- JAX f32 with all cores: 1.9–2.0 TFLOP/s in this run, ~3.2 in the
  mislabelled 1-thread run and in a standalone check. It keeps every core
  busy, E-cores included, so it is the most sensitive to other load. jaxlib
  links no Accelerate; its CPU GEMM is XLA's own (YNNPACK and oneDNN symbols
  are present). Reaching ~3.2 TFLOP/s implies both SME units, but which
  library issues the SME instructions was not determined.
- JAX f64 peaks at 223 GFLOP/s and falls to 177 at N=4096, about half of
  one SME unit: XLA's f64 dot does not get Accelerate's or the SME units'
  f64 rate.

## GPU results (bracketed session, throttled)

GFLOP/s. The Metal rows of the same session measured `MetalMps` f32 at
7,273 / 6,593 (before / after) at N=4096 and the FMA peak at 9.7 / 8.4
TFLOP/s, about two thirds of the published run, so compare within this
table only.

| Call | dtype | max rel. err (N=256) | N=256 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|--:|--:|--:|--:|--:|--:|
| PyTorch `mm`, MPS | f32 | 7.2e-07 | 2,308 | 8,250 | 5,610 | 4,944 | 4,036 |
| PyTorch `mm`, MPS | fp16 | 3.9e-04 | 1,824 | 5,888 | 6,783 | 6,779 | 4,792 |
| PyTorch `mm`, MPS | bf16 | 2.9e-03 | 1,753 | 5,348 | 5,985 | 5,595 | 4,658 |
| JAX `jit(matmul)`, jax-metal | f32 | 7.2e-07 | 1,107 | 12,049 | 4,396 | 4,366 | 3,620 |
| `MetalMps` (MPSMatrixMultiplication), before / after | f32 | — | 3,515 / 2,614 | 7,402 / 6,516 | 5,548 / 5,662 | 7,273 / 6,593 | 4,799 / 4,636 |
| `MetalSimdTg`, before / after | f32 | — | 1,101 / 935 | 5,486 / 5,240 | 5,801 / 6,802 | 5,628 / 4,059 | 2,979 / 3,475 |

The fp16 / bf16 rows round the inputs and, on MPS, the result to 16 bits.
Every f32 row, CPU and GPU, measured the same 7.2e-07.
