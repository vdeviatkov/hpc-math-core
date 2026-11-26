# Low-precision tile and block sweeps, Apple M4 Max

`bench_gemm_lowp`'s `Neon*Tile`, `Neon*Block`, `SmeWideningTile` and
`SmeWideningBlock` rows ([source](../../../../benchmarks/bench_gemm_lowp.cpp)).
Apple M4 Max, one P-core / one SME unit, macOS, Apple Clang 17,
`-O3 -mcpu=apple-m4 -ffast-math -funroll-loops`. The desktop
had background load (load averages per run in the `*load.txt` files,
2–13); runs are randomly interleaved, so rows within one file compare
fairly. Every swept shape is checked against the reference in
`hpc_tests_lowp` (`LowpNeon.SweptTilesMatchNaive`, `LowpSme.*`).

| File | Rows | Repetitions |
|---|---|---|
| [`neon_tiles.json`](neon_tiles.json) | every NEON BF16 / INT8 MMLA shape (11) and FP16 FMLAL (3) / FP16-accumulate (5) shape, N = 256 … 4096 | 3, interleaved |
| [`sme_widening_tiles.json`](sme_widening_tiles.json) | SME widening 1 × 4, 2 × 2, 4 × 1, bf16 / fp16 / INT8 | 3, interleaved |
| [`confirm.json`](confirm.json) | the leading candidates against the runner-up shapes | 7, interleaved |
| [`blocks.json`](blocks.json) | blocks at the chosen tiles (Mc × k-block bytes × Nc for NEON; Mc × k groups × Nc for SME), N = 1024 and 4096 | 3, interleaved |
| [`blocks_confirm.json`](blocks_confirm.json) | each default block against the leading candidates | 7, interleaved |

## Results

| Kernel | Measured tile | Evidence (confirm.json, against the 4 × 4 / 2 × 2 tile) |
|---|---|---|
| NEON FP16 widening (FMLAL) | **4 × 6** | +8.2% to +9.8% at N = 256 … 4096 |
| NEON FP16 accumulate | **4 × 5** | +4.3% at N = 256, +8.6% to +10.0% from 1024; 4 × 6 within 1% |
| SME widening (bf16, fp16, INT8) | **1 × 4** | +0.8% to +6.1% over 2 × 2, all 12 cells |
| NEON BF16 (BFMMLA) | 4 × 4 | every shape 64–68 GFLOP/s, within variation |
| NEON INT8 (SMMLA) | 4 × 4 | 4 × 4 and 2 × 8 tie as the fastest |

Blocks (blocks_confirm.json, GFLOP/s or GOP/s at N = 1024 / 4096): every
default holds. SME widening at its default blocks is the best or within
0.7% (bf16 1,752 / 1,848 against at most 1,750 / 1,861). NEON: Nc = 1024
against the default 512 is within ±0.6% for BF16 and INT8 and +0.6% to
+2.2% for the FP16 kernels; the best FP16-accumulate block (256 × 2048 B
× 1024) leads by 2.0% / 3.7%, within the run's variation (cv ≤ 3.5%), so
the smaller default is kept.
