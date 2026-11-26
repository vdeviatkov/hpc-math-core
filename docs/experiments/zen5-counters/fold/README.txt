B folding in the AVX-512 micro-kernel, GCC 13.3 vs Clang 18.1.3, Zen 5, 2026-11-25.

inst.cpp: explicit instantiations of vector_micro<Zmm<float>, R, V>, compiled with
-std=c++20 -O3 -march=native -ffast-math -funroll-loops; FMAs counted in objdump:
"B from memory" = full-vector memory operand, "A bcast" = {1to16} operand.

            GCC 13: B-from-mem / FMAs      Clang 18: B-from-mem / FMAs
  4x2       24 / 24                        0 / 8
  2x4       56 / 56                        0 / 8
  4x4       48 / 48                        0 / 16
  6x2 8x2 6x3 8x3 12x2 14x2   0 (all)      0 (all)
  16x1      0 (16 A broadcasts)            0 (16 A broadcasts)

Hot loops (perf annotate, bench_gemm Avx512Tile4x2/f32/N=256, GCC build): the
hottest FMAs are vfmadd231ps (%rdx) / 0x40(%rdx): B from memory in the running loop.

gcc.csv / clang.csv: bench_gemm Avx512Tile rows, taskset -c 2, median of 3 interleaved,
GFLOP/s:
  shape   N=256 GCC  Clang     N=1024 GCC  Clang
  4x2     219.4      287.0     141.2       158.4
  4x4     254.7      303.2     229.4       233.4
  2x4     165.1      170.5     109.5       117.8
  8x2     328.8      327.2     282.6       292.7
  8x3     325.9      328.4     312.3       313.3
