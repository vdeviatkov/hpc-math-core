#pragma once

/**
 * @file cuda_variants.hpp
 * @brief The tile variants bench_gemm_cuda and bench_gemm_lowp_cuda sweep:
 *        CudaParams (cuda_params.hpp) with one level's sizes changed.
 *
 * Each list is an X-macro: X(name, type). benchmarks/cuda_instances.cpp
 * instantiates the entry points each variant needs; the benchmarks register
 * one row per variant (<Level>Var/<name>/...). Every variant satisfies its
 * level's constraints (gemm/cuda.hpp, "Tile parameters"): static shared
 * memory ≤ 48 KB for Levels 3 and 7, BM == BN and whole 128-bit vectors for
 * Level 4.
 */

#include "cuda_params.hpp"

namespace hpc::bench::cuda_variants {

// Levels 0-1: Tile x Tile threads per block.
using TL_8  = WithTile<8>;
using TL_16 = WithTile<16>;  // the default
using TL_32 = WithTile<32>;
#define HPC_CUDA_TILE_VARIANTS(X) X(8, TL_8) X(16, TL_16) X(32, TL_32)

// Levels 2-4 in double: the register tile (Levels 2 and 4) and the
// double-buffered tile (Level 3); two 32-deep double tiles of 128 x 128
// exceed 48 KB, so the k steps are 8 and 16.
using F64_128_k16_8x8 = WithF64Tiles<RegTileOf<128, 128, 16, 8, 8>, DoubleBufOf<64, 64, 16, 8, 8, 1>>;  // the default
using F64_128_k8_8x8  = WithF64Tiles<RegTileOf<128, 128, 8, 8, 8>, DoubleBufOf<128, 128, 8, 8, 8, 1>>;
using F64_128_k16_8x4 = WithF64Tiles<RegTileOf<128, 128, 16, 8, 4>, DoubleBufOf<64, 64, 16, 8, 4, 1>>;
using F64_64_k16_8x8  = WithF64Tiles<RegTileOf<64, 64, 16, 8, 8>, DoubleBufOf<64, 64, 8, 8, 8, 1>>;
using F64_64_k16_4x4  = WithF64Tiles<RegTileOf<64, 64, 16, 4, 4>, DoubleBufOf<64, 64, 16, 4, 4, 2>>;
#define HPC_CUDA_F64_VARIANTS(X)                                                                       \
    X(128x128_k16_8x8__db64x64_k16_8x8, F64_128_k16_8x8) X(128x128_k8_8x8__db128x128_k8_8x8, F64_128_k8_8x8) \
    X(128x128_k16_8x4__db64x64_k16_8x4, F64_128_k16_8x4) X(64x64_k16_8x8__db64x64_k8_8x8, F64_64_k16_8x8)    \
    X(64x64_k16_4x4__db64x64_k16_4x4_mb2, F64_64_k16_4x4)

// Levels 2 and 4 (float): block tile, k step, thread tile.
using RT_128x128_k16_8x8  = WithRegTile<RegTileOf<128, 128, 16, 8, 8>>;  // the default
using RT_128x128_k16_8x4  = WithRegTile<RegTileOf<128, 128, 16, 8, 4>>;
using RT_128x128_k16_4x8  = WithRegTile<RegTileOf<128, 128, 16, 4, 8>>;
using RT_128x128_k16_16x8 = WithRegTile<RegTileOf<128, 128, 16, 16, 8>>;
using RT_64x64_k16_8x8    = WithRegTile<RegTileOf<64, 64, 16, 8, 8>>;
using RT_64x64_k16_4x4    = WithRegTile<RegTileOf<64, 64, 16, 4, 4>>;
using RT_128x128_k8_8x8   = WithRegTile<RegTileOf<128, 128, 8, 8, 8>>;
using RT_128x128_k32_8x8  = WithRegTile<RegTileOf<128, 128, 32, 8, 8>>;
#define HPC_CUDA_REGTILE_VARIANTS(X)                                                                   \
    X(128x128_k16_8x8, RT_128x128_k16_8x8) X(128x128_k16_8x4, RT_128x128_k16_8x4)                      \
    X(128x128_k16_4x8, RT_128x128_k16_4x8) X(128x128_k16_16x8, RT_128x128_k16_16x8)                    \
    X(64x64_k16_8x8, RT_64x64_k16_8x8) X(64x64_k16_4x4, RT_64x64_k16_4x4)                              \
    X(128x128_k8_8x8, RT_128x128_k8_8x8) X(128x128_k32_8x8, RT_128x128_k32_8x8)

// Level 3 (float): tile, k step, thread tile, __launch_bounds__ minimum blocks.
using DB_128_k16_8x8_mb2 = WithDoubleBufF32<DoubleBufOf<128, 128, 16, 8, 8, 2>>;  // the default
using DB_128_k16_8x8_mb1 = WithDoubleBufF32<DoubleBufOf<128, 128, 16, 8, 8, 1>>;
using DB_128_k16_8x4_mb1 = WithDoubleBufF32<DoubleBufOf<128, 128, 16, 8, 4, 1>>;
using DB_64_k16_8x8_mb8  = WithDoubleBufF32<DoubleBufOf<64, 64, 16, 8, 8, 8>>;
using DB_64_k16_4x4_mb4  = WithDoubleBufF32<DoubleBufOf<64, 64, 16, 4, 4, 4>>;
using DB_128_k8_8x8_mb2  = WithDoubleBufF32<DoubleBufOf<128, 128, 8, 8, 8, 2>>;
#define HPC_CUDA_DOUBLEBUF_VARIANTS(X)                                                                 \
    X(128x128_k16_8x8_mb2, DB_128_k16_8x8_mb2) X(128x128_k16_8x8_mb1, DB_128_k16_8x8_mb1)              \
    X(128x128_k16_8x4_mb1, DB_128_k16_8x4_mb1) X(64x64_k16_8x8_mb8, DB_64_k16_8x8_mb8)                 \
    X(64x64_k16_4x4_mb4, DB_64_k16_4x4_mb4) X(128x128_k8_8x8_mb2, DB_128_k8_8x8_mb2)

// Levels 5-6: warps per block (WarpsM x WarpsN of 16 x 16) and minimum
// resident blocks, chosen so the SM's 48 warp slots can fill.
using WG_4x4 = WithWarpGrid<WarpGridOf<4, 4, 3>>;  // the default
using WG_2x2 = WithWarpGrid<WarpGridOf<2, 2, 12>>;
using WG_2x4 = WithWarpGrid<WarpGridOf<2, 4, 6>>;
using WG_4x2 = WithWarpGrid<WarpGridOf<4, 2, 6>>;
using WG_2x8 = WithWarpGrid<WarpGridOf<2, 8, 3>>;
using WG_8x2 = WithWarpGrid<WarpGridOf<8, 2, 3>>;
using WG_4x8 = WithWarpGrid<WarpGridOf<4, 8, 1>>;
using WG_8x4 = WithWarpGrid<WarpGridOf<8, 4, 1>>;
#define HPC_CUDA_WARPGRID_VARIANTS(X)                                                                  \
    X(4x4, WG_4x4) X(2x2, WG_2x2) X(2x4, WG_2x4) X(4x2, WG_4x2) X(2x8, WG_2x8) X(8x2, WG_8x2)          \
    X(4x8, WG_4x8) X(8x4, WG_8x4)

// Level 7: block tile, k step, warp tile.
using PP_128x128_k32_w32x64 = WithPipe<PipeOf<128, 128, 32, 32, 64>>;  // the default
using PP_128x128_k32_w64x32 = WithPipe<PipeOf<128, 128, 32, 64, 32>>;
using PP_128x128_k32_w64x64 = WithPipe<PipeOf<128, 128, 32, 64, 64>>;
using PP_128x128_k32_w32x32 = WithPipe<PipeOf<128, 128, 32, 32, 32>>;
using PP_128x128_k16_w32x64 = WithPipe<PipeOf<128, 128, 16, 32, 64>>;
using PP_64x128_k32_w32x64  = WithPipe<PipeOf<64, 128, 32, 32, 64>>;
using PP_128x64_k32_w32x32  = WithPipe<PipeOf<128, 64, 32, 32, 32>>;
using PP_64x64_k32_w32x32   = WithPipe<PipeOf<64, 64, 32, 32, 32>>;
#define HPC_CUDA_PIPE_VARIANTS(X)                                                                      \
    X(128x128_k32_w32x64, PP_128x128_k32_w32x64) X(128x128_k32_w64x32, PP_128x128_k32_w64x32)          \
    X(128x128_k32_w64x64, PP_128x128_k32_w64x64) X(128x128_k32_w32x32, PP_128x128_k32_w32x32)          \
    X(128x128_k16_w32x64, PP_128x128_k16_w32x64) X(64x128_k32_w32x64, PP_64x128_k32_w32x64)            \
    X(128x64_k32_w32x32, PP_128x64_k32_w32x32) X(64x64_k32_w32x32, PP_64x64_k32_w32x32)

// Low-precision cp.async kernel: block tile, warp tile (WarpM rows, WarpsN
// warps across), stages, block-order group height.
using LP_128x128_w64x32_g8  = WithLowp<LowpOf<128, 128, 64, 4, 2, 8>>;  // the default
using LP_128x128_w32x32_g8  = WithLowp<LowpOf<128, 128, 32, 4, 2, 8>>;
using LP_128x128_w64x64_g8  = WithLowp<LowpOf<128, 128, 64, 2, 2, 8>>;
using LP_64x128_w64x32_g8   = WithLowp<LowpOf<64, 128, 64, 4, 2, 8>>;
using LP_128x256_w64x64_g8  = WithLowp<LowpOf<128, 256, 64, 4, 2, 8>>;
using LP_128x128_w64x32_g4  = WithLowp<LowpOf<128, 128, 64, 4, 2, 4>>;
using LP_128x128_w64x32_g16 = WithLowp<LowpOf<128, 128, 64, 4, 2, 16>>;
#define HPC_CUDA_LOWP_VARIANTS(X)                                                                      \
    X(128x128_w64x32_g8, LP_128x128_w64x32_g8) X(128x128_w32x32_g8, LP_128x128_w32x32_g8)              \
    X(128x128_w64x64_g8, LP_128x128_w64x64_g8) X(64x128_w64x32_g8, LP_64x128_w64x32_g8)                \
    X(128x256_w64x64_g8, LP_128x256_w64x64_g8) X(128x128_w64x32_g4, LP_128x128_w64x32_g4)              \
    X(128x128_w64x32_g16, LP_128x128_w64x32_g16)

// Level 8: block rows, stages wide / narrow, group height, forced tile
// (0 = by waves), wide margin in percent.
using TP_128_auto_g8   = WithTma<TmaOf<128, 4, 6, 8, 0, 5>>;  // the default
using TP_128_wide_g8   = WithTma<TmaOf<128, 4, 6, 8, 256, 5>>;
using TP_128_narrow_g8 = WithTma<TmaOf<128, 4, 6, 8, 128, 5>>;
using TP_64_auto_g8    = WithTma<TmaOf<64, 4, 6, 8, 0, 5>>;
using TP_128_auto_g4   = WithTma<TmaOf<128, 4, 6, 4, 0, 5>>;
using TP_128_auto_g16  = WithTma<TmaOf<128, 4, 6, 16, 0, 5>>;
#define HPC_CUDA_TMA_VARIANTS(X)                                                                       \
    X(128_auto_g8, TP_128_auto_g8) X(128_wide_g8, TP_128_wide_g8) X(128_narrow_g8, TP_128_narrow_g8)   \
    X(64_auto_g8, TP_64_auto_g8) X(128_auto_g4, TP_128_auto_g4) X(128_auto_g16, TP_128_auto_g16)

}  // namespace hpc::bench::cuda_variants
