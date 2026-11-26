#pragma once

/**
 * @file cuda_params.hpp
 * @brief Measured tile parameters of the CUDA kernels (gemm/cuda.hpp,
 *        "Tile parameters"), and the variants bench_gemm_cuda sweeps.
 *
 * The kernels take every size from a parameter struct P; CudaParams holds
 * the values the benchmarks and tests use, measured on an RTX 5080 (sm_120,
 * 84 SMs). docs/kernels/cuda.md has the sweeps behind each one. They are the
 * best measured on that GPU, not universal optima.
 */

#include <type_traits>

namespace hpc::bench {

struct CudaParams {
    /// Levels 0-1: 16 x 16 threads per block.
    static constexpr int Tile = 16;

    /// Level 2. Float: 128 x 128 block tile, k step 32, 8 x 8 per thread
    /// (256 threads), 1.3–4.0% faster than k step 16 at N = 1024 … 8192.
    /// Double: 64 x 64, k step 16, 4 x 4 per thread, 10% faster than
    /// 128 x 128 / 8 x 8 at N = 4096, 1% at 1024.
    template <typename T>
    struct RegTile {
        static constexpr bool kF32 = sizeof(T) == 4;
        static constexpr int BM = kF32 ? 128 : 64, BN = BM, BK = kF32 ? 32 : 16, TM = kF32 ? 8 : 4, TN = TM;
    };

    /// Level 4. Float: 128 x 128, k step 16, 16 x 8 per thread (128
    /// threads), 6–9% faster than 8 x 8 at N ≥ 4096 and level at 1024.
    /// Double: 64 x 64, k step 16, 4 x 4 per thread, 7% faster than
    /// 128 x 128 / 8 x 8 at N = 4096.
    template <typename T>
    struct Vectorized {
        static constexpr bool kF32 = sizeof(T) == 4;
        static constexpr int BM = kF32 ? 128 : 64, BN = BM, BK = 16, TM = kF32 ? 16 : 4, TN = kF32 ? 8 : 4;
    };

    /// Level 3. Float: 128 x 128, k step 16, 8 x 8 per thread, (256, 2)
    /// launch bounds: the best at N ≥ 4096 of six variants (two blocks per
    /// SM cap the kernel at 128 registers; one block is 4–7% slower).
    /// Double: 64 x 64, k step 16, 4 x 4 per thread, two blocks: 1.5–2.3%
    /// faster than 8 x 8 (two 128 x 128 double buffers would exceed 48 KB).
    template <typename T>
    struct DoubleBuf {
        static constexpr bool kF32 = sizeof(T) == 4;
        static constexpr int BM = kF32 ? 128 : 64, BN = BM, BK = 16, TM = kF32 ? 8 : 4, TN = TM;
        static constexpr int MinBlocks = 2;
    };

    /// Levels 5-6: 2 x 8 warps of 16 x 16, a 32 x 128 tile, three
    /// 512-thread blocks per SM: 38–53% faster than 4 x 4 (Level 5) and
    /// 4–10% (Level 6) at N = 1024 … 8192.
    struct WarpGrid {
        static constexpr int WarpsM = 2, WarpsN = 8, MinBlocks = 3;
    };

    /// Level 7: 128 x 128 block tile, k step 32, 64 x 32 per warp (8 warps):
    /// 3–4% faster than 32 x 64 at N ≥ 4096, level at 1024.
    struct Pipe {
        static constexpr int BM = 128, BN = 128, BK = 32, WarpM = 64, WarpN = 32;
    };

    /// Low-precision cp.async kernel: 128 x 128, 2 x 4 warps of 64 x 32,
    /// 2 stages (8–18% faster than 3 or 4), blocks grouped by 8 tile rows.
    struct Lowp {
        static constexpr int BM = 128, BN = 128, WarpM = 64, WarpsN = 4, Stages = 2, GroupM = 8;
    };

    /// Level 8: 128 x 256 with 4 stages and 128 x 128 with 6, chosen per
    /// launch (the wide tile unless the narrow one fills the waves more than
    /// 5% better); blocks grouped by 8 tile rows.
    struct Tma {
        static constexpr int BM = 128, StagesWide = 4, StagesNarrow = 6, GroupM = 8, Tile = 0, WideMarginPct = 5;
    };
};

// ---------------------------------------------------------------------------
// Sweep variants: CudaParams with one level's struct replaced.
// ---------------------------------------------------------------------------

template <int TILE>
struct WithTile : CudaParams {
    static constexpr int Tile = TILE;
};

/// Levels 2, 3 and 4 variant for double (float keeps CudaParams' tiles).
template <typename RT, typename DB>
struct WithF64Tiles : CudaParams {
    template <typename T>
    struct RegTile : std::conditional_t<sizeof(T) == 8, RT, CudaParams::RegTile<T>> {};
    template <typename T>
    struct Vectorized : std::conditional_t<sizeof(T) == 8, RT, CudaParams::Vectorized<T>> {};
    template <typename T>
    struct DoubleBuf : std::conditional_t<sizeof(T) == 8, DB, CudaParams::DoubleBuf<T>> {};
};

template <int BM_, int BN_, int BK_, int TM_, int TN_>
struct RegTileOf {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, TM = TM_, TN = TN_;
};
/// Levels 2 and 4 variant for float (double keeps CudaParams' tiles).
template <typename RT>
struct WithRegTile : CudaParams {
    template <typename T>
    struct RegTile : std::conditional_t<sizeof(T) == 4, RT, CudaParams::RegTile<T>> {};
    template <typename T>
    struct Vectorized : std::conditional_t<sizeof(T) == 4, RT, CudaParams::Vectorized<T>> {};
};

template <int BM_, int BN_, int BK_, int TM_, int TN_, int MIN_BLOCKS>
struct DoubleBufOf {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, TM = TM_, TN = TN_, MinBlocks = MIN_BLOCKS;
};
/// Level 3 variant for float (double keeps CudaParams' tile).
template <typename DB>
struct WithDoubleBufF32 : CudaParams {
    template <typename T>
    struct DoubleBuf : std::conditional_t<sizeof(T) == 4, DB, CudaParams::DoubleBuf<T>> {};
};

template <int WARPS_M, int WARPS_N, int MIN_BLOCKS>
struct WarpGridOf {
    static constexpr int WarpsM = WARPS_M, WarpsN = WARPS_N, MinBlocks = MIN_BLOCKS;
};
template <typename WG>
struct WithWarpGrid : CudaParams {
    using WarpGrid = WG;
};

template <int BM_, int BN_, int BK_, int WARP_M, int WARP_N>
struct PipeOf {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, WarpM = WARP_M, WarpN = WARP_N;
};
template <typename PP>
struct WithPipe : CudaParams {
    using Pipe = PP;
};

template <int BM_, int BN_, int WARP_M, int WARPS_N, int STAGES, int GROUP_M>
struct LowpOf {
    static constexpr int BM = BM_, BN = BN_, WarpM = WARP_M, WarpsN = WARPS_N, Stages = STAGES, GroupM = GROUP_M;
};
template <typename LP>
struct WithLowp : CudaParams {
    using Lowp = LP;
};

template <int BM_, int STAGES_WIDE, int STAGES_NARROW, int GROUP_M, int TILE, int WIDE_MARGIN_PCT>
struct TmaOf {
    static constexpr int BM = BM_, StagesWide = STAGES_WIDE, StagesNarrow = STAGES_NARROW, GroupM = GROUP_M,
                         Tile = TILE, WideMarginPct = WIDE_MARGIN_PCT;
};
template <typename TP>
struct WithTma : CudaParams {
    using Tma = TP;
};

}  // namespace hpc::bench
