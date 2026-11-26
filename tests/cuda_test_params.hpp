#pragma once

/**
 * @file cuda_test_params.hpp
 * @brief The parameter sets the CUDA tests run every kernel with: the
 *        measured one (benchmarks/cuda_params.hpp) and one where every size
 *        differs from it, so the tests cover the templates, not one shape.
 */

#include "cuda_params.hpp"

namespace hpc::test {

using CudaMeasured = hpc::bench::CudaParams;

/// Every size different from CudaParams, each within its level's constraints.
struct CudaOdd {
    static constexpr int Tile = 8;
    template <typename T>
    struct RegTile {
        static constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    };
    template <typename T>
    struct Vectorized {
        static constexpr int BM = 32, BN = 32, BK = 8, TM = 4, TN = 4;
    };
    template <typename T>
    struct DoubleBuf {
        static constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4, MinBlocks = 1;
    };
    struct WarpGrid {
        static constexpr int WarpsM = 2, WarpsN = 4, MinBlocks = 1;
    };
    struct Pipe {
        static constexpr int BM = 64, BN = 128, BK = 16, WarpM = 32, WarpN = 64;
    };
    struct Lowp {
        static constexpr int BM = 64, BN = 128, WarpM = 32, WarpsN = 4, Stages = 3, GroupM = 4;
    };
    struct Tma {
        static constexpr int BM = 64, StagesWide = 3, StagesNarrow = 5, GroupM = 4, Tile = 0, WideMarginPct = 5;
    };
};

}  // namespace hpc::test
