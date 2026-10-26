#pragma once

/**
 * @file sme_params.hpp
 * @brief Measured parameters of the SME kernels (gemm/sme.hpp), for the
 *        benchmarks and experiments.
 *
 * The kernels take the ZA shape and the cache blocks as template parameters
 * with no defaults; these are the values the sweeps chose on Apple M4 Max
 * (bench_gemm's SmeTile, SmeBlock and SmeMtT rows, sme_block_sweep), with
 * the data and reasoning in docs/kernels/sme.md and docs/experiments/.
 * They are the best on the machine measured, not universal optima.
 *
 *   R × V        ZA tiles: R rows × V columns of SVL × SVL tiles
 *   Mc/Kc/Nc     the cache block in elements (Kc a whole number of k groups)
 *   Threads      gemm_sme_mt's thread count; MtMc its A block height
 */

#include "gemm/sme.hpp"
#include "hpc/matrix.hpp"

#include <cstddef>

namespace hpc::bench {

template <typename T>
struct SmeParams {
    /// 32-bit ZA tiles (FP32 and the widening types) or 64-bit (FP64).
    static constexpr bool kWord32 = sizeof(gemm::sme_word_t<T>) == 4;

    // Every ZA tile, in two rows: 2 × 2 za32 (a 32 × 32 C block at M4's SVL)
    // or 2 × 4 za64 (16 × 32). A 2 × 2 FP64 layout (4 of 8 tiles) measured
    // 9% slower at N = 4096 (sme.hpp, "gemm_sme").
    static constexpr int R = 2, V = kWord32 ? 2 : 4;

    // Swept over Mc 128–2048, Kc 1024–4096, Nc 1024–4096 at N = 512 … 4096:
    // the A block (Mc × Kc) is 8 MiB in both FP types, half the L2, and each
    // B strip is reused by Mc / 32 (f32) or Mc / 16 (f64) A strips in a row
    // (docs/experiments/sme2-gemm). The widening types keep FP32's sizes in
    // k groups, 2,048 words: Kc = 4,096 fp16 / bf16 or 8,192 INT8 elements,
    // an 8 MiB A block too; they were not swept.
    static constexpr std::size_t Mc = kWord32 ? 1024 : 512;
    static constexpr std::size_t Kc = 2048 * gemm::kSmeG<T>;
    static constexpr std::size_t Nc = 4096;

    // gemm_sme_mt: 8 threads were best or within 1.1% of the best at every
    // N, with gemm_sme's A block height (docs/kernels/sme.md, "Several SME
    // units"; docs/experiments/sme-mt).
    static constexpr unsigned Threads = 8;
    static constexpr std::size_t MtMc = Mc;
};

/// gemm_sme_tiles at the measured ZA shape (no cache blocks to choose).
inline constexpr auto sme_tiles = []<typename T>(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    using P = SmeParams<T>;
    gemm::gemm_sme_tiles<T, P::R, P::V>(A, B, C);
};

/// gemm_sme_blocked at the measured parameters.
inline constexpr auto sme_blocked = []<typename T>(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    using P = SmeParams<T>;
    gemm::gemm_sme_blocked<T, P::R, P::V, P::Mc, P::Kc, P::Nc>(A, B, C);
};

/// gemm_sme at the measured parameters, any of the five input types.
inline constexpr auto sme = []<typename T>(const Matrix<T>& A, const Matrix<T>& B,
                                           Matrix<gemm::sme_out_t<T>>& C) {
    using P = SmeParams<T>;
    gemm::gemm_sme<T, P::R, P::V, P::Mc, P::Kc, P::Nc>(A, B, C);
};

/// gemm_sme_mt at the measured parameters and thread count.
inline constexpr auto sme_mt = []<typename T>(const Matrix<T>& A, const Matrix<T>& B,
                                              Matrix<gemm::sme_out_t<T>>& C) {
    using P = SmeParams<T>;
    gemm::gemm_sme_mt<T, P::R, P::V, P::MtMc, P::Kc, P::Nc>(A, B, C, P::Threads);
};

}  // namespace hpc::bench
