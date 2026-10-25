#pragma once

/**
 * @file simd_params.hpp
 * @brief Measured parameters of the generic SIMD kernels (gemm/simd.hpp,
 *        gemm/simd_lowp.hpp), for the benchmarks and experiments.
 *
 * The kernels take the register tile and the cache blocks as template
 * parameters with no defaults; these are the values the sweeps in
 * bench_gemm chose (<Isa>Tile, <Isa>Block, <Isa>PackBlock rows), with the
 * data and reasoning in docs/kernels/ and docs/experiments/packed-b/.
 * They are the best on the machines measured (AMD Zen 5 for AVX2 and
 * AVX-512, Apple M4 for NEON), not universal optima.
 *
 *   R × V               register tile: R rows × V vectors of C
 *   TileM/TileK/TileN   gemm_simd_blocked's cache block
 *   Mc/Kc/Nc            gemm_simd_packed's cache block
 *
 * LowpParams: gemm_simd_lowp's tile (R × V accumulators) and cache block
 * (Mc × Kc × Nc elements; Kc depends on the input type), from
 * docs/articles/lowp-gemm.md and bench_gemm_lowp's tile sweeps.
 */

#include "gemm/simd.hpp"
#include "gemm/simd_lowp.hpp"

#include <cstddef>
#include <type_traits>

namespace hpc::bench {

template <template <typename> class Ops>
struct SimdParams;

// Zen 5. Six rows give the 12 independent FMA chains Zen 5 needs (2 FMAs per
// cycle × 4 cycles) and still fit the 16 YMM registers: 12 accumulators,
// 2 B vectors, 1–2 broadcasts of A; seven rows spill. The B block is 256 KiB
// in either type (docs/kernels/avx2.md, blocks.md).
template <>
struct SimdParams<gemm::Ymm> {
    static constexpr int R = 6, V = 2;
    static constexpr std::size_t TileM = 384, TileK = 256;
    template <typename T>
    static constexpr std::size_t TileN = 1024 / sizeof(T);  // 256 f32, 128 f64
    static constexpr std::size_t Mc = 256, Kc = 256, Nc = 512;
};

// Zen 5. 24 accumulators in 28 of the 32 ZMM registers; with 6 or more rows
// GCC 13 loads B once per k step instead of in every FMA. The B block is
// 512 KiB, half of Zen 5's L2 (docs/kernels/avx512.md, blocks.md).
template <>
struct SimdParams<gemm::Zmm> {
    static constexpr int R = 8, V = 3;
    static constexpr std::size_t TileM = 512, TileK = 256;
    template <typename T>
    static constexpr std::size_t TileN = 2048 / sizeof(T);  // 512 f32, 256 f64
    static constexpr std::size_t Mc = 128, Kc = 512, Nc = 512;
};

// Apple M4. 24 accumulators, 4 B vectors and one element of A (29 of 32
// registers); the P-core needs 12–16 accumulators for its 4 FMA pipes
// (docs/kernels/neon.md, blocks.md).
template <>
struct SimdParams<gemm::Q> {
    static constexpr int R = 6, V = 4;
    static constexpr std::size_t TileM = 384, TileK = 256;
    template <typename T>
    static constexpr std::size_t TileN = 256;
    static constexpr std::size_t Mc = 256, Kc = 512, Nc = 512;
};

template <template <typename> class Ops>
struct LowpParams;

// Zen 5. 8 × 2 ZMM accumulators (8 rows × 32 columns of C); a k block of
// 1 KiB (512 bf16, 1,024 INT8) keeps a B strip, 32 KiB, inside the 48 KiB L1.
template <>
struct LowpParams<gemm::ZmmDot> {
    static constexpr int R = 8, V = 2;
    static constexpr std::size_t Mc = 128, Nc = 512;
    template <typename In>
    static constexpr std::size_t Kc = 1024 / sizeof(In);
};

// Apple M4. 4 × 4 accumulators of 2 × 2: an 8 × 8 block of C in 16 registers.
template <>
struct LowpParams<gemm::QMmla> {
    static constexpr int R = 4, V = 4;
    static constexpr std::size_t Mc = 256, Nc = 512;
    template <typename In>
    static constexpr std::size_t Kc = 2048 / sizeof(In);  // 1,024 bf16, 2,048 INT8
};

// Apple M4. 4 rows (the lanes of A) × 4 accumulators: 4 × 16 (FMLAL) or
// 4 × 32 (FMLA .8h) elements of C.
template <>
struct LowpParams<gemm::QFmlal> {
    static constexpr int R = 4, V = 4;
    static constexpr std::size_t Mc = 256, Nc = 512;
    template <typename In>
    static constexpr std::size_t Kc = 512;
};
template <>
struct LowpParams<gemm::QFmlaF16> : LowpParams<gemm::QFmlal> {};

/// gemm_simd_lowp at its measured parameters, as a function pointer.
template <template <typename> class Ops, typename In>
inline constexpr auto simd_lowp = +[](const Matrix<In>& A, const Matrix<In>& B, Matrix<gemm::lowp_out_t<In>>& C) {
    using P = LowpParams<Ops>;
    gemm::gemm_simd_lowp<Ops, P::R, P::V, P::Mc, P::template Kc<In>, P::Nc>(A, B, C);
};

/// gemm_simd_reordered, which has no parameters to tune, as a kernel object.
template <template <typename> class Ops>
inline constexpr auto simd_reordered = [](const auto& A, const auto& B, auto& C) {
    gemm::gemm_simd_reordered<Ops>(A, B, C);
};

/// gemm_simd_blocked at its measured parameters; PfDist > 0 adds prefetch.
template <template <typename> class Ops, std::size_t PfDist = 0>
inline constexpr auto simd_blocked = [](const auto& A, const auto& B, auto& C) {
    using T = typename std::remove_cvref_t<decltype(C)>::value_type;
    using P = SimdParams<Ops>;
    gemm::gemm_simd_blocked<Ops, P::R, P::V, P::TileM, P::TileK, P::template TileN<T>, PfDist>(A, B, C);
};

/// gemm_simd_packed at its measured parameters; PackA packs A too.
template <template <typename> class Ops, bool PackA = false>
inline constexpr auto simd_packed = [](const auto& A, const auto& B, auto& C) {
    using P = SimdParams<Ops>;
    gemm::gemm_simd_packed<Ops, P::R, P::V, P::Mc, P::Kc, P::Nc, PackA>(A, B, C);
};

}  // namespace hpc::bench
