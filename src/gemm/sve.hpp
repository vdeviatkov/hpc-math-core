#pragma once

/**
 * @file sve.hpp
 * @brief ARM SVE counterparts of the reordered and blocked SIMD kernels
 *        (simd.hpp).
 *
 * Unlike NEON (128-bit), AVX2 (256-bit) and AVX-512 (512-bit), SVE's vector
 * length is implementation-defined (128–2048 bits) and read at runtime with
 * svcntw() (f32 lanes) / svcntd() (f64 lanes). Loop steps are computed from
 * it, so one binary runs on 128-bit (Neoverse N2/V2, Graviton4), 256-bit
 * (Neoverse V1, Graviton3) and 512-bit (A64FX) hardware.
 *
 * Every load, store and FMA takes a predicate. svwhilelt_b32(j, N) is
 * active only for lanes with j + lane < N, so the last partial vector is
 * handled by the same loop body — there is no scalar tail loop.
 *
 *  gemm_sve_reordered  i → k → j: broadcast A(i,k), FMA against B and C rows.
 *  gemm_sve_blocked    tiled i → k → j with a 4-row × 2-vector C register
 *                      tile by default; R and V are template parameters.
 *
 * Not measured: no SVE hardware was available. Apple Silicon has no
 * non-streaming SVE (see sme.hpp).
 *
 * HPC_HAS_SVE (hpc/isa.hpp) follows __ARM_FEATURE_SVE, set by e.g.
 * -march=armv8.2-a+sve, -mcpu=neoverse-v1, or -march=native on SVE hardware.
 * Where it is 0 the kernels are declared `= delete`.
 */

#include "gemm/simd.hpp"  // detail::pf
#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"

#if HPC_HAS_SVE
    #include <arm_sve.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpc::gemm {

// ============================================================================
// Kernel 1: gemm_sve_reordered  —  i → k → j,  SVE on the j-loop (VLA)
// ============================================================================

/**
 * @brief SVE GEMM with cache-friendly i-k-j loop order — vector-length agnostic.
 *
 * Counterpart of gemm_simd_reordered (simd.hpp). The j-step is
 * svcntw()/svcntd(); the last partial vector uses an svwhilelt_b32(j, N)
 * predicate, so one loop covers any N.
 *
 * Not measured (no SVE hardware available). On every measured family the
 * explicit-SIMD reordered kernel performs about the same as the
 * auto-vectorised scalar one.
 */
#if !HPC_HAS_SVE
template <typename T>
void gemm_sve_reordered(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // ARM_FEATURE_SVE not available on this target
#else
template <typename T>
void gemm_sve_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sve_reordered: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    if constexpr (sizeof(T) == 4) {
        const std::uint64_t vl = svcntw();
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t k = 0; k < K; ++k) {
                const svfloat32_t a_broad = svdup_n_f32(A.data()[i * lda + k]);
                const float* b_row        = B.data() + k * ldb;
                float* c_row              = C.data() + i * ldc;

                // VLA j-loop: step = vl, last iteration uses tail predicate.
                std::uint64_t j = 0;
                for (svbool_t pg = svwhilelt_b32(j, (std::uint64_t)N);
                     svptest_any(svptrue_b32(), pg);
                     j += vl, pg = svwhilelt_b32(j, (std::uint64_t)N)) {
                    const svfloat32_t b_vec = svld1_f32(pg, b_row + j);
                    const svfloat32_t c_vec = svld1_f32(pg, c_row + j);
                    svst1_f32(pg, c_row + j, svmla_f32_x(pg, c_vec, a_broad, b_vec));
                }
            }
        }
    } else {
        const std::uint64_t vl = svcntd();
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t k = 0; k < K; ++k) {
                const svfloat64_t a_broad = svdup_n_f64(A.data()[i * lda + k]);
                const double* b_row       = B.data() + k * ldb;
                double* c_row             = C.data() + i * ldc;

                std::uint64_t j = 0;
                for (svbool_t pg = svwhilelt_b64(j, (std::uint64_t)N);
                     svptest_any(svptrue_b64(), pg);
                     j += vl, pg = svwhilelt_b64(j, (std::uint64_t)N)) {
                    const svfloat64_t b_vec = svld1_f64(pg, b_row + j);
                    const svfloat64_t c_vec = svld1_f64(pg, c_row + j);
                    svst1_f64(pg, c_row + j, svmla_f64_x(pg, c_vec, a_broad, b_vec));
                }
            }
        }
    }
}
#endif  // HPC_HAS_SVE

// ============================================================================
// For Kernel 2: tile sizes (also used by gemm_sve_blocked_prefetch)
// ============================================================================
// The j-step is V * svcntw()/svcntd(), read at runtime. Register shape R/V
// and the outer blocking tile sizes are compile-time parameters.

inline constexpr std::size_t kSveTileM = 64;   // outer i-tile (rows of A / C)
inline constexpr std::size_t kSveTileK = 256;  // outer k-tile (contraction width)
inline constexpr std::size_t kSveTileN = 512;  // outer j-tile (cols of B / C)

// Default register tile, retained without claiming an unmeasured optimum:
// 4 rows × 2 vectors; at 256-bit VL this is 4 × 16 f32 (256 B).
inline constexpr std::size_t kSveRegRows = 4;
// Number of SVE vectors per C row in the micro-kernel.
// 2 vectors × VL elements = 2*svcntw() f32 or 2*svcntd() f64 per row.
inline constexpr std::size_t kSveRegCols = 2;

// ============================================================================
// For Kernel 2: the register-tile micro-kernels
// ============================================================================

#if HPC_HAS_SVE

namespace sve_detail {

// SVE vectors and predicates are sizeless: C++ arrays/aggregate members are
// forbidden. Parameter packs give every accumulator a separate local instead.
template <typename T>
struct Ops;
template <>
struct Ops<float> {
    static std::size_t lanes() { return svcntw(); }
    static svbool_t predicate(std::uint64_t first, std::uint64_t end) {
        return svwhilelt_b32(first, end);
    }
    static svfloat32_t load(svbool_t pg, const float* p) { return svld1_f32(pg, p); }
    static void store(svbool_t pg, float* p, svfloat32_t x) { svst1_f32(pg, p, x); }
    static svfloat32_t load_a(const float* p) { return svdup_f32(*p); }
    static svfloat32_t fma(svbool_t pg, svfloat32_t acc, svfloat32_t a, svfloat32_t b) {
        return svmla_f32_x(pg, acc, a, b);
    }
};
template <>
struct Ops<double> {
    static std::size_t lanes() { return svcntd(); }
    static svbool_t predicate(std::uint64_t first, std::uint64_t end) {
        return svwhilelt_b64(first, end);
    }
    static svfloat64_t load(svbool_t pg, const double* p) { return svld1_f64(pg, p); }
    static void store(svbool_t pg, double* p, svfloat64_t x) { svst1_f64(pg, p, x); }
    static svfloat64_t load_a(const double* p) { return svdup_f64(*p); }
    static svfloat64_t fma(svbool_t pg, svfloat64_t acc, svfloat64_t a, svfloat64_t b) {
        return svmla_f64_x(pg, acc, a, b);
    }
};

// Compile-time indexing into a pack; always inlined, with no runtime lookup.
template <std::size_t I, typename First, typename... Rest>
[[gnu::always_inline]] inline decltype(auto) at(First& first, Rest&... rest) noexcept {
    if constexpr (I == 0)
        return (first);
    else
        return at<I - 1>(rest...);
}

template <int V, typename T, std::size_t... I, std::size_t... Rows, std::size_t... Cols,
          typename... Acc>
[[gnu::always_inline]] inline void accumulate(const T* a, const T* b, T* c, std::size_t lda,
                                              std::size_t ldb, std::size_t ldc, std::size_t k_len,
                                              std::size_t cols, std::size_t vl,
                                              std::index_sequence<I...>,
                                              std::index_sequence<Rows...>,
                                              std::index_sequence<Cols...>, Acc... acc) noexcept {
    using O       = Ops<T>;
    const auto pg = [=](std::size_t v) {
        return O::predicate(v * vl, cols);
    };
    // Entirely inactive vectors use a valid row-end pointer, never an address
    // beyond the allocation. Stores use the same predicates as the loads.
    const auto offset = [=](std::size_t v) {
        return std::min(v * vl, cols);
    };
    for (std::size_t k = 0; k < k_len; ++k) {
        const auto step = [&](auto... bv) {
            const auto row = [&]<std::size_t R>() {
                const auto ar = O::load_a(a + R * lda + k);
                ((at<R * V + Cols>(acc...) = O::fma(pg(Cols), at<R * V + Cols>(acc...), ar,
                                                    at<Cols>(bv...))),
                 ...);
            };
            (row.template operator()<Rows>(), ...);
        };
        // Load V vectors of B once, then reuse each across all rows.
        step(O::load(pg(Cols), b + k * ldb + offset(Cols))...);
    }
    (O::store(pg(I % V), c + (I / V) * ldc + offset(I % V), acc), ...);
}

template <int R, int V, typename T, std::size_t... I>
[[gnu::always_inline]] inline void micro(const T* a, const T* b, T* c, std::size_t lda,
                                         std::size_t ldb, std::size_t ldc, std::size_t k_len,
                                         std::size_t cols,
                                         std::index_sequence<I...> indices) noexcept {
    using O              = Ops<T>;
    const std::size_t vl = O::lanes();
    accumulate<V>(a, b, c, lda, ldb, ldc, k_len, cols, vl, indices, std::make_index_sequence<R>{},
                  std::make_index_sequence<V>{},
                  O::load(O::predicate((I % V) * vl, cols),
                          c + (I / V) * ldc + std::min((I % V) * vl, cols))...);
}

}  // namespace sve_detail

/// C[R][cols] += A[R][k_len] * B[k_len][cols], R rows by V scalable vectors.
/// cols must be in 1..V*VL and all R rows must exist. Omit cols for a full tile.
/// R*V independent accumulators remain live through the K loop; inactive column
/// lanes are predicated. Very large shapes may spill, just as with other ISAs.
template <int R, int V, typename T>
inline void sve_micro(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                      std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t k_len,
                      std::size_t cols = V * sve_detail::Ops<T>::lanes()) noexcept {
    static_assert(R > 0 && V > 0, "SVE register tile dimensions must be positive");
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    sve_detail::micro<R, V>(a, b, c, lda, ldb, ldc, k_len, cols, std::make_index_sequence<R * V>{});
}

namespace detail {

// Select a smaller row specialization for a partial register tile. Column
// tails stay predicated, including entirely inactive vectors in a wide tile.
template <int R, int V, typename T>
[[gnu::always_inline]] inline void sve_run_rows(int rows, const T* a, const T* b, T* c,
                                                std::size_t lda, std::size_t ldb, std::size_t ldc,
                                                std::size_t k_len, std::size_t cols) noexcept {
    if (rows == R)
        sve_micro<R, V>(a, b, c, lda, ldb, ldc, k_len, cols);
    else if constexpr (R > 1)
        sve_run_rows<R - 1, V>(rows, a, b, c, lda, ldb, ldc, k_len, cols);
}

/// Shared traversal; PfDist counts R-row strips. Zero removes all prefetch
/// hints. Cache tiles and the register tile are independent template parameters.
template <typename T, int R, int V, std::size_t TileM, std::size_t TileK, std::size_t TileN,
          std::size_t PfDist = 0>
void sve_blocked_gemm(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(R > 0 && V > 0, "SVE register tile dimensions must be positive");
    static_assert(TileM > 0 && TileK > 0 && TileN > 0,
                  "SVE cache tile dimensions must be positive");
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    const std::size_t j_step = V * sve_detail::Ops<T>::lanes();
    for (std::size_t i_blk = 0; i_blk < M; i_blk += TileM) {
        const std::size_t i_end = std::min(i_blk + TileM, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += TileK) {
            const std::size_t k_len = std::min(TileK, K - k_blk);
            if constexpr (PfDist > 0)
                if (k_blk + TileK < K)
                    pf<0, 2>(B.data() + (k_blk + TileK) * N);
            for (std::size_t j_blk = 0; j_blk < N; j_blk += TileN) {
                const std::size_t j_end = std::min(j_blk + TileN, N);
                for (std::size_t i = i_blk; i < i_end; i += R) {
                    if constexpr (PfDist > 0)
                        if (i + R * PfDist < i_end) {
                            pf<0, 2>(A.data() + (i + R * PfDist) * K + k_blk);
                            pf<1, 3>(C.data() + (i + R * PfDist) * N + j_blk);
                        }
                    const int rows = static_cast<int>(std::min<std::size_t>(R, i_end - i));
                    for (std::size_t j = j_blk; j < j_end; j += j_step)
                        sve_run_rows<R, V>(rows, A.data() + i * K + k_blk, B.data() + k_blk * N + j,
                                           C.data() + i * N + j, K, N, N, k_len,
                                           std::min(j_step, j_end - j));
                }
            }
        }
    }
}

}  // namespace detail
#endif  // HPC_HAS_SVE

// ============================================================================
// Kernel 2: gemm_sve_blocked  —  tiled i → k → j,  VLA register tile
// ============================================================================

/**
 * @brief SVE GEMM with cache-blocking and VLA register-tiled micro-kernel.
 *
 * Counterpart of gemm_simd_blocked (simd.hpp): i-k-j order, 3-level
 * tiling (TileM × TileK × TileN), and an R-row × V-vector C register tile
 * held for a whole K block. Width is V*svcntw()/svcntd(); column tails use
 * predicates and partial row tiles use smaller specializations. Defaults
 * remain R=4, V=2; no SVE hardware measurements establish a better choice.
 *
 * Tile footprints (f32): A 64 KB, B 512 KB (L2-resident on Neoverse V1's
 * 1 MB L2), C 128 KB streamed.
 */
#if !HPC_HAS_SVE
template <typename T, int R = kSveRegRows, int V = kSveRegCols, std::size_t TileM = kSveTileM,
          std::size_t TileK = kSveTileK, std::size_t TileN = kSveTileN>
void gemm_sve_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;
#else
template <typename T, int R = kSveRegRows, int V = kSveRegCols, std::size_t TileM = kSveTileM,
          std::size_t TileK = kSveTileK, std::size_t TileN = kSveTileN>
void gemm_sve_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    detail::sve_blocked_gemm<T, R, V, TileM, TileK, TileN>(A, B, C);
}
#endif  // HPC_HAS_SVE

}  // namespace hpc::gemm
