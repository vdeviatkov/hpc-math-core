#pragma once

/**
 * @file prefetch.hpp
 * @brief Software-prefetch variants of every blocked GEMM kernel.
 *
 * ============================================================
 *  What software prefetch is and why it helps
 * ============================================================
 *
 * Modern CPUs have a hardware prefetcher that detects sequential or
 * strided access patterns and issues cache-line fills ahead of time.
 * It works well for simple streams but struggles with:
 *
 *   1. The k-stride walk over A rows — stride = N*sizeof(T) bytes,
 *      may not be predicted reliably at large N.
 *   2. The next B k-tile — after finishing k_blk..k_blk+TK the next
 *      tile starts at a cold address.
 *   3. The C write rows — first write to a C row that is no longer in
 *      L1 incurs a load-for-ownership miss.
 *
 *
 * ============================================================
 *  __builtin_prefetch(addr, rw, locality)
 * ============================================================
 *
 *   rw        0 = prefetch for read       (PREFETCHT* / PRFM PLDL*KEEP)
 *             1 = prefetch for read+write  (PREFETCHW  / PRFM PSTL*KEEP)
 *   locality  2 → L2     ← used for A/B data used in ~4–16 iterations
 *             3 → L1     ← used for C data we are about to write
 *
 *
 * ============================================================
 *  Prefetch distance
 * ============================================================
 *
 * Distance D = rows ahead of the current micro-kernel call.
 * D is exposed as template parameter PfDist (default = 4).
 * The benchmark sweeps PfDist in {2, 4, 8, 16} per ISA family.
 *
 *
 * ============================================================
 *  Design
 * ============================================================
 *
 * Each wrapper:
 *  1. Has the *exact same* outer tiled loop as the base kernel.
 *  2. Adds __builtin_prefetch before each micro-kernel call.
 *  3. Calls the *same* micro-kernel — no SIMD code duplication.
 *  4. Compiles to a no-op body (benchmark SKIPPED) on unsupported ISA.
 */

#include "gemm/avx2.hpp"
#include "gemm/avx512.hpp"
#include "gemm/blocked.hpp"
#include "gemm/neon.hpp"
#include "gemm/sve.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace hpc::gemm {

/// Default prefetch distance in micro-kernel rows.
inline constexpr std::size_t kDefaultPrefetchDist = 4;

/// Prefetch helper. rw: 0=read, 1=write. loc: 2=L2, 3=L1.
template <int Rw = 0, int Loc = 2>
[[gnu::always_inline]] inline void pf(const void* addr) noexcept {
    __builtin_prefetch(addr, Rw, Loc);
}

// ============================================================================
// 1.  Scalar blocked + prefetch
// ============================================================================
template <typename T, std::size_t PfDist = kDefaultPrefetchDist,
          std::size_t Tile = kDefaultTile>
void gemm_blocked_prefetch(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    for (std::size_t i_blk = 0; i_blk < M; i_blk += Tile) {
        const std::size_t i_end = std::min(i_blk + Tile, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += Tile) {
            const std::size_t k_end = std::min(k_blk + Tile, K);
            // [PF-B] prefetch next B k-tile into L2
            if (k_blk + Tile < K)
                pf<0, 2>(B.data() + (k_blk + Tile) * ldb);
            for (std::size_t j_blk = 0; j_blk < N; j_blk += Tile) {
                const std::size_t j_end = std::min(j_blk + Tile, N);
                for (std::size_t i = i_blk; i < i_end; ++i) {
                    // [PF-A] next A row into L2
                    if (i + PfDist < i_end)
                        pf<0, 2>(A.data() + (i + PfDist) * lda + k_blk);
                    // [PF-C] next C write row into L1
                    if (i + PfDist < i_end)
                        pf<1, 3>(C.data() + (i + PfDist) * ldc + j_blk);
                    for (std::size_t k = k_blk; k < k_end; ++k) {
                        const T a_ik = A(i, k);
                        for (std::size_t j = j_blk; j < j_end; ++j)
                            C(i, j) += a_ik * B(k, j);
                    }
                }
            }
        }
    }
}

// ============================================================================
// 2.  AVX2 blocked + prefetch
// ============================================================================
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_avx2_blocked_prefetch(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __AVX2__
    (void)A; (void)B; (void)C;
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K, ldb = N, ldc = N;
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    constexpr std::size_t kSimdW   = (sizeof(T) == 4) ? 8u : 4u;
    constexpr std::size_t kRegRows = (sizeof(T) == 4) ? kF32RegRows : kF64RegRows;
    constexpr std::size_t kRegCols = (sizeof(T) == 4) ? kF32RegCols : kF64RegCols;
    constexpr std::size_t kJStep   = kSimdW * kRegCols;

    for (std::size_t i_blk = 0; i_blk < M; i_blk += kAvx2TileM) {
        const std::size_t i_end = std::min(i_blk + kAvx2TileM, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += kAvx2TileK) {
            const std::size_t k_end = std::min(k_blk + kAvx2TileK, K);
            const std::size_t k_len = k_end - k_blk;
            if (k_blk + kAvx2TileK < K)
                pf<0, 2>(B.data() + (k_blk + kAvx2TileK) * ldb);
            for (std::size_t j_blk = 0; j_blk < N; j_blk += kAvx2TileN) {
                const std::size_t j_end = std::min(j_blk + kAvx2TileN, N);
                std::size_t i = i_blk;
                for (; i + kRegRows <= i_end; i += kRegRows) {
                    if (i + kRegRows * PfDist < i_end) {
                        pf<0, 2>(A.data() + (i + kRegRows * PfDist) * lda + k_blk);  // [PF-A]
                        pf<1, 3>(C.data() + (i + kRegRows * PfDist) * ldc + j_blk);  // [PF-C]
                    }
                    const T* a_ptr = A.data() + i * lda + k_blk;
                    const T* b_ptr = B.data() + k_blk * ldb + j_blk;
                    std::size_t j = j_blk;
                    for (; j + kJStep <= j_end; j += kJStep) {
                        T* c0 = C.data() + (i + 0) * ldc + j;
                        T* c1 = C.data() + (i + 1) * ldc + j;
                        T* c2 = C.data() + (i + 2) * ldc + j;
                        T* c3 = C.data() + (i + 3) * ldc + j;
                        if constexpr (sizeof(T) == 4)
                            avx2_micro_f32_4x16(reinterpret_cast<const float*>(a_ptr),
                                reinterpret_cast<const float*>(b_ptr + (j - j_blk)),
                                reinterpret_cast<float*>(c0), reinterpret_cast<float*>(c1),
                                reinterpret_cast<float*>(c2), reinterpret_cast<float*>(c3),
                                lda, ldb, k_len);
                        else
                            avx2_micro_f64_4x8(reinterpret_cast<const double*>(a_ptr),
                                reinterpret_cast<const double*>(b_ptr + (j - j_blk)),
                                reinterpret_cast<double*>(c0), reinterpret_cast<double*>(c1),
                                reinterpret_cast<double*>(c2), reinterpret_cast<double*>(c3),
                                lda, ldb, k_len);
                    }
                    for (; j < j_end; ++j)
                        for (std::size_t ii = i; ii < i + kRegRows; ++ii) {
                            T acc{};
                            for (std::size_t k = k_blk; k < k_end; ++k) acc += A(ii, k) * B(k, j);
                            C(ii, j) += acc;
                        }
                }
                for (; i < i_end; ++i)
                    for (std::size_t k = k_blk; k < k_end; ++k) {
                        const T a_ik = A(i, k);
                        for (std::size_t j = j_blk; j < j_end; ++j) C(i, j) += a_ik * B(k, j);
                    }
            }
        }
    }
#endif
}

// ============================================================================
// 3.  AVX-512 blocked + prefetch
// ============================================================================
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_avx512_blocked_prefetch(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __AVX512F__
    (void)A; (void)B; (void)C;
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K, ldb = N, ldc = N;
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    constexpr std::size_t kSimdW   = (sizeof(T) == 4) ? 16u : 8u;
    constexpr std::size_t kRegRows = (sizeof(T) == 4) ? kAvx512F32RegRows : kAvx512F64RegRows;
    constexpr std::size_t kRegCols = (sizeof(T) == 4) ? kAvx512F32RegCols : kAvx512F64RegCols;
    constexpr std::size_t kJStep   = kSimdW * kRegCols;

    for (std::size_t i_blk = 0; i_blk < M; i_blk += kAvx512TileM) {
        const std::size_t i_end = std::min(i_blk + kAvx512TileM, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += kAvx512TileK) {
            const std::size_t k_end = std::min(k_blk + kAvx512TileK, K);
            const std::size_t k_len = k_end - k_blk;
            if (k_blk + kAvx512TileK < K)
                pf<0, 2>(B.data() + (k_blk + kAvx512TileK) * ldb);
            for (std::size_t j_blk = 0; j_blk < N; j_blk += kAvx512TileN) {
                const std::size_t j_end = std::min(j_blk + kAvx512TileN, N);
                std::size_t i = i_blk;
                for (; i + kRegRows <= i_end; i += kRegRows) {
                    if (i + kRegRows * PfDist < i_end) {
                        pf<0, 2>(A.data() + (i + kRegRows * PfDist) * lda + k_blk);
                        pf<1, 3>(C.data() + (i + kRegRows * PfDist) * ldc + j_blk);
                    }
                    const T* a_ptr = A.data() + i * lda + k_blk;
                    const T* b_ptr = B.data() + k_blk * ldb + j_blk;
                    std::size_t j = j_blk;
                    for (; j + kJStep <= j_end; j += kJStep) {
                        T* c0 = C.data() + (i + 0) * ldc + j;
                        T* c1 = C.data() + (i + 1) * ldc + j;
                        T* c2 = C.data() + (i + 2) * ldc + j;
                        T* c3 = C.data() + (i + 3) * ldc + j;
                        if constexpr (sizeof(T) == 4)
                            avx512_micro_f32_4x32(reinterpret_cast<const float*>(a_ptr),
                                reinterpret_cast<const float*>(b_ptr + (j - j_blk)),
                                reinterpret_cast<float*>(c0), reinterpret_cast<float*>(c1),
                                reinterpret_cast<float*>(c2), reinterpret_cast<float*>(c3),
                                lda, ldb, k_len);
                        else
                            avx512_micro_f64_4x16(reinterpret_cast<const double*>(a_ptr),
                                reinterpret_cast<const double*>(b_ptr + (j - j_blk)),
                                reinterpret_cast<double*>(c0), reinterpret_cast<double*>(c1),
                                reinterpret_cast<double*>(c2), reinterpret_cast<double*>(c3),
                                lda, ldb, k_len);
                    }
                    for (; j < j_end; ++j)
                        for (std::size_t ii = i; ii < i + kRegRows; ++ii) {
                            T acc{};
                            for (std::size_t k = k_blk; k < k_end; ++k) acc += A(ii, k) * B(k, j);
                            C(ii, j) += acc;
                        }
                }
                for (; i < i_end; ++i)
                    for (std::size_t k = k_blk; k < k_end; ++k) {
                        const T a_ik = A(i, k);
                        for (std::size_t j = j_blk; j < j_end; ++j) C(i, j) += a_ik * B(k, j);
                    }
            }
        }
    }
#endif
}

// ============================================================================
// 4.  NEON blocked + prefetch
// ============================================================================
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_neon_blocked_prefetch(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __ARM_NEON
    (void)A; (void)B; (void)C;
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K, ldb = N, ldc = N;
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    // NEON: 4 f32 per lane, kNeonF32RegCols=4 Q-vecs → kJStep=16
    //       2 f64 per lane, kNeonF64RegCols=2 Q-vecs → kJStep=4
    constexpr std::size_t kSimdW   = (sizeof(T) == 4) ? 4u : 2u;
    constexpr std::size_t kRegRows = (sizeof(T) == 4) ? kNeonF32RegRows : kNeonF64RegRows;
    constexpr std::size_t kRegCols = (sizeof(T) == 4) ? kNeonF32RegCols : kNeonF64RegCols;
    constexpr std::size_t kJStep   = kSimdW * kRegCols;

    for (std::size_t i_blk = 0; i_blk < M; i_blk += kNeonTileM) {
        const std::size_t i_end = std::min(i_blk + kNeonTileM, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += kNeonTileK) {
            const std::size_t k_end = std::min(k_blk + kNeonTileK, K);
            const std::size_t k_len = k_end - k_blk;
            if (k_blk + kNeonTileK < K)
                pf<0, 2>(B.data() + (k_blk + kNeonTileK) * ldb);  // [PF-B]
            for (std::size_t j_blk = 0; j_blk < N; j_blk += kNeonTileN) {
                const std::size_t j_end = std::min(j_blk + kNeonTileN, N);
                std::size_t i = i_blk;
                for (; i + kRegRows <= i_end; i += kRegRows) {
                    if (i + kRegRows * PfDist < i_end) {
                        pf<0, 2>(A.data() + (i + kRegRows * PfDist) * lda + k_blk);  // [PF-A]
                        pf<1, 3>(C.data() + (i + kRegRows * PfDist) * ldc + j_blk);  // [PF-C]
                    }
                    const T* a_ptr = A.data() + i * lda + k_blk;
                    const T* b_ptr = B.data() + k_blk * ldb + j_blk;
                    std::size_t j = j_blk;
                    for (; j + kJStep <= j_end; j += kJStep) {
                        T* c0 = C.data() + (i + 0) * ldc + j;
                        T* c1 = C.data() + (i + 1) * ldc + j;
                        T* c2 = C.data() + (i + 2) * ldc + j;
                        T* c3 = C.data() + (i + 3) * ldc + j;
                        if constexpr (sizeof(T) == 4)
                            neon_micro_f32_4x16(reinterpret_cast<const float*>(a_ptr),
                                reinterpret_cast<const float*>(b_ptr + (j - j_blk)),
                                reinterpret_cast<float*>(c0), reinterpret_cast<float*>(c1),
                                reinterpret_cast<float*>(c2), reinterpret_cast<float*>(c3),
                                lda, ldb, k_len);
                        else
                            neon_micro_f64_4x4(reinterpret_cast<const double*>(a_ptr),
                                reinterpret_cast<const double*>(b_ptr + (j - j_blk)),
                                reinterpret_cast<double*>(c0), reinterpret_cast<double*>(c1),
                                reinterpret_cast<double*>(c2), reinterpret_cast<double*>(c3),
                                lda, ldb, k_len);
                    }
                    for (; j < j_end; ++j)
                        for (std::size_t ii = i; ii < i + kRegRows; ++ii) {
                            T acc{};
                            for (std::size_t k = k_blk; k < k_end; ++k) acc += A(ii, k) * B(k, j);
                            C(ii, j) += acc;
                        }
                }
                for (; i < i_end; ++i)
                    for (std::size_t k = k_blk; k < k_end; ++k) {
                        const T a_ik = A(i, k);
                        for (std::size_t j = j_blk; j < j_end; ++j) C(i, j) += a_ik * B(k, j);
                    }
            }
        }
    }
#endif
}

// ============================================================================
// 5.  SVE blocked + prefetch
// ============================================================================
template <typename T, std::size_t PfDist = kDefaultPrefetchDist>
void gemm_sve_blocked_prefetch(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __ARM_FEATURE_SVE
    (void)A; (void)B; (void)C;
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K, ldb = N, ldc = N;
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    const std::size_t vl     = (sizeof(T) == 4) ? svcntw() : svcntd();
    const std::size_t kJStep = kSveRegCols * vl;

    for (std::size_t i_blk = 0; i_blk < M; i_blk += kSveTileM) {
        const std::size_t i_end = std::min(i_blk + kSveTileM, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += kSveTileK) {
            const std::size_t k_end = std::min(k_blk + kSveTileK, K);
            const std::size_t k_len = k_end - k_blk;
            if (k_blk + kSveTileK < K)
                pf<0, 2>(B.data() + (k_blk + kSveTileK) * ldb);  // [PF-B]
            for (std::size_t j_blk = 0; j_blk < N; j_blk += kSveTileN) {
                const std::size_t j_end = std::min(j_blk + kSveTileN, N);
                std::size_t i = i_blk;
                for (; i + kSveRegRows <= i_end; i += kSveRegRows) {
                    if (i + kSveRegRows * PfDist < i_end) {
                        pf<0, 2>(A.data() + (i + kSveRegRows * PfDist) * lda + k_blk);  // [PF-A]
                        pf<1, 3>(C.data() + (i + kSveRegRows * PfDist) * ldc + j_blk);  // [PF-C]
                    }
                    const T* a_ptr = A.data() + i * lda + k_blk;
                    const T* b_ptr = B.data() + k_blk * ldb + j_blk;
                    for (std::size_t j = j_blk; j < j_end; j += kJStep) {
                        const std::size_t rem  = (j_end > j) ? j_end - j : 0;
                        const std::size_t rem1 = (rem > vl) ? rem - vl : 0;
                        T* c0 = C.data() + (i + 0) * ldc + j;
                        T* c1 = C.data() + (i + 1) * ldc + j;
                        T* c2 = C.data() + (i + 2) * ldc + j;
                        T* c3 = C.data() + (i + 3) * ldc + j;
                        if constexpr (sizeof(T) == 4) {
                            const svbool_t pg0 = (rem >= vl) ? svptrue_b32()
                                : svwhilelt_b32((uint64_t)0, (uint64_t)rem);
                            const svbool_t pg1 = (rem1 >= vl) ? svptrue_b32()
                                : svwhilelt_b32((uint64_t)0, (uint64_t)rem1);
                            if (svptest_any(svptrue_b32(), pg0))
                                sve_micro_f32_4x2v(reinterpret_cast<const float*>(a_ptr),
                                    reinterpret_cast<const float*>(b_ptr + (j - j_blk)),
                                    reinterpret_cast<float*>(c0), reinterpret_cast<float*>(c1),
                                    reinterpret_cast<float*>(c2), reinterpret_cast<float*>(c3),
                                    lda, ldb, k_len, pg0, pg1);
                        } else {
                            const svbool_t pg0 = (rem >= vl) ? svptrue_b64()
                                : svwhilelt_b64((uint64_t)0, (uint64_t)rem);
                            const svbool_t pg1 = (rem1 >= vl) ? svptrue_b64()
                                : svwhilelt_b64((uint64_t)0, (uint64_t)rem1);
                            if (svptest_any(svptrue_b64(), pg0))
                                sve_micro_f64_4x2v(reinterpret_cast<const double*>(a_ptr),
                                    reinterpret_cast<const double*>(b_ptr + (j - j_blk)),
                                    reinterpret_cast<double*>(c0), reinterpret_cast<double*>(c1),
                                    reinterpret_cast<double*>(c2), reinterpret_cast<double*>(c3),
                                    lda, ldb, k_len, pg0, pg1);
                        }
                    }
                }
                for (; i < i_end; ++i)
                    for (std::size_t k = k_blk; k < k_end; ++k) {
                        const T a_ik = A(i, k);
                        for (std::size_t j = j_blk; j < j_end; ++j) C(i, j) += a_ik * B(k, j);
                    }
            }
        }
    }
#endif
}

}  // namespace hpc::gemm

