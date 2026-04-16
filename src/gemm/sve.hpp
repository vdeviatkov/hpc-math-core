#pragma once

/**
 * @file sve.hpp
 * @brief Three progressive ARM SVE / SVE2 GEMM kernels.
 *
 * ============================================================
 *  Why SVE is fundamentally different from NEON and AVX
 * ============================================================
 *
 * NEON Q-registers are always 128-bit wide.
 * AVX2 YMM-registers are always 256-bit wide.
 * AVX-512 ZMM-registers are always 512-bit wide.
 *
 * SVE (Scalable Vector Extension, ARMv8.2-A, 2016) breaks this model:
 *   • The hardware vector length VL is implementation-defined,
 *     ranging from 128 bits to 2048 bits in 128-bit increments.
 *   • VL is NOT known at compile time.  It is queried at runtime via:
 *       svcntw()  — number of 32-bit (float)  elements per vector
 *       svcntd()  — number of 64-bit (double) elements per vector
 *   • A single binary compiled with -march=armv8.2-a+sve runs correctly
 *     on 128-bit SVE (Neoverse N1), 256-bit SVE (Neoverse V1, Fugaku),
 *     512-bit SVE (A64FX), or 2048-bit SVE (future), without recompilation.
 *   • This is "vector-length agnostic" (VLA) programming — the loop step
 *     is computed from svcntw()/svcntd(), not a compile-time constant.
 *
 * SVE2 (ARMv9.0-A, 2021) is a superset of SVE that adds:
 *   • Matrix outer-product instructions (FMOPA / FMOPS) via SME extension
 *   • More complex integer/polynomial operations
 *   For GEMM purposes, SVE and SVE2 are equivalent; the kernels below
 *   target SVE (available on Neoverse V1, A64FX, some Cortex-X cores).
 *
 *
 * ============================================================
 *  SVE predicate registers
 * ============================================================
 *
 * Every SVE load/store/FMA operates under a predicate register (p0..p15)
 * that masks individual lanes.  This elegantly handles loop tails:
 *   instead of a scalar fallback loop, the last iteration uses a predicate
 *   that has 1s only for the remaining elements.
 *
 *   svwhilelt_b32(i, N)  — creates a predicate with 1 in lane k iff (i+k < N)
 *   svptrue_b32()        — all-true predicate (all lanes active)
 *
 * This means SVE kernels have NO scalar tail loops for the j-dimension —
 * the predicated last iteration handles any N cleanly.
 *
 *
 * ============================================================
 *  Key SVE intrinsics used in these kernels
 * ============================================================
 *
 *  svfloat32_t, svfloat64_t   — scalable vector types (VL × f32/f64)
 *  svbool_t                   — predicate vector
 *
 *  svptrue_b32/b64()          — all-lanes-active predicate
 *  svwhilelt_b32/b64(i, N)    — tail predicate: lane k active iff i+k < N
 *
 *  svld1_f32/f64(pg, ptr)     — predicated load
 *  svst1_f32/f64(pg, ptr, v)  — predicated store
 *  svdup_n_f32/f64(s)         — broadcast scalar to all lanes
 *  svmla_f32/f64_x(pg, a, b, c) — a = a + b * c  (FMA, pg-controlled)
 *
 *  svcntw() / svcntd()        — runtime VL query (elements per vector)
 *
 *  svaddv_f32/f64(pg, v)      — horizontal reduce → scalar (for naive kernel)
 *
 *
 * ============================================================
 *  Three kernels — same pedagogical structure as AVX2 / NEON
 * ============================================================
 *
 *  Kernel 1 — gemm_sve_naive
 *    Loop order: i → j → k
 *    SIMD on k-loop: accumulate A row × B column using VL-wide FMA.
 *    B access: stride-N column gather — fill a vl-element heap buffer
 *    (B(k,j)..B(k+vl-1,j)) then load as a contiguous SVE vector.
 *    No scalar tail: svaddv reduces the final accumulator to a scalar.
 *    Expected: GFLOP/s ≈ scalar naive (bandwidth-bound from gather).
 *
 *  Kernel 2 — gemm_sve_reordered
 *    Loop order: i → k → j
 *    SIMD on j-loop: broadcast A(i,k), VL-wide FMA against B row / C row.
 *    No scalar j-tail: svwhilelt predicate handles remainder lanes.
 *    Expected: ~VL/sizeof(T) × scalar reordered.  On 256-bit SVE:
 *      f32: 8 lanes ≈ 2× NEON, ~8× scalar
 *      f64: 4 lanes ≈ 2× NEON, ~4× scalar
 *
 *  Kernel 3 — gemm_sve_blocked
 *    Loop order: tiled i → k → j
 *    Outer L2 blocking + inner VLA register tile.
 *    Unlike NEON/AVX the tile width (kJStep) is computed at runtime
 *    from svcntw()/svcntd(), so the micro-kernel automatically scales
 *    with the hardware VL.
 *    Expected: highest GFLOP/s on SVE hardware.
 *
 *
 * ============================================================
 *  Hardware availability
 * ============================================================
 *
 * SVE is available on:
 *   AWS Graviton3/4 (256-bit, Neoverse V1/V2)
 *   Fujitsu A64FX   (512-bit, used in Fugaku supercomputer)
 *   ARM Neoverse N2 (256-bit)
 *   Some Cortex-X3/X4 mobile cores (128-bit)
 *   NOT on: Apple Silicon (M-series uses NEON only, not SVE)
 *   NOT on: x86 (Intel/AMD)
 *
 * The header detects __ARM_FEATURE_SVE at compile time.
 * If absent, all three kernels fall back to their NEON equivalents
 * (which themselves fall back to AVX2, then scalar).
 *
 * To compile with SVE on GCC/Clang targeting Graviton3:
 *   -march=armv8.2-a+sve   (explicit)
 *   -march=neoverse-v1     (CPU-specific, enables SVE automatically)
 *   -march=native          (auto-detects on the build machine)
 *
 * To compile with SVE2:
 *   -march=armv9-a+sve2
 */

#include "gemm/neon.hpp"  // fallback chain: SVE → NEON → AVX2 → scalar
#include "hpc/matrix.hpp"

#ifdef __ARM_FEATURE_SVE
    #include <arm_sve.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace hpc::gemm {

// ============================================================================
// SVE tile constants
// ============================================================================
// Unlike NEON/AVX, the j-step width is NOT a compile-time constant —
// it depends on svcntw()/svcntd() which vary per CPU.  The outer blocking
// tile dimensions (M, K) are fixed at values that work well across all
// known SVE implementations (128-bit through 512-bit).

inline constexpr std::size_t kSveTileM = 64;   // outer i-tile (rows of A / C)
inline constexpr std::size_t kSveTileK = 256;  // outer k-tile (contraction width)
inline constexpr std::size_t kSveTileN = 512;  // outer j-tile (cols of B / C)
                                               // 512 × 4B = 2 KB per B row-tile;
                                               // on 256-bit SVE (8 f32/vec) this is
                                               // 64 vectors — well within L1 TLB.

// Number of C rows accumulated simultaneously in the blocked micro-kernel.
// Fixed at 4: 4 rows × 2 vectors × VL floats each.  With VL=256b this is
// 4 × 2 × 8 = 64 f32 = 256 B of C held in registers — same as AVX2 blocked.
inline constexpr std::size_t kSveRegRows = 4;
// Number of SVE vectors per C row in the micro-kernel.
// 2 vectors × VL elements = 2*svcntw() f32 or 2*svcntd() f64 per row.
inline constexpr std::size_t kSveRegCols = 2;

// ============================================================================
// SVE micro-kernels (used only by gemm_sve_blocked)
// ============================================================================

#ifdef __ARM_FEATURE_SVE

/**
 * @brief SVE f32 micro-kernel: C[i..i+3][j..j+2*VL) += A[i..i+3][k_blk..k_end) × B[..][j..)
 *
 * Register tile: 4 rows × 2 SVE vectors = 4 × (2 * svcntw()) f32.
 *
 * On 256-bit SVE (VL=8 f32): tile = 4 × 16 f32 = 64 B  (same as AVX2)
 * On 512-bit SVE (VL=16 f32): tile = 4 × 32 f32 = 128 B
 * On 128-bit SVE (VL=4 f32): tile = 4 × 8 f32  = 32 B
 *
 * The predicate pg is all-true for full-width iterations; the caller passes
 * svptrue_b32() for all tiles except the j-tail where svwhilelt_b32 is used.
 *
 * Instruction note: svmla_f32_x(pg, acc, a, b)
 *   acc = acc + a * b  (FMA, predicated by pg, "don't care" for inactive lanes)
 *   Using _x (don't-care) rather than _z (zero) or _m (merge) because
 *   inactive lanes hold stale accumulator values that we don't want zeroed.
 *
 * @param a      A(i, k_blk) — row stride lda
 * @param b0     B(k_blk, j) — first SVE-width block
 * @param b1     B(k_blk, j + svcntw()) — second SVE-width block
 * @param c0..c3 C(i+0..3, j)
 * @param pg0/1  Predicates for the two B/C vector blocks (all-true or tail)
 */
inline void sve_micro_f32_4x2v(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ c0, float* __restrict__ c1,
                               float* __restrict__ c2, float* __restrict__ c3, std::size_t lda,
                               std::size_t ldb, std::size_t k_len, svbool_t pg0,
                               svbool_t pg1) noexcept {
    const std::uint64_t vl = svcntw();  // elements per SVE vector (runtime)

    // Load 4×2 SVE-vector C tile.
    svfloat32_t c00 = svld1_f32(pg0, c0), c01 = svld1_f32(pg1, c0 + vl);
    svfloat32_t c10 = svld1_f32(pg0, c1), c11 = svld1_f32(pg1, c1 + vl);
    svfloat32_t c20 = svld1_f32(pg0, c2), c21 = svld1_f32(pg1, c2 + vl);
    svfloat32_t c30 = svld1_f32(pg0, c3), c31 = svld1_f32(pg1, c3 + vl);

    for (std::size_t k = 0; k < k_len; ++k) {
        // Load 2 SVE vectors of B row k.
        const svfloat32_t b0 = svld1_f32(pg0, b + k * ldb);
        const svfloat32_t b1 = svld1_f32(pg1, b + k * ldb + vl);

        // Broadcast A(i+r, k) to all lanes — single scalar → full vector.
        const svfloat32_t a0 = svdup_n_f32(a[0 * lda + k]);
        const svfloat32_t a1 = svdup_n_f32(a[1 * lda + k]);
        const svfloat32_t a2 = svdup_n_f32(a[2 * lda + k]);
        const svfloat32_t a3 = svdup_n_f32(a[3 * lda + k]);

        // 8 predicated FMA instructions.
        c00 = svmla_f32_x(pg0, c00, a0, b0);
        c01 = svmla_f32_x(pg1, c01, a0, b1);
        c10 = svmla_f32_x(pg0, c10, a1, b0);
        c11 = svmla_f32_x(pg1, c11, a1, b1);
        c20 = svmla_f32_x(pg0, c20, a2, b0);
        c21 = svmla_f32_x(pg1, c21, a2, b1);
        c30 = svmla_f32_x(pg0, c30, a3, b0);
        c31 = svmla_f32_x(pg1, c31, a3, b1);
    }

    // Store 4×2 SVE-vector C tile back.
    svst1_f32(pg0, c0, c00);
    svst1_f32(pg1, c0 + vl, c01);
    svst1_f32(pg0, c1, c10);
    svst1_f32(pg1, c1 + vl, c11);
    svst1_f32(pg0, c2, c20);
    svst1_f32(pg1, c2 + vl, c21);
    svst1_f32(pg0, c3, c30);
    svst1_f32(pg1, c3 + vl, c31);
}

/**
 * @brief SVE f64 micro-kernel: C[i..i+3][j..j+2*VL) += A[i..i+3][k_blk..k_end) × B[..][j..)
 *
 * Register tile: 4 rows × 2 SVE vectors = 4 × (2 * svcntd()) f64.
 * Same structure as f32 but using f64 intrinsics and svcntd() for VL.
 */
inline void sve_micro_f64_4x2v(const double* __restrict__ a, const double* __restrict__ b,
                               double* __restrict__ c0, double* __restrict__ c1,
                               double* __restrict__ c2, double* __restrict__ c3, std::size_t lda,
                               std::size_t ldb, std::size_t k_len, svbool_t pg0,
                               svbool_t pg1) noexcept {
    const std::uint64_t vl = svcntd();

    svfloat64_t c00 = svld1_f64(pg0, c0), c01 = svld1_f64(pg1, c0 + vl);
    svfloat64_t c10 = svld1_f64(pg0, c1), c11 = svld1_f64(pg1, c1 + vl);
    svfloat64_t c20 = svld1_f64(pg0, c2), c21 = svld1_f64(pg1, c2 + vl);
    svfloat64_t c30 = svld1_f64(pg0, c3), c31 = svld1_f64(pg1, c3 + vl);

    for (std::size_t k = 0; k < k_len; ++k) {
        const svfloat64_t b0 = svld1_f64(pg0, b + k * ldb);
        const svfloat64_t b1 = svld1_f64(pg1, b + k * ldb + vl);

        const svfloat64_t a0 = svdup_n_f64(a[0 * lda + k]);
        const svfloat64_t a1 = svdup_n_f64(a[1 * lda + k]);
        const svfloat64_t a2 = svdup_n_f64(a[2 * lda + k]);
        const svfloat64_t a3 = svdup_n_f64(a[3 * lda + k]);

        c00 = svmla_f64_x(pg0, c00, a0, b0);
        c01 = svmla_f64_x(pg1, c01, a0, b1);
        c10 = svmla_f64_x(pg0, c10, a1, b0);
        c11 = svmla_f64_x(pg1, c11, a1, b1);
        c20 = svmla_f64_x(pg0, c20, a2, b0);
        c21 = svmla_f64_x(pg1, c21, a2, b1);
        c30 = svmla_f64_x(pg0, c30, a3, b0);
        c31 = svmla_f64_x(pg1, c31, a3, b1);
    }

    svst1_f64(pg0, c0, c00);
    svst1_f64(pg1, c0 + vl, c01);
    svst1_f64(pg0, c1, c10);
    svst1_f64(pg1, c1 + vl, c11);
    svst1_f64(pg0, c2, c20);
    svst1_f64(pg1, c2 + vl, c21);
    svst1_f64(pg0, c3, c30);
    svst1_f64(pg1, c3 + vl, c31);
}

#endif  // __ARM_FEATURE_SVE

// ============================================================================
// Kernel 1: gemm_sve_naive  —  i → j → k,  SVE on the k-loop
// ============================================================================

/**
 * @brief SVE GEMM with naive i-j-k loop order.
 *
 * Direct SVE counterpart of gemm_neon_naive / gemm_avx2_naive.
 *
 * Loop structure:  for i: for j: for k:  C(i,j) += A(i,k) * B(k,j)
 *
 * SVE specifics vs NEON naive:
 *   • Inner k-loop step = svcntw() f32 / svcntd() f64 (runtime VL).
 *   • B column access B(k..k+VL-1, j) is stride-ldb gather — cache-hostile.
 *     On SVE this is done with a manual scalar load into a VLA-allocated
 *     stack array, then svld1.  (SVE does have gather-load instructions
 *     via svld1_gather_index, but the memory access pattern is identical —
 *     each element is a cache miss at large N.)
 *   • Final tail: svaddv_f32/f64 reduces the accumulated vector to a scalar
 *     in a single instruction.  No scalar fallback needed.
 *
 * Expected result: GFLOP/s ≈ scalar naive — gather saturates bandwidth.
 * Pedagogical purpose: confirms that VLA makes no difference for
 * cache-hostile access patterns (same lesson as AVX2/NEON naive).
 *
 * Falls back to gemm_neon_naive on non-SVE targets.
 */
template <typename T>
void gemm_sve_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __ARM_FEATURE_SVE
    gemm_neon_naive(A, B, C);
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sve_naive: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    if constexpr (sizeof(T) == 4) {
        const std::size_t vl = svcntw();  // elements per SVE vector (runtime)
        // VLA stack buffer: vl floats for gathering one B column segment.
        // Using std::vector avoids a VLA (which is a GCC extension, not C++20).
        std::vector<float> b_col(vl);
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                svfloat32_t acc = svdup_n_f32(0.f);
                std::size_t k   = 0;
                for (; k + vl <= K; k += vl) {
                    // A(i, k..k+vl): sequential load — cache friendly.
                    const svfloat32_t a_vec = svld1_f32(svptrue_b32(), A.data() + i * lda + k);
                    // Gather B column j: fill buffer element by element,
                    // then load as a contiguous SVE vector.
                    for (std::size_t lane = 0; lane < vl; ++lane)
                        b_col[lane] = B.data()[(k + lane) * ldb + j];
                    const svfloat32_t b_vec = svld1_f32(svptrue_b32(), b_col.data());
                    acc = svmla_f32_x(svptrue_b32(), acc, a_vec, b_vec);
                }
                // Horizontal reduce the SVE accumulator to a scalar.
                float s = svaddv_f32(svptrue_b32(), acc);
                // Scalar tail for k % vl elements.
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            }
        }
    } else {
        const std::size_t vl = svcntd();
        std::vector<double> b_col(vl);
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                svfloat64_t acc = svdup_n_f64(0.0);
                std::size_t k   = 0;
                for (; k + vl <= K; k += vl) {
                    const svfloat64_t a_vec = svld1_f64(svptrue_b64(), A.data() + i * lda + k);
                    for (std::size_t lane = 0; lane < vl; ++lane)
                        b_col[lane] = B.data()[(k + lane) * ldb + j];
                    const svfloat64_t b_vec = svld1_f64(svptrue_b64(), b_col.data());
                    acc = svmla_f64_x(svptrue_b64(), acc, a_vec, b_vec);
                }
                double s = svaddv_f64(svptrue_b64(), acc);
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            }
        }

    }
#endif
}

// ============================================================================
// Kernel 2: gemm_sve_reordered  —  i → k → j,  SVE on the j-loop (VLA)
// ============================================================================

/**
 * @brief SVE GEMM with cache-friendly i-k-j loop order — vector-length agnostic.
 *
 * Direct SVE counterpart of gemm_neon_reordered / gemm_avx2_reordered.
 *
 * Loop structure:  for i: for k: a_broad = A(i,k);  for j: C(i,j) += a_broad * B(k,j)
 *
 * SVE specifics vs NEON reordered:
 *   • j-loop step = svcntw() / svcntd() — determined at runtime.
 *   • No scalar j-tail: the final (partial) iteration uses
 *     svwhilelt_b32(j, N) which produces a predicate with 1s only for
 *     in-bounds lanes.  svld1 / svmla / svst1 with this predicate handle
 *     the tail exactly, without any scalar fallback code.
 *   • This is the key SVE elegance: one predicated loop covers all N.
 *
 * Expected result: ~svcntw()/svcntd() × scalar reordered.  On 256-bit SVE:
 *   f32: 8 lanes → ~8× scalar reordered (≈ 2× NEON, same as AVX2)
 *   f64: 4 lanes → ~4× scalar reordered (≈ 2× NEON, same as AVX2)
 *
 * Falls back to gemm_neon_reordered on non-SVE targets.
 */
template <typename T>
void gemm_sve_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __ARM_FEATURE_SVE
    gemm_neon_reordered(A, B, C);
#else
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
#endif
}

// ============================================================================
// Kernel 3: gemm_sve_blocked  —  tiled i → k → j,  VLA register tile
// ============================================================================

/**
 * @brief SVE GEMM with cache-blocking and VLA register-tiled micro-kernel.
 *
 * Direct SVE counterpart of gemm_neon_blocked / gemm_avx2_blocked.
 *
 * Combines all three techniques with SVE-specific adaptations:
 *   1. i-k-j loop order      (stride-1 B and C access)
 *   2. 3-level L2 cache tiling (kSveTileM × kSveTileK × kSveTileN)
 *   3. VLA register tile:
 *      - Tile width = kSveRegCols × svcntw/d() — scales with hardware VL.
 *      - 4 rows × 2 SVE vectors of C held in scalable registers for the
 *        full k-tile, eliminating C load/store traffic during k-iteration.
 *      - j-tail handled entirely by predicates in the micro-kernel —
 *        no separate scalar tail loop.
 *
 * Working set (256-bit SVE, f32, default tile sizes):
 *   A tile: 64 × 256 × 4 B  =  64 KB  (L2 resident)
 *   B tile: 256 × 512 × 4 B = 512 KB  (L2 resident on Neoverse V1: 1 MB L2)
 *   C tile: 4 rows in registers + 64 × 512 × 4 B = 128 KB streamed
 *
 * Portability: because kJStep = kSveRegCols × svcntw/d() is computed at
 * runtime, the same binary delivers correct and efficient code on:
 *   128-bit SVE (kJStep = 2×4  =  8 f32)
 *   256-bit SVE (kJStep = 2×8  = 16 f32)  — Graviton3, Neoverse V1
 *   512-bit SVE (kJStep = 2×16 = 32 f32)  — A64FX (Fugaku)
 *   2048-bit SVE(kJStep = 2×64 =128 f32)  — future SVE2 / SME hardware
 *
 * Falls back to gemm_neon_blocked on non-SVE targets.
 */
template <typename T>
void gemm_sve_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __ARM_FEATURE_SVE
    gemm_neon_blocked(A, B, C);
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_sve_blocked: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    // VL is queried once here; it is guaranteed constant for the process lifetime.
    const std::size_t vl     = (sizeof(T) == 4) ? svcntw() : svcntd();
    const std::size_t kJStep = kSveRegCols * vl;  // j-elements per micro-kernel call

    for (std::size_t i_blk = 0; i_blk < M; i_blk += kSveTileM) {
        const std::size_t i_end = std::min(i_blk + kSveTileM, M);

        for (std::size_t k_blk = 0; k_blk < K; k_blk += kSveTileK) {
            const std::size_t k_end = std::min(k_blk + kSveTileK, K);
            const std::size_t k_len = k_end - k_blk;

            for (std::size_t j_blk = 0; j_blk < N; j_blk += kSveTileN) {
                const std::size_t j_end = std::min(j_blk + kSveTileN, N);

                // --- SVE hot path: kSveRegRows rows × kJStep cols per call ---
                std::size_t i = i_blk;
                for (; i + kSveRegRows <= i_end; i += kSveRegRows) {
                    const T* a_ptr = A.data() + i * lda + k_blk;
                    const T* b_ptr = B.data() + k_blk * ldb + j_blk;

                    std::size_t j = j_blk;
                    for (; j < j_end; j += kJStep) {
                        // Compute remaining j elements to build predicates.
                        const std::size_t rem0 = (j_end > j) ? j_end - j : 0;
                        const std::size_t rem1 = (rem0 > vl) ? rem0 - vl : 0;

                        T* c0 = C.data() + (i + 0) * ldc + j;
                        T* c1 = C.data() + (i + 1) * ldc + j;
                        T* c2 = C.data() + (i + 2) * ldc + j;
                        T* c3 = C.data() + (i + 3) * ldc + j;

                        if constexpr (sizeof(T) == 4) {
                            // pg0: first VL elements — full if rem0 >= vl, else tail.
                            const svbool_t pg0 = (rem0 >= vl) ? svptrue_b32()
                                                              : svwhilelt_b32((std::uint64_t)0,
                                                                              (std::uint64_t)rem0);
                            // pg1: second VL elements — full if rem1 >= vl, else tail.
                            const svbool_t pg1 = (rem1 >= vl) ? svptrue_b32()
                                                              : svwhilelt_b32((std::uint64_t)0,
                                                                              (std::uint64_t)rem1);

                            // Only call the micro-kernel if the first block has work.
                            if (svptest_any(svptrue_b32(), pg0)) {
                                sve_micro_f32_4x2v(
                                    reinterpret_cast<const float*>(a_ptr),
                                    reinterpret_cast<const float*>(b_ptr + (j - j_blk)),
                                    reinterpret_cast<float*>(c0), reinterpret_cast<float*>(c1),
                                    reinterpret_cast<float*>(c2), reinterpret_cast<float*>(c3), lda,
                                    ldb, k_len, pg0, pg1);
                            }
                        } else {
                            const svbool_t pg0 = (rem0 >= vl) ? svptrue_b64()
                                                              : svwhilelt_b64((std::uint64_t)0,
                                                                              (std::uint64_t)rem0);
                            const svbool_t pg1 = (rem1 >= vl) ? svptrue_b64()
                                                              : svwhilelt_b64((std::uint64_t)0,
                                                                              (std::uint64_t)rem1);

                            if (svptest_any(svptrue_b64(), pg0)) {
                                sve_micro_f64_4x2v(
                                    reinterpret_cast<const double*>(a_ptr),
                                    reinterpret_cast<const double*>(b_ptr + (j - j_blk)),
                                    reinterpret_cast<double*>(c0), reinterpret_cast<double*>(c1),
                                    reinterpret_cast<double*>(c2), reinterpret_cast<double*>(c3),
                                    lda, ldb, k_len, pg0, pg1);
                            }
                        }
                    }
                }

                // Scalar i-tail (M not a multiple of kSveRegRows).
                for (; i < i_end; ++i)
                    for (std::size_t k = k_blk; k < k_end; ++k) {
                        const T a_ik = A(i, k);
                        for (std::size_t j = j_blk; j < j_end; ++j)
                            C(i, j) += a_ik * B(k, j);
                    }
            }
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Convenience alias: gemm_sve → gemm_sve_blocked
// ---------------------------------------------------------------------------
template <typename T>
inline void gemm_sve(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    gemm_sve_blocked(A, B, C);
}

}  // namespace hpc::gemm
