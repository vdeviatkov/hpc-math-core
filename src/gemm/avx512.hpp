#pragma once

/**
 * @file avx512.hpp
 * @brief Three progressive AVX-512 + FMA GEMM kernels.
 *
 * ============================================================
 *  Design philosophy: mirrors avx2.hpp exactly, one technique per kernel
 * ============================================================
 *
 * The three kernels are direct AVX-512 counterparts of the AVX2 family,
 * so the benchmark numbers answer precisely: "how much does doubling the
 * SIMD register width (256-bit → 512-bit) contribute at each optimisation
 * level?"
 *
 *  Kernel 1 — gemm_avx512_naive
 *    Loop order: i → j → k  (same as scalar/AVX2 naive)
 *    New vs AVX2 naive: ZMM register holds 16 f32 / 8 f64 instead of 8/4.
 *    Remaining bottleneck: column-stride B access is still a stride-N gather.
 *    SIMD width is irrelevant when every load is a cache miss.
 *    Expected: GFLOP/s ≈ scalar naive — proves wider SIMD still cannot fix
 *    cache-hostile access.
 *
 *  Kernel 2 — gemm_avx512_reordered
 *    Loop order: i → k → j  (same as scalar/AVX2 reordered)
 *    New vs AVX2 reordered: 16 f32 / 8 f64 per FMA instead of 8/4.
 *    B and C accessed stride-1 → every cache line fully consumed.
 *    Theoretical peak vs AVX2: 2× FLOP/cycle for same frequency.
 *    Expected: ~2× AVX2 reordered for f32; ~2× for f64.
 *
 *  Kernel 3 — gemm_avx512_blocked
 *    Loop order: tiled i → k → j  (same as scalar/AVX2 blocked)
 *    New vs AVX2 blocked: micro-kernel register tile is 4×32 f32 / 4×16 f64
 *    instead of 4×16 / 4×8.  4 rows × 2 ZMM per row = 8 accumulators.
 *    Outer L2 tiling unchanged.
 *    Expected: highest GFLOP/s — combines L2-resident tiles with 512-bit FMA.
 *
 *
 * ============================================================
 *  AVX-512 register file and FMA throughput
 * ============================================================
 *
 * AVX-512: 32 × 512-bit ZMM registers.
 *   f32: 16 floats  per ZMM  (64 B — exactly one cache line)
 *   f64:  8 doubles per ZMM  (64 B)
 *
 * FMA: vfmadd231ps / vfmadd231pd  (zmm variants, EVEX prefix)
 *   acc = acc + a * b   (2 FLOP, latency 4 cyc, throughput 0.5 cyc on ICL)
 *
 * Theoretical peak (single core, 3 GHz, Ice Lake):
 *   f32: 2 ports × 16 lanes × 2 FLOP = 64 FLOP/cycle → 192 GFLOP/s
 *   f64: 2 ports ×  8 lanes × 2 FLOP = 32 FLOP/cycle →  96 GFLOP/s
 *
 * Key AVX-512 advantages over AVX2:
 *   • 2× SIMD width → 2× FLOP/cycle (when compute-bound)
 *   • 32 ZMM registers vs 16 YMM → room for larger register tiles without
 *     spilling accumulators to the stack
 *   • Embedded broadcast (vfmadd231ps zmm, zmm, mem{1to16}) eliminates
 *     explicit broadcast instructions for A scalars
 *   • One ZMM load covers a full 64-byte cache line exactly
 *
 * Micro-kernel register tile (f32, 4×32):
 *   f32: 4 rows × 2 ZMM per row = 8 accumulator registers (zmm0..zmm7)
 *        4 broadcasts = zmm8..zmm11
 *        2 B loads    = zmm12..zmm13
 *        Total: 14 of 32 ZMM — leaves 18 free for prefetch / software pipeline
 *
 * Micro-kernel register tile (f64, 4×16):
 *   f64: 4 rows × 2 ZMM per row = 8 accumulator registers (zmm0..zmm7)
 *        same broadcast/load structure
 *        Total: 14 of 32 ZMM
 *
 *
 * ============================================================
 *  Portability guard
 * ============================================================
 *
 * AVX-512F + AVX-512DQ is available on:
 *   Intel: Skylake-SP/X (2017), Cascade Lake, Ice Lake, Rocket Lake,
 *          Alder Lake P-cores (AVX-512 disabled by Intel), Sapphire Rapids
 *   AMD:   Zen 4 (2022) and later
 *   NOT:   Apple Silicon (ARM), pre-Skylake Intel, AMD pre-Zen 4,
 *          Alder Lake (E-cores have no AVX-512; Intel disabled it)
 *
 * The header detects __AVX512F__ at compile time.
 * If absent, all three kernels fall back to their AVX2 equivalents
 * (which themselves fall back to scalar if __AVX2__ is also absent).
 * The binary is always correct, never generates SIGILL.
 *
 * To enable on x86 without -march=native:
 *   cmake -DCMAKE_CXX_FLAGS="-mavx512f -mavx512dq -mfma" ...
 * or use the CMake option:
 *   cmake -DHPC_ENABLE_AVX512=ON ...
 */

#include "hpc/matrix.hpp"

#include "gemm/avx2.hpp"  // fallback implementations + AVX2 tile constants

#ifdef __AVX512F__
    #include <immintrin.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace hpc::gemm {

// ============================================================================
// AVX-512 tile / unroll constants
// ============================================================================

inline constexpr std::size_t kAvx512TileM = 64;   // outer i-tile (same as AVX2)
inline constexpr std::size_t kAvx512TileK = 256;  // outer k-tile (same as AVX2)
inline constexpr std::size_t kAvx512TileN = 512;  // outer j-tile — wider: 512 f32 = 2 KB/row

// Micro-kernel unroll: 4 C rows × 2 ZMM vectors per row.
// f32: 2 × 16 = 32 j-elements per call  (vs 16 in AVX2)
// f64: 2 ×  8 = 16 j-elements per call  (vs  8 in AVX2)
inline constexpr std::size_t kAvx512F32RegRows = 4;
inline constexpr std::size_t kAvx512F32RegCols = 2;  // 2 ZMM → 32 f32
inline constexpr std::size_t kAvx512F64RegRows = 4;
inline constexpr std::size_t kAvx512F64RegCols = 2;  // 2 ZMM → 16 f64

// ============================================================================
// AVX-512 micro-kernels
// ============================================================================

#ifdef __AVX512F__

/**
 * @brief AVX-512 f32 micro-kernel: C[i..i+3][j..j+31] += A[i..i+3][k_blk..k_end) × B[..][j..)
 *
 * Register allocation (14 of 32 ZMM used — AVX-512 has 32 registers):
 *   zmm0..zmm7   — 4 rows × 2 ZMM = 8 C accumulators (C(i+r, j..j+31))
 *   zmm8..zmm11  — broadcast(A(i+r, k)) for rows r=0..3
 *   zmm12..zmm13 — B(k, j..j+15) and B(k, j+16..j+31)
 *
 * Note on embedded broadcast:
 *   Intel AVX-512 supports a memory-source broadcast operand in FMA:
 *     vfmadd231ps zmm_acc, zmm_a_broad, [mem]{1to16}
 *   This encodes broadcast + FMA in a single instruction with no extra register.
 *   Compilers with -O3 often exploit this automatically; writing explicit
 *   _mm512_set1_ps / _mm512_fmadd_ps makes the intent clear and lets us
 *   verify the instruction count.
 *
 * @param a       Pointer to A(i, k_blk) — row-stride lda
 * @param b       Pointer to B(k_blk, j) — row-stride ldb
 * @param c0..c3  Pointers to C(i+0..3, j)
 * @param lda     Leading dimension of A (= K for row-major)
 * @param ldb     Leading dimension of B (= N for row-major)
 * @param k_len   Number of k iterations to process
 */
inline void avx512_micro_f32_4x32(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ c0, float* __restrict__ c1,
                                  float* __restrict__ c2, float* __restrict__ c3, std::size_t lda,
                                  std::size_t ldb, std::size_t k_len) noexcept {
    // Load 4×32 C tile: 8 ZMM accumulators.
    // Each ZMM covers 16 f32 = 64 bytes = 1 cache line.
    __m512 c00 = _mm512_loadu_ps(c0), c01 = _mm512_loadu_ps(c0 + 16);
    __m512 c10 = _mm512_loadu_ps(c1), c11 = _mm512_loadu_ps(c1 + 16);
    __m512 c20 = _mm512_loadu_ps(c2), c21 = _mm512_loadu_ps(c2 + 16);
    __m512 c30 = _mm512_loadu_ps(c3), c31 = _mm512_loadu_ps(c3 + 16);

    for (std::size_t k = 0; k < k_len; ++k) {
        // Load two ZMM vectors of B row k (32 consecutive floats).
        const __m512 b0 = _mm512_loadu_ps(b + k * ldb);
        const __m512 b1 = _mm512_loadu_ps(b + k * ldb + 16);

        // Broadcast A(i+r, k) — _mm512_set1_ps replicates one scalar to all 16 lanes.
        // On Skylake/ICL this compiles to vpbroadcastd + vfmadd or the embedded-broadcast form.
        const __m512 a0 = _mm512_set1_ps(a[0 * lda + k]);
        const __m512 a1 = _mm512_set1_ps(a[1 * lda + k]);
        const __m512 a2 = _mm512_set1_ps(a[2 * lda + k]);
        const __m512 a3 = _mm512_set1_ps(a[3 * lda + k]);

        // 8 FMA instructions — fills both FMA execution ports each cycle.
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c11 = _mm512_fmadd_ps(a1, b1, c11);
        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c21 = _mm512_fmadd_ps(a2, b1, c21);
        c30 = _mm512_fmadd_ps(a3, b0, c30);
        c31 = _mm512_fmadd_ps(a3, b1, c31);
    }

    // Store 4×32 C tile back.
    _mm512_storeu_ps(c0, c00);
    _mm512_storeu_ps(c0 + 16, c01);
    _mm512_storeu_ps(c1, c10);
    _mm512_storeu_ps(c1 + 16, c11);
    _mm512_storeu_ps(c2, c20);
    _mm512_storeu_ps(c2 + 16, c21);
    _mm512_storeu_ps(c3, c30);
    _mm512_storeu_ps(c3 + 16, c31);
}

/**
 * @brief AVX-512 f64 micro-kernel: C[i..i+3][j..j+15] += A[i..i+3][k_blk..k_end) × B[..][j..)
 *
 * Register allocation (14 of 32 ZMM):
 *   zmm0..zmm7   — 8 C accumulators  (4 rows × 2 ZMM, each ZMM = 8 f64)
 *   zmm8..zmm11  — broadcast(A(i+r, k))
 *   zmm12..zmm13 — B(k, j..j+7) and B(k, j+8..j+15)
 */
inline void avx512_micro_f64_4x16(const double* __restrict__ a, const double* __restrict__ b,
                                  double* __restrict__ c0, double* __restrict__ c1,
                                  double* __restrict__ c2, double* __restrict__ c3, std::size_t lda,
                                  std::size_t ldb, std::size_t k_len) noexcept {
    __m512d c00 = _mm512_loadu_pd(c0), c01 = _mm512_loadu_pd(c0 + 8);
    __m512d c10 = _mm512_loadu_pd(c1), c11 = _mm512_loadu_pd(c1 + 8);
    __m512d c20 = _mm512_loadu_pd(c2), c21 = _mm512_loadu_pd(c2 + 8);
    __m512d c30 = _mm512_loadu_pd(c3), c31 = _mm512_loadu_pd(c3 + 8);

    for (std::size_t k = 0; k < k_len; ++k) {
        const __m512d b0 = _mm512_loadu_pd(b + k * ldb);
        const __m512d b1 = _mm512_loadu_pd(b + k * ldb + 8);

        const __m512d a0 = _mm512_set1_pd(a[0 * lda + k]);
        const __m512d a1 = _mm512_set1_pd(a[1 * lda + k]);
        const __m512d a2 = _mm512_set1_pd(a[2 * lda + k]);
        const __m512d a3 = _mm512_set1_pd(a[3 * lda + k]);

        c00 = _mm512_fmadd_pd(a0, b0, c00);
        c01 = _mm512_fmadd_pd(a0, b1, c01);
        c10 = _mm512_fmadd_pd(a1, b0, c10);
        c11 = _mm512_fmadd_pd(a1, b1, c11);
        c20 = _mm512_fmadd_pd(a2, b0, c20);
        c21 = _mm512_fmadd_pd(a2, b1, c21);
        c30 = _mm512_fmadd_pd(a3, b0, c30);
        c31 = _mm512_fmadd_pd(a3, b1, c31);
    }

    _mm512_storeu_pd(c0, c00);
    _mm512_storeu_pd(c0 + 8, c01);
    _mm512_storeu_pd(c1, c10);
    _mm512_storeu_pd(c1 + 8, c11);
    _mm512_storeu_pd(c2, c20);
    _mm512_storeu_pd(c2 + 8, c21);
    _mm512_storeu_pd(c3, c30);
    _mm512_storeu_pd(c3 + 8, c31);
}

#endif  // __AVX512F__

// ============================================================================
// Kernel 1: gemm_avx512_naive  —  i → j → k,  512-bit SIMD on the k-loop
// ============================================================================

/**
 * @brief AVX-512 GEMM with naive i-j-k loop order.
 *
 * Direct counterpart of gemm_avx2_naive with ZMM (512-bit) registers.
 *
 * Loop structure:  for i: for j: for k:  C(i,j) += A(i,k) * B(k,j)
 *
 * AVX-512 change vs AVX2 naive:
 *   16 f32 / 8 f64 elements per FMA instead of 8/4.
 *   B column access pattern is identical — stride-N gather.
 *
 * Expected result: GFLOP/s ≈ scalar naive, ≈ AVX2 naive.
 * The gather overhead and cache-miss rate dominate; wider registers help nothing.
 * This is the control measurement: "does the instruction width matter when
 * you're memory-bandwidth bound on random-stride accesses?"  Answer: No.
 *
 * Falls back to gemm_avx2_naive (then gemm_naive) on non-AVX-512 targets.
 */
template <typename T>
void gemm_avx512_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __AVX512F__
    gemm_avx2_naive(A, B, C);
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_avx512_naive: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    // ZMM SIMD width in elements.
    constexpr std::size_t W = (sizeof(T) == 4) ? 16 : 8;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            if constexpr (sizeof(T) == 4) {
                __m512 acc    = _mm512_setzero_ps();
                std::size_t k = 0;
                for (; k + W <= K; k += W) {
                    // A(i, k..k+15): sequential load from row i.
                    const __m512 a_vec = _mm512_loadu_ps(A.data() + i * lda + k);
                    // B column j: gather 16 elements spaced ldb apart.
                    // Each access is a separate cache line for large N.
                    alignas(64) float b_col[16] = {
                        B.data()[(k + 0) * ldb + j],  B.data()[(k + 1) * ldb + j],
                        B.data()[(k + 2) * ldb + j],  B.data()[(k + 3) * ldb + j],
                        B.data()[(k + 4) * ldb + j],  B.data()[(k + 5) * ldb + j],
                        B.data()[(k + 6) * ldb + j],  B.data()[(k + 7) * ldb + j],
                        B.data()[(k + 8) * ldb + j],  B.data()[(k + 9) * ldb + j],
                        B.data()[(k + 10) * ldb + j], B.data()[(k + 11) * ldb + j],
                        B.data()[(k + 12) * ldb + j], B.data()[(k + 13) * ldb + j],
                        B.data()[(k + 14) * ldb + j], B.data()[(k + 15) * ldb + j],
                    };
                    const __m512 b_vec = _mm512_load_ps(b_col);
                    acc                = _mm512_fmadd_ps(a_vec, b_vec, acc);
                }
                // Horizontal reduce ZMM → scalar.
                float s = _mm512_reduce_add_ps(acc);
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            } else {
                __m512d acc   = _mm512_setzero_pd();
                std::size_t k = 0;
                for (; k + W <= K; k += W) {
                    const __m512d a_vec         = _mm512_loadu_pd(A.data() + i * lda + k);
                    alignas(64) double b_col[8] = {
                        B.data()[(k + 0) * ldb + j], B.data()[(k + 1) * ldb + j],
                        B.data()[(k + 2) * ldb + j], B.data()[(k + 3) * ldb + j],
                        B.data()[(k + 4) * ldb + j], B.data()[(k + 5) * ldb + j],
                        B.data()[(k + 6) * ldb + j], B.data()[(k + 7) * ldb + j],
                    };
                    const __m512d b_vec = _mm512_load_pd(b_col);
                    acc                 = _mm512_fmadd_pd(a_vec, b_vec, acc);
                }
                double s = _mm512_reduce_add_pd(acc);
                for (; k < K; ++k)
                    s += A(i, k) * B(k, j);
                C(i, j) = static_cast<T>(s);
            }
        }
    }
#endif
}

// ============================================================================
// Kernel 2: gemm_avx512_reordered  —  i → k → j,  512-bit SIMD on the j-loop
// ============================================================================

/**
 * @brief AVX-512 GEMM with cache-friendly i-k-j loop order.
 *
 * Direct counterpart of gemm_avx2_reordered with ZMM registers.
 *
 * Loop structure:  for i: for k: a_broad = A(i,k);  for j: C(i,j) += a_broad * B(k,j)
 *
 * AVX-512 change vs AVX2 reordered:
 *   _mm512_set1_ps broadcasts A scalar to 16 lanes (vs 8 in AVX2).
 *   Inner j-loop processes 16 f32 / 8 f64 per FMA instead of 8/4.
 *   B and C accessed stride-1 — every cache line fully consumed.
 *
 * Expected result: ~2× AVX2 reordered GFLOP/s (compute-bound regime).
 * At large N, C row eviction from L1 is identical to AVX2 — the degradation
 * slope matches but starts from a higher baseline.
 *
 * Falls back to gemm_avx2_reordered on non-AVX-512 targets.
 */
template <typename T>
void gemm_avx512_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __AVX512F__
    gemm_avx2_reordered(A, B, C);
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_avx512_reordered: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    constexpr std::size_t W = (sizeof(T) == 4) ? 16 : 8;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            if constexpr (sizeof(T) == 4) {
                const __m512 a_broad = _mm512_set1_ps(A.data()[i * lda + k]);
                const float* b_row   = B.data() + k * ldb;
                float* c_row         = C.data() + i * ldc;

                std::size_t j = 0;
                for (; j + W <= N; j += W) {
                    const __m512 b_vec = _mm512_loadu_ps(b_row + j);
                    const __m512 c_vec = _mm512_loadu_ps(c_row + j);
                    _mm512_storeu_ps(c_row + j, _mm512_fmadd_ps(a_broad, b_vec, c_vec));
                }
                const float a_scalar = A.data()[i * lda + k];
                for (; j < N; ++j)
                    c_row[j] += a_scalar * b_row[j];
            } else {
                const __m512d a_broad = _mm512_set1_pd(A.data()[i * lda + k]);
                const double* b_row   = B.data() + k * ldb;
                double* c_row         = C.data() + i * ldc;

                std::size_t j = 0;
                for (; j + W <= N; j += W) {
                    const __m512d b_vec = _mm512_loadu_pd(b_row + j);
                    const __m512d c_vec = _mm512_loadu_pd(c_row + j);
                    _mm512_storeu_pd(c_row + j, _mm512_fmadd_pd(a_broad, b_vec, c_vec));
                }
                const double a_scalar = A.data()[i * lda + k];
                for (; j < N; ++j)
                    c_row[j] += a_scalar * b_row[j];
            }
        }
    }
#endif
}

// ============================================================================
// Kernel 3: gemm_avx512_blocked  —  tiled i → k → j,  512-bit register tile
// ============================================================================

/**
 * @brief AVX-512 GEMM with cache-blocking and 512-bit register-tiled micro-kernel.
 *
 * Direct counterpart of gemm_avx2_blocked with ZMM registers.
 *
 * Combines all three techniques:
 *   1. i-k-j loop order      (stride-1 access to B and C)
 *   2. 3-level L2 cache tiling (outer tile, same structure as scalar blocked)
 *   3. Register tile: 4 rows × 2 ZMM per row = 4×32 f32 / 4×16 f64 of C
 *      held in registers for the entire k-tile — no C store-reload during k.
 *
 * Working set of the micro-kernel (f32, tile=64×256×512):
 *   A tile: 64 × 256 × 4 B =  64 KB  (fits in L2)
 *   B tile: 256 × 512 × 4 B = 512 KB  (fits in L2 on the benchmark machine)
 *   C tile: 64 × 512 × 4 B = 128 KB  (registers + L2)
 *
 * Register allocation (14 of 32 ZMM):
 *   zmm0..zmm7   — 8 C accumulators (4 rows × 2 ZMM = 4×32 f32)
 *   zmm8..zmm11  — broadcast(A(i+r, k)) for r=0..3
 *   zmm12..zmm13 — B(k, j..j+31) split into two 16-wide ZMM loads
 *   zmm14..zmm31 — free: ideal for software pipelining / prefetch in next step
 *
 * vs gemm_avx512_reordered:
 *   At large N the reordered kernel evicts C row i from L1 between k-iterations.
 *   The register tile retains C(i+0..3, j..j+31) in ZMM for kAvx512TileK steps,
 *   then stores once — reducing L1 store traffic by a factor of kAvx512TileK.
 *
 * Falls back to gemm_avx2_blocked on non-AVX-512 targets.
 */
template <typename T>
void gemm_avx512_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef __AVX512F__
    gemm_avx2_blocked(A, B, C);
#else
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_avx512_blocked: T must be float or double");

    const std::size_t M   = A.rows();
    const std::size_t K   = A.cols();
    const std::size_t N   = B.cols();
    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();

    // ZMM SIMD width and micro-kernel step.
    constexpr std::size_t kSimdW   = (sizeof(T) == 4) ? 16 : 8;
    constexpr std::size_t kRegRows = (sizeof(T) == 4) ? kAvx512F32RegRows : kAvx512F64RegRows;
    constexpr std::size_t kRegCols = (sizeof(T) == 4) ? kAvx512F32RegCols : kAvx512F64RegCols;
    constexpr std::size_t kJStep   = kSimdW * kRegCols;  // 32 f32 or 16 f64 per micro-kernel call

    for (std::size_t i_blk = 0; i_blk < M; i_blk += kAvx512TileM) {
        const std::size_t i_end = std::min(i_blk + kAvx512TileM, M);

        for (std::size_t k_blk = 0; k_blk < K; k_blk += kAvx512TileK) {
            const std::size_t k_end = std::min(k_blk + kAvx512TileK, K);
            const std::size_t k_len = k_end - k_blk;

            for (std::size_t j_blk = 0; j_blk < N; j_blk += kAvx512TileN) {
                const std::size_t j_end = std::min(j_blk + kAvx512TileN, N);

                // --- AVX-512 hot path ---
                std::size_t i = i_blk;
                for (; i + kRegRows <= i_end; i += kRegRows) {
                    const T* a_ptr = A.data() + i * lda + k_blk;
                    const T* b_ptr = B.data() + k_blk * ldb + j_blk;

                    std::size_t j = j_blk;
                    for (; j + kJStep <= j_end; j += kJStep) {
                        T* c0 = C.data() + (i + 0) * ldc + j;
                        T* c1 = C.data() + (i + 1) * ldc + j;
                        T* c2 = C.data() + (i + 2) * ldc + j;
                        T* c3 = C.data() + (i + 3) * ldc + j;
                        if constexpr (sizeof(T) == 4) {
                            avx512_micro_f32_4x32(
                                reinterpret_cast<const float*>(a_ptr),
                                reinterpret_cast<const float*>(b_ptr + (j - j_blk)),
                                reinterpret_cast<float*>(c0), reinterpret_cast<float*>(c1),
                                reinterpret_cast<float*>(c2), reinterpret_cast<float*>(c3), lda,
                                ldb, k_len);
                        } else {
                            avx512_micro_f64_4x16(
                                reinterpret_cast<const double*>(a_ptr),
                                reinterpret_cast<const double*>(b_ptr + (j - j_blk)),
                                reinterpret_cast<double*>(c0), reinterpret_cast<double*>(c1),
                                reinterpret_cast<double*>(c2), reinterpret_cast<double*>(c3), lda,
                                ldb, k_len);
                        }
                    }
                    // Scalar j-tail (j not a multiple of kJStep)
                    for (; j < j_end; ++j)
                        for (std::size_t ii = i; ii < i + kRegRows; ++ii) {
                            T acc{};
                            for (std::size_t k = k_blk; k < k_end; ++k)
                                acc += A(ii, k) * B(k, j);
                            C(ii, j) += acc;
                        }
                }
                // Scalar i-tail (M not a multiple of kRegRows)
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
// Convenience alias: gemm_avx512 → gemm_avx512_blocked
// ---------------------------------------------------------------------------
template <typename T>
inline void gemm_avx512(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    gemm_avx512_blocked(A, B, C);
}

}  // namespace hpc::gemm
