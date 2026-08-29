#pragma once

/**
 * @file amx.hpp
 * @brief GEMM via Apple's AMX coprocessor, accessed through Accelerate.framework.
 *
 * ============================================================
 *  Which "AMX" this is (and which one it isn't)
 * ============================================================
 *
 * "AMX" names two, architecturally unrelated, matrix-multiply accelerators
 * that happen to share an acronym:
 *
 *   - Intel AMX (Advanced Matrix Extensions): x86 tile registers + TMUL,
 *     Sapphire Rapids+ Xeon only, programmed via public <immintrin.h>
 *     intrinsics (_tile_loadd, _tile_dpbf16ps, ...).
 *   - Apple AMX (Apple Matrix coprocessor): a coprocessor block present in
 *     every Apple Silicon SoC since the M1, used internally by Apple's own
 *     libraries. This is Apple AMX — the kernels in this file target it.
 *
 * Apple has never published instruction-level documentation or an ACLE-
 * style intrinsic header for its AMX coprocessor (unlike ARM SME, which IS
 * a public, documented ISA extension — see gemm/sme.hpp). The instruction
 * encodings are known only through third-party reverse engineering, are
 * explicitly unsupported for direct use, and are not something this
 * repository will emit directly. The one Apple-sanctioned, stable way to
 * benefit from the AMX coprocessor's throughput is **Accelerate.framework**
 * — its BLAS implementation (`cblas_sgemm` / `cblas_dgemm`) is Apple's own,
 * and the reverse-engineering community (and Apple's own performance
 * guidance to "use Accelerate for matrix math") indicates it dispatches to
 * AMX blocks where beneficial. This file is therefore a thin, correct,
 * *verified* wrapper around Accelerate's BLAS — not a hand-written
 * tile-multiply kernel.
 *
 * This means gemm_amx_* in this repo answers a different question than
 * every other kernel family: not "how fast can a hand-written GEMM in this
 * style go on this ISA", but "what does Apple's own vendor-tuned
 * implementation achieve, as a ceiling to compare our other kernels
 * against". See the "Naive / Reordered / Blocked are intentionally
 * identical here" section below for why.
 *
 *
 * ============================================================
 *  Verification status
 * ============================================================
 *
 * VERIFIED on Apple M4 Max (macOS, Apple Clang 17) — see README.md for
 * measured GFLOP/s. Accelerate.framework ships in every macOS SDK, so
 * unlike this file's previous Intel-AMX design (which needed a runtime
 * CPUID + Linux kernel permission dance and could not be tested on this
 * arm64 machine at all), there is no meaningful "unsupported hardware"
 * failure mode to guard against beyond "not building for Apple platforms".
 *
 *
 * ============================================================
 *  No precision trade-off
 * ============================================================
 *
 * Unlike Intel AMX (bf16-in/fp32-accumulate only) and gemm_cuda_wmma
 * (fp16-in/fp32-accumulate), Accelerate's cblas_sgemm/cblas_dgemm compute
 * at full fp32/fp64 precision throughout — whatever the AMX coprocessor
 * does internally, it does not force a reduced-precision input format the
 * way Intel AMX's tile-multiply instructions do. So gemm_amx_naive/
 * reordered/blocked support both float AND double at full precision, with
 * no bf16-style relative-error caveat.
 *
 *
 * ============================================================
 *  Naive / Reordered / Blocked are intentionally identical here
 * ============================================================
 *
 * Every other kernel family in this repo progresses through three
 * genuinely different implementations (bad access pattern → cache-friendly
 * → cache-blocked + register-tiled). Accelerate's BLAS is an opaque,
 * already-optimal vendor implementation: it exposes no algorithm-staging
 * knob, no tile-size parameter, nothing to reorder or block from the
 * caller's side. All three gemm_amx_* entry points below call the exact
 * same cblas_sgemm/cblas_dgemm wrapper. They exist as separate, identically
 * named functions purely so this family's benchmarks and tests slot into
 * the same BM_Amx{Naive,Reordered,Blocked} / gemm_amx_{...} naming
 * convention as every other family, for filtering and comparison
 * convenience — not because there are three different algorithms here.
 *
 *
 * ============================================================
 *  Threading note
 * ============================================================
 *
 * Accelerate's BLAS may use multiple CPU cores internally for large
 * matrices (undocumented, size-dependent heuristic) — unlike every other
 * CPU kernel in this repo, which is strictly single-threaded. This makes
 * gemm_amx_* numbers a "best vendor-library throughput on this machine"
 * reference point, not an apples-to-apples single-core comparison against
 * gemm_sme_*, gemm_avx512_*, or gemm_neon_*. See README.md for the measured
 * numbers and this caveat repeated in context.
 *
 *
 * ============================================================
 *  Hardware / platform availability
 * ============================================================
 *
 * Accelerate.framework (and therefore gemm_amx_*'s real path): macOS and
 * iOS only, on Apple Silicon (M1 and later) or Intel Macs (where it
 * dispatches to AVX instead of AMX — still a valid, fast BLAS, just not
 * exercising the AMX coprocessor this file is about).
 * NOT on: Linux, Windows, or any non-Apple platform — falls back to
 * gemm_avx512_blocked (which itself falls back further down the x86/scalar
 * chain).
 *
 * The header detects HPC_HAS_AMX, defined by CMakeLists.txt's
 * HPC_ENABLE_AMX option (default ON on Apple platforms, since linking
 * Accelerate.framework carries none of the SIGILL/runtime-permission risk
 * that gated HPC_ENABLE_AVX512/HPC_ENABLE_SME being opt-in).
 */

#include "gemm/avx512.hpp"  // fallback chain: AMX → AVX-512 → AVX2 → scalar
#include "hpc/matrix.hpp"

#if defined(HPC_HAS_AMX) && defined(__APPLE__)
    #ifndef ACCELERATE_NEW_LAPACK
        #define ACCELERATE_NEW_LAPACK
    #endif
    #include <Accelerate/Accelerate.h>
    #define HPC_AMX_ACTIVE 1
#endif

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace hpc::gemm {

#ifdef HPC_AMX_ACTIVE

/// Row-major C = A * B via Accelerate's single-precision BLAS (cblas_sgemm).
inline void amx_accelerate_gemm(const float* A, const float* B, float* C, std::size_t M,
                                 std::size_t N, std::size_t K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(M),
                static_cast<int>(N), static_cast<int>(K), 1.0f, A, static_cast<int>(K), B,
                static_cast<int>(N), 0.0f, C, static_cast<int>(N));
}

/// Row-major C = A * B via Accelerate's double-precision BLAS (cblas_dgemm).
inline void amx_accelerate_gemm(const double* A, const double* B, double* C, std::size_t M,
                                 std::size_t N, std::size_t K) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(M),
                static_cast<int>(N), static_cast<int>(K), 1.0, A, static_cast<int>(K), B,
                static_cast<int>(N), 0.0, C, static_cast<int>(N));
}

template <typename T>
void amx_dispatch(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_amx_*: T must be float or double");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    amx_accelerate_gemm(A.data(), B.data(), C.data(), M, N, K);
}

#endif  // HPC_AMX_ACTIVE

/**
 * @brief Always-declared runtime AMX availability check (mirrors
 *        hpc::gemm::cuda_device_count() in cuda.hpp). Trivial here — unlike
 *        the Intel-AMX design this replaced, Accelerate.framework carries
 *        no runtime permission handshake — but kept as a function (rather
 *        than a compile-time constant benchmarks read directly) so
 *        bench_gemm.cpp's skip-guard code is identical across ISA families.
 */
inline bool amx_runtime_available() {
#ifdef HPC_AMX_ACTIVE
    return true;
#else
    return false;
#endif
}

// ============================================================================
// gemm_amx_naive / gemm_amx_reordered / gemm_amx_blocked
//
// All three call the identical Accelerate BLAS wrapper — see the file
// header ("Naive / Reordered / Blocked are intentionally identical here")
// for why there is only one real implementation in this family.
// ============================================================================

/// @copydoc amx_dispatch — see file header for why this equals gemm_amx_reordered/_blocked.
template <typename T>
void gemm_amx_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef HPC_AMX_ACTIVE
    gemm_avx512_naive(A, B, C);
#else
    amx_dispatch(A, B, C);
#endif
}

/// @copydoc amx_dispatch — see file header for why this equals gemm_amx_naive/_blocked.
template <typename T>
void gemm_amx_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef HPC_AMX_ACTIVE
    gemm_avx512_reordered(A, B, C);
#else
    amx_dispatch(A, B, C);
#endif
}

/// @copydoc amx_dispatch — see file header for why this equals gemm_amx_naive/_reordered.
template <typename T>
void gemm_amx_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
#ifndef HPC_AMX_ACTIVE
    gemm_avx512_blocked(A, B, C);
#else
    amx_dispatch(A, B, C);
#endif
}

// ---------------------------------------------------------------------------
// Convenience alias: gemm_amx → gemm_amx_blocked
// ---------------------------------------------------------------------------
template <typename T>
inline void gemm_amx(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    gemm_amx_blocked(A, B, C);
}

}  // namespace hpc::gemm
