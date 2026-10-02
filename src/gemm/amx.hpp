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
 * This means gemm_amx answers a different question than every other kernel
 * family: not "how fast can a hand-written GEMM in this style go on this
 * ISA", but "what does Apple's own vendor-tuned implementation achieve, as a
 * ceiling to compare our other kernels against".
 *
 *
 * ============================================================
 *  Verification status
 * ============================================================
 *
 * Verified on Apple M4 Max (macOS, Apple Clang 17) — see docs/benchmarks.md
 * for measured GFLOP/s. Accelerate.framework ships in every macOS SDK, so
 * unlike Intel AMX (runtime CPUID + Linux kernel permission request) there
 * is no "unsupported hardware" failure mode to guard against beyond "not
 * building for Apple platforms".
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
 * way Intel AMX's tile-multiply instructions do. So gemm_amx supports both
 * float AND double at full precision, with no bf16-style relative-error
 * caveat.
 *
 *
 * ============================================================
 *  One function, not three
 * ============================================================
 *
 * Every other CPU family in this repo progresses through genuinely
 * different implementations (bad access pattern -> cache-friendly ->
 * cache-blocked + register-tiled). Accelerate's BLAS is an opaque vendor
 * implementation with no tile size, blocking or packing knob to stage from
 * the caller's side, so this family is a single gemm_amx.
 *
 *
 * ============================================================
 *  Threading note
 * ============================================================
 *
 * Accelerate's BLAS may use multiple CPU cores internally for large
 * matrices (undocumented, size-dependent heuristic) — unlike every other
 * CPU kernel in this repo, which is strictly single-threaded. This makes
 * gemm_amx numbers a "best vendor-library throughput on this machine"
 * reference point, not an apples-to-apples single-core comparison against
 * gemm_sme, gemm_avx512_*, or gemm_neon_*. See docs/benchmarks.md for the
 * measured numbers, including a single-thread comparison
 * (VECLIB_MAXIMUM_THREADS=1).
 *
 *
 * ============================================================
 *  Hardware / platform availability
 * ============================================================
 *
 * Accelerate.framework (and therefore gemm_amx's real path): macOS and
 * iOS only, on Apple Silicon (M1 and later) or Intel Macs (where it
 * dispatches to AVX instead of AMX — still a valid, fast BLAS, just not
 * exercising the AMX coprocessor this file is about).
 * NOT on: Linux, Windows, or any non-Apple platform.
 *
 * The header uses HPC_HAS_AMX from hpc/isa.hpp, which is 1 only when
 * CMakeLists.txt's HPC_ENABLE_AMX option found Accelerate.framework
 * (default ON on Apple platforms, since linking Accelerate carries none of
 * the SIGILL/runtime-permission risk that keeps HPC_ENABLE_AVX512 and
 * HPC_ENABLE_SME opt-in). Where it is 0 gemm_amx is declared
 * `= delete`: calling them is a compile-time error, never a silent
 * substitution of a different kernel under the AMX name.
 */

#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"

#if HPC_HAS_AMX
    #ifndef ACCELERATE_NEW_LAPACK
        #define ACCELERATE_NEW_LAPACK
    #endif
    #include <Accelerate/Accelerate.h>
#endif

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace hpc::gemm {

#if !HPC_HAS_AMX

template <typename T>
void gemm_amx(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // Accelerate.framework not available on this target

#else

/// Row-major C = A * B via Accelerate's BLAS (cblas_sgemm / cblas_dgemm).
template <typename T>
void gemm_amx(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "gemm_amx: T must be float or double");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    if constexpr (std::is_same_v<T, float>)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, A.data(), k,
                    B.data(), n, 0.0f, C.data(), n);
    else
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A.data(), k,
                    B.data(), n, 0.0, C.data(), n);
}

#endif  // HPC_HAS_AMX

}  // namespace hpc::gemm
