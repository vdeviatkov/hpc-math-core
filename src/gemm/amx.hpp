#pragma once

/**
 * @file amx.hpp
 * @brief GEMM via Accelerate.framework's BLAS, Apple's route to its AMX
 *        matrix coprocessor.
 *
 * "AMX" names two unrelated accelerators. Intel AMX is a public x86 ISA
 * extension (tile registers + TMUL, Sapphire Rapids+). Apple AMX is a
 * coprocessor in every Apple Silicon SoC since M1, with no public
 * instruction set or intrinsics; its encodings are known only from reverse
 * engineering. The supported way to use it is Accelerate's BLAS
 * (cblas_sgemm / cblas_dgemm), which Apple's guidance recommends for matrix
 * math and which is understood to dispatch to the coprocessor. On M4 it most
 * likely runs on the same matrix unit SME exposes (sme.hpp).
 *
 * So gemm_amx is a thin wrapper around a vendor library, not a hand-written
 * kernel: it measures what Apple's own implementation achieves, as a ceiling
 * for the other kernels. Unlike the other CPU families there is one
 * function, not a naive → reordered → blocked progression — Accelerate has
 * no tiling or blocking knob to stage from the caller's side.
 *
 * Precision: full fp32/fp64 (no reduced-precision inputs, unlike Intel AMX's
 * bf16 or gemm_cuda_wmma's fp16).
 *
 * Threading: Accelerate may use several cores for large matrices, unlike
 * every other CPU kernel here. docs/benchmarks.md has both the default and
 * the single-thread (VECLIB_MAXIMUM_THREADS=1) numbers.
 *
 * Availability: macOS/iOS only (on Intel Macs Accelerate uses AVX instead).
 * HPC_HAS_AMX (hpc/isa.hpp) is 1 when CMake's HPC_ENABLE_AMX (default ON on
 * Apple) found Accelerate.framework; elsewhere gemm_amx is declared
 * `= delete`.
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
