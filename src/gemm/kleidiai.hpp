#pragma once

/**
 * @file kleidiai.hpp
 * @brief Reference f32 GEMM via Arm KleidiAI's SME2 micro-kernels.
 *
 * KleidiAI (github.com/ARM-software/kleidiai) is Arm's open-source library
 * of hand-written matmul micro-kernels; XNNPACK, llama.cpp, ONNX Runtime
 * and PyTorch use it on SME hardware. It is the closest open reference for
 * gemm_sme (sme.hpp): same hardware, same FMOPA/ZA primitive, but written in
 * assembly by Arm. A gap between the two is the headroom left in
 * gemm_sme's micro-kernel, while a gap to Accelerate also includes Apple's
 * private tuning.
 *
 * Kernel used: kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa
 *   - 2VL×2VL output block per step: all four f32 ZA tiles (2×2), the same
 *     micro-tile shape gemm_sme uses
 *   - LHS packed by kai_lhs_pack_f32p2vlx1_f32_sme (2VL-row strips)
 *   - RHS packed by kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme
 *     (2VL-column strips plus a bias vector; bias = 0 here)
 *   - clamp disabled (±FLT_MAX)
 *
 * Both packing steps run on every call, so timings include them. gemm_sme
 * packs on every call too, which keeps the comparison fair.
 *
 * f32 only. KleidiAI has no f64 matmul, so gemm_kleidiai<double> is deleted.
 * Single-threaded: KleidiAI is a kernel library with no threading of its own.
 *
 * Build: CMake option HPC_ENABLE_KLEIDIAI (default ON when the SME probe
 * passed) fetches KleidiAI with FetchContent. It needs SME2 (Apple M4+).
 */

#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"

#if HPC_HAS_KLEIDIAI
    #include <kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h>
    #include <kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h>
    #include <kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h>

    #include <cfloat>
    #include <vector>
#endif

#include <cassert>
#include <cstddef>

namespace hpc::gemm {

#if !HPC_HAS_KLEIDIAI

template <typename T>
void gemm_kleidiai(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // KleidiAI not built (needs SME2 + HPC_ENABLE_KLEIDIAI=ON)

#else

template <typename T>
void gemm_kleidiai(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) = delete;  // KleidiAI provides f32 matmul only

/// Row-major C = A · B (f32) via KleidiAI's SME2 FMOPA micro-kernel.
template <>
inline void gemm_kleidiai<float>(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);

    const std::size_t mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    const std::size_t nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    const std::size_t kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    const std::size_t sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();

    // Sizes are in bytes.
    const std::size_t lhs_bytes = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr);
    const std::size_t rhs_bytes = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(N, K);
    Matrix<float> lhs_packed(1, lhs_bytes / sizeof(float) + 1);  // 64-byte aligned scratch
    Matrix<float> rhs_packed(1, rhs_bytes / sizeof(float) + 1);
    const std::vector<float> bias(N, 0.0f);

    kai_run_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr, 0, A.data(), K * sizeof(float),
                                       lhs_packed.data());
    kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(1, N, K, nr, kr, sr, N * sizeof(float),
                                                      B.data(), bias.data(), nullptr,
                                                      rhs_packed.data(), 0, nullptr);
    kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
        M, N, K, lhs_packed.data(), rhs_packed.data(), C.data(), N * sizeof(float), sizeof(float),
        -FLT_MAX, FLT_MAX);
}

#endif  // HPC_HAS_KLEIDIAI

}  // namespace hpc::gemm
