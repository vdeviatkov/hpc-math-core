#pragma once

/**
 * @file lowp_reference.hpp
 * @brief Reference for the low-precision GEMMs: C = A · B with bf16 / fp16
 *        inputs and FP32 output, or INT8 inputs and INT32 output.
 *
 * Every input is converted to double (exactly) and the products are summed
 * in double, then rounded once to float; INT8 products are summed in
 * int64_t, which is exact. The kernels accumulate in FP32 or INT32, so they
 * differ from this reference only by FP32 rounding in a different order
 * (floating point) or not at all (INT8, as long as no INT32 sum overflows:
 * K ≤ 133,000 for any INT8 inputs).
 */

#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace hpc::gemm {

/// Output element type of a low-precision GEMM: FP32, or INT32 for INT8.
template <typename In>
using lowp_out_t = std::conditional_t<std::is_same_v<In, std::int8_t>, std::int32_t, float>;

template <typename In>
void gemm_lowp_reference(const Matrix<In>& A, const Matrix<In>& B, Matrix<lowp_out_t<In>>& C) {
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            if constexpr (std::is_same_v<In, std::int8_t>) {
                std::int64_t acc = 0;
                for (std::size_t k = 0; k < K; ++k)
                    acc += std::int64_t{A(i, k)} * std::int64_t{B(k, j)};
                C(i, j) = static_cast<std::int32_t>(acc);
            } else {
                double acc = 0;
                for (std::size_t k = 0; k < K; ++k)
                    acc += double(to_float(A(i, k))) * double(to_float(B(k, j)));
                C(i, j) = static_cast<float>(acc);
            }
        }
}

}  // namespace hpc::gemm
