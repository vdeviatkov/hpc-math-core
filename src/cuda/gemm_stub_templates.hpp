#pragma once
/**
 * @file gemm_stub_templates.hpp
 * @brief Definitions of the parameterised CUDA entry points (gemm/cuda.hpp)
 *        for builds without a CUDA toolkit: every one throws.
 *
 * The CPU-only counterpart of src/cuda/gemm_kernels.cuh. Targets include it
 * instead of the kernels in a stub build and instantiate the same entry
 * points with cuda/gemm_instantiate.hpp, so they link either way; tests and
 * benchmarks skip before calling one (cuda_device_count() returns 0).
 */

#include "gemm/cuda.hpp"

#include <stdexcept>

namespace hpc::gemm {

namespace cuda_stub {
/// Every entry point that needs a GPU ends here.
[[noreturn]] inline void unavailable() { throw std::runtime_error("built without CUDA support"); }
}  // namespace cuda_stub

#define HPC_CUDA_STUB_FMA(fn)                                                                    \
    template <typename P, typename T>                                                            \
    void fn(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) { cuda_stub::unavailable(); }        \
    template <typename P, typename T>                                                            \
    void fn##_device(const T*, const T*, T*, int, int, int) { cuda_stub::unavailable(); }
HPC_CUDA_STUB_FMA(gemm_cuda_naive)
HPC_CUDA_STUB_FMA(gemm_cuda_blocked)
HPC_CUDA_STUB_FMA(gemm_cuda_reg_tile)
HPC_CUDA_STUB_FMA(gemm_cuda_double_buf)
HPC_CUDA_STUB_FMA(gemm_cuda_vectorized)
#undef HPC_CUDA_STUB_FMA

template <typename P>
void gemm_cuda_wmma(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) { cuda_stub::unavailable(); }
template <typename P>
void gemm_cuda_wmma_device(const float*, const float*, float*, int, int, int) { cuda_stub::unavailable(); }
template <typename P>
void gemm_cuda_mma_ldmatrix(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) { cuda_stub::unavailable(); }
template <typename P>
void gemm_cuda_mma_ldmatrix_device(const float*, const float*, float*, int, int, int) { cuda_stub::unavailable(); }
template <typename P>
void gemm_cuda_wmma_pipelined(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) {
    cuda_stub::unavailable();
}
template <typename P>
void gemm_cuda_wmma_pipelined_device(const void*, const void*, float*, int, int, int) { cuda_stub::unavailable(); }
template <typename P>
void gemm_cuda_tma(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) { cuda_stub::unavailable(); }
template <typename P>
void gemm_cuda_tma_device(const void*, const void*, float*, int, int, int) { cuda_stub::unavailable(); }

template <typename P>
void gemm_cuda_lowp_device(CudaLowp, const void*, const void*, void*, int, int, int) { cuda_stub::unavailable(); }
template <typename P>
void gemm_cuda_lowp_tma_device(CudaLowp, const void*, const void*, void*, int, int, int) {
    cuda_stub::unavailable();
}
template <typename P>
void gemm_cuda_bf16(const Matrix<bf16>&, const Matrix<bf16>&, Matrix<float>&, CudaLowpImpl) {
    cuda_stub::unavailable();
}
template <typename P>
void gemm_cuda_fp16(const Matrix<fp16>&, const Matrix<fp16>&, Matrix<float>&, CudaLowpImpl) {
    cuda_stub::unavailable();
}
template <typename P>
void gemm_cuda_fp16_acc16(const Matrix<fp16>&, const Matrix<fp16>&, Matrix<float>&, CudaLowpImpl) {
    cuda_stub::unavailable();
}
template <typename P>
void gemm_cuda_i8(const Matrix<std::int8_t>&, const Matrix<std::int8_t>&, Matrix<std::int32_t>&, CudaLowpImpl) {
    cuda_stub::unavailable();
}
template <typename P>
void gemm_cuda_fp8(const Matrix<fp8_e4m3>&, const Matrix<fp8_e4m3>&, Matrix<float>&, CudaLowpImpl) {
    cuda_stub::unavailable();
}

}  // namespace hpc::gemm
