/**
 * @file gemm_lowp_stub.cpp
 * @brief CPU-only stand-in for src/cuda/gemm_lowp.cu.
 *
 * Built when there is no CUDA toolkit, so code that uses gemm/cuda_lowp.hpp
 * still links. No format is supported and every kernel throws; the tests and
 * benchmarks check cuda_device_count() / cuda_lowp_supported() and skip.
 */
#include "gemm/cuda_lowp.hpp"

#include <stdexcept>

namespace hpc::gemm {

namespace {
[[noreturn]] void no_cuda() { throw std::runtime_error("hpc_math_core was built without CUDA"); }
}  // namespace

bool cuda_lowp_supported(CudaLowp) noexcept { return false; }
int cuda_lowp_k_multiple(CudaLowp format) noexcept {
    return format == CudaLowp::int8 || format == CudaLowp::fp8_e4m3 ? 64 : 32;
}

bool cuda_lowp_fp8_block_scaled() { return false; }

void gemm_cuda_lowp_device(CudaLowp, const void*, const void*, void*, int, int, int) { no_cuda(); }
void gemm_cuda_cublas_lowp_device(CudaLowp, const void*, const void*, void*, int, int, int) { no_cuda(); }

void gemm_cuda_bf16(const Matrix<bf16>&, const Matrix<bf16>&, Matrix<float>&, bool) { no_cuda(); }
void gemm_cuda_fp16(const Matrix<fp16>&, const Matrix<fp16>&, Matrix<float>&, bool) { no_cuda(); }
void gemm_cuda_fp16_acc16(const Matrix<fp16>&, const Matrix<fp16>&, Matrix<float>&, bool) { no_cuda(); }
void gemm_cuda_i8(const Matrix<std::int8_t>&, const Matrix<std::int8_t>&, Matrix<std::int32_t>&, bool) {
    no_cuda();
}
void gemm_cuda_fp8(const Matrix<fp8_e4m3>&, const Matrix<fp8_e4m3>&, Matrix<float>&, bool) { no_cuda(); }

}  // namespace hpc::gemm
