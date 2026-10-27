/**
 * @file gemm_stub.cpp
 * @brief The API of gemm/cuda.hpp for builds without a CUDA toolkit.
 *
 * cuda_device_count() returns 0 and no low-precision format is supported,
 * so tests and benchmarks skip before calling a kernel. The other queries
 * return false and free is a no-op; compute, allocation, copies,
 * synchronization and timing throw if reached anyway.
 */

#include "gemm/cuda.hpp"

#include <cstddef>
#include <stdexcept>

namespace hpc::gemm {

namespace {
/// Every entry point that needs a GPU ends here.
[[noreturn]] void unavailable() { throw std::runtime_error("built without CUDA support"); }
}  // namespace

int cuda_device_count() noexcept { return 0; }
bool cuda_has_tensor_cores() noexcept { return false; }
bool cuda_has_ampere() noexcept { return false; }
bool cuda_has_tma() noexcept { return false; }
bool cuda_lowp_supported(CudaLowp) noexcept { return false; }
int cuda_lowp_k_multiple(CudaLowp format) noexcept {
    return format == CudaLowp::int8 || format == CudaLowp::fp8_e4m3 ? 64 : 32;
}
bool cuda_lowp_fp8_block_scaled() { return false; }

template <typename T>
void gemm_cuda_naive(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) { unavailable(); }
template <typename T>
void gemm_cuda_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) { unavailable(); }
template <typename T>
void gemm_cuda_reg_tile(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) { unavailable(); }
template <typename T>
void gemm_cuda_double_buf(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) { unavailable(); }
template <typename T>
void gemm_cuda_vectorized(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) { unavailable(); }
template <typename T>
void gemm_cuda_cublas(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) { unavailable(); }
void gemm_cuda_wmma(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) { unavailable(); }
void gemm_cuda_mma_ldmatrix(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) { unavailable(); }
void gemm_cuda_wmma_pipelined(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) { unavailable(); }
void gemm_cuda_tma(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) { unavailable(); }
void gemm_cuda_cublas_tf32(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) { unavailable(); }
void gemm_cuda_cublas_fp16(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) { unavailable(); }

void gemm_cuda_bf16(const Matrix<bf16>&, const Matrix<bf16>&, Matrix<float>&, CudaLowpImpl) { unavailable(); }
void gemm_cuda_fp16(const Matrix<fp16>&, const Matrix<fp16>&, Matrix<float>&, CudaLowpImpl) { unavailable(); }
void gemm_cuda_fp16_acc16(const Matrix<fp16>&, const Matrix<fp16>&, Matrix<float>&, CudaLowpImpl) {
    unavailable();
}
void gemm_cuda_i8(const Matrix<std::int8_t>&, const Matrix<std::int8_t>&, Matrix<std::int32_t>&, CudaLowpImpl) {
    unavailable();
}
void gemm_cuda_fp8(const Matrix<fp8_e4m3>&, const Matrix<fp8_e4m3>&, Matrix<float>&, CudaLowpImpl) {
    unavailable();
}

template <typename T>
void gemm_cuda_naive_device(const T*, const T*, T*, int, int, int) { unavailable(); }
template <typename T>
void gemm_cuda_blocked_device(const T*, const T*, T*, int, int, int) { unavailable(); }
template <typename T>
void gemm_cuda_reg_tile_device(const T*, const T*, T*, int, int, int) { unavailable(); }
template <typename T>
void gemm_cuda_double_buf_device(const T*, const T*, T*, int, int, int) { unavailable(); }
template <typename T>
void gemm_cuda_vectorized_device(const T*, const T*, T*, int, int, int) { unavailable(); }
template <typename T>
void gemm_cuda_cublas_device(const T*, const T*, T*, int, int, int) { unavailable(); }
void gemm_cuda_wmma_device(const float*, const float*, float*, int, int, int) { unavailable(); }
void gemm_cuda_mma_ldmatrix_device(const float*, const float*, float*, int, int, int) { unavailable(); }
void gemm_cuda_wmma_pipelined_device(const void*, const void*, float*, int, int, int) { unavailable(); }
void gemm_cuda_tma_device(const void*, const void*, float*, int, int, int) { unavailable(); }
void gemm_cuda_lowp_device(CudaLowp, const void*, const void*, void*, int, int, int) { unavailable(); }
void gemm_cuda_lowp_tma_device(CudaLowp, const void*, const void*, void*, int, int, int) { unavailable(); }
void gemm_cuda_cublas_tf32_device(const float*, const float*, float*, int, int, int) { unavailable(); }
void gemm_cuda_cublas_fp16_device(const void*, const void*, float*, int, int, int) { unavailable(); }
void gemm_cuda_cublas_lowp_device(CudaLowp, const void*, const void*, void*, int, int, int) { unavailable(); }
void gemm_cuda_convert_f32_to_f16_device(const float*, void*, int) { unavailable(); }

double gemm_cuda_time_ms(const std::function<void()>&) { unavailable(); }
void* gemm_cuda_malloc(std::size_t) { unavailable(); }
void gemm_cuda_free(void*) {}  // nothing can have been allocated
void gemm_cuda_memcpy_h2d(void*, const void*, std::size_t) { unavailable(); }
void gemm_cuda_memcpy_d2h(void*, const void*, std::size_t) { unavailable(); }
void gemm_cuda_device_synchronize() { unavailable(); }

#define HPC_CUDA_INSTANTIATE(fn)                                                             \
    template void fn<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);     \
    template void fn<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&); \
    template void fn##_device<float>(const float*, const float*, float*, int, int, int);    \
    template void fn##_device<double>(const double*, const double*, double*, int, int, int);
HPC_CUDA_INSTANTIATE(gemm_cuda_naive)
HPC_CUDA_INSTANTIATE(gemm_cuda_blocked)
HPC_CUDA_INSTANTIATE(gemm_cuda_reg_tile)
HPC_CUDA_INSTANTIATE(gemm_cuda_double_buf)
HPC_CUDA_INSTANTIATE(gemm_cuda_vectorized)
HPC_CUDA_INSTANTIATE(gemm_cuda_cublas)
#undef HPC_CUDA_INSTANTIATE

}  // namespace hpc::gemm
