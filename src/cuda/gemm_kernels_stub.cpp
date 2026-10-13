/**
 * @file gemm_kernels_stub.cpp
 * @brief The CUDA API of gemm/cuda.hpp for builds without a CUDA toolkit.
 *
 * cuda_device_count() returns 0, so tests and benchmarks skip before calling
 * anything else; every other function throws if it is reached anyway.
 */

#include "gemm/cuda.hpp"

#include <cstddef>
#include <stdexcept>

namespace hpc::gemm {

namespace {
[[noreturn]] void unavailable() {
    throw std::runtime_error("built without CUDA support");
}
}  // namespace

int cuda_device_count() noexcept { return 0; }
bool cuda_has_tensor_cores() noexcept { return false; }
bool cuda_has_ampere() noexcept { return false; }

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
void gemm_cuda_mma_ldmatrix(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) {
    unavailable();
}
void gemm_cuda_wmma_pipelined(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) {
    unavailable();
}
void gemm_cuda_cublas_tf32(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) {
    unavailable();
}
void gemm_cuda_cublas_fp16(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) {
    unavailable();
}

void gemm_cuda_cublas_device_f32(const float*, const float*, float*, int, int, int) { unavailable(); }
void gemm_cuda_cublas_tf32_device(const float*, const float*, float*, int, int, int) { unavailable(); }
void gemm_cuda_convert_f32_to_f16_device(const float*, void*, int) { unavailable(); }
void gemm_cuda_cublas_fp16_device(const void*, const void*, float*, int, int, int) { unavailable(); }
void gemm_cuda_wmma_pipelined_device(const void*, const void*, float*, int, int, int) {
    unavailable();
}

void* gemm_cuda_malloc(std::size_t) { unavailable(); }
void gemm_cuda_free(void*) {}  // nothing can have been allocated
void gemm_cuda_memcpy_h2d(void*, const void*, std::size_t) { unavailable(); }
void gemm_cuda_device_synchronize() { unavailable(); }

#define HPC_INSTANTIATE(fn)                                                                  \
    template void fn<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);     \
    template void fn<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
HPC_INSTANTIATE(gemm_cuda_naive)
HPC_INSTANTIATE(gemm_cuda_blocked)
HPC_INSTANTIATE(gemm_cuda_reg_tile)
HPC_INSTANTIATE(gemm_cuda_double_buf)
HPC_INSTANTIATE(gemm_cuda_vectorized)
HPC_INSTANTIATE(gemm_cuda_cublas)
#undef HPC_INSTANTIATE

}  // namespace hpc::gemm
