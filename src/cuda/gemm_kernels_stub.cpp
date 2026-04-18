/**
 * @file gemm_kernels_stub.cpp
 * @brief CPU-only stub for CUDA GEMM symbols.
 *
 * Built when no CUDA toolkit is found. All kernel functions throw
 * std::runtime_error if called (they never will be because cuda_device_count()
 * returns 0 and the benchmarks/tests skip on that condition).
 */

#include "gemm/cuda.hpp"
#include "hpc/matrix.hpp"
#include <stdexcept>

namespace hpc::gemm {

int  cuda_device_count()     noexcept { return 0; }
bool cuda_has_tensor_cores() noexcept { return false; }
bool cuda_has_ampere()       noexcept { return false; }

template <typename T>
void gemm_cuda_naive(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) {
    throw std::runtime_error("CUDA not available: built without CUDA support");
}
template <typename T>
void gemm_cuda_reordered(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) {
    throw std::runtime_error("CUDA not available: built without CUDA support");
}
template <typename T>
void gemm_cuda_blocked(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) {
    throw std::runtime_error("CUDA not available: built without CUDA support");
}
template <typename T>
void gemm_cuda_reg_tile(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) {
    throw std::runtime_error("CUDA not available: built without CUDA support");
}
template <typename T>
void gemm_cuda_double_buf(const Matrix<T>&, const Matrix<T>&, Matrix<T>&) {
    throw std::runtime_error("CUDA not available: built without CUDA support");
}
void gemm_cuda_wmma(const Matrix<float>&, const Matrix<float>&, Matrix<float>&) {
    throw std::runtime_error("CUDA not available: built without CUDA support");
}

// Explicit instantiations -- required so the linker finds the symbols.
template void gemm_cuda_naive<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_naive<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_reordered<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_reordered<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_blocked<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_blocked<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_reg_tile<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_reg_tile<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_double_buf<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_double_buf<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);

}  // namespace hpc::gemm

