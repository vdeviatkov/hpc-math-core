#pragma once
/**
 * @file gemm_instantiate.hpp
 * @brief Explicit-instantiation macros for the CUDA entry points of
 *        gemm/cuda.hpp, for one parameter struct P.
 *
 * Include it after the definitions: src/cuda/gemm_kernels.cuh in a CUDA
 * build, src/cuda/gemm_stub_templates.hpp in a CPU-only one. The target that
 * calls the entry points instantiates them once per P it uses
 * (benchmarks/cuda_instances.cpp, tests/cuda_instances.cpp). P must be a
 * type name without commas (use a type alias).
 */

#include "gemm/cuda.hpp"

/// One FMA level (Levels 0-4), f32 and f64: Matrix and device entry points.
#define HPC_CUDA_INSTANTIATE_FMA_T(P, T, fn)                                                                    \
    template void hpc::gemm::fn<P, T>(const hpc::Matrix<T>&, const hpc::Matrix<T>&, hpc::Matrix<T>&);          \
    template void hpc::gemm::fn##_device<P, T>(const T*, const T*, T*, int, int, int);
#define HPC_CUDA_INSTANTIATE_FMA(P, fn) HPC_CUDA_INSTANTIATE_FMA_T(P, float, fn) HPC_CUDA_INSTANTIATE_FMA_T(P, double, fn)

/// A float-only level with float device buffers (Levels 5-6).
#define HPC_CUDA_INSTANTIATE_F32(P, fn)                                                                          \
    template void hpc::gemm::fn<P>(const hpc::Matrix<float>&, const hpc::Matrix<float>&, hpc::Matrix<float>&);   \
    template void hpc::gemm::fn##_device<P>(const float*, const float*, float*, int, int, int);

/// A level whose device entry point takes fp16 buffers (Levels 7-8).
#define HPC_CUDA_INSTANTIATE_F16(P, fn)                                                                          \
    template void hpc::gemm::fn<P>(const hpc::Matrix<float>&, const hpc::Matrix<float>&, hpc::Matrix<float>&);   \
    template void hpc::gemm::fn##_device<P>(const void*, const void*, float*, int, int, int);

/// The low-precision kernels and their Matrix wrappers.
#define HPC_CUDA_INSTANTIATE_LOWP(P)                                                                             \
    template void hpc::gemm::gemm_cuda_lowp_device<P>(hpc::gemm::CudaLowp, const void*, const void*, void*, int, \
                                                      int, int);                                                 \
    template void hpc::gemm::gemm_cuda_lowp_tma_device<P>(hpc::gemm::CudaLowp, const void*, const void*, void*,  \
                                                          int, int, int);                                        \
    template void hpc::gemm::gemm_cuda_bf16<P>(const hpc::Matrix<hpc::bf16>&, const hpc::Matrix<hpc::bf16>&,     \
                                               hpc::Matrix<float>&, hpc::gemm::CudaLowpImpl);                    \
    template void hpc::gemm::gemm_cuda_fp16<P>(const hpc::Matrix<hpc::fp16>&, const hpc::Matrix<hpc::fp16>&,     \
                                               hpc::Matrix<float>&, hpc::gemm::CudaLowpImpl);                    \
    template void hpc::gemm::gemm_cuda_fp16_acc16<P>(const hpc::Matrix<hpc::fp16>&,                              \
                                                     const hpc::Matrix<hpc::fp16>&, hpc::Matrix<float>&,         \
                                                     hpc::gemm::CudaLowpImpl);                                   \
    template void hpc::gemm::gemm_cuda_i8<P>(const hpc::Matrix<std::int8_t>&, const hpc::Matrix<std::int8_t>&,   \
                                             hpc::Matrix<std::int32_t>&, hpc::gemm::CudaLowpImpl);               \
    template void hpc::gemm::gemm_cuda_fp8<P>(const hpc::Matrix<hpc::fp8_e4m3>&,                                 \
                                              const hpc::Matrix<hpc::fp8_e4m3>&, hpc::Matrix<float>&,            \
                                              hpc::gemm::CudaLowpImpl);

/// Every entry point that takes P.
#define HPC_CUDA_INSTANTIATE_ALL(P)                        \
    HPC_CUDA_INSTANTIATE_FMA(P, gemm_cuda_naive)           \
    HPC_CUDA_INSTANTIATE_FMA(P, gemm_cuda_blocked)         \
    HPC_CUDA_INSTANTIATE_FMA(P, gemm_cuda_reg_tile)        \
    HPC_CUDA_INSTANTIATE_FMA(P, gemm_cuda_double_buf)      \
    HPC_CUDA_INSTANTIATE_FMA(P, gemm_cuda_vectorized)      \
    HPC_CUDA_INSTANTIATE_F32(P, gemm_cuda_wmma)            \
    HPC_CUDA_INSTANTIATE_F32(P, gemm_cuda_mma_ldmatrix)    \
    HPC_CUDA_INSTANTIATE_F16(P, gemm_cuda_wmma_pipelined)  \
    HPC_CUDA_INSTANTIATE_F16(P, gemm_cuda_tma)             \
    HPC_CUDA_INSTANTIATE_LOWP(P)
