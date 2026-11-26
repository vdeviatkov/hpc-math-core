/**
 * @file cuda_instances.cpp
 * @brief Explicit instantiations of the CUDA entry points for the benchmarks:
 *        the measured parameters (CudaParams) and every sweep variant.
 *
 * Compiled as CUDA when the toolkit is present (CMake sets the source
 * language and HPC_CUDA_KERNELS), as C++ against the throwing stub
 * templates otherwise, so bench_gemm_cuda and bench_gemm_lowp_cuda link in
 * both builds.
 */

#include "cuda_params.hpp"
#include "cuda_variants.hpp"

#if HPC_CUDA_KERNELS
#include "cuda/gemm_kernels.cuh"
#else
#include "cuda/gemm_stub_templates.hpp"
#endif
#include "cuda/gemm_instantiate.hpp"

using Measured = hpc::bench::CudaParams;
HPC_CUDA_INSTANTIATE_ALL(Measured)

namespace v = hpc::bench::cuda_variants;

#define HPC_REGTILE(name, P)                                                                       \
    template void hpc::gemm::gemm_cuda_reg_tile_device<v::P, float>(const float*, const float*, float*, int, int, \
                                                                     int);                         \
    template void hpc::gemm::gemm_cuda_vectorized_device<v::P, float>(const float*, const float*, float*, int,   \
                                                                       int, int);
HPC_CUDA_REGTILE_VARIANTS(HPC_REGTILE)
#undef HPC_REGTILE

#define HPC_TILE(name, P)                                                                          \
    template void hpc::gemm::gemm_cuda_naive_device<v::P, float>(const float*, const float*, float*, int, int, int); \
    template void hpc::gemm::gemm_cuda_blocked_device<v::P, float>(const float*, const float*, float*, int, int,    \
                                                                    int);
HPC_CUDA_TILE_VARIANTS(HPC_TILE)
#undef HPC_TILE

#define HPC_F64(name, P)                                                                           \
    template void hpc::gemm::gemm_cuda_reg_tile_device<v::P, double>(const double*, const double*, double*, int, \
                                                                      int, int);                   \
    template void hpc::gemm::gemm_cuda_double_buf_device<v::P, double>(const double*, const double*, double*,    \
                                                                        int, int, int);            \
    template void hpc::gemm::gemm_cuda_vectorized_device<v::P, double>(const double*, const double*, double*,    \
                                                                        int, int, int);
HPC_CUDA_F64_VARIANTS(HPC_F64)
#undef HPC_F64

#define HPC_DOUBLEBUF(name, P)                                                                     \
    template void hpc::gemm::gemm_cuda_double_buf_device<v::P, float>(const float*, const float*, float*, int,   \
                                                                       int, int);
HPC_CUDA_DOUBLEBUF_VARIANTS(HPC_DOUBLEBUF)
#undef HPC_DOUBLEBUF

#define HPC_WARPGRID(name, P)                                                                      \
    template void hpc::gemm::gemm_cuda_wmma_device<v::P>(const float*, const float*, float*, int, int, int);    \
    template void hpc::gemm::gemm_cuda_mma_ldmatrix_device<v::P>(const float*, const float*, float*, int, int,  \
                                                                 int);
HPC_CUDA_WARPGRID_VARIANTS(HPC_WARPGRID)
#undef HPC_WARPGRID

#define HPC_PIPE(name, P)                                                                          \
    template void hpc::gemm::gemm_cuda_wmma_pipelined_device<v::P>(const void*, const void*, float*, int, int, \
                                                                   int);
HPC_CUDA_PIPE_VARIANTS(HPC_PIPE)
#undef HPC_PIPE

#define HPC_LOWP(name, P)                                                                          \
    template void hpc::gemm::gemm_cuda_lowp_device<v::P>(hpc::gemm::CudaLowp, const void*, const void*, void*, \
                                                         int, int, int);
HPC_CUDA_LOWP_VARIANTS(HPC_LOWP)
#undef HPC_LOWP

#define HPC_TMA(name, P)                                                                           \
    template void hpc::gemm::gemm_cuda_lowp_tma_device<v::P>(hpc::gemm::CudaLowp, const void*, const void*,    \
                                                             void*, int, int, int);
HPC_CUDA_TMA_VARIANTS(HPC_TMA)
#undef HPC_TMA
