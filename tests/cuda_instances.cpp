/**
 * @file cuda_instances.cpp
 * @brief Explicit instantiations of the CUDA entry points for the tests:
 *        the measured parameters and CudaOdd (cuda_test_params.hpp). CUDA
 *        when the toolkit is present, the throwing stub templates otherwise.
 */

#include "cuda_test_params.hpp"

#if HPC_CUDA_KERNELS
#include "cuda/gemm_kernels.cuh"
#else
#include "cuda/gemm_stub_templates.hpp"
#endif
#include "cuda/gemm_instantiate.hpp"

using Measured = hpc::test::CudaMeasured;
using Odd      = hpc::test::CudaOdd;
HPC_CUDA_INSTANTIATE_ALL(Measured)
HPC_CUDA_INSTANTIATE_ALL(Odd)
