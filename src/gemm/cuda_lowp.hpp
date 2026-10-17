#pragma once
/**
 * @file cuda_lowp.hpp
 * @brief Host interface to the low-precision CUDA GEMMs (src/cuda/gemm_lowp.cu).
 *
 * One Tensor Core kernel, instantiated for five input formats. Only the
 * mma.sync instruction and the element width change between them; tiles,
 * shared-memory layout, ldmatrix loads and the cp.async pipeline are shared:
 *
 *   format      instruction (per warp)            accumulate  C
 *   bf16        m16n8k16 .f32.bf16.bf16.f32       fp32        float
 *   fp16        m16n8k16 .f32.f16.f16.f32         fp32        float
 *   fp16_acc16  m16n8k16 .f16.f16.f16.f16         fp16        fp16
 *   int8        m16n8k32 .s32.s8.s8.s32           int32       int32
 *   fp8_e4m3    m16n8k32 .f32.e4m3.e4m3.f32       fp32        float
 *               (or kind::mxf8f6f4.block_scale with unit scales, built for sm_120a)
 *
 * Both operands are K-major: A is M x K row-major and B is passed
 * transposed, Bt = N x K row-major. An mma.sync fragment of an 8-bit
 * operand holds 4 consecutive k per register, and ldmatrix's transpose
 * moves 16-bit elements, so a K-major B is what lets one kernel serve
 * 8- and 16-bit formats alike. It is also the layout ("TN") cuBLASLt
 * requires for FP8.
 *
 * The *_device entry points take device buffers and only launch (they are
 * what the benchmarks time). M and N must be multiples of 128 and K of
 * cuda_lowp_k_multiple(format) (32 for 16-bit inputs, 64 for 8-bit). The
 * Matrix-based wrappers take any shape: they pad with zeros, transpose B,
 * copy to the GPU and back.
 *
 * gemm_cuda_cublas_lowp* run the same products through cuBLAS
 * (cublasGemmEx; cublasLtMatmul for FP8): the vendor reference.
 *
 * Requirements: sm_80+ for bf16, fp16 and int8, sm_89+ for FP8; otherwise
 * the entry points throw. Without a CUDA toolkit, src/cuda/gemm_lowp_stub.cpp
 * provides the same API and every kernel throws.
 */
#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"

#include <cstdint>

namespace hpc::gemm {

enum class CudaLowp { bf16, fp16, fp16_acc16, int8, fp8_e4m3 };

/// True if the selected device runs the format's mma.sync (sm_80+, FP8 sm_89+).
bool cuda_lowp_supported(CudaLowp format) noexcept;
/// K must be a multiple of this on the device entry points.
int cuda_lowp_k_multiple(CudaLowp format) noexcept;
/// True if the FP8 kernel runs the full-rate block-scaled mma (library built
/// for sm_120a, see gemm_lowp.cu); false for the half-rate sm_89 form.
bool cuda_lowp_fp8_block_scaled();

// Device entry points. A: M x K, Bt: N x K, C: M x N, all row-major. Element
// types by format (see the table above); the buffers are void* because this
// header is also compiled against the CPU-only stub.
void gemm_cuda_lowp_device(CudaLowp format, const void* dA, const void* dBt, void* dC,
                           int M, int K, int N);
void gemm_cuda_cublas_lowp_device(CudaLowp format, const void* dA, const void* dBt, void* dC,
                                  int M, int K, int N);

// Matrix wrappers: any shape, B as an ordinary K x N matrix. fp16_acc16
// results are widened to float on the host. `cublas` selects the vendor
// reference instead of the hand-written kernel.
void gemm_cuda_bf16(const Matrix<bf16>& A, const Matrix<bf16>& B, Matrix<float>& C, bool cublas = false);
void gemm_cuda_fp16(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C, bool cublas = false);
void gemm_cuda_fp16_acc16(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C,
                          bool cublas = false);
void gemm_cuda_i8(const Matrix<std::int8_t>& A, const Matrix<std::int8_t>& B, Matrix<std::int32_t>& C,
                  bool cublas = false);
void gemm_cuda_fp8(const Matrix<fp8_e4m3>& A, const Matrix<fp8_e4m3>& B, Matrix<float>& C,
                   bool cublas = false);

}  // namespace hpc::gemm
