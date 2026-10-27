#pragma once
/**
 * @file cuda.hpp
 * @brief Host interface to every CUDA GEMM (src/cuda/gemm.cu): Levels 0-8,
 *        the low-precision formats and the cuBLAS reference.
 *
 * Nine levels, each adding one technique to the one it builds on. Levels
 * 0-4 run on the FMA (SIMT) cores in f32 and f64, Levels 5-8 on Tensor Cores:
 *
 *   0  gemm_cuda_naive           one thread per C(i,j), global memory only
 *   1  gemm_cuda_blocked         16x16 shared-memory tiles
 *   2  gemm_cuda_reg_tile        128x128 block tile, 8x8 register tile per thread
 *   3  gemm_cuda_double_buf      Level 2 + double-buffered shared memory (cp.async on sm_80+)
 *   4  gemm_cuda_vectorized      Level 2 + 128-bit loads + XOR-swizzled shared memory
 *   5  gemm_cuda_wmma            Tensor Cores through the wmma:: API, 64x64 tiles
 *   6  gemm_cuda_mma_ldmatrix    Level 5 through raw ldmatrix + mma.sync PTX
 *   7  gemm_cuda_wmma_pipelined  wmma:: with 128x128 tiles, 8 fragments per warp,
 *                                cp.async double buffering
 *   8  gemm_cuda_tma             raw mma.sync fed by TMA into swizzled shared
 *                                memory, 4-6-stage mbarrier pipeline, producer warp,
 *                                128x256 tiles (128x128 for small grids)
 *
 * Levels 5-8 are float-only: fp32 in and out, fp16 multiply, fp32
 * accumulate (input rounding at roughly the 1e-3 scale; output relative
 * error is not bounded near cancellation, and fp16 has a smaller range).
 * A level whose requirement is not met runs a level below it instead:
 *
 *   Level 4  needs K, N multiples of 4 (f32) / 2 (f64)       else Level 2
 *   Level 5  needs sm_70+                                   else Level 3
 *   Level 6  needs sm_80+                                   else Level 5
 *   Level 7  needs sm_70+, M and N multiples of 128, K of 32 else Level 5
 *   Level 8  needs sm_90+ (TMA)                              else Level 7
 *
 * Low-precision formats. Level 8's kernel and a cp.async kernel (Level 6's
 * mma.sync grown to Level 7's tiles and pipeline) also run five input
 * formats; only the mma.sync instruction and the element width change:
 *
 *   format      instruction (per warp)            accumulate  C
 *   bf16        m16n8k16 .f32.bf16.bf16.f32       fp32        float
 *   fp16        m16n8k16 .f32.f16.f16.f32         fp32        float
 *   fp16_acc16  m16n8k16 .f16.f16.f16.f16         fp16        fp16
 *   int8        m16n8k32 .s32.s8.s8.s32           int32       int32
 *   fp8_e4m3    m16n8k32 .f32.e4m3.e4m3.f32       fp32        float
 *               (or kind::mxf8f6f4.block_scale with unit scales, built for sm_120a)
 *
 * Both operands of the low-precision kernels are K-major: A is M x K
 * row-major and B is passed transposed, Bt = N x K row-major. An mma.sync
 * fragment of an 8-bit operand holds 4 consecutive k per register, and
 * ldmatrix's transpose moves 16-bit elements, so a K-major B is what lets
 * one kernel serve 8- and 16-bit formats alike. It is also the layout
 * ("TN") cuBLASLt requires for FP8. They need sm_80+ (FP8 sm_89+).
 *
 * gemm_cuda_cublas* are not part of the ladder: they are the vendor
 * reference the hand-written kernels are measured against.
 *
 * Every Matrix-based entry point allocates device buffers, copies A and B
 * in, runs the kernel and copies C back. The *_device entry points take
 * buffers already on the GPU and only launch the GEMM: they are what the
 * benchmarks time, so GPU numbers, like CPU ones, measure the
 * multiplication alone.
 *
 * Without a CUDA toolkit the same API is provided by src/cuda/gemm_stub.cpp:
 * cuda_device_count() returns 0, no format is supported, and the kernels
 * throw.
 */
#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace hpc::gemm {

// ---------------------------------------------------------------------------
// Device queries
// ---------------------------------------------------------------------------

/// Number of CUDA devices; 0 in a build without CUDA.
int cuda_device_count() noexcept;
/// True if the selected device is sm_70+ (Tensor Cores).
bool cuda_has_tensor_cores() noexcept;
/// True if the selected device is sm_80+ (cp.async, mma.sync m16n8k16, TF32).
bool cuda_has_ampere() noexcept;
/// True if the selected device is sm_90+ (TMA, mbarrier transactions): Level 8.
bool cuda_has_tma() noexcept;

/// The low-precision input formats (see the table above).
enum class CudaLowp { bf16, fp16, fp16_acc16, int8, fp8_e4m3 };

/// Which implementation a low-precision Matrix wrapper runs.
enum class CudaLowpImpl {
    mma,     ///< cp.async kernel, gemm_cuda_lowp_device
    tma,     ///< Level 8, gemm_cuda_lowp_tma_device (sm_90+)
    cublas,  ///< vendor reference, gemm_cuda_cublas_lowp_device
};

/// True if the selected device runs the format's mma.sync (sm_80+, FP8 sm_89+).
bool cuda_lowp_supported(CudaLowp format) noexcept;
/// K must be a multiple of this on the low-precision device entry points:
/// 32 for 16-bit inputs, 64 for 8-bit ones (64 bytes per pipeline stage).
int cuda_lowp_k_multiple(CudaLowp format) noexcept;
/// True if the FP8 kernels run the full-rate block-scaled mma (library built
/// for sm_120a); false for the half-rate sm_89 form.
bool cuda_lowp_fp8_block_scaled();

// ---------------------------------------------------------------------------
// Matrix-based entry points: any shape, B as an ordinary K x N matrix.
// ---------------------------------------------------------------------------

/// Level 0: one thread per C(i, j).
template <typename T>
void gemm_cuda_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
/// Level 1: 16 x 16 shared-memory tiles.
template <typename T>
void gemm_cuda_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
/// Level 2: 128 x 128 block tile, 8 x 8 register tile per thread.
template <typename T>
void gemm_cuda_reg_tile(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
/// Level 3: Level 2, double-buffered.
template <typename T>
void gemm_cuda_double_buf(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
/// Level 4: Level 2 with 128-bit loads and swizzled shared memory.
template <typename T>
void gemm_cuda_vectorized(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
/// Level 5: wmma:: Tensor Cores, fp32 inputs converted to fp16 per tile.
void gemm_cuda_wmma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
/// Level 6: Level 5 through ldmatrix + mma.sync PTX.
void gemm_cuda_mma_ldmatrix(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
/// Level 7: pipelined wmma::, inputs converted to fp16 on the GPU first.
void gemm_cuda_wmma_pipelined(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
/// Level 8: TMA + mbarrier pipeline, inputs rounded to fp16, padded and B transposed on the host.
void gemm_cuda_tma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);

/// cuBLAS SGEMM / DGEMM.
template <typename T>
void gemm_cuda_cublas(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
/// cuBLAS on TF32 Tensor Cores (fp32 in and out).
void gemm_cuda_cublas_tf32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
/// cuBLAS on fp16 Tensor Cores, inputs converted to fp16 first, fp32 accumulate.
void gemm_cuda_cublas_fp16(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);

/// Low-precision GEMMs through `impl`. They pad with zeros to tile
/// multiples, transpose B, copy to the GPU and back. fp16_acc16 results are
/// widened to float on the host.
void gemm_cuda_bf16(const Matrix<bf16>& A, const Matrix<bf16>& B, Matrix<float>& C,
                    CudaLowpImpl impl = CudaLowpImpl::mma);
void gemm_cuda_fp16(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C,
                    CudaLowpImpl impl = CudaLowpImpl::mma);
void gemm_cuda_fp16_acc16(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C,
                          CudaLowpImpl impl = CudaLowpImpl::mma);
void gemm_cuda_i8(const Matrix<std::int8_t>& A, const Matrix<std::int8_t>& B, Matrix<std::int32_t>& C,
                  CudaLowpImpl impl = CudaLowpImpl::mma);
void gemm_cuda_fp8(const Matrix<fp8_e4m3>& A, const Matrix<fp8_e4m3>& B, Matrix<float>& C,
                   CudaLowpImpl impl = CudaLowpImpl::mma);

// ---------------------------------------------------------------------------
// Device entry points: the same GEMMs on buffers the caller has allocated and
// filled on the GPU, row-major. Each enqueues work on the default stream,
// asynchronously, with the same fallback as its Matrix-based counterpart.
//
// Preconditions: dimensions and linear indices fit signed int; buffers do
// not overlap and are aligned for the selected path (16-byte vector loads,
// 32-byte WMMA output). Negative dimensions throw. K = 0 clears C; M = 0 or
// N = 0 returns.
//
// fp16 and low-precision buffers are typed void*: this header is also
// compiled against the CPU-only stub, where <cuda_fp16.h> does not exist.
// ---------------------------------------------------------------------------

/// Level 0 on device buffers: A M x K, B K x N, C M x N.
template <typename T>
void gemm_cuda_naive_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
/// Level 1 on device buffers.
template <typename T>
void gemm_cuda_blocked_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
/// Level 2 on device buffers.
template <typename T>
void gemm_cuda_reg_tile_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
/// Level 3 on device buffers.
template <typename T>
void gemm_cuda_double_buf_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
/// Level 4 on device buffers.
template <typename T>
void gemm_cuda_vectorized_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
/// Level 5 on device buffers.
void gemm_cuda_wmma_device(const float* dA, const float* dB, float* dC, int M, int K, int N);
/// Level 6 on device buffers.
void gemm_cuda_mma_ldmatrix_device(const float* dA, const float* dB, float* dC, int M, int K, int N);
/// Level 7 on fp16 A (M x K) and B (K x N); convert them with
/// gemm_cuda_convert_f32_to_f16_device. No fallback: M and N multiples of
/// 128, K of 32, sm_70+.
void gemm_cuda_wmma_pipelined_device(const void* dA16, const void* dB16, float* dC, int M, int K, int N);
/// Level 8 on fp16 A (M x K) and fp16 B transposed (N x K): the layout its
/// TMA copies and non-transposing ldmatrix read. No fallback: M and N
/// multiples of 128, K of 32, sm_90+.
void gemm_cuda_tma_device(const void* dA16, const void* dBt16, float* dC, int M, int K, int N);

/// The low-precision cp.async kernel. A: M x K, Bt: N x K, C: M x N, element
/// types by format (see the table above). M and N must be multiples of 128,
/// K of cuda_lowp_k_multiple(format).
void gemm_cuda_lowp_device(CudaLowp format, const void* dA, const void* dBt, void* dC, int M, int K, int N);
/// Level 8 for every low-precision format; requirements as above, sm_90+.
void gemm_cuda_lowp_tma_device(CudaLowp format, const void* dA, const void* dBt, void* dC, int M, int K, int N);

/// cuBLAS SGEMM / DGEMM on device buffers.
template <typename T>
void gemm_cuda_cublas_device(const T* dA, const T* dB, T* dC, int M, int K, int N);
/// cuBLAS TF32 on device buffers.
void gemm_cuda_cublas_tf32_device(const float* dA, const float* dB, float* dC, int M, int K, int N);
/// cuBLAS fp16 on fp16 A (M x K) and B (K x N), fp32 C.
void gemm_cuda_cublas_fp16_device(const void* dA16, const void* dB16, float* dC, int M, int K, int N);
/// cuBLAS (cublasGemmEx; cublasLtMatmul for FP8) on the low-precision
/// layout: A M x K, Bt N x K. Any shape.
void gemm_cuda_cublas_lowp_device(CudaLowp format, const void* dA, const void* dBt, void* dC, int M, int K, int N);

/// dst[i] = fp16(src[i]) for i < count, on the GPU.
void gemm_cuda_convert_f32_to_f16_device(const float* src, void* dst, int count);

// ---------------------------------------------------------------------------
// Timing and device memory, for the benchmarks (so they need not include
// <cuda_runtime.h>, which CPU-only builds lack).
// ---------------------------------------------------------------------------

/// GPU time of the work `launch` enqueues, in milliseconds: CUDA events are
/// recorded on the GPU before and after it, so CPU work and scheduling noise
/// outside the event interval are not counted. GPU idle gaps while the host
/// submits work can be included, especially for tiny kernels. Waits for the
/// work to finish.
double gemm_cuda_time_ms(const std::function<void()>& launch);
/// cudaMalloc; throws on failure.
void* gemm_cuda_malloc(std::size_t bytes);
/// cudaFree; null is a no-op.
void gemm_cuda_free(void* ptr);
/// Host -> device copy.
void gemm_cuda_memcpy_h2d(void* dst, const void* src, std::size_t bytes);
/// Device -> host copy.
void gemm_cuda_memcpy_d2h(void* dst, const void* src, std::size_t bytes);
/// cudaDeviceSynchronize; throws on a pending error.
void gemm_cuda_device_synchronize();

// ---------------------------------------------------------------------------
// Explicit instantiation declarations (definitions in src/cuda/gemm.cu)
// ---------------------------------------------------------------------------
#define HPC_CUDA_EXTERN(fn)                                                                         \
    extern template void fn<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);     \
    extern template void fn<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&); \
    extern template void fn##_device<float>(const float*, const float*, float*, int, int, int);    \
    extern template void fn##_device<double>(const double*, const double*, double*, int, int, int);
HPC_CUDA_EXTERN(gemm_cuda_naive)
HPC_CUDA_EXTERN(gemm_cuda_blocked)
HPC_CUDA_EXTERN(gemm_cuda_reg_tile)
HPC_CUDA_EXTERN(gemm_cuda_double_buf)
HPC_CUDA_EXTERN(gemm_cuda_vectorized)
HPC_CUDA_EXTERN(gemm_cuda_cublas)
#undef HPC_CUDA_EXTERN

}  // namespace hpc::gemm
