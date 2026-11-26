/**
 * @file gemm.cu
 * @brief The size-independent part of the CUDA API (gemm/cuda.hpp): device
 *        queries, the cuBLAS / cuBLASLt references, fp32 -> fp16 conversion,
 *        timing and device memory.
 *
 * The kernels and their entry points are templates over the tile sizes, in
 * src/cuda/gemm_kernels.cuh; the targets that call them instantiate them
 * with their parameters (benchmarks/cuda_instances.cpp,
 * tests/cuda_instances.cpp). This file includes the header for its
 * building blocks and instantiates nothing.
 */

#include "cuda/gemm_kernels.cuh"

#include <cublasLt.h>
#include <cublas_v2.h>

namespace {

using hpc::Matrix;
using hpc::gemm::CudaLowp;
using namespace hpc::gemm::cuda_detail;

/// Throw if a cuBLAS call failed.
void check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed: status " + std::to_string(int(status)));
}

/// Compute capability of the selected device (major * 10 + minor), 0 on
/// error. Cached per host thread and device: cudaSetDevice may change the
/// selection between calls, so only cudaGetDevice runs on a cache hit.
int device_capability() noexcept {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return 0;
    static thread_local std::map<int, int> cache;
    if (const auto it = cache.find(device); it != cache.end())
        return it->second;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        return 0;
    const int capability = prop.major * 10 + prop.minor;
    try {
        cache.emplace(device, capability);
    } catch (...) {  // allocation failure must not escape a noexcept query
    }
    return capability;
}

/// dst[i] = fp16(src[i]): converts a whole matrix in global memory ahead of
/// time, for kernel_wmma_pipelined (cp.async cannot convert) and the cuBLAS
/// fp16 reference.
__global__ void kernel_f32_to_f16(const float* __restrict__ src, __half* __restrict__ dst, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = __float2half(src[i]);
}

/// Writes 1 when this build's FP8 mma is the block-scaled form, else 0.
__global__ void kernel_fp8_probe(int* out) {
#if defined(HPC_FP8_BLOCK_SCALED)
    *out = 1;
#else
    *out = 0;
#endif
}

/// Lazily created, process-lifetime cuBLAS handle, bound to the device
/// selected on first use. Not thread-safe; do not switch or reset that device.
cublasHandle_t cublas() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        check_cublas(cublasCreate(&h), "cublasCreate");
        return h;
    }();
    return handle;
}

/// The cuBLASLt handle for the FP8 reference, like cublas().
cublasLtHandle_t cublas_lt() {
    static cublasLtHandle_t handle = [] {
        cublasLtHandle_t h;
        check_cublas(cublasLtCreate(&h), "cublasLtCreate");
        return h;
    }();
    return handle;
}

/// Row-major C (M x N) = A (M x K) * B through cublasGemmEx, B either K x N
/// or, when b_transposed, N x K. cuBLAS is column-major, and row-major C is
/// column-major C^T = B^T A^T over the same memory, so B goes first and M
/// and N swap. A row-major K x N B is column-major B^T (op N); a row-major
/// N x K Bt is column-major B, which op T turns into B^T. That second, "TN"
/// form is the one cuBLASLt requires for FP8.
void cublas_gemm_ex(const void* A, const void* B, bool b_transposed, void* C, cudaDataType in, cudaDataType out,
                    cublasComputeType_t compute, const void* alpha, const void* beta, int M, int K, int N) {
    check_cublas(cublasGemmEx(cublas(), b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, B, in,
                              b_transposed ? K : N, A, in, K, beta, C, out, N, compute, CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx");
}

/// FP8 reference: row-major C = A * Bt^T through cublasLtMatmul ("TN", as in
/// cublas_gemm_ex), e4m3 inputs, fp32 output.
void cublas_fp8(const void* A, const void* Bt, float* C, int M, int K, int N) {
    constexpr std::size_t kWorkspace = 32u << 20;
    static void* workspace = [] {
        void* p = nullptr;
        HPC_CUDA_CHECK(cudaMalloc(&p, kWorkspace));
        return p;
    }();
    struct Desc {
        cublasLtMatmulDesc_t op{};
        cublasLtMatrixLayout_t a{}, b{}, c{};
        cublasLtMatmulPreference_t pref{};
        ~Desc() {
            if (pref) cublasLtMatmulPreferenceDestroy(pref);
            if (c) cublasLtMatrixLayoutDestroy(c);
            if (b) cublasLtMatrixLayoutDestroy(b);
            if (a) cublasLtMatrixLayoutDestroy(a);
            if (op) cublasLtMatmulDescDestroy(op);
        }
    } d;
    check_cublas(cublasLtMatmulDescCreate(&d.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
    const cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    check_cublas(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof opT), "TRANSA");
    check_cublas(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof opN), "TRANSB");
    check_cublas(cublasLtMatrixLayoutCreate(&d.a, CUDA_R_8F_E4M3, K, N, K), "layout A");
    check_cublas(cublasLtMatrixLayoutCreate(&d.b, CUDA_R_8F_E4M3, K, M, K), "layout B");
    check_cublas(cublasLtMatrixLayoutCreate(&d.c, CUDA_R_32F, N, M, N), "layout C");
    check_cublas(cublasLtMatmulPreferenceCreate(&d.pref), "preference");
    check_cublas(cublasLtMatmulPreferenceSetAttribute(d.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspace,
                                                      sizeof kWorkspace),
                 "workspace");
    cublasLtMatmulHeuristicResult_t heuristic{};
    int found = 0;
    check_cublas(cublasLtMatmulAlgoGetHeuristic(cublas_lt(), d.op, d.a, d.b, d.c, d.c, d.pref, 1, &heuristic, &found),
                 "cublasLtMatmulAlgoGetHeuristic");
    if (found == 0)
        throw std::runtime_error("cuBLASLt has no FP8 algorithm for this shape");
    const float alpha = 1.0f, beta = 0.0f;
    check_cublas(cublasLtMatmul(cublas_lt(), d.op, &alpha, Bt, d.a, A, d.b, &beta, C, d.c, C, d.c, &heuristic.algo,
                                workspace, kWorkspace, nullptr),
                 "cublasLtMatmul (FP8)");
}

}  // namespace

namespace hpc::gemm {

using namespace cuda_detail;

int cuda_device_count() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}
bool cuda_has_tensor_cores() noexcept { return device_capability() >= 70; }
bool cuda_has_ampere() noexcept { return device_capability() >= 80; }
bool cuda_has_tma() noexcept { return device_capability() >= 90; }

bool cuda_lowp_supported(CudaLowp format) noexcept {
    return device_capability() >= (format == CudaLowp::fp8_e4m3 ? 89 : 80);
}
int cuda_lowp_k_multiple(CudaLowp format) noexcept { return kKBytes / input_bytes(format); }

bool cuda_lowp_fp8_block_scaled() {
    if (!cuda_lowp_supported(CudaLowp::fp8_e4m3))
        return false;
    DeviceBuffer<int> d(1);
    int h = 0;
    kernel_fp8_probe<<<1, 1>>>(d.ptr);
    HPC_CUDA_CHECK(cudaMemcpy(&h, d.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    return h == 1;
}

void gemm_cuda_convert_f32_to_f16_device(const float* src, void* dst, int count) {
    if (count < 0)
        throw std::invalid_argument("conversion count must be nonnegative");
    if (count == 0)
        return;
    constexpr int kThreads = 256;
    launch<kernel_f32_to_f16>(dim3((count - 1) / kThreads + 1), dim3(kThreads), 0, src, static_cast<__half*>(dst),
                              count);
}

// ---------------------------------------------------------------------------
// cuBLAS reference
// ---------------------------------------------------------------------------

template <typename T>
void gemm_cuda_cublas_device(const T* A, const T* B, T* C, int M, int K, int N) {
    if (empty_product(C, sizeof(T), M, K, N))
        return;
    const T one = 1, zero = 0;
    constexpr bool f32 = std::is_same_v<T, float>;
    cublas_gemm_ex(A, B, false, C, f32 ? CUDA_R_32F : CUDA_R_64F, f32 ? CUDA_R_32F : CUDA_R_64F,
                   f32 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_64F, &one, &zero, M, K, N);
}

void gemm_cuda_cublas_tf32_device(const float* A, const float* B, float* C, int M, int K, int N) {
    if (empty_product(C, sizeof(float), M, K, N))
        return;
    const float one = 1, zero = 0;
    cublas_gemm_ex(A, B, false, C, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, &one, &zero, M, K, N);
}

void gemm_cuda_cublas_fp16_device(const void* A16, const void* B16, float* C, int M, int K, int N) {
    if (empty_product(C, sizeof(float), M, K, N))
        return;
    const float one = 1, zero = 0;
    cublas_gemm_ex(A16, B16, false, C, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, &one, &zero, M, K, N);
}

void gemm_cuda_cublas_lowp_device(CudaLowp f, const void* A, const void* Bt, void* C, int M, int K, int N) {
    if (lowp_empty_product(f, C, M, K, N))
        return;
    const float one = 1, zero = 0;
    const __half one_h = __float2half(1.0f), zero_h = __float2half(0.0f);
    const int one_i = 1, zero_i = 0;
    switch (f) {
        case CudaLowp::bf16:
            return cublas_gemm_ex(A, Bt, true, C, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, &one, &zero, M, K, N);
        case CudaLowp::fp16:
            return cublas_gemm_ex(A, Bt, true, C, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, &one, &zero, M, K, N);
        case CudaLowp::fp16_acc16:
            return cublas_gemm_ex(A, Bt, true, C, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, &one_h, &zero_h, M, K,
                                  N);
        case CudaLowp::int8:
            return cublas_gemm_ex(A, Bt, true, C, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, &one_i, &zero_i, M, K, N);
        case CudaLowp::fp8_e4m3: return cublas_fp8(A, Bt, static_cast<float*>(C), M, K, N);
    }
}

template <typename T>
void gemm_cuda_cublas(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, gemm_cuda_cublas_device<T>);
}
void gemm_cuda_cublas_tf32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, gemm_cuda_cublas_tf32_device);
}
void gemm_cuda_cublas_fp16(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, [](const float* dA, const float* dB, float* dC, int M, int K, int N) {
        with_fp16_copies(dA, dB, M, K, N,
                         [&](const __half* A16, const __half* B16) {
                             gemm_cuda_cublas_fp16_device(A16, B16, dC, M, K, N);
                         });
    });
}

// ---------------------------------------------------------------------------
// Timing and device-memory helpers for the benchmarks
// ---------------------------------------------------------------------------

double gemm_cuda_time_ms(const std::function<void()>& launch_fn) {
    struct Event {
        cudaEvent_t value{};
        Event() { HPC_CUDA_CHECK(cudaEventCreate(&value)); }
        ~Event() { cudaEventDestroy(value); }
        Event(const Event&)            = delete;
        Event& operator=(const Event&) = delete;
    };
    Event start, stop;
    HPC_CUDA_CHECK(cudaEventRecord(start.value));
    launch_fn();
    HPC_CUDA_CHECK(cudaEventRecord(stop.value));
    HPC_CUDA_CHECK(cudaEventSynchronize(stop.value));
    float ms = 0;
    HPC_CUDA_CHECK(cudaEventElapsedTime(&ms, start.value, stop.value));
    return ms;
}

void* gemm_cuda_malloc(std::size_t bytes) {
    void* ptr = nullptr;
    HPC_CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}
void gemm_cuda_free(void* ptr) {
    if (ptr)
        cudaFree(ptr);
}
void gemm_cuda_memcpy_h2d(void* dst, const void* src, std::size_t bytes) {
    HPC_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
void gemm_cuda_memcpy_d2h(void* dst, const void* src, std::size_t bytes) {
    HPC_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}
void gemm_cuda_device_synchronize() { HPC_CUDA_CHECK(cudaDeviceSynchronize()); }


template void gemm_cuda_cublas<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_cublas<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_cublas_device<float>(const float*, const float*, float*, int, int, int);
template void gemm_cuda_cublas_device<double>(const double*, const double*, double*, int, int, int);

}  // namespace hpc::gemm
