/**
 * @file gemm_lowp.cu
 * @brief Low-precision Tensor Core GEMM: bf16, fp16, int8 and FP8 inputs.
 *
 * The public API and the format table are in gemm/cuda_lowp.hpp. This is
 * the raw mma.sync kernel of Level 6 (gemm_kernels.cu) grown to the tile
 * sizes and pipeline of Level 7:
 *
 *   - 128 x 128 block tile, 64 bytes of K per stage (32 16-bit or 64 8-bit
 *     elements), 8 warps of 64 x 32 each.
 *   - A and Bt are both K-major, so every operand tile is "rows of K bytes"
 *     and one non-transposing ldmatrix.x4 fills the fragments of every
 *     format: a 16 x 32-byte A fragment is 16 k-values of a 16-bit type or
 *     32 of an 8-bit type, and mma.sync's register layout is the same
 *     either way (m16n8k16 for 16-bit, m16n8k32 for 8-bit).
 *   - A kStages-deep cp.async pipeline (16-byte chunks, no conversion).
 *     Two stages (40 KiB of shared memory) let two blocks share an SM;
 *     three or four stages (60 / 80 KiB) allow only one and measured
 *     10–15% slower on an RTX 5080 at every size and format.
 *   - Grouped block order (kGroupM, below), for L2 reuse.
 *   - Shared-memory rows padded by 16 bytes (80-byte stride): the eight
 *     16-byte rows one ldmatrix phase reads start in banks 0, 20, 8, 28,
 *     16, 4, 24, 12, so they never conflict.
 *
 * Shapes are exact tile multiples (the Matrix wrappers pad), so the kernel
 * has no bounds checks.
 */

#include "gemm/cuda_lowp.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#define LOWP_CUDA_CHECK(expr)                                                                \
    do {                                                                                    \
        cudaError_t _e = (expr);                                                            \
        if (_e != cudaSuccess)                                                              \
            throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") +           \
                                     std::to_string(__LINE__) + ": " + cudaGetErrorString(_e)); \
    } while (0)

namespace {

using hpc::gemm::CudaLowp;

// ============================================================================
// Formats: element width, accumulator registers and the one mma.sync that
// differs between them. Each mma covers 16 (M) x 8 (N) x 32 bytes of K.
// a[4] / b[2] are the fragments ldmatrix produced; c is this lane's share of
// the 16x8 accumulator tile: 4 fp32/int32 values, or 4 fp16 values packed
// in 2 registers.
// ============================================================================
struct Bf16 {
    static constexpr int kBytes = 2, kAccRegs = 4;
    using Acc                   = float;
    __device__ static void mma(Acc* c, const unsigned* a, const unsigned* b) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct Fp16 {
    static constexpr int kBytes = 2, kAccRegs = 4;
    using Acc                   = float;
    __device__ static void mma(Acc* c, const unsigned* a, const unsigned* b) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct Fp16Acc16 {
    static constexpr int kBytes = 2, kAccRegs = 2;
    using Acc                   = unsigned;  // two packed fp16 per register
    __device__ static void mma(Acc* c, const unsigned* a, const unsigned* b) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                     "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                     : "+r"(c[0]), "+r"(c[1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct Int8 {
    static constexpr int kBytes = 1, kAccRegs = 4;
    using Acc                   = int;
    __device__ static void mma(Acc* c, const unsigned* a, const unsigned* b) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
// FP8 has two instructions on sm_120 (RTX 50). The sm_89 form below runs
// at half the Tensor Core rate with an fp32 accumulator (249 vs 497
// TFLOP/s on an RTX 5080, benchmarks/experiments/cuda_peak.cu). The
// block-scaled "MX" form runs at full rate: it multiplies each operand by a
// per-32-element power-of-two scale (UE8M0), and with every scale = 2^0 its
// result is the plain product. It needs the arch-specific target, so it is
// used when the library is built with CMAKE_CUDA_ARCHITECTURES=120a.
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ == 1200
#define HPC_LOWP_FP8_BLOCK_SCALED 1
#endif
struct Fp8 {
    static constexpr int kBytes = 1, kAccRegs = 4;
    using Acc                   = float;
    __device__ static void mma(Acc* c, const unsigned* a, const unsigned* b) {
#if defined(HPC_LOWP_FP8_BLOCK_SCALED)
        const unsigned scale = 0x7f7f7f7fu;  // UE8M0 127 = 2^0, in every byte
        const unsigned short selector = 0;   // which byte of `scale` (one scale per operand row)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                     ".f32.e4m3.e4m3.f32.ue8m0 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%11,%11}, %10, {%11,%11};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(scale),
                       "h"(selector));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
        __trap();  // the host checks cuda_lowp_supported first
#endif
    }
};

// ============================================================================
// Tiling
// ============================================================================
#ifndef HPC_LOWP_BN
#define HPC_LOWP_BN 128
#endif
constexpr int kBM = 128, kBN = HPC_LOWP_BN;  // block tile
constexpr int kBK = 64;               // bytes of K per pipeline stage
#ifndef HPC_LOWP_STAGES
#define HPC_LOWP_STAGES 2
#endif
#ifndef HPC_LOWP_GROUP_M
#define HPC_LOWP_GROUP_M 8
#endif
constexpr int kStages = HPC_LOWP_STAGES;  // cp.async pipeline depth
constexpr int kGroupM = HPC_LOWP_GROUP_M; // block rasterization, see kernel_lowp
constexpr int kRowBytes = kBK + 16;   // padded shared-memory row: 80 B
constexpr int kWarpsN = 4;
constexpr int kWarpM = 64, kWarpN = kBN / kWarpsN;
constexpr int kThreads = (kBM / kWarpM) * kWarpsN * 32;  // 256
constexpr int kFragM = kWarpM / 16;                      // 4 m16 tiles per warp
constexpr int kFragN = kWarpN / 8;                       // 4 n8 tiles per warp
constexpr int kStageBytes = (kBM + kBN) * kRowBytes;     // 20,480
constexpr int kSmemBytes = kStages * kStageBytes;        // 40,960

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}

// Copy a ROWS x 64-byte tile (rows `ld_bytes` apart) into shared memory as
// 16-byte cp.async chunks.
template <int ROWS>
__device__ __forceinline__ void load_tile(unsigned char* dst, const unsigned char* src,
                                          std::size_t ld_bytes) {
    #pragma unroll
    for (int i = threadIdx.x; i < ROWS * (kBK / 16); i += kThreads) {
        const int r = i / (kBK / 16), c = i % (kBK / 16) * 16;
        cp_async_16(dst + r * kRowBytes + c, src + r * ld_bytes + c);
    }
}

// C = A * Bt^T for one 128 x 128 tile. A: M x K, Bt: N x K (K-major), C: M x N.
template <typename F, typename Out>
__global__ void __launch_bounds__(kThreads)
kernel_lowp(const unsigned char* __restrict__ A, const unsigned char* __restrict__ Bt,
            Out* __restrict__ C, int M, int K, int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char smem[];
    const std::size_t ld = static_cast<std::size_t>(K) * F::kBytes;  // bytes per row of A and Bt

    // Block rasterization: consecutive blocks walk down a column of kGroupM
    // tile rows before moving right, so the blocks resident at one time
    // share a few A strips and a few Bt strips, which then stay in L2.
    // Row-major order would stream all of Bt through L2 for every row of
    // tiles.
    const int tiles_m = M / kBM, tiles_n = N / kBN;
    const int group = blockIdx.x / (kGroupM * tiles_n), first_m = group * kGroupM;
    const int group_m = min(tiles_m - first_m, kGroupM);
    const int in_group = blockIdx.x % (kGroupM * tiles_n);
    const int tile_m = first_m + in_group % group_m, tile_n = in_group / group_m;
    A += static_cast<std::size_t>(tile_m) * kBM * ld;
    Bt += static_cast<std::size_t>(tile_n) * kBN * ld;
    C += static_cast<std::size_t>(tile_m) * kBM * N + tile_n * kBN;

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / kWarpsN * kWarpM, wn = warp % kWarpsN * kWarpN;
    const int q = lane / 8;  // which 8x8 matrix of an ldmatrix.x4 this lane addresses

    typename F::Acc acc[kFragM][kFragN][F::kAccRegs] = {};

    auto stage_a = [&](int s) { return smem + s * kStageBytes; };
    auto stage_b = [&](int s) { return smem + s * kStageBytes + kBM * kRowBytes; };
    const int k_tiles = static_cast<int>(ld / kBK);

    // Prologue: put the first kStages - 1 tiles in flight.
    #pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < k_tiles) {
            load_tile<kBM>(stage_a(s), A + s * kBK, ld);
            load_tile<kBN>(stage_b(s), Bt + s * kBK, ld);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    for (int kt = 0; kt < k_tiles; ++kt) {
        // Tile kt has landed once at most kStages - 2 younger groups are pending.
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
        __syncthreads();  // ... for every thread, and stage (kt-1) is free to refill

        const int next = kt + kStages - 1;
        if (next < k_tiles) {
            load_tile<kBM>(stage_a(next % kStages), A + static_cast<std::size_t>(next) * kBK, ld);
            load_tile<kBN>(stage_b(next % kStages), Bt + static_cast<std::size_t>(next) * kBK, ld);
        }
        asm volatile("cp.async.commit_group;\n");  // possibly empty: keeps the group count uniform

        const unsigned char* As = stage_a(kt % kStages);
        const unsigned char* Bs = stage_b(kt % kStages);
        #pragma unroll
        for (int kk = 0; kk < kBK; kk += 32) {
            // A, 16 rows x 32 bytes per fragment: the four 8x8 (16-bit) matrices
            // in mma order (rows 0-7, bytes 0-15), (8-15, 0-15), (0-7, 16-31),
            // (8-15, 16-31). Lane group q supplies the row addresses of matrix q.
            unsigned a[kFragM][4];
            #pragma unroll
            for (int i = 0; i < kFragM; ++i)
                ldmatrix_x4(a[i], As + (wm + i * 16 + lane % 8 + q % 2 * 8) * kRowBytes + kk + q / 2 * 16);
            // Bt, two n8 tiles per ldmatrix.x4: (n 0-7, bytes 0-15), (0-7, 16-31),
            // (8-15, 0-15), (8-15, 16-31) -> b[j] and b[j+1].
            unsigned b[kFragN][2];
            #pragma unroll
            for (int j = 0; j < kFragN; j += 2)
                ldmatrix_x4(&b[j][0], Bs + (wn + j * 8 + lane % 8 + q / 2 * 8) * kRowBytes + kk + q % 2 * 16);
            #pragma unroll
            for (int i = 0; i < kFragM; ++i)
                #pragma unroll
                for (int j = 0; j < kFragN; ++j)
                    F::mma(acc[i][j], a[i], b[j]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

    // Lane (g = lane/4, t = lane%4) holds C(g, 2t..2t+1) and C(g+8, 2t..2t+1)
    // of each 16x8 tile: one 8-byte (4-byte for fp16) store per pair.
    const int g = lane / 4, t = lane % 4;
    #pragma unroll
    for (int i = 0; i < kFragM; ++i)
        #pragma unroll
        for (int j = 0; j < kFragN; ++j)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                Out* dst = C + static_cast<std::size_t>(wm + i * 16 + g + h * 8) * N + wn + j * 8 + t * 2;
                if constexpr (F::kAccRegs == 2)
                    *reinterpret_cast<unsigned*>(dst) = acc[i][j][h];
                else {
                    using Pair = std::conditional_t<std::is_same_v<Out, float>, float2, int2>;
                    *reinterpret_cast<Pair*>(dst) = Pair{acc[i][j][2 * h], acc[i][j][2 * h + 1]};
                }
            }
#else
    __trap();
#endif
}

__global__ void kernel_fp8_probe(int* out) {
#if defined(HPC_LOWP_FP8_BLOCK_SCALED)
    *out = 1;
#else
    *out = 0;
#endif
}

template <typename F, typename Out>
void launch(const void* A, const void* Bt, void* C, int M, int K, int N) {
    static const bool configured = [] {
        LOWP_CUDA_CHECK(cudaFuncSetAttribute(kernel_lowp<F, Out>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
        return true;
    }();
    (void)configured;
    const unsigned blocks = static_cast<unsigned>(M / kBM) * static_cast<unsigned>(N / kBN);
    kernel_lowp<F, Out><<<blocks, kThreads, kSmemBytes>>>(static_cast<const unsigned char*>(A),
                                                           static_cast<const unsigned char*>(Bt),
                                                           static_cast<Out*>(C), M, K, N);
    LOWP_CUDA_CHECK(cudaGetLastError());
}

int element_bytes(CudaLowp f) { return f == CudaLowp::int8 || f == CudaLowp::fp8_e4m3 ? 1 : 2; }
int output_bytes(CudaLowp f) { return f == CudaLowp::fp16_acc16 ? 2 : 4; }

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
    const int cc = prop.major * 10 + prop.minor;
    try { cache.emplace(device, cc); } catch (...) {}
    return cc;
}

// Shared argument checks. Returns true when there is nothing to multiply.
bool prepare(CudaLowp f, void* C, int M, int K, int N) {
    if (M < 0 || K < 0 || N < 0)
        throw std::invalid_argument("low-precision CUDA GEMM: negative dimension");
    if (!hpc::gemm::cuda_lowp_supported(f))
        throw std::runtime_error("low-precision CUDA GEMM: format not supported on this device");
    if (M == 0 || N == 0)
        return true;
    if (K == 0) {
        LOWP_CUDA_CHECK(cudaMemsetAsync(C, 0, static_cast<std::size_t>(M) * N * output_bytes(f)));
        return true;
    }
    return false;
}

// ============================================================================
// cuBLAS reference. cuBLAS is column-major, so it computes the column-major
// C^T = B^T A^T (N x M). The row-major N x K buffer Bt is, column-major, a
// K x N matrix: op T turns it into B^T. The row-major M x K buffer A is,
// column-major, K x M = A^T, used as is (op N). That is the "TN" call
// cuBLASLt requires for FP8; every format uses it here.
// ============================================================================
cublasHandle_t cublas() {
    static cublasHandle_t h = [] {
        cublasHandle_t handle;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasCreate failed");
        return handle;
    }();
    return h;
}

cublasLtHandle_t cublas_lt() {
    static cublasLtHandle_t h = [] {
        cublasLtHandle_t handle;
        if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasLtCreate failed");
        return handle;
    }();
    return h;
}

void check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed: status " + std::to_string(int(st)));
}

void cublas_fp8(const void* A, const void* Bt, float* C, int M, int K, int N) {
    constexpr std::size_t kWorkspace = 32u << 20;
    static void* workspace = [] {
        void* p = nullptr;
        LOWP_CUDA_CHECK(cudaMalloc(&p, kWorkspace));
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
    check(cublasLtMatmulDescCreate(&d.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
    const cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof opT), "TRANSA");
    check(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof opN), "TRANSB");
    check(cublasLtMatrixLayoutCreate(&d.a, CUDA_R_8F_E4M3, K, N, K), "layout A");
    check(cublasLtMatrixLayoutCreate(&d.b, CUDA_R_8F_E4M3, K, M, K), "layout B");
    check(cublasLtMatrixLayoutCreate(&d.c, CUDA_R_32F, N, M, N), "layout C");
    check(cublasLtMatmulPreferenceCreate(&d.pref), "preference");
    check(cublasLtMatmulPreferenceSetAttribute(d.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspace, sizeof kWorkspace),
          "workspace");
    cublasLtMatmulHeuristicResult_t heuristic{};
    int found = 0;
    check(cublasLtMatmulAlgoGetHeuristic(cublas_lt(), d.op, d.a, d.b, d.c, d.c, d.pref, 1, &heuristic, &found),
          "cublasLtMatmulAlgoGetHeuristic");
    if (found == 0)
        throw std::runtime_error("cuBLASLt has no FP8 algorithm for this shape");
    const float alpha = 1.0f, beta = 0.0f;
    check(cublasLtMatmul(cublas_lt(), d.op, &alpha, Bt, d.a, A, d.b, &beta, C, d.c, C, d.c,
                         &heuristic.algo, workspace, kWorkspace, nullptr),
          "cublasLtMatmul (FP8)");
}

// ============================================================================
// Matrix wrappers: pad to tile multiples, transpose B, run, copy back.
// ============================================================================
struct DeviceBytes {
    void* ptr = nullptr;
    explicit DeviceBytes(std::size_t n) { LOWP_CUDA_CHECK(cudaMalloc(&ptr, n ? n : 1)); }
    ~DeviceBytes() { cudaFree(ptr); }
    DeviceBytes(const DeviceBytes&)            = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
};

int round_up(std::size_t v, int m) {
    const std::size_t r = (v + m - 1) / m * m;
    if (r > static_cast<std::size_t>(INT32_MAX))
        throw std::invalid_argument("low-precision CUDA GEMM: dimension too large");
    return static_cast<int>(r);
}

template <typename In, typename Out>
void run_padded(CudaLowp f, bool use_cublas, const hpc::Matrix<In>& A, const hpc::Matrix<In>& B,
                hpc::Matrix<Out>& C) {
    if (A.cols() != B.rows() || C.rows() != A.rows() || C.cols() != B.cols())
        throw std::invalid_argument("low-precision CUDA GEMM: shape mismatch");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    if (M == 0 || N == 0)
        return;
    const int Mp = round_up(M, kBM), Np = round_up(N, kBN);
    const int Kp = round_up(K, hpc::gemm::cuda_lowp_k_multiple(f));

    // Zero-padded A (Mp x Kp) and transposed B (Np x Kp) on the host. Zero
    // is all-zero bits in every format here.
    static_assert(sizeof(In) == 1 || sizeof(In) == 2);
    std::vector<In> a(static_cast<std::size_t>(Mp) * Kp), bt(static_cast<std::size_t>(Np) * Kp);
    std::memset(a.data(), 0, a.size() * sizeof(In));
    std::memset(bt.data(), 0, bt.size() * sizeof(In));
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k)
            a[i * Kp + k] = A(i, k);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j)
            bt[j * Kp + k] = B(k, j);

    const std::size_t c_bytes = static_cast<std::size_t>(Mp) * Np * output_bytes(f);
    DeviceBytes dA(a.size() * sizeof(In)), dB(bt.size() * sizeof(In)), dC(c_bytes);
    LOWP_CUDA_CHECK(cudaMemcpy(dA.ptr, a.data(), a.size() * sizeof(In), cudaMemcpyHostToDevice));
    LOWP_CUDA_CHECK(cudaMemcpy(dB.ptr, bt.data(), bt.size() * sizeof(In), cudaMemcpyHostToDevice));
    if (use_cublas)
        hpc::gemm::gemm_cuda_cublas_lowp_device(f, dA.ptr, dB.ptr, dC.ptr, Mp, Kp, Np);
    else
        hpc::gemm::gemm_cuda_lowp_device(f, dA.ptr, dB.ptr, dC.ptr, Mp, Kp, Np);
    LOWP_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned char> c(c_bytes);
    LOWP_CUDA_CHECK(cudaMemcpy(c.data(), dC.ptr, c_bytes, cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            const unsigned char* p = c.data() + (i * Np + j) * output_bytes(f);
            if (f == CudaLowp::fp16_acc16) {
                hpc::fp16 h;
                std::memcpy(&h.bits, p, 2);
                C(i, j) = static_cast<Out>(hpc::to_float(h));
            } else {
                std::memcpy(&C(i, j), p, sizeof(Out));
            }
        }
}

}  // namespace

namespace hpc::gemm {

bool cuda_lowp_supported(CudaLowp format) noexcept {
    const int cc = device_capability();
    return format == CudaLowp::fp8_e4m3 ? cc >= 89 : cc >= 80;
}

int cuda_lowp_k_multiple(CudaLowp format) noexcept { return kBK / element_bytes(format); }

bool cuda_lowp_fp8_block_scaled() {
    if (!cuda_lowp_supported(CudaLowp::fp8_e4m3))
        return false;
    int* d = nullptr;
    int h  = 0;
    LOWP_CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
    kernel_fp8_probe<<<1, 1>>>(d);
    const cudaError_t e = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    LOWP_CUDA_CHECK(e);
    return h == 1;
}

void gemm_cuda_lowp_device(CudaLowp f, const void* A, const void* Bt, void* C, int M, int K, int N) {
    if (prepare(f, C, M, K, N))
        return;
    if (M % kBM || N % kBN || K % cuda_lowp_k_multiple(f))
        throw std::invalid_argument("low-precision CUDA GEMM: M, N must be multiples of 128 and K of " +
                                    std::to_string(cuda_lowp_k_multiple(f)));
    switch (f) {
        case CudaLowp::bf16:       return launch<Bf16, float>(A, Bt, C, M, K, N);
        case CudaLowp::fp16:       return launch<Fp16, float>(A, Bt, C, M, K, N);
        case CudaLowp::fp16_acc16: return launch<Fp16Acc16, __half>(A, Bt, C, M, K, N);
        case CudaLowp::int8:       return launch<Int8, int>(A, Bt, C, M, K, N);
        case CudaLowp::fp8_e4m3:   return launch<Fp8, float>(A, Bt, C, M, K, N);
    }
}

void gemm_cuda_cublas_lowp_device(CudaLowp f, const void* A, const void* Bt, void* C, int M, int K, int N) {
    if (prepare(f, C, M, K, N))
        return;
    const float one = 1.0f, zero = 0.0f;
    const __half one_h = __float2half(1.0f), zero_h = __float2half(0.0f);
    const int one_i = 1, zero_i = 0;
    auto gemm_ex = [&](cudaDataType in, cudaDataType out, cublasComputeType_t compute, const void* alpha,
                       const void* beta) {
        check(cublasGemmEx(cublas(), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha, Bt, in, K, A, in, K, beta, C,
                           out, N, compute, CUBLAS_GEMM_DEFAULT),
              "cublasGemmEx");
    };
    switch (f) {
        case CudaLowp::bf16:       return gemm_ex(CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, &one, &zero);
        case CudaLowp::fp16:       return gemm_ex(CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, &one, &zero);
        case CudaLowp::fp16_acc16: return gemm_ex(CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, &one_h, &zero_h);
        case CudaLowp::int8:       return gemm_ex(CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, &one_i, &zero_i);
        case CudaLowp::fp8_e4m3:   return cublas_fp8(A, Bt, static_cast<float*>(C), M, K, N);
    }
}

void gemm_cuda_bf16(const Matrix<bf16>& A, const Matrix<bf16>& B, Matrix<float>& C, bool use_cublas) {
    run_padded(CudaLowp::bf16, use_cublas, A, B, C);
}
void gemm_cuda_fp16(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C, bool use_cublas) {
    run_padded(CudaLowp::fp16, use_cublas, A, B, C);
}
void gemm_cuda_fp16_acc16(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C, bool use_cublas) {
    run_padded(CudaLowp::fp16_acc16, use_cublas, A, B, C);
}
void gemm_cuda_i8(const Matrix<std::int8_t>& A, const Matrix<std::int8_t>& B, Matrix<std::int32_t>& C,
                  bool use_cublas) {
    run_padded(CudaLowp::int8, use_cublas, A, B, C);
}
void gemm_cuda_fp8(const Matrix<fp8_e4m3>& A, const Matrix<fp8_e4m3>& B, Matrix<float>& C, bool use_cublas) {
    run_padded(CudaLowp::fp8_e4m3, use_cublas, A, B, C);
}

}  // namespace hpc::gemm
