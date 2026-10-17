/**
 * @file gemm_lowp.cu
 * @brief Low-precision Tensor Core GEMM: bf16, fp16, int8 and FP8 inputs.
 *
 * The public API and the format table are in gemm/cuda_lowp.hpp. Two
 * kernels share the format structs below: kernel_lowp, described here, and
 * kernel_tma (Level 8 of gemm/cuda.hpp), described at its definition.
 *
 * kernel_lowp is the raw mma.sync kernel of Level 6 (gemm_kernels.cu) grown
 * to the tile sizes and pipeline of Level 7:
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
 *     8–18% slower on an RTX 5080 at every size and format.
 *   - Grouped block order (kGroupM, below), for L2 reuse.
 *   - Shared-memory rows padded by 16 bytes (80-byte stride): the eight
 *     16-byte rows one ldmatrix phase reads start in banks 0, 20, 8, 28,
 *     16, 4, 24, 12, so they never conflict.
 *
 * Shapes are exact tile multiples (the Matrix wrappers pad), so the kernel
 * has no bounds checks.
 */

#include "gemm/cuda.hpp"
#include "gemm/cuda_lowp.hpp"

#include <cublasLt.h>
#include <cuda.h>  // CUtensorMap, cuTensorMapEncodeTiled (Level 8)
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
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
// Tiling of kernel_lowp. HPC_LOWP_BN / _STAGES / _GROUP_M override the
// measured defaults (docs/experiments/lowp/tuning_cp_async.csv was swept
// with -DCMAKE_CUDA_FLAGS="-DHPC_LOWP_STAGES=3 ..."); normal builds leave
// them alone.
// ============================================================================
#ifndef HPC_LOWP_BN
#define HPC_LOWP_BN 128
#endif
#ifndef HPC_LOWP_STAGES
#define HPC_LOWP_STAGES 2
#endif
#ifndef HPC_LOWP_GROUP_M
#define HPC_LOWP_GROUP_M 8
#endif
constexpr int kBM = 128, kBN = HPC_LOWP_BN;  // block tile
constexpr int kBK = 64;                     // bytes of K per pipeline stage
constexpr int kStages = HPC_LOWP_STAGES;    // cp.async pipeline depth
constexpr int kGroupM = HPC_LOWP_GROUP_M;   // block rasterization, see kernel_lowp
constexpr int kRowBytes = kBK + 16;         // padded shared-memory row: 80 B
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

// ============================================================================
// Level 8: TMA, mbarriers and warp specialisation (sm_90+).
//
// kernel_lowp spends most of its instructions on moving data: every thread
// computes addresses and issues cp.async for 16 bytes at a time, and every
// ldmatrix address goes through the padded-row arithmetic. On the RTX 5080
// it executes 69 M warp instructions for the 16.8 M mma of an N = 4096 FP8
// GEMM, cuBLAS 37 M. Level 8 removes that overhead the way cuBLAS does:
//
//   1. TMA (cp.async.bulk.tensor): one thread copies a whole 128-row (A)
//      or 256-row (Bt) x 64-byte tile with one instruction. The copy engine walks
//      the tensor described by a CUtensorMap built on the host.
//   2. Hardware swizzle instead of padding. The tensor map asks TMA to
//      store each tile with the 64-byte swizzle: 16-byte chunk c of row r
//      lands at chunk c ^ ((r / 2) % 4). The eight 16-byte rows one ldmatrix
//      phase reads then start in banks 0, 16, 4, 20, 8, 24, 12, 28 (for
//      chunk 0; a permutation of them for the others): conflict-free, with
//      no padding bytes, and the XOR term is a per-lane constant.
//   3. Warp specialisation. One producer warp only issues TMA copies; the
//      consumer warps only run ldmatrix and mma.sync. They synchronise
//      through two mbarriers per stage: `full` (the producer announces the
//      bytes it expects; TMA completes the transaction) and `empty` (each
//      consumer warp arrives when it has read the stage). No __syncthreads()
//      in the main loop.
//   4. A 128 x 256 block tile, 8 consumer warps of 64 x 64, 4 stages of
//      24 KiB = 96 KiB: one block per SM. Each warp's 4 A and 4 B
//      ldmatrix.x4 feed 32 mma per 32 bytes of K (kernel_lowp: 6 for 16),
//      and each byte brought into shared memory feeds twice the mma it does
//      in a 128 x 128 tile. When 128 x 256 tiles would leave SMs idle, a
//      128 x 128 tile (4 consumer warps, 6 stages) runs instead (launch_tma).
//
// Measured on an RTX 5080 at N = 2048 and 8192, all formats: 128 x 256 with
// 3 or 4 stages is 3-11% faster than 128 x 128 with 4, 5 or 6 stages; the
// stage count changes little once there are 3.
//
// Shapes must be tile multiples, M and N of 128 (the wrappers pad).
// Block order is kernel_lowp's grouped order.
// ============================================================================
constexpr int kTmaBM = 128;

// Block tile 128 x BN with BN = 256 (8 consumer warps, 4 stages of 24 KiB)
// or 128 (4 consumer warps, 6 stages of 16 KiB); both fill 96 KiB.
template <int BN>
struct TmaTile {
    static constexpr int kStages     = BN == 256 ? 4 : 6;
    static constexpr int kWarpsN     = BN / 64;
    static constexpr int kConsumers  = (kTmaBM / 64) * kWarpsN;
    static constexpr int kThreads    = (kConsumers + 1) * 32;  // + 1 producer warp
    static constexpr int kStageA     = kTmaBM * kBK;
    static constexpr int kStageBytes = kStageA + BN * kBK;
    static constexpr int kSmemBytes  = kStages * kStageBytes + 1024 + 2 * kStages * 8;
    static_assert(kSmemBytes <= 101376, "sm_120 allows 99 KiB of shared memory per block");
};

__device__ __forceinline__ void mbar_init(std::uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(std::uint64_t* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void mbar_arrive(std::uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_addr(bar)) : "memory");
}
// Spin until the barrier's phase with this parity has completed.
__device__ __forceinline__ void mbar_wait(std::uint64_t* bar, unsigned parity) {
    unsigned done = 0;
    while (!done)
        asm volatile("{\n .reg .pred p;\n"
                     " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     " selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(done)
                     : "r"(smem_addr(bar)), "r"(parity)
                     : "memory");
}
// TMA: copy the box at (byte column x, row y) of the tensor into dst and
// complete `bytes` of the barrier's transaction count.
__device__ __forceinline__ void tma_load(void* dst, const CUtensorMap* map, std::uint64_t* bar, int x, int y) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4}], [%2];\n" ::"r"(smem_addr(dst)),
                 "l"(reinterpret_cast<std::uint64_t>(map)), "r"(smem_addr(bar)), "r"(x), "r"(y)
                 : "memory");
}

template <typename F, typename Out, int BN>
__global__ void __launch_bounds__(TmaTile<BN>::kThreads, 1)
kernel_tma(const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b,
           Out* __restrict__ C, int M, int K, int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using T = TmaTile<BN>;
    constexpr int kTmaStages = T::kStages, kTmaStageBytes = T::kStageBytes, kTmaStageA = T::kStageA;
    constexpr int kTmaConsumers = T::kConsumers, kTmaWarpsN = T::kWarpsN, kTmaBN = BN;
    extern __shared__ unsigned char smem_raw[];
    // The swizzle pattern is a function of the shared-memory address: stages
    // must start on a 1024-byte boundary (512 suffices for 64-byte swizzle).
    unsigned char* smem = smem_raw + ((1024 - (smem_addr(smem_raw) & 1023)) & 1023);
    auto stage_a = [&](int s) { return smem + s * kTmaStageBytes; };
    auto stage_b = [&](int s) { return smem + s * kTmaStageBytes + kTmaStageA; };
    std::uint64_t* full  = reinterpret_cast<std::uint64_t*>(smem + kTmaStages * kTmaStageBytes);
    std::uint64_t* empty = full + kTmaStages;

    const int tiles_m = M / kTmaBM, tiles_n = N / kTmaBN;
    const int group = blockIdx.x / (kGroupM * tiles_n), first_m = group * kGroupM;
    const int group_m = min(tiles_m - first_m, kGroupM);
    const int in_group = blockIdx.x % (kGroupM * tiles_n);
    const int tile_m = first_m + in_group % group_m, tile_n = in_group / group_m;
    const int k_tiles = K * F::kBytes / kBK;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

    if (threadIdx.x == 0) {
        for (int s = 0; s < kTmaStages; ++s) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], kTmaConsumers);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (warp == kTmaConsumers) {  // producer
        if (lane == 0) {
            for (int kt = 0; kt < k_tiles; ++kt) {
                const int s = kt % kTmaStages;
                if (kt >= kTmaStages)  // wait until every consumer has read the previous use of s
                    mbar_wait(&empty[s], (kt / kTmaStages - 1) & 1);
                mbar_expect_tx(&full[s], kTmaStageBytes);
                tma_load(stage_a(s), &map_a, &full[s], kt * kBK, tile_m * kTmaBM);
                tma_load(stage_b(s), &map_b, &full[s], kt * kBK, tile_n * kTmaBN);
            }
        }
        return;
    }

    // Consumers: warp tile 64 x 64 = 4 m16 x 8 n8 mma tiles.
    const int wm = warp / kTmaWarpsN * 64, wn = warp % kTmaWarpsN * 64;
    const int q = lane / 8;
    const int x = (lane % 8) / 2;  // swizzle term: (row / 2) % 4 of every row this lane addresses
    typename F::Acc acc[4][8][F::kAccRegs] = {};

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int s = kt % kTmaStages;
        mbar_wait(&full[s], (kt / kTmaStages) & 1);
        const unsigned char* As = stage_a(s);
        const unsigned char* Bs = stage_b(s);
        #pragma unroll
        for (int kk = 0; kk < kBK / 16; kk += 2) {  // 16-byte chunk index: 0 and 2
            unsigned a[4][4], b[8][2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                ldmatrix_x4(a[i], As + (wm + i * 16 + lane % 8 + q % 2 * 8) * kBK + ((kk + q / 2) ^ x) * 16);
            #pragma unroll
            for (int j = 0; j < 8; j += 2)
                ldmatrix_x4(&b[j][0], Bs + (wn + j * 8 + lane % 8 + q / 2 * 8) * kBK + ((kk + q % 2) ^ x) * 16);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    F::mma(acc[i][j], a[i], b[j]);
        }
        __syncwarp();
        if (lane == 0)
            mbar_arrive(&empty[s]);  // after the mma that consumed every fragment of stage s
    }

    C += static_cast<std::size_t>(tile_m) * kTmaBM * N + static_cast<std::size_t>(tile_n) * kTmaBN;
    const int g = lane / 4, t = lane % 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
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

// A K-major operand (rows x row_bytes, row-major) as a TMA tensor of bytes,
// copied in boxes of `box_rows` x 64 bytes with the 64-byte swizzle. Encoding
// costs a few microseconds of host time, so maps are cached per buffer.
const CUtensorMap* tensor_map(const void* ptr, int rows, std::size_t row_bytes, int box_rows) {
    static std::mutex mutex;
    static std::map<std::tuple<const void*, int, std::size_t, int>, CUtensorMap> cache;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_tuple(ptr, rows, row_bytes, box_rows);
    if (const auto it = cache.find(key); it != cache.end())
        return &it->second;
    CUtensorMap map{};
    const cuuint64_t dims[2]    = {row_bytes, static_cast<cuuint64_t>(rows)};
    const cuuint64_t strides[1] = {row_bytes};
    const cuuint32_t box[2]     = {static_cast<cuuint32_t>(kBK), static_cast<cuuint32_t>(box_rows)};
    const cuuint32_t step[2]    = {1, 1};
    const CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims,
                                              strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                              CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS)
        throw std::runtime_error("cuTensorMapEncodeTiled failed: " + std::to_string(int(r)));
    if (cache.size() > 256)  // buffers come and go; keep the cache bounded
        cache.clear();
    return &cache.emplace(key, map).first->second;
}

int sm_count() {
    int device = 0, count = 0;
    LOWP_CUDA_CHECK(cudaGetDevice(&device));
    LOWP_CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
    return count;
}

template <typename F, typename Out, int BN>
void launch_tma_tile(const void* A, const void* Bt, void* C, int M, int K, int N) {
    using T = TmaTile<BN>;
    static const bool configured = [] {
        LOWP_CUDA_CHECK(cudaFuncSetAttribute(kernel_tma<F, Out, BN>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             T::kSmemBytes));
        return true;
    }();
    (void)configured;
    const std::size_t row_bytes = static_cast<std::size_t>(K) * F::kBytes;
    const CUtensorMap map_a = *tensor_map(A, M, row_bytes, kTmaBM);
    const CUtensorMap map_b = *tensor_map(Bt, N, row_bytes, BN);
    const unsigned blocks = static_cast<unsigned>(M / kTmaBM) * static_cast<unsigned>(N / BN);
    kernel_tma<F, Out, BN><<<blocks, T::kThreads, T::kSmemBytes>>>(map_a, map_b, static_cast<Out*>(C), M, K, N);
    LOWP_CUDA_CHECK(cudaGetLastError());
}

// Tile choice by wave quantisation. One block runs per SM, so a grid of B
// blocks takes ceil(B / SMs) waves and the last one may run part-empty:
// the useful fraction is B / (ceil(B / SMs) * SMs). The 128 x 256 tile
// makes the GEMM 3-11% faster when both tiles fill the GPU equally
// (measured at N = 2048 and 8192); the rule assumes 5% and chooses it
// unless the 128 x 128 tile's twice as many, smaller blocks fill the waves
// more than 5% better. On an RTX 5080 (84 SMs) that picks 128 x 128 at
// N = 1024 (32 wide blocks: 38% of one wave) and N = 4096 (512 wide blocks:
// 6.1 waves, 87% useful, against 94%), 128 x 256 at N = 2048, 8192, 16384.
double wave_efficiency(long blocks, int sms) {
    const long waves = (blocks + sms - 1) / sms;
    return double(blocks) / double(waves * sms);
}

template <typename F, typename Out>
void launch_tma(const void* A, const void* Bt, void* C, int M, int K, int N) {
    static const int sms = sm_count();
    const long tiles_m = M / kTmaBM;
    if (N % 256 == 0 &&
        1.05 * wave_efficiency(tiles_m * (N / 256), sms) >= wave_efficiency(tiles_m * (N / 128), sms))
        launch_tma_tile<F, Out, 256>(A, Bt, C, M, K, N);
    else
        launch_tma_tile<F, Out, 128>(A, Bt, C, M, K, N);
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
void run_padded(CudaLowp f, hpc::gemm::CudaLowpImpl impl, const hpc::Matrix<In>& A, const hpc::Matrix<In>& B,
                hpc::Matrix<Out>& C) {
    using hpc::gemm::CudaLowpImpl;
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
    switch (impl) {
        case CudaLowpImpl::mma:    hpc::gemm::gemm_cuda_lowp_device(f, dA.ptr, dB.ptr, dC.ptr, Mp, Kp, Np); break;
        case CudaLowpImpl::tma:    hpc::gemm::gemm_cuda_lowp_tma_device(f, dA.ptr, dB.ptr, dC.ptr, Mp, Kp, Np); break;
        case CudaLowpImpl::cublas: hpc::gemm::gemm_cuda_cublas_lowp_device(f, dA.ptr, dB.ptr, dC.ptr, Mp, Kp, Np); break;
    }
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

void gemm_cuda_lowp_tma_device(CudaLowp f, const void* A, const void* Bt, void* C, int M, int K, int N) {
    if (prepare(f, C, M, K, N))
        return;
    if (!cuda_has_tma())
        throw std::runtime_error("Level 8 (TMA) needs sm_90+");
    if (M % kTmaBM || N % 128 || K % cuda_lowp_k_multiple(f))
        throw std::invalid_argument("Level 8 CUDA GEMM: M, N must be multiples of 128 and K of " +
                                    std::to_string(cuda_lowp_k_multiple(f)));
    switch (f) {
        case CudaLowp::bf16:       return launch_tma<Bf16, float>(A, Bt, C, M, K, N);
        case CudaLowp::fp16:       return launch_tma<Fp16, float>(A, Bt, C, M, K, N);
        case CudaLowp::fp16_acc16: return launch_tma<Fp16Acc16, __half>(A, Bt, C, M, K, N);
        case CudaLowp::int8:       return launch_tma<Int8, int>(A, Bt, C, M, K, N);
        case CudaLowp::fp8_e4m3:   return launch_tma<Fp8, float>(A, Bt, C, M, K, N);
    }
}

void gemm_cuda_tma_device(const void* A16, const void* Bt16, float* C, int M, int K, int N) {
    gemm_cuda_lowp_tma_device(CudaLowp::fp16, A16, Bt16, C, M, K, N);
}

// Level 8 on FP32 matrices: inputs rounded to fp16 (as Levels 5-7 do), B
// transposed to N x K while converting; no TMA (pre-sm_90) runs Level 7.
void gemm_cuda_tma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    if (!cuda_has_tma())
        return gemm_cuda_wmma_pipelined(A, B, C);
    run_padded(CudaLowp::fp16, CudaLowpImpl::tma, convert<fp16>(A), convert<fp16>(B), C);
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

void gemm_cuda_bf16(const Matrix<bf16>& A, const Matrix<bf16>& B, Matrix<float>& C, CudaLowpImpl impl) {
    run_padded(CudaLowp::bf16, impl, A, B, C);
}
void gemm_cuda_fp16(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C, CudaLowpImpl impl) {
    run_padded(CudaLowp::fp16, impl, A, B, C);
}
void gemm_cuda_fp16_acc16(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C, CudaLowpImpl impl) {
    run_padded(CudaLowp::fp16_acc16, impl, A, B, C);
}
void gemm_cuda_i8(const Matrix<std::int8_t>& A, const Matrix<std::int8_t>& B, Matrix<std::int32_t>& C,
                  CudaLowpImpl impl) {
    run_padded(CudaLowp::int8, impl, A, B, C);
}
void gemm_cuda_fp8(const Matrix<fp8_e4m3>& A, const Matrix<fp8_e4m3>& B, Matrix<float>& C, CudaLowpImpl impl) {
    run_padded(CudaLowp::fp8_e4m3, impl, A, B, C);
}

}  // namespace hpc::gemm
