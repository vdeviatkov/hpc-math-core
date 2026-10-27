/**
 * @file gemm.cu
 * @brief Every CUDA GEMM: Levels 0-8, the low-precision formats and the
 *        cuBLAS reference. The public API and the level table are in
 *        gemm/cuda.hpp.
 *
 * The file reads top to bottom in four parts:
 *
 *   1. Device building blocks: PTX wrappers, tile copies, the cp.async
 *      pipeline. Every kernel below is assembled from them.
 *   2. Kernels, Level 0 to Level 8. Each one's comment says what it adds
 *      over the level it builds on.
 *   3. Host launchers: one *_device entry point per kernel. Each checks its
 *      hardware and shape requirements and falls back to a lower level.
 *   4. The cuBLAS reference and the Matrix wrappers (copy in, run, copy out).
 *
 * Addressing convention: a kernel first moves A, B and C to the top-left
 * corner of its block's tile (block_tile), then works in tile-local
 * coordinates. `rows` and `cols` are how much of the C tile lies inside the
 * M x N matrix, so a bounds check is a comparison against them, or against
 * K - k0 along K. The Tensor Core kernels of Levels 7-8 and the
 * low-precision formats take exact tile multiples and have no bounds checks.
 *
 * The GPU memory hierarchy these kernels work against: docs/cache-behavior.md,
 * section 9.
 */

#include "gemm/cuda.hpp"
#include "hpc/matrix.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>  // CUtensorMap, cuTensorMapEncodeTiled (Level 8)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace nvcuda;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define HPC_SM80 1  // cp.async, ldmatrix, mma.sync m16n8k16
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define HPC_SM90 1  // TMA, mbarrier transactions
#endif
// FP8 has two mma instructions on sm_120 (RTX 50). The sm_89 form runs at
// half the Tensor Core rate with an fp32 accumulator (249 vs 497 TFLOP/s on
// an RTX 5080, benchmarks/experiments/cuda_peak.cu). The block-scaled "MX"
// form runs at full rate: it multiplies each operand by a per-32-element
// power-of-two scale (UE8M0), and with every scale = 2^0 its result is the
// plain product. It needs the arch-specific target, so it is used when the
// library is built with CMAKE_CUDA_ARCHITECTURES=120a.
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ == 1200
#define HPC_FP8_BLOCK_SCALED 1
#endif

#define CUDA_CHECK(expr)                                                                       \
    do {                                                                                       \
        const cudaError_t _e = (expr);                                                         \
        if (_e != cudaSuccess)                                                                 \
            throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") +              \
                                     std::to_string(__LINE__) + ": " + cudaGetErrorString(_e)); \
    } while (0)

namespace {

using hpc::Matrix;
using hpc::gemm::CudaLowp;
using hpc::gemm::CudaLowpImpl;

// ############################################################################
// Part 1. Device building blocks
// ############################################################################

/// Shared-memory address of a generic pointer, as PTX's .shared operands take it.
__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

/// Copy one 4- or 8-byte element from global `src` to shared `dst`, or write
/// zeros when !valid (`src` must still be a readable address). On sm_80+
/// this is an asynchronous cp.async, finished by cp_async_wait; older GPUs
/// copy synchronously.
template <int BYTES>
__device__ __forceinline__ void cp_async_or_zero(void* dst, const void* src, bool valid) {
    static_assert(BYTES == 4 || BYTES == 8, "an element is 4 or 8 bytes");
#ifdef HPC_SM80
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_addr(dst)), "l"(src), "n"(BYTES),
                 "r"(valid ? BYTES : 0));
#else
    using Word = std::conditional_t<BYTES == 4, unsigned, uint2>;
    *static_cast<Word*>(dst) = valid ? *static_cast<const Word*>(src) : Word{};
#endif
}

/// Copy one 16-byte chunk from global `src` to shared `dst`, both 16-byte
/// aligned: cp.async on sm_80+ (.cg bypasses L1; a block reads each tile
/// once), a synchronous copy on older GPUs.
__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
#ifdef HPC_SM80
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
#else
    *static_cast<uint4*>(dst) = *static_cast<const uint4*>(src);
#endif
}

/// Close the group of cp.async copies this thread issued since the last commit.
__device__ __forceinline__ void cp_async_commit() {
#ifdef HPC_SM80
    asm volatile("cp.async.commit_group;\n");
#endif
}

/// Wait until at most PENDING of this thread's committed groups are in
/// flight. Not a compiler memory barrier: follow it with __syncthreads(),
/// as cp_async_pipeline does, before reading the copied tile.
template <int PENDING>
__device__ __forceinline__ void cp_async_wait() {
#ifdef HPC_SM80
    asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING));
#endif
}

/// Stream `k_tiles` k-tiles through STAGES shared-memory buffers:
/// load(kt, stage) issues the copies of k-tile kt into a buffer, compute(stage)
/// consumes one. Copies run STAGES - 1 tiles ahead of the math, so on sm_80+
/// they overlap it. The one __syncthreads() per tile both publishes tile kt to
/// every thread and frees the buffer tile kt - 1 used, which is refilled next.
template <int STAGES, typename Load, typename Compute>
__device__ __forceinline__ void cp_async_pipeline(int k_tiles, Load load, Compute compute) {
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < k_tiles)
            load(s, s);
        cp_async_commit();
    }
    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_async_wait<STAGES - 2>();  // this thread's copies of tile kt have landed
        __syncthreads();              // ... every thread's, and nobody still reads tile kt - 1
        const int next = kt + STAGES - 1;
        if (next < k_tiles)
            load(next, next % STAGES);
        cp_async_commit();  // possibly empty: keeps the group count uniform
        compute(kt % STAGES);
    }
    cp_async_wait<0>();
}

/// ldmatrix.x4: four 8x8 matrices of 16-bit elements from shared memory.
/// Lane l supplies the address of row l % 8 of matrix l / 8; r[i] receives
/// this lane's part of matrix i, in the layout mma.sync expects.
__device__ __forceinline__ void ldmatrix_x4(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}

/// ldmatrix.x2.trans: two 8x8 matrices, each transposed on the way into
/// registers. Lanes 0-15 supply the row addresses; lanes 16-31 must pass
/// valid addresses too, which are ignored.
__device__ __forceinline__ void ldmatrix_x2_trans(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(smem_addr(p)));
}

/// Initialise an mbarrier that completes a phase after `count` arrivals.
__device__ __forceinline__ void mbar_init(std::uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(count));
}

/// Arrive on the barrier and announce `bytes` that TMA copies will deliver to it.
__device__ __forceinline__ void mbar_expect_tx(std::uint64_t* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(bytes)
                 : "memory");
}

/// Arrive on the barrier.
__device__ __forceinline__ void mbar_arrive(std::uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_addr(bar)) : "memory");
}

/// Spin until the barrier's phase with this parity has completed.
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

/// TMA: copy the box at (byte column x, row y) of the tensor `map` describes
/// into shared `dst`, completing its bytes on barrier `bar`.
__device__ __forceinline__ void tma_load(void* dst, const CUtensorMap* map, std::uint64_t* bar, int x, int y) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4}], [%2];\n" ::"r"(smem_addr(dst)),
                 "l"(reinterpret_cast<std::uint64_t>(map)), "r"(smem_addr(bar)), "r"(x), "r"(y)
                 : "memory");
}

/// Move A, B and C to the top-left corner of this block's BM x BN tile of C
/// and return how many of its rows (x) and columns (y) lie inside the matrix.
template <int BM, int BN, typename PA, typename PC>
__device__ __forceinline__ int2 block_tile(PA& A, PA& B, PC& C, int M, int K, int N) {
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    A += row0 * K;
    B += col0;
    C += row0 * N + col0;
    return {min(M - row0, BM), min(N - col0, BN)};
}

/// Load the ROWS x COLS tile at `src` (row-major, leading dimension ld) into
/// shared memory at dst (leading dimension dst_ld), converting S to D and
/// writing zeros outside the `rows` x `cols` part inside the matrix.
/// Consecutive threads take consecutive columns, so the global reads
/// coalesce. TRANSPOSED stores element (r, c) at dst[c][r]. ASYNC copies
/// with cp.async, which cannot convert.
template <int ROWS, int COLS, bool TRANSPOSED, bool ASYNC = false, typename D, typename S>
__device__ __forceinline__ void load_tile(D* dst, int dst_ld, const S* src, int ld, int rows, int cols) {
    for (int e = threadIdx.x; e < ROWS * COLS; e += blockDim.x) {
        const int r = e / COLS, c = e % COLS;
        const bool inside = r < rows && c < cols;
        D* d = dst + (TRANSPOSED ? c * dst_ld + r : r * dst_ld + c);
        if constexpr (ASYNC) {
            static_assert(std::is_same_v<D, S>, "cp.async copies bytes, it cannot convert");
            cp_async_or_zero<sizeof(D)>(d, src + (inside ? r * ld + c : 0), inside);
        } else {
            *d = static_cast<D>(inside ? src[r * ld + c] : S{0});
        }
    }
}

/// Copy a ROWS x ROW_BYTES tile from global memory (rows src_ld bytes apart)
/// to shared memory (rows dst_ld bytes apart) in 16-byte cp.async chunks, by
/// THREADS threads. No bounds checks: callers pass whole, 16-byte-aligned tiles.
template <int ROWS, int ROW_BYTES, int THREADS>
__device__ __forceinline__ void copy_tile_16B(void* dst, int dst_ld, const void* src, std::size_t src_ld) {
    constexpr int kChunks = ROW_BYTES / 16;
    auto* d       = static_cast<unsigned char*>(dst);
    const auto* s = static_cast<const unsigned char*>(src);
    #pragma unroll
    for (int i = threadIdx.x; i < ROWS * kChunks; i += THREADS) {
        const int r = i / kChunks, c = i % kChunks * 16;
        cp_async_16(d + r * dst_ld + c, s + r * src_ld + c);
    }
}

// ############################################################################
// Part 2. Kernels
// ############################################################################

// ============================================================================
// Level 0: Naive
//
// One thread computes one C(i,j), reading A and B straight from global
// memory. Adjacent threads of a warp read adjacent B[k][j], so the B loads
// coalesce, and L1/L2 absorb much of the reuse — which makes this baseline
// harder to beat than it looks.
// ============================================================================
constexpr int kTile = 16;  // Levels 0 and 1: 16 x 16 threads per block

/// C(i, j) = sum_k A(i, k) B(k, j) for the one (i, j) this thread owns.
template <typename T>
__global__ void kernel_naive(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                             int M, int K, int N) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= N)
        return;
    T acc = T{0};
    for (int k = 0; k < K; ++k)
        acc += A[i * K + k] * B[k * N + j];
    C[i * N + j] = acc;
}

// ============================================================================
// Level 1: Shared-memory tiling, 16 x 16
//
// A 16x16 thread block stages each 16x16 k-tile of A and B in shared memory,
// so each global element is loaded once per block instead of once per
// thread: 16x fewer global loads. A warp covers two 16-thread rows:
// As[ty][p] broadcasts one value per half-warp, and Bs[p][tx] reads the same
// contiguous segment twice. The +1 padding separates the double-precision
// A broadcasts, which otherwise map to the same banks. Each thread still owns
// one output, so the two __syncthreads() per k-tile pay for only 16 FMAs.
// ============================================================================

/// Level 0's one-output-per-thread GEMM, with A and B staged through shared memory.
template <typename T>
__global__ void kernel_blocked(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                               int M, int K, int N) {
    __shared__ T As[kTile][kTile + 1];
    __shared__ T Bs[kTile][kTile + 1];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i = blockIdx.y * kTile + ty;
    const int j = blockIdx.x * kTile + tx;
    T acc = T{0};
    for (int k0 = 0; k0 < K; k0 += kTile) {
        As[ty][tx] = (i < M && k0 + tx < K) ? A[i * K + k0 + tx] : T{0};
        Bs[ty][tx] = (k0 + ty < K && j < N) ? B[(k0 + ty) * N + j] : T{0};
        __syncthreads();
        #pragma unroll
        for (int p = 0; p < kTile; ++p)
            acc += As[ty][p] * Bs[p][tx];
        __syncthreads();
    }
    if (i < M && j < N)
        C[i * N + j] = acc;
}

// ============================================================================
// Shared by Levels 2-4: the 8x8 register tile
//
// Each thread owns a kTM x kTN tile of C in registers. Per k step it loads
// kTM values of A and kTN values of B from shared memory, then accumulates
// their outer product: 64 FMAs from 16 shared-memory reads.
// ============================================================================
constexpr int kBM = 128;  // block tile rows of C
constexpr int kBN = 128;  // block tile columns of C
constexpr int kBK = 16;   // k-step per shared-memory tile
constexpr int kTM = 8;    // rows of C per thread
constexpr int kTN = 8;    // columns of C per thread
constexpr int kRegTileThreads = (kBM / kTM) * (kBN / kTN);  // 256

/// c += a * b^T: the outer product of a thread's A column and B row.
template <typename T>
__device__ __forceinline__ void fma_outer(T (&c)[kTM][kTN], const T (&a)[kTM], const T (&b)[kTN]) {
    #pragma unroll
    for (int m = 0; m < kTM; ++m)
        #pragma unroll
        for (int n = 0; n < kTN; ++n)
            c[m][n] += a[m] * b[n];
}

/// Accumulate one shared-memory k-tile into the register tile at (tm, tn)
/// of the block tile. As is transposed (As[k][m]), Bs natural (Bs[k][n]).
template <typename T, int LDA, int LDB>
__device__ __forceinline__ void tile_fma(T (&acc)[kTM][kTN], const T (&As)[kBK][LDA], const T (&Bs)[kBK][LDB],
                                         int tm, int tn) {
    #pragma unroll
    for (int k = 0; k < kBK; ++k) {
        T a[kTM], b[kTN];
        #pragma unroll
        for (int m = 0; m < kTM; ++m)
            a[m] = As[k][tm + m];
        #pragma unroll
        for (int n = 0; n < kTN; ++n)
            b[n] = Bs[k][tn + n];
        fma_outer(acc, a, b);
    }
}

/// Write a thread's register tile to C at (tm, tn) of the block tile,
/// clipped to the `rows` x `cols` part inside the matrix.
template <typename T>
__device__ __forceinline__ void store_tile(T* C, int ldc, const T (&acc)[kTM][kTN], int tm, int tn, int rows,
                                           int cols) {
    #pragma unroll
    for (int m = 0; m < kTM; ++m)
        #pragma unroll
        for (int n = 0; n < kTN; ++n)
            if (tm + m < rows && tn + n < cols)
                C[(tm + m) * ldc + tn + n] = acc[m][n];
}

// ============================================================================
// Level 2: Register-tiled
//
// Thread block: kBM x kBN = 128 x 128 outputs, 256 threads, each owning an
// 8x8 register tile of C.
//
// Shared memory: As[kBK][kBM+1], A's tile transposed so a thread's 8 rows of
// A are contiguous; Bs[kBK][kBN+1], B's tile as is. The +1 padding keeps the
// transposed stores of A free of bank conflicts.
//
// Inner loop: outer product of an As column and a Bs row -> 8x8 FMAs per k
// step, all from registers. Arithmetic intensity per block is
// 2·128·128·K FLOP / ((128 + 128)·K·4 B) ≈ 32 FLOP/byte, against ~2 for
// Level 1, where each thread owns one output.
// ============================================================================

/// 128 x 128 block tile, 8 x 8 outputs per thread, single shared-memory buffer.
template <typename T>
__global__ void __launch_bounds__(kRegTileThreads)
kernel_reg_tile(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int K, int N) {
    __shared__ T As[kBK][kBM + 1];
    __shared__ T Bs[kBK][kBN + 1];

    const int2 size = block_tile<kBM, kBN>(A, B, C, M, K, N);
    const int rows = size.x, cols = size.y;
    const int tm = threadIdx.x / (kBN / kTN) * kTM;  // this thread's 8x8 tile
    const int tn = threadIdx.x % (kBN / kTN) * kTN;

    T acc[kTM][kTN] = {};
    for (int k0 = 0; k0 < K; k0 += kBK) {
        load_tile<kBM, kBK, true>(&As[0][0], kBM + 1, A + k0, K, rows, K - k0);
        load_tile<kBK, kBN, false>(&Bs[0][0], kBN + 1, B + k0 * N, N, K - k0, cols);
        __syncthreads();
        tile_fma(acc, As, Bs, tm, tn);
        __syncthreads();
    }
    store_tile(C, N, acc, tm, tn, rows, cols);
}

// ============================================================================
// Level 3: Double-buffered register tile
//
// Level 2 with two shared-memory buffers, run by cp_async_pipeline: tile
// k+1 is copied while tile k is computed. On sm_80+ the copies are cp.async,
// which moves global -> shared without blocking the threads, so the copy
// overlaps the FMAs. On older GPUs the copies are synchronous: the
// structure is the same, without the overlap.
//
// Static shared memory is limited to 48 KB, so double uses a 64x64 tile:
//   f32, 128x128: 2 * 16 * (128+1 + 128+1) * 4 = 33,024 B
//   f64, 128x128: 2 * 16 * (128+1 + 128+1) * 8 = 66,048 B  (too large)
//   f64,   64x64: 2 * 16 * ( 64+1 +  64+1) * 8 = 33,280 B
// ============================================================================
template <typename T>
constexpr int kDBufTile = sizeof(T) == 8 ? 64 : 128;  // block tile edge, BM = BN

/// Level 2 with a two-stage cp.async pipeline. For float, (256, 2): two
/// blocks per SM, so at most 128 registers per thread; one more (registers
/// are allocated in steps of 8) would leave room for only one block, ~4%
/// slower at N = 4096. Double's 64-thread blocks need no bound, and 128
/// registers would make it spill.
template <typename T>
__global__ void __launch_bounds__(kRegTileThreads, sizeof(T) == 4 ? 2 : 1)
kernel_double_buf(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int K, int N) {
    constexpr int BT = kDBufTile<T>;
    __shared__ T As[2][kBK][BT + 1];
    __shared__ T Bs[2][kBK][BT + 1];

    const int2 size = block_tile<BT, BT>(A, B, C, M, K, N);
    const int rows = size.x, cols = size.y;
    const int tm = threadIdx.x / (BT / kTN) * kTM;
    const int tn = threadIdx.x % (BT / kTN) * kTN;

    T acc[kTM][kTN] = {};
    cp_async_pipeline<2>(
        (K + kBK - 1) / kBK,
        [&](int kt, int s) {
            const int k0 = kt * kBK;
            load_tile<BT, kBK, true, true>(&As[s][0][0], BT + 1, A + k0, K, rows, K - k0);
            load_tile<kBK, BT, false, true>(&Bs[s][0][0], BT + 1, B + k0 * N, N, K - k0, cols);
        },
        [&](int s) { tile_fma(acc, As[s], Bs[s], tm, tn); });
    store_tile(C, N, acc, tm, tn, rows, cols);
}

// ============================================================================
// Level 4: Vectorized loads (float4/double2) + shared-memory XOR swizzle
//
// Same register-tile shape as kernel_reg_tile (128x128 block, BK=16, 8x8 per
// thread), with two changes:
//
//  1. Global -> shared loads use 128-bit vectors (float4 / double2), cutting
//     load instructions by 4x / 2x. B's contiguous dimension (N) is also Bs's,
//     so B is loaded and stored as vectors. A is contiguous along K but As is
//     stored transposed (As[k][m]), so A is loaded as a vector and then
//     scattered with scalar stores into W rows of As.
//
//     Vector loads need 16-byte-aligned addresses. cudaMalloc aligns the base
//     pointers, and every offset here is a multiple of the vector width as
//     long as K and N are too; the launcher checks that and otherwise runs
//     kernel_reg_tile.
//
//  2. Shared memory uses an XOR swizzle instead of +1 padding to spread
//     accesses across banks without a wasted column (padding would also
//     break the alignment the vector stores need). swizzle() permutes which
//     vector slot of a row an element lands in; the same function is used at
//     every write and read, so correctness does not depend on the
//     permutation itself.
// ============================================================================

/// 128-bit vector type: float4 for float, double2 for double.
template <typename T> struct VecTraits;
template <> struct VecTraits<float>  { using Vec = float4;  static constexpr int kWidth = 4; };
template <> struct VecTraits<double> { using Vec = double2; static constexpr int kWidth = 2; };

/// Offset of logical column `col` within row `row` of a swizzled tile: the
/// W-element vector slot is XORed with the row (SLOTS, the slots per row, is
/// a power of two), the position inside the slot is kept. A column that
/// starts a slot maps to the start of a slot, so vector stores stay aligned.
template <int W, int SLOTS>
__device__ __forceinline__ int swizzle(int row, int col) {
    return ((col / W) ^ (row & (SLOTS - 1))) * W + col % W;
}

/// Read one aligned vector from shared memory into W consecutive registers.
template <typename Vec, typename T>
__device__ __forceinline__ void load_vec(T* dst, const T* src) {
    const Vec v = *reinterpret_cast<const Vec*>(src);
    const T* e  = reinterpret_cast<const T*>(&v);
    #pragma unroll
    for (int j = 0; j < int(sizeof(Vec) / sizeof(T)); ++j)
        dst[j] = e[j];
}

/// Level 2 with 128-bit global and shared-memory accesses. K, N % W == 0.
template <typename T>
__global__ void __launch_bounds__(kRegTileThreads)
kernel_vectorized(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int K, int N) {
    using Vec          = typename VecTraits<T>::Vec;
    constexpr int W    = VecTraits<T>::kWidth;
    constexpr int kSlots = kBM / W;  // vector slots per row; kBM == kBN

    __shared__ alignas(16) T As[kBK][kBM];  // As[k][swizzle(k, m)] = A(m, k)
    __shared__ alignas(16) T Bs[kBK][kBN];  // Bs[k][swizzle(k, n)] = B(k, n)

    const int2 size = block_tile<kBM, kBN>(A, B, C, M, K, N);
    const int rows = size.x, cols = size.y;
    const int tm = threadIdx.x / (kBN / kTN) * kTM;
    const int tn = threadIdx.x % (kBN / kTN) * kTN;

    T acc[kTM][kTN] = {};
    for (int k0 = 0; k0 < K; k0 += kBK) {
        // A: each vector is W consecutive k of row m, scattered into W rows
        // of As. K % W == 0, so a vector that starts inside K ends inside it.
        for (int v = threadIdx.x; v < kBM * kBK / W; v += blockDim.x) {
            const int m = v / (kBK / W), k = v % (kBK / W) * W;
            Vec x{};
            if (m < rows && k0 + k < K)
                x = *reinterpret_cast<const Vec*>(A + m * K + k0 + k);
            const T* e = reinterpret_cast<const T*>(&x);
            #pragma unroll
            for (int j = 0; j < W; ++j)
                As[k + j][swizzle<W, kSlots>(k + j, m)] = e[j];
        }
        // B: each vector is W consecutive n of row k, stored as a vector.
        // N % W == 0 likewise.
        for (int v = threadIdx.x; v < kBK * kBN / W; v += blockDim.x) {
            const int k = v / (kBN / W), n = v % (kBN / W) * W;
            Vec x{};
            if (k0 + k < K && n < cols)
                x = *reinterpret_cast<const Vec*>(B + (k0 + k) * N + n);
            *reinterpret_cast<Vec*>(&Bs[k][swizzle<W, kSlots>(k, n)]) = x;
        }
        __syncthreads();

        // A thread's kTM values of A and kTN of B are whole vector slots
        // (tm, tn are multiples of W), so they are read as vectors too.
        #pragma unroll
        for (int k = 0; k < kBK; ++k) {
            T a[kTM], b[kTN];
            #pragma unroll
            for (int v = 0; v < kTM / W; ++v)
                load_vec<Vec>(&a[v * W], &As[k][swizzle<W, kSlots>(k, tm + v * W)]);
            #pragma unroll
            for (int v = 0; v < kTN / W; ++v)
                load_vec<Vec>(&b[v * W], &Bs[k][swizzle<W, kSlots>(k, tn + v * W)]);
            fma_outer(acc, a, b);
        }
        __syncthreads();
    }
    store_tile(C, N, acc, tm, tn, rows, cols);
}

// ============================================================================
// Tensor Core formats: element width, accumulator and the one mma.sync
// instruction that differs between them. Level 6 uses Fp16; the
// low-precision kernels (kernel_lowp, Level 8's kernel_tma) every format.
//
// Each mma covers 16 (M) x 8 (N) x 32 bytes of K: m16n8k16 for 16-bit
// inputs, m16n8k32 for 8-bit ones, with the same register layout. a[4] and
// b[2] are the fragments ldmatrix produced; c is this lane's share of the
// 16x8 accumulator tile: 4 fp32/int32 values, or 4 fp16 packed in 2 registers.
// Out is C's element type.
// ============================================================================
struct Bf16 {
    static constexpr int kBytes = 2, kAccRegs = 4;
    using Acc = float;
    using Out = float;
    /// c += a * b, bf16 inputs, fp32 accumulate.
    __device__ static void mma(Acc* c, const unsigned* a, const unsigned* b) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct Fp16 {
    static constexpr int kBytes = 2, kAccRegs = 4;
    using Acc = float;
    using Out = float;
    /// c += a * b, fp16 inputs, fp32 accumulate.
    __device__ static void mma(Acc* c, const unsigned* a, const unsigned* b) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct Fp16Acc16 {
    static constexpr int kBytes = 2, kAccRegs = 2;
    using Acc = unsigned;  // two packed fp16 per register
    using Out = __half;
    /// c += a * b, fp16 inputs, fp16 accumulate.
    __device__ static void mma(Acc* c, const unsigned* a, const unsigned* b) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                     "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                     : "+r"(c[0]), "+r"(c[1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct Int8 {
    static constexpr int kBytes = 1, kAccRegs = 4;
    using Acc = int;
    using Out = int;
    /// c += a * b, int8 inputs, int32 accumulate.
    __device__ static void mma(Acc* c, const unsigned* a, const unsigned* b) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct Fp8 {
    static constexpr int kBytes = 1, kAccRegs = 4;
    using Acc = float;
    using Out = float;
    /// c += a * b, e4m3 inputs, fp32 accumulate: the full-rate block-scaled
    /// form with unit scales on sm_120a, the sm_89 form elsewhere.
    __device__ static void mma(Acc* c, const unsigned* a, const unsigned* b) {
#if defined(HPC_FP8_BLOCK_SCALED)
        const unsigned scale          = 0x7f7f7f7fu;  // UE8M0 127 = 2^0, in every byte
        const unsigned short selector = 0;            // which byte of `scale` (one scale per operand row)
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
// Level 5: Tensor Cores via WMMA -- sm_70+
//
// Each warp computes a 16x16 output tile of C with wmma::mma_sync.
// Thread block: 4 x 4 warps = 512 threads, a 64x64 tile of C.
//
// A and B are fp32; each tile is converted to fp16 on its way into shared
// memory, multiplied with wmma::mma_sync (fp16 x fp16 -> fp32) and
// accumulated in fp32 fragments. The conversion costs ~1e-3 relative error,
// hence the relaxed test tolerance.
// ============================================================================
constexpr int kWmma    = 16;            // the WMMA fragment is 16 x 16 x 16, fixed by the API
constexpr int kWarpsM  = 4;             // Levels 5 and 6: 4 x 4 warps ...
constexpr int kWarpsN  = 4;
constexpr int kBlockMN = kWarpsM * 16;  // ... of 16 x 16 outputs: a 64 x 64 block tile
constexpr int kWarpGridThreads = kWarpsM * kWarpsN * 32;  // 512

/// Level 5's block: the 64 x 64 tile of C, one 16 x 16 wmma fragment per warp.
/// Also Level 6's body on GPUs compiled for below sm_80 (see kernel_mma_ldmatrix).
__device__ __forceinline__ void wmma_tile(const float* __restrict__ A, const float* __restrict__ B,
                                          float* __restrict__ C, int M, int K, int N) {
    // No +1 padding: wmma::load_matrix_sync requires the leading dimension of
    // a __half fragment to be a multiple of 8 elements (16 bytes). 64 is; 65
    // would fail with cudaErrorMisalignedAddress.
    __shared__ alignas(32) __half As[kWmma][kBlockMN];  // transposed: As[k][m] = A(m, k)
    __shared__ alignas(32) __half Bs[kWmma][kBlockMN];  // natural:    Bs[k][n] = B(k, n)

    const int2 size = block_tile<kBlockMN, kBlockMN>(A, B, C, M, K, N);
    const int rows = size.x, cols = size.y;
    const int warp = threadIdx.x / 32;
    const int wm = warp / kWarpsN * kWmma;  // this warp's 16x16 tile
    const int wn = warp % kWarpsN * kWmma;
    const bool active = wm < rows && wn < cols;

    // The major-order tags must match the shared-memory layout; a mismatch
    // silently transposes the operand (wrong values, no error).
    //
    // As[k][m] = A[m][k]. With col_major, element(row,col) = ptr + row +
    // col*ldm = As[k=col][m=row] = A[row][col], the (row=M, col=K) view
    // mma_sync expects of matrix_a.
    //
    // Bs[k][n] = B[k][n]. With row_major, element(row,col) = ptr + row*ldm +
    // col = B[row][col], the (row=K, col=N) view mma_sync expects of matrix_b.
    wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, __half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += kWmma) {
        load_tile<kBlockMN, kWmma, true>(&As[0][0], kBlockMN, A + k0, K, rows, K - k0);
        load_tile<kWmma, kBlockMN, false>(&Bs[0][0], kBlockMN, B + k0 * N, N, K - k0, cols);
        __syncthreads();
        if (active) {
            wmma::load_matrix_sync(a_frag, &As[0][wm], kBlockMN);
            wmma::load_matrix_sync(b_frag, &Bs[0][wn], kBlockMN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    // store_matrix_sync always writes a full 16x16 tile and needs a
    // 32-byte-aligned destination, so a direct store is only safe for a tile
    // wholly inside C with N % 8 == 0. Any other tile goes through a per-warp
    // shared-memory staging tile and is copied out element by element.
    __shared__ alignas(32) float Cs[kWarpsM * kWarpsN][kWmma * kWmma];
    if (!active)
        return;
    if (wm + kWmma <= rows && wn + kWmma <= cols && N % 8 == 0) {
        wmma::store_matrix_sync(C + wm * N + wn, c_frag, N, wmma::mem_row_major);
    } else {
        float* tile = Cs[warp];
        wmma::store_matrix_sync(tile, c_frag, kWmma, wmma::mem_row_major);
        __syncwarp();
        for (int e = threadIdx.x % 32; e < kWmma * kWmma; e += 32) {
            const int r = wm + e / kWmma, c = wn + e % kWmma;
            if (r < rows && c < cols)
                C[r * N + c] = tile[e];
        }
    }
}

/// Level 5. (512, 3): three 512-thread blocks fill an SM. Without the
/// minimum the compiler may use up to 64 registers per thread, which fits
/// only two blocks; the kernel needs 40.
__global__ void __launch_bounds__(kWarpGridThreads, 3)
kernel_wmma(const float* A, const float* B, float* C, int M, int K, int N) {
    wmma_tile(A, B, C, M, K, N);
}

// ============================================================================
// Level 6: Raw Tensor Core MMA via mma.sync + ldmatrix -- sm_80+
//
// The same computation as kernel_wmma, one level lower. WMMA's
// load_matrix_sync / mma_sync leave the per-lane register mapping to the
// compiler; here ldmatrix loads shared memory straight into the registers
// mma.sync.m16n8k16 (Fp16::mma) expects, and the kernel chooses which
// address each lane supplies and whether the load is transposed. A mistake
// in that mapping produces wrong values, not a build or launch error. The
// native tile is 16x8x16, so each warp issues two MMAs side by side to
// cover the 16x16 area kernel_wmma covers with one.
//
// Operand layouts (the only f16 m16n8k16 variant is .row.col):
//   A `.row` (K contiguous): As is stored [m][k], A's own layout, so no
//     transpose — unlike kernel_wmma and the FMA kernels, which store As as
//     [k][m].
//   B `.col` (K contiguous): Bs is stored [k][n] (N contiguous), so B's
//     ldmatrix uses `.trans`. `.trans` only changes the register shuffle,
//     not which address each lane supplies.
//
// mma.sync.m16n8k16.f32 accumulator layout (PTX ISA, "Matrix Fragments for
// mma.m16n8k16"), groupID = lane / 4, threadInGroup = lane % 4:
//   acc[0] -> C[groupID,     threadInGroup*2]
//   acc[1] -> C[groupID,     threadInGroup*2 + 1]
//   acc[2] -> C[groupID + 8, threadInGroup*2]
//   acc[3] -> C[groupID + 8, threadInGroup*2 + 1]
//
// sm_70-75 have no f16 m16n8k16 shape; the launcher runs kernel_wmma there.
// A binary compiled only for sm_70 PTX may still run on a newer GPU, so the
// pre-sm_80 compilation of this kernel is wmma_tile rather than a no-op.
// ============================================================================

/// Level 5's tiling (4 x 4 warps, 16 x 16 per warp) with ldmatrix + mma.sync PTX.
__global__ void __launch_bounds__(kWarpGridThreads)
kernel_mma_ldmatrix(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                    int M, int K, int N) {
#ifdef HPC_SM80
    __shared__ __half As[kBlockMN][16];  // natural: As[m][k], matches `.row`
    __shared__ __half Bs[16][kBlockMN];  // natural: Bs[k][n], needs `.trans`

    const int2 size = block_tile<kBlockMN, kBlockMN>(A, B, C, M, K, N);
    const int rows = size.x, cols = size.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / kWarpsN * 16;  // this warp's 16x16 tile
    const int wn = warp % kWarpsN * 16;

    float acc[2][4] = {};  // two 16x8 tiles, 4 fp32 registers per lane each
    for (int k0 = 0; k0 < K; k0 += 16) {
        load_tile<kBlockMN, 16, false>(&As[0][0], 16, A + k0, K, rows, K - k0);
        load_tile<16, kBlockMN, false>(&Bs[0][0], kBlockMN, B + k0 * N, N, K - k0, cols);
        __syncthreads();
        if (wm < rows) {
            // A: the 16(M) x 16(K) tile as four 8x8 quadrants. mma.sync
            // expects them as (M0-7,K0-7), (M8-15,K0-7), (M0-7,K8-15),
            // (M8-15,K8-15): lane group q = lane/8 supplies rows of quadrant
            // q, M half q%2, K half q/2.
            const int q = lane / 8;
            unsigned a[4];
            ldmatrix_x4(a, &As[wm + lane % 8 + q % 2 * 8][q / 2 * 8]);
            // B: two 16(K) x 8(N) tiles. Lanes 0-15 address rows K = 0..15;
            // lanes 16-31 repeat them.
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                unsigned b[2];
                ldmatrix_x2_trans(b, &Bs[lane % 16][wn + t * 8]);
                Fp16::mma(acc[t], a, b);
            }
        }
        __syncthreads();
    }

    // Accumulator element i of tile t is C(wm + lane/4 + i/2*8, wn + t*8 + lane%4*2 + i%2).
    #pragma unroll
    for (int t = 0; t < 2; ++t)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = wm + lane / 4 + i / 2 * 8;
            const int c = wn + t * 8 + lane % 4 * 2 + i % 2;
            if (r < rows && c < cols)
                C[r * N + c] = acc[t][i];
        }
#else
    wmma_tile(A, B, C, M, K, N);
#endif
}

// ============================================================================
// Level 7: Pipelined Tensor Cores via WMMA -- bigger tiles + cp.async
// double buffering.
//
// kernel_wmma (64x64 block tile, single-buffered, one mma_sync per warp per
// k-step) runs an order of magnitude below cuBLAS FP16 on the same Tensor
// Cores. This kernel closes most of that gap the way CUTLASS-style kernels
// do, still on the wmma:: C++ API:
//
//   1. Bigger block tile: 128x128 with BK=32 (vs 64x64, BK=16), so more work
//      per shared-memory round trip and per __syncthreads().
//   2. Bigger warp tile: each of 8 warps owns 32x64 = 2x4 fragments and issues
//      8 mma_sync per k-sub-step. Each b_frag is loaded once and reused for
//      both fm, each a_frag once for all 4 fn — the register blocking of
//      kernel_reg_tile, at fragment granularity.
//   3. Double buffering with cp_async_pipeline, as in kernel_double_buf.
//   4. Padded shared-memory leading dimensions (kPipeAsLd / kPipeBsLd below).
//
// Constraints that follow from cp.async:
//
//   - It copies bytes without converting, so A and B are converted to fp16
//     in global memory first (kernel_f32_to_f16, as gemm_cuda_cublas_fp16
//     does).
//   - It copies contiguous runs only, so As keeps A16's row-major layout
//     (As[m][k], K contiguous) instead of kernel_wmma's transposed As[k][m],
//     and a_frag is therefore row_major — the tag follows the layout.
//     Bs[k][n] and b_frag (row_major) are as in kernel_wmma.
//   - Every transfer is a full 16-byte chunk (8 halves). M and N must be
//     multiples of 128 and K of 32 (the launcher falls back to kernel_wmma
//     otherwise): then every chunk starts at an element offset that is a
//     multiple of 8, i.e. 16-byte aligned, and no bounds checks are needed.
// ============================================================================
constexpr int kPipeBM = 128;                     // thread-block output rows
constexpr int kPipeBN = 128;                     // thread-block output cols
constexpr int kPipeBK = 32;                      // k per stage: 2 WMMA k-steps
constexpr int kPipeWarpM = 32;                   // per-warp output rows: 2 fragments
constexpr int kPipeWarpN = 64;                   // per-warp output cols: 4 fragments
constexpr int kPipeFragM = kPipeWarpM / kWmma;   // 2
constexpr int kPipeFragN = kPipeWarpN / kWmma;   // 4
constexpr int kPipeWarpCols = kPipeBN / kPipeWarpN;                          // 2
constexpr int kPipeThreads = (kPipeBM / kPipeWarpM) * kPipeWarpCols * 32;    // 256

// Shared-memory leading dimensions, padded by 8 halves (16 bytes).
//
// A load_matrix_sync fragment reads 16 rows of 16 halves; a row starts in bank
// (row * ld * 2 / 4) % 32. Unpadded, ld = kPipeBK = 32 (64 B/row) gives only
// 2 distinct starting banks over 16 rows (8-way conflict), and
// ld = kPipeBN = 128 (256 B/row, two full bank cycles) puts every row in the
// same bank (16-way). Padding by 8 (ld = 40 / 136) gives 8 distinct starts:
// a 2-way conflict. Measured with Nsight Compute on RTX 5080: shared-load
// conflicts fall from 85% to 1% of wavefronts, 80.8 → 100.5 TFLOP/s.
//
// The pad is 8, not the usual 1: load_matrix_sync needs ld to be a multiple
// of 8 halves and cp.async needs a 16-byte-aligned destination. An XOR
// swizzle (no memory cost, as in kernel_vectorized) is not possible here —
// load_matrix_sync takes a plain (pointer, ld) pair.
//
// Cost: 32 → 37 KB of shared memory per block, lowering the shared-memory
// occupancy limit from 3 to 2 blocks/SM. The kernel's 118 registers/thread
// already cap it at 2, so occupancy is unchanged.
constexpr int kPipeAsLd = kPipeBK + 8;  // 40 halves =  80 B
constexpr int kPipeBsLd = kPipeBN + 8;  // 136 halves = 272 B

/// fp16 A (M x K) times fp16 B (K x N), 128 x 128 block tile, 2 x 4 wmma
/// fragments per warp, double-buffered with cp.async. Exact tile multiples.
__global__ void __launch_bounds__(kPipeThreads)
kernel_wmma_pipelined(const __half* __restrict__ A16, const __half* __restrict__ B16, float* __restrict__ C,
                      int M, int K, int N) {
    __shared__ alignas(32) __half As[2][kPipeBM][kPipeAsLd];  // natural: As[m][k], padded ld
    __shared__ alignas(32) __half Bs[2][kPipeBK][kPipeBsLd];  // natural: Bs[k][n], padded ld

    const int row0 = blockIdx.y * kPipeBM, col0 = blockIdx.x * kPipeBN;
    A16 += static_cast<std::size_t>(row0) * K;
    B16 += col0;
    C += static_cast<std::size_t>(row0) * N + col0;
    const int warp = threadIdx.x / 32;
    const int wm = warp / kPipeWarpCols * kPipeWarpM;  // this warp's 32x64 tile
    const int wn = warp % kPipeWarpCols * kPipeWarpN;

    wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float> c_frag[kPipeFragM][kPipeFragN];
    #pragma unroll
    for (int fm = 0; fm < kPipeFragM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < kPipeFragN; ++fn)
            wmma::fill_fragment(c_frag[fm][fn], 0.0f);

    auto load = [&](int kt, int s) {
        const int k0 = kt * kPipeBK;
        copy_tile_16B<kPipeBM, kPipeBK * 2, kPipeThreads>(&As[s][0][0], kPipeAsLd * 2, A16 + k0,
                                                          std::size_t(K) * 2);
        copy_tile_16B<kPipeBK, kPipeBN * 2, kPipeThreads>(&Bs[s][0][0], kPipeBsLd * 2,
                                                          B16 + static_cast<std::size_t>(k0) * N,
                                                          std::size_t(N) * 2);
    };
    auto compute = [&](int s) {
        #pragma unroll
        for (int ks = 0; ks < kPipeBK; ks += kWmma) {
            // Each fn's b_frag is loaded once per k-sub-step and reused for
            // every fm below: the register blocking of kernel_reg_tile.
            wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, __half, wmma::row_major> b_frag[kPipeFragN];
            #pragma unroll
            for (int fn = 0; fn < kPipeFragN; ++fn)
                wmma::load_matrix_sync(b_frag[fn], &Bs[s][ks][wn + fn * kWmma], kPipeBsLd);
            #pragma unroll
            for (int fm = 0; fm < kPipeFragM; ++fm) {
                wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, __half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, &As[s][wm + fm * kWmma][ks], kPipeAsLd);
                #pragma unroll
                for (int fn = 0; fn < kPipeFragN; ++fn)
                    wmma::mma_sync(c_frag[fm][fn], a_frag, b_frag[fn], c_frag[fm][fn]);
            }
        }
    };
    cp_async_pipeline<2>(K / kPipeBK, load, compute);

    #pragma unroll
    for (int fm = 0; fm < kPipeFragM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < kPipeFragN; ++fn)
            wmma::store_matrix_sync(C + (wm + fm * kWmma) * N + wn + fn * kWmma, c_frag[fm][fn], N,
                                    wmma::mem_row_major);
}

/// dst[i] = fp16(src[i]): converts a whole matrix in global memory ahead of
/// time, for kernel_wmma_pipelined (cp.async cannot convert) and the cuBLAS
/// fp16 reference.
__global__ void kernel_f32_to_f16(const float* __restrict__ src, __half* __restrict__ dst, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = __float2half(src[i]);
}

// ============================================================================
// Low-precision kernels: kernel_lowp and Level 8's kernel_tma
//
// Both run every format above on K-major operands: A is M x K and B is
// passed transposed, Bt = N x K, both row-major. Every operand tile is then
// "rows of K bytes", and one non-transposing ldmatrix.x4 fills the
// fragments of every format: a 16 x 32-byte fragment is 16 k-values of a
// 16-bit type or 32 of an 8-bit type, and mma.sync's register layout is the
// same either way. Shapes are exact tile multiples (the Matrix wrappers pad),
// so neither kernel has bounds checks.
//
// The two kernels share the math (warp_mma_k32), the epilogue (store_acc)
// and the block order (grouped_tile); they differ only in how tiles reach
// shared memory.
// ============================================================================
constexpr int kKBytes = 64;  // bytes of K per pipeline stage: 32 16-bit or 64 8-bit elements
constexpr int kGroupM = 8;   // tile rows per group of grouped_tile

/// Grouped block order: consecutive blocks walk down a column of kGroupM
/// tile rows before moving right, so the blocks resident at one time share
/// a few A strips and a few Bt strips, which then stay in L2. Row-major
/// order would stream all of Bt through L2 for every row of tiles.
/// Returns this block's (tile row, tile column).
__device__ __forceinline__ int2 grouped_tile(int tiles_m, int tiles_n) {
    const int per_group = kGroupM * tiles_n;
    const int first_m   = blockIdx.x / per_group * kGroupM;
    const int group_m   = min(tiles_m - first_m, kGroupM);
    const int in_group  = blockIdx.x % per_group;
    return {first_m + in_group % group_m, in_group / group_m};
}

/// kernel_lowp's shared-memory rows: 64 bytes padded to 80. The eight
/// 16-byte rows one ldmatrix phase reads start in banks 0, 20, 8, 28, 16, 4,
/// 24, 12, so they never conflict. Returns the offset of 16-byte chunk
/// `chunk` of row `row`.
struct PaddedRows {
    static constexpr int kRowBytes = kKBytes + 16;
    __device__ static int offset(int row, int chunk) { return row * kRowBytes + chunk * 16; }
};

/// kernel_tma's shared-memory rows: 64 bytes, unpadded, stored by TMA with
/// its 64-byte swizzle: chunk c of row r lands at chunk c ^ (r / 2 % 4). The
/// eight rows one ldmatrix phase reads then start in eight different bank
/// groups: conflict-free with no padding bytes.
struct Swizzle64 {
    __device__ static int offset(int row, int chunk) { return row * kKBytes + ((chunk ^ (row / 2 % 4)) * 16); }
};

/// One 32-byte k step of a warp's (16 FM) x (8 FN) tile: ldmatrix FM
/// fragments of A (rows wm..) and FN of Bt (rows wn..) from 16-byte chunks
/// `chunk` and `chunk` + 1 of a stage laid out as LAYOUT, then FM x FN mma.
template <typename F, typename LAYOUT, int FM, int FN>
__device__ __forceinline__ void warp_mma_k32(typename F::Acc (&acc)[FM][FN][F::kAccRegs], const unsigned char* As,
                                             const unsigned char* Bs, int wm, int wn, int chunk) {
    const int lane = threadIdx.x % 32, q = lane / 8;  // q: which 8x8 matrix this lane addresses
    // A, 16 rows x 32 bytes per fragment: the four 8x8 (16-bit) matrices in
    // mma order (rows 0-7, bytes 0-15), (8-15, 0-15), (0-7, 16-31), (8-15, 16-31).
    unsigned a[FM][4];
    #pragma unroll
    for (int i = 0; i < FM; ++i)
        ldmatrix_x4(a[i], As + LAYOUT::offset(wm + i * 16 + lane % 8 + q % 2 * 8, chunk + q / 2));
    // Bt, two n8 tiles per ldmatrix.x4: (n 0-7, bytes 0-15), (0-7, 16-31),
    // (8-15, 0-15), (8-15, 16-31) -> b[j] and b[j+1].
    unsigned b[FN][2];
    #pragma unroll
    for (int j = 0; j < FN; j += 2)
        ldmatrix_x4(&b[j][0], Bs + LAYOUT::offset(wn + j * 8 + lane % 8 + q / 2 * 8, chunk + q % 2));
    #pragma unroll
    for (int i = 0; i < FM; ++i)
        #pragma unroll
        for (int j = 0; j < FN; ++j)
            F::mma(acc[i][j], a[i], b[j]);
}

/// Write a warp's FM x FN grid of 16x8 accumulator tiles to C, which points
/// at the warp tile's top-left element (row stride ldc). Lane (g = lane/4,
/// t = lane%4) holds C(g, 2t..2t+1) and C(g+8, 2t..2t+1) of each tile: one
/// 8-byte (4-byte for fp16) store per pair.
template <typename F, int FM, int FN>
__device__ __forceinline__ void store_acc(typename F::Out* C, int ldc,
                                          const typename F::Acc (&acc)[FM][FN][F::kAccRegs]) {
    using Out  = typename F::Out;
    const int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    #pragma unroll
    for (int i = 0; i < FM; ++i)
        #pragma unroll
        for (int j = 0; j < FN; ++j)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                Out* dst = C + static_cast<std::size_t>(i * 16 + g + h * 8) * ldc + j * 8 + t * 2;
                if constexpr (F::kAccRegs == 2) {
                    *reinterpret_cast<unsigned*>(dst) = acc[i][j][h];
                } else {
                    using Pair = std::conditional_t<std::is_same_v<Out, float>, float2, int2>;
                    *reinterpret_cast<Pair*>(dst) = Pair{acc[i][j][2 * h], acc[i][j][2 * h + 1]};
                }
            }
}

// ============================================================================
// kernel_lowp: the raw mma.sync kernel of Level 6 grown to the tile sizes
// and pipeline of Level 7.
//
//   - 128 x 128 block tile, 8 warps of 64 x 32 each.
//   - A two-stage cp_async_pipeline of 16-byte chunks (no conversion). Two
//     stages (40 KiB of shared memory) let two blocks share an SM; three or
//     four stages (60 / 80 KiB) allow only one and measured 8–18% slower on
//     an RTX 5080 at every size and format
//     (docs/experiments/lowp/tuning_cp_async.csv).
//   - Grouped block order (grouped_tile), PaddedRows shared memory.
// ============================================================================
constexpr int kLowpBM = 128, kLowpBN = 128;
constexpr int kLowpWarpsN  = 4;                                   // 2 x 4 warps ...
constexpr int kLowpWarpM   = 64, kLowpWarpN = kLowpBN / kLowpWarpsN;  // ... of 64 x 32
constexpr int kLowpThreads = (kLowpBM / kLowpWarpM) * kLowpWarpsN * 32;  // 256
constexpr int kLowpStages  = 2;
constexpr int kLowpStageA  = kLowpBM * PaddedRows::kRowBytes;              // A rows, then Bt rows
constexpr int kLowpStageBytes = (kLowpBM + kLowpBN) * PaddedRows::kRowBytes;  // 20,480
constexpr int kLowpSmemBytes  = kLowpStages * kLowpStageBytes;                // 40,960

/// C = A * Bt^T for one 128 x 128 tile. A: M x K, Bt: N x K, C: M x N.
template <typename F>
__global__ void __launch_bounds__(kLowpThreads)
kernel_lowp(const unsigned char* __restrict__ A, const unsigned char* __restrict__ Bt,
            typename F::Out* __restrict__ C, int M, int K, int N) {
#ifdef HPC_SM80
    extern __shared__ __align__(16) unsigned char smem[];
    const std::size_t ld = static_cast<std::size_t>(K) * F::kBytes;  // bytes per row of A and Bt

    const int2 tile = grouped_tile(M / kLowpBM, N / kLowpBN);
    A += static_cast<std::size_t>(tile.x) * kLowpBM * ld;
    Bt += static_cast<std::size_t>(tile.y) * kLowpBN * ld;
    const int warp = threadIdx.x / 32;
    const int wm = warp / kLowpWarpsN * kLowpWarpM, wn = warp % kLowpWarpsN * kLowpWarpN;

    typename F::Acc acc[kLowpWarpM / 16][kLowpWarpN / 8][F::kAccRegs] = {};
    auto stage = [&](int s) { return smem + s * kLowpStageBytes; };
    cp_async_pipeline<kLowpStages>(
        static_cast<int>(ld / kKBytes),
        [&](int kt, int s) {
            const std::size_t k = static_cast<std::size_t>(kt) * kKBytes;
            copy_tile_16B<kLowpBM, kKBytes, kLowpThreads>(stage(s), PaddedRows::kRowBytes, A + k, ld);
            copy_tile_16B<kLowpBN, kKBytes, kLowpThreads>(stage(s) + kLowpStageA, PaddedRows::kRowBytes,
                                                          Bt + k, ld);
        },
        [&](int s) {
            #pragma unroll
            for (int chunk = 0; chunk < kKBytes / 16; chunk += 2)
                warp_mma_k32<F, PaddedRows>(acc, stage(s), stage(s) + kLowpStageA, wm, wn, chunk);
        });

    C += (static_cast<std::size_t>(tile.x) * kLowpBM + wm) * N + static_cast<std::size_t>(tile.y) * kLowpBN + wn;
    store_acc<F>(C, N, acc);
#else
    __trap();
#endif
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
//      or 256-row (Bt) x 64-byte tile with one instruction. The copy engine
//      walks the tensor described by a CUtensorMap built on the host.
//   2. Hardware swizzle instead of padding (Swizzle64).
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
// stage count changes little once there are 3 (tuning_level8.csv).
// ============================================================================
constexpr int kTmaBM = 128;

/// Level 8's block tile, 128 x BN with BN = 256 (8 consumer warps, 4 stages
/// of 24 KiB) or 128 (4 consumer warps, 6 stages of 16 KiB); both fill 96 KiB.
template <int BN>
struct TmaTile {
    static constexpr int kStages     = BN == 256 ? 4 : 6;
    static constexpr int kWarpsN     = BN / 64;
    static constexpr int kConsumers  = (kTmaBM / 64) * kWarpsN;
    static constexpr int kThreads    = (kConsumers + 1) * 32;  // + 1 producer warp
    static constexpr int kStageA     = kTmaBM * kKBytes;
    static constexpr int kStageBytes = kStageA + BN * kKBytes;
    // + 1024 to align the stages (see kernel_tma), + the two barriers per stage
    static constexpr int kSmemBytes  = kStages * kStageBytes + 1024 + 2 * kStages * 8;
    static_assert(kSmemBytes <= 101376, "sm_120 allows 99 KiB of shared memory per block");
};

/// C = A * Bt^T for one 128 x BN tile, A and Bt read through TMA tensor maps.
template <typename F, int BN>
__global__ void __launch_bounds__(TmaTile<BN>::kThreads, 1)
kernel_tma(const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b,
           typename F::Out* __restrict__ C, int M, int K, int N) {
#ifdef HPC_SM90
    using T = TmaTile<BN>;
    extern __shared__ unsigned char smem_raw[];
    // The swizzle pattern is a function of the shared-memory address: stages
    // must start on a 1024-byte boundary (512 suffices for 64-byte swizzle).
    unsigned char* smem = smem_raw + ((1024 - (smem_addr(smem_raw) & 1023)) & 1023);
    auto stage_a = [&](int s) { return smem + s * T::kStageBytes; };
    auto stage_b = [&](int s) { return smem + s * T::kStageBytes + T::kStageA; };
    std::uint64_t* full  = reinterpret_cast<std::uint64_t*>(smem + T::kStages * T::kStageBytes);
    std::uint64_t* empty = full + T::kStages;

    const int2 tile   = grouped_tile(M / kTmaBM, N / BN);
    const int k_tiles = K * F::kBytes / kKBytes;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

    if (threadIdx.x == 0) {
        for (int s = 0; s < T::kStages; ++s) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], T::kConsumers);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (warp == T::kConsumers) {  // producer
        if (lane == 0) {
            for (int kt = 0; kt < k_tiles; ++kt) {
                const int s = kt % T::kStages;
                if (kt >= T::kStages)  // wait until every consumer has read the previous use of s
                    mbar_wait(&empty[s], (kt / T::kStages - 1) & 1);
                mbar_expect_tx(&full[s], T::kStageBytes);
                tma_load(stage_a(s), &map_a, &full[s], kt * kKBytes, tile.x * kTmaBM);
                tma_load(stage_b(s), &map_b, &full[s], kt * kKBytes, tile.y * BN);
            }
        }
        return;
    }

    // Consumers: warp tile 64 x 64 = 4 m16 x 8 n8 mma tiles.
    const int wm = warp / T::kWarpsN * 64, wn = warp % T::kWarpsN * 64;
    typename F::Acc acc[4][8][F::kAccRegs] = {};
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int s = kt % T::kStages;
        mbar_wait(&full[s], (kt / T::kStages) & 1);
        #pragma unroll
        for (int chunk = 0; chunk < kKBytes / 16; chunk += 2)
            warp_mma_k32<F, Swizzle64>(acc, stage_a(s), stage_b(s), wm, wn, chunk);
        __syncwarp();
        if (lane == 0)
            mbar_arrive(&empty[s]);  // after the mma that consumed every fragment of stage s
    }

    C += (static_cast<std::size_t>(tile.x) * kTmaBM + wm) * N + static_cast<std::size_t>(tile.y) * BN + wn;
    store_acc<F>(C, N, acc);
#else
    __trap();
#endif
}

/// Writes 1 when this build's FP8 mma is the block-scaled form, else 0.
__global__ void kernel_fp8_probe(int* out) {
#if defined(HPC_FP8_BLOCK_SCALED)
    *out = 1;
#else
    *out = 0;
#endif
}

// ############################################################################
// Part 3. Host launchers
// ############################################################################

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

/// Number of SMs of the selected device.
int sm_count() {
    int device = 0, count = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
    return count;
}

/// An owning device allocation of n elements of T.
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(&ptr, (n ? n : 1) * sizeof(T))); }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

/// Launch KERNEL on the default stream with `smem` bytes of dynamic shared
/// memory, raising the kernel's limit once when it needs more than the
/// default 48 KiB, and throw if the launch fails.
template <auto KERNEL, typename... Args>
void launch(dim3 grid, dim3 block, int smem, Args... args) {
    if (smem > 48 * 1024) {
        static const cudaError_t configured =
            cudaFuncSetAttribute(KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        CUDA_CHECK(configured);
    }
    KERNEL<<<grid, block, smem>>>(args...);
    CUDA_CHECK(cudaGetLastError());
}

/// The 2-D grid of BM x BN block tiles that covers an M x N output.
dim3 tile_grid(int M, int N, int BM, int BN) {
    return dim3(static_cast<unsigned>((N + BN - 1) / BN), static_cast<unsigned>((M + BM - 1) / BM));
}

/// Argument checks shared by every device entry point: throw on a negative
/// dimension; return true when there is nothing to multiply — M or N is 0,
/// or K is 0, after clearing C (elem_bytes per element) on the default stream.
bool empty_product(void* C, std::size_t elem_bytes, int M, int K, int N) {
    if (M < 0 || K < 0 || N < 0)
        throw std::invalid_argument("CUDA GEMM dimensions must be nonnegative");
    if (M == 0 || N == 0)
        return true;
    if (K != 0)
        return false;
    CUDA_CHECK(cudaMemsetAsync(C, 0, static_cast<std::size_t>(M) * N * elem_bytes));
    return true;
}

/// Bytes per input element and per output element of a low-precision format.
int input_bytes(CudaLowp f) { return f == CudaLowp::int8 || f == CudaLowp::fp8_e4m3 ? 1 : 2; }
int output_bytes(CudaLowp f) { return f == CudaLowp::fp16_acc16 ? 2 : 4; }

/// Call fn(format struct) for a low-precision format: the switch every
/// low-precision entry point dispatches through.
template <typename Fn>
void with_format(CudaLowp f, Fn fn) {
    switch (f) {
        case CudaLowp::bf16:       return fn(Bf16{});
        case CudaLowp::fp16:       return fn(Fp16{});
        case CudaLowp::fp16_acc16: return fn(Fp16Acc16{});
        case CudaLowp::int8:       return fn(Int8{});
        case CudaLowp::fp8_e4m3:   return fn(Fp8{});
    }
}

/// empty_product for a low-precision GEMM, which also throws when the
/// device lacks the format.
bool lowp_empty_product(CudaLowp f, void* C, int M, int K, int N) {
    if (M < 0 || K < 0 || N < 0)
        throw std::invalid_argument("CUDA GEMM dimensions must be nonnegative");
    if (!hpc::gemm::cuda_lowp_supported(f))
        throw std::runtime_error("low-precision CUDA GEMM: format not supported on this device");
    return empty_product(C, output_bytes(f), M, K, N);
}

/// Throw unless M and N are multiples of the block tile and K of the
/// format's k multiple: the low-precision kernels have no bounds checks.
void require_tiles(CudaLowp f, int M, int K, int N, int BM, int BN) {
    const int k_multiple = hpc::gemm::cuda_lowp_k_multiple(f);
    if (M % BM || N % BN || K % k_multiple)
        throw std::invalid_argument("low-precision CUDA GEMM: M, N must be multiples of " + std::to_string(BM) +
                                    " and K of " + std::to_string(k_multiple));
}

/// A K-major operand (rows x row_bytes, row-major) as a TMA tensor of bytes,
/// copied in boxes of `box_rows` x 64 bytes with the 64-byte swizzle.
/// Encoding costs a few microseconds of host time, so maps are cached per
/// buffer and shape.
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
    const cuuint32_t box[2]     = {static_cast<cuuint32_t>(kKBytes), static_cast<cuuint32_t>(box_rows)};
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

/// Run kernel_tma with a 128 x BN block tile.
template <typename F, int BN>
void launch_tma_tile(const void* A, const void* Bt, void* C, int M, int K, int N) {
    using T = TmaTile<BN>;
    const std::size_t row_bytes = static_cast<std::size_t>(K) * F::kBytes;
    const CUtensorMap map_a = *tensor_map(A, M, row_bytes, kTmaBM);
    const CUtensorMap map_b = *tensor_map(Bt, N, row_bytes, BN);
    const unsigned blocks = static_cast<unsigned>(M / kTmaBM) * static_cast<unsigned>(N / BN);
    launch<kernel_tma<F, BN>>(dim3(blocks), dim3(T::kThreads), T::kSmemBytes, map_a, map_b,
                              static_cast<typename F::Out*>(C), M, K, N);
}

/// Fraction of the SM slots a grid of `blocks` one-per-SM blocks keeps busy:
/// blocks / (ceil(blocks / sms) * sms).
double wave_efficiency(long blocks, int sms) {
    const long waves = (blocks + sms - 1) / sms;
    return double(blocks) / double(waves * sms);
}

/// Run Level 8, choosing the tile by wave quantisation. The 128 x 256 tile
/// makes the GEMM 3-11% faster when both tiles fill the GPU equally
/// (measured at N = 2048 and 8192); the rule assumes 5% and chooses it
/// unless the 128 x 128 tile's twice as many, smaller blocks fill the waves
/// more than 5% better. On an RTX 5080 (84 SMs) that picks 128 x 128 at
/// N = 1024 (32 wide blocks: 38% of one wave) and N = 4096 (512 wide blocks:
/// 6.1 waves, 87% useful, against 94%), 128 x 256 at N = 2048, 8192, 16384.
template <typename F>
void launch_tma(const void* A, const void* Bt, void* C, int M, int K, int N) {
    static const int sms = sm_count();
    const long tiles_m   = M / kTmaBM;
    if (N % 256 == 0 &&
        1.05 * wave_efficiency(tiles_m * (N / 256), sms) >= wave_efficiency(tiles_m * (N / 128), sms))
        launch_tma_tile<F, 256>(A, Bt, C, M, K, N);
    else
        launch_tma_tile<F, 128>(A, Bt, C, M, K, N);
}

// ############################################################################
// Part 4. cuBLAS reference and Matrix-wrapper internals
// ############################################################################

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
        CUDA_CHECK(cudaMalloc(&p, kWorkspace));
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

/// Copy A and B to the device, run launch(dA, dB, dC, M, K, N), copy C back.
/// Every Matrix<T>-based entry point of the f32/f64 levels goes through this.
template <typename T, typename Launch>
void run_on_device(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, Launch launch_fn) {
    assert(A.cols() == B.rows() && C.rows() == A.rows() && C.cols() == B.cols());
    // Kernels and conversion counts use signed 32-bit element indices.
    for (const std::size_t n : {A.rows(), A.cols(), B.cols(), A.size(), B.size(), C.size()})
        if (n > INT_MAX)
            throw std::length_error("CUDA GEMM exceeds 32-bit indexing limits");
    if (C.size() == 0)
        return;
    if (A.cols() == 0) {
        C.zero();
        return;
    }
    const int M = static_cast<int>(A.rows()), K = static_cast<int>(A.cols()), N = static_cast<int>(B.cols());
    DeviceBuffer<T> dA(A.size()), dB(B.size()), dC(C.size());
    CUDA_CHECK(cudaMemcpy(dA.ptr, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, B.data(), B.size() * sizeof(T), cudaMemcpyHostToDevice));
    launch_fn(dA.ptr, dB.ptr, dC.ptr, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC.ptr, C.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

/// Convert device fp32 A (M x K) and B (K x N) to fp16 copies, run
/// fn(A16, B16) on them and wait for it before the copies are freed.
template <typename Fn>
void with_fp16_copies(const float* A, const float* B, int M, int K, int N, Fn fn) {
    DeviceBuffer<__half> A16(static_cast<std::size_t>(M) * K), B16(static_cast<std::size_t>(K) * N);
    hpc::gemm::gemm_cuda_convert_f32_to_f16_device(A, A16.ptr, M * K);
    hpc::gemm::gemm_cuda_convert_f32_to_f16_device(B, B16.ptr, K * N);
    fn(A16.ptr, B16.ptr);
    CUDA_CHECK(cudaDeviceSynchronize());
}

/// v rounded up to a multiple of m, as an int.
int round_up(std::size_t v, int m) {
    const std::size_t r = (v + m - 1) / m * m;
    if (r > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument("low-precision CUDA GEMM: dimension too large");
    return static_cast<int>(r);
}

/// The low-precision Matrix wrappers: zero-pad A to Mp x Kp and B,
/// transposed, to Np x Kp (tile multiples), run `impl` on the GPU, copy the
/// M x N corner of C back. fp16_acc16 results are widened to Out on the host.
template <typename In, typename Out>
void run_padded(CudaLowp f, CudaLowpImpl impl, const Matrix<In>& A, const Matrix<In>& B, Matrix<Out>& C) {
    if (A.cols() != B.rows() || C.rows() != A.rows() || C.cols() != B.cols())
        throw std::invalid_argument("low-precision CUDA GEMM: shape mismatch");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    if (M == 0 || N == 0)
        return;
    const int Mp = round_up(M, kLowpBM), Np = round_up(N, kLowpBN);
    const int Kp = round_up(K, hpc::gemm::cuda_lowp_k_multiple(f));

    // Zero is all-zero bits in every format here.
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
    DeviceBuffer<In> dA(a.size()), dB(bt.size());
    DeviceBuffer<unsigned char> dC(c_bytes);
    CUDA_CHECK(cudaMemcpy(dA.ptr, a.data(), a.size() * sizeof(In), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, bt.data(), bt.size() * sizeof(In), cudaMemcpyHostToDevice));
    switch (impl) {
        case CudaLowpImpl::mma:    hpc::gemm::gemm_cuda_lowp_device(f, dA.ptr, dB.ptr, dC.ptr, Mp, Kp, Np); break;
        case CudaLowpImpl::tma:    hpc::gemm::gemm_cuda_lowp_tma_device(f, dA.ptr, dB.ptr, dC.ptr, Mp, Kp, Np); break;
        case CudaLowpImpl::cublas: hpc::gemm::gemm_cuda_cublas_lowp_device(f, dA.ptr, dB.ptr, dC.ptr, Mp, Kp, Np); break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned char> c(c_bytes);
    CUDA_CHECK(cudaMemcpy(c.data(), dC.ptr, c_bytes, cudaMemcpyDeviceToHost));
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

// ############################################################################
// Public API (gemm/cuda.hpp)
// ############################################################################
namespace hpc::gemm {

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
    CUDA_CHECK(cudaMemcpy(&h, d.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    return h == 1;
}

// ---------------------------------------------------------------------------
// Levels 0-4: f32 and f64 on the FMA cores
// ---------------------------------------------------------------------------

template <typename T>
void gemm_cuda_naive_device(const T* A, const T* B, T* C, int M, int K, int N) {
    if (empty_product(C, sizeof(T), M, K, N))
        return;
    launch<kernel_naive<T>>(tile_grid(M, N, kTile, kTile), dim3(kTile, kTile), 0, A, B, C, M, K, N);
}

template <typename T>
void gemm_cuda_blocked_device(const T* A, const T* B, T* C, int M, int K, int N) {
    if (empty_product(C, sizeof(T), M, K, N))
        return;
    launch<kernel_blocked<T>>(tile_grid(M, N, kTile, kTile), dim3(kTile, kTile), 0, A, B, C, M, K, N);
}

template <typename T>
void gemm_cuda_reg_tile_device(const T* A, const T* B, T* C, int M, int K, int N) {
    if (empty_product(C, sizeof(T), M, K, N))
        return;
    launch<kernel_reg_tile<T>>(tile_grid(M, N, kBM, kBN), dim3(kRegTileThreads), 0, A, B, C, M, K, N);
}

template <typename T>
void gemm_cuda_double_buf_device(const T* A, const T* B, T* C, int M, int K, int N) {
    if (empty_product(C, sizeof(T), M, K, N))
        return;
    // One thread per kTM x kTN sub-tile: 256 for float, 64 for double (kDBufTile).
    constexpr int BT = kDBufTile<T>;
    launch<kernel_double_buf<T>>(tile_grid(M, N, BT, BT), dim3((BT / kTM) * (BT / kTN)), 0, A, B, C, M, K, N);
}

// Vector loads need K and N to be multiples of the 128-bit vector width
// (4 for float, 2 for double); otherwise Level 2 runs.
template <typename T>
void gemm_cuda_vectorized_device(const T* A, const T* B, T* C, int M, int K, int N) {
    if (empty_product(C, sizeof(T), M, K, N))
        return;
    constexpr int W = VecTraits<T>::kWidth;
    if (K % W != 0 || N % W != 0)
        return gemm_cuda_reg_tile_device(A, B, C, M, K, N);
    launch<kernel_vectorized<T>>(tile_grid(M, N, kBM, kBN), dim3(kRegTileThreads), 0, A, B, C, M, K, N);
}

// ---------------------------------------------------------------------------
// Levels 5-8: fp16 multiply on Tensor Cores
// ---------------------------------------------------------------------------

// Tensor Cores need sm_70+; otherwise Level 3 runs.
void gemm_cuda_wmma_device(const float* A, const float* B, float* C, int M, int K, int N) {
    if (empty_product(C, sizeof(float), M, K, N))
        return;
    if (!cuda_has_tensor_cores())
        return gemm_cuda_double_buf_device(A, B, C, M, K, N);
    launch<kernel_wmma>(tile_grid(M, N, kBlockMN, kBlockMN), dim3(kWarpGridThreads), 0, A, B, C, M, K, N);
}

// The f16 m16n8k16 mma.sync shape needs sm_80+; otherwise Level 5 runs.
void gemm_cuda_mma_ldmatrix_device(const float* A, const float* B, float* C, int M, int K, int N) {
    if (empty_product(C, sizeof(float), M, K, N))
        return;
    if (!cuda_has_ampere())
        return gemm_cuda_wmma_device(A, B, C, M, K, N);
    launch<kernel_mma_ldmatrix>(tile_grid(M, N, kBlockMN, kBlockMN), dim3(kWarpGridThreads), 0, A, B, C, M, K, N);
}

// fp16 inputs, no fallback: M, N multiples of 128, K of 32, sm_70+.
void gemm_cuda_wmma_pipelined_device(const void* A16, const void* B16, float* C, int M, int K, int N) {
    if (empty_product(C, sizeof(float), M, K, N))
        return;
    if (M % kPipeBM || N % kPipeBN || K % kPipeBK)
        throw std::invalid_argument("pipelined WMMA needs M/N multiples of 128 and K of 32");
    if (!cuda_has_tensor_cores())
        throw std::runtime_error("pipelined WMMA needs sm_70+");
    launch<kernel_wmma_pipelined>(tile_grid(M, N, kPipeBM, kPipeBN), dim3(kPipeThreads), 0,
                                  static_cast<const __half*>(A16), static_cast<const __half*>(B16), C, M, K, N);
}

void gemm_cuda_tma_device(const void* A16, const void* Bt16, float* C, int M, int K, int N) {
    gemm_cuda_lowp_tma_device(CudaLowp::fp16, A16, Bt16, C, M, K, N);
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
// Low-precision formats
// ---------------------------------------------------------------------------

void gemm_cuda_lowp_device(CudaLowp f, const void* A, const void* Bt, void* C, int M, int K, int N) {
    if (lowp_empty_product(f, C, M, K, N))
        return;
    require_tiles(f, M, K, N, kLowpBM, kLowpBN);
    with_format(f, [&](auto format) {
        using F = decltype(format);
        const unsigned blocks = static_cast<unsigned>(M / kLowpBM) * static_cast<unsigned>(N / kLowpBN);
        launch<kernel_lowp<F>>(dim3(blocks), dim3(kLowpThreads), kLowpSmemBytes,
                               static_cast<const unsigned char*>(A), static_cast<const unsigned char*>(Bt),
                               static_cast<typename F::Out*>(C), M, K, N);
    });
}

void gemm_cuda_lowp_tma_device(CudaLowp f, const void* A, const void* Bt, void* C, int M, int K, int N) {
    if (lowp_empty_product(f, C, M, K, N))
        return;
    if (!cuda_has_tma())
        throw std::runtime_error("Level 8 (TMA) needs sm_90+");
    require_tiles(f, M, K, N, kTmaBM, 128);  // 128 x 256 tiles also need N % 256: launch_tma
    with_format(f, [&](auto format) { launch_tma<decltype(format)>(A, Bt, C, M, K, N); });
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

// ---------------------------------------------------------------------------
// Matrix-based entry points
// ---------------------------------------------------------------------------

template <typename T>
void gemm_cuda_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, gemm_cuda_naive_device<T>);
}
template <typename T>
void gemm_cuda_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, gemm_cuda_blocked_device<T>);
}
template <typename T>
void gemm_cuda_reg_tile(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, gemm_cuda_reg_tile_device<T>);
}
template <typename T>
void gemm_cuda_double_buf(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, gemm_cuda_double_buf_device<T>);
}
template <typename T>
void gemm_cuda_vectorized(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, gemm_cuda_vectorized_device<T>);
}
void gemm_cuda_wmma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, gemm_cuda_wmma_device);
}
void gemm_cuda_mma_ldmatrix(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, gemm_cuda_mma_ldmatrix_device);
}

// Level 7 with A and B converted to fp16 on the GPU first; shapes that are
// not exact tile multiples, and GPUs without Tensor Cores, run Level 5.
void gemm_cuda_wmma_pipelined(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, [](const float* dA, const float* dB, float* dC, int M, int K, int N) {
        if (!cuda_has_tensor_cores() || M % kPipeBM || N % kPipeBN || K % kPipeBK)
            return gemm_cuda_wmma_device(dA, dB, dC, M, K, N);
        with_fp16_copies(dA, dB, M, K, N,
                         [&](const __half* A16, const __half* B16) {
                             gemm_cuda_wmma_pipelined_device(A16, B16, dC, M, K, N);
                         });
    });
}

// Level 8 with inputs rounded to fp16 on the host (as Levels 5-7 do) and B
// transposed to N x K while padding; without TMA (pre-sm_90) Level 7 runs.
void gemm_cuda_tma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    if (!cuda_has_tma())
        return gemm_cuda_wmma_pipelined(A, B, C);
    run_padded(CudaLowp::fp16, CudaLowpImpl::tma, convert<fp16>(A), convert<fp16>(B), C);
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

// ---------------------------------------------------------------------------
// Timing and device-memory helpers for the benchmarks
// ---------------------------------------------------------------------------

double gemm_cuda_time_ms(const std::function<void()>& launch_fn) {
    struct Event {
        cudaEvent_t value{};
        Event() { CUDA_CHECK(cudaEventCreate(&value)); }
        ~Event() { cudaEventDestroy(value); }
        Event(const Event&)            = delete;
        Event& operator=(const Event&) = delete;
    };
    Event start, stop;
    CUDA_CHECK(cudaEventRecord(start.value));
    launch_fn();
    CUDA_CHECK(cudaEventRecord(stop.value));
    CUDA_CHECK(cudaEventSynchronize(stop.value));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.value, stop.value));
    return ms;
}

void* gemm_cuda_malloc(std::size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}
void gemm_cuda_free(void* ptr) {
    if (ptr)
        cudaFree(ptr);
}
void gemm_cuda_memcpy_h2d(void* dst, const void* src, std::size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
void gemm_cuda_memcpy_d2h(void* dst, const void* src, std::size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}
void gemm_cuda_device_synchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
#define HPC_CUDA_INSTANTIATE(fn)                                                             \
    template void fn<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);     \
    template void fn<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&); \
    template void fn##_device<float>(const float*, const float*, float*, int, int, int);    \
    template void fn##_device<double>(const double*, const double*, double*, int, int, int);
HPC_CUDA_INSTANTIATE(gemm_cuda_naive)
HPC_CUDA_INSTANTIATE(gemm_cuda_blocked)
HPC_CUDA_INSTANTIATE(gemm_cuda_reg_tile)
HPC_CUDA_INSTANTIATE(gemm_cuda_double_buf)
HPC_CUDA_INSTANTIATE(gemm_cuda_vectorized)
HPC_CUDA_INSTANTIATE(gemm_cuda_cublas)
#undef HPC_CUDA_INSTANTIATE

}  // namespace hpc::gemm
