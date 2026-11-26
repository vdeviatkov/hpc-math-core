#pragma once
/**
 * @file gemm_kernels.cuh
 * @brief Every CUDA GEMM kernel and its launcher as templates over the tile
 *        sizes: Levels 0-8 and the low-precision kernels. The public API and
 *        the level table are in gemm/cuda.hpp.
 *
 * The sizes are not in this file. Every entry point takes a parameter
 * struct P (see gemm/cuda.hpp, "Tile parameters") whose members hold each
 * level's sizes; the measured values are in benchmarks/cuda_params.hpp.
 * Only sizes the hardware or an API fixes are constants here (kWmma, the
 * WMMA fragment; kKBytes, the 64-byte rows of the TMA swizzle; the 128-bit
 * vector widths).
 *
 * This header is compiled by nvcc in the target that instantiates the
 * entry points (benchmarks/cuda_instances.cpp, tests/cuda_instances.cpp),
 * with the instantiation macros of cuda/gemm_instantiate.hpp. The parts
 * that do not depend on sizes (device queries, cuBLAS, conversion, timing)
 * are compiled once, in src/cuda/gemm.cu.
 *
 * It reads top to bottom in three parts: device building blocks (PTX
 * wrappers, tile copies, the cp.async pipeline), the kernels from Level 0
 * to Level 8, and the host launchers with the entry points.
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
#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"

#include <cuda.h>      // CUtensorMap, cuTensorMapEncodeTiled (Level 8)
#include <cuda/ptx>    // mbarrier and TMA instructions (Level 8)
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
// code is built with CMAKE_CUDA_ARCHITECTURES=120a.
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ == 1200
#define HPC_FP8_BLOCK_SCALED 1
#endif

#define HPC_CUDA_CHECK(expr)                                                                   \
    do {                                                                                       \
        const cudaError_t _e = (expr);                                                         \
        if (_e != cudaSuccess)                                                                 \
            throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") +              \
                                     std::to_string(__LINE__) + ": " + cudaGetErrorString(_e)); \
    } while (0)

namespace hpc::gemm::cuda_detail {

namespace wmma = nvcuda::wmma;
using hpc::Matrix;

// ############################################################################
// Part 1. Device building blocks
// ############################################################################

/// Shared-memory address of a generic pointer, as PTX's .shared operands take it.
__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// Asynchronous copies (sm_80+).
//
// The cp_async_* helpers are hand-written versions of CUDA's pipeline
// primitives (<cuda_pipeline_primitives.h>); each one does what its
// counterpart does:
//
//   cp_async_or_zero<B>(d, s, v)  __pipeline_memcpy_async(d, s, B, v ? 0 : B)
//   cp_async_16(d, s)             __pipeline_memcpy_async(d, s, 16)
//   cp_async_commit()             __pipeline_commit()
//   cp_async_wait<N>()            __pipeline_wait_prior(N)
//
// Both sides lower to the same three PTX instructions:
//
//   cp.async.{ca,cg}.shared.global [dst], [src], cp-size{, src-size};
//       Start a copy of cp-size bytes (4, 8 or 16) from global to shared
//       memory and return at once; the thread keeps running while the
//       bytes move (SASS LDGSTS, which does not pass through registers).
//       Only src-size bytes are read and the rest of cp-size is zeroed:
//       __pipeline_memcpy_async's zfill is cp-size - src-size. .ca caches
//       in L1 and L2, .cg in L2 only and is allowed for 16 bytes only.
//       dst must not be read until a wait_group covers the copy.
//   cp.async.commit_group;
//       Close the copies this thread issued since the last commit into a
//       group (SASS LDGDEPBAR). Groups are per thread and complete in
//       order; an empty group is allowed.
//   cp.async.wait_group N;
//       Block until at most the N newest groups of this thread are still
//       in flight (SASS DEPBAR.LE SB0, N). N must be a constant. The
//       copied bytes are then visible to this thread only, so other
//       threads read them after a __syncthreads().
//
// The PTX is written by hand rather than through __pipeline_* for two
// reasons. The cache operator is stated here (.ca for 4/8 bytes, .cg for
// 16, where a block reads each tile once and L1 would only be polluted)
// instead of being chosen inside the header. And the #else branches show
// what runs below sm_80: a plain synchronous copy, with commit and wait
// compiling to nothing, so cp_async_pipeline keeps its structure and
// results and only loses the overlap.
//
// cp_async_pipeline is the usual memcpy_async / commit / wait_prior /
// __syncthreads loop built on these.

/// Copy one 4- or 8-byte element from global `src` to shared `dst`, or write
/// zeros when !valid (`src` must still be a readable address). On sm_80+
/// this is an asynchronous cp.async, finished by cp_async_wait; older GPUs
/// copy synchronously. Same as __pipeline_memcpy_async(dst, src, BYTES,
/// valid ? 0 : BYTES).
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
/// once), a synchronous copy on older GPUs. Same as
/// __pipeline_memcpy_async(dst, src, 16).
__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
#ifdef HPC_SM80
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
#else
    *static_cast<uint4*>(dst) = *static_cast<const uint4*>(src);
#endif
}

/// Close the group of cp.async copies this thread issued since the last
/// commit. Same as __pipeline_commit().
__device__ __forceinline__ void cp_async_commit() {
#ifdef HPC_SM80
    asm volatile("cp.async.commit_group;\n");
#endif
}

/// Wait until at most PENDING of this thread's committed groups are in
/// flight. Same as __pipeline_wait_prior(PENDING). Not a compiler memory
/// barrier: follow it with __syncthreads(), as cp_async_pipeline does,
/// before reading the copied tile.
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

// mbarrier and TMA (sm_90+), through cuda::ptx (<cuda/ptx>). Each
// cuda::ptx function emits exactly the PTX instruction it is named after,
// with the qualifiers passed as tag arguments, so the comments name the
// instruction.

/// Initialise an mbarrier that completes a phase after `count` arrivals.
/// mbarrier.init.shared.b64
__device__ __forceinline__ void mbar_init(std::uint64_t* bar, unsigned count) {
    cuda::ptx::mbarrier_init(bar, count);
}

/// Make the initialised barriers visible to the async proxy (TMA) before any
/// copy completes on them. fence.mbarrier_init.release.cluster
__device__ __forceinline__ void mbar_init_fence() {
    cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
}

/// Arrive on the barrier and announce `bytes` that TMA copies will deliver to it.
/// mbarrier.arrive.expect_tx.release.cta.shared::cta.b64
__device__ __forceinline__ void mbar_expect_tx(std::uint64_t* bar, unsigned bytes) {
    cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared,
                                         bar, bytes);
}

/// Arrive on the barrier. mbarrier.arrive.shared.b64
__device__ __forceinline__ void mbar_arrive(std::uint64_t* bar) {
    cuda::ptx::mbarrier_arrive(bar);
}

/// Spin until the barrier's phase with this parity has completed.
/// mbarrier.try_wait.parity.shared::cta.b64
__device__ __forceinline__ void mbar_wait(std::uint64_t* bar, unsigned parity) {
    while (!cuda::ptx::mbarrier_try_wait_parity(bar, parity)) {
    }
}

/// TMA: copy the box at (byte column x, row y) of the tensor `map` describes
/// into shared `dst`, completing its bytes on barrier `bar`.
/// cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
__device__ __forceinline__ void tma_load(void* dst, const CUtensorMap* map, std::uint64_t* bar, int x, int y) {
    const std::int32_t coords[2] = {x, y};
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster, cuda::ptx::space_global, dst, map, coords, bar);
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
// harder to beat than it looks. The block is P::Tile x P::Tile threads.
// ============================================================================

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
// Level 1: Shared-memory tiling, TILE x TILE
//
// A TILE x TILE thread block stages each TILE x TILE k-tile of A and B in
// shared memory, so each global element is loaded once per block instead of
// once per thread. With TILE = 16 a warp covers two 16-thread rows:
// As[ty][p] broadcasts one value per half-warp, and Bs[p][tx] reads the same
// contiguous segment twice. The +1 padding separates the double-precision
// A broadcasts, which otherwise map to the same banks. Each thread still owns
// one output, so the two __syncthreads() per k-tile pay for only TILE FMAs.
// ============================================================================

/// Level 0's one-output-per-thread GEMM, with A and B staged through shared memory.
template <typename T, int TILE>
__global__ void kernel_blocked(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                               int M, int K, int N) {
    __shared__ T As[TILE][TILE + 1];
    __shared__ T Bs[TILE][TILE + 1];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i = blockIdx.y * TILE + ty;
    const int j = blockIdx.x * TILE + tx;
    T acc = T{0};
    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[ty][tx] = (i < M && k0 + tx < K) ? A[i * K + k0 + tx] : T{0};
        Bs[ty][tx] = (k0 + ty < K && j < N) ? B[(k0 + ty) * N + j] : T{0};
        __syncthreads();
        #pragma unroll
        for (int p = 0; p < TILE; ++p)
            acc += As[ty][p] * Bs[p][tx];
        __syncthreads();
    }
    if (i < M && j < N)
        C[i * N + j] = acc;
}

// ============================================================================
// Shared by Levels 2-4: the register tile
//
// RT (P::RegTile<T>, P::DoubleBuf<T> or P::Vectorized<T>) gives the block tile
// RT::BM x RT::BN of C, the k step RT::BK of one shared-memory tile, and the
// RT::TM x RT::TN tile of C each thread keeps in registers. Per k step a
// thread loads TM values of A and TN values of B from shared memory and
// accumulates their outer product: TM·TN FMAs from TM + TN reads.
// ============================================================================

/// Threads of a register-tiled block: one per TM x TN sub-tile.
template <typename RT>
constexpr int reg_tile_threads() {
    static_assert(RT::BM % RT::TM == 0 && RT::BN % RT::TN == 0, "the thread tile divides the block tile");
    return (RT::BM / RT::TM) * (RT::BN / RT::TN);
}

/// c += a * b^T: the outer product of a thread's A column and B row.
template <typename T, int TM, int TN>
__device__ __forceinline__ void fma_outer(T (&c)[TM][TN], const T (&a)[TM], const T (&b)[TN]) {
    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n)
            c[m][n] += a[m] * b[n];
}

/// Accumulate one shared-memory k-tile into the register tile at (tm, tn)
/// of the block tile. As is transposed (As[k][m]), Bs natural (Bs[k][n]).
template <typename T, int BK, int TM, int TN, int LDA, int LDB>
__device__ __forceinline__ void tile_fma(T (&acc)[TM][TN], const T (&As)[BK][LDA], const T (&Bs)[BK][LDB],
                                         int tm, int tn) {
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        T a[TM], b[TN];
        #pragma unroll
        for (int m = 0; m < TM; ++m)
            a[m] = As[k][tm + m];
        #pragma unroll
        for (int n = 0; n < TN; ++n)
            b[n] = Bs[k][tn + n];
        fma_outer(acc, a, b);
    }
}

/// Write a thread's register tile to C at (tm, tn) of the block tile,
/// clipped to the `rows` x `cols` part inside the matrix.
template <typename T, int TM, int TN>
__device__ __forceinline__ void store_tile(T* C, int ldc, const T (&acc)[TM][TN], int tm, int tn, int rows,
                                           int cols) {
    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n)
            if (tm + m < rows && tn + n < cols)
                C[(tm + m) * ldc + tn + n] = acc[m][n];
}

// ============================================================================
// Level 2: Register-tiled
//
// Shared memory: As[BK][BM+1], A's tile transposed so a thread's TM rows of
// A are contiguous; Bs[BK][BN+1], B's tile as is. The +1 padding spreads
// the transposed stores of A over the banks: unpadded, a float warp's 32
// stores would fall in two banks.
//
// Inner loop: outer product of an As column and a Bs row -> TM x TN FMAs per
// k step, all from registers. Arithmetic intensity per block is
// 2·BM·BN·K FLOP / ((BM + BN)·K·sizeof(T) B): 32 FLOP/byte for a 128 x 128
// float tile, against 4 for Level 1's 16 x 16 tile.
// ============================================================================

/// BM x BN block tile, TM x TN outputs per thread, single shared-memory buffer.
template <typename T, typename RT>
__global__ void __launch_bounds__(reg_tile_threads<RT>())
kernel_reg_tile(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int K, int N) {
    constexpr int BM = RT::BM, BN = RT::BN, BK = RT::BK, TM = RT::TM, TN = RT::TN;
    __shared__ T As[BK][BM + 1];
    __shared__ T Bs[BK][BN + 1];

    const int2 size = block_tile<BM, BN>(A, B, C, M, K, N);
    const int rows = size.x, cols = size.y;
    const int tm = threadIdx.x / (BN / TN) * TM;  // this thread's TM x TN tile
    const int tn = threadIdx.x % (BN / TN) * TN;

    T acc[TM][TN] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tile<BM, BK, true>(&As[0][0], BM + 1, A + k0, K, rows, K - k0);
        load_tile<BK, BN, false>(&Bs[0][0], BN + 1, B + k0 * N, N, K - k0, cols);
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
// DB = P::DoubleBuf<T>, per element type, because static shared memory is
// limited to 48 KB: two buffers of a 128 x 128 tile take
// 2 * 16 * (128+1 + 128+1) * 8 = 66,048 B in double (too large) and
// 33,024 B in float; a 64 x 64 double tile takes 33,280 B. DB::MinBlocks is
// the __launch_bounds__ minimum of resident blocks per SM.
// ============================================================================

/// Level 2 with a two-stage cp.async pipeline.
template <typename T, typename DB>
__global__ void __launch_bounds__(reg_tile_threads<DB>(), DB::MinBlocks)
kernel_double_buf(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int K, int N) {
    constexpr int BM = DB::BM, BN = DB::BN, BK = DB::BK, TM = DB::TM, TN = DB::TN;
    static_assert(2 * BK * (BM + 1 + BN + 1) * sizeof(T) <= 48 * 1024, "static shared memory is 48 KB");
    __shared__ T As[2][BK][BM + 1];
    __shared__ T Bs[2][BK][BN + 1];

    const int2 size = block_tile<BM, BN>(A, B, C, M, K, N);
    const int rows = size.x, cols = size.y;
    const int tm = threadIdx.x / (BN / TN) * TM;
    const int tn = threadIdx.x % (BN / TN) * TN;

    T acc[TM][TN] = {};
    cp_async_pipeline<2>(
        (K + BK - 1) / BK,
        [&](int kt, int s) {
            const int k0 = kt * BK;
            load_tile<BM, BK, true, true>(&As[s][0][0], BM + 1, A + k0, K, rows, K - k0);
            load_tile<BK, BN, false, true>(&Bs[s][0][0], BN + 1, B + k0 * N, N, K - k0, cols);
        },
        [&](int s) { tile_fma(acc, As[s], Bs[s], tm, tn); });
    store_tile(C, N, acc, tm, tn, rows, cols);
}

// ============================================================================
// Level 4: Vectorized loads (float4/double2) + shared-memory XOR swizzle
//
// Level 2's register tile structure (RT = P::Vectorized<T>) with two changes:
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
template <typename T, typename RT>
__global__ void __launch_bounds__(reg_tile_threads<RT>())
kernel_vectorized(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int K, int N) {
    using Vec          = typename VecTraits<T>::Vec;
    constexpr int W    = VecTraits<T>::kWidth;
    constexpr int BM = RT::BM, BN = RT::BN, BK = RT::BK, TM = RT::TM, TN = RT::TN;
    constexpr int kSlots = BM / W;  // vector slots per row
    static_assert(BM == BN, "one swizzle serves both tiles");
    static_assert((kSlots & (kSlots - 1)) == 0, "the swizzle needs a power-of-two number of slots");
    static_assert(TM % W == 0 && TN % W == 0 && BK % W == 0, "thread tiles and k step are whole vectors");

    __shared__ alignas(16) T As[BK][BM];  // As[k][swizzle(k, m)] = A(m, k)
    __shared__ alignas(16) T Bs[BK][BN];  // Bs[k][swizzle(k, n)] = B(k, n)

    const int2 size = block_tile<BM, BN>(A, B, C, M, K, N);
    const int rows = size.x, cols = size.y;
    const int tm = threadIdx.x / (BN / TN) * TM;
    const int tn = threadIdx.x % (BN / TN) * TN;

    T acc[TM][TN] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        // A: each vector is W consecutive k of row m, scattered into W rows
        // of As. K % W == 0, so a vector that starts inside K ends inside it.
        for (int v = threadIdx.x; v < BM * BK / W; v += blockDim.x) {
            const int m = v / (BK / W), k = v % (BK / W) * W;
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
        for (int v = threadIdx.x; v < BK * BN / W; v += blockDim.x) {
            const int k = v / (BN / W), n = v % (BN / W) * W;
            Vec x{};
            if (k0 + k < K && n < cols)
                x = *reinterpret_cast<const Vec*>(B + (k0 + k) * N + n);
            *reinterpret_cast<Vec*>(&Bs[k][swizzle<W, kSlots>(k, n)]) = x;
        }
        __syncthreads();

        // A thread's TM values of A and TN of B are whole vector slots
        // (tm, tn are multiples of W), so they are read as vectors too.
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            T a[TM], b[TN];
            #pragma unroll
            for (int v = 0; v < TM / W; ++v)
                load_vec<Vec>(&a[v * W], &As[k][swizzle<W, kSlots>(k, tm + v * W)]);
            #pragma unroll
            for (int v = 0; v < TN / W; ++v)
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
// Each warp computes a 16x16 output tile of C with wmma::mma_sync. A block
// is WG::WarpsM x WG::WarpsN warps (WG = P::WarpGrid), a (16·WarpsM) x
// (16·WarpsN) tile of C.
//
// A and B are fp32; each tile is converted to fp16 on its way into shared
// memory, multiplied with wmma::mma_sync (fp16 x fp16 -> fp32) and
// accumulated in fp32 fragments. The conversion costs ~1e-3 relative error,
// hence the relaxed test tolerance.
// ============================================================================
constexpr int kWmma = 16;  // the WMMA fragment is 16 x 16 x 16, fixed by the API

template <typename WG>
constexpr int warp_grid_threads() {
    return WG::WarpsM * WG::WarpsN * 32;
}

/// Level 5's block: one 16 x 16 wmma fragment per warp. Also Level 6's body
/// on GPUs compiled for below sm_80 (see kernel_mma_ldmatrix).
template <typename WG>
__device__ __forceinline__ void wmma_tile(const float* __restrict__ A, const float* __restrict__ B,
                                          float* __restrict__ C, int M, int K, int N) {
    constexpr int BM = WG::WarpsM * kWmma, BN = WG::WarpsN * kWmma;
    // No +1 padding: wmma::load_matrix_sync requires the leading dimension of
    // a __half fragment to be a multiple of 8 elements (16 bytes); BM and BN
    // are multiples of 16, BM + 1 would fail with cudaErrorMisalignedAddress.
    __shared__ alignas(32) __half As[kWmma][BM];  // transposed: As[k][m] = A(m, k)
    __shared__ alignas(32) __half Bs[kWmma][BN];  // natural:    Bs[k][n] = B(k, n)

    const int2 size = block_tile<BM, BN>(A, B, C, M, K, N);
    const int rows = size.x, cols = size.y;
    const int warp = threadIdx.x / 32;
    const int wm = warp / WG::WarpsN * kWmma;  // this warp's 16x16 tile
    const int wn = warp % WG::WarpsN * kWmma;
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
        load_tile<BM, kWmma, true>(&As[0][0], BM, A + k0, K, rows, K - k0);
        load_tile<kWmma, BN, false>(&Bs[0][0], BN, B + k0 * N, N, K - k0, cols);
        __syncthreads();
        if (active) {
            wmma::load_matrix_sync(a_frag, &As[0][wm], BM);
            wmma::load_matrix_sync(b_frag, &Bs[0][wn], BN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    // store_matrix_sync always writes a full 16x16 tile and needs a
    // 32-byte-aligned destination, so a direct store is only safe for a tile
    // wholly inside C with N % 8 == 0. Any other tile goes through a per-warp
    // shared-memory staging tile and is copied out element by element.
    __shared__ alignas(32) float Cs[WG::WarpsM * WG::WarpsN][kWmma * kWmma];
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

/// Level 5. WG::MinBlocks is the __launch_bounds__ minimum of resident
/// blocks per SM (with the measured 4 x 4 grid: three 512-thread blocks,
/// which caps the kernel at 40 registers instead of 64).
template <typename WG>
__global__ void __launch_bounds__(warp_grid_threads<WG>(), WG::MinBlocks)
kernel_wmma(const float* A, const float* B, float* C, int M, int K, int N) {
    wmma_tile<WG>(A, B, C, M, K, N);
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
// How each WMMA call of kernel_wmma maps onto this kernel:
//
//   fill_fragment(c_frag, 0)     float acc[2][4] = {}: two 16x8 tiles, four
//                                fp32 per lane each.
//   load_matrix_sync(a_frag, …)  ldmatrix_x4: the 16x16 A tile as four 8x8
//                                matrices, lane l addressing row l % 8 of
//                                matrix l / 8.
//   load_matrix_sync(b_frag, …)  two ldmatrix_x2_trans: the left and right
//                                16x8 halves of the 16x16 B tile.
//   mma_sync(c, a, b, c)         two Fp16::mma (mma.sync.m16n8k16) that
//                                share the A registers: columns 0-7, 8-15.
//   store_matrix_sync(C, …)      each lane stores its four accumulators per
//                                tile at the positions in the layout below.
//   row_major / col_major tags   whether ldmatrix uses .trans. mma.sync
//                                has only .row.col for f16, so an operand
//                                stored the other way round is transposed
//                                on load; kernel_wmma's col_major A
//                                (As[k][m]) would need .trans, so As is
//                                stored [m][k] here instead.
//
// nvcc lowers the WMMA calls themselves to PTX wmma.load / wmma.mma, and
// ptxas turns those into the same Tensor Core instructions (a 16x16x16
// wmma.mma is two m16n8k16 HMMAs on sm_80+). The difference is who
// chooses: WMMA leaves the fragment layout unspecified and picks the
// shared-memory loads itself from a (pointer, ld) pair, so it can read only
// plain strided tiles; with ldmatrix each lane supplies its own row
// address, which is what lets kernel_lowp and kernel_tma read padded and
// swizzled layouts, and mma.sync's documented accumulator layout is what
// lets store_acc write C straight from registers.
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

/// Level 5's tiling (WarpsM x WarpsN warps, 16 x 16 per warp) with ldmatrix + mma.sync PTX.
template <typename WG>
__global__ void __launch_bounds__(warp_grid_threads<WG>())
kernel_mma_ldmatrix(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                    int M, int K, int N) {
#ifdef HPC_SM80
    constexpr int BM = WG::WarpsM * 16, BN = WG::WarpsN * 16;
    __shared__ __half As[BM][16];  // natural: As[m][k], matches `.row`
    __shared__ __half Bs[16][BN];  // natural: Bs[k][n], needs `.trans`

    const int2 size = block_tile<BM, BN>(A, B, C, M, K, N);
    const int rows = size.x, cols = size.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / WG::WarpsN * 16;  // this warp's 16x16 tile
    const int wn = warp % WG::WarpsN * 16;

    float acc[2][4] = {};  // two 16x8 tiles, 4 fp32 registers per lane each
    for (int k0 = 0; k0 < K; k0 += 16) {
        load_tile<BM, 16, false>(&As[0][0], 16, A + k0, K, rows, K - k0);
        load_tile<16, BN, false>(&Bs[0][0], BN, B + k0 * N, N, K - k0, cols);
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
    wmma_tile<WG>(A, B, C, M, K, N);
#endif
}

// ============================================================================
// Level 7: Pipelined Tensor Cores via WMMA -- bigger tiles + cp.async
// double buffering.
//
// kernel_wmma (64x64 block tile, single-buffered, one mma_sync per warp per
// k-step) runs at about an eighth of cuBLAS FP16 on the same Tensor
// Cores. This kernel closes most of that gap the way CUTLASS-style kernels
// do, still on the wmma:: C++ API. PP = P::Pipe gives the sizes:
//
//   1. A bigger block tile, PP::BM x PP::BN with k step PP::BK, so more work
//      per shared-memory round trip and per __syncthreads().
//   2. A bigger warp tile: each warp owns PP::WarpM x PP::WarpN outputs,
//      (WarpM/16) x (WarpN/16) fragments, and issues that many mma_sync per
//      k-sub-step. Each b_frag is loaded once and reused for every fm, each
//      a_frag once for all fn — the register blocking of kernel_reg_tile,
//      at fragment granularity.
//   3. Double buffering with cp_async_pipeline, as in kernel_double_buf.
//   4. Padded shared-memory leading dimensions (pipe_lda / pipe_ldb below).
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
//     multiples of BM and BN and K of BK (the launcher falls back to
//     kernel_wmma otherwise): then every chunk starts at an element offset
//     that is a multiple of 8, i.e. 16-byte aligned, and no bounds checks
//     are needed.
//
// Shared-memory leading dimensions are padded by 8 halves (16 bytes). A
// load_matrix_sync fragment reads 16 rows of 16 halves; a row starts in bank
// (row * ld * 2 / 4) % 32. Unpadded, ld = 32 (64 B/row) gives only 2
// distinct starting banks over 16 rows (8-way conflict), and ld = 128
// (256 B/row) puts every row in the same bank (16-way). Padding by 8 gives 8
// distinct starts: a 2-way conflict. Measured with Nsight Compute on an
// RTX 5080 (128 x 128, BK = 32): shared-load conflicts fall from 85% to 1%
// of wavefronts, 80.8 → 100.5 TFLOP/s. The pad is 8, not 1:
// load_matrix_sync needs ld to be a multiple of 8 halves and cp.async a
// 16-byte-aligned destination. An XOR swizzle (as in kernel_vectorized) is
// not possible here: load_matrix_sync takes a plain (pointer, ld) pair.
// ============================================================================

template <typename PP>
struct PipeShape {
    static constexpr int kFragM    = PP::WarpM / kWmma;
    static constexpr int kFragN    = PP::WarpN / kWmma;
    static constexpr int kWarpCols = PP::BN / PP::WarpN;
    static constexpr int kThreads  = (PP::BM / PP::WarpM) * kWarpCols * 32;
    static constexpr int kLda      = PP::BK + 8;  // padded leading dimensions, in halves
    static constexpr int kLdb      = PP::BN + 8;
    static_assert(PP::BM % PP::WarpM == 0 && PP::BN % PP::WarpN == 0 && PP::BK % kWmma == 0 &&
                      PP::WarpM % kWmma == 0 && PP::WarpN % kWmma == 0,
                  "warp tiles divide the block tile; all are whole WMMA fragments");
    static_assert(2 * (PP::BM * kLda + PP::BK * kLdb) * 2 <= 48 * 1024, "static shared memory is 48 KB");
};

/// fp16 A (M x K) times fp16 B (K x N), BM x BN block tile, WarpM x WarpN
/// per warp, double-buffered with cp.async. Exact tile multiples.
template <typename PP>
__global__ void __launch_bounds__(PipeShape<PP>::kThreads)
kernel_wmma_pipelined(const __half* __restrict__ A16, const __half* __restrict__ B16, float* __restrict__ C,
                      int M, int K, int N) {
    using S = PipeShape<PP>;
    constexpr int BM = PP::BM, BN = PP::BN, BK = PP::BK;
    __shared__ alignas(32) __half As[2][BM][S::kLda];  // natural: As[m][k], padded ld
    __shared__ alignas(32) __half Bs[2][BK][S::kLdb];  // natural: Bs[k][n], padded ld

    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    A16 += static_cast<std::size_t>(row0) * K;
    B16 += col0;
    C += static_cast<std::size_t>(row0) * N + col0;
    const int warp = threadIdx.x / 32;
    const int wm = warp / S::kWarpCols * PP::WarpM;  // this warp's WarpM x WarpN tile
    const int wn = warp % S::kWarpCols * PP::WarpN;

    wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float> c_frag[S::kFragM][S::kFragN];
    #pragma unroll
    for (int fm = 0; fm < S::kFragM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < S::kFragN; ++fn)
            wmma::fill_fragment(c_frag[fm][fn], 0.0f);

    auto load = [&](int kt, int s) {
        const int k0 = kt * BK;
        copy_tile_16B<BM, BK * 2, S::kThreads>(&As[s][0][0], S::kLda * 2, A16 + k0, std::size_t(K) * 2);
        copy_tile_16B<BK, BN * 2, S::kThreads>(&Bs[s][0][0], S::kLdb * 2, B16 + static_cast<std::size_t>(k0) * N,
                                               std::size_t(N) * 2);
    };
    auto compute = [&](int s) {
        #pragma unroll
        for (int ks = 0; ks < BK; ks += kWmma) {
            // Each fn's b_frag is loaded once per k-sub-step and reused for
            // every fm below: the register blocking of kernel_reg_tile.
            wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, __half, wmma::row_major> b_frag[S::kFragN];
            #pragma unroll
            for (int fn = 0; fn < S::kFragN; ++fn)
                wmma::load_matrix_sync(b_frag[fn], &Bs[s][ks][wn + fn * kWmma], S::kLdb);
            #pragma unroll
            for (int fm = 0; fm < S::kFragM; ++fm) {
                wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, __half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, &As[s][wm + fm * kWmma][ks], S::kLda);
                #pragma unroll
                for (int fn = 0; fn < S::kFragN; ++fn)
                    wmma::mma_sync(c_frag[fm][fn], a_frag, b_frag[fn], c_frag[fm][fn]);
            }
        }
    };
    cp_async_pipeline<2>(K / BK, load, compute);

    #pragma unroll
    for (int fm = 0; fm < S::kFragM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < S::kFragN; ++fn)
            wmma::store_matrix_sync(C + (wm + fm * kWmma) * N + wn + fn * kWmma, c_frag[fm][fn], N,
                                    wmma::mem_row_major);
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
constexpr int kKBytes = 64;  // bytes of K per pipeline stage: the 64-byte rows of the TMA swizzle

/// Grouped block order: consecutive blocks walk down a column of GROUP_M
/// tile rows before moving right, so the blocks resident at one time share
/// a few A strips and a few Bt strips, which then stay in L2. Row-major
/// order would stream all of Bt through L2 for every row of tiles.
/// Returns this block's (tile row, tile column).
template <int GROUP_M>
__device__ __forceinline__ int2 grouped_tile(int tiles_m, int tiles_n) {
    const int per_group = GROUP_M * tiles_n;
    const int first_m   = blockIdx.x / per_group * GROUP_M;
    const int group_m   = min(tiles_m - first_m, GROUP_M);
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
// and pipeline of Level 7. LP = P::Lowp gives the block tile LP::BM x LP::BN,
// LP::WarpsN warps across it, each LP::WarpM rows deep, the cp.async
// pipeline depth LP::Stages and the grouped block order's LP::GroupM. Shared
// memory per block is Stages · (BM + BN) · 80 bytes (PaddedRows).
// ============================================================================
template <typename LP>
struct LowpShape {
    static constexpr int kWarpN      = LP::BN / LP::WarpsN;
    static constexpr int kThreads    = (LP::BM / LP::WarpM) * LP::WarpsN * 32;
    static constexpr int kStageA     = LP::BM * PaddedRows::kRowBytes;  // A rows, then Bt rows
    static constexpr int kStageBytes = (LP::BM + LP::BN) * PaddedRows::kRowBytes;
    static constexpr int kSmemBytes  = LP::Stages * kStageBytes;
    static_assert(LP::BM % LP::WarpM == 0 && LP::WarpM % 16 == 0 && kWarpN % 16 == 0,
                  "warp tiles are whole m16 x (2 n8) mma tiles");
    static_assert(kSmemBytes <= 101376, "sm_120 allows 99 KiB of shared memory per block");
};

/// C = A * Bt^T for one BM x BN tile. A: M x K, Bt: N x K, C: M x N.
template <typename F, typename LP>
__global__ void __launch_bounds__(LowpShape<LP>::kThreads)
kernel_lowp(const unsigned char* __restrict__ A, const unsigned char* __restrict__ Bt,
            typename F::Out* __restrict__ C, int M, int K, int N) {
#ifdef HPC_SM80
    using S = LowpShape<LP>;
    extern __shared__ __align__(16) unsigned char smem[];
    const std::size_t ld = static_cast<std::size_t>(K) * F::kBytes;  // bytes per row of A and Bt

    const int2 tile = grouped_tile<LP::GroupM>(M / LP::BM, N / LP::BN);
    A += static_cast<std::size_t>(tile.x) * LP::BM * ld;
    Bt += static_cast<std::size_t>(tile.y) * LP::BN * ld;
    const int warp = threadIdx.x / 32;
    const int wm = warp / LP::WarpsN * LP::WarpM, wn = warp % LP::WarpsN * S::kWarpN;

    typename F::Acc acc[LP::WarpM / 16][S::kWarpN / 8][F::kAccRegs] = {};
    auto stage = [&](int s) { return smem + s * S::kStageBytes; };
    cp_async_pipeline<LP::Stages>(
        static_cast<int>(ld / kKBytes),
        [&](int kt, int s) {
            const std::size_t k = static_cast<std::size_t>(kt) * kKBytes;
            copy_tile_16B<LP::BM, kKBytes, S::kThreads>(stage(s), PaddedRows::kRowBytes, A + k, ld);
            copy_tile_16B<LP::BN, kKBytes, S::kThreads>(stage(s) + S::kStageA, PaddedRows::kRowBytes, Bt + k, ld);
        },
        [&](int s) {
            #pragma unroll
            for (int chunk = 0; chunk < kKBytes / 16; chunk += 2)
                warp_mma_k32<F, PaddedRows>(acc, stage(s), stage(s) + S::kStageA, wm, wn, chunk);
        });

    C += (static_cast<std::size_t>(tile.x) * LP::BM + wm) * N + static_cast<std::size_t>(tile.y) * LP::BN + wn;
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
//   1. TMA (cp.async.bulk.tensor): one thread copies a whole BM-row (A)
//      or BN-row (Bt) x 64-byte tile with one instruction. The copy engine
//      walks the tensor described by a CUtensorMap built on the host.
//   2. Hardware swizzle instead of padding (Swizzle64).
//   3. Warp specialisation. One producer warp only issues TMA copies; the
//      consumer warps only run ldmatrix and mma.sync, each on a 64 x 64
//      tile. They synchronise through two mbarriers per stage: `full` (the
//      producer announces the bytes it expects; TMA completes the
//      transaction) and `empty` (each consumer warp arrives when it has read
//      the stage). No __syncthreads() in the main loop.
//   4. Two block tiles, TP::BM x 256 ("wide", TP::StagesWide stages) and
//      TP::BM x 128 ("narrow", TP::StagesNarrow stages); the host picks one
//      per launch by how full the last wave of blocks is (launch_tma).
//      TP = P::Tma.
// ============================================================================

/// Level 8's block tile: BM x BN with BN = 256 (wide) or 128 (narrow).
template <typename TP, int BN>
struct TmaTile {
    static constexpr int kBM         = TP::BM;
    static constexpr int kStages     = BN == 256 ? TP::StagesWide : TP::StagesNarrow;
    static constexpr int kWarpsN     = BN / 64;
    static constexpr int kConsumers  = (kBM / 64) * kWarpsN;
    static constexpr int kThreads    = (kConsumers + 1) * 32;  // + 1 producer warp
    static constexpr int kStageA     = kBM * kKBytes;
    static constexpr int kStageBytes = kStageA + BN * kKBytes;
    // + 1024 to align the stages (see kernel_tma), + the two barriers per stage
    static constexpr int kSmemBytes  = kStages * kStageBytes + 1024 + 2 * kStages * 8;
    static_assert(BN == 128 || BN == 256, "Level 8 has a 128- and a 256-column tile");
    static_assert(kBM % 64 == 0, "consumer warps own 64 x 64 tiles");
    static_assert(kSmemBytes <= 101376, "sm_120 allows 99 KiB of shared memory per block");
};

/// C = A * Bt^T for one BM x BN tile, A and Bt read through TMA tensor maps.
template <typename F, typename TP, int BN>
__global__ void __launch_bounds__(TmaTile<TP, BN>::kThreads, 1)
kernel_tma(const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b,
           typename F::Out* __restrict__ C, int M, int K, int N) {
#ifdef HPC_SM90
    using T = TmaTile<TP, BN>;
    extern __shared__ unsigned char smem_raw[];
    // The swizzle pattern is a function of the shared-memory address: stages
    // must start on a 1024-byte boundary (512 suffices for 64-byte swizzle).
    unsigned char* smem = smem_raw + ((1024 - (smem_addr(smem_raw) & 1023)) & 1023);
    auto stage_a = [&](int s) { return smem + s * T::kStageBytes; };
    auto stage_b = [&](int s) { return smem + s * T::kStageBytes + T::kStageA; };
    std::uint64_t* full  = reinterpret_cast<std::uint64_t*>(smem + T::kStages * T::kStageBytes);
    std::uint64_t* empty = full + T::kStages;

    const int2 tile   = grouped_tile<TP::GroupM>(M / T::kBM, N / BN);
    const int k_tiles = K * F::kBytes / kKBytes;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

    if (threadIdx.x == 0) {
        for (int s = 0; s < T::kStages; ++s) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], T::kConsumers);
        }
        mbar_init_fence();
    }
    __syncthreads();

    if (warp == T::kConsumers) {  // producer
        if (lane == 0) {
            for (int kt = 0; kt < k_tiles; ++kt) {
                const int s = kt % T::kStages;
                if (kt >= T::kStages)  // wait until every consumer has read the previous use of s
                    mbar_wait(&empty[s], (kt / T::kStages - 1) & 1);
                mbar_expect_tx(&full[s], T::kStageBytes);
                tma_load(stage_a(s), &map_a, &full[s], kt * kKBytes, tile.x * T::kBM);
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

    C += (static_cast<std::size_t>(tile.x) * T::kBM + wm) * N + static_cast<std::size_t>(tile.y) * BN + wn;
    store_acc<F>(C, N, acc);
#else
    __trap();
#endif
}

// ############################################################################
// Part 3. Host launchers
// ############################################################################

/// Number of SMs of the selected device.
inline int sm_count() {
    int device = 0, count = 0;
    HPC_CUDA_CHECK(cudaGetDevice(&device));
    HPC_CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
    return count;
}

/// An owning device allocation of n elements of T.
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(std::size_t n) { HPC_CUDA_CHECK(cudaMalloc(&ptr, (n ? n : 1) * sizeof(T))); }
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
        HPC_CUDA_CHECK(configured);
    }
    KERNEL<<<grid, block, smem>>>(args...);
    HPC_CUDA_CHECK(cudaGetLastError());
}

/// The 2-D grid of BM x BN block tiles that covers an M x N output.
inline dim3 tile_grid(int M, int N, int BM, int BN) {
    return dim3(static_cast<unsigned>((N + BN - 1) / BN), static_cast<unsigned>((M + BM - 1) / BM));
}

/// Argument checks shared by every device entry point: throw on a negative
/// dimension; return true when there is nothing to multiply — M or N is 0,
/// or K is 0, after clearing C (elem_bytes per element) on the default stream.
inline bool empty_product(void* C, std::size_t elem_bytes, int M, int K, int N) {
    if (M < 0 || K < 0 || N < 0)
        throw std::invalid_argument("CUDA GEMM dimensions must be nonnegative");
    if (M == 0 || N == 0)
        return true;
    if (K != 0)
        return false;
    HPC_CUDA_CHECK(cudaMemsetAsync(C, 0, static_cast<std::size_t>(M) * N * elem_bytes));
    return true;
}

/// Bytes per input element and per output element of a low-precision format.
inline int input_bytes(CudaLowp f) { return f == CudaLowp::int8 || f == CudaLowp::fp8_e4m3 ? 1 : 2; }
inline int output_bytes(CudaLowp f) { return f == CudaLowp::fp16_acc16 ? 2 : 4; }

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
inline bool lowp_empty_product(CudaLowp f, void* C, int M, int K, int N) {
    if (M < 0 || K < 0 || N < 0)
        throw std::invalid_argument("CUDA GEMM dimensions must be nonnegative");
    if (!cuda_lowp_supported(f))
        throw std::runtime_error("low-precision CUDA GEMM: format not supported on this device");
    return empty_product(C, output_bytes(f), M, K, N);
}

/// Throw unless M and N are multiples of the block tile and K of the
/// format's k multiple: the low-precision kernels have no bounds checks.
inline void require_tiles(CudaLowp f, int M, int K, int N, int BM, int BN) {
    const int k_multiple = cuda_lowp_k_multiple(f);
    if (M % BM || N % BN || K % k_multiple)
        throw std::invalid_argument("low-precision CUDA GEMM: M, N must be multiples of " + std::to_string(BM) +
                                    ", " + std::to_string(BN) + " and K of " + std::to_string(k_multiple));
}

/// A K-major operand (rows x row_bytes, row-major) as a TMA tensor of bytes,
/// copied in boxes of `box_rows` x 64 bytes with the 64-byte swizzle.
/// Encoding costs a few microseconds of host time, so maps are cached per
/// buffer and shape.
inline const CUtensorMap* tensor_map(const void* ptr, int rows, std::size_t row_bytes, int box_rows) {
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

/// Run kernel_tma with a BM x BN block tile.
template <typename F, typename TP, int BN>
void launch_tma_tile(const void* A, const void* Bt, void* C, int M, int K, int N) {
    using T = TmaTile<TP, BN>;
    const std::size_t row_bytes = static_cast<std::size_t>(K) * F::kBytes;
    const CUtensorMap map_a = *tensor_map(A, M, row_bytes, T::kBM);
    const CUtensorMap map_b = *tensor_map(Bt, N, row_bytes, BN);
    const unsigned blocks = static_cast<unsigned>(M / T::kBM) * static_cast<unsigned>(N / BN);
    launch<kernel_tma<F, TP, BN>>(dim3(blocks), dim3(T::kThreads), T::kSmemBytes, map_a, map_b,
                                  static_cast<typename F::Out*>(C), M, K, N);
}

/// Fraction of the SM slots a grid of `blocks` one-per-SM blocks keeps busy:
/// blocks / (ceil(blocks / sms) * sms).
inline double wave_efficiency(long blocks, int sms) {
    const long waves = (blocks + sms - 1) / sms;
    return double(blocks) / double(waves * sms);
}

/// Run Level 8, choosing the tile by wave quantisation: the wide tile unless
/// the narrow tile's twice as many, smaller blocks fill the waves more than
/// TP::WideMarginPct percent better. TP::Tile forces one tile (256 or 128)
/// instead; 0 chooses by waves.
template <typename F, typename TP>
void launch_tma(const void* A, const void* Bt, void* C, int M, int K, int N) {
    static const int sms = sm_count();
    const long tiles_m   = M / TP::BM;
    bool wide            = N % 256 == 0;
    if constexpr (TP::Tile == 128)
        wide = false;
    else if constexpr (TP::Tile == 0)
        wide = wide && (1.0 + TP::WideMarginPct / 100.0) * wave_efficiency(tiles_m * (N / 256), sms) >=
                           wave_efficiency(tiles_m * (N / 128), sms);
    if (wide)
        launch_tma_tile<F, TP, 256>(A, Bt, C, M, K, N);
    else
        launch_tma_tile<F, TP, 128>(A, Bt, C, M, K, N);
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
    HPC_CUDA_CHECK(cudaMemcpy(dA.ptr, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice));
    HPC_CUDA_CHECK(cudaMemcpy(dB.ptr, B.data(), B.size() * sizeof(T), cudaMemcpyHostToDevice));
    launch_fn(dA.ptr, dB.ptr, dC.ptr, M, K, N);
    HPC_CUDA_CHECK(cudaDeviceSynchronize());
    HPC_CUDA_CHECK(cudaMemcpy(C.data(), dC.ptr, C.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

/// Convert device fp32 A (M x K) and B (K x N) to fp16 copies, run
/// fn(A16, B16) on them and wait for it before the copies are freed.
template <typename Fn>
void with_fp16_copies(const float* A, const float* B, int M, int K, int N, Fn fn) {
    DeviceBuffer<__half> A16(static_cast<std::size_t>(M) * K), B16(static_cast<std::size_t>(K) * N);
    gemm_cuda_convert_f32_to_f16_device(A, A16.ptr, M * K);
    gemm_cuda_convert_f32_to_f16_device(B, B16.ptr, K * N);
    fn(A16.ptr, B16.ptr);
    HPC_CUDA_CHECK(cudaDeviceSynchronize());
}

/// v rounded up to a multiple of m, as an int.
inline int round_up(std::size_t v, int m) {
    const std::size_t r = (v + m - 1) / m * m;
    if (r > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument("low-precision CUDA GEMM: dimension too large");
    return static_cast<int>(r);
}

}  // namespace hpc::gemm::cuda_detail

// ############################################################################
// Entry points (declared in gemm/cuda.hpp)
// ############################################################################
namespace hpc::gemm {

// ---------------------------------------------------------------------------
// Levels 0-4: f32 and f64 on the FMA cores
// ---------------------------------------------------------------------------

template <typename P, typename T>
void gemm_cuda_naive_device(const T* A, const T* B, T* C, int M, int K, int N) {
    using namespace cuda_detail;
    if (empty_product(C, sizeof(T), M, K, N))
        return;
    launch<kernel_naive<T>>(tile_grid(M, N, P::Tile, P::Tile), dim3(P::Tile, P::Tile), 0, A, B, C, M, K, N);
}

template <typename P, typename T>
void gemm_cuda_blocked_device(const T* A, const T* B, T* C, int M, int K, int N) {
    using namespace cuda_detail;
    if (empty_product(C, sizeof(T), M, K, N))
        return;
    launch<kernel_blocked<T, P::Tile>>(tile_grid(M, N, P::Tile, P::Tile), dim3(P::Tile, P::Tile), 0, A, B, C, M,
                                        K, N);
}

template <typename P, typename T>
void gemm_cuda_reg_tile_device(const T* A, const T* B, T* C, int M, int K, int N) {
    using namespace cuda_detail;
    using RT = typename P::template RegTile<T>;
    if (empty_product(C, sizeof(T), M, K, N))
        return;
    launch<kernel_reg_tile<T, RT>>(tile_grid(M, N, RT::BM, RT::BN), dim3(reg_tile_threads<RT>()), 0, A, B, C, M,
                                   K, N);
}

template <typename P, typename T>
void gemm_cuda_double_buf_device(const T* A, const T* B, T* C, int M, int K, int N) {
    using namespace cuda_detail;
    using DB = typename P::template DoubleBuf<T>;
    if (empty_product(C, sizeof(T), M, K, N))
        return;
    launch<kernel_double_buf<T, DB>>(tile_grid(M, N, DB::BM, DB::BN), dim3(reg_tile_threads<DB>()), 0, A, B, C,
                                     M, K, N);
}

// Vector loads need K and N to be multiples of the 128-bit vector width
// (4 for float, 2 for double); otherwise Level 2 runs.
template <typename P, typename T>
void gemm_cuda_vectorized_device(const T* A, const T* B, T* C, int M, int K, int N) {
    using namespace cuda_detail;
    using RT = typename P::template Vectorized<T>;
    if (empty_product(C, sizeof(T), M, K, N))
        return;
    constexpr int W = VecTraits<T>::kWidth;
    if (K % W != 0 || N % W != 0)
        return gemm_cuda_reg_tile_device<P>(A, B, C, M, K, N);
    launch<kernel_vectorized<T, RT>>(tile_grid(M, N, RT::BM, RT::BN), dim3(reg_tile_threads<RT>()), 0, A, B, C,
                                     M, K, N);
}

// ---------------------------------------------------------------------------
// Levels 5-8: fp16 multiply on Tensor Cores
// ---------------------------------------------------------------------------

// Tensor Cores need sm_70+; otherwise Level 3 runs.
template <typename P>
void gemm_cuda_wmma_device(const float* A, const float* B, float* C, int M, int K, int N) {
    using namespace cuda_detail;
    using WG = typename P::WarpGrid;
    if (empty_product(C, sizeof(float), M, K, N))
        return;
    if (!cuda_has_tensor_cores())
        return gemm_cuda_double_buf_device<P>(A, B, C, M, K, N);
    launch<kernel_wmma<WG>>(tile_grid(M, N, WG::WarpsM * kWmma, WG::WarpsN * kWmma), dim3(warp_grid_threads<WG>()),
                            0, A, B, C, M, K, N);
}

// The f16 m16n8k16 mma.sync shape needs sm_80+; otherwise Level 5 runs.
template <typename P>
void gemm_cuda_mma_ldmatrix_device(const float* A, const float* B, float* C, int M, int K, int N) {
    using namespace cuda_detail;
    using WG = typename P::WarpGrid;
    if (empty_product(C, sizeof(float), M, K, N))
        return;
    if (!cuda_has_ampere())
        return gemm_cuda_wmma_device<P>(A, B, C, M, K, N);
    launch<kernel_mma_ldmatrix<WG>>(tile_grid(M, N, WG::WarpsM * 16, WG::WarpsN * 16),
                                    dim3(warp_grid_threads<WG>()), 0, A, B, C, M, K, N);
}

// fp16 inputs, no fallback: M, N multiples of the block tile, K of its k step, sm_70+.
template <typename P>
void gemm_cuda_wmma_pipelined_device(const void* A16, const void* B16, float* C, int M, int K, int N) {
    using namespace cuda_detail;
    using PP = typename P::Pipe;
    if (empty_product(C, sizeof(float), M, K, N))
        return;
    if (M % PP::BM || N % PP::BN || K % PP::BK)
        throw std::invalid_argument("pipelined WMMA needs M, N multiples of " + std::to_string(PP::BM) + ", " +
                                    std::to_string(PP::BN) + " and K of " + std::to_string(PP::BK));
    if (!cuda_has_tensor_cores())
        throw std::runtime_error("pipelined WMMA needs sm_70+");
    launch<kernel_wmma_pipelined<PP>>(tile_grid(M, N, PP::BM, PP::BN), dim3(PipeShape<PP>::kThreads), 0,
                                      static_cast<const __half*>(A16), static_cast<const __half*>(B16), C, M, K, N);
}

// ---------------------------------------------------------------------------
// Low-precision formats
// ---------------------------------------------------------------------------

template <typename P>
void gemm_cuda_lowp_device(CudaLowp f, const void* A, const void* Bt, void* C, int M, int K, int N) {
    using namespace cuda_detail;
    using LP = typename P::Lowp;
    if (lowp_empty_product(f, C, M, K, N))
        return;
    require_tiles(f, M, K, N, LP::BM, LP::BN);
    with_format(f, [&](auto format) {
        using F = decltype(format);
        const unsigned blocks = static_cast<unsigned>(M / LP::BM) * static_cast<unsigned>(N / LP::BN);
        launch<kernel_lowp<F, LP>>(dim3(blocks), dim3(LowpShape<LP>::kThreads), LowpShape<LP>::kSmemBytes,
                                   static_cast<const unsigned char*>(A), static_cast<const unsigned char*>(Bt),
                                   static_cast<typename F::Out*>(C), M, K, N);
    });
}

template <typename P>
void gemm_cuda_lowp_tma_device(CudaLowp f, const void* A, const void* Bt, void* C, int M, int K, int N) {
    using namespace cuda_detail;
    using TP = typename P::Tma;
    if (lowp_empty_product(f, C, M, K, N))
        return;
    if (!cuda_has_tma())
        throw std::runtime_error("Level 8 (TMA) needs sm_90+");
    require_tiles(f, M, K, N, TP::BM, TP::Tile == 256 ? 256 : 128);  // the wide tile also needs N % 256: launch_tma
    with_format(f, [&](auto format) { launch_tma<decltype(format), TP>(A, Bt, C, M, K, N); });
}

template <typename P>
void gemm_cuda_tma_device(const void* A16, const void* Bt16, float* C, int M, int K, int N) {
    gemm_cuda_lowp_tma_device<P>(CudaLowp::fp16, A16, Bt16, C, M, K, N);
}

// ---------------------------------------------------------------------------
// Matrix-based entry points
// ---------------------------------------------------------------------------

template <typename P, typename T>
void gemm_cuda_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    cuda_detail::run_on_device(A, B, C, gemm_cuda_naive_device<P, T>);
}
template <typename P, typename T>
void gemm_cuda_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    cuda_detail::run_on_device(A, B, C, gemm_cuda_blocked_device<P, T>);
}
template <typename P, typename T>
void gemm_cuda_reg_tile(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    cuda_detail::run_on_device(A, B, C, gemm_cuda_reg_tile_device<P, T>);
}
template <typename P, typename T>
void gemm_cuda_double_buf(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    cuda_detail::run_on_device(A, B, C, gemm_cuda_double_buf_device<P, T>);
}
template <typename P, typename T>
void gemm_cuda_vectorized(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    cuda_detail::run_on_device(A, B, C, gemm_cuda_vectorized_device<P, T>);
}
template <typename P>
void gemm_cuda_wmma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    cuda_detail::run_on_device(A, B, C, gemm_cuda_wmma_device<P>);
}
template <typename P>
void gemm_cuda_mma_ldmatrix(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    cuda_detail::run_on_device(A, B, C, gemm_cuda_mma_ldmatrix_device<P>);
}

// Level 7 with A and B converted to fp16 on the GPU first; shapes that are
// not exact tile multiples, and GPUs without Tensor Cores, run Level 5.
template <typename P>
void gemm_cuda_wmma_pipelined(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    using PP = typename P::Pipe;
    cuda_detail::run_on_device(A, B, C, [](const float* dA, const float* dB, float* dC, int M, int K, int N) {
        if (!cuda_has_tensor_cores() || M % PP::BM || N % PP::BN || K % PP::BK)
            return gemm_cuda_wmma_device<P>(dA, dB, dC, M, K, N);
        cuda_detail::with_fp16_copies(dA, dB, M, K, N, [&](const __half* A16, const __half* B16) {
            gemm_cuda_wmma_pipelined_device<P>(A16, B16, dC, M, K, N);
        });
    });
}

/// The low-precision Matrix wrappers: zero-pad A to Mp x Kp and B,
/// transposed, to Np x Kp (multiples of the chosen kernel's tile), run
/// `impl` on the GPU, copy the M x N corner of C back. fp16_acc16 results
/// are widened to Out on the host.
template <typename P, typename In, typename Out>
void gemm_cuda_lowp_padded(CudaLowp f, CudaLowpImpl impl, const Matrix<In>& A, const Matrix<In>& B,
                           Matrix<Out>& C) {
    using namespace cuda_detail;
    if (A.cols() != B.rows() || C.rows() != A.rows() || C.cols() != B.cols())
        throw std::invalid_argument("low-precision CUDA GEMM: shape mismatch");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    if (M == 0 || N == 0)
        return;
    const int tile_m = impl == CudaLowpImpl::tma ? P::Tma::BM : P::Lowp::BM;
    const int tile_n = impl == CudaLowpImpl::tma ? (P::Tma::Tile == 256 ? 256 : 128) : P::Lowp::BN;
    const int Mp = round_up(M, tile_m), Np = round_up(N, tile_n);
    const int Kp = round_up(K, cuda_lowp_k_multiple(f));

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
    HPC_CUDA_CHECK(cudaMemcpy(dA.ptr, a.data(), a.size() * sizeof(In), cudaMemcpyHostToDevice));
    HPC_CUDA_CHECK(cudaMemcpy(dB.ptr, bt.data(), bt.size() * sizeof(In), cudaMemcpyHostToDevice));
    switch (impl) {
        case CudaLowpImpl::mma:    gemm_cuda_lowp_device<P>(f, dA.ptr, dB.ptr, dC.ptr, Mp, Kp, Np); break;
        case CudaLowpImpl::tma:    gemm_cuda_lowp_tma_device<P>(f, dA.ptr, dB.ptr, dC.ptr, Mp, Kp, Np); break;
        case CudaLowpImpl::cublas: gemm_cuda_cublas_lowp_device(f, dA.ptr, dB.ptr, dC.ptr, Mp, Kp, Np); break;
    }
    HPC_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned char> c(c_bytes);
    HPC_CUDA_CHECK(cudaMemcpy(c.data(), dC.ptr, c_bytes, cudaMemcpyDeviceToHost));
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

// Level 8 with inputs rounded to fp16 on the host (Levels 5-7 round on the
// GPU) and B transposed to N x K while padding; without TMA (pre-sm_90)
// Level 7 runs.
template <typename P>
void gemm_cuda_tma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    if (!cuda_has_tma())
        return gemm_cuda_wmma_pipelined<P>(A, B, C);
    gemm_cuda_lowp_padded<P>(CudaLowp::fp16, CudaLowpImpl::tma, convert<fp16>(A), convert<fp16>(B), C);
}

template <typename P>
void gemm_cuda_bf16(const Matrix<bf16>& A, const Matrix<bf16>& B, Matrix<float>& C, CudaLowpImpl impl) {
    gemm_cuda_lowp_padded<P>(CudaLowp::bf16, impl, A, B, C);
}
template <typename P>
void gemm_cuda_fp16(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C, CudaLowpImpl impl) {
    gemm_cuda_lowp_padded<P>(CudaLowp::fp16, impl, A, B, C);
}
template <typename P>
void gemm_cuda_fp16_acc16(const Matrix<fp16>& A, const Matrix<fp16>& B, Matrix<float>& C, CudaLowpImpl impl) {
    gemm_cuda_lowp_padded<P>(CudaLowp::fp16_acc16, impl, A, B, C);
}
template <typename P>
void gemm_cuda_i8(const Matrix<std::int8_t>& A, const Matrix<std::int8_t>& B, Matrix<std::int32_t>& C,
                  CudaLowpImpl impl) {
    gemm_cuda_lowp_padded<P>(CudaLowp::int8, impl, A, B, C);
}
template <typename P>
void gemm_cuda_fp8(const Matrix<fp8_e4m3>& A, const Matrix<fp8_e4m3>& B, Matrix<float>& C, CudaLowpImpl impl) {
    gemm_cuda_lowp_padded<P>(CudaLowp::fp8_e4m3, impl, A, B, C);
}

}  // namespace hpc::gemm
