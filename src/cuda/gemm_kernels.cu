/**
 * @file gemm_kernels.cu
 * @brief CUDA GEMM kernel implementations -- progressive optimization ladder.
 *
 * Level 0: kernel_naive          -- global memory only, 1 thread -> 1 C(i,j)
 * Level 1: kernel_blocked        -- TILE=16 shared-memory tiling
 * Level 2: kernel_reg_tile       -- 128x128 thread block, each thread owns 8x8 C tile
 * Level 3: kernel_double_buf     -- Level 2 + double buffering (cp.async on Ampere+)
 * Level 4: kernel_wmma           -- Tensor Cores via WMMA (fp16 in, fp32 accumulate)
 * Level 5: kernel_vectorized     -- Level 2 + float4/double2 loads + XOR-swizzled smem
 * Level 6: kernel_mma_ldmatrix   -- Tensor Cores via raw ldmatrix + mma.sync PTX
 * Level 7: kernel_wmma_pipelined -- WMMA with 128x128 tiles + cp.async double buffering
 * Reference: cuBLAS (SGEMM/DGEMM, TF32, FP16) -- vendor ceiling, not part of the ladder
 *
 * Each level's comment below explains what it adds. The GPU memory
 * hierarchy these kernels work against is described in
 * docs/cache-behavior.md, section 9.
 */

#include "gemm/cuda.hpp"
#include "hpc/matrix.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#include <cublas_v2.h>  // the reference entry points at the end of the file

// cp.async requires sm_80+ (Ampere)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  #include <cuda_pipeline_primitives.h>
  #define HPC_HAVE_CP_ASYNC 1
#endif

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string("CUDA error at " __FILE__      \
                                                 ":" + std::to_string(__LINE__) \
                                                 + ": ") +                      \
                                     cudaGetErrorString(_e));                   \
        }                                                                       \
    } while (0)

// ============================================================================
// Compile-time constants
// ============================================================================

static constexpr int kTile = 16;   // Level 0/1 tile

// Level 2/3 register-tile parameters
static constexpr int kBM = 128;   // thread-block output rows
static constexpr int kBN = 128;   // thread-block output cols
static constexpr int kBK = 16;    // k-step per shared-memory tile
static constexpr int kTM = 8;     // output rows per thread
static constexpr int kTN = 8;     // output cols per thread
// threads per block = (kBM/kTM) * (kBN/kTN) = 16 * 16 = 256

// Level 4 WMMA tile -- fixed by the WMMA API
static constexpr int kWMMA_M = 16;
static constexpr int kWMMA_N = 16;
static constexpr int kWMMA_K = 16;

// ============================================================================
// Level 0: Naive
// ============================================================================
template <typename T>
__global__ void kernel_naive(const T* __restrict__ A,
                              const T* __restrict__ B,
                              T* __restrict__ C,
                              int M, int K, int N) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= N) return;
    T acc = T{0};
    for (int k = 0; k < K; ++k)
        acc += A[i * K + k] * B[k * N + j];
    C[i * N + j] = acc;
}

// ============================================================================
// Level 1: Blocked / Tiled  (TILE=16 shared-memory)
// ============================================================================
template <typename T>
__global__ void kernel_blocked(const T* __restrict__ A,
                                const T* __restrict__ B,
                                T* __restrict__ C,
                                int M, int K, int N) {
    __shared__ T As[kTile][kTile + 1];
    __shared__ T Bs[kTile][kTile + 1];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i  = blockIdx.y * kTile + ty;
    const int j  = blockIdx.x * kTile + tx;
    T acc = T{0};

    const int nTilesK = (K + kTile - 1) / kTile;
    for (int tileK = 0; tileK < nTilesK; ++tileK) {
        const int kA = tileK * kTile + tx;
        const int kB = tileK * kTile + ty;
        As[ty][tx] = (i < M && kA < K) ? A[i * K + kA] : T{0};
        Bs[ty][tx] = (kB < K && j < N) ? B[kB * N + j] : T{0};
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
// 8x8 register-tile micro-kernel shared by Levels 2, 3 and 5
//
// Each thread owns a kTM x kTN tile of C in registers. Per k step it loads
// kTM values of A and kTN values of B from shared memory (how depends on the
// level's shared-memory layout), then accumulates their outer product.
// ============================================================================
template <typename T>
__device__ __forceinline__ void fma_outer(T (&c)[kTM][kTN], const T (&a)[kTM], const T (&b)[kTN]) {
    #pragma unroll
    for (int m = 0; m < kTM; ++m)
        #pragma unroll
        for (int n = 0; n < kTN; ++n)
            c[m][n] += a[m] * b[n];
}

// Write a thread's register tile to C at (row, col), clipped to M x N.
template <typename T>
__device__ __forceinline__ void store_tile(T* C, const T (&c)[kTM][kTN], int row, int col,
                                           int M, int N) {
    #pragma unroll
    for (int m = 0; m < kTM; ++m)
        #pragma unroll
        for (int n = 0; n < kTN; ++n) {
            const int gi = row + m, gj = col + n;
            if (gi < M && gj < N)
                C[gi * N + gj] = c[m][n];
        }
}

// ============================================================================
// Level 2: Register-tiled
//
// Thread block: kBMxkBN = 128x128 outputs
// Threads/block: (kBM/kTM) x (kBN/kTN) = 16 x 16 = 256
// Each thread owns a kTMxkTN = 8x8 register tile of C.
//
// Shared memory layout:
//   As[kBK][kBM] = 16 x 128 -- A sub-tile transposed for column access
//   Bs[kBK][kBN] = 16 x 128 -- B sub-tile in row-major
//
// Inner loop: outer product of As column and Bs row -> 8x8 FMAs per k step,
// all from registers. Arithmetic intensity per block is
// 2·128·128·K FLOP / ((128 + 128)·K·4 B) ≈ 32 FLOP/byte, against ~2 for
// Level 1, where each thread owns one output.
// Bank conflict avoidance: +1 padding on the inner dimension.
// Shared-memory tiles are filled with a strided loop: 256 threads x 8
// iterations = the 2048 elements of each 16x128 tile.
// ============================================================================
template <typename T>
__global__ void __launch_bounds__(256)
kernel_reg_tile(const T* __restrict__ A,
                const T* __restrict__ B,
                T* __restrict__ C,
                int M, int K, int N) {
    // Position of this thread block's output tile.
    const int blockRow = blockIdx.y;  // which 128-row block of C
    const int blockCol = blockIdx.x;  // which 128-col block of C

    // Thread indices within the block.
    const int threadRow = threadIdx.x / (kBN / kTN);  // 0..15
    const int threadCol = threadIdx.x % (kBN / kTN);  // 0..15

    // Global C position for this thread's top-left corner.
    const int cRow = blockRow * kBM + threadRow * kTM;
    const int cCol = blockCol * kBN + threadCol * kTN;

    // Shared memory tiles (padded to avoid bank conflicts).
    __shared__ T As[kBK][kBM + 1];  // kBK x kBM, transposed: column-major A tile
    __shared__ T Bs[kBK][kBN + 1];  // kBK x kBN, row-major B tile

    // Register accumulator tile: kTM x kTN = 8x8 = 64 registers per thread.
    T reg_C[kTM][kTN] = {};

    // 256 threads load 128*16 = 2048 elements of As (8 each) and
    // 16*128 = 2048 elements of Bs (8 each) via a strided loop.
    constexpr int kAElems = kBK * kBM;  // 2048
    constexpr int kBElems = kBK * kBN;  // 2048

    const int nTilesK = (K + kBK - 1) / kBK;

    for (int tileK = 0; tileK < nTilesK; ++tileK) {
        // Load A sub-tile into As[kBK][kBM] (transposed for column-major access).
        // A[blockRow*kBM + c][tileK*kBK + r]
        for (int idx = threadIdx.x; idx < kAElems; idx += blockDim.x) {
            const int r = idx / kBM;   // 0..15  (kBK dimension)
            const int c = idx % kBM;   // 0..127 (kBM dimension)
            const int aRow = blockRow * kBM + c;
            const int aCol = tileK * kBK + r;
            As[r][c] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : T{0};
        }

        // Load B sub-tile into Bs[kBK][kBN].
        // B[tileK*kBK + r][blockCol*kBN + c]
        for (int idx = threadIdx.x; idx < kBElems; idx += blockDim.x) {
            const int r = idx / kBN;   // 0..15  (kBK dimension)
            const int c = idx % kBN;   // 0..127 (kBN dimension)
            const int bRow = tileK * kBK + r;
            const int bCol = blockCol * kBN + c;
            Bs[r][c] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : T{0};
        }

        __syncthreads();

        // Inner loop: walk the kBK dimension, accumulate outer products of
        // A column k (this thread's kTM rows) and B row k (its kTN cols).
        #pragma unroll
        for (int k = 0; k < kBK; ++k) {
            T reg_A[kTM], reg_B[kTN];
            #pragma unroll
            for (int m = 0; m < kTM; ++m)
                reg_A[m] = As[k][threadRow * kTM + m];
            #pragma unroll
            for (int n = 0; n < kTN; ++n)
                reg_B[n] = Bs[k][threadCol * kTN + n];
            fma_outer(reg_C, reg_A, reg_B);
        }

        __syncthreads();
    }

    store_tile(C, reg_C, cRow, cCol, M, N);
}

// ============================================================================
// Level 3: Double-buffered register tile
//
// Same register-tiling as Level 2, but uses two ping-pong shared-memory
// buffers to overlap loading of tile k+1 with computation of tile k.
//
// On sm_80+ the loads use cp.async (__pipeline_memcpy_async / _commit /
// _wait_prior), which copies global -> shared without blocking the threads,
// so the copy overlaps the FMAs. On older GPUs the loads are synchronous:
// the structure is the same, without the overlap.
// ============================================================================
// Static shared memory is limited to 48 KB, so double uses a 64x64 tile:
//   f32, 128x128: 2 * 16 * (128+1 + 128+1) * 4 = 33,024 B
//   f64, 128x128: 2 * 16 * (128+1 + 128+1) * 8 = 66,048 B  (too large)
//   f64,   64x64: 2 * 16 * ( 64+1 +  64+1) * 8 = 33,280 B
template <typename T>
static constexpr int kDBufBM = (sizeof(T) == 8) ? 64 : kBM;
template <typename T>
static constexpr int kDBufBN = (sizeof(T) == 8) ? 64 : kBN;

template <typename T>
__global__ void __launch_bounds__(256)
kernel_double_buf(const T* __restrict__ A,
                  const T* __restrict__ B,
                  T* __restrict__ C,
                  int M, int K, int N) {
    constexpr int LBM = kDBufBM<T>;
    constexpr int LBN = kDBufBN<T>;
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int threadRow = threadIdx.x / (LBN / kTN);
    const int threadCol = threadIdx.x % (LBN / kTN);
    const int cRow = blockRow * LBM + threadRow * kTM;
    const int cCol = blockCol * LBN + threadCol * kTN;

    // Double-buffered shared memory: index 0 and 1 alternate.
    __shared__ T As[2][kBK][LBM + 1];
    __shared__ T Bs[2][kBK][LBN + 1];

    T reg_C[kTM][kTN] = {};

    // Strided tile loads, as in kernel_reg_tile.
    constexpr int kAElems = kBK * LBM;
    constexpr int kBElems = kBK * LBN;

    const int nTilesK = (K + kBK - 1) / kBK;

    // -------------------------------------------------------------------
    // Helper lambda: load tile tileK into shared-memory buffer buf.
    // On Ampere+: issues an async copy and does not synchronise.
    // On older:   copies synchronously and issues __syncthreads.
    //
    // cp.async copies global -> shared only, so its source must be a real
    // global address (not a local variable). Out-of-bounds elements use the
    // zfill form: copy 0 bytes from a dummy in-bounds address (never read)
    // and zero-fill the shared-memory destination.
    // -------------------------------------------------------------------
    auto load_tile = [&](int tileK, int buf) {
        for (int idx = threadIdx.x; idx < kAElems; idx += blockDim.x) {
            const int r = idx / LBM;
            const int c = idx % LBM;
            const int aRow = blockRow * LBM + c;
            const int aCol = tileK * kBK + r;
            const bool aInBounds = (aRow < M && aCol < K);
#ifdef HPC_HAVE_CP_ASYNC
            const T* aSrc = aInBounds ? &A[aRow * K + aCol] : &A[0];
            __pipeline_memcpy_async(&As[buf][r][c], aSrc, sizeof(T),
                                     aInBounds ? 0 : sizeof(T));
#else
            As[buf][r][c] = aInBounds ? A[aRow * K + aCol] : T{0};
#endif
        }
        for (int idx = threadIdx.x; idx < kBElems; idx += blockDim.x) {
            const int r = idx / LBN;
            const int c = idx % LBN;
            const int bRow = tileK * kBK + r;
            const int bCol = blockCol * LBN + c;
            const bool bInBounds = (bRow < K && bCol < N);
#ifdef HPC_HAVE_CP_ASYNC
            const T* bSrc = bInBounds ? &B[bRow * N + bCol] : &B[0];
            __pipeline_memcpy_async(&Bs[buf][r][c], bSrc, sizeof(T),
                                     bInBounds ? 0 : sizeof(T));
#else
            Bs[buf][r][c] = bInBounds ? B[bRow * N + bCol] : T{0};
#endif
        }
#ifdef HPC_HAVE_CP_ASYNC
        __pipeline_commit();
#endif
    };

    // Wait for the in-flight tile, then make it visible to the whole block.
    auto wait_tile = [] {
#ifdef HPC_HAVE_CP_ASYNC
        __pipeline_wait_prior(0);
#endif
        __syncthreads();
    };

    // Prefetch tile 0 into buffer 0.
    load_tile(0, 0);
    wait_tile();

    for (int tileK = 0; tileK < nTilesK; ++tileK) {
        const int cur = tileK & 1;       // current buffer
        const int nxt = 1 - cur;         // next buffer

        // Prefetch next tile while computing current.
        if (tileK + 1 < nTilesK) {
            load_tile(tileK + 1, nxt);
        }

        // Compute outer products from current buffer.
        #pragma unroll
        for (int k = 0; k < kBK; ++k) {
            T reg_A[kTM], reg_B[kTN];
            #pragma unroll
            for (int m = 0; m < kTM; ++m)
                reg_A[m] = As[cur][k][threadRow * kTM + m];
            #pragma unroll
            for (int n = 0; n < kTN; ++n)
                reg_B[n] = Bs[cur][k][threadCol * kTN + n];
            fma_outer(reg_C, reg_A, reg_B);
        }

        // Wait for the next tile to finish loading before swapping.
        if (tileK + 1 < nTilesK)
            wait_tile();
    }

    store_tile(C, reg_C, cRow, cCol, M, N);
}

// ============================================================================
// Level 5: Vectorized loads (float4/double2) + shared-memory XOR swizzle
//
// Same register-tile shape as kernel_reg_tile (128x128 block, BK=16, 8x8 per
// thread), with two changes:
//
//  1. Global -> shared loads use 128-bit vectors (float4 / double2), cutting
//     load instructions by 4x / 2x. B's contiguous dimension (N) is also Bs's,
//     so B is loaded and stored as vectors. A is contiguous along K but As is
//     stored transposed (As[k][m]), so A is loaded as a vector and then
//     scattered with scalar stores into kVecW rows of As.
//
//     Vector loads need 16-byte-aligned addresses. cudaMalloc aligns the base
//     pointers, and every offset here is a multiple of the vector width as
//     long as K and N are too; launch_vectorized checks that and otherwise
//     runs kernel_reg_tile.
//
//  2. Shared memory uses an XOR swizzle instead of +1 padding to spread
//     accesses across banks without a wasted column (padding would also
//     break the alignment the vector stores need). swizzle_slot() permutes
//     which physical vector slot a logical (row, slot) maps to; the same
//     function is used at every write and read, so correctness does not
//     depend on the permutation itself.
// ============================================================================

// 128-bit vector type selector: float4 for float, double2 for double.
template <typename T> struct VecTraits;
template <> struct VecTraits<float>  { using Vec = float4;  static constexpr int kWidth = 4; };
template <> struct VecTraits<double> { using Vec = double2; static constexpr int kWidth = 2; };

// XOR swizzle over vector-slots within one row of a tile. `slots_per_row`
// must be a power of two (true for every instantiation in this file: 32 for
// float's kBM/kBN=128 with kWidth=4, 64 for double's kBM/kBN=128 with
// kWidth=2). Self-inverse: calling this twice with the same (row,
// slots_per_row) undoes itself, since XOR-by-a-constant is its own inverse.
__device__ __forceinline__ int swizzle_slot(int row, int slot, int slots_per_row) {
    return slot ^ (row & (slots_per_row - 1));
}

template <typename T>
__global__ void __launch_bounds__(256)
kernel_vectorized(const T* __restrict__ A,
                  const T* __restrict__ B,
                  T* __restrict__ C,
                  int M, int K, int N) {
    using Vec = typename VecTraits<T>::Vec;
    constexpr int kVecW   = VecTraits<T>::kWidth;
    constexpr int kASlots = kBM / kVecW;  // 32 (f32) / 64 (f64)
    constexpr int kBSlots = kBN / kVecW;  // 32 (f32) / 64 (f64)

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int threadRow = threadIdx.x / (kBN / kTN);
    const int threadCol = threadIdx.x % (kBN / kTN);
    const int cRow = blockRow * kBM + threadRow * kTM;
    const int cCol = blockCol * kBN + threadCol * kTN;

    // No +1 padding here -- swizzle_slot() handles bank conflicts instead,
    // and padding would break the alignment vectorized stores rely on.
    __shared__ alignas(16) T As[kBK][kBM];
    __shared__ alignas(16) T Bs[kBK][kBN];

    T reg_C[kTM][kTN] = {};

    const int nTilesK = (K + kBK - 1) / kBK;

    // A: vectorize along K (A's own contiguous dimension). kAVecsPerCol
    // vector-loads per output column m, each yielding kVecW consecutive
    // k-values that get scattered (scalar stores) into kVecW different rows
    // of the transposed As at the same column m.
    constexpr int kAVecsPerCol = kBK / kVecW;          // 4 (f32) / 8 (f64)
    constexpr int kATotalVecs  = kBM * kAVecsPerCol;   // 512 (f32) / 1024 (f64)
    // B: vectorize along N (B's own contiguous dimension, and Bs's fast
    // dimension too) -- both the load AND the store are vectorized here.
    constexpr int kBTotalVecs  = kBK * kBSlots;        // 512 (f32) / 1024 (f64)

    for (int tileK = 0; tileK < nTilesK; ++tileK) {
        // --- A: vectorized load, scalar scatter-store (transposed) ---
        for (int idx = threadIdx.x; idx < kATotalVecs; idx += blockDim.x) {
            const int m    = idx / kAVecsPerCol;         // 0..kBM-1
            const int kvec = idx % kAVecsPerCol;         // 0..kAVecsPerCol-1
            const int aRow = blockRow * kBM + m;
            const int aColBase = tileK * kBK + kvec * kVecW;

            Vec v{};
            // K % kVecW == 0 (checked by launch_vectorized), so aColBase < K
            // means the whole vector is in bounds.
            if (aRow < M && aColBase < K) {
                v = *reinterpret_cast<const Vec*>(&A[aRow * K + aColBase]);
            }
            const T* velems = reinterpret_cast<const T*>(&v);
            #pragma unroll
            for (int e = 0; e < kVecW; ++e) {
                const int k = kvec * kVecW + e;
                const int logical_slot = m / kVecW;
                const int lane         = m % kVecW;
                const int phys_slot    = swizzle_slot(k, logical_slot, kASlots);
                As[k][phys_slot * kVecW + lane] = velems[e];
            }
        }

        // --- B: vectorized load AND vectorized store ---
        for (int idx = threadIdx.x; idx < kBTotalVecs; idx += blockDim.x) {
            const int k    = idx / kBSlots;              // 0..kBK-1
            const int nvec = idx % kBSlots;               // 0..kBSlots-1 (logical)
            const int bRow = tileK * kBK + k;
            const int bColBase = blockCol * kBN + nvec * kVecW;

            Vec v{};
            // Likewise N % kVecW == 0.
            if (bRow < K && bColBase < N) {
                v = *reinterpret_cast<const Vec*>(&B[bRow * N + bColBase]);
            }
            const int phys_slot = swizzle_slot(k, nvec, kBSlots);
            *reinterpret_cast<Vec*>(&Bs[k][phys_slot * kVecW]) = v;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kBK; ++k) {
            T reg_A[kTM], reg_B[kTN];
            #pragma unroll
            for (int m = 0; m < kTM; ++m) {
                const int col           = threadRow * kTM + m;
                const int logical_slot  = col / kVecW;
                const int lane          = col % kVecW;
                const int phys_slot     = swizzle_slot(k, logical_slot, kASlots);
                reg_A[m] = As[k][phys_slot * kVecW + lane];
            }
            #pragma unroll
            for (int n = 0; n < kTN; ++n) {
                const int col           = threadCol * kTN + n;
                const int logical_slot  = col / kVecW;
                const int lane          = col % kVecW;
                const int phys_slot     = swizzle_slot(k, logical_slot, kBSlots);
                reg_B[n] = Bs[k][phys_slot * kVecW + lane];
            }
            fma_outer(reg_C, reg_A, reg_B);
        }

        __syncthreads();
    }

    store_tile(C, reg_C, cRow, cCol, M, N);
}

// ============================================================================
// Level 4: Tensor Cores via WMMA -- fp32 only, sm_70+
//
// Each warp computes a kWMMA_M x kWMMA_N = 16x16 output tile of C.
// Thread block: 4 warps in X x 4 warps in Y = 16 warps = 512 threads.
// Thread block output: (4*16) x (4*16) = 64x64.
//
// The input matrices are fp32. We convert to fp16 into shared memory,
// run wmma::mma_sync (fp16 x fp16 -> fp32), and accumulate in fp32 fragments.
//
// Key concepts:
//   wmma::fragment  -- opaque per-warp register file holding a matrix tile.
//   wmma::load_matrix_sync  -- cooperative warp load from shared memory.
//   wmma::mma_sync  -- 16x16x16 Tensor Core MMA.
//   wmma::store_matrix_sync -- cooperative warp store to global memory.
//
// The fp16 conversion introduces ~1e-3 relative error vs fp32 GEMM; the
// tests use a relaxed tolerance.
//
// Falls back to gemm_cuda_double_buf on non-WMMA targets.
// ============================================================================

// WMMA warp tile grid inside the thread block.
static constexpr int kWarpM  = 4;   // warps in M dimension
static constexpr int kWarpN  = 4;   // warps in N dimension
// Thread block output: (kWarpM * kWMMA_M) x (kWarpN * kWMMA_N) = 64x64
static constexpr int kBlockM = kWarpM * kWMMA_M;  // 64
static constexpr int kBlockN = kWarpN * kWMMA_N;  // 64
static constexpr int kBlockK = kWMMA_K;            // 16 (k-step)


__global__ void __launch_bounds__(512)
kernel_wmma(const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            int M, int K, int N) {
    // Which 64x64 output tile does this block own?
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Which 16x16 WMMA tile does this warp own within the block?
    const int warpId  = threadIdx.x / 32;
    const int warpRow = warpId / kWarpN;   // 0..3
    const int warpCol = warpId % kWarpN;   // 0..3

    // Global row/col of this warp's C tile.
    const int cWarpRow = blockRow * kBlockM + warpRow * kWMMA_M;
    const int cWarpCol = blockCol * kBlockN + warpCol * kWMMA_N;

    // Shared memory: fp16 sub-tiles for Tensor Core input. No +1 padding:
    // wmma::load_matrix_sync requires the leading dimension of a __half
    // fragment to be a multiple of 8 elements (16 bytes). kBlockM/N = 64
    // already are; 65 would fail with cudaErrorMisalignedAddress.
    __shared__ __half As[kBlockK][kBlockM];  // 16 x 64
    __shared__ __half Bs[kBlockK][kBlockN];  // 16 x 64

    // WMMA fragments for this warp. The major-order tags must match the
    // physical shared-memory layout; a mismatch silently transposes the
    // operand (wrong values, no error).
    //
    // As is stored transposed, As[k][m] = A[m][k]. With col_major,
    // element(row,col) = ptr + row + col*ldm = As[k=col][m=row] = A[row][col],
    // the (row=M, col=K) view mma_sync expects of matrix_a.
    //
    // Bs is stored naturally, Bs[k][n] = B[k][n]. With row_major,
    // element(row,col) = ptr + row*ldm + col = B[row][col], the (row=K,
    // col=N) view mma_sync expects of matrix_b.
    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K, __half,
                   wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K, __half,
                   wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int nTilesK = (K + kBlockK - 1) / kBlockK;

    // Shared-memory load helpers (all threads participate).
    // 512 threads load 16x64 = 1024 As elements (2 each) and
    //                  16x64 = 1024 Bs elements (2 each) per k-tile.
    const int tid = threadIdx.x;

    for (int tileK = 0; tileK < nTilesK; ++tileK) {
        // Load A sub-tile: rows blockRow*64..+64, cols tileK*16..+16.
        // Convert fp32 -> fp16 on the fly.
        for (int idx = tid; idx < kBlockM * kBlockK; idx += blockDim.x) {
            const int aRow = blockRow * kBlockM + (idx / kBlockK);
            const int aCol = tileK * kBlockK + (idx % kBlockK);
            const float val = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.f;
            As[idx % kBlockK][idx / kBlockK] = __float2half(val);
        }
        // Load B sub-tile: rows tileK*16..+16, cols blockCol*64..+64.
        for (int idx = tid; idx < kBlockK * kBlockN; idx += blockDim.x) {
            const int bRow = tileK * kBlockK + (idx / kBlockN);
            const int bCol = blockCol * kBlockN + (idx % kBlockN);
            const float val = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.f;
            Bs[idx / kBlockN][idx % kBlockN] = __float2half(val);
        }

        __syncthreads();

        // Each warp performs its 16x16x16 Tensor Core MMA.
        if (cWarpRow < M && cWarpCol < N) {
            // Pointers to this warp's 16x16 fragment within shared memory.
            // As is stored As[k][m] (transposed relative to A) and Bs is
            // stored Bs[k][n] (B's natural orientation) -- see the a_frag/
            // b_frag declaration comment above for why that means a_frag
            // must be col_major and b_frag row_major. Stride = kBlockM/N
            // (no padding -- see the As/Bs declaration comment above).
            const __half* as_ptr = &As[0][warpRow * kWMMA_M];
            const __half* bs_ptr = &Bs[0][warpCol * kWMMA_N];

            wmma::load_matrix_sync(a_frag, as_ptr, kBlockM);
            wmma::load_matrix_sync(b_frag, bs_ptr, kBlockN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    // Store the accumulated fp32 fragment. store_matrix_sync always writes a
    // full 16x16 tile and needs a 32-byte-aligned destination, so a direct
    // store is only safe for a tile wholly inside C with N % 8 == 0. Any
    // other tile goes through a per-warp shared-memory staging tile and is
    // copied out element by element.
    __shared__ float Cs[kWarpM * kWarpN][kWMMA_M * kWMMA_N];
    if (cWarpRow < M && cWarpCol < N) {
        if (cWarpRow + kWMMA_M <= M && cWarpCol + kWMMA_N <= N && N % 8 == 0) {
            wmma::store_matrix_sync(C + cWarpRow * N + cWarpCol, c_frag, N,
                                    wmma::mem_row_major);
        } else {
            float* tile = Cs[warpId];
            wmma::store_matrix_sync(tile, c_frag, kWMMA_N, wmma::mem_row_major);
            __syncwarp();
            for (int e = threadIdx.x % 32; e < kWMMA_M * kWMMA_N; e += 32) {
                const int r = cWarpRow + e / kWMMA_N, c = cWarpCol + e % kWMMA_N;
                if (r < M && c < N)
                    C[r * N + c] = tile[e];
            }
        }
    }
}

// ============================================================================
// Level 7: Pipelined Tensor Cores via WMMA -- bigger tiles + cp.async
// double buffering.
//
// kernel_wmma (64x64 block tile, single-buffered, one mma_sync per warp per
// k-step) reaches ~5–6 TFLOP/s on RTX 5080; cuBLAS FP16 (compute-only)
// reaches ~120 on the same Tensor Cores. This kernel closes most of that
// gap the way CUTLASS-style kernels do, still on the wmma:: C++ API:
//
//   1. Bigger block tile: 128x128 with BK=32 (vs 64x64, BK=16), so more work
//      per shared-memory round trip and per __syncthreads() pair.
//   2. Bigger warp tile: each of 8 warps owns 32x64 = 2x4 fragments and issues
//      8 mma_sync per k-sub-step. Each b_frag is loaded once and reused for
//      both fm, each a_frag once for all 4 fn — the register blocking of
//      kernel_reg_tile, at fragment granularity.
//   3. Double buffering with cp.async (sm_80+): the next k-tile's copy
//      overlaps the current tile's MMAs, with kernel_double_buf's control
//      flow. Pre-Ampere it falls back to a synchronous float4 copy.
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
//     multiples of 128 and K of 32 (launch_wmma_pipelined falls back to
//     kernel_wmma otherwise): then every chunk starts at an element offset
//     that is a multiple of 8, i.e. 16-byte aligned, and no bounds checks or
//     zfill are needed.
// ============================================================================

static constexpr int kPipeBM = 128;   // thread-block output rows
static constexpr int kPipeBN = 128;   // thread-block output cols
static constexpr int kPipeBK = 32;    // k-step per shared-memory tile (2 WMMA k-steps of 16)
static constexpr int kPipeWarpM = 32; // per-warp output rows (2x kWMMA_M)
static constexpr int kPipeWarpN = 64; // per-warp output cols (4x kWMMA_N)
static constexpr int kPipeWarpRows = kPipeBM / kPipeWarpM;          // 4
static constexpr int kPipeWarpCols = kPipeBN / kPipeWarpN;          // 2
static constexpr int kPipeNumWarps = kPipeWarpRows * kPipeWarpCols; // 8 (256 threads)
static constexpr int kPipeFragM = kPipeWarpM / kWMMA_M;             // 2
static constexpr int kPipeFragN = kPipeWarpN / kWMMA_N;             // 4
static constexpr int kPipeKSteps = kPipeBK / kWMMA_K;               // 2

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
// occupancy limit from 3 to 2 blocks/SM. The kernel's 126 registers/thread
// already cap it at 2, so occupancy is unchanged.
static constexpr int kPipeAsLd = kPipeBK + 8;   // 40 halves =  80 B
static constexpr int kPipeBsLd = kPipeBN + 8;   // 136 halves = 272 B

__global__ void __launch_bounds__(kPipeNumWarps * 32)
kernel_wmma_pipelined(const __half* __restrict__ A16,   // MxK, row-major, fp16
                      const __half* __restrict__ B16,   // KxN, row-major, fp16
                      float* __restrict__ C,
                      int M, int K, int N) {
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int warpId  = threadIdx.x / 32;
    const int warpRow = warpId / kPipeWarpCols;   // 0..3
    const int warpCol = warpId % kPipeWarpCols;   // 0..1
    const int cWarpRow = blockRow * kPipeBM + warpRow * kPipeWarpM;
    const int cWarpCol = blockCol * kPipeBN + warpCol * kPipeWarpN;

    __shared__ alignas(16) __half As[2][kPipeBM][kPipeAsLd];  // natural: As[m][k], padded ld
    __shared__ alignas(16) __half Bs[2][kPipeBK][kPipeBsLd];  // natural: Bs[k][n], padded ld

    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K, float> c_frag[kPipeFragM][kPipeFragN];
    #pragma unroll
    for (int fm = 0; fm < kPipeFragM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < kPipeFragN; ++fn)
            wmma::fill_fragment(c_frag[fm][fn], 0.0f);

    const int nTilesK = K / kPipeBK;  // exact -- see file comment above
    const int tid = threadIdx.x;

    // Each thread copies 16-byte (8-half) chunks. As has kPipeBM*(kPipeBK/8)
    // chunks, Bs has kPipeBK*(kPipeBN/8) chunks -- both equal 512 with the
    // constants above, and kPipeNumWarps*32 = 256 threads, so each thread
    // handles exactly 2 chunks per call (512/256).
    auto load_tile = [&](int tileK, int buf) {
        constexpr int kAChunksPerRow = kPipeBK / 8;
        constexpr int kAChunks = kPipeBM * kAChunksPerRow;
        for (int c = tid; c < kAChunks; c += blockDim.x) {
            const int row  = c / kAChunksPerRow;
            const int kOff = (c % kAChunksPerRow) * 8;
            const __half* src = A16 + static_cast<std::size_t>(blockRow * kPipeBM + row) * K
                                     + (tileK * kPipeBK + kOff);
            __half* dst = &As[buf][row][kOff];
#ifdef HPC_HAVE_CP_ASYNC
            __pipeline_memcpy_async(dst, src, 16);
#else
            *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
#endif
        }
        constexpr int kBChunksPerRow = kPipeBN / 8;
        constexpr int kBChunks = kPipeBK * kBChunksPerRow;
        for (int c = tid; c < kBChunks; c += blockDim.x) {
            const int row  = c / kBChunksPerRow;
            const int nOff = (c % kBChunksPerRow) * 8;
            const __half* src = B16 + static_cast<std::size_t>(tileK * kPipeBK + row) * N
                                     + (blockCol * kPipeBN + nOff);
            __half* dst = &Bs[buf][row][nOff];
#ifdef HPC_HAVE_CP_ASYNC
            __pipeline_memcpy_async(dst, src, 16);
#else
            *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
#endif
        }
#ifdef HPC_HAVE_CP_ASYNC
        __pipeline_commit();
#endif
    };
    auto wait_tile = [] {
#ifdef HPC_HAVE_CP_ASYNC
        __pipeline_wait_prior(0);
#endif
        __syncthreads();
    };

    load_tile(0, 0);
    wait_tile();

    for (int tileK = 0; tileK < nTilesK; ++tileK) {
        const int cur = tileK & 1;
        const int nxt = 1 - cur;

        if (tileK + 1 < nTilesK)
            load_tile(tileK + 1, nxt);

        #pragma unroll
        for (int kSub = 0; kSub < kPipeKSteps; ++kSub) {
            // Load each fn's b_frag once per k-sub-step, reuse across all
            // fm below (register-blocking, same idea as kernel_reg_tile's
            // reg_A/reg_B reuse across its m/n FMA loop).
            wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K, __half, wmma::row_major> b_frag[kPipeFragN];
            #pragma unroll
            for (int fn = 0; fn < kPipeFragN; ++fn)
                wmma::load_matrix_sync(b_frag[fn],
                    &Bs[cur][kSub * kWMMA_K][warpCol * kPipeWarpN + fn * kWMMA_N], kPipeBsLd);

            #pragma unroll
            for (int fm = 0; fm < kPipeFragM; ++fm) {
                wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K, __half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag,
                    &As[cur][warpRow * kPipeWarpM + fm * kWMMA_M][kSub * kWMMA_K], kPipeAsLd);
                #pragma unroll
                for (int fn = 0; fn < kPipeFragN; ++fn)
                    wmma::mma_sync(c_frag[fm][fn], a_frag, b_frag[fn], c_frag[fm][fn]);
            }
        }

        if (tileK + 1 < nTilesK)
            wait_tile();
    }

    #pragma unroll
    for (int fm = 0; fm < kPipeFragM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < kPipeFragN; ++fn)
            wmma::store_matrix_sync(
                C + (cWarpRow + fm * kWMMA_M) * N + (cWarpCol + fn * kWMMA_N),
                c_frag[fm][fn], N, wmma::mem_row_major);
}

// ============================================================================
// Level 6: Raw Tensor Core MMA via mma.sync + ldmatrix
//
// The same computation as kernel_wmma, one level lower. WMMA's
// load_matrix_sync / mma_sync leave the per-lane register mapping to the
// compiler; here ldmatrix.sync loads shared memory straight into the
// registers mma.sync.m16n8k16 expects, and the kernel chooses which address
// each lane supplies and whether the load is transposed. A mistake in that
// mapping produces wrong values, not a build or launch error. The native
// tile is 16x8x16, so each warp issues two MMAs side by side to cover the
// 16x16 area kernel_wmma covers with one.
//
// Operand layouts (the only f16 m16n8k16 variant is .row.col):
//   A `.row` (K contiguous): As is stored [m][k], A's own layout, so no
//     transpose — unlike the FMA kernels, which store As as [k][m].
//   B `.col` (K contiguous): Bs is stored [k][n] (N contiguous), so B's
//     ldmatrix uses `.trans`. `.trans` only changes the register shuffle,
//     not which address each lane supplies.
//
// ldmatrix addressing (PTX ISA, "ldmatrix"): with `.x4`, lane l supplies the
// start address of row l%8 of 8x8 quadrant l/8. `.x2` uses lanes 0-15 only,
// but every lane must pass a valid address, so lanes 16-31 repeat them
// (lane % 16).
//
// mma.sync.m16n8k16.f32 accumulator layout (PTX ISA, "Matrix Fragments for
// mma.m16n8k16"), groupID = lane / 4, threadInGroup = lane % 4:
//   acc[0] -> C[groupID,     threadInGroup*2]
//   acc[1] -> C[groupID,     threadInGroup*2 + 1]
//   acc[2] -> C[groupID + 8, threadInGroup*2]
//   acc[3] -> C[groupID + 8, threadInGroup*2 + 1]
//
// Requires sm_80+ for the f16 m16n8k16 shape; launch_mma_ldmatrix falls back
// to kernel_wmma on sm_70-75.
// ============================================================================

static constexpr int kMmaM = 16;   // mma.sync m16n8k16 native shape
static constexpr int kMmaN = 8;
static constexpr int kMmaK = 16;

// 4x4 warps per block; each warp owns TWO side-by-side 16x8 tiles (16x16
// total) to match kernel_wmma's per-warp output area for a fair comparison.
static constexpr int kMmaWarpM  = 4;
static constexpr int kMmaWarpN  = 4;
static constexpr int kMmaBlockM = kMmaWarpM * kMmaM;         // 64
static constexpr int kMmaBlockN = kMmaWarpN * kMmaN * 2;     // 64
static constexpr int kMmaBlockK = kMmaK;                      // 16

__global__ void __launch_bounds__(512)
kernel_mma_ldmatrix(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    int M, int K, int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int warpId  = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int warpRow = warpId / kMmaWarpN;   // 0..3
    const int warpCol = warpId % kMmaWarpN;   // 0..3

    const int cWarpRow  = blockRow * kMmaBlockM + warpRow * kMmaM;
    const int cWarpCol0 = blockCol * kMmaBlockN + warpCol * (kMmaN * 2);
    const int cWarpCol1 = cWarpCol0 + kMmaN;

    // A: natural row-major layout (K contiguous) -- matches `.row` directly.
    // B: natural row-major layout (N contiguous) -- needs `.trans` below.
    __shared__ __half As[kMmaBlockM][kMmaBlockK];
    __shared__ __half Bs[kMmaBlockK][kMmaBlockN];

    // mma.m16n8k16.f32 output: 4 f32 registers/thread per 16x8 tile.
    float acc0[4] = {0.f, 0.f, 0.f, 0.f};
    float acc1[4] = {0.f, 0.f, 0.f, 0.f};

    const int nTilesK = (K + kMmaBlockK - 1) / kMmaBlockK;
    const int tid = threadIdx.x;

    for (int tileK = 0; tileK < nTilesK; ++tileK) {
        // Shared-memory load (fp32 -> fp16), same strided idiom as kernel_wmma.
        for (int idx = tid; idx < kMmaBlockM * kMmaBlockK; idx += blockDim.x) {
            const int m = idx / kMmaBlockK;
            const int k = idx % kMmaBlockK;
            const int aRow = blockRow * kMmaBlockM + m;
            const int aCol = tileK * kMmaBlockK + k;
            const float val = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.f;
            As[m][k] = __float2half(val);
        }
        for (int idx = tid; idx < kMmaBlockK * kMmaBlockN; idx += blockDim.x) {
            const int k = idx / kMmaBlockN;
            const int n = idx % kMmaBlockN;
            const int bRow = tileK * kMmaBlockK + k;
            const int bCol = blockCol * kMmaBlockN + n;
            const float val = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.f;
            Bs[k][n] = __float2half(val);
        }
        __syncthreads();

        if (cWarpRow < M) {
            // --- A fragment: 16(M)x16(K) tile, 4 quadrants, ldmatrix.x4 (no .trans) ---
            // ldmatrix.x4 loads four 8x8 chunks in lane-group order
            // (chunk = lane/8 -> a_frag[chunk]), and mma.sync.m16n8k16
            // expects them as chunk0=(M0-7,K0-7), chunk1=(M8-15,K0-7),
            // chunk2=(M0-7,K8-15), chunk3=(M8-15,K8-15): the M half varies
            // fastest (quadIdx%2), the K half slowest (quadIdx/2).
            const int quadIdx  = lane / 8;              // 0..3
            const int quadRow  = lane % 8;               // 0..7
            const int aM = warpRow * kMmaM + quadRow + (quadIdx % 2) * 8;
            const int aK = (quadIdx / 2) * 8;
            const __half* a_addr = &As[aM][aK];

            unsigned a_frag[4];
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "l"(__cvta_generic_to_shared(a_addr)));

            // --- B fragment: 16(K)x8(N) tile, 2 quadrants, ldmatrix.x2 + .trans ---
            // Lanes 0-15 address rows K=0..15 (quadrant = lane/8); lanes
            // 16-31 repeat them, since every lane must supply a valid address.
            const int bK = lane % 16;
            for (int which = 0; which < 2; ++which) {
                const int bN = warpCol * (kMmaN * 2) + which * kMmaN;
                const __half* b_addr = &Bs[bK][bN];

                unsigned b_frag[2];
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[0]), "=r"(b_frag[1])
                    : "l"(__cvta_generic_to_shared(b_addr)));

                float* acc = (which == 0) ? acc0 : acc1;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1]));
            }
        }

        __syncthreads();
    }

    if (cWarpRow < M) {
        const int groupID = lane / 4;
        const int tig     = lane % 4;
        auto store_frag = [&](const float* acc, int cCol) {
            const int rows[2] = {groupID, groupID + 8};
            const int cols[2] = {tig * 2, tig * 2 + 1};
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr)
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int gi = cWarpRow + rows[rr];
                    const int gj = cCol + cols[cc];
                    if (gi < M && gj < N)
                        C[gi * N + gj] = acc[rr * 2 + cc];
                }
        };
        store_frag(acc0, cWarpCol0);
        store_frag(acc1, cWarpCol1);
    }
#else
    // sm_75 and below have no f16 m16n8k16 shape. Never launched there
    // (launch_mma_ldmatrix checks cuda_has_ampere()); this branch only keeps
    // the file compiling for older -arch targets.
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)N;
#endif
}

// ----------------------------------------------------------------------------
// fp32 -> fp16 staging kernel: converts a whole matrix in global memory ahead
// of time, unlike kernel_wmma / kernel_mma_ldmatrix, which convert on the fly
// per shared-memory tile. Used by kernel_wmma_pipelined (cp.async is a
// same-dtype byte copy, not a converting load) and by the cuBLAS fp16
// reference.
// ----------------------------------------------------------------------------
__global__ void kernel_f32_to_f16(const float* __restrict__ src,
                                  __half* __restrict__ dst,
                                  int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        dst[idx] = __float2half(src[idx]);
}

// ============================================================================
// Host-side device count query
// ============================================================================

namespace hpc::gemm {

int cuda_device_count() noexcept {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return count;
}

// Returns true if any device has compute capability >= the given (major, minor).
static bool device_has_capability(int major, int minor) noexcept {
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess) return false;
    for (int d = 0; d < devCount; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) == cudaSuccess)
            if (prop.major > major || (prop.major == major && prop.minor >= minor))
                return true;
    }
    return false;
}

bool cuda_has_tensor_cores() noexcept { return device_has_capability(7, 0); }
bool cuda_has_ampere()       noexcept { return device_has_capability(8, 0); }


// ============================================================================
// RAII device buffer
// ============================================================================

template <typename T>
struct DeviceBuffer {
    T*          ptr  = nullptr;
    std::size_t size = 0;
    explicit DeviceBuffer(std::size_t n) : size(n) {
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&)             = delete;
    DeviceBuffer& operator=(const DeviceBuffer&)  = delete;
};

// ============================================================================
// Host launchers -- one per level, on device pointers. Each falls back to the
// level below it when its hardware or shape requirement isn't met.
// ============================================================================

template <typename T>
static void launch_naive(const T* A, const T* B, T* C, int M, int K, int N) {
    const dim3 block(kTile, kTile), grid((N + kTile-1)/kTile, (M + kTile-1)/kTile);
    kernel_naive<T><<<grid, block>>>(A, B, C, M, K, N);
}

template <typename T>
static void launch_blocked(const T* A, const T* B, T* C, int M, int K, int N) {
    const dim3 block(kTile, kTile), grid((N + kTile-1)/kTile, (M + kTile-1)/kTile);
    kernel_blocked<T><<<grid, block>>>(A, B, C, M, K, N);
}

template <typename T>
static void launch_reg_tile(const T* A, const T* B, T* C, int M, int K, int N) {
    const dim3 block(256), grid((N + kBN-1)/kBN, (M + kBM-1)/kBM);
    kernel_reg_tile<T><<<grid, block>>>(A, B, C, M, K, N);
}

template <typename T>
static void launch_double_buf(const T* A, const T* B, T* C, int M, int K, int N) {
    constexpr int LBM = kDBufBM<T>, LBN = kDBufBN<T>;
    // One thread per kTM x kTN sub-tile: 256 threads for float
    // (128x128 tile), 64 for double (64x64 tile -- see kDBufBM).
    const dim3 block((LBM / kTM) * (LBN / kTN)), grid((N + LBN-1)/LBN, (M + LBM-1)/LBM);
    kernel_double_buf<T><<<grid, block>>>(A, B, C, M, K, N);
}

// Vectorized loads require K and N to be multiples of the 128-bit vector
// width (4 for float, 2 for double); otherwise fall back to Level 2.
template <typename T>
static void launch_vectorized(const T* A, const T* B, T* C, int M, int K, int N) {
    constexpr int kVecW = VecTraits<T>::kWidth;
    if (K % kVecW != 0 || N % kVecW != 0)
        return launch_reg_tile(A, B, C, M, K, N);
    const dim3 block(256), grid((N + kBN-1)/kBN, (M + kBM-1)/kBM);
    kernel_vectorized<T><<<grid, block>>>(A, B, C, M, K, N);
}

// Tensor Cores need sm_70+; otherwise fall back to Level 3.
static void launch_wmma(const float* A, const float* B, float* C, int M, int K, int N) {
    if (!cuda_has_tensor_cores())
        return launch_double_buf(A, B, C, M, K, N);
    const dim3 block(kWarpM * kWarpN * 32);  // 4*4*32 = 512 threads
    const dim3 grid((N + kBlockN-1)/kBlockN, (M + kBlockM-1)/kBlockM);
    kernel_wmma<<<grid, block>>>(A, B, C, M, K, N);
}

// The f16 m16n8k16 mma.sync shape needs sm_80+; otherwise fall back to WMMA.
static void launch_mma_ldmatrix(const float* A, const float* B, float* C, int M, int K, int N) {
    if (!cuda_has_ampere())
        return launch_wmma(A, B, C, M, K, N);
    const dim3 block(kMmaWarpM * kMmaWarpN * 32);  // 512 threads
    const dim3 grid((N + kMmaBlockN-1)/kMmaBlockN, (M + kMmaBlockM-1)/kMmaBlockM);
    kernel_mma_ldmatrix<<<grid, block>>>(A, B, C, M, K, N);
}

// fp32 -> fp16 conversion of a device buffer (cp.async, used by Level 7, is a
// same-dtype byte copy and can't convert while loading).
static void convert_f32_to_f16(const float* src, __half* dst, int count) {
    const int threads = 256;
    kernel_f32_to_f16<<<(count + threads - 1) / threads, threads>>>(src, dst, count);
    CUDA_CHECK(cudaGetLastError());
}

// Requires sm_70+ (the cp.async overlap needs sm_80+, see HPC_HAVE_CP_ASYNC's
// #else fallback in kernel_wmma_pipelined), M/N exact multiples of 128 and K
// an exact multiple of 32 (no tail handling); otherwise falls back to WMMA.
static void launch_wmma_pipelined(const float* A, const float* B, float* C, int M, int K, int N) {
    const bool exactTiles = (M % kPipeBM == 0) && (N % kPipeBN == 0) && (K % kPipeBK == 0);
    if (!cuda_has_tensor_cores() || !exactTiles)
        return launch_wmma(A, B, C, M, K, N);
    DeviceBuffer<__half> A16(static_cast<std::size_t>(M) * K);
    DeviceBuffer<__half> B16(static_cast<std::size_t>(K) * N);
    convert_f32_to_f16(A, A16.ptr, M * K);
    convert_f32_to_f16(B, B16.ptr, K * N);
    const dim3 block(kPipeNumWarps * 32), grid(N / kPipeBN, M / kPipeBM);
    kernel_wmma_pipelined<<<grid, block>>>(A16.ptr, B16.ptr, C, M, K, N);
}

// Copy A and B to the device, run `launch` on device pointers, copy C back.
// Every Matrix<T>-based entry point goes through this, so every one of them
// is timed the same way: cudaMalloc + H2D + compute + D2H on each call.
template <typename T, typename Launch>
static void run_on_device(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, Launch launch) {
    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());
    const std::size_t a_n = static_cast<std::size_t>(M) * K;
    const std::size_t b_n = static_cast<std::size_t>(K) * N;
    const std::size_t c_n = static_cast<std::size_t>(M) * N;

    DeviceBuffer<T> dA(a_n), dB(b_n), dC(c_n);
    CUDA_CHECK(cudaMemcpy(dA.ptr, A.data(), a_n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, B.data(), b_n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC.ptr, 0, c_n * sizeof(T)));

    launch(dA.ptr, dB.ptr, dC.ptr, M, K, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC.ptr, c_n * sizeof(T), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Public API
// ============================================================================

template <typename T>
void gemm_cuda_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, launch_naive<T>);
}
template <typename T>
void gemm_cuda_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, launch_blocked<T>);
}
template <typename T>
void gemm_cuda_reg_tile(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, launch_reg_tile<T>);
}
template <typename T>
void gemm_cuda_double_buf(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, launch_double_buf<T>);
}
template <typename T>
void gemm_cuda_vectorized(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, launch_vectorized<T>);
}
// The Tensor Core levels are float-only: fp32 in/out, fp16 compute.
void gemm_cuda_wmma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, launch_wmma);
}
void gemm_cuda_mma_ldmatrix(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, launch_mma_ldmatrix);
}
void gemm_cuda_wmma_pipelined(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, launch_wmma_pipelined);
}

// Compute-only entry point (see cuda.hpp): takes fp16 device buffers, so the
// timing excludes conversion and transfers. No fallback — the caller must
// pass M, N multiples of 128 and K a multiple of 32.
void gemm_cuda_wmma_pipelined_device(const void* dA16, const void* dB16, float* dC,
                                     int M, int K, int N) {
    const dim3 block(kPipeNumWarps * 32), grid(N / kPipeBN, M / kPipeBM);
    kernel_wmma_pipelined<<<grid, block>>>(
        static_cast<const __half*>(dA16), static_cast<const __half*>(dB16), dC, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Reference -- cuBLAS (vendor-tuned upper bound, not part of the hand-
// written kernel ladder above)
//
// NVIDIA's production GEMM, used as the realistic ceiling the hand-written
// kernels are measured against.
//
// gemm_cuda_cublas<T>      -- plain SGEMM/DGEMM, cuBLAS's own tuned SIMT
//                              kernel. Ceiling for the FMA-based kernels
//                              (naive/blocked/reg_tile/double_buf/vectorized).
// gemm_cuda_cublas_tf32    -- fp32 in/out, TF32 Tensor Core compute
//                              (10-bit mantissa, same precision class as
//                              the fp16 kernels above). Requires sm_80+; on
//                              older hardware cuBLAS falls back to plain fp32.
// gemm_cuda_cublas_fp16    -- fp16 in, fp32 accumulate. Ceiling for the
//                              Tensor Core kernels (wmma/mma_ldmatrix/
//                              wmma_pipelined).
//
// hpc::Matrix is row-major; cuBLAS is column-major. Row-major C = A*B is
// exactly column-major C^T = B^T*A^T over the same memory, so every call
// below swaps A<->B (and M<->N) and asks for a plain no-transpose GEMM --
// no data movement, no transpose flags.
// ============================================================================

// Lazily-created, process-lifetime handle. Not thread-safe, matching this
// file's single-threaded benchmark/test usage (every other kernel here is
// launched the same way, with no concurrent-stream support).
static cublasHandle_t cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasCreate failed");
        return h;
    }();
    return handle;
}

static void check_cublas(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed, status=" +
                                 std::to_string(static_cast<int>(st)));
}

// C = A*B via cublasGemmEx with the given storage and compute types.
static void cublas_gemm_ex(const void* A, const void* B, void* C, cudaDataType in_type,
                           cudaDataType out_type, cublasComputeType_t compute,
                           int M, int K, int N, const char* what) {
    const float alpha = 1.0f, beta = 0.0f;
    check_cublas(cublasGemmEx(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                              B, in_type, N, A, in_type, K, &beta, C, out_type, N,
                              compute, CUBLAS_GEMM_DEFAULT),
                 what);
}

template <typename T>
static void cublas_gemm(const T* A, const T* B, T* C, int M, int K, int N) {
    const T alpha = T{1}, beta = T{0};
    if constexpr (std::is_same_v<T, float>)
        check_cublas(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, B, N, A, K, &beta, C, N), "cublasSgemm");
    else
        check_cublas(cublasDgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, B, N, A, K, &beta, C, N), "cublasDgemm");
}

template <typename T>
void gemm_cuda_cublas(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    run_on_device(A, B, C, cublas_gemm<T>);
}

void gemm_cuda_cublas_tf32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, gemm_cuda_cublas_tf32_device);
}

// Matrix<float>-based wrapper (converts internally) -- for correctness
// testing; the compute-only benchmark uses the raw-pointer entry points below
// directly, converting once outside the timed region.
void gemm_cuda_cublas_fp16(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, [](const float* dA, const float* dB, float* dC, int M, int K, int N) {
        DeviceBuffer<__half> A16(static_cast<std::size_t>(M) * K);
        DeviceBuffer<__half> B16(static_cast<std::size_t>(K) * N);
        convert_f32_to_f16(dA, A16.ptr, M * K);
        convert_f32_to_f16(dB, B16.ptr, K * N);
        gemm_cuda_cublas_fp16_device(A16.ptr, B16.ptr, dC, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());  // before A16/B16 are freed
    });
}

// ============================================================================
// Reference -- cuBLAS, compute-only entry points (see cuda.hpp). Same calls as
// the Matrix-based wrappers above, on caller-provided device buffers; the
// wrappers' tests cover their correctness.
// ============================================================================
void gemm_cuda_cublas_device_f32(const float* dA, const float* dB, float* dC,
                                 int M, int K, int N) {
    cublas_gemm(dA, dB, dC, M, K, N);
}

void gemm_cuda_cublas_tf32_device(const float* dA, const float* dB, float* dC,
                                  int M, int K, int N) {
    cublas_gemm_ex(dA, dB, dC, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, M, K, N,
                   "cublasGemmEx (TF32)");
}

// fp16-in/fp32-accumulate. Dense FP16 runs at roughly 2x TF32 throughput
// (half the bits per element through the same tensor pipe): ~120 vs ~60
// TFLOP/s compute-only on RTX 5080. fp16 buffers are void* in the public API
// because cuda.hpp must compile without <cuda_fp16.h> (CPU-only stub builds).
void gemm_cuda_convert_f32_to_f16_device(const float* src, void* dst, int count) {
    convert_f32_to_f16(src, static_cast<__half*>(dst), count);
}

void gemm_cuda_cublas_fp16_device(const void* dA16, const void* dB16, float* dC,
                                  int M, int K, int N) {
    cublas_gemm_ex(dA16, dB16, dC, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, M, K, N,
                   "cublasGemmEx (FP16)");
}

// ============================================================================
// Device-memory helpers for the compute-only benchmarks (see cuda.hpp).
// ============================================================================
void* gemm_cuda_malloc(std::size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}
void gemm_cuda_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}
void gemm_cuda_memcpy_h2d(void* dst, const void* src, std::size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
void gemm_cuda_device_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Explicit instantiations
// ============================================================================

template void gemm_cuda_naive<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_naive<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_blocked<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_blocked<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_reg_tile<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_reg_tile<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_double_buf<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_double_buf<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_vectorized<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_vectorized<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_cublas<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_cublas<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);

}  // namespace hpc::gemm

