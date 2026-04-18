/**
 * @file gemm_kernels.cu
 * @brief CUDA GEMM kernel implementations -- progressive optimization ladder.
 *
 * Level 0: kernel_naive        -- global memory only, 1 thread -> 1 C(i,j)
 * Level 1: kernel_blocked      -- TILE=16 shared-memory tiling
 * Level 2: kernel_reg_tile     -- 128x128 thread block, each thread owns 8x8 C tile
 * Level 3: kernel_double_buf   -- Level 2 + double buffering (cp.async on Ampere+)
 * Level 4: kernel_wmma         -- Tensor Cores via WMMA (f16->f32, f32 only)
 *
 * ============================================================
 *  GPU memory hierarchy reminder
 * ============================================================
 *
 *  Global memory   : off-chip DRAM, ~400-600 GB/s (A100), ~1 TB/s (H100).
 *                    High latency (~200-800 cycles).
 *
 *  Shared memory   : on-chip SRAM, ~10-20 TB/s, ~4-32 cycles.
 *                    32 banks (4-byte interleaved) -- parallel access required.
 *
 *  Registers       : per-thread, ~1 cycle.  Spill to local memory if overused.
 *
 *  Tensor Cores    : dedicated MMA units (Volta+).
 *                    Operate on 16x16x16 matrix fragments.
 *                    ~8x throughput vs SIMT FP32 on the same SM.
 *
 * ============================================================
 *  Level 2 -- Register tile: why each thread should own many outputs
 * ============================================================
 *
 *  In kernel_blocked (Level 1), each thread owns 1 output element.
 *  Per k-step: 2 __syncthreads + 16 shared loads + 16 FMAs = low ratio.
 *
 *  In kernel_reg_tile (Level 2), each thread owns TMxTN = 8x8 = 64 outputs.
 *  Thread block: BMxBN = 128x128 outputs, BK=16 k-step.
 *  Threads per block: (BM/TM) x (BN/TN) = 16 x 16 = 256.
 *
 *  Per k-step:
 *    - Load BMxBK = 128x16 A sub-tile into shared memory
 *    - Load BKxBN = 16x128 B sub-tile into shared memory
 *    - Each thread: TMxTN outer product = 8x8 = 64 FMAs from registers
 *    - 2 __syncthreads + ~2x128x16/256 = 16 global loads per thread + 64 FMAs
 *
 *  Arithmetic intensity = (2 x 128 x 128 x K) / ((128xK + 128xK) x 4B)
 *                       ~= 128/4 = 32 FLOP/byte   (vs ~2 for Level 1)
 *
 * ============================================================
 *  Level 3 -- Double buffering: hiding __syncthreads latency
 * ============================================================
 *
 *  __syncthreads creates a global barrier -- all threads idle while the
 *  next tile loads.  Double buffering uses two ping-pong shared buffers:
 *    - While computing tile k from buffer A, prefetch tile k+1 into buffer B.
 *    - Swap buffers, repeat.
 *
 *  On Ampere+ (SM80+) `__pipeline_memcpy_async` / cp.async moves data
 *  from global to shared memory asynchronously -- completely hidden behind
 *  compute.  Falls back to synchronous load on older GPUs.
 *
 * ============================================================
 *  Level 4 -- Tensor Cores (WMMA)
 * ============================================================
 *
 *  NVIDIA Tensor Cores (Volta+, SM70+) perform a 16x16x16 matrix-multiply
 *  in a single warp-synchronous instruction:
 *    D[16x16] += A[16x16] * B[16x16]
 *
 *  WMMA (Warp Matrix Multiply Accumulate) API fragments the tile across
 *  all 32 threads in a warp using an opaque layout.
 *
 *  This kernel:
 *    - Loads A and B sub-tiles as fp16 into shared memory (even when host
 *      matrices are fp32 -- we convert on the fly)
 *    - Uses wmma::mma_sync to run the 16x16x16 Tensor Core MMA
 *    - Accumulates into fp32 wmma fragment
 *    - Stores result back to host fp32 matrix
 *
 *  Available only when __CUDA_ARCH__ >= 700 (Volta+).
 *  Falls back to gemm_cuda_double_buf on older GPUs / non-WMMA builds.
 */

#include "gemm/cuda.hpp"
#include "hpc/matrix.hpp"

#include <cuda_runtime.h>

// WMMA requires sm_70+; guard so it compiles on all CUDA installations
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  #include <mma.h>
  using namespace nvcuda;
  #define HPC_HAVE_WMMA 1
#endif

// cp.async requires sm_80+ (Ampere)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  #include <cuda_pipeline_primitives.h>
  #define HPC_HAVE_CP_ASYNC 1
#endif

#include <cstddef>
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
// Kernel 1: Naive
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
// Kernel 2: Reordered (CPU naming symmetry -- same as naive on GPU)
// ============================================================================
template <typename T>
__global__ void kernel_reordered(const T* __restrict__ A,
                                  const T* __restrict__ B,
                                  T* __restrict__ C,
                                  int M, int K, int N) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= N) return;
    T acc = T{0};
    const T* a_row = A + i * K;
    for (int k = 0; k < K; ++k)
        acc += a_row[k] * B[k * N + j];
    C[i * N + j] = acc;
}

// ============================================================================
// Kernel 3: Blocked / Tiled  (TILE=16 shared-memory)
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
// Kernel 4: Register-tiled (Level 2)
//
// Thread block: kBMxkBN = 128x128 outputs
// Threads/block: (kBM/kTM) x (kBN/kTN) = 16 x 16 = 256
// Each thread owns a kTMxkTN = 8x8 register tile of C.
//
// Shared memory layout:
//   As[kBK][kBM] = 16 x 128 -- A sub-tile transposed for column access
//   Bs[kBK][kBN] = 16 x 128 -- B sub-tile in row-major
//
// Inner loop: outer product of As column and Bs row -> 8x8 FMAs per k step.
// Bank conflict avoidance: +1 padding on the inner dimension.
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

    // Registers to cache A and B columns/rows during the inner loop.
    T reg_A[kTM] = {};
    T reg_B[kTN] = {};

    // Thread's responsibility for loading shared memory.
    // 256 threads load 128*16 = 2048 elements of As (8 each).
    // 256 threads load 16*128 = 2048 elements of Bs (8 each).
    const int loadARow = threadIdx.x / kBM;   // 0..15  (kBK dimension)
    const int loadACol = threadIdx.x % kBM;   // 0..127 (kBM dimension)
    const int loadBRow = threadIdx.x / kBN;   // 0..15  (kBK dimension)
    const int loadBCol = threadIdx.x % kBN;   // 0..127 (kBN dimension)

    const int nTilesK = (K + kBK - 1) / kBK;

    for (int tileK = 0; tileK < nTilesK; ++tileK) {
        // Load A sub-tile into As[kBK][kBM] (transposed for column-major access).
        // A[blockRow*kBM + loadACol][tileK*kBK + loadARow]
        const int aRow = blockRow * kBM + loadACol;
        const int aCol = tileK * kBK + loadARow;
        As[loadARow][loadACol] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : T{0};

        // Load B sub-tile into Bs[kBK][kBN].
        // B[tileK*kBK + loadBRow][blockCol*kBN + loadBCol]
        const int bRow = tileK * kBK + loadBRow;
        const int bCol = blockCol * kBN + loadBCol;
        Bs[loadBRow][loadBCol] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : T{0};

        __syncthreads();

        // Inner loop: walk the kBK dimension, accumulate outer products.
        #pragma unroll
        for (int k = 0; k < kBK; ++k) {
            // Load A column k (kTM elements for this thread's rows).
            #pragma unroll
            for (int m = 0; m < kTM; ++m)
                reg_A[m] = As[k][threadRow * kTM + m];
            // Load B row k (kTN elements for this thread's cols).
            #pragma unroll
            for (int n = 0; n < kTN; ++n)
                reg_B[n] = Bs[k][threadCol * kTN + n];
            // Outer product -> accumulate into register tile.
            #pragma unroll
            for (int m = 0; m < kTM; ++m)
                #pragma unroll
                for (int n = 0; n < kTN; ++n)
                    reg_C[m][n] += reg_A[m] * reg_B[n];
        }

        __syncthreads();
    }

    // Write register tile back to global memory C.
    #pragma unroll
    for (int m = 0; m < kTM; ++m) {
        #pragma unroll
        for (int n = 0; n < kTN; ++n) {
            const int gi = cRow + m;
            const int gj = cCol + n;
            if (gi < M && gj < N)
                C[gi * N + gj] = reg_C[m][n];
        }
    }
}

// ============================================================================
// Kernel 5: Double-buffered register tile (Level 3)
//
// Same register-tiling as Level 2, but uses two ping-pong shared-memory
// buffers to overlap loading of tile k+1 with computation of tile k.
//
// On Ampere+ (sm_80+):
//   Uses __pipeline_memcpy_async / __pipeline_commit / __pipeline_wait_prior
//   for truly asynchronous global->shared DMA (cp.async instruction).
//   This hides memory latency completely behind FMA execution.
//
// On older GPUs (sm_70..79):
//   Falls back to synchronous loads with __syncthreads barriers.
//   The double-buffer structure is preserved for code clarity, but the
//   overlap benefit requires the hardware async copy support.
// ============================================================================
template <typename T>
__global__ void __launch_bounds__(256)
kernel_double_buf(const T* __restrict__ A,
                  const T* __restrict__ B,
                  T* __restrict__ C,
                  int M, int K, int N) {
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int threadRow = threadIdx.x / (kBN / kTN);
    const int threadCol = threadIdx.x % (kBN / kTN);
    const int cRow = blockRow * kBM + threadRow * kTM;
    const int cCol = blockCol * kBN + threadCol * kTN;

    // Double-buffered shared memory: index 0 and 1 alternate.
    __shared__ T As[2][kBK][kBM + 1];
    __shared__ T Bs[2][kBK][kBN + 1];

    T reg_C[kTM][kTN] = {};
    T reg_A[kTM] = {};
    T reg_B[kTN] = {};

    const int loadARow = threadIdx.x / kBM;
    const int loadACol = threadIdx.x % kBM;
    const int loadBRow = threadIdx.x / kBN;
    const int loadBCol = threadIdx.x % kBN;

    const int nTilesK = (K + kBK - 1) / kBK;

    // -------------------------------------------------------------------
    // Helper lambda: load tile tileK into shared-memory buffer buf.
    // On Ampere+: issues async copy and does NOT synchronise.
    // On older:   copies synchronously and issues __syncthreads.
    // -------------------------------------------------------------------
    auto load_tile = [&](int tileK, int buf) {
        const int aRow = blockRow * kBM + loadACol;
        const int aCol = tileK * kBK + loadARow;
        const T a_val  = (aRow < M && aCol < K) ? A[aRow * K + aCol] : T{0};

        const int bRow = tileK * kBK + loadBRow;
        const int bCol = blockCol * kBN + loadBCol;
        const T b_val  = (bRow < K && bCol < N) ? B[bRow * N + bCol] : T{0};

#ifdef HPC_HAVE_CP_ASYNC
        // Async copy: write directly to shared memory without occupying
        // registers or stalling the warp.
        __pipeline_memcpy_async(&As[buf][loadARow][loadACol], &a_val, sizeof(T));
        __pipeline_memcpy_async(&Bs[buf][loadBRow][loadBCol], &b_val, sizeof(T));
        __pipeline_commit();
#else
        As[buf][loadARow][loadACol] = a_val;
        Bs[buf][loadBRow][loadBCol] = b_val;
#endif
    };

    auto wait_tile = []([[maybe_unused]] int n_ahead) {
#ifdef HPC_HAVE_CP_ASYNC
        __pipeline_wait_prior(n_ahead);
#else
        __syncthreads();
#endif
    };

    // Prefetch tile 0 into buffer 0.
    load_tile(0, 0);
    wait_tile(0);
    __syncthreads();

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
            #pragma unroll
            for (int m = 0; m < kTM; ++m)
                reg_A[m] = As[cur][k][threadRow * kTM + m];
            #pragma unroll
            for (int n = 0; n < kTN; ++n)
                reg_B[n] = Bs[cur][k][threadCol * kTN + n];
            #pragma unroll
            for (int m = 0; m < kTM; ++m)
                #pragma unroll
                for (int n = 0; n < kTN; ++n)
                    reg_C[m][n] += reg_A[m] * reg_B[n];
        }

        // Wait for the next tile to finish loading before swapping.
        if (tileK + 1 < nTilesK) {
            wait_tile(0);
            __syncthreads();
        }
    }

    // Store register tile.
    #pragma unroll
    for (int m = 0; m < kTM; ++m)
        #pragma unroll
        for (int n = 0; n < kTN; ++n) {
            const int gi = cRow + m, gj = cCol + n;
            if (gi < M && gj < N)
                C[gi * N + gj] = reg_C[m][n];
        }
}

// ============================================================================
// Kernel 6: Tensor Cores via WMMA (Level 4) -- fp32 only, sm_70+
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
// The fp16 conversion introduces ~1e-3 relative error vs fp32 GEMM --
// acceptable for training but not exact.  The test uses a relaxed tolerance.
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700

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

    // Shared memory: store fp16 sub-tiles for Tensor Core input.
    // +1 padding avoids bank conflicts on 16-wide warp access.
    __shared__ __half As[kBlockK][kBlockM + 1];  // 16 x 65
    __shared__ __half Bs[kBlockK][kBlockN + 1];  // 16 x 65

    // WMMA fragments for this warp.
    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K, __half,
                   wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K, __half,
                   wmma::col_major> b_frag;
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
            // As is stored as As[k][m]: row = warpRow*kWMMA_M, col = k
            // We need As[k][warpRow*16 .. warpRow*16+15] as a row-major 16x16.
            // The layout is: As[k_row][m_col], so for wmma::row_major we need
            // A[m][k] -- i.e., As^T. Stride = kBlockM+1.
            const __half* as_ptr = &As[0][warpRow * kWMMA_M];
            const __half* bs_ptr = &Bs[0][warpCol * kWMMA_N];

            wmma::load_matrix_sync(a_frag, as_ptr, kBlockM + 1);
            wmma::load_matrix_sync(b_frag, bs_ptr, kBlockN + 1);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    // Store the accumulated fp32 fragment back to global memory.
    if (cWarpRow < M && cWarpCol < N) {
        wmma::store_matrix_sync(C + cWarpRow * N + cWarpCol, c_frag, N,
                                wmma::mem_row_major);
    }
}

#endif  // __CUDA_ARCH__ >= 700

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
// Generic host launcher
// ============================================================================

enum class GemmKind { Naive, Reordered, Blocked, RegTile, DoubleBuf, Wmma };

template <typename T>
static void launch(GemmKind kind, const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    DeviceBuffer<T> dA(M * K), dB(K * N), dC(M * N);
    CUDA_CHECK(cudaMemcpy(dA.ptr, A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC.ptr, 0, M * N * sizeof(T)));

    if (kind == GemmKind::Naive) {
        const dim3 block(kTile, kTile);
        const dim3 grid((N + kTile-1)/kTile, (M + kTile-1)/kTile);
        kernel_naive<T><<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);

    } else if (kind == GemmKind::Reordered) {
        const dim3 block(kTile, kTile);
        const dim3 grid((N + kTile-1)/kTile, (M + kTile-1)/kTile);
        kernel_reordered<T><<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);

    } else if (kind == GemmKind::Blocked) {
        const dim3 block(kTile, kTile);
        const dim3 grid((N + kTile-1)/kTile, (M + kTile-1)/kTile);
        kernel_blocked<T><<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);

    } else if (kind == GemmKind::RegTile) {
        // 256 threads/block, grid sized in units of kBM x kBN.
        const dim3 block(256);
        const dim3 grid((N + kBN-1)/kBN, (M + kBM-1)/kBM);
        kernel_reg_tile<T><<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);

    } else if (kind == GemmKind::DoubleBuf) {
        const dim3 block(256);
        const dim3 grid((N + kBN-1)/kBN, (M + kBM-1)/kBM);
        kernel_double_buf<T><<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);

    } else if (kind == GemmKind::Wmma) {
        // WMMA is fp32-only in this implementation (converts to fp16 internally).
        static_assert(std::is_same_v<T, float>,
                      "gemm_cuda_wmma is only supported for float");
        const dim3 block(kWarpM * kWarpN * 32);  // 4*4*32 = 512 threads
        const dim3 grid((N + kBlockN-1)/kBlockN, (M + kBlockM-1)/kBlockM);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        kernel_wmma<<<grid, block>>>(
            reinterpret_cast<const float*>(dA.ptr),
            reinterpret_cast<const float*>(dB.ptr),
            reinterpret_cast<float*>(dC.ptr), M, K, N);
#else
        // Host side: if Tensor Cores are available at runtime, run WMMA kernel.
        // The kernel was compiled for the target arch by nvcc; if sm_70+ is
        // the target, it is available. We check at runtime for safety.
        if (cuda_has_tensor_cores()) {
            kernel_wmma<<<grid, block>>>(
                reinterpret_cast<const float*>(dA.ptr),
                reinterpret_cast<const float*>(dB.ptr),
                reinterpret_cast<float*>(dC.ptr), M, K, N);
        } else {
            // Fallback: use double-buf kernel (always correct).
            const dim3 block2(256);
            const dim3 grid2((N + kBN-1)/kBN, (M + kBM-1)/kBM);
            kernel_double_buf<T><<<grid2, block2>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);
        }
#endif
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC.ptr, M * N * sizeof(T), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Public API
// ============================================================================

template <typename T>
void gemm_cuda_naive(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    launch<T>(GemmKind::Naive, A, B, C);
}
template <typename T>
void gemm_cuda_reordered(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    launch<T>(GemmKind::Reordered, A, B, C);
}
template <typename T>
void gemm_cuda_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    launch<T>(GemmKind::Blocked, A, B, C);
}
template <typename T>
void gemm_cuda_reg_tile(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    launch<T>(GemmKind::RegTile, A, B, C);
}
template <typename T>
void gemm_cuda_double_buf(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    launch<T>(GemmKind::DoubleBuf, A, B, C);
}
// WMMA is float-only.
void gemm_cuda_wmma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    launch<float>(GemmKind::Wmma, A, B, C);
}

// ============================================================================
// Explicit instantiations
// ============================================================================

template void gemm_cuda_naive<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_naive<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_reordered<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_reordered<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_blocked<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_blocked<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_reg_tile<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_reg_tile<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_cuda_double_buf<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_double_buf<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);

}  // namespace hpc::gemm

