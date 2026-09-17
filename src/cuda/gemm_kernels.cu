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
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// cuBLAS -- backs the vendor-tuned reference kernels (gemm_cuda_cublas /
// gemm_cuda_cublas_tf32) near the bottom of this file. Unlike every kernel
// above, these call into NVIDIA's own production GEMM implementation
// instead of hand-written PTX/intrinsics, to answer "what does this GPU
// actually achieve at its realistic peak" as a ceiling for the hand-written
// kernels to be measured against.
#include <cublas_v2.h>

// Driver API -- needed only for the Hopper TMA descriptor
// (cuTensorMapEncodeTiled has no CUDA-runtime-API equivalent). Always
// included (it is a plain host header, part of every CUDA toolkit
// installation); the functions it declares are only ever CALLED when
// cuda_has_hopper() is true at runtime. Requires linking CUDA::cuda_driver
// (see CMakeLists.txt) in addition to the usual CUDA::cudart.
//
// NOTE: CUtensorMap, cuTensorMapEncodeTiled, and __grid_constant__ (used by
// kernel_hopper_wgmma further down) are CUDA 12.0+ additions. Verified
// building and running against CUDA 13.2 on real hardware (RTX 5080,
// Blackwell sm_120) -- building against CUDA <12.0 would fail to compile
// this translation unit at all, not just this one kernel. That same
// verification run is what found and fixed the real bugs described
// throughout this file (search "found running on real hardware"); the
// Hopper-specific kernel_hopper_wgmma itself remains genuinely unverified
// since no Hopper (sm_90a) hardware has been available to test it on --
// see its much larger "UNVERIFIED" caveat below.
#include <cuda.h>

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
//
// CORRECTNESS FIX (see git history): the shared-memory load previously used
// `threadIdx.x / kBM` / `threadIdx.x % kBM` directly as the (row, col) index
// into the kBK x kBM tile. With 256 threads and kBM=128, that expression can
// only ever produce row in {0, 1} -- rows 2..15 of As/Bs were silently left
// uninitialized before being read by every k-iteration below. Fixed with a
// strided loop (256 threads x 8 iterations = 2048 elements), matching the
// already-correct pattern used by kernel_wmma further down this file.
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
    // 256 threads load 128*16 = 2048 elements of As (8 each) and
    // 16*128 = 2048 elements of Bs (8 each), via a strided loop --
    // NOT a single direct index (256 threads cannot cover 2048 elements
    // one-to-one; see kAElems/kBElems below).
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
// For double, halve BM to stay within 48 KB shared memory limit.
// f32: 2 * 16 * (128+1+128+1) * 4 = 33,024 B ✓
// f64: 2 * 16 * (128+1+128+1) * 8 = 66,048 B ✗  →  use BM=64:
//      2 * 16 * (64+1+128+1) * 8   = 49,664 B ✗  →  use BM=64,BN=64:
//      2 * 16 * (64+1+64+1) * 8    = 33,280 B ✓
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
    T reg_A[kTM] = {};
    T reg_B[kTN] = {};

    // CORRECTNESS FIX (see git history / kernel_reg_tile above): a single
    // `threadIdx.x / LBM` cannot enumerate all kBK=16 rows with only 256
    // threads and LBM<=128 -- the original code left most of As/Bs
    // uninitialized. Use a strided loop instead, same as kernel_reg_tile.
    constexpr int kAElems = kBK * LBM;
    constexpr int kBElems = kBK * LBN;

    const int nTilesK = (K + kBK - 1) / kBK;

    // -------------------------------------------------------------------
    // Helper lambda: load tile tileK into shared-memory buffer buf.
    // On Ampere+: issues async copy and does NOT synchronise.
    // On older:   copies synchronously and issues __syncthreads.
    //
    // CORRECTNESS FIX (found running on real Ampere-class hardware for the
    // first time -- RTX 5080/Blackwell, sm_120 -- see cuda_has_hopper()
    // above for a different instance of the same "never verified on real
    // hardware" class of bug): the previous version read the boundary-
    // checked element into a local `a_val`/`b_val` register and passed
    // `&a_val` as __pipeline_memcpy_async's source. cp.async is a
    // global-memory-to-shared-memory instruction ONLY -- the source
    // address must resolve to the global address space, and a local
    // (register/stack) variable's address does not. Every call aborted at
    // runtime with cudaErrorNotSupported ("operation not supported on
    // global/shared address space"), which then poisoned the CUDA context
    // for the rest of the process (every subsequent CUDA call, including
    // unrelated cudaMalloc calls in later tests, failed with the same
    // sticky error). Fixed by passing the real global pointer and using
    // the zfill overload (copy `sizeof(T) - zfill` bytes from src, zero
    // the rest) so out-of-bounds elements still zero-fill shared memory
    // without ever dereferencing out-of-range global memory -- the dummy
    // in-bounds address substituted for the OOB case is never read.
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
// Kernel 5b: Vectorized loads (float4/double2) + shared-memory XOR swizzle
// (Level 5)
//
// Same register-tile shape as kernel_reg_tile (kBM x kBN = 128x128,
// kBK=16, kTM x kTN = 8x8 per thread), with two changes:
//
//  1. Global -> shared loads use 128-bit vector instructions (float4 for
//     float, double2 for double) instead of one scalar per thread per
//     element, cutting the instruction count for the load phase by 4x/2x.
//
//     - B's fast (contiguous) dimension in global memory is N, which is
//       ALSO Bs's fast dimension in shared memory -- so B's load is a
//       straight vectorized load *and* a vectorized store.
//     - A's fast (contiguous) dimension in global memory is K, but As is
//       stored TRANSPOSED (As[k][m], to give the compute loop column
//       access) -- so A's load is a vectorized LOAD (4/2 consecutive K
//       values for one fixed row) followed by a SCALAR scatter-store (each
//       of those K values lands in a different As row, same column).
//
//     Vectorized loads require the source address to be 16-byte aligned.
//     cudaMalloc guarantees the base pointer is (well) aligned, and every
//     offset used here (`aColBase`, `bColBase`) is constructed to be a
//     multiple of the vector width -- but only if K (for A) and N (for B)
//     are ALSO multiples of the vector width. The host-side launcher
//     therefore only dispatches to this kernel when that holds; otherwise
//     it falls back to kernel_reg_tile (see `launch()` below).
//
//  2. Shared memory uses an XOR "swizzle" instead of the +1-padding trick
//     used everywhere else in this file, to spread accesses across banks
//     without wasting a column. `swizzle_slot(row, slot, slots_per_row)`
//     permutes which physical vector-slot a logical (row, slot) pair maps
//     to. CORRECTNESS DOES NOT DEPEND ON THIS BEING BANK-CONFLICT-FREE: the
//     exact same function is called at every write site (A's scalar
//     scatter-store, B's vectorized store) and every read site (the k-loop
//     below), so whatever permutation it computes is applied and undone
//     consistently. Only the *performance* claim (fewer bank conflicts than
//     padding) has not been checked with a profiler (e.g. Nsight Compute)
//     against the padding alternative -- the *result* is correct
//     regardless (confirmed by CudaVectorizedFloat/Double's GTest cases on
//     real hardware), which is why this technique is safe to include even
//     without a profiler run to validate the perf benefit specifically.
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
    T reg_A[kTM] = {};
    T reg_B[kTN] = {};

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
            // K % kVecW == 0 is guaranteed by the host-side dispatch check
            // (see launch() below), so `aColBase < K` alone is sufficient
            // to guarantee the whole vector [aColBase, aColBase+kVecW) is
            // in-bounds -- see the file's kernel_vectorized dispatch note.
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
            // N % kVecW == 0 is likewise guaranteed by the host dispatch.
            if (bRow < K && bColBase < N) {
                v = *reinterpret_cast<const Vec*>(&B[bRow * N + bColBase]);
            }
            const int phys_slot = swizzle_slot(k, nvec, kBSlots);
            *reinterpret_cast<Vec*>(&Bs[k][phys_slot * kVecW]) = v;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kBK; ++k) {
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
            #pragma unroll
            for (int m = 0; m < kTM; ++m)
                #pragma unroll
                for (int n = 0; n < kTN; ++n)
                    reg_C[m][n] += reg_A[m] * reg_B[n];
        }

        __syncthreads();
    }

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
    // CORRECTNESS FIX (found running on real hardware for the first time --
    // see the cp.async/cuda_has_hopper()/DoubleBuf-launch-config fixes
    // elsewhere in this file for the same story): this used to be padded
    // +1 (As/Bs[kBlockK][kBlockM+1]) for the usual scalar shared-memory
    // bank-conflict trick. wmma::load_matrix_sync does NOT tolerate that --
    // for a __half fragment it requires the leading dimension to be a
    // multiple of 8 elements (16 bytes), and kBlockM+1 = 65 is odd. Every
    // call aborted at runtime with cudaErrorMisalignedAddress. kBlockM/N
    // (64) are themselves already multiples of 8, so simply dropping the
    // padding satisfies the alignment requirement; the padding's bank-
    // conflict benefit was never realized here anyway since
    // load_matrix_sync is a single cooperative warp-wide instruction, not
    // per-thread strided scalar loads.
    __shared__ __half As[kBlockK][kBlockM];  // 16 x 64
    __shared__ __half Bs[kBlockK][kBlockN];  // 16 x 64

    // WMMA fragments for this warp.
    //
    // CORRECTNESS FIX (found running on real hardware for the first time --
    // see this file's other "found running on real hardware" comments):
    // these major-order tags were swapped relative to how As/Bs are
    // physically laid out, and load_matrix_sync trusted them blindly --
    // wrong VALUES, not a crash, so this survived compiling and even
    // running without complaint until compared against the reference GEMM.
    //
    // As is stored As[k][m] (transposed -- see its declaration comment
    // above: As[k][m] = A[m][k]), so with load_matrix_sync's row_major
    // convention (address(row,col) = ptr + row*ldm + col) and
    // as_ptr/ldm=kBlockM below, element(row,col) resolves to
    // As[k=row][m=col] = A[M=col, K=row] -- row and col land on the WRONG
    // axis (A's K ended up as the fragment's "row"/M axis and vice versa).
    // matrix_a needs col_major instead: address(row,col) = ptr + row +
    // col*ldm resolves to As[k=col][m=row] = A[M=row, K=col], which is
    // exactly the (row=M, col=K) semantics wmma::mma_sync expects of
    // matrix_a.
    //
    // Bs is stored Bs[k][n] in ITS natural (non-transposed) orientation,
    // so the same row_major convention that was wrong for As is the
    // correct one for Bs: address(row,col) = ptr + row*ldm + col resolves
    // to Bs[k=row][n=col] = B[K=row, N=col] -- exactly matrix_b's expected
    // (row=K, col=N) semantics. col_major (the previous tag) would have
    // swapped it the same way row_major swapped matrix_a above.
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

    // Store the accumulated fp32 fragment back to global memory.
    if (cWarpRow < M && cWarpCol < N) {
        wmma::store_matrix_sync(C + cWarpRow * N + cWarpCol, c_frag, N,
                                wmma::mem_row_major);
    }
}

// ============================================================================
// Kernel 6b: Pipelined Tensor Cores via WMMA (Level 8) -- bigger tiles +
// cp.async double buffering. NEW kernel, added after the "Reference --
// cuBLAS" section far below this file measured this GPU's realistic
// Tensor Core ceiling.
//
// kernel_wmma above (64x64 block tile, single-buffered, one wmma::mma_sync
// per warp per k-step) measured ~5 TFLOP/s on RTX 5080 -- cuBLAS's dense
// FP16 Tensor Core path (gemm_cuda_cublas_fp16, compute-only) measured
// ~118 TFLOP/s on the same GPU. That ~24x gap is almost entirely
// pipelining and tile size, not precision or instruction choice (both
// already use fp16 Tensor Cores) -- see "Reference -- cuBLAS" below for
// the measurements that motivated this kernel.
//
// This kernel closes part of that gap the way CUTLASS-style kernels do,
// while staying on the documented wmma:: C++ API (not raw mma.sync/
// ldmatrix PTX -- kernel_mma_ldmatrix below already demonstrates, and
// this file's fix history already proves, how much easier it is to
// introduce a silent wrong-value bug in hand-written register-mapping PTX
// than in compiler-managed WMMA fragments):
//
//   1. Bigger thread-block tile: 128x128 (vs 64x64) with BK=32 (vs 16),
//      so more work is done per shared-memory round trip and per
//      __syncthreads() pair.
//   2. Bigger per-warp tile: each of 8 warps (256 threads/block) owns a
//      32x64 output region == 2x4 = 8 WMMA 16x16x16 fragments, issuing
//      8 wmma::mma_sync calls per k-sub-step instead of kernel_wmma's 1,
//      amortizing load/sync overhead across more compute. A/B fragments
//      are each loaded once per k-sub-step and reused across the other
//      dimension (a_frag reused across all fn, b_frag[] reused across
//      all fm) -- the same register-blocking structure kernel_reg_tile/
//      kernel_double_buf already use for their scalar FMA micro-kernel.
//   3. Double-buffered shared memory loaded via cp.async (Ampere+; see
//      HPC_HAVE_CP_ASYNC above), so the NEXT k-tile's global->shared
//      copy overlaps the CURRENT k-tile's Tensor Core compute -- the
//      same structural fix already proven correct in kernel_double_buf's
//      cp.async bug fix elsewhere in this file, applied here to fp16
//      Tensor Core input instead of scalar FMA input. Falls back to a
//      synchronous vectorized copy (still double-buffered, just without
//      the async overlap) on pre-Ampere targets, matching kernel_double_
//      buf's own #ifdef HPC_HAVE_CP_ASYNC / #else pattern.
//
// Design choices that keep this correctness-tractable (unlike kernel_
// mma_ldmatrix's hand-mapped PTX registers, or kernel_hopper_wgmma's
// from-scratch descriptor layout):
//
//   - A and B are pre-converted to fp16 in GLOBAL memory once (via the
//     existing kernel_f32_to_f16, the same staging step gemm_cuda_cublas_
//     fp16 and kernel_hopper_wgmma already use) before this kernel
//     launches. cp.async is a same-dtype byte copy, not a converting
//     load -- it cannot do the fp32->fp16 narrowing kernel_wmma's
//     synchronous load does on the fly, so the conversion has to happen
//     as a separate step whenever cp.async is used at all.
//   - Unlike kernel_wmma, As is stored NATURALLY as As[m][k] (matching
//     A16's own row-major layout exactly, K contiguous) rather than
//     transposed as As[k][m]. This is a deliberate departure from
//     kernel_wmma's layout: cp.async can only copy a CONTIGUOUS run of
//     bytes to a CONTIGUOUS destination, and A16's natural per-row K
//     contiguity only lines up with a per-row-in-K destination too (i.e.
//     no transpose) -- so a_frag below is `row_major`, the OPPOSITE of
//     kernel_wmma's `col_major` a_frag. This is not a bug -- it is the
//     correct tag for THIS kernel's different (untransposed) physical
//     layout, exactly analogous to how kernel_wmma's own fix above
//     required matching its tag to ITS layout. Bs stays natural
//     (Bs[k][n], same as kernel_wmma), so b_frag stays `row_major` here
//     too, same as kernel_wmma.
//   - Every cp.async transfer moves a full 16-byte (8 x __half) chunk,
//     the largest size __pipeline_memcpy_async supports, chosen so a
//     single instruction per thread per chunk both maximizes throughput
//     and keeps every source/destination address provably 16-byte
//     aligned by construction (see the alignment argument below) --
//     no zfill/boundary-tile logic is needed at all because...
//   - ...this kernel deliberately requires M, N to be exact multiples of
//     128 and K an exact multiple of 32 (no tail handling, the same
//     scoping decision kernel_hopper_wgmma already makes for its own
//     tile shape) -- the host dispatch below falls back to the always-
//     correct kernel_wmma otherwise. Every alignment argument above
//     depends on this: cudaMalloc'd buffers are >=256-byte aligned, and
//     with K/N multiples of 32/128 (hence of 8), every row of A16/B16
//     this kernel reads a 16-byte chunk from starts at a byte offset
//     that is itself a multiple of 16 (offset-in-elements is always a
//     multiple of 8 given those divisibility constraints, so offset-in-
//     bytes = that * 2 is always a multiple of 16).
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

    __shared__ alignas(16) __half As[2][kPipeBM][kPipeBK];  // natural: As[m][k]
    __shared__ alignas(16) __half Bs[2][kPipeBK][kPipeBN];  // natural: Bs[k][n]

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
                    &Bs[cur][kSub * kWMMA_K][warpCol * kPipeWarpN + fn * kWMMA_N], kPipeBN);

            #pragma unroll
            for (int fm = 0; fm < kPipeFragM; ++fm) {
                wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K, __half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag,
                    &As[cur][warpRow * kPipeWarpM + fm * kWMMA_M][kSub * kWMMA_K], kPipeBK);
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
// Kernel 7: Raw Tensor Core MMA via mma.sync + ldmatrix (Level 6)
// Deliberately kept as a SEPARATE kernel from kernel_wmma above, not a
// refactor of it -- the point is to compare the two abstraction levels.
//
// *** VERIFIED on real hardware (RTX 5080, Blackwell sm_120, CUDA 13.2) ***
// Originally written with no CUDA toolkit or GPU available anywhere in the
// project, following the PTX ISA's documented instruction shapes and the
// standard published ldmatrix+mma.sync idiom as precisely as could be
// reproduced without a reference compile. As anticipated by this comment's
// original warning below, that produced exactly the predicted failure mode:
// it compiled and ran without complaint but computed numerically WRONG
// results (not a crash) once actually exercised against a reference GEMM.
// Root cause: the A-fragment's ldmatrix.x4 quadrant-to-register mapping had
// its row/col bits swapped (`aM` used `quadIdx/2` and `aK` used
// `quadIdx%2`, the reverse of the correct assignment) -- see the fix
// comment at the `aM`/`aK` computation below for the corrected formula and
// the reference implementation it was cross-checked against. All three
// GTest cases (N=64/128/256) now pass against the reference GEMM.
//
// Unlike a missing #include or a type error, that wrong-thread-to-fragment
// mapping bug silently produced numerically wrong output rather than
// failing to build or crashing -- this is exactly the class of bug the
// rest of this file's kernels (which use documented C++ APIs: wmma::,
// __pipeline_memcpy_async, or plain indexed loads) do not risk, and it is
// exactly why the WMMA kernel above exists as a *separate*, comparatively
// lower-risk Tensor Core kernel.
//
// What this kernel does, one level below WMMA:
//   WMMA (kernel_wmma):  wmma::load_matrix_sync / wmma::mma_sync -- the
//                        compiler manages which register holds which matrix
//                        element; native tile is 16x16x16.
//   Here:                ldmatrix.sync.aligned.m8n8.x{2,4}.shared.b16 loads
//                        raw shared-memory addresses into the *exact*
//                        per-thread registers mma.sync.m16n8k16 expects --
//                        the hardware does the 32-thread distribution, but
//                        WHICH address each thread supplies, and whether the
//                        load needs `.trans`, is this kernel's responsibility
//                        by hand. Native tile is 16x8x16 (note: 8 wide, not
//                        16 -- mma.sync's f16 m16n8k16 shape has a narrower
//                        N than WMMA's 16x16x16, so each warp here issues
//                        TWO side-by-side MMAs to cover the same 16x16 area
//                        kernel_wmma computes with one wmma::mma_sync call).
//
// Operand layout requirement (fixed by the instruction: only ".row.col" is
// defined for this shape/type combination -- there is no ".row.row" f16
// m16n8k16 variant):
//   A operand must be `.row`  (M x K, K the fast/contiguous axis)
//   B operand must be `.col`  (K x N, K the fast/contiguous axis)
//
// This kernel stores As[[M][K]] in shared memory in A's OWN natural
// row-major layout (K contiguous) specifically so the A operand needs NO
// transpose -- unlike every FMA-based kernel earlier in this file, which
// transposes A into As[K][M] for compute-loop column access. That
// optimization doesn't apply here (mma.sync does the whole 16x8x16 MMA in
// one hardware instruction; there is no manual per-element compute loop to
// optimize for). B is stored Bs[K][N] (N contiguous, B's own natural
// row-major layout) which is the OPPOSITE of the `.col` (K-contiguous)
// operand B needs -- so B's ldmatrix call below uses `.trans` to have the
// instruction transpose it during the load. ldmatrix's `.trans` only
// changes the internal register shuffle, not the per-thread source-address
// convention, so this does not change how `b_addr` is computed.
//
// ldmatrix address convention (PTX ISA "Warp-level Matrix Load
// Instruction: ldmatrix"): for `.x4`, each of the 32 lanes supplies ONE
// address; lanes are grouped in fours of eight (lane/8 = which of the 4
// 8x8 quadrants, lane%8 = which row within that quadrant), and each
// supplied address is the START of an 8-contiguous-element row read from
// shared memory. `.x2` uses only the first 16 lanes' addresses (lane/8 = 0
// or 1, lane%8 = row); this kernel has every lane (including 16-31)
// compute a valid, in-bounds address via `lane % 16` for the `.x2` (B)
// call, since ldmatrix is a warp-collective instruction and every
// participating lane must supply *some* valid address even where the
// result is unused.
//
// mma.sync.m16n8k16.f32 accumulator layout (PTX ISA "Matrix Fragments for
// mma.m16n8k16" -- the one piece of this kernel with an independent,
// well-known citation trail beyond this author's reconstruction):
//   groupID = lane / 4, threadInGroup = lane % 4
//   acc[0] -> C[groupID,          threadInGroup*2]
//   acc[1] -> C[groupID,          threadInGroup*2 + 1]
//   acc[2] -> C[groupID + 8,      threadInGroup*2]
//   acc[3] -> C[groupID + 8,      threadInGroup*2 + 1]
//
// Requires sm_80+ (Ampere) for the f16 m16n8k16 shape (guarded by
// __CUDA_ARCH__ below; the host dispatch additionally checks
// cuda_has_ampere() before ever launching this kernel). Falls back to
// kernel_wmma on sm_70-75 (Volta/Turing) via the host dispatch.
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
            // CORRECTNESS FIX (found running on real hardware for the first
            // time -- this kernel's file-level comment already flagged
            // itself as the single most likely place in this file for a
            // silent wrong-value bug, and this is it): ldmatrix.x4 loads
            // 4 fixed 8x8 chunks in lane-group order (chunk = lane/8 ->
            // a_frag[chunk]), and mma.sync.m16n8k16 expects those four
            // chunks in a SPECIFIC physical order -- row-quadrant fastest,
            // col-quadrant slowest, i.e. chunk0=(M0-7,K0-7),
            // chunk1=(M8-15,K0-7), chunk2=(M0-7,K8-15), chunk3=(M8-15,K8-15).
            // The previous formula had this backwards (quadIdx/2 selecting
            // the M half, quadIdx%2 selecting the K half), silently
            // transposing which quadrant landed in which a_frag register.
            // Cross-checked against a working reference implementation
            // (am17an.bearblog.dev's mma-tensor-cores GEMM writeup): row
            // (M) must vary with quadIdx%2 (the fast-varying bit), col (K)
            // with quadIdx/2 (the slow-varying bit) -- the reverse of what
            // was here. The B fragment loop below was already correct
            // against that same reference (bK = bLane reconstructs
            // lane%16 exactly, matching the reference's row=lane%16).
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

            for (int which = 0; which < 2; ++which) {
                // --- B fragment: 16(K)x8(N) tile, 2 quadrants, ldmatrix.x2 + .trans ---
                const int bLane    = lane % 16;
                const int bQuadIdx = bLane / 8;          // 0..1
                const int bQuadRow = bLane % 8;           // 0..7
                const int bK = bQuadIdx * 8 + bQuadRow;
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
    // sm_75 and below: the f16 m16n8k16 mma.sync shape used above does not
    // exist. This branch is never launched on such hardware (host dispatch
    // checks cuda_has_ampere() first) -- it exists only so the file still
    // compiles when -arch targets sm_75 or older.
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)N;
#endif
}

// ============================================================================
// Kernel 8: Hopper warp specialization + TMA (wgmma) -- Level 7
//
// ****************************************************************
// *** BEST-EFFORT, LIKELY-BROKEN, EXPLICITLY UNVERIFIED KERNEL. ***
// ****************************************************************
// The user asked for this specific technique with the explicit
// understanding, agreed in advance, that it would be written as an honest
// best-effort sketch rather than working code: sm_90a wgmma + TMA has no
// public C++ intrinsic surface at all (unlike WMMA, unlike even mma.sync/
// ldmatrix above) -- every input is raw inline PTX and a driver-API
// tensor-map descriptor, hand-written against the PTX ISA's prose
// description with no compiler or hardware available anywhere in this
// project to check it against. Hand-written kernels using these primitives
// essentially do not exist outside CUTLASS/cuDNN internals; even NVIDIA's
// own examples build this through the CUTLASS template library, not by
// hand. Treat every bit-layout and register-mapping comment below as "my
// best reading of the documentation", not as a verified fact -- several are
// flagged with an explicit confidence level.
//
// What this kernel demonstrates (the two requested techniques):
//
//   TMA (Tensor Memory Accelerator): a single thread issues one
//   instruction (`cp.async.bulk.tensor.2d...`) that asynchronously copies
//   an entire 2-D tile from global to shared memory, using a descriptor
//   (`CUtensorMap`) built ONCE on the host via the driver API
//   (`cuTensorMapEncodeTiled`) that encodes the tensor's global shape,
//   strides, and box (tile) size. This replaces the "every thread computes
//   its own address and issues its own load" pattern every other kernel in
//   this file uses.
//
//   Warp specialization: threads in a thread block take on ROLES rather
//   than all executing identical code. Here, one warpgroup (128 threads)
//   is the PRODUCER -- it does nothing but issue TMA loads and signal
//   completion via an mbarrier -- while a second warpgroup is the CONSUMER
//   -- it waits on that mbarrier, then issues `wgmma.mma_async`
//   (warpgroup-wide MMA, operating on all 128 consumer threads at once)
//   directly against the shared-memory tile the producer just staged, with
//   no per-thread fragment loading step at all (wgmma reads its operands
//   from shared memory via a 64-bit "matrix descriptor", not from
//   registers the way mma.sync does).
//
// Scope deliberately kept minimal (a single-buffered, non-deeply-pipelined
// producer/consumer handshake, exact-multiple-of-tile-size M/N/K only, no
// tail handling) -- a full multi-stage pipeline is exactly the kind of
// thing CUTLASS exists to get right, and adding more untestable complexity
// here would not make this kernel more trustworthy.
//
// Requires sm_90a specifically (not just sm_90 -- wgmma/TMA are excluded
// from the portable "family" compute-capability feature set and need the
// architecture-specific target). Falls back to kernel_mma_ldmatrix (or
// kernel_wmma) via the host dispatch on non-Hopper hardware.
// ============================================================================

// ----------------------------------------------------------------------------
// fp32 -> fp16 staging kernel: TMA needs its source tensor already resident
// in global memory in the target element type (fp16 here), unlike the WMMA/
// mma.sync kernels above, which convert on-the-fly per shared-memory tile.
// Confidence: HIGH (this is a completely ordinary elementwise kernel).
// ----------------------------------------------------------------------------
__global__ void kernel_f32_to_f16(const float* __restrict__ src,
                                  __half* __restrict__ dst,
                                  int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        dst[idx] = __float2half(src[idx]);
}

// Thread-block tile: one warpgroup (128 threads) computes one 64x64 output
// tile per k-step of 16 (wgmma.m64n64k16 f16 native shape). Declared
// OUTSIDE the sm_90a guard below (plain compile-time constants, not
// device-arch-specific) so the host-side launch() dispatcher can also see
// them for grid-size computation.
static constexpr int kWgmmaM = 64;
static constexpr int kWgmmaN = 64;
static constexpr int kWgmmaK = 16;
static constexpr int kWgmmaWarpgroupThreads = 128;

// == 900, not >= 900: wgmma/TMA are Hopper-exclusive (sm_90/sm_90a) and are
// NOT part of Blackwell's (sm_100/sm_110/sm_120/sm_121, __CUDA_ARCH__ 1000+)
// forward-compatible feature set -- ptxas rejects wgmma.mma_async and
// friends outright for those targets. Verified on real Blackwell hardware
// (RTX 5080, sm_120): the old ">= 900" guard let this branch compile for
// -arch=native there and ptxas aborted the build. See cuda_has_hopper()
// below for the matching runtime-dispatch fix.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900

// ----------------------------------------------------------------------------
// Shared-memory matrix descriptor for wgmma operands (PTX ISA "Asynchronous
// Warpgroup Level Matrix Shared Memory Layout" / matrix descriptor format).
// Confidence: MEDIUM -- the field existence (start address, leading-dim
// offset, stride-dim offset, swizzle mode, all in units of 16 bytes) is
// well attested across public Hopper-kernel writeups; the EXACT bit offsets
// below are this author's best reconstruction and are the single most
// likely place in this file for a silent mismatch.
// ----------------------------------------------------------------------------
__device__ __forceinline__ uint64_t make_smem_desc(const void* smem_ptr,
                                                    int leading_dim_bytes,
                                                    int stride_dim_bytes) {
    uint64_t addr = static_cast<uint64_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (addr >> 4) & 0x3FFF;                                   // bits 0-13
    desc |= (static_cast<uint64_t>(leading_dim_bytes >> 4) & 0x3FFF) << 16;  // bits 16-29
    desc |= (static_cast<uint64_t>(stride_dim_bytes  >> 4) & 0x3FFF) << 32;  // bits 32-45
    // Swizzle mode left at 0 (none) -- bits 62-63. A real implementation
    // would match this to the swizzle mode baked into the TMA descriptor
    // that populated this shared-memory tile; left at "none" here to avoid
    // compounding an already-uncertain bit layout with an unverified
    // swizzle-mode interaction.
    return desc;
}

__global__ void __launch_bounds__(256)  // 2 warpgroups: producer + consumer
kernel_hopper_wgmma(const __half* __restrict__ A16,  // pre-converted, row-major MxK
                    const __half* __restrict__ B16,  // pre-converted, row-major KxN
                    float* __restrict__ C,
                    int M, int K, int N,
                    const __grid_constant__ CUtensorMap tensorMapA,
                    const __grid_constant__ CUtensorMap tensorMapB) {
    const int warpgroupId = threadIdx.x / kWgmmaWarpgroupThreads;  // 0 = producer, 1 = consumer
    const bool isProducer  = (warpgroupId == 0);

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    __shared__ alignas(128) __half As[kWgmmaM][kWgmmaK];
    __shared__ alignas(128) __half Bs[kWgmmaK][kWgmmaN];
    // Two mbarriers: `full` (producer -> consumer: "tile is loaded"),
    // `empty` (consumer -> producer: "tile has been consumed, reuse it").
    // Single-buffered by design (see file-level scope note above) -- a
    // production pipeline would use N buffers and N mbarrier pairs.
    __shared__ uint64_t full_bar;
    __shared__ uint64_t empty_bar;

    if (threadIdx.x == 0) {
        // mbarrier.init expects the *thread count* that will arrive on it.
        // `full_bar`: 1 arrival expected (the single TMA-issuing thread,
        // whose "arrive" is implicit in the TMA instruction's
        // mbarrier::complete_tx qualifier). `empty_bar`: all 128 consumer
        // threads must finish reading before the producer reuses the tile.
        asm volatile("mbarrier.init.shared.b64 [%0], 1;\n"
                    :: "l"(__cvta_generic_to_shared(&full_bar)));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                    :: "l"(__cvta_generic_to_shared(&empty_bar)),
                       "r"(kWgmmaWarpgroupThreads));
    }
    __syncthreads();

    // Accumulator: wgmma.m64n64k16.f32 output distributed across the 128
    // consumer threads. Confidence: LOW on the exact per-thread (row,col)
    // mapping used at STORE time below -- see the comment there. The
    // register COUNT (32 f32 per thread for a 64x64 tile / 128 threads =
    // 32 elements/thread) is a simple area/thread-count computation and is
    // high confidence; which 32 (row,col) pairs a given thread owns is not.
    float acc[32] = {};

    const int nTilesK = (K + kWgmmaK - 1) / kWgmmaK;

    for (int tileK = 0; tileK < nTilesK; ++tileK) {
        if (isProducer) {
            if (threadIdx.x == 0) {
                if (tileK > 0) {
                    // Wait for the consumer to finish with the PREVIOUS
                    // tile's data before overwriting it (single-buffered).
                    asm volatile(
                        "{\n"
                        ".reg .pred p;\n"
                        "L_WAIT_EMPTY:\n"
                        "mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n"
                        "@!p bra L_WAIT_EMPTY;\n"
                        "}\n"
                        :: "l"(__cvta_generic_to_shared(&empty_bar)));
                }
                // Issue the two TMA bulk-tensor loads (A tile, B tile).
                // Coordinates are in ELEMENTS, per the CUtensorMap's own
                // element type -- (col, row) order per TMA's convention of
                // fastest-varying dimension first.
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                    "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                    :: "l"(__cvta_generic_to_shared(&As[0][0])),
                       "l"(reinterpret_cast<uint64_t>(&tensorMapA)),
                       "r"(tileK * kWgmmaK), "r"(blockRow * kWgmmaM),
                       "l"(__cvta_generic_to_shared(&full_bar)));
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                    "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                    :: "l"(__cvta_generic_to_shared(&Bs[0][0])),
                       "l"(reinterpret_cast<uint64_t>(&tensorMapB)),
                       "r"(blockCol * kWgmmaN), "r"(tileK * kWgmmaK),
                       "l"(__cvta_generic_to_shared(&full_bar)));
            }
            // Non-issuing producer threads simply idle this iteration --
            // real warp-specialized kernels usually give the producer
            // warpgroup additional prefetch/bookkeeping work; omitted here
            // to keep an already-speculative kernel as small as possible.
        } else {
            // Consumer: wait for the producer's TMA loads to complete.
            if (threadIdx.x == kWgmmaWarpgroupThreads) {
                asm volatile(
                    "{\n"
                    ".reg .pred p;\n"
                    "L_WAIT_FULL:\n"
                    "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
                    "@!p bra L_WAIT_FULL;\n"
                    "}\n"
                    :: "l"(__cvta_generic_to_shared(&full_bar)), "r"(tileK & 1));
            }
            __syncwarp();  // only meaningful within the issuing warp; see note below

            const uint64_t descA = make_smem_desc(&As[0][0], kWgmmaK * 2, kWgmmaM * kWgmmaK * 2);
            const uint64_t descB = make_smem_desc(&Bs[0][0], kWgmmaN * 2, kWgmmaK * kWgmmaN * 2);

            // wgmma.mma_async: warpgroup-wide, all 128 consumer threads
            // issue the IDENTICAL instruction (SIMT-cooperative, like
            // wmma:: / mma.sync, but at warpgroup granularity). Accumulator
            // registers persist across calls (scale-d=1 after the first).
            // Confidence: MEDIUM on the instruction syntax/operand count
            // (32 accumulator registers matches the documented m64n64k16.f32
            // shape); LOW on whether `p` (scale-d) and the two trailing
            // 0-immediates (trans-a, trans-b) are in the right operand
            // positions for this PTX ISA version.
            asm volatile(
                "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
                "%32, %33, %34, 0, 0;\n"
                : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
                  "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
                  "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
                  "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
                  "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
                  "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
                  "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
                  "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31])
                : "l"(descA), "l"(descB), "r"(tileK > 0 ? 1 : 0));
            asm volatile("wgmma.commit_group.sync.aligned;\n");
            asm volatile("wgmma.wait_group.sync.aligned 0;\n");

            // Signal the producer that this tile's shared-memory buffer is
            // free to be overwritten with the next one.
            __syncthreads();  // all 128 consumer threads done reading
            if (threadIdx.x == kWgmmaWarpgroupThreads) {
                asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n"
                            :: "l"(__cvta_generic_to_shared(&empty_bar)));
            }
        }
    }

    // Store the accumulator to global C.
    // Confidence: LOW -- see the accumulator declaration comment above.
    // This uses a placeholder linear decomposition (NOT a verified wgmma
    // output layout) purely so the kernel has SOME defined store behaviour;
    // treat any numerical result from this kernel as unverified even in
    // the cases where it happens to compile and run.
    if (!isProducer) {
        const int consumerTid = threadIdx.x - kWgmmaWarpgroupThreads;  // 0..127
        #pragma unroll
        for (int e = 0; e < 32; ++e) {
            const int flat = consumerTid * 32 + e;   // 0..4095 -- placeholder mapping
            const int r = flat / kWgmmaN;
            const int c = flat % kWgmmaN;
            const int gi = blockRow * kWgmmaM + r;
            const int gj = blockCol * kWgmmaN + c;
            if (gi < M && gj < N)
                C[gi * N + gj] = acc[e];
        }
    }
}

#else  // __CUDA_ARCH__ != 900 (Blackwell, pre-Hopper, or host compilation pass)

// sm_90a-only: this branch exists purely so the translation unit compiles
// when no -arch target is Hopper. Never launched on such hardware (host
// dispatch checks cuda_has_hopper() first).
__global__ void kernel_hopper_wgmma(const __half* __restrict__, const __half* __restrict__,
                                    float* __restrict__, int, int, int,
                                    const __grid_constant__ CUtensorMap,
                                    const __grid_constant__ CUtensorMap) {}

#endif  // __CUDA_ARCH__ == 900

// ----------------------------------------------------------------------------
// Host-side TMA descriptor construction (driver API).
// Confidence: MEDIUM -- cuTensorMapEncodeTiled's signature and parameter
// meanings are drawn from the CUDA driver API reference; the specific
// element/data-type and swizzle/interleave/L2-promotion/OOB-fill enum
// values chosen below (all "none"/"default") are the least risky choice
// for each field, not a performance-tuned configuration.
// ----------------------------------------------------------------------------
static CUtensorMap make_tensor_map_2d(const __half* globalAddr, std::uint64_t rows,
                                      std::uint64_t cols, std::uint32_t boxRows,
                                      std::uint32_t boxCols) {
    CUtensorMap tensorMap{};
    // TMA addresses tensors as {fastest-varying dim, ..., slowest-varying
    // dim}; for a row-major [rows x cols] fp16 matrix, cols is fastest.
    const cuuint64_t globalDim[2]     = {cols, rows};
    const cuuint64_t globalStrides[1] = {cols * sizeof(__half)};  // rank-1: only the non-fastest dim needs an explicit stride
    const cuuint32_t boxDim[2]        = {boxCols, boxRows};
    const cuuint32_t elementStrides[2] = {1, 1};

    const CUresult res = cuTensorMapEncodeTiled(
        &tensorMap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        /*tensorRank=*/2,
        const_cast<void*>(static_cast<const void*>(globalAddr)),
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuTensorMapEncodeTiled failed (code " +
                                 std::to_string(static_cast<int>(res)) + ")");
    }
    return tensorMap;
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

// NOT device_has_capability(9, 0): wgmma/TMA (sm_90a) are Hopper-exclusive
// instructions, not part of the forward-compatible "family" feature set --
// ptxas rejects wgmma.mma_async/commit_group/wait_group outright when
// targeting Blackwell (sm_100/sm_110/sm_120/sm_121, compute major 10/12),
// even though those report compute capability >= 9.0. Verified against
// real Blackwell hardware (RTX 5080, sm_120): device_has_capability(9, 0)'s
// ">=" semantics -- correct for tensor-core/Ampere gating below, since
// those ARE forward-compatible -- silently mis-selected kernel_hopper_wgmma
// on non-Hopper hardware and ptxas aborted the whole build.
bool cuda_has_hopper() noexcept {
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess) return false;
    for (int d = 0; d < devCount; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) == cudaSuccess)
            if (prop.major == 9) return true;
    }
    return false;
}

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

enum class GemmKind { Naive, Reordered, Blocked, RegTile, DoubleBuf, Wmma,
                      Vectorized, MmaLdmatrix, HopperWgmma, WmmaPipelined };

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
        constexpr int LBM = kDBufBM<T>;
        constexpr int LBN = kDBufBN<T>;
        // CORRECTNESS FIX (found running on real hardware for the first
        // time -- see cp.async / cuda_has_hopper() fixes above for the
        // same story): this was hardcoded to 256 threads, correct only by
        // coincidence for float (kDBufBM/BN=kBM/kBN=128 -> (128/8)*(128/8)
        // = 256). For double, kDBufBM/BN halve to 64 (see the comment on
        // kDBufBM above) so only (64/8)*(64/8) = 64 threads' worth of
        // (threadRow, threadCol) mapping is valid -- the other 192 threads
        // computed threadRow up to 31 instead of the valid 0..7, reading
        // As/Bs[...][threadRow*kTM+m] far past each row's real LBM+1=65
        // elements (out-of-bounds shared-memory read) and, whenever that
        // block happened to land in-bounds of a LARGER matrix (multi-block
        // grids, i.e. N/M > 64), writing the resulting garbage over a
        // different block's already-correct C tile. Single-block cases
        // (N<=64) masked this because gi<M's bounds check discarded every
        // phantom thread's store. Kept generic (not hardcoded 64) so a
        // future T with a different kDBufBM/BN doesn't reintroduce this.
        const dim3 block((LBM / kTM) * (LBN / kTN));
        const dim3 grid((N + LBN-1)/LBN, (M + LBM-1)/LBM);
        kernel_double_buf<T><<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);

    } else if (kind == GemmKind::Wmma) {
        // WMMA is fp32-only in this implementation (converts to fp16 internally).
        if constexpr (!std::is_same_v<T, float>) {
            throw std::runtime_error("gemm_cuda_wmma is only supported for float");
        } else {
            const dim3 block(kWarpM * kWarpN * 32);  // 4*4*32 = 512 threads
            const dim3 grid((N + kBlockN-1)/kBlockN, (M + kBlockM-1)/kBlockM);
            if (cuda_has_tensor_cores()) {
                kernel_wmma<<<grid, block>>>(
                    reinterpret_cast<const float*>(dA.ptr),
                    reinterpret_cast<const float*>(dB.ptr),
                    reinterpret_cast<float*>(dC.ptr), M, K, N);
            } else {
                // Fallback: use double-buf kernel (always correct).
                constexpr int FLBM = kDBufBM<float>;
                constexpr int FLBN = kDBufBN<float>;
                const dim3 block2(256);
                const dim3 grid2((N + FLBN-1)/FLBN, (M + FLBM-1)/FLBM);
                kernel_double_buf<T><<<grid2, block2>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);
            }
        }

    } else if (kind == GemmKind::Vectorized) {
        // Vectorized loads require K and N to be multiples of the 128-bit
        // vector width (4 elements for float, 2 for double) -- see
        // kernel_vectorized's file comment. Falls back to the (now-fixed)
        // kernel_reg_tile otherwise, which is always correct for any shape.
        constexpr int kVecW = VecTraits<T>::kWidth;
        if (K % kVecW == 0 && N % kVecW == 0) {
            const dim3 block(256);
            const dim3 grid((N + kBN-1)/kBN, (M + kBM-1)/kBM);
            kernel_vectorized<T><<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);
        } else {
            const dim3 block(256);
            const dim3 grid((N + kBN-1)/kBN, (M + kBM-1)/kBM);
            kernel_reg_tile<T><<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);
        }

    } else if (kind == GemmKind::MmaLdmatrix) {
        // Raw mma.sync + ldmatrix is fp32-only (like WMMA) and needs sm_80+
        // for the f16 m16n8k16 shape used. Falls back to kernel_wmma
        // (sm_70+) otherwise.
        if constexpr (!std::is_same_v<T, float>) {
            throw std::runtime_error("gemm_cuda_mma_ldmatrix is only supported for float");
        } else {
            if (cuda_has_ampere()) {
                const dim3 block(kMmaWarpM * kMmaWarpN * 32);  // 512 threads
                const dim3 grid((N + kMmaBlockN-1)/kMmaBlockN, (M + kMmaBlockM-1)/kMmaBlockM);
                kernel_mma_ldmatrix<<<grid, block>>>(
                    reinterpret_cast<const float*>(dA.ptr),
                    reinterpret_cast<const float*>(dB.ptr),
                    reinterpret_cast<float*>(dC.ptr), M, K, N);
            } else if (cuda_has_tensor_cores()) {
                const dim3 block(kWarpM * kWarpN * 32);
                const dim3 grid((N + kBlockN-1)/kBlockN, (M + kBlockM-1)/kBlockM);
                kernel_wmma<<<grid, block>>>(
                    reinterpret_cast<const float*>(dA.ptr),
                    reinterpret_cast<const float*>(dB.ptr),
                    reinterpret_cast<float*>(dC.ptr), M, K, N);
            } else {
                constexpr int FLBM = kDBufBM<float>;
                constexpr int FLBN = kDBufBN<float>;
                const dim3 block2(256);
                const dim3 grid2((N + FLBN-1)/FLBN, (M + FLBM-1)/FLBM);
                kernel_double_buf<T><<<grid2, block2>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);
            }
        }

    } else if (kind == GemmKind::HopperWgmma) {
        // UNVERIFIED -- see kernel_hopper_wgmma's file comment. fp32-only,
        // sm_90a-only; requires M/N/K to be exact multiples of the wgmma
        // tile shape (no tail handling in this deliberately-minimal sketch).
        // Falls back to MmaLdmatrix/Wmma/double_buf otherwise, in that order.
        if constexpr (!std::is_same_v<T, float>) {
            throw std::runtime_error("gemm_cuda_hopper_wgmma is only supported for float");
        } else {
            const bool exactTiles = (M % kWgmmaM == 0) && (N % kWgmmaN == 0) && (K % kWgmmaK == 0);
            if (cuda_has_hopper() && exactTiles) {
                // Stage A, B into fp16 global buffers -- TMA needs its
                // source tensor already resident in the target element type.
                DeviceBuffer<__half> dA16(static_cast<std::size_t>(M) * K);
                DeviceBuffer<__half> dB16(static_cast<std::size_t>(K) * N);
                {
                    const int threads = 256;
                    const int blocksA = (M * K + threads - 1) / threads;
                    const int blocksB = (K * N + threads - 1) / threads;
                    kernel_f32_to_f16<<<blocksA, threads>>>(
                        reinterpret_cast<const float*>(dA.ptr), dA16.ptr, M * K);
                    kernel_f32_to_f16<<<blocksB, threads>>>(
                        reinterpret_cast<const float*>(dB.ptr), dB16.ptr, K * N);
                    CUDA_CHECK(cudaGetLastError());
                }
                const CUtensorMap tensorMapA = make_tensor_map_2d(
                    dA16.ptr, static_cast<std::uint64_t>(M), static_cast<std::uint64_t>(K),
                    kWgmmaM, kWgmmaK);
                const CUtensorMap tensorMapB = make_tensor_map_2d(
                    dB16.ptr, static_cast<std::uint64_t>(K), static_cast<std::uint64_t>(N),
                    kWgmmaK, kWgmmaN);
                const dim3 block(2 * kWgmmaWarpgroupThreads);  // producer + consumer warpgroups
                const dim3 grid(N / kWgmmaN, M / kWgmmaM);
                kernel_hopper_wgmma<<<grid, block>>>(
                    dA16.ptr, dB16.ptr, reinterpret_cast<float*>(dC.ptr), M, K, N,
                    tensorMapA, tensorMapB);
            } else if (cuda_has_ampere()) {
                const dim3 block(kMmaWarpM * kMmaWarpN * 32);
                const dim3 grid((N + kMmaBlockN-1)/kMmaBlockN, (M + kMmaBlockM-1)/kMmaBlockM);
                kernel_mma_ldmatrix<<<grid, block>>>(
                    reinterpret_cast<const float*>(dA.ptr),
                    reinterpret_cast<const float*>(dB.ptr),
                    reinterpret_cast<float*>(dC.ptr), M, K, N);
            } else if (cuda_has_tensor_cores()) {
                const dim3 block(kWarpM * kWarpN * 32);
                const dim3 grid((N + kBlockN-1)/kBlockN, (M + kBlockM-1)/kBlockM);
                kernel_wmma<<<grid, block>>>(
                    reinterpret_cast<const float*>(dA.ptr),
                    reinterpret_cast<const float*>(dB.ptr),
                    reinterpret_cast<float*>(dC.ptr), M, K, N);
            } else {
                constexpr int FLBM = kDBufBM<float>;
                constexpr int FLBN = kDBufBN<float>;
                const dim3 block2(256);
                const dim3 grid2((N + FLBN-1)/FLBN, (M + FLBM-1)/FLBM);
                kernel_double_buf<T><<<grid2, block2>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);
            }
        }

    } else if (kind == GemmKind::WmmaPipelined) {
        // NEW kernel (Level 8) -- see kernel_wmma_pipelined's file comment
        // for the full rationale. fp32-only, sm_70+ (Tensor Cores; the
        // cp.async double-buffering benefit specifically needs sm_80+,
        // but the kernel is still correct without it -- see HPC_HAVE_
        // CP_ASYNC's #else fallback in kernel_wmma_pipelined). Requires
        // M/N exact multiples of 128 and K an exact multiple of 32 (no
        // tail handling, same scoping decision as kernel_hopper_wgmma);
        // falls back to the always-correct kernel_wmma otherwise.
        if constexpr (!std::is_same_v<T, float>) {
            throw std::runtime_error("gemm_cuda_wmma_pipelined is only supported for float");
        } else {
            const bool exactTiles = (M % kPipeBM == 0) && (N % kPipeBN == 0) && (K % kPipeBK == 0);
            if (cuda_has_tensor_cores() && exactTiles) {
                DeviceBuffer<__half> dA16(static_cast<std::size_t>(M) * K);
                DeviceBuffer<__half> dB16(static_cast<std::size_t>(K) * N);
                {
                    const int threads = 256;
                    const int blocksA = (M * K + threads - 1) / threads;
                    const int blocksB = (K * N + threads - 1) / threads;
                    kernel_f32_to_f16<<<blocksA, threads>>>(
                        reinterpret_cast<const float*>(dA.ptr), dA16.ptr, M * K);
                    kernel_f32_to_f16<<<blocksB, threads>>>(
                        reinterpret_cast<const float*>(dB.ptr), dB16.ptr, K * N);
                    CUDA_CHECK(cudaGetLastError());
                }
                const dim3 block(kPipeNumWarps * 32);
                const dim3 grid(N / kPipeBN, M / kPipeBM);
                kernel_wmma_pipelined<<<grid, block>>>(
                    dA16.ptr, dB16.ptr, reinterpret_cast<float*>(dC.ptr), M, K, N);
            } else if (cuda_has_tensor_cores()) {
                const dim3 block(kWarpM * kWarpN * 32);
                const dim3 grid((N + kBlockN-1)/kBlockN, (M + kBlockM-1)/kBlockM);
                kernel_wmma<<<grid, block>>>(
                    reinterpret_cast<const float*>(dA.ptr),
                    reinterpret_cast<const float*>(dB.ptr),
                    reinterpret_cast<float*>(dC.ptr), M, K, N);
            } else {
                constexpr int FLBM = kDBufBM<float>;
                constexpr int FLBN = kDBufBN<float>;
                const dim3 block2(256);
                const dim3 grid2((N + FLBN-1)/FLBN, (M + FLBM-1)/FLBM);
                kernel_double_buf<T><<<grid2, block2>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);
            }
        }
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
template <typename T>
void gemm_cuda_vectorized(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    launch<T>(GemmKind::Vectorized, A, B, C);
}
// Raw mma.sync + ldmatrix is float-only, like WMMA.
void gemm_cuda_mma_ldmatrix(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    launch<float>(GemmKind::MmaLdmatrix, A, B, C);
}
// Hopper wgmma+TMA is float-only. UNVERIFIED -- see kernel_hopper_wgmma.
void gemm_cuda_hopper_wgmma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    launch<float>(GemmKind::HopperWgmma, A, B, C);
}
// Pipelined WMMA (Level 8, NEW) is float-only, like WMMA/mma_ldmatrix
// above -- see kernel_wmma_pipelined's file comment.
void gemm_cuda_wmma_pipelined(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    launch<float>(GemmKind::WmmaPipelined, A, B, C);
}

// Raw-device-pointer, compute-only entry point -- same reasoning as the
// cuBLAS raw-device-pointer functions further down this file: for a fair
// compute-only comparison against gemm_cuda_cublas_fp16_device (measured
// ~118 TFLOP/s), timing must exclude the fp32->fp16 conversion and
// cudaMalloc/H2D/D2H that gemm_cuda_wmma_pipelined's Matrix<float>-based
// wrapper above always pays. Caller must guarantee M/N/K satisfy the
// exact-tile requirement (M,N multiples of 128; K a multiple of 32) --
// unlike the wrapper above, this does NOT fall back to kernel_wmma, since
// there is no non-fp16 input to fall back from at this layer.
void gemm_cuda_wmma_pipelined_device(const void* dA16, const void* dB16, float* dC,
                                     int M, int K, int N) {
    const dim3 block(kPipeNumWarps * 32);
    const dim3 grid(N / kPipeBN, M / kPipeBM);
    kernel_wmma_pipelined<<<grid, block>>>(
        static_cast<const __half*>(dA16), static_cast<const __half*>(dB16), dC, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Reference -- cuBLAS (vendor-tuned upper bound, not part of the hand-
// written kernel ladder above)
//
// Added to answer a concrete question after the Levels 0-6 real-hardware
// verification pass above: the hand-written Tensor Core kernels
// (gemm_cuda_wmma, gemm_cuda_mma_ldmatrix) measured ~5 TFLOP/s on this
// RTX 5080 -- well under the "100-200 TFLOP/s" a well-tuned Tensor Core
// GEMM should reach on Blackwell -- because they are small (64x64 tiles),
// single-buffered, and unpipelined. Before attempting a much larger
// hand-written rewrite (bigger tiles, multi-stage cp.async pipelining,
// larger per-warp MMA fragments) to close that gap, gemm_cuda_cublas_tf32
// establishes what NVIDIA's own production GEMM actually achieves here as
// the realistic ceiling to rewrite toward.
//
// gemm_cuda_cublas<T>      -- plain SGEMM/DGEMM, cuBLAS's own tuned SIMT
//                              kernel. Ceiling for the FMA-based kernels
//                              (naive/blocked/reg_tile/double_buf/vectorized).
// gemm_cuda_cublas_tf32    -- fp32 in/out, TF32 Tensor Core compute
//                              (10-bit mantissa, same precision class as
//                              the fp16 kernels above). Ceiling for the
//                              Tensor Core kernels (wmma/mma_ldmatrix).
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

template <typename T>
void gemm_cuda_cublas(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    DeviceBuffer<T> dA(static_cast<std::size_t>(M) * K);
    DeviceBuffer<T> dB(static_cast<std::size_t>(K) * N);
    DeviceBuffer<T> dC(static_cast<std::size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(dA.ptr, A.data(), static_cast<std::size_t>(M) * K * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, B.data(), static_cast<std::size_t>(K) * N * sizeof(T), cudaMemcpyHostToDevice));

    const T alpha = T{1}, beta = T{0};
    // hpc::Matrix is row-major; cuBLAS is column-major. Row-major C = A*B
    // is exactly column-major C^T = B^T*A^T over the SAME memory -- so
    // swapping A<->B (and M<->N) and asking cuBLAS for an ordinary
    // (no-transpose) column-major C^T=B^T*A^T reproduces our row-major
    // C=A*B with no data movement and no transpose flags. Standard trick
    // for using a column-major BLAS from row-major storage.
    cublasStatus_t st;
    if constexpr (std::is_same_v<T, float>) {
        st = cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                          N, M, K, &alpha, dB.ptr, N, dA.ptr, K, &beta, dC.ptr, N);
    } else {
        st = cublasDgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                          N, M, K, &alpha, dB.ptr, N, dA.ptr, K, &beta, dC.ptr, N);
    }
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasSgemm/cublasDgemm failed, status=" + std::to_string(static_cast<int>(st)));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC.ptr, static_cast<std::size_t>(M) * N * sizeof(T), cudaMemcpyDeviceToHost));
}

// fp32-only, like WMMA/mma_ldmatrix above -- TF32 (10-bit mantissa, ~1e-3
// relative error, same precision class as those fp16 kernels) via
// cublasGemmEx's fast-TF32 compute type. Requires sm_80+ (Ampere+); on
// older hardware cuBLAS itself transparently falls back to a plain fp32
// SIMT path (no explicit fallback needed here, unlike the hand-written
// kernels above).
void gemm_cuda_cublas_tf32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    DeviceBuffer<float> dA(static_cast<std::size_t>(M) * K);
    DeviceBuffer<float> dB(static_cast<std::size_t>(K) * N);
    DeviceBuffer<float> dC(static_cast<std::size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(dA.ptr, A.data(), static_cast<std::size_t>(M) * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, B.data(), static_cast<std::size_t>(K) * N * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
    // Same row-major/column-major swap trick as gemm_cuda_cublas above.
    const cublasStatus_t st = cublasGemmEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha,
        dB.ptr, CUDA_R_32F, N,
        dA.ptr, CUDA_R_32F, K,
        &beta,
        dC.ptr, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasGemmEx (TF32) failed, status=" + std::to_string(static_cast<int>(st)));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC.ptr, static_cast<std::size_t>(M) * N * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Reference -- cuBLAS, raw-device-pointer entry points (peak-compute-only
// benchmarking)
//
// Every kernel above -- including gemm_cuda_cublas/_tf32 just above -- times
// a FULL round trip (cudaMalloc + H2D copy + compute + D2H copy) on every
// call, by design, so every kernel in this file is measured the same way
// (see docs/benchmarks.md's "all CUDA benchmarks include host<->device
// transfer time" note). That is the wrong methodology to answer "what is
// this GPU's actual achievable compute throughput" at problem sizes large
// enough to matter: at N=16384 the ~3.2 GB of host<->device traffic plus
// per-call cudaMalloc of ~1 GB buffers dominates wall-clock time far more
// than the matmul itself, which is exactly why gemm_cuda_cublas_tf32's
// end-to-end throughput (measured ~22 TFLOP/s at N=16384 on RTX 5080)
// badly understates the GPU's real Tensor Core throughput.
//
// These two entry points take pre-allocated, already-resident device
// pointers and do nothing but issue the GEMM call -- allocate and copy
// once outside the timed region (see BM_CudaCublas*ComputeOnly in
// bench_gemm_cuda.cpp), then call these repeatedly inside it. They call
// the exact same cublasSgemm/cublasGemmEx as the Matrix<T>-based wrappers
// above (same row-major/column-major swap trick), so their correctness is
// already covered by CudaCublasFloat/CudaCublasTf32Float's GTest cases --
// no separate correctness test needed for boilerplate that skips a memcpy.
// ============================================================================
void gemm_cuda_cublas_device_f32(const float* dA, const float* dB, float* dC,
                                 int M, int K, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                          N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasSgemm (device) failed, status=" + std::to_string(static_cast<int>(st)));
}

void gemm_cuda_cublas_tf32_device(const float* dA, const float* dB, float* dC,
                                  int M, int K, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha,
        dB, CUDA_R_32F, N,
        dA, CUDA_R_32F, K,
        &beta,
        dC, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasGemmEx (TF32, device) failed, status=" + std::to_string(static_cast<int>(st)));
}

// fp16-in/fp32-accumulate compute-only entry point -- TF32 (compute-only
// measured ~59 TFLOP/s on RTX 5080) is still well under the "100-200
// TFLOP/s" range asked about; dense FP16 Tensor Core throughput is
// roughly 2x TF32 on Ampere-and-later (TF32 occupies twice the bits per
// element, so half as many elements move through the tensor pipe per
// cycle), so this is the natural next data point. Reuses
// kernel_f32_to_f16 (defined above, used identically by the Level 7
// wgmma dispatch) for the one-time fp32->fp16 staging.
//
// Uses void* rather than __half* in every signature below (and in the
// cuda.hpp declarations) even though this file is happy to use __half
// internally: cuda.hpp is included by gemm_kernels_stub.cpp and by every
// host .cpp file (bench_gemm_cuda.cpp, test_gemm_cuda.cpp) that must
// still compile on a genuinely CPU-only machine with NO CUDA toolkit
// installed at all (see the build-cuda-stub CI job) -- <cuda_fp16.h>
// itself would not be found there, so __half cannot appear in a type
// that header exposes. float*/void*/int/size_t are all the public API
// above and below may safely use.
void gemm_cuda_convert_f32_to_f16_device(const float* src, void* dst, int count) {
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    kernel_f32_to_f16<<<blocks, threads>>>(src, static_cast<__half*>(dst), count);
    CUDA_CHECK(cudaGetLastError());
}

void gemm_cuda_cublas_fp16_device(const void* dA16, const void* dB16, float* dC,
                                  int M, int K, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha,
        dB16, CUDA_R_16F, N,
        dA16, CUDA_R_16F, K,
        &beta,
        dC, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasGemmEx (FP16, device) failed, status=" + std::to_string(static_cast<int>(st)));
}

// Matrix<float>-based wrapper (converts internally) -- for correctness
// testing only; the compute-only benchmark uses the raw-pointer entry
// points above directly, converting once outside the timed region.
void gemm_cuda_cublas_fp16(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    DeviceBuffer<float> dA(static_cast<std::size_t>(M) * K);
    DeviceBuffer<float> dB(static_cast<std::size_t>(K) * N);
    DeviceBuffer<float> dC(static_cast<std::size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(dA.ptr, A.data(), static_cast<std::size_t>(M) * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, B.data(), static_cast<std::size_t>(K) * N * sizeof(float), cudaMemcpyHostToDevice));

    DeviceBuffer<__half> dA16(static_cast<std::size_t>(M) * K);
    DeviceBuffer<__half> dB16(static_cast<std::size_t>(K) * N);
    gemm_cuda_convert_f32_to_f16_device(dA.ptr, dA16.ptr, M * K);
    gemm_cuda_convert_f32_to_f16_device(dB.ptr, dB16.ptr, K * N);

    gemm_cuda_cublas_fp16_device(dA16.ptr, dB16.ptr, dC.ptr, M, K, N);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC.ptr, static_cast<std::size_t>(M) * N * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Reference -- generic device-memory helpers for compute-only benchmarking
//
// Thin, toolkit-type-free wrappers (void*/size_t only, same reasoning as
// above) around cudaMalloc/cudaMemcpy/cudaFree/cudaDeviceSynchronize, so
// bench_gemm_cuda.cpp's compute-only benchmarks (BM_CudaCublas*ComputeOnly)
// can pre-stage device buffers without including <cuda_runtime.h> itself --
// that header isn't available on a CPU-only machine with no CUDA toolkit,
// and bench_gemm_cuda.cpp (like this whole file's public API) must still
// compile there against the stub library.
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
template void gemm_cuda_reordered<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_reordered<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
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

