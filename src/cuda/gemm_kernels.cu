/**
 * @file gemm_kernels.cu
 * @brief CUDA GEMM kernel implementations: naive, reordered, tiled-blocked.
 *
 * Compiled by nvcc (or clang with CUDA support).  Produces a normal object
 * file that is linked into the benchmark / test binaries.
 *
 * ============================================================
 *  GPU memory hierarchy reminder
 * ============================================================
 *
 *  Global memory   : off-chip DRAM, ~400–600 GB/s (A100), ~1 TB/s (H100).
 *                    High latency (~200–800 cycles).  All threads can access.
 *                    Coalesced access (warp reads 128 contiguous bytes) is
 *                    essential for peak bandwidth.
 *
 *  Shared memory   : on-chip SRAM, ~10–20 TB/s bandwidth, ~4–32 cycles.
 *                    Visible to all threads in a block.  Organised in 32
 *                    banks (4-byte interleaved).  Parallel access to the
 *                    same bank by threads in the same warp causes a "bank
 *                    conflict" — accesses are serialised.
 *
 *  Registers       : per-thread, ~1 cycle access.  spill to global memory
 *                    if a kernel uses too many (register pressure).
 *
 *  L1/L2 cache     : automatic, ~50 GB/s–2 TB/s effective.  Shared memory
 *                    occupies part of the same on-chip SRAM as L1 (configurable).
 *
 *
 * ============================================================
 *  Warp coalescence rules
 * ============================================================
 *
 *  A warp = 32 consecutive threads (threadIdx.x = 0..31 in the same block).
 *  When a warp issues a load/store, the hardware coalesces it into as few
 *  128-byte transactions as possible:
 *
 *    • Ideal:  warp loads A[row][0..31] → 1 transaction (32×4 = 128 bytes).
 *    • Bad:    warp loads A[0..31][col] → 32 separate transactions
 *              (stride = N elements apart, each touching a different cache line).
 *
 *  In our kernels: thread (ty, tx) computes C(i=blockRow*TILE+ty, j=blockCol*TILE+tx).
 *  All threads in the same warp have the same ty and differ in tx (0..TILE-1=15).
 *  Reading B[k][j]: j varies linearly → coalesced ✓
 *  Reading A[i][k]: all threads have the same i, same k → scalar broadcast ✓ (or cached)
 *
 *
 * ============================================================
 *  Tiled GEMM algorithm (gemm_cuda_blocked)
 * ============================================================
 *
 *  Outer loop over k-tiles of size TILE:
 *    1. All TILE×TILE threads cooperatively load:
 *         As[ty][tx] = A[i][kTile*TILE + tx]   // TILE×TILE tile of A
 *         Bs[ty][tx] = B[kTile*TILE + ty][j]   // TILE×TILE tile of B
 *       into shared memory.
 *    2. __syncthreads() — ensure the tile is fully loaded.
 *    3. Each thread accumulates: acc += As[ty][p] * Bs[p][tx], p=0..TILE-1.
 *    4. __syncthreads() — ensure all threads finished reading before next tile.
 *  End.  C[i][j] = acc.
 *
 *  Bank conflict avoidance:
 *    Bs[TILE][TILE+1] — the extra column pads each row to a non-power-of-2
 *    stride, so Bs[0][tx], Bs[1][tx], … map to different banks.
 *    As[TILE][TILE+1] — same padding for As column accesses in step 3.
 */

#include "gemm/cuda.hpp"
#include "hpc/matrix.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// Compile-time tile size.  16×16 = 256 threads per block — a common sweet
// spot: enough threads to hide latency, fits in shared memory budget.
// ---------------------------------------------------------------------------
static constexpr int kTile = 16;

// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error at " __FILE__     \
                                                 ":" + std::to_string(__LINE__) + \
                                                 ": ") +                       \
                                     cudaGetErrorString(_e));                  \
        }                                                                      \
    } while (0)

// ============================================================================
// Kernel 1: Naive — one thread computes one C(i,j), no shared memory
// ============================================================================

/**
 * @brief Naive GEMM kernel.
 *
 * Thread (tx, ty) in block (bx, by) computes:
 *   i = by * blockDim.y + ty
 *   j = bx * blockDim.x + tx
 *   C[i][j] = Σ_k A[i][k] * B[k][j]
 *
 * Memory access:
 *   A[i][k]: all k values for fixed i  → sequential row scan  (coalesced within block)
 *   B[k][j]: all k values for fixed j  → column scan (stride N) → NOT coalesced.
 *   C[i][j]: one write per thread      → coalesced.
 *
 * This is the most straightforward mapping: no optimisation, pure baseline.
 */
template <typename T>
__global__ void kernel_naive(const T* __restrict__ A,
                              const T* __restrict__ B,
                              T* __restrict__ C,
                              int M, int K, int N) {
    const int i = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= M || j >= N) return;

    T acc = T{0};
    for (int k = 0; k < K; ++k)
        acc += A[i * K + k] * B[k * N + j];
    C[i * N + j] = acc;
}

// ============================================================================
// Kernel 2: Reordered — same one-thread mapping, explicit row-major read.
// ============================================================================

/**
 * @brief Reordered GEMM kernel.
 *
 * Structurally identical to naive on GPU; included for naming symmetry with
 * the CPU kernels.  The thread mapping and memory access are the same.
 * On modern GPUs with large L2 (A100: 40 MB; RTX 4090: 72 MB), the difference
 * between naive and reordered at moderate sizes is negligible because the
 * repeated B column accesses land in L2.
 *
 * This kernel explicitly computes the dot-product with A and B in row-major
 * order (walking k innermost), which is what the CPU i-k-j reordering achieves.
 * On GPU this is the natural order for the inner product formulation.
 */
template <typename T>
__global__ void kernel_reordered(const T* __restrict__ A,
                                  const T* __restrict__ B,
                                  T* __restrict__ C,
                                  int M, int K, int N) {
    const int i = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= M || j >= N) return;

    // Walk k in the inner loop — same as naive, but we make it explicit
    // that both A(i,k) and B(k,j) are indexed with k varying fastest.
    T acc = T{0};
    const T* a_row = A + i * K;   // pointer to start of A row i
    for (int k = 0; k < K; ++k)
        acc += a_row[k] * B[k * N + j];
    C[i * N + j] = acc;
}

// ============================================================================
// Kernel 3: Blocked / Tiled — shared memory sub-tiles
// ============================================================================

/**
 * @brief Tiled GEMM kernel using TILE×TILE shared-memory sub-tiles.
 *
 * Each thread block of TILE×TILE threads computes a TILE×TILE sub-tile of C.
 *
 * Shared memory buffers (with +1 padding to avoid bank conflicts):
 *   As[kTile][kTile+1]  — sub-tile of A: rows i..i+TILE, cols kBlk..kBlk+TILE
 *   Bs[kTile][kTile+1]  — sub-tile of B: rows kBlk..kBlk+TILE, cols j..j+TILE
 *
 * Each thread loads one element of As and one of Bs per k-tile iteration.
 * The inner loop over p=0..TILE-1 runs entirely from shared memory (~4 cycles/access).
 *
 * Boundary handling:
 *   Out-of-bounds threads load 0.0 into shared memory to avoid UB.
 *   C is only written when i < M && j < N.
 */
template <typename T>
__global__ void kernel_blocked(const T* __restrict__ A,
                                const T* __restrict__ B,
                                T* __restrict__ C,
                                int M, int K, int N) {
    // Padded shared memory to avoid bank conflicts on column access.
    __shared__ T As[kTile][kTile + 1];
    __shared__ T Bs[kTile][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i  = static_cast<int>(blockIdx.y) * kTile + ty;  // global row
    const int j  = static_cast<int>(blockIdx.x) * kTile + tx;  // global col

    T acc = T{0};

    // Iterate over k-tiles.
    const int nTilesK = (K + kTile - 1) / kTile;
    for (int tileK = 0; tileK < nTilesK; ++tileK) {
        // Column/row index within this k-tile.
        const int kA = tileK * kTile + tx;  // A column for this thread
        const int kB = tileK * kTile + ty;  // B row for this thread

        // Cooperatively load A sub-tile: A[i][kA]
        As[ty][tx] = (i < M && kA < K) ? A[i * K + kA] : T{0};
        // Cooperatively load B sub-tile: B[kB][j]
        Bs[ty][tx] = (kB < K && j < N) ? B[kB * N + j] : T{0};

        __syncthreads();  // Ensure full tile is in shared memory.

        // Accumulate partial dot-product from shared memory.
        #pragma unroll
        for (int p = 0; p < kTile; ++p)
            acc += As[ty][p] * Bs[p][tx];

        __syncthreads();  // Ensure all threads finished reading before next tile.
    }

    if (i < M && j < N)
        C[i * N + j] = acc;
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

// ============================================================================
// RAII helper for device memory
// ============================================================================

template <typename T>
struct DeviceBuffer {
    T*           ptr  = nullptr;
    std::size_t  size = 0;  // in elements

    explicit DeviceBuffer(std::size_t n) : size(n) {
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) noexcept        = delete;
};

// ============================================================================
// Generic launcher: alloc, copy, launch, sync, copy back
// ============================================================================

enum class GemmKind { Naive, Reordered, Blocked };

template <typename T>
static void launch(GemmKind kind, const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    // Allocate and upload.
    DeviceBuffer<T> dA(M * K), dB(K * N), dC(M * N);
    CUDA_CHECK(cudaMemcpy(dA.ptr, A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC.ptr, 0, M * N * sizeof(T)));

    // Grid and block dimensions.
    const dim3 block(kTile, kTile);
    const dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);

    if (kind == GemmKind::Naive)
        kernel_naive<T><<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);
    else if (kind == GemmKind::Reordered)
        kernel_reordered<T><<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);
    else
        kernel_blocked<T><<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, K, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back.
    CUDA_CHECK(cudaMemcpy(C.data(), dC.ptr, M * N * sizeof(T), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Public API — template definitions
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

// ============================================================================
// Explicit instantiations
// ============================================================================

template void gemm_cuda_naive<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_naive<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);

template void gemm_cuda_reordered<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_reordered<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);

template void gemm_cuda_blocked<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_cuda_blocked<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);

}  // namespace hpc::gemm

