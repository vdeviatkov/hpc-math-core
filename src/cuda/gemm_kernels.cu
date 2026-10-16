/**
 * @file gemm_kernels.cu
 * @brief The CUDA GEMM kernels, Levels 0-7, and the cuBLAS reference.
 *
 * The levels and their public entry points are listed in gemm/cuda.hpp.
 * Each kernel's comment below explains what it adds over the level it
 * builds on; the host launchers near the end of the file pick each level's
 * fallback when its hardware or shape requirement is not met. The GPU memory
 * hierarchy these kernels work against: docs/cache-behavior.md, section 9.
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
// Addressing convention
//
// Every kernel first moves A, B and C to the top-left corner of its block's
// tile, then works in tile-local coordinates. `rows` and `cols` are how much
// of the C tile lies inside the M x N matrix (at most the tile size), so a
// bounds check is a comparison against them, or against K - k0 along K.
// ============================================================================

// ============================================================================
// Level 0: Naive
//
// One thread computes one C(i,j), reading A and B straight from global
// memory. Adjacent threads of a warp read adjacent B[k][j], so the B loads
// coalesce, and L1/L2 absorb much of the reuse — which makes this baseline
// harder to beat than it looks.
// ============================================================================
static constexpr int kTile = 16;  // Levels 0 and 1: 16 x 16 threads per block

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
// Level 1: Shared-memory tiling, 16 x 16
//
// A 16x16 thread block stages each 16x16 k-tile of A and B in shared memory,
// so each global element is loaded once per block instead of once per
// thread: 16x fewer global loads. The shared-memory reads are a broadcast
// (As[ty][p]) and a contiguous row (Bs[p][tx]), so they are free of bank
// conflicts; the +1 padding is not needed for that. Each thread still owns
// one output, so the two __syncthreads() per k-tile pay for only 16 FMAs.
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
// Shared by Levels 2, 3, 5 and 6: tile loads
// ============================================================================

// Copy src[offset] to shared memory, or write zero when the element is
// outside the matrix. With Async on sm_80+ the copy is a cp.async, which
// proceeds while the thread moves on; its zfill form writes the zero, but
// still needs a valid source address, so out-of-range elements read
// src[0], the tile origin.
template <bool Async, typename D, typename S>
__device__ __forceinline__ void copy_or_zero(D* dst, const S* src, int offset, bool in_range) {
#ifdef HPC_HAVE_CP_ASYNC
    if constexpr (Async) {
        static_assert(std::is_same_v<D, S>, "cp.async copies bytes, it cannot convert");
        __pipeline_memcpy_async(dst, src + (in_range ? offset : 0), sizeof(D),
                                in_range ? 0 : sizeof(D));
        return;
    }
#endif
    *dst = static_cast<D>(in_range ? src[offset] : S{0});
}

// Load the ROWS x COLS tile whose top-left element is `src` (row-major,
// leading dimension ld) into shared memory at dst (leading dimension
// dst_ld), converting S to D and zero-filling outside the `rows` x `cols`
// part that lies inside the matrix. Consecutive threads take consecutive
// columns, so the global reads coalesce. Transposed stores element (r, c)
// at dst[c][r] instead of dst[r][c].
template <int ROWS, int COLS, bool Transposed, bool Async = false, typename D, typename S>
__device__ __forceinline__ void load_tile(D* dst, int dst_ld, const S* src, int ld,
                                          int rows, int cols) {
    for (int e = threadIdx.x; e < ROWS * COLS; e += blockDim.x) {
        const int r = e / COLS, c = e % COLS;
        const bool in_range = r < rows && c < cols;
        copy_or_zero<Async>(dst + (Transposed ? c * dst_ld + r : r * dst_ld + c), src,
                            r * ld + c, in_range);
    }
}

// ============================================================================
// Shared by Levels 2-4: the 8x8 register tile
//
// Each thread owns a kTM x kTN tile of C in registers. Per k step it loads
// kTM values of A and kTN values of B from shared memory, then accumulates
// their outer product: 64 FMAs from 16 shared-memory reads.
// ============================================================================
static constexpr int kBM = 128;  // block tile rows of C
static constexpr int kBN = 128;  // block tile columns of C
static constexpr int kBK = 16;   // k-step per shared-memory tile
static constexpr int kTM = 8;    // rows of C per thread
static constexpr int kTN = 8;    // columns of C per thread
// threads per block = (kBM / kTM) * (kBN / kTN) = 16 * 16 = 256

template <typename T>
__device__ __forceinline__ void fma_outer(T (&c)[kTM][kTN], const T (&a)[kTM], const T (&b)[kTN]) {
    #pragma unroll
    for (int m = 0; m < kTM; ++m)
        #pragma unroll
        for (int n = 0; n < kTN; ++n)
            c[m][n] += a[m] * b[n];
}

// One shared-memory k-tile into the register tile at (tm, tn) of the block
// tile: As is transposed (As[k][m]), Bs natural (Bs[k][n]).
template <typename T, int LDA, int LDB>
__device__ __forceinline__ void tile_fma(T (&acc)[kTM][kTN], const T (&As)[kBK][LDA],
                                         const T (&Bs)[kBK][LDB], int tm, int tn) {
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

// Write a thread's register tile to C at (tm, tn) of the block tile, clipped
// to the `rows` x `cols` part inside the matrix.
template <typename T>
__device__ __forceinline__ void store_tile(T* C, int ldc, const T (&acc)[kTM][kTN], int tm, int tn,
                                           int rows, int cols) {
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
template <typename T>
__global__ void __launch_bounds__(256)
kernel_reg_tile(const T* __restrict__ A,
                const T* __restrict__ B,
                T* __restrict__ C,
                int M, int K, int N) {
    __shared__ T As[kBK][kBM + 1];
    __shared__ T Bs[kBK][kBN + 1];

    const int row0 = blockIdx.y * kBM, col0 = blockIdx.x * kBN;
    A += row0 * K;
    B += col0;
    C += row0 * N + col0;
    const int rows = min(M - row0, kBM), cols = min(N - col0, kBN);
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
// Level 2 with two shared-memory buffers: tile k+1 is loaded while tile k is
// computed. On sm_80+ the loads use cp.async (__pipeline_memcpy_async /
// _commit / _wait_prior), which copies global -> shared without blocking the
// threads, so the copy overlaps the FMAs. On older GPUs the loads are
// synchronous: the structure is the same, without the overlap.
//
// Static shared memory is limited to 48 KB, so double uses a 64x64 tile:
//   f32, 128x128: 2 * 16 * (128+1 + 128+1) * 4 = 33,024 B
//   f64, 128x128: 2 * 16 * (128+1 + 128+1) * 8 = 66,048 B  (too large)
//   f64,   64x64: 2 * 16 * ( 64+1 +  64+1) * 8 = 33,280 B
// ============================================================================
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
    constexpr int BM = kDBufBM<T>, BN = kDBufBN<T>;
    __shared__ T As[2][kBK][BM + 1];
    __shared__ T Bs[2][kBK][BN + 1];

    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    A += row0 * K;
    B += col0;
    C += row0 * N + col0;
    const int rows = min(M - row0, BM), cols = min(N - col0, BN);
    const int tm = threadIdx.x / (BN / kTN) * kTM;
    const int tn = threadIdx.x % (BN / kTN) * kTN;

    // Start the copy of the k-tile at k0 into buffer buf.
    auto load = [&](int k0, int buf) {
        load_tile<BM, kBK, true, true>(&As[buf][0][0], BM + 1, A + k0, K, rows, K - k0);
        load_tile<kBK, BN, false, true>(&Bs[buf][0][0], BN + 1, B + k0 * N, N, K - k0, cols);
#ifdef HPC_HAVE_CP_ASYNC
        __pipeline_commit();
#endif
    };
    // Wait for the copy in flight, then make it visible to the whole block.
    auto wait = [] {
#ifdef HPC_HAVE_CP_ASYNC
        __pipeline_wait_prior(0);
#endif
        __syncthreads();
    };

    T acc[kTM][kTN] = {};
    load(0, 0);
    wait();
    for (int k0 = 0, cur = 0; k0 < K; k0 += kBK, cur ^= 1) {
        const bool more = k0 + kBK < K;
        if (more)
            load(k0 + kBK, cur ^ 1);
        tile_fma(acc, As[cur], Bs[cur], tm, tn);
        if (more)
            wait();
    }
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
//     long as K and N are too; launch_vectorized checks that and otherwise
//     runs kernel_reg_tile.
//
//  2. Shared memory uses an XOR swizzle instead of +1 padding to spread
//     accesses across banks without a wasted column (padding would also
//     break the alignment the vector stores need). swizzle() permutes which
//     vector slot of a row an element lands in; the same function is used at
//     every write and read, so correctness does not depend on the
//     permutation itself.
// ============================================================================

// 128-bit vector type: float4 for float, double2 for double.
template <typename T> struct VecTraits;
template <> struct VecTraits<float>  { using Vec = float4;  static constexpr int kWidth = 4; };
template <> struct VecTraits<double> { using Vec = double2; static constexpr int kWidth = 2; };

// Offset of logical column `col` within row `row` of a swizzled tile: the
// W-element vector slot is XORed with the row (SLOTS, the slots per row, is
// a power of two), the position inside the slot is kept. A column that
// starts a slot maps to the start of a slot, so vector stores stay aligned.
template <int W, int SLOTS>
__device__ __forceinline__ int swizzle(int row, int col) {
    return ((col / W) ^ (row & (SLOTS - 1))) * W + col % W;
}

// Read one aligned vector from shared memory into W consecutive registers.
template <typename Vec, typename T>
__device__ __forceinline__ void load_vec(T* dst, const T* src) {
    const Vec v = *reinterpret_cast<const Vec*>(src);
    const T* e  = reinterpret_cast<const T*>(&v);
    #pragma unroll
    for (int j = 0; j < int(sizeof(Vec) / sizeof(T)); ++j)
        dst[j] = e[j];
}

template <typename T>
__global__ void __launch_bounds__(256)
kernel_vectorized(const T* __restrict__ A,
                  const T* __restrict__ B,
                  T* __restrict__ C,
                  int M, int K, int N) {
    using Vec = typename VecTraits<T>::Vec;
    constexpr int W = VecTraits<T>::kWidth;
    constexpr int kSlots = kBM / W;  // vector slots per row; kBM == kBN

    __shared__ alignas(16) T As[kBK][kBM];  // As[k][swizzle(k, m)] = A(m, k)
    __shared__ alignas(16) T Bs[kBK][kBN];  // Bs[k][swizzle(k, n)] = B(k, n)

    const int row0 = blockIdx.y * kBM, col0 = blockIdx.x * kBN;
    A += row0 * K;
    B += col0;
    C += row0 * N + col0;
    const int rows = min(M - row0, kBM), cols = min(N - col0, kBN);
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
// Levels 5-7 run on Tensor Cores and are float-only: fp32 in and out, fp16
// multiply, fp32 accumulate.
// ============================================================================
static constexpr int kWMMA_M = 16;  // the WMMA fragment shape, fixed by the API
static constexpr int kWMMA_N = 16;
static constexpr int kWMMA_K = 16;

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
static constexpr int kWarpM  = 4;                  // warps along M
static constexpr int kWarpN  = 4;                  // warps along N
static constexpr int kBlockM = kWarpM * kWMMA_M;   // 64
static constexpr int kBlockN = kWarpN * kWMMA_N;   // 64
static constexpr int kBlockK = kWMMA_K;            // 16

// (512, 3): three 512-thread blocks fill an SM. Without the minimum the
// compiler may use up to 64 registers per thread, which fits only two
// blocks; the kernel needs 40.
__global__ void __launch_bounds__(512, 3)
kernel_wmma(const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            int M, int K, int N) {
    // No +1 padding: wmma::load_matrix_sync requires the leading dimension of
    // a __half fragment to be a multiple of 8 elements (16 bytes). 64 is; 65
    // would fail with cudaErrorMisalignedAddress.
    __shared__ __half As[kBlockK][kBlockM];  // transposed: As[k][m] = A(m, k)
    __shared__ __half Bs[kBlockK][kBlockN];  // natural:    Bs[k][n] = B(k, n)

    const int row0 = blockIdx.y * kBlockM, col0 = blockIdx.x * kBlockN;
    A += row0 * K;
    B += col0;
    C += row0 * N + col0;
    const int rows = min(M - row0, kBlockM), cols = min(N - col0, kBlockN);
    const int warp = threadIdx.x / 32;
    const int wm = warp / kWarpN * kWMMA_M;  // this warp's 16x16 tile
    const int wn = warp % kWarpN * kWMMA_N;
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
    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K, __half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += kBlockK) {
        load_tile<kBlockM, kBlockK, true>(&As[0][0], kBlockM, A + k0, K, rows, K - k0);
        load_tile<kBlockK, kBlockN, false>(&Bs[0][0], kBlockN, B + k0 * N, N, K - k0, cols);
        __syncthreads();
        if (active) {
            wmma::load_matrix_sync(a_frag, &As[0][wm], kBlockM);
            wmma::load_matrix_sync(b_frag, &Bs[0][wn], kBlockN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    // store_matrix_sync always writes a full 16x16 tile and needs a
    // 32-byte-aligned destination, so a direct store is only safe for a tile
    // wholly inside C with N % 8 == 0. Any other tile goes through a per-warp
    // shared-memory staging tile and is copied out element by element.
    __shared__ float Cs[kWarpM * kWarpN][kWMMA_M * kWMMA_N];
    if (!active)
        return;
    if (wm + kWMMA_M <= rows && wn + kWMMA_N <= cols && N % 8 == 0) {
        wmma::store_matrix_sync(C + wm * N + wn, c_frag, N, wmma::mem_row_major);
    } else {
        float* tile = Cs[warp];
        wmma::store_matrix_sync(tile, c_frag, kWMMA_N, wmma::mem_row_major);
        __syncwarp();
        for (int e = threadIdx.x % 32; e < kWMMA_M * kWMMA_N; e += 32) {
            const int r = wm + e / kWMMA_N, c = wn + e % kWMMA_N;
            if (r < rows && c < cols)
                C[r * N + c] = tile[e];
        }
    }
}

// ============================================================================
// Level 6: Raw Tensor Core MMA via mma.sync + ldmatrix -- sm_80+
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
//     transpose — unlike kernel_wmma and the FMA kernels, which store As as
//     [k][m].
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
// sm_70-75 have no f16 m16n8k16 shape; launch_mma_ldmatrix falls back to
// kernel_wmma there.
// ============================================================================
static constexpr int kMmaM = 16;  // mma.sync m16n8k16 native shape
static constexpr int kMmaN = 8;
static constexpr int kMmaK = 16;

// 4x4 warps per block; each warp owns two side-by-side 16x8 tiles (16x16
// total), kernel_wmma's per-warp output area, for a like-for-like comparison.
static constexpr int kMmaWarpM  = 4;
static constexpr int kMmaWarpN  = 4;
static constexpr int kMmaBlockM = kMmaWarpM * kMmaM;      // 64
static constexpr int kMmaBlockN = kMmaWarpN * kMmaN * 2;  // 64
static constexpr int kMmaBlockK = kMmaK;                  // 16

__global__ void __launch_bounds__(512)
kernel_mma_ldmatrix(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    int M, int K, int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __half As[kMmaBlockM][kMmaBlockK];  // natural: As[m][k], matches `.row`
    __shared__ __half Bs[kMmaBlockK][kMmaBlockN];  // natural: Bs[k][n], needs `.trans`

    const int row0 = blockIdx.y * kMmaBlockM, col0 = blockIdx.x * kMmaBlockN;
    A += row0 * K;
    B += col0;
    C += row0 * N + col0;
    const int rows = min(M - row0, kMmaBlockM), cols = min(N - col0, kMmaBlockN);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / kMmaWarpN * kMmaM;        // this warp's 16x16 tile
    const int wn = warp % kMmaWarpN * kMmaN * 2;

    float acc[2][4] = {};  // two 16x8 tiles, 4 fp32 registers per lane each

    for (int k0 = 0; k0 < K; k0 += kMmaBlockK) {
        load_tile<kMmaBlockM, kMmaBlockK, false>(&As[0][0], kMmaBlockK, A + k0, K, rows, K - k0);
        load_tile<kMmaBlockK, kMmaBlockN, false>(&Bs[0][0], kMmaBlockN, B + k0 * N, N, K - k0, cols);
        __syncthreads();

        if (wm < rows) {
            // A: the 16(M) x 16(K) tile as four 8x8 quadrants, ldmatrix.x4.
            // mma.sync expects them as (M0-7,K0-7), (M8-15,K0-7),
            // (M0-7,K8-15), (M8-15,K8-15): lane group q = lane/8 supplies
            // rows of quadrant q, M half q%2, K half q/2.
            const int q = lane / 8;
            const __half* a_addr = &As[wm + lane % 8 + q % 2 * 8][q / 2 * 8];
            unsigned a_frag[4];
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "l"(__cvta_generic_to_shared(a_addr)));

            // B: two 16(K) x 8(N) tiles, ldmatrix.x2 + .trans. Lanes 0-15
            // address rows K = 0..15; lanes 16-31 repeat them.
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                const __half* b_addr = &Bs[lane % 16][wn + t * kMmaN];
                unsigned b_frag[2];
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[0]), "=r"(b_frag[1])
                    : "l"(__cvta_generic_to_shared(b_addr)));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[t][0]), "+f"(acc[t][1]), "+f"(acc[t][2]), "+f"(acc[t][3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1]));
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
            const int c = wn + t * kMmaN + lane % 4 * 2 + i % 2;
            if (r < rows && c < cols)
                C[r * N + c] = acc[t][i];
        }
#else
    // Never launched below sm_80 (launch_mma_ldmatrix checks
    // cuda_has_ampere()); this branch only keeps the file compiling for
    // older -arch targets.
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)N;
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
//      per shared-memory round trip and per __syncthreads() pair.
//   2. Bigger warp tile: each of 8 warps owns 32x64 = 2x4 fragments and issues
//      8 mma_sync per k-sub-step. Each b_frag is loaded once and reused for
//      both fm, each a_frag once for all 4 fn — the register blocking of
//      kernel_reg_tile, at fragment granularity.
//   3. Double buffering with cp.async (sm_80+): the next k-tile's copy
//      overlaps the current tile's MMAs, with kernel_double_buf's control
//      flow. Pre-Ampere it falls back to a synchronous 16-byte copy.
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
static constexpr int kPipeWarpCols = kPipeBN / kPipeWarpN;                        // 2
static constexpr int kPipeNumWarps = (kPipeBM / kPipeWarpM) * kPipeWarpCols;      // 8 (256 threads)
static constexpr int kPipeFragM = kPipeWarpM / kWMMA_M;                           // 2
static constexpr int kPipeFragN = kPipeWarpN / kWMMA_N;                           // 4

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
static constexpr int kPipeAsLd = kPipeBK + 8;   // 40 halves =  80 B
static constexpr int kPipeBsLd = kPipeBN + 8;   // 136 halves = 272 B

// Copy the ROWS x COLS fp16 tile at src (leading dimension ld) to dst
// (leading dimension dst_ld) in 16-byte chunks of 8 halves: cp.async on
// sm_80+, a plain 16-byte copy before. No bounds checks (see above).
template <int ROWS, int COLS>
__device__ __forceinline__ void copy_tile_16B(__half* dst, int dst_ld, const __half* src, int ld) {
    constexpr int kChunksPerRow = COLS / 8;
    for (int i = threadIdx.x; i < ROWS * kChunksPerRow; i += blockDim.x) {
        const int r = i / kChunksPerRow, c = i % kChunksPerRow * 8;
#ifdef HPC_HAVE_CP_ASYNC
        __pipeline_memcpy_async(dst + r * dst_ld + c, src + static_cast<std::size_t>(r) * ld + c, 16);
#else
        *reinterpret_cast<float4*>(dst + r * dst_ld + c) =
            *reinterpret_cast<const float4*>(src + static_cast<std::size_t>(r) * ld + c);
#endif
    }
}

__global__ void __launch_bounds__(kPipeNumWarps * 32)
kernel_wmma_pipelined(const __half* __restrict__ A16,   // MxK, row-major, fp16
                      const __half* __restrict__ B16,   // KxN, row-major, fp16
                      float* __restrict__ C,
                      int M, int K, int N) {
    __shared__ alignas(16) __half As[2][kPipeBM][kPipeAsLd];  // natural: As[m][k], padded ld
    __shared__ alignas(16) __half Bs[2][kPipeBK][kPipeBsLd];  // natural: Bs[k][n], padded ld

    // M, N and K are exact multiples of the tiles: no `rows`/`cols` clipping.
    const int row0 = blockIdx.y * kPipeBM, col0 = blockIdx.x * kPipeBN;
    A16 += static_cast<std::size_t>(row0) * K;
    B16 += col0;
    C += static_cast<std::size_t>(row0) * N + col0;
    const int warp = threadIdx.x / 32;
    const int wm = warp / kPipeWarpCols * kPipeWarpM;  // this warp's 32x64 tile
    const int wn = warp % kPipeWarpCols * kPipeWarpN;

    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K, float> c_frag[kPipeFragM][kPipeFragN];
    #pragma unroll
    for (int fm = 0; fm < kPipeFragM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < kPipeFragN; ++fn)
            wmma::fill_fragment(c_frag[fm][fn], 0.0f);

    auto load = [&](int k0, int buf) {
        copy_tile_16B<kPipeBM, kPipeBK>(&As[buf][0][0], kPipeAsLd, A16 + k0, K);
        copy_tile_16B<kPipeBK, kPipeBN>(&Bs[buf][0][0], kPipeBsLd,
                                        B16 + static_cast<std::size_t>(k0) * N, N);
#ifdef HPC_HAVE_CP_ASYNC
        __pipeline_commit();
#endif
    };
    auto wait = [] {
#ifdef HPC_HAVE_CP_ASYNC
        __pipeline_wait_prior(0);
#endif
        __syncthreads();
    };

    load(0, 0);
    wait();
    for (int k0 = 0, cur = 0; k0 < K; k0 += kPipeBK, cur ^= 1) {
        const bool more = k0 + kPipeBK < K;
        if (more)
            load(k0 + kPipeBK, cur ^ 1);

        #pragma unroll
        for (int ks = 0; ks < kPipeBK; ks += kWMMA_K) {
            // Load each fn's b_frag once per k-sub-step and reuse it for every
            // fm below: the register blocking of kernel_reg_tile.
            wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K, __half, wmma::row_major> b_frag[kPipeFragN];
            #pragma unroll
            for (int fn = 0; fn < kPipeFragN; ++fn)
                wmma::load_matrix_sync(b_frag[fn], &Bs[cur][ks][wn + fn * kWMMA_N], kPipeBsLd);

            #pragma unroll
            for (int fm = 0; fm < kPipeFragM; ++fm) {
                wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K, __half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, &As[cur][wm + fm * kWMMA_M][ks], kPipeAsLd);
                #pragma unroll
                for (int fn = 0; fn < kPipeFragN; ++fn)
                    wmma::mma_sync(c_frag[fm][fn], a_frag, b_frag[fn], c_frag[fm][fn]);
            }
        }

        if (more)
            wait();
    }

    #pragma unroll
    for (int fm = 0; fm < kPipeFragM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < kPipeFragN; ++fn)
            wmma::store_matrix_sync(C + (wm + fm * kWMMA_M) * N + wn + fn * kWMMA_N,
                                    c_frag[fm][fn], N, wmma::mem_row_major);
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

// Queried once: every launch checks these, and a property query costs far
// more than a launch.
bool cuda_has_tensor_cores() noexcept {
    static const bool has = device_has_capability(7, 0);
    return has;
}
bool cuda_has_ampere() noexcept {
    static const bool has = device_has_capability(8, 0);
    return has;
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
// Device entry points -- one per level, on device pointers (see cuda.hpp).
// Each falls back to a level below it when its hardware or shape requirement
// isn't met, and only launches: the work runs asynchronously.
// ============================================================================

template <typename T>
void gemm_cuda_naive_device(const T* A, const T* B, T* C, int M, int K, int N) {
    const dim3 block(kTile, kTile), grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
    kernel_naive<T><<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void gemm_cuda_blocked_device(const T* A, const T* B, T* C, int M, int K, int N) {
    const dim3 block(kTile, kTile), grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
    kernel_blocked<T><<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void gemm_cuda_reg_tile_device(const T* A, const T* B, T* C, int M, int K, int N) {
    const dim3 block(256), grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
    kernel_reg_tile<T><<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void gemm_cuda_double_buf_device(const T* A, const T* B, T* C, int M, int K, int N) {
    constexpr int BM = kDBufBM<T>, BN = kDBufBN<T>;
    // One thread per kTM x kTN sub-tile: 256 threads for float
    // (128x128 tile), 64 for double (64x64 tile -- see kDBufBM).
    const dim3 block((BM / kTM) * (BN / kTN)), grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    kernel_double_buf<T><<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// Level 4: vectorized loads require K and N to be multiples of the 128-bit
// vector width (4 for float, 2 for double); otherwise fall back to Level 2.
template <typename T>
void gemm_cuda_vectorized_device(const T* A, const T* B, T* C, int M, int K, int N) {
    constexpr int W = VecTraits<T>::kWidth;
    if (K % W != 0 || N % W != 0)
        return gemm_cuda_reg_tile_device(A, B, C, M, K, N);
    const dim3 block(256), grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
    kernel_vectorized<T><<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// Level 5: Tensor Cores need sm_70+; otherwise fall back to Level 3.
void gemm_cuda_wmma_device(const float* A, const float* B, float* C, int M, int K, int N) {
    if (!cuda_has_tensor_cores())
        return gemm_cuda_double_buf_device(A, B, C, M, K, N);
    const dim3 block(kWarpM * kWarpN * 32);  // 512 threads
    const dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);
    kernel_wmma<<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// Level 6: the f16 m16n8k16 mma.sync shape needs sm_80+; otherwise fall back
// to Level 5.
void gemm_cuda_mma_ldmatrix_device(const float* A, const float* B, float* C, int M, int K, int N) {
    if (!cuda_has_ampere())
        return gemm_cuda_wmma_device(A, B, C, M, K, N);
    const dim3 block(kMmaWarpM * kMmaWarpN * 32);  // 512 threads
    const dim3 grid((N + kMmaBlockN - 1) / kMmaBlockN, (M + kMmaBlockM - 1) / kMmaBlockM);
    kernel_mma_ldmatrix<<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// Level 7 on fp16 inputs. No fallback: M and N must be multiples of 128 and
// K a multiple of 32.
void gemm_cuda_wmma_pipelined_device(const void* A16, const void* B16, float* C,
                                     int M, int K, int N) {
    const dim3 block(kPipeNumWarps * 32), grid(N / kPipeBN, M / kPipeBM);
    kernel_wmma_pipelined<<<grid, block>>>(static_cast<const __half*>(A16),
                                           static_cast<const __half*>(B16), C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// fp32 -> fp16 conversion of a device buffer (cp.async, used by Level 7, is a
// same-dtype byte copy and can't convert while loading).
void gemm_cuda_convert_f32_to_f16_device(const float* src, void* dst, int count) {
    const int threads = 256;
    kernel_f32_to_f16<<<(count + threads - 1) / threads, threads>>>(
        src, static_cast<__half*>(dst), count);
    CUDA_CHECK(cudaGetLastError());
}

// Level 7 on fp32 inputs, for the Matrix-based entry point: converts A and B
// to fp16 first. Requires sm_70+ (the cp.async overlap needs sm_80+, see
// copy_tile_16B), M/N exact multiples of 128 and K an exact multiple of 32
// (no tail handling); otherwise falls back to Level 5.
static void wmma_pipelined_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    const bool exactTiles = (M % kPipeBM == 0) && (N % kPipeBN == 0) && (K % kPipeBK == 0);
    if (!cuda_has_tensor_cores() || !exactTiles)
        return gemm_cuda_wmma_device(A, B, C, M, K, N);
    DeviceBuffer<__half> A16(static_cast<std::size_t>(M) * K);
    DeviceBuffer<__half> B16(static_cast<std::size_t>(K) * N);
    gemm_cuda_convert_f32_to_f16_device(A, A16.ptr, M * K);
    gemm_cuda_convert_f32_to_f16_device(B, B16.ptr, K * N);
    gemm_cuda_wmma_pipelined_device(A16.ptr, B16.ptr, C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());  // before A16/B16 are freed
}

double gemm_cuda_time_ms(const std::function<void()>& launch) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Copy A and B to the device, run `launch` on device pointers, copy C back.
// Every Matrix<T>-based entry point goes through this.
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
// Public API, Matrix-based
// ============================================================================

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
// The Tensor Core levels are float-only: fp32 in/out, fp16 compute.
void gemm_cuda_wmma(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, gemm_cuda_wmma_device);
}
void gemm_cuda_mma_ldmatrix(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, gemm_cuda_mma_ldmatrix_device);
}
void gemm_cuda_wmma_pipelined(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, wmma_pipelined_f32);
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
void gemm_cuda_cublas_device(const T* A, const T* B, T* C, int M, int K, int N) {
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
    run_on_device(A, B, C, gemm_cuda_cublas_device<T>);
}

void gemm_cuda_cublas_tf32(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, gemm_cuda_cublas_tf32_device);
}

// Matrix<float>-based wrapper (converts internally) -- for correctness
// testing; the benchmark uses the device entry points below directly,
// converting once outside the timed region.
void gemm_cuda_cublas_fp16(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    run_on_device(A, B, C, [](const float* dA, const float* dB, float* dC, int M, int K, int N) {
        DeviceBuffer<__half> A16(static_cast<std::size_t>(M) * K);
        DeviceBuffer<__half> B16(static_cast<std::size_t>(K) * N);
        gemm_cuda_convert_f32_to_f16_device(dA, A16.ptr, M * K);
        gemm_cuda_convert_f32_to_f16_device(dB, B16.ptr, K * N);
        gemm_cuda_cublas_fp16_device(A16.ptr, B16.ptr, dC, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());  // before A16/B16 are freed
    });
}

// ============================================================================
// Reference -- cuBLAS, device entry points (see cuda.hpp). Same calls as the
// Matrix-based wrappers above, on caller-provided device buffers; the
// wrappers' tests cover their correctness.
// ============================================================================
void gemm_cuda_cublas_tf32_device(const float* dA, const float* dB, float* dC,
                                  int M, int K, int N) {
    cublas_gemm_ex(dA, dB, dC, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, M, K, N,
                   "cublasGemmEx (TF32)");
}

// fp16 in, fp32 accumulate. fp16 buffers are void* in the public API
// because cuda.hpp must compile without <cuda_fp16.h> (CPU-only stub builds).
void gemm_cuda_cublas_fp16_device(const void* dA16, const void* dB16, float* dC,
                                  int M, int K, int N) {
    cublas_gemm_ex(dA16, dB16, dC, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, M, K, N,
                   "cublasGemmEx (FP16)");
}

// ============================================================================
// Device-memory helpers for the benchmarks (see cuda.hpp).
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
#define HPC_CUDA_DEVICE_INSTANTIATE(fn)                                        \
    template void fn<float>(const float*, const float*, float*, int, int, int); \
    template void fn<double>(const double*, const double*, double*, int, int, int);
HPC_CUDA_DEVICE_INSTANTIATE(gemm_cuda_naive_device)
HPC_CUDA_DEVICE_INSTANTIATE(gemm_cuda_blocked_device)
HPC_CUDA_DEVICE_INSTANTIATE(gemm_cuda_reg_tile_device)
HPC_CUDA_DEVICE_INSTANTIATE(gemm_cuda_double_buf_device)
HPC_CUDA_DEVICE_INSTANTIATE(gemm_cuda_vectorized_device)
HPC_CUDA_DEVICE_INSTANTIATE(gemm_cuda_cublas_device)
#undef HPC_CUDA_DEVICE_INSTANTIATE

}  // namespace hpc::gemm

