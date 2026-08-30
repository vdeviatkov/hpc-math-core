/**
 * @file bench_gemm_cuda.cpp
 * @brief Google Benchmark driver for CUDA GEMM kernels (Levels 0-8, plus a
 *        cuBLAS reference).
 *
 * Levels:
 *   CudaNaive         -- Level 0: global memory only
 *   CudaReordered     -- Level 0b: CPU-symmetry baseline
 *   CudaBlocked       -- Level 1: TILE=16 shared-memory tiling
 *   CudaRegTile       -- Level 2: 128x128 block, 8x8 register tile per thread
 *   CudaDoubleBuf     -- Level 3: Level 2 + double buffering (cp.async on Ampere+)
 *   CudaWmma          -- Level 4: Tensor Cores via WMMA (fp32 only, sm_70+)
 *   CudaVectorized    -- Level 5: float4/double2 loads + shared-memory XOR swizzle
 *   CudaMmaLdmatrix   -- Level 6: raw Tensor Cores via mma.sync+ldmatrix (fp32 only, sm_80+)
 *   CudaHopperWgmma   -- Level 7: warp specialization + TMA (fp32 only, sm_90a; unverified)
 *   CudaWmmaPipelined -- Level 8: WMMA, 128x128 tiles + cp.async double buffering
 *                        (fp32 only, sm_70+; NEW, added after the cuBLAS reference
 *                        below measured this GPU's real Tensor Core ceiling)
 *   CudaCublas        -- Reference: cuBLAS SGEMM/DGEMM, ceiling for the FMA kernels above
 *   CudaCublasTf32    -- Reference: cuBLAS TF32 Tensor Cores (fp32 only, sm_80+),
 *                        ceiling for the Tensor Core kernels above
 *   CudaCublas*ComputeOnly -- same two cuBLAS kernels, but timing ONLY the
 *                        GEMM call against pre-staged device buffers (no
 *                        per-iteration cudaMalloc/H2D/D2H) -- see
 *                        gemm_kernels.cu's "raw-device-pointer entry
 *                        points" comment for why every OTHER benchmark
 *                        here (including CudaCublas/CudaCublasTf32 above)
 *                        badly understates achievable throughput at large N.
 *
 * Runtime guards:
 *   All kernels check cuda_device_count() > 0 -> SKIPPED on CPU-only machines.
 *   CudaWmma additionally checks cuda_has_tensor_cores() -> SKIPPED on pre-Volta.
 *   CudaMmaLdmatrix/CudaCublasTf32(*) additionally check cuda_has_ampere() -> SKIPPED pre-Ampere.
 *   CudaHopperWgmma additionally checks cuda_has_hopper() -> SKIPPED on non-Hopper.
 *   CudaDoubleBuf reports whether cp.async (Ampere+) is active.
 */

#include "gemm/cuda.hpp"
#include "hpc/matrix.hpp"

#include <benchmark/benchmark.h>

// Deliberately NOT <cuda_runtime.h> -- this file must still compile on a
// genuinely CPU-only machine with no CUDA toolkit at all (build-cuda-stub
// CI), so it only ever touches device memory through gemm_cuda_malloc/
// _free/_memcpy_h2d/_device_synchronize (declared in gemm/cuda.hpp,
// toolkit-type-free by design -- see that header's comment).
#include <cstddef>
#include <random>
#include <type_traits>

template <typename T>
static void fill_random(hpc::Matrix<T>& M, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(T{-1}, T{1});
    for (std::size_t i = 0; i < M.rows(); ++i)
        for (std::size_t j = 0; j < M.cols(); ++j)
            M(i, j) = dist(rng);
}

static inline double flops(std::size_t N) {
    return 2.0 * double(N) * double(N) * double(N);
}

// ---------------------------------------------------------------------------
// Level 0 -- Naive
// ---------------------------------------------------------------------------
template <std::size_t N, typename T = double>
static void BM_CudaNaive(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_naive(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = double(sizeof(T) * 8);
}

// ---------------------------------------------------------------------------
// Level 0b -- Reordered
// ---------------------------------------------------------------------------
template <std::size_t N, typename T = double>
static void BM_CudaReordered(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_reordered(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = double(sizeof(T) * 8);
}

// ---------------------------------------------------------------------------
// Level 1 -- Blocked (TILE=16)
// ---------------------------------------------------------------------------
template <std::size_t N, typename T = double>
static void BM_CudaBlocked(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_blocked(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = double(sizeof(T) * 8);
    state.counters["tile"] = 16;
}

// ---------------------------------------------------------------------------
// Level 2 -- Register tile (128x128 block, 8x8 per thread)
// ---------------------------------------------------------------------------
template <std::size_t N, typename T = double>
static void BM_CudaRegTile(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_reg_tile(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = double(sizeof(T) * 8);
    state.counters["block"] = 128;
    state.counters["reg_tile"] = 64;  // 8x8
}

// ---------------------------------------------------------------------------
// Level 3 -- Double-buffered register tile
// ---------------------------------------------------------------------------
template <std::size_t N, typename T = double>
static void BM_CudaDoubleBuf(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_double_buf(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = double(sizeof(T) * 8);
    state.counters["ampere_async"] = hpc::gemm::cuda_has_ampere() ? 1.0 : 0.0;
}

// ---------------------------------------------------------------------------
// Level 4 -- Tensor Cores (WMMA) -- fp32 only, sm_70+
// ---------------------------------------------------------------------------
template <std::size_t N>
static void BM_CudaWmma(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    if (!hpc::gemm::cuda_has_tensor_cores()) { state.SkipWithMessage("Tensor Cores not available (requires sm_70+)"); return; }
    hpc::Matrix<float> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_wmma(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = 16;  // fp16 MMA
    state.counters["tensor_cores"] = 1;
}

// ---------------------------------------------------------------------------
// Level 5 -- Vectorized loads (float4/double2) + shared-memory XOR swizzle.
// Falls back to RegTile internally when K or N isn't a multiple of the
// vector width -- always correct, always runs (given a CUDA device).
// ---------------------------------------------------------------------------
template <std::size_t N, typename T = double>
static void BM_CudaVectorized(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_vectorized(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = double(sizeof(T) * 8);
    state.counters["vec_width"] = double(16 / sizeof(T));
}

// ---------------------------------------------------------------------------
// Level 6 -- Raw Tensor Cores via mma.sync + ldmatrix -- fp32 only, sm_80+.
// VERIFIED on real hardware (RTX 5080, Blackwell sm_120): the A-fragment
// ldmatrix.x4 address mapping had its row/col quadrant bits swapped
// (produced numerically wrong output, not a crash) -- fixed and
// cross-checked against a reference implementation; see gemm_kernels.cu's
// kernel_mma_ldmatrix file comment for the full writeup.
// ---------------------------------------------------------------------------
template <std::size_t N>
static void BM_CudaMmaLdmatrix(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    if (!hpc::gemm::cuda_has_ampere()) { state.SkipWithMessage("mma.sync m16n8k16 requires sm_80+ (Ampere)"); return; }
    hpc::Matrix<float> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_mma_ldmatrix(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = 16;  // fp16 MMA
    state.counters["tensor_cores"] = 1;
}

// ---------------------------------------------------------------------------
// Level 7 -- Hopper warp specialization + TMA (wgmma) -- fp32 only, sm_90a.
// BEST-EFFORT, EXPLICITLY UNVERIFIED, LIKELY NON-FUNCTIONAL: see
// gemm_kernels.cu's kernel_hopper_wgmma file comment. Requires M=N=64k,
// K=16k exactly (all sizes below satisfy this) and real Hopper hardware,
// which was not available anywhere in this project -- this benchmark will
// SKIP on every machine this repo has actually been run on.
// ---------------------------------------------------------------------------
template <std::size_t N>
static void BM_CudaHopperWgmma(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    if (!hpc::gemm::cuda_has_hopper()) { state.SkipWithMessage("wgmma/TMA requires sm_90a (Hopper) -- UNVERIFIED code path, see src/gemm/README.md"); return; }
    hpc::Matrix<float> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_hopper_wgmma(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = 16;
    state.counters["unverified"] = 1;
}

// ---------------------------------------------------------------------------
// Level 8 -- Pipelined WMMA (bigger tiles + cp.async double buffering) --
// fp32 only, sm_70+. NEW kernel, added after the cuBLAS reference below
// measured this GPU's real Tensor Core ceiling. See gemm_kernels.cu's
// kernel_wmma_pipelined file comment for the full design rationale.
// N=64/128 are below or barely at the 128x128x32 exact-tile requirement
// (N=64 always falls back to gemm_cuda_wmma; N=128 needs K=128, a
// multiple of 32, which it is) -- kept in the standard size sweep for
// direct comparison against every other Level 0-7 kernel at the same
// sizes; N=4096 is where the fast path matters most.
// ---------------------------------------------------------------------------
template <std::size_t N>
static void BM_CudaWmmaPipelined(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    if (!hpc::gemm::cuda_has_tensor_cores()) { state.SkipWithMessage("Tensor Cores not available (requires sm_70+)"); return; }
    hpc::Matrix<float> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_wmma_pipelined(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = 16;  // fp16 MMA
    state.counters["tensor_cores"] = 1;
    state.counters["exact_tiles"] = (N % 128 == 0) ? 1 : 0;  // 1 = fast path, 0 = kernel_wmma fallback
}

// ---------------------------------------------------------------------------
// Reference -- cuBLAS (vendor-tuned upper bound, not part of the Level
// 0-7 ladder above). See gemm_kernels.cu's "Reference -- cuBLAS" section
// for why this exists: the hand-written Tensor Core kernels above measured
// only ~5 TFLOP/s on this RTX 5080, well under Blackwell's realistic
// Tensor Core potential -- this establishes what's actually achievable
// here before attempting a larger hand-written rewrite to close that gap.
// ---------------------------------------------------------------------------
template <std::size_t N, typename T = double>
static void BM_CudaCublas(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_cublas(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = double(sizeof(T) * 8);
}

// fp32-only: TF32 Tensor Core compute via cublasGemmEx.
template <std::size_t N>
static void BM_CudaCublasTf32(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    if (!hpc::gemm::cuda_has_ampere()) { state.SkipWithMessage("TF32 Tensor Cores require sm_80+ (Ampere)"); return; }
    hpc::Matrix<float> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1); fill_random(B, 2);
    for (auto _ : state) { hpc::gemm::gemm_cuda_cublas_tf32(A, B, C); benchmark::DoNotOptimize(C.data()); benchmark::ClobberMemory(); }
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["precision"] = 32;  // fp32 storage in/out; TF32 (10-bit mantissa) internally
    state.counters["tensor_cores"] = 1;
}

// ---------------------------------------------------------------------------
// Reference -- cuBLAS, compute-only (device-resident buffers, allocated
// and filled ONCE outside the timed loop -- no per-iteration cudaMalloc/
// H2D/D2H). See gemm_kernels.cu's "raw-device-pointer entry points"
// comment: BM_CudaCublas/BM_CudaCublasTf32 above time a full round trip
// every iteration, which for large N is dominated by ~GB-scale data
// movement and allocation, not the matmul itself -- these measure ONLY
// the GEMM call, to answer "what can this GPU's Tensor Cores actually do".
// Uses only the toolkit-type-free gemm_cuda_malloc/_free/_memcpy_h2d/
// _device_synchronize wrappers (see gemm/cuda.hpp) so this file needs no
// <cuda_runtime.h> include.
// ---------------------------------------------------------------------------
template <std::size_t N>
static void BM_CudaCublasComputeOnly(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    const std::size_t bytes = N * N * sizeof(float);
    float* dA = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes));
    float* dB = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes));
    float* dC = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes));
    hpc::Matrix<float> A(N, N), B(N, N);
    fill_random(A, 1); fill_random(B, 2);
    hpc::gemm::gemm_cuda_memcpy_h2d(dA, A.data(), bytes);
    hpc::gemm::gemm_cuda_memcpy_h2d(dB, B.data(), bytes);
    for (auto _ : state) {
        hpc::gemm::gemm_cuda_cublas_device_f32(dA, dB, dC, int(N), int(N), int(N));
        hpc::gemm::gemm_cuda_device_synchronize();
        benchmark::ClobberMemory();
    }
    hpc::gemm::gemm_cuda_free(dA); hpc::gemm::gemm_cuda_free(dB); hpc::gemm::gemm_cuda_free(dC);
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["compute_only"] = 1;
}

template <std::size_t N>
static void BM_CudaCublasTf32ComputeOnly(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    if (!hpc::gemm::cuda_has_ampere()) { state.SkipWithMessage("TF32 Tensor Cores require sm_80+ (Ampere)"); return; }
    const std::size_t bytes = N * N * sizeof(float);
    float* dA = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes));
    float* dB = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes));
    float* dC = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes));
    hpc::Matrix<float> A(N, N), B(N, N);
    fill_random(A, 1); fill_random(B, 2);
    hpc::gemm::gemm_cuda_memcpy_h2d(dA, A.data(), bytes);
    hpc::gemm::gemm_cuda_memcpy_h2d(dB, B.data(), bytes);
    for (auto _ : state) {
        hpc::gemm::gemm_cuda_cublas_tf32_device(dA, dB, dC, int(N), int(N), int(N));
        hpc::gemm::gemm_cuda_device_synchronize();
        benchmark::ClobberMemory();
    }
    hpc::gemm::gemm_cuda_free(dA); hpc::gemm::gemm_cuda_free(dB); hpc::gemm::gemm_cuda_free(dC);
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["compute_only"] = 1;
    state.counters["tensor_cores"] = 1;
}

// fp16-in/fp32-accumulate compute-only -- the natural next data point
// after TF32 compute-only (~59 TFLOP/s), since dense FP16 Tensor Core
// throughput is roughly 2x TF32's on Ampere-and-later. Converts once
// outside the timed region via gemm_cuda_convert_f32_to_f16_device.
template <std::size_t N>
static void BM_CudaCublasFp16ComputeOnly(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    if (!hpc::gemm::cuda_has_tensor_cores()) { state.SkipWithMessage("FP16 Tensor Cores require sm_70+ (Volta)"); return; }
    const std::size_t bytes32 = N * N * sizeof(float);
    const std::size_t bytes16 = N * N * 2;  // __half is 2 bytes; kept toolkit-type-free (see cuda.hpp)
    float* dA32 = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes32));
    float* dB32 = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes32));
    void*  dA16 = hpc::gemm::gemm_cuda_malloc(bytes16);
    void*  dB16 = hpc::gemm::gemm_cuda_malloc(bytes16);
    float* dC   = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes32));
    hpc::Matrix<float> A(N, N), B(N, N);
    fill_random(A, 1); fill_random(B, 2);
    hpc::gemm::gemm_cuda_memcpy_h2d(dA32, A.data(), bytes32);
    hpc::gemm::gemm_cuda_memcpy_h2d(dB32, B.data(), bytes32);
    hpc::gemm::gemm_cuda_convert_f32_to_f16_device(dA32, dA16, int(N * N));
    hpc::gemm::gemm_cuda_convert_f32_to_f16_device(dB32, dB16, int(N * N));
    hpc::gemm::gemm_cuda_device_synchronize();
    for (auto _ : state) {
        hpc::gemm::gemm_cuda_cublas_fp16_device(dA16, dB16, dC, int(N), int(N), int(N));
        hpc::gemm::gemm_cuda_device_synchronize();
        benchmark::ClobberMemory();
    }
    hpc::gemm::gemm_cuda_free(dA32); hpc::gemm::gemm_cuda_free(dB32);
    hpc::gemm::gemm_cuda_free(dA16); hpc::gemm::gemm_cuda_free(dB16);
    hpc::gemm::gemm_cuda_free(dC);
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["compute_only"] = 1;
    state.counters["tensor_cores"] = 1;
}

// Level 8's compute-only counterpart -- same pre-staging as
// BM_CudaCublasFp16ComputeOnly above, so the two numbers are directly
// comparable: this is "how close does the NEW hand-written kernel get to
// cuBLAS's ~118 TFLOP/s dense-FP16 ceiling once transfer/conversion
// overhead is excluded from both". Requires exact-tile N (multiple of
// 128) -- gemm_cuda_wmma_pipelined_device has no fallback at this layer.
template <std::size_t N>
static void BM_CudaWmmaPipelinedComputeOnly(benchmark::State& state) {
    static_assert(N % 128 == 0, "BM_CudaWmmaPipelinedComputeOnly requires N a multiple of 128");
    if (hpc::gemm::cuda_device_count() == 0) { state.SkipWithMessage("No CUDA device available"); return; }
    if (!hpc::gemm::cuda_has_tensor_cores()) { state.SkipWithMessage("Tensor Cores not available (requires sm_70+)"); return; }
    const std::size_t bytes32 = N * N * sizeof(float);
    const std::size_t bytes16 = N * N * 2;
    float* dA32 = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes32));
    float* dB32 = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes32));
    void*  dA16 = hpc::gemm::gemm_cuda_malloc(bytes16);
    void*  dB16 = hpc::gemm::gemm_cuda_malloc(bytes16);
    float* dC   = static_cast<float*>(hpc::gemm::gemm_cuda_malloc(bytes32));
    hpc::Matrix<float> A(N, N), B(N, N);
    fill_random(A, 1); fill_random(B, 2);
    hpc::gemm::gemm_cuda_memcpy_h2d(dA32, A.data(), bytes32);
    hpc::gemm::gemm_cuda_memcpy_h2d(dB32, B.data(), bytes32);
    hpc::gemm::gemm_cuda_convert_f32_to_f16_device(dA32, dA16, int(N * N));
    hpc::gemm::gemm_cuda_convert_f32_to_f16_device(dB32, dB16, int(N * N));
    hpc::gemm::gemm_cuda_device_synchronize();
    for (auto _ : state) {
        hpc::gemm::gemm_cuda_wmma_pipelined_device(dA16, dB16, dC, int(N), int(N), int(N));
        hpc::gemm::gemm_cuda_device_synchronize();
        benchmark::ClobberMemory();
    }
    hpc::gemm::gemm_cuda_free(dA32); hpc::gemm::gemm_cuda_free(dB32);
    hpc::gemm::gemm_cuda_free(dA16); hpc::gemm::gemm_cuda_free(dB16);
    hpc::gemm::gemm_cuda_free(dC);
    state.counters["GFLOP/s"] = benchmark::Counter(flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = double(N);
    state.counters["compute_only"] = 1;
    state.counters["tensor_cores"] = 1;
}

// ---------------------------------------------------------------------------
// Registrations
// ---------------------------------------------------------------------------
#define HPC_REG_CUDA_T(TMPL)                                                             \
    BENCHMARK((TMPL<64,   double>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f64/N=64");    \
    BENCHMARK((TMPL<256,  double>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f64/N=256");   \
    BENCHMARK((TMPL<512,  double>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f64/N=512");   \
    BENCHMARK((TMPL<1024, double>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f64/N=1024");  \
    BENCHMARK((TMPL<4096, double>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f64/N=4096");  \
    BENCHMARK((TMPL<64,   float>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f32/N=64");     \
    BENCHMARK((TMPL<256,  float>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f32/N=256");    \
    BENCHMARK((TMPL<512,  float>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f32/N=512");    \
    BENCHMARK((TMPL<1024, float>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f32/N=1024");   \
    BENCHMARK((TMPL<4096, float>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f32/N=4096")

// WMMA is float-only -- separate macro.
#define HPC_REG_CUDA_WMMA(TMPL)                                                          \
    BENCHMARK((TMPL<64>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f32/N=64");      \
    BENCHMARK((TMPL<256>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f32/N=256");    \
    BENCHMARK((TMPL<512>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f32/N=512");    \
    BENCHMARK((TMPL<1024>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f32/N=1024");  \
    BENCHMARK((TMPL<4096>))->Unit(benchmark::kMicrosecond)->Name(#TMPL "/f32/N=4096")

HPC_REG_CUDA_T(BM_CudaNaive);
HPC_REG_CUDA_T(BM_CudaReordered);
HPC_REG_CUDA_T(BM_CudaBlocked);
HPC_REG_CUDA_T(BM_CudaRegTile);
HPC_REG_CUDA_T(BM_CudaDoubleBuf);
HPC_REG_CUDA_T(BM_CudaVectorized);
HPC_REG_CUDA_WMMA(BM_CudaWmma);
HPC_REG_CUDA_WMMA(BM_CudaMmaLdmatrix);
HPC_REG_CUDA_WMMA(BM_CudaHopperWgmma);
HPC_REG_CUDA_WMMA(BM_CudaWmmaPipelined);
HPC_REG_CUDA_T(BM_CudaCublas);
HPC_REG_CUDA_WMMA(BM_CudaCublasTf32);

// Larger problem sizes -- N=4096 (the size shared with every other kernel
// above) is too small for a GEMM to reach a GPU's asymptotic compute-bound
// peak. cuBLAS gets N=8192/16384 to show what it can do; BM_CudaWmmaPipelined
// gets the same sizes to see how close the new hand-written kernel gets.
BENCHMARK((BM_CudaCublas<8192,  float>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublas/f32/N=8192");
BENCHMARK((BM_CudaCublas<16384, float>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublas/f32/N=16384");
BENCHMARK((BM_CudaCublasTf32<8192>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublasTf32/f32/N=8192");
BENCHMARK((BM_CudaCublasTf32<16384>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublasTf32/f32/N=16384");
BENCHMARK((BM_CudaWmmaPipelined<8192>))->Unit(benchmark::kMillisecond)->Name("BM_CudaWmmaPipelined/f32/N=8192");
BENCHMARK((BM_CudaWmmaPipelined<16384>))->Unit(benchmark::kMillisecond)->Name("BM_CudaWmmaPipelined/f32/N=16384");

// Same sizes, compute-only (see BM_CudaCublas*ComputeOnly's comment above
// for why this is a different, higher number than the end-to-end rows
// above it at the same N).
BENCHMARK((BM_CudaCublasComputeOnly<4096>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublasComputeOnly/f32/N=4096");
BENCHMARK((BM_CudaCublasComputeOnly<8192>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublasComputeOnly/f32/N=8192");
BENCHMARK((BM_CudaCublasComputeOnly<16384>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublasComputeOnly/f32/N=16384");
BENCHMARK((BM_CudaCublasTf32ComputeOnly<4096>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublasTf32ComputeOnly/f32/N=4096");
BENCHMARK((BM_CudaCublasTf32ComputeOnly<8192>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublasTf32ComputeOnly/f32/N=8192");
BENCHMARK((BM_CudaCublasTf32ComputeOnly<16384>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublasTf32ComputeOnly/f32/N=16384");
BENCHMARK((BM_CudaCublasFp16ComputeOnly<4096>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublasFp16ComputeOnly/f32/N=4096");
BENCHMARK((BM_CudaCublasFp16ComputeOnly<8192>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublasFp16ComputeOnly/f32/N=8192");
BENCHMARK((BM_CudaCublasFp16ComputeOnly<16384>))->Unit(benchmark::kMillisecond)->Name("BM_CudaCublasFp16ComputeOnly/f32/N=16384");
BENCHMARK((BM_CudaWmmaPipelinedComputeOnly<4096>))->Unit(benchmark::kMillisecond)->Name("BM_CudaWmmaPipelinedComputeOnly/f32/N=4096");
BENCHMARK((BM_CudaWmmaPipelinedComputeOnly<8192>))->Unit(benchmark::kMillisecond)->Name("BM_CudaWmmaPipelinedComputeOnly/f32/N=8192");
BENCHMARK((BM_CudaWmmaPipelinedComputeOnly<16384>))->Unit(benchmark::kMillisecond)->Name("BM_CudaWmmaPipelinedComputeOnly/f32/N=16384");

#undef HPC_REG_CUDA_T
#undef HPC_REG_CUDA_WMMA

BENCHMARK_MAIN();

