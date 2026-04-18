/**
 * @file bench_gemm_cuda.cpp
 * @brief Google Benchmark driver for CUDA GEMM kernels (all 5 levels).
 *
 * Levels:
 *   CudaNaive      -- Level 0: global memory only
 *   CudaReordered  -- Level 0b: CPU-symmetry baseline
 *   CudaBlocked    -- Level 1: TILE=16 shared-memory tiling
 *   CudaRegTile    -- Level 2: 128x128 block, 8x8 register tile per thread
 *   CudaDoubleBuf  -- Level 3: Level 2 + double buffering (cp.async on Ampere+)
 *   CudaWmma       -- Level 4: Tensor Cores via WMMA (fp32 only, sm_70+)
 *
 * Runtime guards:
 *   All kernels check cuda_device_count() > 0 -> SKIPPED on CPU-only machines.
 *   CudaWmma additionally checks cuda_has_tensor_cores() -> SKIPPED on pre-Volta.
 *   CudaDoubleBuf reports whether cp.async (Ampere+) is active.
 */

#include "gemm/cuda.hpp"
#include "hpc/matrix.hpp"

#include <benchmark/benchmark.h>

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
HPC_REG_CUDA_WMMA(BM_CudaWmma);

#undef HPC_REG_CUDA_T
#undef HPC_REG_CUDA_WMMA

BENCHMARK_MAIN();

