/**
 * @file bench_gemm_cuda.cpp
 * @brief Google Benchmark driver for CUDA GEMM kernels.
 *
 * ============================================================
 *  Design
 * ============================================================
 *
 * This file benchmarks three CUDA GEMM strategies (naive, reordered,
 * blocked/tiled) across two precisions (f32, f64) and five matrix sizes
 * (N = 64, 256, 512, 1024, 4096).
 *
 * Runtime device detection:
 *   At the start of each benchmark function, we call cuda_device_count().
 *   If it returns 0 the benchmark is skipped with a human-readable message.
 *   This means the binary compiles on CPU-only machines (macOS, CI without
 *   GPU) and produces `SKIPPED` rows rather than crashing.
 *
 * Each benchmark:
 *   1. Allocates A, B, C on the host and fills A/B with random data.
 *   2. Calls the kernel launcher (which internally allocates device memory,
 *      copies data, runs the kernel, synchronises, copies back).
 *   3. Uses DoNotOptimize / ClobberMemory on C so the compiler cannot
 *      eliminate the call.
 *   4. Reports GFLOP/s = (2 × N³) / (time_ns).
 *
 * Note on timing:
 *   The timings include host↔device data transfer.  For a fair arithmetic
 *   throughput comparison with CPU kernels this is the right thing to
 *   measure (end-to-end time visible to the caller).
 *   If you want to measure compute-only kernel time, use CUDA events and
 *   a custom benchmark harness.
 *
 * ============================================================
 *  Naming convention
 * ============================================================
 *
 *   CudaNaive/<prec>/N=<size>
 *   CudaReordered/<prec>/N=<size>
 *   CudaBlocked/<prec>/N=<size>
 *
 *   Filter examples:
 *     ./build/benchmarks/bench_gemm_cuda --benchmark_filter="CudaBlocked"
 *     ./build/benchmarks/bench_gemm_cuda --benchmark_filter="f32"
 *     ./build/benchmarks/bench_gemm_cuda --benchmark_filter="N=1024"
 */

#include "gemm/cuda.hpp"
#include "hpc/matrix.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <type_traits>

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

template <typename T>
static void fill_random(hpc::Matrix<T>& M, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(T{-1}, T{1});
    for (std::size_t i = 0; i < M.rows(); ++i)
        for (std::size_t j = 0; j < M.cols(); ++j)
            M(i, j) = dist(rng);
}

static inline double flops(std::size_t N) {
    return 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
}

// ---------------------------------------------------------------------------
// Benchmark templates
// ---------------------------------------------------------------------------

template <std::size_t N, typename T = double>
static void BM_CudaNaive(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) {
        state.SkipWithMessage("No CUDA device available");
        return;
    }
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_cuda_naive(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
}

template <std::size_t N, typename T = double>
static void BM_CudaReordered(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) {
        state.SkipWithMessage("No CUDA device available");
        return;
    }
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_cuda_reordered(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
}

template <std::size_t N, typename T = double>
static void BM_CudaBlocked(benchmark::State& state) {
    if (hpc::gemm::cuda_device_count() == 0) {
        state.SkipWithMessage("No CUDA device available");
        return;
    }
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_cuda_blocked(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
}

// ---------------------------------------------------------------------------
// Registrations
// ---------------------------------------------------------------------------

// Double precision
#define HPC_REG_CUDA(TEMPLATE, PNAME)                                                  \
    BENCHMARK((TEMPLATE<64,   double>))->Unit(benchmark::kMicrosecond)->Name(#TEMPLATE "/f64/N=64");   \
    BENCHMARK((TEMPLATE<256,  double>))->Unit(benchmark::kMicrosecond)->Name(#TEMPLATE "/f64/N=256");  \
    BENCHMARK((TEMPLATE<512,  double>))->Unit(benchmark::kMicrosecond)->Name(#TEMPLATE "/f64/N=512");  \
    BENCHMARK((TEMPLATE<1024, double>))->Unit(benchmark::kMicrosecond)->Name(#TEMPLATE "/f64/N=1024"); \
    BENCHMARK((TEMPLATE<4096, double>))->Unit(benchmark::kMicrosecond)->Name(#TEMPLATE "/f64/N=4096"); \
    BENCHMARK((TEMPLATE<64,   float>))->Unit(benchmark::kMicrosecond)->Name(#TEMPLATE "/f32/N=64");    \
    BENCHMARK((TEMPLATE<256,  float>))->Unit(benchmark::kMicrosecond)->Name(#TEMPLATE "/f32/N=256");   \
    BENCHMARK((TEMPLATE<512,  float>))->Unit(benchmark::kMicrosecond)->Name(#TEMPLATE "/f32/N=512");   \
    BENCHMARK((TEMPLATE<1024, float>))->Unit(benchmark::kMicrosecond)->Name(#TEMPLATE "/f32/N=1024"); \
    BENCHMARK((TEMPLATE<4096, float>))->Unit(benchmark::kMicrosecond)->Name(#TEMPLATE "/f32/N=4096")

HPC_REG_CUDA(BM_CudaNaive,    "");
HPC_REG_CUDA(BM_CudaReordered,"");
HPC_REG_CUDA(BM_CudaBlocked,  "");

#undef HPC_REG_CUDA

BENCHMARK_MAIN();

