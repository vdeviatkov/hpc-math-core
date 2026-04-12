/**
 * @file bench_gemm.cpp
 * @brief Google Benchmark driver for GEMM kernels.
 *
 * Running the benchmarks
 * ======================
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
 *   ./build/benchmarks/bench_gemm --benchmark_format=console
 *
 * Interpreting results
 * ====================
 * The benchmark reports wall-clock time in microseconds per iteration.
 * "Iterations" is automatically chosen by Google Benchmark to make the
 * total measurement time statistically stable (≥ 0.5 s by default).
 *
 * Derived GFLOP/s
 * ---------------
 * A square N×N GEMM performs 2*N³ floating-point operations (N³ multiplies
 * + N³ adds, ignoring the zero-init sweep). Convert µs → GFLOP/s as:
 *
 *   GFLOP/s = (2 * N^3) / (time_µs * 1e3)
 *
 * A custom counter `items_per_second` is registered so that Google Benchmark
 * can display this automatically.
 *
 * Allocation note
 * ===============
 * Matrices are allocated *outside* the benchmark loop (in the fixture body)
 * so that only the actual GEMM computation is measured. Allocating inside
 * `for (auto _ : state)` would measure memory-allocation overhead, not the
 * kernel we care about.
 */

#include "gemm/naive.hpp"
#include "gemm/reordered.hpp"
#include "hpc/matrix.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>

using hpc::MatrixD;

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

static void fill_random(MatrixD& M, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (std::size_t i = 0; i < M.rows(); ++i)
        for (std::size_t j = 0; j < M.cols(); ++j)
            M(i, j) = dist(rng);
}

/// Compute the number of floating-point operations for a square N×N GEMM.
static constexpr double flops(std::size_t N) {
    return 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
}

// ---------------------------------------------------------------------------
// Benchmark templates
// ---------------------------------------------------------------------------

/**
 * @brief Benchmark gemm_naive for a given matrix size N.
 *
 * The template parameter allows the compiler to fully specialise and inline
 * the benchmark body for each N without runtime branching.
 */
template <std::size_t N>
static void BM_Naive(benchmark::State& state) {
    MatrixD A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);

    for (auto _ : state) {
        hpc::gemm::gemm_naive(A, B, C);
        // Prevent the compiler from optimising away the computation.
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    // Report GFLOP/s as a custom counter. CounterFlags::kIsRate divides by
    // the elapsed time automatically.
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = static_cast<double>(N);
}

/**
 * @brief Benchmark gemm_reordered for a given matrix size N.
 */
template <std::size_t N>
static void BM_Reordered(benchmark::State& state) {
    MatrixD A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);

    for (auto _ : state) {
        hpc::gemm::gemm_reordered(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"] = static_cast<double>(N);
}

// ---------------------------------------------------------------------------
// Register benchmarks
//
// Sizes chosen to exercise different cache levels:
//   N=64   → A+B+C ≈   96 KB  (fits in L2, 256 KB typical)
//   N=256  → A+B+C ≈    1.5 MB (fits in L3, 8–32 MB typical)
//   N=512  → A+B+C ≈    6 MB  (fits in L3 on server CPUs)
//   N=1024 → A+B+C ≈   24 MB  (likely spills to DRAM)
//   N=4096 → A+B+C ≈  384 MB  (deep DRAM pressure; stresses memory bandwidth ceiling)
// ---------------------------------------------------------------------------

BENCHMARK(BM_Naive<64>)->Unit(benchmark::kMicrosecond)->Name("Naive/N=64");
BENCHMARK(BM_Naive<256>)->Unit(benchmark::kMicrosecond)->Name("Naive/N=256");
BENCHMARK(BM_Naive<512>)->Unit(benchmark::kMicrosecond)->Name("Naive/N=512");
BENCHMARK(BM_Naive<1024>)->Unit(benchmark::kMicrosecond)->Name("Naive/N=1024");
BENCHMARK(BM_Naive<4096>)->Unit(benchmark::kMicrosecond)->Name("Naive/N=4096");

BENCHMARK(BM_Reordered<64>)->Unit(benchmark::kMicrosecond)->Name("Reordered/N=64");
BENCHMARK(BM_Reordered<256>)->Unit(benchmark::kMicrosecond)->Name("Reordered/N=256");
BENCHMARK(BM_Reordered<512>)->Unit(benchmark::kMicrosecond)->Name("Reordered/N=512");
BENCHMARK(BM_Reordered<1024>)->Unit(benchmark::kMicrosecond)->Name("Reordered/N=1024");
BENCHMARK(BM_Reordered<4096>)->Unit(benchmark::kMicrosecond)->Name("Reordered/N=4096");

BENCHMARK_MAIN();
