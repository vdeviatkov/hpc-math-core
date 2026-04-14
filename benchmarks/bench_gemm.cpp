/**
 * @file bench_gemm.cpp
 * @brief Google Benchmark driver for GEMM kernels — float and double.
 *
 * Running the benchmarks
 * ======================
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
 *   ./build/benchmarks/bench_gemm --benchmark_format=console
 *
 * Precision variants
 * ==================
 * Every kernel is benchmarked for both element types:
 *
 *   double (f64) — 8 bytes per element, IEEE-754 64-bit.
 *                  Reference precision; matches most scientific code.
 *   float  (f32) — 4 bytes per element, IEEE-754 32-bit.
 *                  Twice as many elements fit per cache line / SIMD register,
 *                  so a well-vectorised float kernel can deliver up to 2× the
 *                  GFLOP/s of its double equivalent on the same hardware.
 *                  Widely used in ML inference and HFT risk engines where
 *                  ~7 significant decimal digits are sufficient.
 *
 * With -ffast-math + -march=native the compiler is free to:
 *   • Contract multiply-add pairs into FMA instructions.
 *   • Vectorise the inner j-loop using SIMD (SSE/AVX/AVX-512).
 *   • Reorder floating-point operations for better pipeline utilisation.
 *
 * GFLOP/s formula
 * ---------------
 * A square N×N GEMM performs 2*N³ floating-point operations.
 * Convert µs → GFLOP/s:  (2 * N^3) / (time_µs * 1e3)
 * The formula is identical for float and double — only the throughput differs.
 *
 * Allocation note
 * ===============
 * Matrices are allocated *outside* the benchmark loop so that only the
 * GEMM kernel itself is measured.
 */

#include "gemm/blocked.hpp"
#include "gemm/naive.hpp"
#include "gemm/reordered.hpp"
#include "hpc/matrix.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <type_traits>

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

template <typename T>
static void fill_random(hpc::Matrix<T>& M, unsigned seed) {
    std::mt19937_64 rng(seed);
    // Use the appropriate distribution for the element type.
    std::uniform_real_distribution<T> dist(T{0}, T{1});
    for (std::size_t i = 0; i < M.rows(); ++i)
        for (std::size_t j = 0; j < M.cols(); ++j)
            M(i, j) = dist(rng);
}

/// Compute the number of floating-point operations for a square N×N GEMM.
static constexpr double flops(std::size_t N) {
    return 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
}

/// Human-readable precision label used in benchmark names.
template <typename T>
constexpr const char* precision_label() {
    if constexpr (std::is_same_v<T, float>)
        return "f32";
    else
        return "f64";
}

// ---------------------------------------------------------------------------
// Benchmark templates — templated on both matrix size N and element type T.
// ---------------------------------------------------------------------------

/**
 * @brief Benchmark gemm_naive<T> for a given matrix size N.
 */
template <std::size_t N, typename T = double>
static void BM_Naive(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);

    for (auto _ : state) {
        hpc::gemm::gemm_naive(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);  // 32 or 64
}

/**
 * @brief Benchmark gemm_reordered<T> for a given matrix size N.
 */
template <std::size_t N, typename T = double>
static void BM_Reordered(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);

    for (auto _ : state) {
        hpc::gemm::gemm_reordered(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
}

/**
 * @brief Benchmark gemm_blocked<T> for a given matrix size N.
 *
 * Uses the default compile-time tile size (kDefaultTile = 64). To benchmark
 * a different tile size, pass it as a third template argument to gemm_blocked
 * directly, e.g.: hpc::gemm::gemm_blocked<T, 32>(A, B, C).
 */
template <std::size_t N, typename T = double>
static void BM_Blocked(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);

    for (auto _ : state) {
        hpc::gemm::gemm_blocked(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["tile"]      = static_cast<double>(hpc::gemm::kDefaultTile);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
}

// ---------------------------------------------------------------------------
// Register benchmarks
//
// Naming convention:  <Kernel>/<precision>/N=<size>
//   precision = f64 (double, 8 B/elem) or f32 (float, 4 B/elem)
//
// Working-set sizes per matrix (A + B + C = 3 matrices):
//   N=64   f64:  96 KB   f32:  48 KB
//   N=256  f64:   1.5 MB f32: 768 KB
//   N=512  f64:   6 MB   f32:   3 MB
//   N=1024 f64:  24 MB   f32:  12 MB
//   N=4096 f64: 384 MB   f32: 192 MB
//
// float matrices are half the size, so they fit in faster cache levels at
// larger N, which amplifies the benefit of both tiling and SIMD.
// ---------------------------------------------------------------------------

// ---- double (f64) ----------------------------------------------------------
BENCHMARK(BM_Naive<64>)->Unit(benchmark::kMicrosecond)->Name("Naive/f64/N=64");
BENCHMARK(BM_Naive<256>)->Unit(benchmark::kMicrosecond)->Name("Naive/f64/N=256");
BENCHMARK(BM_Naive<512>)->Unit(benchmark::kMicrosecond)->Name("Naive/f64/N=512");
BENCHMARK(BM_Naive<1024>)->Unit(benchmark::kMicrosecond)->Name("Naive/f64/N=1024");
BENCHMARK(BM_Naive<4096>)->Unit(benchmark::kMicrosecond)->Name("Naive/f64/N=4096");

BENCHMARK(BM_Reordered<64>)->Unit(benchmark::kMicrosecond)->Name("Reordered/f64/N=64");
BENCHMARK(BM_Reordered<256>)->Unit(benchmark::kMicrosecond)->Name("Reordered/f64/N=256");
BENCHMARK(BM_Reordered<512>)->Unit(benchmark::kMicrosecond)->Name("Reordered/f64/N=512");
BENCHMARK(BM_Reordered<1024>)->Unit(benchmark::kMicrosecond)->Name("Reordered/f64/N=1024");
BENCHMARK(BM_Reordered<4096>)->Unit(benchmark::kMicrosecond)->Name("Reordered/f64/N=4096");

BENCHMARK(BM_Blocked<64>)->Unit(benchmark::kMicrosecond)->Name("Blocked/f64/N=64");
BENCHMARK(BM_Blocked<256>)->Unit(benchmark::kMicrosecond)->Name("Blocked/f64/N=256");
BENCHMARK(BM_Blocked<512>)->Unit(benchmark::kMicrosecond)->Name("Blocked/f64/N=512");
BENCHMARK(BM_Blocked<1024>)->Unit(benchmark::kMicrosecond)->Name("Blocked/f64/N=1024");
BENCHMARK(BM_Blocked<4096>)->Unit(benchmark::kMicrosecond)->Name("Blocked/f64/N=4096");

// ---- float (f32) -----------------------------------------------------------
BENCHMARK(BM_Naive<64, float>)->Unit(benchmark::kMicrosecond)->Name("Naive/f32/N=64");
BENCHMARK(BM_Naive<256, float>)->Unit(benchmark::kMicrosecond)->Name("Naive/f32/N=256");
BENCHMARK(BM_Naive<512, float>)->Unit(benchmark::kMicrosecond)->Name("Naive/f32/N=512");
BENCHMARK(BM_Naive<1024, float>)->Unit(benchmark::kMicrosecond)->Name("Naive/f32/N=1024");
BENCHMARK(BM_Naive<4096, float>)->Unit(benchmark::kMicrosecond)->Name("Naive/f32/N=4096");

BENCHMARK(BM_Reordered<64, float>)->Unit(benchmark::kMicrosecond)->Name("Reordered/f32/N=64");
BENCHMARK(BM_Reordered<256, float>)->Unit(benchmark::kMicrosecond)->Name("Reordered/f32/N=256");
BENCHMARK(BM_Reordered<512, float>)->Unit(benchmark::kMicrosecond)->Name("Reordered/f32/N=512");
BENCHMARK(BM_Reordered<1024, float>)->Unit(benchmark::kMicrosecond)->Name("Reordered/f32/N=1024");
BENCHMARK(BM_Reordered<4096, float>)->Unit(benchmark::kMicrosecond)->Name("Reordered/f32/N=4096");

BENCHMARK(BM_Blocked<64, float>)->Unit(benchmark::kMicrosecond)->Name("Blocked/f32/N=64");
BENCHMARK(BM_Blocked<256, float>)->Unit(benchmark::kMicrosecond)->Name("Blocked/f32/N=256");
BENCHMARK(BM_Blocked<512, float>)->Unit(benchmark::kMicrosecond)->Name("Blocked/f32/N=512");
BENCHMARK(BM_Blocked<1024, float>)->Unit(benchmark::kMicrosecond)->Name("Blocked/f32/N=1024");
BENCHMARK(BM_Blocked<4096, float>)->Unit(benchmark::kMicrosecond)->Name("Blocked/f32/N=4096");

BENCHMARK_MAIN();
