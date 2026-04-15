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
#include "gemm/neon.hpp"
#include "hpc/matrix.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <type_traits>

#include "gemm/avx2.hpp"
#include "gemm/avx512.hpp"

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

/**
 * @brief Benchmark gemm_avx2_naive<T> — i-j-k order, SIMD on the k-loop.
 *
 * Expected result: similar GFLOP/s to scalar naive — the column-stride B
 * access pattern is still cache-hostile regardless of SIMD width.
 * This benchmark answers: "does SIMD alone fix bad memory access?"  (No.)
 */
template <std::size_t N, typename T = double>
static void BM_Avx2Naive(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_avx2_naive(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
}

/**
 * @brief Benchmark gemm_avx2_reordered<T> — i-k-j order, SIMD on the j-loop.
 *
 * Expected result: ~SIMD_WIDTH × scalar reordered. B and C are accessed
 * stride-1, so every cache line is fully consumed. No blocking — degrades
 * at large N when C row i is evicted from L1 between k-iterations.
 */
template <std::size_t N, typename T = double>
static void BM_Avx2Reordered(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_avx2_reordered(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
}

/**
 * @brief Benchmark gemm_avx2_blocked<T> — tiled i-k-j, register-tiled micro-kernel.
 *
 * Expected result: highest GFLOP/s of the three. Outer tiling keeps the
 * working set in L2; the register tile eliminates C reload traffic and
 * drives both FMA ports at near-peak utilisation.
 */
template <std::size_t N, typename T = double>
static void BM_Avx2Blocked(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_avx2_blocked(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
#ifdef __AVX2__
    state.counters["avx2"] = 1;
#else
    state.counters["avx2"] = 0;
#endif
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

// ---- AVX2 Naive: SIMD on k-loop, i-j-k order (cache-hostile B access) ------
BENCHMARK(BM_Avx2Naive<64>)->Unit(benchmark::kMicrosecond)->Name("Avx2Naive/f64/N=64");
BENCHMARK(BM_Avx2Naive<256>)->Unit(benchmark::kMicrosecond)->Name("Avx2Naive/f64/N=256");
BENCHMARK(BM_Avx2Naive<512>)->Unit(benchmark::kMicrosecond)->Name("Avx2Naive/f64/N=512");
BENCHMARK(BM_Avx2Naive<1024>)->Unit(benchmark::kMicrosecond)->Name("Avx2Naive/f64/N=1024");
BENCHMARK(BM_Avx2Naive<4096>)->Unit(benchmark::kMicrosecond)->Name("Avx2Naive/f64/N=4096");

BENCHMARK(BM_Avx2Naive<64, float>)->Unit(benchmark::kMicrosecond)->Name("Avx2Naive/f32/N=64");
BENCHMARK(BM_Avx2Naive<256, float>)->Unit(benchmark::kMicrosecond)->Name("Avx2Naive/f32/N=256");
BENCHMARK(BM_Avx2Naive<512, float>)->Unit(benchmark::kMicrosecond)->Name("Avx2Naive/f32/N=512");
BENCHMARK(BM_Avx2Naive<1024, float>)->Unit(benchmark::kMicrosecond)->Name("Avx2Naive/f32/N=1024");
BENCHMARK(BM_Avx2Naive<4096, float>)->Unit(benchmark::kMicrosecond)->Name("Avx2Naive/f32/N=4096");

// ---- AVX2 Reordered: SIMD on j-loop, i-k-j order (cache-friendly, no tiling) ---
BENCHMARK(BM_Avx2Reordered<64>)->Unit(benchmark::kMicrosecond)->Name("Avx2Reordered/f64/N=64");
BENCHMARK(BM_Avx2Reordered<256>)->Unit(benchmark::kMicrosecond)->Name("Avx2Reordered/f64/N=256");
BENCHMARK(BM_Avx2Reordered<512>)->Unit(benchmark::kMicrosecond)->Name("Avx2Reordered/f64/N=512");
BENCHMARK(BM_Avx2Reordered<1024>)->Unit(benchmark::kMicrosecond)->Name("Avx2Reordered/f64/N=1024");
BENCHMARK(BM_Avx2Reordered<4096>)->Unit(benchmark::kMicrosecond)->Name("Avx2Reordered/f64/N=4096");

BENCHMARK(BM_Avx2Reordered<64, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx2Reordered/f32/N=64");
BENCHMARK(BM_Avx2Reordered<256, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx2Reordered/f32/N=256");
BENCHMARK(BM_Avx2Reordered<512, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx2Reordered/f32/N=512");
BENCHMARK(BM_Avx2Reordered<1024, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx2Reordered/f32/N=1024");
BENCHMARK(BM_Avx2Reordered<4096, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx2Reordered/f32/N=4096");

// ---- AVX2 Blocked: register-tiled micro-kernel + outer L2 tiling (full) ----
BENCHMARK(BM_Avx2Blocked<64>)->Unit(benchmark::kMicrosecond)->Name("Avx2Blocked/f64/N=64");
BENCHMARK(BM_Avx2Blocked<256>)->Unit(benchmark::kMicrosecond)->Name("Avx2Blocked/f64/N=256");
BENCHMARK(BM_Avx2Blocked<512>)->Unit(benchmark::kMicrosecond)->Name("Avx2Blocked/f64/N=512");
BENCHMARK(BM_Avx2Blocked<1024>)->Unit(benchmark::kMicrosecond)->Name("Avx2Blocked/f64/N=1024");
BENCHMARK(BM_Avx2Blocked<4096>)->Unit(benchmark::kMicrosecond)->Name("Avx2Blocked/f64/N=4096");

BENCHMARK(BM_Avx2Blocked<64, float>)->Unit(benchmark::kMicrosecond)->Name("Avx2Blocked/f32/N=64");
BENCHMARK(BM_Avx2Blocked<256, float>)->Unit(benchmark::kMicrosecond)->Name("Avx2Blocked/f32/N=256");
BENCHMARK(BM_Avx2Blocked<512, float>)->Unit(benchmark::kMicrosecond)->Name("Avx2Blocked/f32/N=512");
BENCHMARK(BM_Avx2Blocked<1024, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx2Blocked/f32/N=1024");
BENCHMARK(BM_Avx2Blocked<4096, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx2Blocked/f32/N=4096");

// ============================================================================
// AVX-512 benchmarks
// On non-AVX-512 targets each kernel transparently delegates to its AVX2
// equivalent, so these registrations are always safe to include.
// ============================================================================

/**
 * @brief Benchmark gemm_avx512_naive<T> — i-j-k, 512-bit SIMD on k-loop.
 * Expected: GFLOP/s ≈ scalar naive — gather is still cache-miss bound.
 */
template <std::size_t N, typename T = double>
static void BM_Avx512Naive(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_avx512_naive(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
#ifdef __AVX512F__
    state.counters["avx512"] = 1;
#else
    state.counters["avx512"] = 0;
#endif
}

/**
 * @brief Benchmark gemm_avx512_reordered<T> — i-k-j, 512-bit SIMD on j-loop.
 * Expected: ~2× AVX2 reordered (16 vs 8 f32 per FMA, stride-1 access).
 */
template <std::size_t N, typename T = double>
static void BM_Avx512Reordered(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_avx512_reordered(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
#ifdef __AVX512F__
    state.counters["avx512"] = 1;
#else
    state.counters["avx512"] = 0;
#endif
}

/**
 * @brief Benchmark gemm_avx512_blocked<T> — tiled i-k-j + 512-bit register tile.
 * Expected: highest GFLOP/s. L2 tiling + 4×32 f32 C tile held in ZMM registers.
 */
template <std::size_t N, typename T = double>
static void BM_Avx512Blocked(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_avx512_blocked(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
#ifdef __AVX512F__
    state.counters["avx512"] = 1;
#else
    state.counters["avx512"] = 0;
#endif
}

// ---- AVX-512 Naive ----------------------------------------------------------
BENCHMARK(BM_Avx512Naive<64>)->Unit(benchmark::kMicrosecond)->Name("Avx512Naive/f64/N=64");
BENCHMARK(BM_Avx512Naive<256>)->Unit(benchmark::kMicrosecond)->Name("Avx512Naive/f64/N=256");
BENCHMARK(BM_Avx512Naive<512>)->Unit(benchmark::kMicrosecond)->Name("Avx512Naive/f64/N=512");
BENCHMARK(BM_Avx512Naive<1024>)->Unit(benchmark::kMicrosecond)->Name("Avx512Naive/f64/N=1024");
BENCHMARK(BM_Avx512Naive<4096>)->Unit(benchmark::kMicrosecond)->Name("Avx512Naive/f64/N=4096");
BENCHMARK(BM_Avx512Naive<64, float>)->Unit(benchmark::kMicrosecond)->Name("Avx512Naive/f32/N=64");
BENCHMARK(BM_Avx512Naive<256, float>)->Unit(benchmark::kMicrosecond)->Name("Avx512Naive/f32/N=256");
BENCHMARK(BM_Avx512Naive<512, float>)->Unit(benchmark::kMicrosecond)->Name("Avx512Naive/f32/N=512");
BENCHMARK(BM_Avx512Naive<1024, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Naive/f32/N=1024");
BENCHMARK(BM_Avx512Naive<4096, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Naive/f32/N=4096");

// ---- AVX-512 Reordered ------------------------------------------------------
BENCHMARK(BM_Avx512Reordered<64>)->Unit(benchmark::kMicrosecond)->Name("Avx512Reordered/f64/N=64");
BENCHMARK(BM_Avx512Reordered<256>)->Unit(benchmark::kMicrosecond)->Name("Avx512Reordered/f64/N=256");
BENCHMARK(BM_Avx512Reordered<512>)->Unit(benchmark::kMicrosecond)->Name("Avx512Reordered/f64/N=512");
BENCHMARK(BM_Avx512Reordered<1024>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Reordered/f64/N=1024");
BENCHMARK(BM_Avx512Reordered<4096>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Reordered/f64/N=4096");
BENCHMARK(BM_Avx512Reordered<64, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Reordered/f32/N=64");
BENCHMARK(BM_Avx512Reordered<256, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Reordered/f32/N=256");
BENCHMARK(BM_Avx512Reordered<512, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Reordered/f32/N=512");
BENCHMARK(BM_Avx512Reordered<1024, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Reordered/f32/N=1024");
BENCHMARK(BM_Avx512Reordered<4096, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Reordered/f32/N=4096");

// ---- AVX-512 Blocked --------------------------------------------------------
BENCHMARK(BM_Avx512Blocked<64>)->Unit(benchmark::kMicrosecond)->Name("Avx512Blocked/f64/N=64");
BENCHMARK(BM_Avx512Blocked<256>)->Unit(benchmark::kMicrosecond)->Name("Avx512Blocked/f64/N=256");
BENCHMARK(BM_Avx512Blocked<512>)->Unit(benchmark::kMicrosecond)->Name("Avx512Blocked/f64/N=512");
BENCHMARK(BM_Avx512Blocked<1024>)->Unit(benchmark::kMicrosecond)->Name("Avx512Blocked/f64/N=1024");
BENCHMARK(BM_Avx512Blocked<4096>)->Unit(benchmark::kMicrosecond)->Name("Avx512Blocked/f64/N=4096");
BENCHMARK(BM_Avx512Blocked<64, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Blocked/f32/N=64");
BENCHMARK(BM_Avx512Blocked<256, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Blocked/f32/N=256");
BENCHMARK(BM_Avx512Blocked<512, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Blocked/f32/N=512");
BENCHMARK(BM_Avx512Blocked<1024, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Blocked/f32/N=1024");
BENCHMARK(BM_Avx512Blocked<4096, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("Avx512Blocked/f32/N=4096");

// ============================================================================
// NEON benchmarks
// On x86 targets each kernel transparently delegates to its AVX2 equivalent,
// so these registrations are always safe to include in the binary.
// On Apple Silicon / AArch64 the NEON code path is active.
// ============================================================================

/**
 * @brief Benchmark gemm_neon_naive<T> — i-j-k, NEON on k-loop.
 * Expected: GFLOP/s ≈ scalar naive — column-stride gather is still
 * cache-miss bound regardless of SIMD width.
 */
template <std::size_t N, typename T = double>
static void BM_NeonNaive(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_neon_naive(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
#ifdef __ARM_NEON
    state.counters["neon"] = 1;
#else
    state.counters["neon"] = 0;
#endif
}

/**
 * @brief Benchmark gemm_neon_reordered<T> — i-k-j, NEON on j-loop.
 * Expected: ~4× scalar reordered (f32, 4-wide NEON); ~2× (f64, 2-wide NEON).
 */
template <std::size_t N, typename T = double>
static void BM_NeonReordered(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_neon_reordered(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
#ifdef __ARM_NEON
    state.counters["neon"] = 1;
#else
    state.counters["neon"] = 0;
#endif
}

/**
 * @brief Benchmark gemm_neon_blocked<T> — tiled i-k-j + NEON register tile.
 * Expected: highest GFLOP/s on ARM. L2 tiling + 4×16 f32 C tile in Q registers.
 */
template <std::size_t N, typename T = double>
static void BM_NeonBlocked(benchmark::State& state) {
    hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::gemm_neon_blocked(A, B, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["N"]         = static_cast<double>(N);
    state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
#ifdef __ARM_NEON
    state.counters["neon"] = 1;
#else
    state.counters["neon"] = 0;
#endif
}

// ---- NEON Naive ------------------------------------------------------------
BENCHMARK(BM_NeonNaive<64>)->Unit(benchmark::kMicrosecond)->Name("NeonNaive/f64/N=64");
BENCHMARK(BM_NeonNaive<256>)->Unit(benchmark::kMicrosecond)->Name("NeonNaive/f64/N=256");
BENCHMARK(BM_NeonNaive<512>)->Unit(benchmark::kMicrosecond)->Name("NeonNaive/f64/N=512");
BENCHMARK(BM_NeonNaive<1024>)->Unit(benchmark::kMicrosecond)->Name("NeonNaive/f64/N=1024");
BENCHMARK(BM_NeonNaive<4096>)->Unit(benchmark::kMicrosecond)->Name("NeonNaive/f64/N=4096");
BENCHMARK(BM_NeonNaive<64, float>)->Unit(benchmark::kMicrosecond)->Name("NeonNaive/f32/N=64");
BENCHMARK(BM_NeonNaive<256, float>)->Unit(benchmark::kMicrosecond)->Name("NeonNaive/f32/N=256");
BENCHMARK(BM_NeonNaive<512, float>)->Unit(benchmark::kMicrosecond)->Name("NeonNaive/f32/N=512");
BENCHMARK(BM_NeonNaive<1024, float>)->Unit(benchmark::kMicrosecond)->Name("NeonNaive/f32/N=1024");
BENCHMARK(BM_NeonNaive<4096, float>)->Unit(benchmark::kMicrosecond)->Name("NeonNaive/f32/N=4096");

// ---- NEON Reordered --------------------------------------------------------
BENCHMARK(BM_NeonReordered<64>)->Unit(benchmark::kMicrosecond)->Name("NeonReordered/f64/N=64");
BENCHMARK(BM_NeonReordered<256>)->Unit(benchmark::kMicrosecond)->Name("NeonReordered/f64/N=256");
BENCHMARK(BM_NeonReordered<512>)->Unit(benchmark::kMicrosecond)->Name("NeonReordered/f64/N=512");
BENCHMARK(BM_NeonReordered<1024>)->Unit(benchmark::kMicrosecond)->Name("NeonReordered/f64/N=1024");
BENCHMARK(BM_NeonReordered<4096>)->Unit(benchmark::kMicrosecond)->Name("NeonReordered/f64/N=4096");
BENCHMARK(BM_NeonReordered<64, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("NeonReordered/f32/N=64");
BENCHMARK(BM_NeonReordered<256, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("NeonReordered/f32/N=256");
BENCHMARK(BM_NeonReordered<512, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("NeonReordered/f32/N=512");
BENCHMARK(BM_NeonReordered<1024, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("NeonReordered/f32/N=1024");
BENCHMARK(BM_NeonReordered<4096, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("NeonReordered/f32/N=4096");

// ---- NEON Blocked ----------------------------------------------------------
BENCHMARK(BM_NeonBlocked<64>)->Unit(benchmark::kMicrosecond)->Name("NeonBlocked/f64/N=64");
BENCHMARK(BM_NeonBlocked<256>)->Unit(benchmark::kMicrosecond)->Name("NeonBlocked/f64/N=256");
BENCHMARK(BM_NeonBlocked<512>)->Unit(benchmark::kMicrosecond)->Name("NeonBlocked/f64/N=512");
BENCHMARK(BM_NeonBlocked<1024>)->Unit(benchmark::kMicrosecond)->Name("NeonBlocked/f64/N=1024");
BENCHMARK(BM_NeonBlocked<4096>)->Unit(benchmark::kMicrosecond)->Name("NeonBlocked/f64/N=4096");
BENCHMARK(BM_NeonBlocked<64, float>)->Unit(benchmark::kMicrosecond)->Name("NeonBlocked/f32/N=64");
BENCHMARK(BM_NeonBlocked<256, float>)->Unit(benchmark::kMicrosecond)->Name("NeonBlocked/f32/N=256");
BENCHMARK(BM_NeonBlocked<512, float>)->Unit(benchmark::kMicrosecond)->Name("NeonBlocked/f32/N=512");
BENCHMARK(BM_NeonBlocked<1024, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("NeonBlocked/f32/N=1024");
BENCHMARK(BM_NeonBlocked<4096, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("NeonBlocked/f32/N=4096");

BENCHMARK_MAIN();
