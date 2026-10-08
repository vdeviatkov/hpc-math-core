/**
 * @file bench_gemm.cpp
 * @brief Google Benchmark driver for the CPU GEMM kernels, f32 and f64.
 *
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
 *   ./build/benchmarks/bench_gemm --benchmark_format=console
 *
 * Every kernel runs for both element types; f32 fits twice as many elements
 * per cache line and SIMD register. A square N×N GEMM performs 2·N³ FLOPs,
 * reported as the GFLOP/s counter: (2·N³) / (time_µs · 1e3).
 *
 * Matrices are allocated and filled outside the timed loop, so only the
 * kernel is measured.
 */

#include "gemm/amx.hpp"
#include "gemm/blocked.hpp"
#include "gemm/kleidiai.hpp"
#include "gemm/naive.hpp"
#include "gemm/neon.hpp"
#include "gemm/prefetch.hpp"
#include "gemm/reordered.hpp"
#include "gemm/sme.hpp"
#include "gemm/sve.hpp"
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
// ISA availability
//
// hpc/isa.hpp decides at compile time which kernel families exist in this
// build; on a target without an ISA that family's gemm_* functions are
// declared `= delete`. run_gemm() below therefore takes the availability
// flag as a template parameter: when it is false the benchmark reports
// SKIPPED and the `else` branch — the only place the kernel is named — is a
// discarded statement that is never instantiated. The benchmark name still
// appears in the output, so the full kernel catalogue is always visible and
// nothing can silently time a substitute kernel.
// ---------------------------------------------------------------------------

using hpc::kHaveAmx;
using hpc::kHaveAvx2;
using hpc::kHaveAvx512;
using hpc::kHaveNeon;
using hpc::kHaveSme;
using hpc::kHaveSve;

static constexpr const char* kNoAvx2   = "AVX2 not available on this target";
static constexpr const char* kNoAvx512 = "AVX-512 not available on this target";
static constexpr const char* kNoNeon   = "NEON not available on this target";
static constexpr const char* kNoSve    = "SVE not available on this target";
static constexpr const char* kNoSme    = "SME not available on this target (build with -DHPC_ENABLE_SME=ON on Apple M4+)";
static constexpr const char* kNoAmx    = "AMX not available (Accelerate.framework requires Apple platforms; "
                                         "build with -DHPC_ENABLE_AMX=ON, default on Apple)";

/// No-op `extra` for run_gemm.
static void no_extra_counters(benchmark::State&) {}

/**
 * @brief Time one N×N GEMM kernel, or report it as SKIPPED.
 *
 * @tparam N         Matrix dimension.
 * @tparam T         Element type (float / double).
 * @tparam Available Compile-time ISA flag (hpc::kHave*). When false the
 *                   kernel lambda is never instantiated.
 * @param  kernel    Generic callable `(A, B, C)` invoking the kernel — must be
 *                   a generic lambda so that the call stays dependent and is
 *                   only resolved inside the instantiated branch.
 * @param  extra     Callable adding kernel-specific counters (tile, vl, …).
 */
template <std::size_t N, typename T, bool Available, typename Kernel,
          typename Extra = void (*)(benchmark::State&)>
static void run_gemm(benchmark::State& state, const char* unavailable_msg, Kernel kernel,
                     Extra extra = no_extra_counters) {
    if constexpr (!Available) {
        (void)kernel;
        (void)extra;
        state.SkipWithMessage(unavailable_msg);
    } else {
        hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
        fill_random(A, 1);
        fill_random(B, 2);
        for (auto _ : state) {
            kernel(A, B, C);
            benchmark::DoNotOptimize(C.data());
            benchmark::ClobberMemory();
        }
        state.counters["GFLOP/s"] = benchmark::Counter(
            flops(N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
        state.counters["N"]         = static_cast<double>(N);
        state.counters["precision"] = static_cast<double>(sizeof(T) * 8);
        extra(state);
    }
}

// ---------------------------------------------------------------------------
// Benchmark templates — templated on both matrix size N and element type T.
// ---------------------------------------------------------------------------

/**
 * @brief Benchmark gemm_naive<T> for a given matrix size N.
 */
template <std::size_t N, typename T = double>
static void BM_Naive(benchmark::State& state) {
    run_gemm<N, T, true>(
        state, nullptr,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_naive(A, B, C); });
}

/**
 * @brief Benchmark gemm_reordered<T> for a given matrix size N.
 */
template <std::size_t N, typename T = double>
static void BM_Reordered(benchmark::State& state) {
    run_gemm<N, T, true>(
        state, nullptr,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_reordered(A, B, C); });
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
    run_gemm<N, T, true>(
        state, nullptr,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_blocked(A, B, C); },
        [](benchmark::State& s) { s.counters["tile"] = static_cast<double>(hpc::gemm::kDefaultTile); });
}

/**
 * @brief Benchmark gemm_avx2_naive<T> — i-j-k order, SIMD on the k-loop.
 *
 * About scalar-naive speed: the column-stride B access stays cache-hostile
 * regardless of SIMD width.
 */
template <std::size_t N, typename T = double>
static void BM_Avx2Naive(benchmark::State& state) {
    run_gemm<N, T, kHaveAvx2>(
        state, kNoAvx2,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_avx2_naive(A, B, C); });
}

/**
 * @brief Benchmark gemm_avx2_reordered<T> — i-k-j order, SIMD on the j-loop.
 *
 * Measured within a few percent of scalar gemm_reordered on Zen 5: the
 * compiler already auto-vectorises the scalar i-k-j loop at -O3 -ffast-math.
 * No blocking — degrades at large N when C row i is evicted from L1 between
 * k-iterations.
 */
template <std::size_t N, typename T = double>
static void BM_Avx2Reordered(benchmark::State& state) {
    run_gemm<N, T, kHaveAvx2>(
        state, kNoAvx2,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_avx2_reordered(A, B, C); });
}

/**
 * @brief Benchmark gemm_avx2_blocked<T> — tiled i-k-j, register-tiled micro-kernel.
 *
 * The fastest of the three: outer tiling keeps the working set in L2 and the
 * register tile removes C reload traffic.
 */
template <std::size_t N, typename T = double>
static void BM_Avx2Blocked(benchmark::State& state) {
    run_gemm<N, T, kHaveAvx2>(
        state, kNoAvx2,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_avx2_blocked(A, B, C); });
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
// Reported as SKIPPED on targets without AVX-512.
// ============================================================================

/**
 * @brief Benchmark gemm_avx512_naive<T> — i-j-k, 512-bit SIMD on k-loop.
 * About scalar-naive speed: the gather is still cache-miss bound.
 */
template <std::size_t N, typename T = double>
static void BM_Avx512Naive(benchmark::State& state) {
    run_gemm<N, T, kHaveAvx512>(
        state, kNoAvx512,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_avx512_naive(A, B, C); });
}

/**
 * @brief Benchmark gemm_avx512_reordered<T> — i-k-j, 512-bit SIMD on j-loop.
 * Measured: within a few percent of scalar/AVX2 reordered on Zen 5 — the loop
 * is limited by memory traffic, not FMA width.
 */
template <std::size_t N, typename T = double>
static void BM_Avx512Reordered(benchmark::State& state) {
    run_gemm<N, T, kHaveAvx512>(
        state, kNoAvx512,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_avx512_reordered(A, B, C); });
}

/**
 * @brief Benchmark gemm_avx512_blocked<T> — tiled i-k-j + 512-bit register tile.
 * L2 tiling + 4×32 f32 C tile held in ZMM registers; the fastest AVX-512 kernel.
 */
template <std::size_t N, typename T = double>
static void BM_Avx512Blocked(benchmark::State& state) {
    run_gemm<N, T, kHaveAvx512>(
        state, kNoAvx512,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_avx512_blocked(A, B, C); });
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
// Reported as SKIPPED on targets without AArch64 NEON.
// ============================================================================

/**
 * @brief Benchmark gemm_neon_naive<T> — i-j-k, NEON on k-loop.
 * About scalar-naive speed: the column-stride gather is still cache-miss bound.
 */
template <std::size_t N, typename T = double>
static void BM_NeonNaive(benchmark::State& state) {
    run_gemm<N, T, kHaveNeon>(
        state, kNoNeon,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_neon_naive(A, B, C); });
}

/**
 * @brief Benchmark gemm_neon_reordered<T> — i-k-j, NEON on j-loop.
 * Measured on M4 Max: no faster than scalar gemm_reordered, which the compiler
 * auto-vectorises (e.g. f32 N=256: 29.6 vs 32.3 GFLOP/s).
 */
template <std::size_t N, typename T = double>
static void BM_NeonReordered(benchmark::State& state) {
    run_gemm<N, T, kHaveNeon>(
        state, kNoNeon,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_neon_reordered(A, B, C); });
}

/**
 * @brief Benchmark gemm_neon_blocked<T> — tiled i-k-j + NEON register tile.
 * L2 tiling + 4×16 f32 C tile in Q registers; the fastest NEON kernel.
 */
template <std::size_t N, typename T = double>
static void BM_NeonBlocked(benchmark::State& state) {
    run_gemm<N, T, kHaveNeon>(
        state, kNoNeon,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_neon_blocked(A, B, C); });
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

// ============================================================================
// SVE / SVE2 benchmarks
// Reported as SKIPPED on targets without SVE (x86, Apple Silicon).
//
// The "vl" counter reports the actual SVE vector length at runtime:
//   128-bit SVE: vl=4 (f32) / vl=2 (f64)
//   256-bit SVE: vl=8 (f32) / vl=4 (f64)  ← Graviton3, Neoverse V1
//   512-bit SVE: vl=16(f32) / vl=8 (f64)  ← A64FX (Fugaku)
// ============================================================================

/**
 * @brief Benchmark gemm_sve_naive<T> — i-j-k, SVE on k-loop (VLA gather).
 * Not measured (no SVE hardware); the column gather is cache-miss bound.
 */
template <std::size_t N, typename T = double>
static void BM_SveNaive(benchmark::State& state) {
    run_gemm<N, T, kHaveSve>(
        state, kNoSve,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_sve_naive(A, B, C); },
        []([[maybe_unused]] benchmark::State& s) {
#if HPC_HAS_SVE
            s.counters["vl"] = static_cast<double>((sizeof(T) == 4) ? svcntw() : svcntd());
#endif
        });
}

/**
 * @brief Benchmark gemm_sve_reordered<T> — i-k-j, VLA SVE on j-loop.
 * Not measured (no SVE hardware). Predicated tail — no scalar remainder loop.
 */
template <std::size_t N, typename T = double>
static void BM_SveReordered(benchmark::State& state) {
    run_gemm<N, T, kHaveSve>(
        state, kNoSve,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_sve_reordered(A, B, C); },
        []([[maybe_unused]] benchmark::State& s) {
#if HPC_HAS_SVE
            s.counters["vl"] = static_cast<double>((sizeof(T) == 4) ? svcntw() : svcntd());
#endif
        });
}

/**
 * @brief Benchmark gemm_sve_blocked<T> — tiled i-k-j + VLA register tile.
 * Not measured (no SVE hardware). Tile width scales with the hardware VL.
 */
template <std::size_t N, typename T = double>
static void BM_SveBlocked(benchmark::State& state) {
    run_gemm<N, T, kHaveSve>(
        state, kNoSve,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_sve_blocked(A, B, C); },
        []([[maybe_unused]] benchmark::State& s) {
#if HPC_HAS_SVE
            s.counters["vl"] = static_cast<double>((sizeof(T) == 4) ? svcntw() : svcntd());
#endif
        });
}

// ---- SVE Naive -------------------------------------------------------------
BENCHMARK(BM_SveNaive<64>)->Unit(benchmark::kMicrosecond)->Name("SveNaive/f64/N=64");
BENCHMARK(BM_SveNaive<256>)->Unit(benchmark::kMicrosecond)->Name("SveNaive/f64/N=256");
BENCHMARK(BM_SveNaive<512>)->Unit(benchmark::kMicrosecond)->Name("SveNaive/f64/N=512");
BENCHMARK(BM_SveNaive<1024>)->Unit(benchmark::kMicrosecond)->Name("SveNaive/f64/N=1024");
BENCHMARK(BM_SveNaive<4096>)->Unit(benchmark::kMicrosecond)->Name("SveNaive/f64/N=4096");
BENCHMARK(BM_SveNaive<64, float>)->Unit(benchmark::kMicrosecond)->Name("SveNaive/f32/N=64");
BENCHMARK(BM_SveNaive<256, float>)->Unit(benchmark::kMicrosecond)->Name("SveNaive/f32/N=256");
BENCHMARK(BM_SveNaive<512, float>)->Unit(benchmark::kMicrosecond)->Name("SveNaive/f32/N=512");
BENCHMARK(BM_SveNaive<1024, float>)->Unit(benchmark::kMicrosecond)->Name("SveNaive/f32/N=1024");
BENCHMARK(BM_SveNaive<4096, float>)->Unit(benchmark::kMicrosecond)->Name("SveNaive/f32/N=4096");

// ---- SVE Reordered ---------------------------------------------------------
BENCHMARK(BM_SveReordered<64>)->Unit(benchmark::kMicrosecond)->Name("SveReordered/f64/N=64");
BENCHMARK(BM_SveReordered<256>)->Unit(benchmark::kMicrosecond)->Name("SveReordered/f64/N=256");
BENCHMARK(BM_SveReordered<512>)->Unit(benchmark::kMicrosecond)->Name("SveReordered/f64/N=512");
BENCHMARK(BM_SveReordered<1024>)->Unit(benchmark::kMicrosecond)->Name("SveReordered/f64/N=1024");
BENCHMARK(BM_SveReordered<4096>)->Unit(benchmark::kMicrosecond)->Name("SveReordered/f64/N=4096");
BENCHMARK(BM_SveReordered<64, float>)->Unit(benchmark::kMicrosecond)->Name("SveReordered/f32/N=64");
BENCHMARK(BM_SveReordered<256, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("SveReordered/f32/N=256");
BENCHMARK(BM_SveReordered<512, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("SveReordered/f32/N=512");
BENCHMARK(BM_SveReordered<1024, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("SveReordered/f32/N=1024");
BENCHMARK(BM_SveReordered<4096, float>)
    ->Unit(benchmark::kMicrosecond)
    ->Name("SveReordered/f32/N=4096");

// ---- SVE Blocked -----------------------------------------------------------
BENCHMARK(BM_SveBlocked<64>)->Unit(benchmark::kMicrosecond)->Name("SveBlocked/f64/N=64");
BENCHMARK(BM_SveBlocked<256>)->Unit(benchmark::kMicrosecond)->Name("SveBlocked/f64/N=256");
BENCHMARK(BM_SveBlocked<512>)->Unit(benchmark::kMicrosecond)->Name("SveBlocked/f64/N=512");
BENCHMARK(BM_SveBlocked<1024>)->Unit(benchmark::kMicrosecond)->Name("SveBlocked/f64/N=1024");
BENCHMARK(BM_SveBlocked<4096>)->Unit(benchmark::kMicrosecond)->Name("SveBlocked/f64/N=4096");
BENCHMARK(BM_SveBlocked<64, float>)->Unit(benchmark::kMicrosecond)->Name("SveBlocked/f32/N=64");
BENCHMARK(BM_SveBlocked<256, float>)->Unit(benchmark::kMicrosecond)->Name("SveBlocked/f32/N=256");
BENCHMARK(BM_SveBlocked<512, float>)->Unit(benchmark::kMicrosecond)->Name("SveBlocked/f32/N=512");
BENCHMARK(BM_SveBlocked<1024, float>)->Unit(benchmark::kMicrosecond)->Name("SveBlocked/f32/N=1024");
BENCHMARK(BM_SveBlocked<4096, float>)->Unit(benchmark::kMicrosecond)->Name("SveBlocked/f32/N=4096");

// ============================================================================
// SME (Scalable Matrix Extension) benchmark
// On SME hardware (Apple M4/M4 Pro/M4 Max with -DHPC_ENABLE_SME=ON) this is
// the packed, cache-blocked, all-ZA-tiles outer-product kernel — see
// src/gemm/sme.hpp. Elsewhere it reports SKIPPED.
//
// Single-threaded. The fair library comparisons are Accelerate with
// VECLIB_MAXIMUM_THREADS=1 and KleidiAI (always single-threaded).
//
// The "svl" counter reports the streaming vector length at runtime
// (elements per ZA-tile row/column, via svcntsw()/svcntsd()):
//   512-bit SVL: svl=16 (f32) / svl=8 (f64)  ← Apple M4/M4 Pro/M4 Max
// ============================================================================

/**
 * @brief Benchmark gemm_sme<T> — 2×2 za32 / 2×4 za64 tiles, packed A and B.
 */
template <std::size_t N, typename T = double>
static void BM_Sme(benchmark::State& state) {
    run_gemm<N, T, kHaveSme>(
        state, kNoSme,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_sme(A, B, C); },
        []([[maybe_unused]] benchmark::State& s) {
#if HPC_HAS_SME
            s.counters["svl"] = static_cast<double>((sizeof(T) == 4) ? svcntsw() : svcntsd());
#endif
        });
}

BENCHMARK(BM_Sme<64>)->Unit(benchmark::kMicrosecond)->Name("Sme/f64/N=64");
BENCHMARK(BM_Sme<256>)->Unit(benchmark::kMicrosecond)->Name("Sme/f64/N=256");
BENCHMARK(BM_Sme<512>)->Unit(benchmark::kMicrosecond)->Name("Sme/f64/N=512");
BENCHMARK(BM_Sme<1024>)->Unit(benchmark::kMicrosecond)->Name("Sme/f64/N=1024");
BENCHMARK(BM_Sme<2048>)->Unit(benchmark::kMicrosecond)->Name("Sme/f64/N=2048");
BENCHMARK(BM_Sme<4096>)->Unit(benchmark::kMicrosecond)->Name("Sme/f64/N=4096");
BENCHMARK(BM_Sme<64, float>)->Unit(benchmark::kMicrosecond)->Name("Sme/f32/N=64");
BENCHMARK(BM_Sme<256, float>)->Unit(benchmark::kMicrosecond)->Name("Sme/f32/N=256");
BENCHMARK(BM_Sme<512, float>)->Unit(benchmark::kMicrosecond)->Name("Sme/f32/N=512");
BENCHMARK(BM_Sme<1024, float>)->Unit(benchmark::kMicrosecond)->Name("Sme/f32/N=1024");
BENCHMARK(BM_Sme<2048, float>)->Unit(benchmark::kMicrosecond)->Name("Sme/f32/N=2048");
BENCHMARK(BM_Sme<4096, float>)->Unit(benchmark::kMicrosecond)->Name("Sme/f32/N=4096");

// ============================================================================
// AMX (Apple Matrix coprocessor, via Accelerate.framework) benchmark
//
// gemm_amx is a thin wrapper around Accelerate's cblas_sgemm / cblas_dgemm
// (Apple's own vendor-tuned BLAS) — see src/gemm/amx.hpp. Accelerate may use
// multiple cores internally (unlike every other, strictly single-threaded,
// CPU kernel in this repo), so it is timed with UseRealTime(); set
// VECLIB_MAXIMUM_THREADS=1 for a single-core comparison.
//
// Full fp32/fp64 precision throughout — unlike gemm_cuda_wmma, Accelerate's
// BLAS does not force a reduced-precision input format.
// ============================================================================

template <std::size_t N, typename T = double>
static void BM_Amx(benchmark::State& state) {
    run_gemm<N, T, kHaveAmx>(
        state, kNoAmx,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_amx(A, B, C); });
}

BENCHMARK(BM_Amx<64>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f64/N=64");
BENCHMARK(BM_Amx<256>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f64/N=256");
BENCHMARK(BM_Amx<512>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f64/N=512");
BENCHMARK(BM_Amx<1024>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f64/N=1024");
BENCHMARK(BM_Amx<2048>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f64/N=2048");
BENCHMARK(BM_Amx<4096>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f64/N=4096");
BENCHMARK(BM_Amx<64, float>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f32/N=64");
BENCHMARK(BM_Amx<256, float>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f32/N=256");
BENCHMARK(BM_Amx<512, float>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f32/N=512");
BENCHMARK(BM_Amx<1024, float>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f32/N=1024");
BENCHMARK(BM_Amx<2048, float>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f32/N=2048");
BENCHMARK(BM_Amx<4096, float>)->UseRealTime()->Unit(benchmark::kMicrosecond)->Name("Amx/f32/N=4096");

// ============================================================================
// Reference library: Arm KleidiAI
//
// KleidiAI (src/gemm/kleidiai.hpp) — Arm's SME2 assembly micro-kernel,
//   single-threaded, f32 only (no f64 rows). Times include LHS/RHS packing,
//   as gemm_sme's do.
//
// Accelerate (the Amx rows above) is multi-threaded, so it is timed with
// UseRealTime(): Google Benchmark's default is the *calling thread's* CPU
// time, which overstates throughput when worker threads do the work while
// the caller sleeps.
// ============================================================================

static constexpr const char* kNoKleidiAI = "KleidiAI not built (needs SME2; -DHPC_ENABLE_KLEIDIAI=ON)";

template <std::size_t N>
static void BM_KleidiAI(benchmark::State& state) {
    run_gemm<N, float, hpc::kHaveKleidiAI>(
        state, kNoKleidiAI,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_kleidiai(A, B, C); });
}

BENCHMARK(BM_KleidiAI<64>)->Unit(benchmark::kMicrosecond)->Name("KleidiAI/f32/N=64");
BENCHMARK(BM_KleidiAI<256>)->Unit(benchmark::kMicrosecond)->Name("KleidiAI/f32/N=256");
BENCHMARK(BM_KleidiAI<512>)->Unit(benchmark::kMicrosecond)->Name("KleidiAI/f32/N=512");
BENCHMARK(BM_KleidiAI<1024>)->Unit(benchmark::kMicrosecond)->Name("KleidiAI/f32/N=1024");
BENCHMARK(BM_KleidiAI<2048>)->Unit(benchmark::kMicrosecond)->Name("KleidiAI/f32/N=2048");
BENCHMARK(BM_KleidiAI<4096>)->Unit(benchmark::kMicrosecond)->Name("KleidiAI/f32/N=4096");

// ============================================================================
// Software-prefetch benchmark templates
//
// Only the blocked kernel of each family gets a prefetch variant: the naive
// kernels are bound by cache misses prefetch can't hide, and the reordered
// ones stream data the hardware prefetcher already follows.
//
// Naming convention:  <Family>BlockedPf<D>/<prec>/N=<size>
//   D  = prefetch distance in micro-kernel rows (2, 4, 8, 16)
//
// Each template is also ISA-guarded via the same kHave* flags as its base
// kernel — benchmarks for absent ISAs appear as SKIPPED, not as timing.
//
// ============================================================================

// ---------------------------------------------------------------------------
// Prefetch distance sweep helper — one template per (ISA, PfDist)
// ---------------------------------------------------------------------------

/// Scalar blocked + prefetch, distance PfDist.
template <std::size_t N, typename T = double, std::size_t PfDist = hpc::gemm::kDefaultPrefetchDist>
static void BM_BlockedPf(benchmark::State& state) {
    run_gemm<N, T, true>(
        state, nullptr,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_blocked_prefetch<T, PfDist>(A, B, C); },
        [](benchmark::State& s) { s.counters["pf_dist"] = static_cast<double>(PfDist); });
}

/// AVX2 blocked + prefetch, distance PfDist.
template <std::size_t N, typename T = double, std::size_t PfDist = hpc::gemm::kDefaultPrefetchDist>
static void BM_Avx2BlockedPf(benchmark::State& state) {
    run_gemm<N, T, kHaveAvx2>(
        state, kNoAvx2,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_avx2_blocked_prefetch<T, PfDist>(A, B, C); },
        [](benchmark::State& s) { s.counters["pf_dist"] = static_cast<double>(PfDist); });
}

/// AVX-512 blocked + prefetch, distance PfDist.
template <std::size_t N, typename T = double, std::size_t PfDist = hpc::gemm::kDefaultPrefetchDist>
static void BM_Avx512BlockedPf(benchmark::State& state) {
    run_gemm<N, T, kHaveAvx512>(
        state, kNoAvx512,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_avx512_blocked_prefetch<T, PfDist>(A, B, C); },
        [](benchmark::State& s) { s.counters["pf_dist"] = static_cast<double>(PfDist); });
}

/// NEON blocked + prefetch, distance PfDist.
template <std::size_t N, typename T = double, std::size_t PfDist = hpc::gemm::kDefaultPrefetchDist>
static void BM_NeonBlockedPf(benchmark::State& state) {
    run_gemm<N, T, kHaveNeon>(
        state, kNoNeon,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_neon_blocked_prefetch<T, PfDist>(A, B, C); },
        [](benchmark::State& s) { s.counters["pf_dist"] = static_cast<double>(PfDist); });
}

/// SVE blocked + prefetch, distance PfDist.
template <std::size_t N, typename T = double, std::size_t PfDist = hpc::gemm::kDefaultPrefetchDist>
static void BM_SveBlockedPf(benchmark::State& state) {
    run_gemm<N, T, kHaveSve>(
        state, kNoSve,
        [](auto& A, auto& B, auto& C) { hpc::gemm::gemm_sve_blocked_prefetch<T, PfDist>(A, B, C); },
        [](benchmark::State& s) { s.counters["pf_dist"] = static_cast<double>(PfDist); });
}

// ---------------------------------------------------------------------------
// Registrations — distance sweep: D2 / D4 / D8 / D16
// Sizes: N = 256, 512, 1024.
//
// To run only this family:
//   ./build/benchmarks/bench_gemm --benchmark_filter="BlockedPf"
// ---------------------------------------------------------------------------

// ---- Scalar + prefetch (always runs) ----------------------------------------
#define HPC_REG_BLOCKED_PF(N, T, D, TNAME, PNAME)                                         \
    BENCHMARK((BM_BlockedPf<N, T, D>))->Unit(benchmark::kMicrosecond)                     \
        ->Name("BlockedPf" #D "/" PNAME "/N=" #N);

HPC_REG_BLOCKED_PF(256,  double, 2,  f64, "f64")
HPC_REG_BLOCKED_PF(256,  double, 4,  f64, "f64")
HPC_REG_BLOCKED_PF(256,  double, 8,  f64, "f64")
HPC_REG_BLOCKED_PF(256,  double, 16, f64, "f64")
HPC_REG_BLOCKED_PF(512,  double, 2,  f64, "f64")
HPC_REG_BLOCKED_PF(512,  double, 4,  f64, "f64")
HPC_REG_BLOCKED_PF(512,  double, 8,  f64, "f64")
HPC_REG_BLOCKED_PF(512,  double, 16, f64, "f64")
HPC_REG_BLOCKED_PF(1024, double, 2,  f64, "f64")
HPC_REG_BLOCKED_PF(1024, double, 4,  f64, "f64")
HPC_REG_BLOCKED_PF(1024, double, 8,  f64, "f64")
HPC_REG_BLOCKED_PF(1024, double, 16, f64, "f64")

HPC_REG_BLOCKED_PF(256,  float, 2,  f32, "f32")
HPC_REG_BLOCKED_PF(256,  float, 4,  f32, "f32")
HPC_REG_BLOCKED_PF(256,  float, 8,  f32, "f32")
HPC_REG_BLOCKED_PF(256,  float, 16, f32, "f32")
HPC_REG_BLOCKED_PF(512,  float, 2,  f32, "f32")
HPC_REG_BLOCKED_PF(512,  float, 4,  f32, "f32")
HPC_REG_BLOCKED_PF(512,  float, 8,  f32, "f32")
HPC_REG_BLOCKED_PF(512,  float, 16, f32, "f32")
HPC_REG_BLOCKED_PF(1024, float, 2,  f32, "f32")
HPC_REG_BLOCKED_PF(1024, float, 4,  f32, "f32")
HPC_REG_BLOCKED_PF(1024, float, 8,  f32, "f32")
HPC_REG_BLOCKED_PF(1024, float, 16, f32, "f32")

#undef HPC_REG_BLOCKED_PF

// ---- AVX2 + prefetch -------------------------------------------------------
#define HPC_REG_AVX2_PF(N, T, D, PNAME)                                                   \
    BENCHMARK((BM_Avx2BlockedPf<N, T, D>))->Unit(benchmark::kMicrosecond)                 \
        ->Name("Avx2BlockedPf" #D "/" PNAME "/N=" #N);

HPC_REG_AVX2_PF(256,  double, 2,  "f64") HPC_REG_AVX2_PF(256,  double, 4,  "f64")
HPC_REG_AVX2_PF(256,  double, 8,  "f64") HPC_REG_AVX2_PF(256,  double, 16, "f64")
HPC_REG_AVX2_PF(512,  double, 2,  "f64") HPC_REG_AVX2_PF(512,  double, 4,  "f64")
HPC_REG_AVX2_PF(512,  double, 8,  "f64") HPC_REG_AVX2_PF(512,  double, 16, "f64")
HPC_REG_AVX2_PF(1024, double, 2,  "f64") HPC_REG_AVX2_PF(1024, double, 4,  "f64")
HPC_REG_AVX2_PF(1024, double, 8,  "f64") HPC_REG_AVX2_PF(1024, double, 16, "f64")

HPC_REG_AVX2_PF(256,  float, 2,  "f32") HPC_REG_AVX2_PF(256,  float, 4,  "f32")
HPC_REG_AVX2_PF(256,  float, 8,  "f32") HPC_REG_AVX2_PF(256,  float, 16, "f32")
HPC_REG_AVX2_PF(512,  float, 2,  "f32") HPC_REG_AVX2_PF(512,  float, 4,  "f32")
HPC_REG_AVX2_PF(512,  float, 8,  "f32") HPC_REG_AVX2_PF(512,  float, 16, "f32")
HPC_REG_AVX2_PF(1024, float, 2,  "f32") HPC_REG_AVX2_PF(1024, float, 4,  "f32")
HPC_REG_AVX2_PF(1024, float, 8,  "f32") HPC_REG_AVX2_PF(1024, float, 16, "f32")

#undef HPC_REG_AVX2_PF

// ---- AVX-512 + prefetch ----------------------------------------------------
#define HPC_REG_AVX512_PF(N, T, D, PNAME)                                                 \
    BENCHMARK((BM_Avx512BlockedPf<N, T, D>))->Unit(benchmark::kMicrosecond)               \
        ->Name("Avx512BlockedPf" #D "/" PNAME "/N=" #N);

HPC_REG_AVX512_PF(256,  double, 2,  "f64") HPC_REG_AVX512_PF(256,  double, 4,  "f64")
HPC_REG_AVX512_PF(256,  double, 8,  "f64") HPC_REG_AVX512_PF(256,  double, 16, "f64")
HPC_REG_AVX512_PF(512,  double, 2,  "f64") HPC_REG_AVX512_PF(512,  double, 4,  "f64")
HPC_REG_AVX512_PF(512,  double, 8,  "f64") HPC_REG_AVX512_PF(512,  double, 16, "f64")
HPC_REG_AVX512_PF(1024, double, 2,  "f64") HPC_REG_AVX512_PF(1024, double, 4,  "f64")
HPC_REG_AVX512_PF(1024, double, 8,  "f64") HPC_REG_AVX512_PF(1024, double, 16, "f64")

HPC_REG_AVX512_PF(256,  float, 2,  "f32") HPC_REG_AVX512_PF(256,  float, 4,  "f32")
HPC_REG_AVX512_PF(256,  float, 8,  "f32") HPC_REG_AVX512_PF(256,  float, 16, "f32")
HPC_REG_AVX512_PF(512,  float, 2,  "f32") HPC_REG_AVX512_PF(512,  float, 4,  "f32")
HPC_REG_AVX512_PF(512,  float, 8,  "f32") HPC_REG_AVX512_PF(512,  float, 16, "f32")
HPC_REG_AVX512_PF(1024, float, 2,  "f32") HPC_REG_AVX512_PF(1024, float, 4,  "f32")
HPC_REG_AVX512_PF(1024, float, 8,  "f32") HPC_REG_AVX512_PF(1024, float, 16, "f32")

#undef HPC_REG_AVX512_PF

// ---- NEON + prefetch -------------------------------------------------------
#define HPC_REG_NEON_PF(N, T, D, PNAME)                                                   \
    BENCHMARK((BM_NeonBlockedPf<N, T, D>))->Unit(benchmark::kMicrosecond)                 \
        ->Name("NeonBlockedPf" #D "/" PNAME "/N=" #N);

HPC_REG_NEON_PF(256,  double, 2,  "f64") HPC_REG_NEON_PF(256,  double, 4,  "f64")
HPC_REG_NEON_PF(256,  double, 8,  "f64") HPC_REG_NEON_PF(256,  double, 16, "f64")
HPC_REG_NEON_PF(512,  double, 2,  "f64") HPC_REG_NEON_PF(512,  double, 4,  "f64")
HPC_REG_NEON_PF(512,  double, 8,  "f64") HPC_REG_NEON_PF(512,  double, 16, "f64")
HPC_REG_NEON_PF(1024, double, 2,  "f64") HPC_REG_NEON_PF(1024, double, 4,  "f64")
HPC_REG_NEON_PF(1024, double, 8,  "f64") HPC_REG_NEON_PF(1024, double, 16, "f64")

HPC_REG_NEON_PF(256,  float, 2,  "f32") HPC_REG_NEON_PF(256,  float, 4,  "f32")
HPC_REG_NEON_PF(256,  float, 8,  "f32") HPC_REG_NEON_PF(256,  float, 16, "f32")
HPC_REG_NEON_PF(512,  float, 2,  "f32") HPC_REG_NEON_PF(512,  float, 4,  "f32")
HPC_REG_NEON_PF(512,  float, 8,  "f32") HPC_REG_NEON_PF(512,  float, 16, "f32")
HPC_REG_NEON_PF(1024, float, 2,  "f32") HPC_REG_NEON_PF(1024, float, 4,  "f32")
HPC_REG_NEON_PF(1024, float, 8,  "f32") HPC_REG_NEON_PF(1024, float, 16, "f32")

#undef HPC_REG_NEON_PF

// ---- SVE + prefetch --------------------------------------------------------
#define HPC_REG_SVE_PF(N, T, D, PNAME)                                                    \
    BENCHMARK((BM_SveBlockedPf<N, T, D>))->Unit(benchmark::kMicrosecond)                  \
        ->Name("SveBlockedPf" #D "/" PNAME "/N=" #N);

HPC_REG_SVE_PF(256,  double, 2,  "f64") HPC_REG_SVE_PF(256,  double, 4,  "f64")
HPC_REG_SVE_PF(256,  double, 8,  "f64") HPC_REG_SVE_PF(256,  double, 16, "f64")
HPC_REG_SVE_PF(512,  double, 2,  "f64") HPC_REG_SVE_PF(512,  double, 4,  "f64")
HPC_REG_SVE_PF(512,  double, 8,  "f64") HPC_REG_SVE_PF(512,  double, 16, "f64")
HPC_REG_SVE_PF(1024, double, 2,  "f64") HPC_REG_SVE_PF(1024, double, 4,  "f64")
HPC_REG_SVE_PF(1024, double, 8,  "f64") HPC_REG_SVE_PF(1024, double, 16, "f64")

HPC_REG_SVE_PF(256,  float, 2,  "f32") HPC_REG_SVE_PF(256,  float, 4,  "f32")
HPC_REG_SVE_PF(256,  float, 8,  "f32") HPC_REG_SVE_PF(256,  float, 16, "f32")
HPC_REG_SVE_PF(512,  float, 2,  "f32") HPC_REG_SVE_PF(512,  float, 4,  "f32")
HPC_REG_SVE_PF(512,  float, 8,  "f32") HPC_REG_SVE_PF(512,  float, 16, "f32")
HPC_REG_SVE_PF(1024, float, 2,  "f32") HPC_REG_SVE_PF(1024, float, 4,  "f32")
HPC_REG_SVE_PF(1024, float, 8,  "f32") HPC_REG_SVE_PF(1024, float, 16, "f32")

#undef HPC_REG_SVE_PF

BENCHMARK_MAIN();
