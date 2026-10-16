/**
 * @file fma_peak.cpp
 * @brief Vector FMA latency and throughput of one CPU core.
 *
 *   ./build/benchmarks/fma_peak > fma.csv
 *
 * A loop applies acc[c] = fma(acc[c], a, b) to C independent accumulators in
 * turn, on constant vectors, touching no memory. With one accumulator every
 * FMA waits for the previous one, so the time per FMA is its latency; with
 * enough accumulators the FMAs overlap and the time per FMA is the issue
 * rate of the core's FMA pipes. The number of accumulators at which the two
 * meet is latency × pipes: the number of independent accumulators a GEMM
 * micro-kernel needs to keep the FMA units busy.
 *
 * Runs every vector ISA the build targets: NEON (128-bit) on AArch64, AVX2
 * (256-bit) and AVX-512 (512-bit) on x86-64; f32 and f64. Best of 5 runs;
 * stdout gets CSV rows, with times in cycles of the core clock measured at
 * the start (core_clock.hpp).
 */

#include "core_clock.hpp"
#include "simd_isa.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace {

using namespace hpc::experiments;

// Keeps a vector in a register and opaque to the optimiser, so the chain of
// FMAs on it cannot be folded or reordered. Emits no instruction.
template <typename V>
inline void opaque(V& v) {
#if defined(__aarch64__)
    asm volatile("" : "+w"(v));
#else
    asm volatile("" : "+v"(v));
#endif
}

constexpr long kIters = 20'000'000;

// Named accumulators, not an array: an array indexed in the loop can end up
// in memory, and the loop then measures store-to-load forwarding instead of
// the FMA units.
#define HPC_ACC(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) \
    X(12) X(13) X(14) X(15) X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23)

/// Seconds for kIters rounds of one FMA on each of the C accumulators (best of 5).
template <typename Isa, int C>
double run() {
    static_assert(C >= 1 && C <= 24);
    using V   = typename Isa::V;
    const V a = Isa::splat(0.999f), b = Isa::splat(1e-6f);
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
#define HPC_DECL(n) V acc##n = Isa::splat(float(n));
        HPC_ACC(HPC_DECL)
#undef HPC_DECL
        const auto t0 = std::chrono::steady_clock::now();
        for (long i = 0; i < kIters; ++i) {
#define HPC_STEP(n)                               \
    if constexpr (C > n) {                        \
        acc##n = Isa::fma(acc##n, a, b);          \
        opaque(acc##n);                           \
    }
            HPC_ACC(HPC_STEP)
#undef HPC_STEP
        }
        const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
        best = std::min(best, dt.count());
#define HPC_SINK(n) opaque(acc##n);
        HPC_ACC(HPC_SINK)
#undef HPC_SINK
    }
    return best;
}

#undef HPC_ACC

double g_ghz = 0;

template <typename Isa, int C>
void report() {
    const double s       = run<Isa, C>();
    const double fmas    = double(kIters) * C;
    const double cycles  = s * g_ghz * 1e9;
    const double per_fma = cycles / fmas;
    std::printf("%s,f%d,%d,%d,%.3f,%.2f,%.1f\n", Isa::isa, Isa::bits, Isa::lanes, C,
                cycles / double(kIters), 1.0 / per_fma, fmas * Isa::lanes * 2 / s / 1e9);
}

template <typename Isa, int... Cs>
void sweep() {
    (report<Isa, Cs>(), ...);
}

}  // namespace

int main() {
    g_ghz = hpc::experiments::core_ghz();
    std::fprintf(stderr, "core clock: %.3f GHz\n", g_ghz);
    std::printf("isa,type,lanes,accumulators,cycles_per_round,fma_per_cycle,gflops\n");
#if HPC_HAS_NEON
    sweep<NeonF32, 1, 2, 4, 8, 12, 16, 20, 24>();
    sweep<NeonF64, 1, 2, 4, 8, 12, 16, 20, 24>();
#endif
#if HPC_HAS_AVX2
    sweep<Avx2F32, 1, 2, 4, 6, 8, 10, 12>();
    sweep<Avx2F64, 1, 2, 4, 6, 8, 10, 12>();
#endif
#if HPC_HAS_AVX512
    sweep<Avx512F32, 1, 2, 4, 6, 8, 10, 12, 16, 24>();
    sweep<Avx512F64, 1, 2, 4, 6, 8, 10, 12, 16, 24>();
#endif
    std::fprintf(stderr, "core clock after: %.3f GHz\n", hpc::experiments::core_ghz());
}
