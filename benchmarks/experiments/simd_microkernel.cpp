/**
 * @file simd_microkernel.cpp
 * @brief The GEMM register-tile micro-kernel on L1-resident data: how many
 *        accumulators, and which B layout, reach the FMA peak.
 *
 *   ./build/benchmarks/simd_microkernel > microkernel.csv
 *
 * The kernel is the one in neon.hpp / avx2.hpp / avx512.hpp: an R x (V·W)
 * block of C held in R·V vector accumulators, and per k step V vector loads
 * of B, R broadcasts of A and R·V FMAs, for k = 0 … 255 (the kernels' k
 * tile). It runs repeatedly on the same small A, B and C, so the operands
 * come from L1 and nothing but the core is measured. Two variables:
 *
 *   rows R      the number of accumulators is R·V; the FMA latency × the
 *               number of FMA pipes is how many it takes to keep them busy
 *               (fma_peak).
 *   B layout    "packed": B's rows are contiguous, V·W elements each.
 *               "ldb=N":  B's rows are N elements apart, as when the
 *               library kernels read B in place from an N x N matrix.
 *
 * Best of 5; stdout gets CSV rows, GFLOP/s and cycles per k step at the core
 * clock measured at the start (core_clock.hpp).
 */

#include "core_clock.hpp"
#include "simd_isa.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace {

using namespace hpc::experiments;

constexpr int kK = 256;  // k tile of the library kernels

/// C[R][V·W] += A[R][kK] · B[kK][V·W], B rows ldb apart. Fully unrolled over
/// R and V so the accumulators live in registers.
template <typename Isa, int R, int V>
__attribute__((noinline)) void micro(const typename Isa::T* a, const typename Isa::T* b,
                                     typename Isa::T* c, int ldb) {
    using Vec            = typename Isa::V;
    constexpr int W      = Isa::lanes;
    Vec acc[R][V];
#pragma GCC unroll 16
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            acc[r][v] = Isa::load(c + r * V * W + v * W);
    for (int k = 0; k < kK; ++k) {
        Vec bv[V];
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            bv[v] = Isa::load(b + k * ldb + v * W);
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r) {
            const Vec ar = Isa::splat(a[r * kK + k]);
#pragma GCC unroll 16
            for (int v = 0; v < V; ++v)
                acc[r][v] = Isa::fma(acc[r][v], ar, bv[v]);
        }
    }
#pragma GCC unroll 16
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
        for (int v = 0; v < V; ++v)
            Isa::store(c + r * V * W + v * W, acc[r][v]);
}

double g_ghz = 0;

template <typename Isa, int R, int V>
void report(const char* layout, int ldb) {
    using T          = typename Isa::T;
    constexpr int W  = Isa::lanes;
    std::vector<T> a(R * kK, T(0.5)), c(R * V * W, T(0));
    std::vector<T> b(std::size_t(kK) * std::max(ldb, V * W), T(1e-3));
    constexpr long kCalls = 200'000;
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (long i = 0; i < kCalls; ++i)
            micro<Isa, R, V>(a.data(), b.data(), c.data(), ldb);
        const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
        best = std::min(best, dt.count());
    }
    const double flops = 2.0 * R * V * W * kK * kCalls;
    std::printf("%s,f%d,%d,%d,%d,%s,%.1f,%.2f\n", Isa::isa, Isa::bits, R, V, R * V, layout,
                flops / best / 1e9, best * g_ghz * 1e9 / (double(kK) * kCalls));
    if (c[0] == T(-1))  // keeps the results observable
        std::puts("");
}

/// Every row count, B packed and with B rows N = 256 and 1024 elements apart.
template <typename Isa, int V, int... Rs>
void sweep() {
    constexpr int W = Isa::lanes;
    (report<Isa, Rs, V>("packed", V * W), ...);
    (report<Isa, Rs, V>("ldb=256", 256), ...);
    (report<Isa, Rs, V>("ldb=1024", 1024), ...);
}

}  // namespace

int main() {
    g_ghz = hpc::experiments::core_ghz();
    std::fprintf(stderr, "core clock: %.3f GHz\n", g_ghz);
    std::printf("isa,type,rows,vectors_per_row,accumulators,b_layout,gflops,cycles_per_k\n");
#if HPC_HAS_NEON
    sweep<NeonF32, 4, 2, 3, 4, 5, 6>();  // library: 4 x 4 vectors = 16 accumulators
    sweep<NeonF64, 2, 4, 6, 8, 10>();    // library: 8 x 2 vectors = 16
#endif
#if HPC_HAS_AVX2
    sweep<Avx2F32, 2, 4, 5, 6>();        // library: 4 x 2 = 8 (16 registers in all)
    sweep<Avx2F64, 2, 4, 5, 6>();
#endif
#if HPC_HAS_AVX512
    sweep<Avx512F32, 2, 4, 6, 8, 12>();  // library: 8 x 2 = 16
    sweep<Avx512F64, 2, 4, 6, 8, 12>();
#endif
    std::fprintf(stderr, "core clock after: %.3f GHz\n", hpc::experiments::core_ghz());
}
