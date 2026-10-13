/**
 * @file sme_fmopa.cpp
 * @brief Peak FMOPA throughput: dependency chains and threads.
 *
 *   ./build/benchmarks/sme_fmopa > fmopa.csv
 *
 * A streaming loop issues FMOPAs on constant vectors into T ZA tiles in
 * turn, so each tile is a chain of dependent accumulations and the T chains
 * are independent: one tile measures FMOPA latency, all tiles throughput.
 * No memory is touched inside the loop. FP32 uses up to the 4 za32 tiles
 * (16 × 16, 512 flops per FMOPA), FP64 up to the 8 za64 tiles (8 × 8, 128
 * flops). The same loop then runs on 1 … 12 threads at once to show how
 * many SME units the threads share. Best of 5 runs; stdout gets CSV rows,
 * with times also in cycles of the core clock measured at the start
 * (core_clock.hpp).
 */

#include "core_clock.hpp"
#include "hpc/isa.hpp"

#include <algorithm>
#include <arm_sme.h>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#if !HPC_HAS_SME
    #error "sme_fmopa needs SME2 (configure with -DHPC_ENABLE_SME=ON)"
#endif

namespace {

constexpr long kIters = 4'000'000;  // FMOPAs per tile

template <int... Tiles>
__arm_locally_streaming __arm_new("za") void fmopa_f32(long iters, float* out) {
    const svbool_t pg   = svptrue_b32();
    const svfloat32_t a = svdup_f32(1.0f), b = svdup_f32(1e-9f);
    for (long i = 0; i < iters; ++i)
        (svmopa_za32_f32_m(Tiles, pg, pg, a, b), ...);
    svst1_hor_za32(0, 0, pg, out);  // keep the result observable
}

template <int... Tiles>
__arm_locally_streaming __arm_new("za") void fmopa_f64(long iters, double* out) {
    const svbool_t pg   = svptrue_b64();
    const svfloat64_t a = svdup_f64(1.0), b = svdup_f64(1e-9);
    for (long i = 0; i < iters; ++i)
        (svmopa_za64_f64_m(Tiles, pg, pg, a, b), ...);
    svst1_hor_za64(0, 0, pg, out);
}

/// Seconds for `threads` threads to each run fn once (best of 5).
template <typename Fn>
double time_threads(int threads, Fn fn) {
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        std::vector<std::thread> pool;
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < threads; ++t)
            pool.emplace_back(fn);
        for (auto& th : pool)
            th.join();
        best = std::min(best,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

double g_ghz = 0;

void report(const char* type, int tiles, int threads, int flops_per_fmopa, double seconds) {
    const double fmopas = double(kIters) * tiles * threads;
    const double ns     = seconds * 1e9 / (double(kIters) * tiles);
    std::printf("%s,%d,%d,%.3f,%.2f,%.1f\n", type, tiles, threads, ns, ns * g_ghz,
                fmopas * flops_per_fmopa / seconds / 1e9);
}

}  // namespace

int main() {
    g_ghz = hpc::experiments::core_ghz();
    std::fprintf(stderr, "core clock: %.3f GHz\n", g_ghz);
    std::printf("type,tiles,threads,ns_per_fmopa_per_thread,cycles_per_fmopa_per_thread,gflops_total\n");
    float f[64];
    double d[64];
    report("f32", 1, 1, 512, time_threads(1, [&] { fmopa_f32<0>(kIters, f); }));
    report("f32", 2, 1, 512, time_threads(1, [&] { fmopa_f32<0, 1>(kIters, f); }));
    report("f32", 4, 1, 512, time_threads(1, [&] { fmopa_f32<0, 1, 2, 3>(kIters, f); }));
    report("f64", 1, 1, 128, time_threads(1, [&] { fmopa_f64<0>(kIters, d); }));
    report("f64", 2, 1, 128, time_threads(1, [&] { fmopa_f64<0, 1>(kIters, d); }));
    report("f64", 4, 1, 128, time_threads(1, [&] { fmopa_f64<0, 1, 2, 3>(kIters, d); }));
    report("f64", 8, 1, 128, time_threads(1, [&] { fmopa_f64<0, 1, 2, 3, 4, 5, 6, 7>(kIters, d); }));
    for (const int threads : {2, 3, 4, 6, 8, 12})
        report("f32", 4, threads, 512, time_threads(threads, [] {
                   float out[64];
                   fmopa_f32<0, 1, 2, 3>(kIters, out);
               }));
}
