/**
 * @file amx_sme_share.cpp
 * @brief Does Accelerate's BLAS (Apple AMX) run on the matrix unit SME
 *        exposes? Each alone, then both at once.
 *
 *   VECLIB_MAXIMUM_THREADS=1 ./build/benchmarks/amx_sme_share > share.csv
 *
 * Two threads: one calls gemm_sme (FP32, the measured parameters) in a loop,
 * the other cblas_sgemm through gemm_amx, each on its own N = 2048
 * matrices, for 3 s. Run alone, each reaches its single-thread rate; run
 * together, if both use one unit they share its throughput (each falls to
 * about half, the sum stays near one alone), and if they use separate units
 * each keeps its rate. macOS places the threads; two busy threads usually
 * stay on one P-cluster (docs/kernels/sme.md, "Several SME units"), whose
 * SME unit both would then share. As the control for placement, "sme_pair"
 * runs two gemm_sme threads the same way (two threads on one cluster share
 * its unit; the second thread's rate is in the amx column). Five rounds,
 * each alone-SME, alone-AMX, together, sme_pair. Then, independent of
 * placement, both clusters saturated: 6 gemm_sme threads, 6 Accelerate
 * threads, and both groups at once (summed rates per group). stdout gets
 * CSV rows: round, mode, sme_gflops, amx_gflops.
 */

#include "gemm/amx.hpp"
#include "gemm/sme.hpp"
#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"
#include "../sme_params.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#if HPC_HAS_SME && HPC_HAS_AMX

namespace {

using hpc::Matrix;
constexpr std::size_t kN = 2048;
constexpr double kSeconds = 3.0;

void fill(Matrix<float>& m, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1, 1);
    for (std::size_t i = 0; i < m.size(); ++i)
        m.data()[i] = d(rng);
}

/// Calls of `kernel` until `stop`; returns GFLOP/s.
template <typename Kernel>
double run(Kernel kernel, const std::atomic<bool>& stop) {
    Matrix<float> A(kN, kN), B(kN, kN), C(kN, kN);
    fill(A, 1);
    fill(B, 2);
    kernel(A, B, C);  // warm-up
    long calls    = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (!stop.load(std::memory_order_relaxed)) {
        kernel(A, B, C);
        ++calls;
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return 2.0 * double(kN) * kN * kN * calls / sec / 1e9;
}

void sme(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) { hpc::bench::sme(A, B, C); }
void amx(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) { hpc::gemm::gemm_amx(A, B, C); }

/// Run the selected kernels concurrently for kSeconds; `second_sme` runs a
/// second gemm_sme thread in place of the Accelerate one (its rate is
/// reported in the amx column).
void round(int r, const char* mode, bool with_sme, bool with_amx, bool second_sme = false) {
    std::atomic<bool> stop{false};
    double g_sme = 0, g_amx = 0;
    std::thread ts, ta;
    if (with_sme)
        ts = std::thread([&] { g_sme = run(sme, stop); });
    if (with_amx)
        ta = std::thread([&] { g_amx = second_sme ? run(sme, stop) : run(amx, stop); });
    std::this_thread::sleep_for(std::chrono::duration<double>(kSeconds));
    stop = true;
    if (ts.joinable())
        ts.join();
    if (ta.joinable())
        ta.join();
    std::printf("%d,%s,%.0f,%.0f\n", r, mode, g_sme, g_amx);
    std::fflush(stdout);
}

/// k gemm_sme threads and m Accelerate threads at once for kSeconds;
/// prints the summed rate of each group.
void saturate(int r, const char* mode, int k, int m) {
    std::atomic<bool> stop{false};
    std::vector<double> g(k + m, 0.0);
    std::vector<std::thread> ts;
    for (int i = 0; i < k + m; ++i)
        ts.emplace_back([&, i] { g[i] = i < k ? run(sme, stop) : run(amx, stop); });
    std::this_thread::sleep_for(std::chrono::duration<double>(kSeconds));
    stop = true;
    for (auto& t : ts)
        t.join();
    double s = 0, a = 0;
    for (int i = 0; i < k + m; ++i)
        (i < k ? s : a) += g[i];
    std::printf("%d,%s,%.0f,%.0f\n", r, mode, s, a);
    std::fflush(stdout);
}

}  // namespace

int main() {
    std::printf("round,mode,sme_gflops,amx_gflops\n");
    for (int r = 1; r <= 5; ++r) {
        round(r, "sme_alone", true, false);
        round(r, "amx_alone", false, true);
        round(r, "together", true, true);
        round(r, "sme_pair", true, true, true);
    }
    // Both clusters saturated: totals add up only if AMX work runs on other
    // hardware than the SME units.
    for (int r = 1; r <= 3; ++r) {
        saturate(r, "sme6_alone", 6, 0);
        saturate(r, "amx6_alone", 0, 6);
        saturate(r, "sme6_amx6", 6, 6);
    }
}

#else
int main() {
    std::fprintf(stderr, "amx_sme_share needs SME2 and Accelerate\n");
    return 0;
}
#endif
