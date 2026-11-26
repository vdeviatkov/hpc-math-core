/**
 * @file sme_microkernel.cpp
 * @brief The SME macro-kernel alone on cache-resident packed operands: how
 *        close each ZA shape gets to the FMOPA peak without the traffic of
 *        a full GEMM.
 *
 *   ./build/benchmarks/sme_microkernel > microkernel.csv
 *
 * Packs A (MC × KC) and B (KC × NC) once, then calls sme::macro_kernel on
 * them repeatedly, accumulating into the same C, so packing and the
 * streaming of A and B from memory are out of the measurement: the
 * operands stay in the caches (A 1 MiB, B 2 MiB FP32 at KC = 2048; the M4
 * P-cluster L2 is 16 MiB). Shapes: FP32 1×4, 2×2, 4×1; FP64 1×4, 2×4, 2×2,
 * 4×2. KC ∈ {256, 512, 2048}. Best of 7 timing runs of ≥ 0.2 s each;
 * stdout gets CSV rows: type, shape, kc, gflops, share of the measured
 * FMOPA peak (1,963 GFLOP/s FP32, 500 FP64; docs/experiments/sme2-machine).
 */

#include "gemm/sme.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

namespace sme = hpc::gemm::sme_detail;
using hpc::Matrix;

namespace {

template <typename T>
void fill(Matrix<T>& m, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> d(-1, 1);
    for (std::size_t i = 0; i < m.size(); ++i)
        m.data()[i] = d(rng);
}

template <typename T, int R, int V>
void run(const char* type, double peak) {
    constexpr std::size_t MC = sizeof(T) == 4 ? 128 : 64, NC = sizeof(T) == 4 ? 256 : 128;
    const std::size_t s = std::size_t(sme::lanes<T>()), mr = R * s, nr = V * s;
    for (const std::size_t kc : {256, 512, 2048}) {
        Matrix<T> A(MC, kc), B(kc, NC), C(MC, NC);
        fill(A, 1);
        fill(B, 2);
        Matrix<T> a_pack(MC, kc), b_pack(kc, NC);
        sme::pack_a(A.data(), kc, MC, kc, mr, a_pack.data());
        sme::pack_b(B.data(), NC, kc, NC, nr, b_pack.data());
        auto call = [&] {
            sme::macro_kernel<T, R, V>(a_pack.data(), b_pack.data(), std::int64_t(kc), std::int64_t(nr), C.data(),
                                       std::int64_t(NC), std::int64_t(MC), std::int64_t(NC), std::int64_t(kc), true);
        };
        const double flops_per_call = 2.0 * MC * NC * double(kc);
        int calls                   = 1;
        for (;;) {  // calls per timing run: at least 0.2 s
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < calls; ++i)
                call();
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (sec >= 0.2)
                break;
            calls *= 2;
        }
        double best = 0;
        for (int run = 0; run < 7; ++run) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < calls; ++i)
                call();
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            best             = std::max(best, flops_per_call * calls / sec / 1e9);
        }
        std::printf("%s,%dx%d,%zu,%.1f,%.3f\n", type, R, V, kc, best, best / peak);
        std::fflush(stdout);
    }
}

}  // namespace

int main() {
    std::printf("type,shape,kc,gflops,share_of_peak\n");
    run<float, 1, 4>("f32", 1963);
    run<float, 2, 2>("f32", 1963);
    run<float, 4, 1>("f32", 1963);
    run<double, 1, 4>("f64", 500);
    run<double, 2, 4>("f64", 500);
    run<double, 2, 2>("f64", 500);
    run<double, 4, 2>("f64", 500);
}
