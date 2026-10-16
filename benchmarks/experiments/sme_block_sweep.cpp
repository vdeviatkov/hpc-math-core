/**
 * @file sme_block_sweep.cpp
 * @brief gemm_sme with run-time block sizes: which Mc, Kc, Nc are fastest.
 *
 *   ./build/benchmarks/sme_block_sweep > sweep.csv
 *
 * The loop nest of sme_detail::blocked_driver<T, true> (pack B per (jc, pc),
 * pack A per ic, macro-kernel), with Mc, Kc and Nc as arguments instead of
 * the compile-time kSmeMc / kSmeKc / kSmeNc. Every combination of Mc in
 * {256, 512, 1024, 2048}, Kc in {1024, 2048, 4096} and Nc in {1024, 2048,
 * 4096}, plus (128, 1024, 4096), the kernel's first sizing, runs on square
 * matrices N = 512 … 4096, FP32 and FP64. Configurations are timed in turn
 * within each repetition (5 repetitions, 3 at N = 4096) and the median is
 * reported. stdout gets CSV rows.
 */

#include "gemm/sme.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#if !HPC_HAS_SME
    #error "sme_block_sweep needs SME2 (configure with -DHPC_ENABLE_SME=ON)"
#endif

namespace {

using hpc::Matrix;
namespace sme = hpc::gemm::sme_detail;

struct Blocks {
    std::size_t mc, kc, nc;
};

/// C = A · B with packed A and B and the given block sizes.
template <typename T>
void gemm(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, Blocks bl) {
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    const std::size_t s  = sme::kF32<T> ? svcntsw() : svcntsd();
    const std::size_t mr = 2 * s, nr = sme::kCols<T> * s;
    auto round_up        = [](std::size_t x, std::size_t m) { return (x + m - 1) / m * m; };
    Matrix<T> a_pack(round_up(std::min(bl.mc, M), mr), std::min(bl.kc, K));
    Matrix<T> b_pack(std::min(bl.kc, K), round_up(std::min(bl.nc, N), nr));
    for (std::size_t jc = 0; jc < N; jc += bl.nc) {
        const std::size_t nc = std::min(bl.nc, N - jc);
        for (std::size_t pc = 0; pc < K; pc += bl.kc) {
            const std::size_t kc = std::min(bl.kc, K - pc);
            sme::pack_b(B.data() + pc * N + jc, N, kc, nc, nr, b_pack.data());
            for (std::size_t ic = 0; ic < M; ic += bl.mc) {
                const std::size_t mc = std::min(bl.mc, M - ic);
                sme::pack_a(A.data() + ic * K + pc, K, mc, kc, mr, a_pack.data());
                sme::macro_kernel<T>(a_pack.data(), b_pack.data(), std::int64_t(kc),
                                     std::int64_t(nr), C.data() + ic * N + jc, std::int64_t(N),
                                     std::int64_t(mc), std::int64_t(nc), std::int64_t(kc), pc > 0);
            }
        }
    }
}

template <typename T>
void sweep(const char* type) {
    std::vector<Blocks> configs{{128, 1024, 4096}};
    for (std::size_t mc : {256, 512, 1024, 2048})
        for (std::size_t kc : {1024, 2048, 4096})
            for (std::size_t nc : {1024, 2048, 4096})
                configs.push_back({mc, kc, nc});
    std::mt19937 rng(1);
    std::uniform_real_distribution<T> dist(-1, 1);
    for (std::size_t n : {512, 1024, 2048, 4096}) {
        Matrix<T> A(n, n), B(n, n), C(n, n);
        for (std::size_t i = 0; i < A.size(); ++i) {
            A.data()[i] = dist(rng);
            B.data()[i] = dist(rng);
        }
        std::vector<std::vector<double>> seconds(configs.size());
        const int reps = n >= 4096 ? 3 : 5;
        for (int rep = 0; rep < reps; ++rep)
            for (std::size_t c = 0; c < configs.size(); ++c) {
                if (rep == 0)
                    gemm(A, B, C, configs[c]);  // warm-up
                const auto t0 = std::chrono::steady_clock::now();
                gemm(A, B, C, configs[c]);
                seconds[c].push_back(
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
        for (std::size_t c = 0; c < configs.size(); ++c) {
            std::vector<double> v = seconds[c];
            std::sort(v.begin(), v.end());
            std::printf("%s,%zu,%zu,%zu,%zu,%.1f\n", type, n, configs[c].mc, configs[c].kc,
                        configs[c].nc, 2.0 * double(n) * n * n / v[v.size() / 2] / 1e9);
        }
    }
}

}  // namespace

int main() {
    std::printf("type,N,Mc,Kc,Nc,gflops\n");
    sweep<float>("f32");
    sweep<double>("f64");
}
