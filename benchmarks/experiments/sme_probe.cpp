/**
 * @file sme_probe.cpp
 * @brief Two SME follow-ups on Apple M4: where gemm_sme_tiles (no cache
 *        blocking) falls off, and what limits FP64 with B read in place.
 *
 *   ./build/benchmarks/sme_probe tiles > tiles.csv
 *   ./build/benchmarks/sme_probe f64   > f64.csv
 *
 * tiles  gemm_sme_tiles<float> (all of A re-read for every B strip) at N =
 *        1024 … 4096, powers of two and not. If its slowdown at large N is
 *        a capacity effect, it follows the size of A against the 16 MiB L2
 *        (A is 16 MiB FP32 at N = 2048) whatever the stride; a stride effect
 *        would single out the powers of two.
 * f64    The blocked driver with B read in place, FP64, N = 4096 (Mc = 512,
 *        Nc = 4096, the measured shape), B rows N, N + 16, N + 32 or N + 64
 *        doubles apart (+128, +256, +512 B), and strips of Kc = 256, 512,
 *        1024, 2048 rows. A strip of
 *        B is Kc rows of 256 B (1 × 4 FP64 tiles: 32 columns); if the loss
 *        that padding does not remove is capacity, it shrinks with Kc.
 *
 * Median of 5 runs after a warm-up; stdout gets CSV rows.
 */

#include "../sme_params.hpp"
#include "gemm/sme.hpp"
#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#if HPC_HAS_SME

namespace {

using hpc::Matrix;
namespace sme = hpc::gemm::sme_detail;

template <typename T>
void fill(T* p, std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(-1, 1);
    for (std::size_t i = 0; i < n; ++i)
        p[i] = T(d(rng));
}

template <typename F>
double median_gflops(F f, double flops) {
    f();
    std::vector<double> g;
    for (int r = 0; r < 5; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        g.push_back(flops / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9);
    }
    std::sort(g.begin(), g.end());
    return g[2];
}

void tiles() {
    std::printf("N,A_MiB,gflops\n");
    for (const std::size_t n : {1024, 1280, 1536, 1792, 1920, 2000, 2048, 2100, 2304, 2560, 3072, 4000, 4096}) {
        Matrix<float> A(n, n), B(n, n), C(n, n);
        fill(A.data(), A.size(), 1);
        fill(B.data(), B.size(), 2);
        const double g = median_gflops([&] { hpc::bench::sme_tiles(A, B, C); }, 2.0 * double(n) * n * n);
        std::printf("%zu,%.1f,%.1f\n", n, double(n) * n * 4 / (1 << 20), g);
        std::fflush(stdout);
    }
}

template <std::size_t KC>
double f64_in_place(const Matrix<double>& A, const std::vector<double>& B, std::size_t ldb, Matrix<double>& C) {
    using P             = hpc::bench::SmeParams<double>;
    const std::size_t n = A.rows();
    return median_gflops(
        [&] { sme::blocked_driver<double, false, P::R, P::V, P::Mc, KC, P::Nc>(A, B.data(), ldb, C, 1); },
        2.0 * double(n) * n * n);
}

void f64() {
    constexpr std::size_t n = 4096;
    std::printf("ldb,stride_bytes,kc,gflops\n");
    Matrix<double> A(n, n), C(n, n);
    fill(A.data(), A.size(), 1);
    for (const std::size_t ldb : {n, n + 16, n + 32, n + 64}) {
        std::vector<double> B(n * ldb);
        fill(B.data(), B.size(), 2);
        const double g[] = {f64_in_place<256>(A, B, ldb, C), f64_in_place<512>(A, B, ldb, C),
                            f64_in_place<1024>(A, B, ldb, C), f64_in_place<2048>(A, B, ldb, C)};
        const std::size_t kc[] = {256, 512, 1024, 2048};
        for (int i = 0; i < 4; ++i)
            std::printf("%zu,%zu,%zu,%.1f\n", ldb, ldb * 8, kc[i], g[i]);
        std::fflush(stdout);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "tiles")
        tiles();
    else if (mode == "f64")
        f64();
    else {
        std::fprintf(stderr, "usage: %s tiles|f64\n", argv[0]);
        return 2;
    }
}

#else
int main() {
    std::fprintf(stderr, "sme_probe needs SME2\n");
    return 0;
}
#endif
