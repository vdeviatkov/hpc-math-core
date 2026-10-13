/**
 * @file sme_b_stride_probe.cpp
 * @brief Compute-only probes of how B's layout affects the SME macro-kernel.
 *
 *   ./build/benchmarks/sme_b_stride_probe height > height.csv
 *   ./build/benchmarks/sme_b_stride_probe width  > width.csv
 *
 * FP32. A is packed once up front and B is laid out once, so only
 * sme_detail::macro_kernel is timed (over all Mc blocks of an M = 4096
 * problem, one Kc block). Configurations are shuffled within each of 9
 * repetitions; stdout gets one CSV row per sample, stderr a median table.
 *
 *   height  N = 1024, strip height Kc = 8 … 1024 rows. B read in place with
 *           ldb = 4096 (rows 16 KiB apart) and ldb = 4128 (16,512 B), and
 *           packed. Where the two in-place strides part is how many rows
 *           16 KiB apart the conflicting cache holds.
 *   width   Kc = 1024, N = 1024 … 4224. B in place with ldb = N + 32 (one
 *           cache line of padding) against packed: what remains of the
 *           in-place cost once the stride is not a power of two.
 */

#include "gemm/sme.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#if !HPC_HAS_SME
    #error "sme_b_stride_probe needs SME2 (configure with -DHPC_ENABLE_SME=ON)"
#endif

namespace {

using hpc::Matrix;
namespace sme = hpc::gemm::sme_detail;

constexpr std::size_t kM = 4096, kMc = 128, kNr = 32;

/// One configuration: N columns, Kc rows, B packed (ldb = 0) or in place with ldb.
struct Config {
    std::size_t N, Kc, ldb;
};

/// Prepared operands for one configuration.
struct Problem {
    Matrix<float> a_pack, B, C;
    std::int64_t b_step, b_ld;

    explicit Problem(const Config& c)
        : a_pack(kM, c.Kc), B(1, c.ldb ? c.Kc * c.ldb : c.Kc * c.N), C(kM, c.N) {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1, 1);
        Matrix<float> A(kM, c.Kc);
        for (std::size_t i = 0; i < A.size(); ++i)
            A.data()[i] = dist(rng);
        sme::pack_a(A.data(), c.Kc, kM, c.Kc, 2 * svcntsw(), a_pack.data());
        for (std::size_t i = 0; i < B.size(); ++i)
            B.data()[i] = dist(rng);
        // Packed: strip s (32 columns) is Kc contiguous rows of 128 B.
        // In place: row k of strip s is at s·32 + k·ldb.
        b_step = c.ldb ? 1 : std::int64_t(c.Kc);
        b_ld   = c.ldb ? std::int64_t(c.ldb) : std::int64_t(kNr);
    }

    void run(const Config& c) {
        for (std::size_t ic = 0; ic < kM; ic += kMc)
            sme::macro_kernel<float>(a_pack.data() + ic * c.Kc, B.data(), b_step, b_ld,
                                     C.data() + ic * c.N, std::int64_t(c.N), std::int64_t(kMc),
                                     std::int64_t(c.N), std::int64_t(c.Kc), false);
    }
};

double gflops(const Config& c, double seconds) {
    return 2.0 * double(kM) * double(c.N) * double(c.Kc) / seconds / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    std::vector<Config> configs;
    if (mode == "height") {
        for (std::size_t kc : {8, 16, 32, 64, 128, 256, 384, 448, 512, 576, 640, 768, 896, 1024})
            for (std::size_t ldb : {std::size_t(4096), std::size_t(4128), std::size_t(0)})
                configs.push_back({1024, kc, ldb});
    } else if (mode == "width") {
        for (std::size_t n : {1024, 2048, 3072, 3584, 3968, 4000, 4096, 4128, 4224})
            for (std::size_t ldb : {n + 32, std::size_t(0)})
                configs.push_back({n, 1024, ldb});
    } else {
        std::fprintf(stderr, "usage: %s height|width\n", argv[0]);
        return 2;
    }

    constexpr int kReps = 9;
    std::map<std::tuple<std::size_t, std::size_t, std::size_t>, std::vector<double>> results;
    std::printf("mode,N,Kc,ldb,gflops\n");
    std::mt19937 rng(20261012);
    std::vector<std::size_t> order(configs.size());
    for (std::size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    for (int rep = 0; rep < kReps; ++rep) {
        std::shuffle(order.begin(), order.end(), rng);
        for (const std::size_t i : order) {
            const Config& c = configs[i];
            Problem p(c);
            p.run(c);  // warm-up
            const auto t0 = std::chrono::steady_clock::now();
            p.run(c);
            const double s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double g = gflops(c, s);
            results[{c.N, c.Kc, c.ldb}].push_back(g);
            std::printf("%s,%zu,%zu,%zu,%.1f\n", mode.c_str(), c.N, c.Kc, c.ldb, g);
        }
    }

    std::fprintf(stderr, "median GFLOP/s over %d repetitions (ldb 0 = packed)\n", kReps);
    for (const auto& [key, v] : results) {
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        const auto [n, kc, ldb] = key;
        std::fprintf(stderr, "N=%5zu Kc=%5zu ldb=%5zu : %7.0f\n", n, kc, ldb, s[s.size() / 2]);
    }
}
