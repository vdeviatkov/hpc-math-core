/**
 * @file sme_b_layout.cpp
 * @brief Where packed B saves time: phase timing of the SME blocked driver.
 *
 *   ./build/benchmarks/sme_b_layout [N=4096] [reps=15] [batch=3] > timings.csv
 *
 * FP32, M = N = K. Six cases: B with ldb = N, N + 32 and N + 64, each read in
 * place (gemm_sme_blocked's path) and packed (gemm_sme's path). Each case is
 * timed two ways, in an order shuffled per repetition:
 *
 *   end to end  sme_detail::blocked_driver itself, `batch` calls averaged:
 *               scratch allocation, packing and compute, as the kernels run.
 *   by phase    a copy of the driver's loop nest with a clock around each
 *               phase (scratch, pack_a, pack_b, macro_kernel), one call.
 *               Diagnostic only: the clocks add overhead, and a phase's
 *               median need not add up to the total's.
 *
 * stdout gets one CSV row per timed run; stderr gets the validation result
 * and a median table. Every result, including the phase-timed loop's, is
 * checked against the packed driver before timing starts.
 */

#include "../sme_params.hpp"
#include "gemm/sme.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#if !HPC_HAS_SME
    #error "sme_b_layout needs SME2 (configure with -DHPC_ENABLE_SME=ON)"
#endif

namespace {

using hpc::Matrix;
using Clock = std::chrono::steady_clock;
namespace sme = hpc::gemm::sme_detail;
using P         = hpc::bench::SmeParams<float>;  // M4's measured shape and blocks

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

double cpu_ms() {
    timespec t{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return double(t.tv_sec) * 1e3 + double(t.tv_nsec) / 1e6;
}

struct Phases {
    double scratch = 0, pack_a = 0, pack_b = 0, compute = 0;
};

/// sme_detail::blocked_driver<float, PackB> with a clock around each phase.
template <bool PackB>
Phases timed_driver(const Matrix<float>& A, const float* B, std::size_t ldb, Matrix<float>& C) {
    const std::size_t M = A.rows(), K = A.cols(), N = C.cols();
    const std::size_t s = svcntsw(), mr = P::R * s, nr = P::V * s;
    auto round_up       = [](std::size_t x, std::size_t m) { return (x + m - 1) / m * m; };
    const std::size_t kc_max = std::min(P::Kc, K);

    Phases p;
    auto t = Clock::now();
    Matrix<float> a_pack(round_up(std::min(P::Mc, M), mr), kc_max);
    Matrix<float> b_pack(PackB ? kc_max : 0, PackB ? round_up(std::min(P::Nc, N), nr) : 0);
    p.scratch = ms_since(t);

    for (std::size_t jc = 0; jc < N; jc += P::Nc) {
        const std::size_t nc = std::min(P::Nc, N - jc);
        for (std::size_t pc = 0; pc < K; pc += P::Kc) {
            const std::size_t kc = std::min(P::Kc, K - pc);
            const float* b       = B + pc * ldb + jc;
            std::int64_t b_step = 1, b_ld = std::int64_t(ldb);
            if constexpr (PackB) {
                t = Clock::now();
                sme::pack_b(b, ldb, kc, nc, nr, b_pack.data());
                p.pack_b += ms_since(t);
                b      = b_pack.data();
                b_step = std::int64_t(kc);
                b_ld   = std::int64_t(nr);
            }
            for (std::size_t ic = 0; ic < M; ic += P::Mc) {
                const std::size_t mc = std::min(P::Mc, M - ic);
                t                    = Clock::now();
                sme::pack_a(A.data() + ic * K + pc, K, mc, kc, mr, a_pack.data());
                p.pack_a += ms_since(t);
                t = Clock::now();
                sme::macro_kernel<float, P::R, P::V>(a_pack.data(), b, b_step, b_ld, C.data() + ic * N + jc,
                                         std::int64_t(N), std::int64_t(mc), std::int64_t(nc),
                                         std::int64_t(kc), pc > 0);
                p.compute += ms_since(t);
            }
        }
    }
    return p;
}

struct Case {
    std::size_t buffer;  ///< index into the B copies
    std::size_t ldb;
    bool packed;
};

struct Sample {
    Case c;
    bool by_phase;
    double wall_ms, cpu_ms;
    Phases p;
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t N = argc > 1 ? std::stoul(argv[1]) : 4096;
    const int reps      = argc > 2 ? std::stoi(argv[2]) : 15;
    const int batch     = argc > 3 ? std::stoi(argv[3]) : 3;
    if (N == 0 || reps < 1 || batch < 1) {
        std::fprintf(stderr, "usage: %s [N] [reps] [batch]\n", argv[0]);
        return 2;
    }

    // The same logical A and B for every layout; B is stored three times,
    // with 0, 32 and 64 padding columns per row.
    const std::array<std::size_t, 3> pads{0, 32, 64};
    Matrix<float> A(N, N), C(N, N), ref(N, N);
    std::vector<Matrix<float>> B;
    for (const std::size_t pad : pads)
        B.emplace_back(N, N + pad);
    std::mt19937 rng(20261012);
    std::uniform_real_distribution<float> dist(-1, 1);
    for (std::size_t i = 0; i < N * N; ++i)
        A.data()[i] = dist(rng);
    for (std::size_t k = 0; k < N; ++k)
        for (std::size_t j = 0; j < N; ++j) {
            const float v = dist(rng);
            for (auto& b : B)
                b(k, j) = v;
        }

    std::vector<Case> cases;
    for (std::size_t i = 0; i < pads.size(); ++i)
        for (const bool packed : {false, true})
            cases.push_back({i, N + pads[i], packed});
    auto b_of = [&](const Case& c) { return B[c.buffer].data(); };
    auto run  = [&](const Case& c, bool by_phase) {
        if (by_phase)
            return c.packed ? timed_driver<true>(A, b_of(c), c.ldb, C)
                            : timed_driver<false>(A, b_of(c), c.ldb, C);
        if (c.packed)
            sme::blocked_driver<float, true, P::R, P::V, P::Mc, P::Kc, P::Nc>(A, b_of(c), c.ldb, C);
        else
            sme::blocked_driver<float, false, P::R, P::V, P::Mc, P::Kc, P::Nc>(A, b_of(c), c.ldb, C);
        return Phases{};
    };

    sme::blocked_driver<float, true, P::R, P::V, P::Mc, P::Kc, P::Nc>(A, B[0].data(), N, ref);
    for (const Case& c : cases)
        for (const bool by_phase : {false, true}) {
            run(c, by_phase);
            for (std::size_t i = 0; i < C.size(); ++i)
                if (!(std::abs(C.data()[i] - ref.data()[i]) <= 1e-4f * (1 + std::abs(ref.data()[i])))) {
                    std::fprintf(stderr, "ldb=%zu packed=%d by_phase=%d: wrong result\n", c.ldb,
                                 c.packed, by_phase);
                    return 1;
                }
        }
    std::fprintf(stderr, "all %zu layouts, both timing paths: results match\n", cases.size());
    for (std::size_t i = 0; i < B.size(); ++i)
        std::fprintf(stderr, "B buffer %zu (ldb=%zu) starts at offset %zu mod 128\n", i,
                     N + pads[i], std::size_t(reinterpret_cast<std::uintptr_t>(B[i].data()) % 128));

    std::vector<Sample> samples;
    std::vector<int> order(2 * cases.size());
    std::iota(order.begin(), order.end(), 0);
    for (int rep = 0; rep < reps; ++rep) {
        std::shuffle(order.begin(), order.end(), rng);
        for (const int id : order) {
            const Case& c       = cases[std::size_t(id / 2)];
            const bool by_phase = id % 2;
            const int calls     = by_phase ? 1 : batch;
            Phases p;
            const auto t0   = Clock::now();
            const double c0 = cpu_ms();
            for (int i = 0; i < calls; ++i)
                p = run(c, by_phase);
            samples.push_back({c, by_phase, ms_since(t0) / calls, (cpu_ms() - c0) / calls, p});
        }
    }

    std::printf("ldb,packed,by_phase,wall_ms,cpu_ms,scratch_ms,pack_a_ms,pack_b_ms,compute_ms\n");
    for (const Sample& s : samples)
        std::printf("%zu,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", s.c.ldb, s.c.packed, s.by_phase,
                    s.wall_ms, s.cpu_ms, s.p.scratch, s.p.pack_a, s.p.pack_b, s.p.compute);

    std::fprintf(stderr, "\nmedians over %d repetitions, ms (GFLOP/s from the end-to-end time)\n",
                 reps);
    std::fprintf(stderr, "%6s %7s %10s %8s | %8s %8s %8s %8s\n", "ldb", "B", "end2end", "GFLOP/s",
                 "scratch", "pack_a", "pack_b", "compute");
    for (const Case& c : cases) {
        std::vector<double> e2e, sc, pa, pb, cp;
        for (const Sample& s : samples) {
            if (s.c.ldb != c.ldb || s.c.packed != c.packed)
                continue;
            if (!s.by_phase) {
                e2e.push_back(s.wall_ms);
            } else {
                sc.push_back(s.p.scratch);
                pa.push_back(s.p.pack_a);
                pb.push_back(s.p.pack_b);
                cp.push_back(s.p.compute);
            }
        }
        const double t = median(e2e);
        std::fprintf(stderr, "%6zu %7s %10.2f %8.0f | %8.2f %8.2f %8.2f %8.2f\n", c.ldb,
                     c.packed ? "packed" : "inplace", t, 2.0 * double(N) * N * N / (t * 1e6),
                     median(sc), median(pa), median(pb), median(cp));
    }
}
