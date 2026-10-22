/**
 * @file avx512_b_stride.cpp
 * @brief Part 3's B-stride experiment on an x86 core (AMD Zen 5, AVX-512).
 *
 *   ./build/benchmarks/avx512_b_stride chase  > chase.csv
 *   ./build/benchmarks/avx512_b_stride ldb    > ldb.csv
 *   ./build/benchmarks/avx512_b_stride size   > size.csv
 *   ./build/benchmarks/avx512_b_stride height > height.csv
 *
 * chase   A dependent pointer chase over `lines` cache lines placed `stride`
 *         bytes apart, in random order, on 2 MiB transparent huge pages (so
 *         a strided walk does not also miss the TLB on every load). The
 *         number of lines at which the latency steps up is the capacity of
 *         the cache sets the stride leaves to them, per level.
 *
 * ldb     The FP32 GEMM of gemm_avx512_blocked (same 64 x 256 x 512 blocking,
 *         same avx512_micro<8, 2>) with B stored at row stride ldb >= N. Only
 *         the stride changes; the multiplication does not. Each ldb is run
 *         with B read in place, as the library kernel does, and with each
 *         256 x 512 block of B packed into contiguous 32-column strips first
 *         (packing time included).
 *
 * size    The same two variants at N around 1024, 2048 and 4096, ldb = N:
 *         what a caller of the library kernel sees.
 *
 * height  Only the strip height Kc changes (the k extent of the B block the
 *         micro-kernel reuses), for B rows 4, 8 and 16 KiB apart and each
 *         padded by 64 B; M = 512, K = 2048, N = 1024 (B is K x ldb, its
 *         first N columns used), plus 4,352 B and 6 KiB rows. Whether the
 *         loss starts at a height that halves as the stride doubles separates
 *         L2 sets (whose capacity at a stride does) from L1 sets (whose
 *         capacity is the same 12 lines at every multiple of 4 KiB).
 *
 * Every GEMM variant is checked against gemm_avx512_blocked before timing.
 * Times are medians of 5 repetitions, variants interleaved; GFLOP/s =
 * 2 M N K / time. The core clock is measured first (core_clock.hpp) for the
 * chase's cycle column. Pin the process to one core (taskset -c 2).
 */

#include "core_clock.hpp"
#include "gemm/avx512.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <sys/mman.h>
#include <vector>

#if !HPC_HAS_AVX512 || !defined(__linux__)
int main() {
    std::fprintf(stderr, "avx512_b_stride needs AVX-512 and Linux (transparent huge pages)\n");
    return 1;
}
#else

namespace {

using hpc::gemm::avx512_micro;

double g_ghz = 0;

template <typename F>
double median_seconds(F run, int reps = 5) {
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        run();
        t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// ---------------------------------------------------------------------------
// chase
// ---------------------------------------------------------------------------
void chase() {
    std::printf("stride_bytes,lines,ns_per_load,cycles_per_load\n");
    const std::size_t strides[] = {64, 4096, 4160, 8192, 8256, 16384, 16448, 65536, 65600};
    const std::size_t counts[]  = {8,   12,  16,  24,   32,   48,   64,   96,    128,   192,  256,
                                   320, 384, 512, 768,  1024, 2048, 4096, 8192, 16384, 32768};
    constexpr std::size_t kHuge = std::size_t(2) << 20;
    for (const std::size_t stride : strides)
        for (const std::size_t n : counts) {
            const std::size_t bytes = ((n * stride + kHuge) / kHuge + 1) * kHuge;
            if (bytes > (std::size_t(3) << 30))
                continue;
            void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mem == MAP_FAILED) {
                std::perror("mmap");
                std::exit(1);
            }
            madvise(mem, bytes, MADV_HUGEPAGE);
            // Start at a 2 MiB boundary so every line sits where its offset says.
            char* base = reinterpret_cast<char*>((reinterpret_cast<std::uintptr_t>(mem) + kHuge - 1) & ~(kHuge - 1));
            std::vector<std::size_t> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin() + 1, order.end(), std::mt19937(42));
            for (std::size_t i = 0; i < n; ++i)
                *reinterpret_cast<char**>(base + order[i] * stride) = base + order[(i + 1) % n] * stride;
            constexpr long kLoads = 20'000'000;
            char* p = base;
            for (long i = 0; i < kLoads / 10; ++i)  // warm up
                p = *reinterpret_cast<char**>(p);
            double best = 1e9;
            for (int rep = 0; rep < 3; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                for (long i = 0; i < kLoads; ++i)
                    p = *reinterpret_cast<char**>(p);
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            if (p == nullptr)  // keep the chain observable
                std::puts("");
            const double ns = best * 1e9 / kLoads;
            std::printf("%zu,%zu,%.2f,%.1f\n", stride, n, ns, ns * g_ghz);
            std::fflush(stdout);
            munmap(mem, bytes);
        }
}

// ---------------------------------------------------------------------------
// GEMM with B at row stride ldb, in place or packed per block
// ---------------------------------------------------------------------------
// The library's cache block when this experiment was run (2026-10-17); the
// default is 512 × 256 × 512 since 2026-10-22 (docs/kernels/blocks.md).
constexpr std::size_t kTileM = 64;
constexpr std::size_t kTileN = 512;
constexpr std::size_t kJStep = 32;                       // f32: 2 ZMM per micro-kernel call

// C = A (M x K, lda K) * B (K x N, row stride ldb). M a multiple of 8, N of 32.
void gemm_strided(const float* A, const float* B, float* C, std::size_t M, std::size_t K, std::size_t N,
                  std::size_t ldb, std::size_t kc, bool pack, std::vector<float>& buf) {
    std::memset(C, 0, M * N * sizeof(float));
    buf.resize(kc * kTileN);
    for (std::size_t i_blk = 0; i_blk < M; i_blk += kTileM) {
        const std::size_t i_end = std::min(i_blk + kTileM, M);
        for (std::size_t k_blk = 0; k_blk < K; k_blk += kc) {
            const std::size_t k_len = std::min(kc, K - k_blk);
            for (std::size_t j_blk = 0; j_blk < N; j_blk += kTileN) {
                const std::size_t j_len = std::min(kTileN, N - j_blk);
                const float* b_blk = B + k_blk * ldb + j_blk;
                if (pack)  // strip s: k_len rows of 32 floats, contiguous
                    for (std::size_t s = 0; s < j_len / kJStep; ++s)
                        for (std::size_t k = 0; k < k_len; ++k)
                            std::memcpy(&buf[(s * k_len + k) * kJStep], b_blk + k * ldb + s * kJStep,
                                        kJStep * sizeof(float));
                for (std::size_t i = i_blk; i < i_end; i += 8)
                    for (std::size_t j = 0; j < j_len; j += kJStep) {
                        const float* b = pack ? &buf[(j / kJStep) * k_len * kJStep] : b_blk + j;
                        avx512_micro<8, 2>(A + i * K + k_blk, b, C + i * N + j_blk + j, K, pack ? kJStep : ldb, N,
                                         k_len);
                    }
            }
        }
    }
}

struct Problem {
    std::size_t M, K, N, ldb;
    std::vector<float> A, B, C, buf;
    Problem(std::size_t m, std::size_t k, std::size_t n, std::size_t ld) : M(m), K(k), N(n), ldb(ld) {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        A.resize(M * K);
        B.resize(K * ldb);
        C.resize(M * N);
        for (auto& x : A) x = d(rng);
        for (auto& x : B) x = d(rng);
    }
    void run(std::size_t kc, bool pack) { gemm_strided(A.data(), B.data(), C.data(), M, K, N, ldb, kc, pack, buf); }

    // Compare with the library kernel on the same values (B copied to ldb = N).
    void validate(std::size_t kc) {
        hpc::Matrix<float> a(M, K), b(K, N), c(M, N);
        std::copy(A.begin(), A.end(), a.data());
        for (std::size_t k = 0; k < K; ++k)
            std::copy_n(&B[k * ldb], N, b.data() + k * N);
        hpc::gemm::gemm_avx512_blocked(a, b, c);
        for (const bool pack : {false, true}) {
            run(kc, pack);
            for (std::size_t i = 0; i < M * N; ++i)
                if (std::fabs(C[i] - c.data()[i]) > 1e-3f * (1.f + std::fabs(c.data()[i]))) {
                    std::fprintf(stderr, "mismatch M=%zu N=%zu ldb=%zu pack=%d at %zu\n", M, N, ldb, pack, i);
                    std::exit(1);
                }
        }
    }
};

double gflops(const Problem& p, double s) { return 2.0 * double(p.M) * double(p.N) * double(p.K) / s / 1e9; }

// One row per (M, K, N, ldb, kc): in place and packed interleaved, 5 rounds.
void measure(std::size_t M, std::size_t K, std::size_t N, std::size_t ldb, std::size_t kc) {
    Problem p(M, K, N, ldb);
    p.validate(kc);
    p.run(kc, false);
    p.run(kc, true);  // warm up both
    std::vector<double> t_in, t_pk;
    for (int r = 0; r < 5; ++r) {
        t_in.push_back(median_seconds([&] { p.run(kc, false); }, 1));
        t_pk.push_back(median_seconds([&] { p.run(kc, true); }, 1));
    }
    std::sort(t_in.begin(), t_in.end());
    std::sort(t_pk.begin(), t_pk.end());
    const double in = gflops(p, t_in[2]), pk = gflops(p, t_pk[2]);
    std::printf("%zu,%zu,%zu,%zu,%zu,%zu,%.1f,%.1f,%.2f\n", M, K, N, ldb, ldb * sizeof(float), kc, in, pk, in / pk);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    g_ghz = hpc::experiments::core_ghz();
    std::fprintf(stderr, "core clock: %.3f GHz\n", g_ghz);
    if (mode == "chase") {
        chase();
        return 0;
    }
    const char* header = "M,K,N,ldb,stride_bytes,kc,in_place_gflops,packed_gflops,in_place_over_packed\n";
    if (mode == "ldb") {
        std::printf("%s", header);
        for (const std::size_t N : {1024, 2048, 4096})
            for (const std::size_t pad : {0, 16, 32, 64, 128, 512})  // floats: 0, 64 B, 128 B, 256 B, 512 B, 2 KiB
                measure(N, N, N, N + pad, 256);
    } else if (mode == "size") {
        std::printf("%s", header);
        for (const std::size_t N : {512, 768, 992, 1024, 1056, 1536, 2016, 2048, 2080, 3072, 4064, 4096, 4128})
            measure(N, N, N, N, 256);
    } else if (mode == "height") {
        std::printf("%s", header);
        // 4, 8, 16 KiB and each + 64 B; then 4,352 B and 6 KiB, which give a
        // strip's rows 16 and 2 groups of L1 sets instead of 1.
        for (const std::size_t ldb : {1024, 1040, 2048, 2064, 4096, 4112, 1088, 1536})
            for (const std::size_t kc : {16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024})
                measure(512, 2048, 1024, ldb, kc);
    } else {
        std::fprintf(stderr, "usage: %s chase|ldb|size|height\n", argv[0]);
        return 2;
    }
}

#endif  // HPC_HAS_AVX512 && __linux__
