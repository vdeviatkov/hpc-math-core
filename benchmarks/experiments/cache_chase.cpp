/**
 * @file cache_chase.cpp
 * @brief Pointer chases that measure the CPU core's caches and TLBs.
 *
 *   ./build/benchmarks/cache_chase [stride|sizes|tlb] > out.csv
 *
 * Each 128-byte line holds a pointer to the next one, visited in a random
 * cyclic order, so every load depends on the previous one and the hardware
 * prefetcher cannot help; the result is load-to-use latency. Best of 5 runs.
 *
 *   stride  (default) L lines spaced 16,384 B (a power of two) or 16,512 B
 *           apart: how many lines at that stride each cache level keeps.
 *   sizes   L lines packed into a contiguous working set of 8 KiB … 1 GiB:
 *           latency of L1, L2, the system cache and DRAM.
 *   tlb     P lines, one per 16 KiB page, against the same number of lines
 *           packed densely. The line inside each page rotates so the lines
 *           spread over all cache sets; the difference between the two is
 *           the cost of translation once P pages exceed a TLB's reach.
 *
 * Latencies are reported in ns and in cycles of the core's clock, measured
 * at the start of the run (core_clock.hpp). This measures the CPU core's
 * load path; it needs an AArch64 core, and the tlb mode assumes 16 KiB
 * pages (Apple Silicon).
 */

#include "core_clock.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kLine = 128, kPage = 16384;

/// ns per dependent load over the given line addresses, visited in random order.
double chase(std::vector<char*> lines, std::mt19937& rng) {
    std::shuffle(lines.begin(), lines.end(), rng);
    for (std::size_t i = 0; i < lines.size(); ++i)
        *reinterpret_cast<char**>(lines[i]) = lines[(i + 1) % lines.size()];
    char* p = lines[0];
    for (std::size_t i = 0; i < 8 * lines.size(); ++i)  // warm-up
        p = *reinterpret_cast<char**>(p);
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        constexpr long kLoads = 4'000'000;
        const auto t0 = std::chrono::steady_clock::now();
        for (long i = 0; i < kLoads; ++i)
            p = *reinterpret_cast<char**>(p);
        const std::chrono::duration<double, std::nano> dt = std::chrono::steady_clock::now() - t0;
        best = std::min(best, dt.count() / kLoads);
    }
    if (p == nullptr)  // keeps the chase observable
        std::abort();
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "stride";
    constexpr std::size_t kArena = std::size_t(1) << 30;
    auto* arena = static_cast<char*>(std::aligned_alloc(kPage, kArena));
    if (arena == nullptr)
        return 1;
    std::memset(arena, 0, kArena);  // fault every page in
    std::mt19937 rng(1);
    const double ghz = hpc::experiments::core_ghz();
    std::fprintf(stderr, "core clock: %.3f GHz\n", ghz);

    if (mode == "stride") {
        std::printf("stride_bytes,lines,ns_per_load,cycles\n");
        for (const std::size_t stride : {16384, 16512})
            for (const std::size_t n :
                 {8, 16, 64, 256, 384, 448, 512, 576, 640, 704, 768, 896, 1024, 1536, 2048}) {
                std::vector<char*> lines(n);
                for (std::size_t i = 0; i < n; ++i)
                    lines[i] = arena + i * stride;
                const double ns = chase(lines, rng);
                std::printf("%zu,%zu,%.2f,%.1f\n", stride, n, ns, ns * ghz);
            }
    } else if (mode == "sizes") {
        std::printf("working_set_bytes,ns_per_load,cycles\n");
        for (std::size_t bytes = 8 << 10; bytes <= kArena; bytes *= 2) {
            std::vector<char*> lines(bytes / kLine);
            for (std::size_t i = 0; i < lines.size(); ++i)
                lines[i] = arena + i * kLine;
            const double ns = chase(lines, rng);
            std::printf("%zu,%.2f,%.1f\n", bytes, ns, ns * ghz);
        }
    } else if (mode == "tlb") {
        std::printf("pages,ns_one_line_per_page,ns_dense,cycles_one_line_per_page,cycles_dense\n");
        for (const std::size_t pages :
             {16, 64, 128, 136, 144, 152, 160, 176, 192, 256, 512, 1024, 2048, 2560, 3072, 3328,
              3584, 3840, 4096, 6144, 8192, 16384, 32768, 65536}) {
            std::vector<char*> spread(pages), dense(pages);
            for (std::size_t i = 0; i < pages; ++i) {
                spread[i] = arena + i * kPage + (i % (kPage / kLine)) * kLine;
                dense[i]  = arena + i * kLine;
            }
            const double s = chase(spread, rng), d = chase(dense, rng);
            std::printf("%zu,%.2f,%.2f,%.1f,%.1f\n", pages, s, d, s * ghz, d * ghz);
        }
    } else {
        std::fprintf(stderr, "usage: %s [stride|sizes|tlb]\n", argv[0]);
        return 2;
    }
    std::free(arena);
}
