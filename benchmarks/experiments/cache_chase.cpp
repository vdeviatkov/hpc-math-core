/**
 * @file cache_chase.cpp
 * @brief Pointer chase: load latency over L cache lines spaced `stride` bytes.
 *
 *   ./build/benchmarks/cache_chase > chase.csv
 *
 * Each line holds a pointer to the next one, visited in a random cyclic
 * order, so every load depends on the previous one and the hardware
 * prefetcher cannot help. The latency step as L grows shows how many lines
 * at that stride a cache level keeps. Two strides: 16,384 B (a power of two)
 * and 16,512 B (one 128-byte line more). This measures the CPU core's load
 * path; it runs on any machine.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

int main() {
    constexpr std::size_t kArena = std::size_t(1) << 30;
    auto* arena = static_cast<char*>(std::aligned_alloc(1 << 14, kArena));
    if (arena == nullptr)
        return 1;
    std::memset(arena, 0, kArena);
    std::mt19937 rng(1);

    std::printf("stride_bytes,lines,ns_per_load\n");
    for (const std::size_t stride : {16384, 16512}) {
        for (const std::size_t lines :
             {8, 16, 64, 256, 384, 448, 512, 576, 640, 704, 768, 896, 1024, 1536, 2048}) {
            std::vector<std::size_t> idx(lines);
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);
            for (std::size_t i = 0; i < lines; ++i)
                *reinterpret_cast<char**>(arena + idx[i] * stride) =
                    arena + idx[(i + 1) % lines] * stride;

            char* p = arena + idx[0] * stride;
            for (std::size_t i = 0; i < 8 * lines; ++i)  // warm-up
                p = *reinterpret_cast<char**>(p);
            double best = 1e9;
            for (int rep = 0; rep < 5; ++rep) {
                constexpr long kLoads = 4'000'000;
                const auto t0 = std::chrono::steady_clock::now();
                for (long i = 0; i < kLoads; ++i)
                    p = *reinterpret_cast<char**>(p);
                const std::chrono::duration<double, std::nano> dt =
                    std::chrono::steady_clock::now() - t0;
                best = std::min(best, dt.count() / kLoads);
            }
            if (p == nullptr)  // keeps the chase observable
                return 1;
            std::printf("%zu,%zu,%.2f\n", stride, lines, best);
        }
    }
    std::free(arena);
}
