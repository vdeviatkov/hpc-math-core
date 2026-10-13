#pragma once

/**
 * @file core_clock.hpp
 * @brief The running core's clock, measured with a chain of dependent adds.
 *
 * An integer add whose input is the previous add's result completes one per
 * cycle on every AArch64 core, so the time per add is one cycle. macOS does
 * not report the clock; measuring it in the same process, right before the
 * experiment, also captures the frequency the core actually runs at.
 */

#include <algorithm>
#include <chrono>

#if !defined(__aarch64__)
    #error "core_clock.hpp measures AArch64 cores"
#endif

namespace hpc::experiments {

/// Cycles per nanosecond (= GHz) of the core running this thread; best of 20.
inline double core_ghz() {
    double best_ns = 1e9;
    for (int rep = 0; rep < 20; ++rep) {
        long x                = 0;
        constexpr long kLoops = 2'000'000;
        const auto t0         = std::chrono::steady_clock::now();
        for (long i = 0; i < kLoops; ++i)
            asm volatile(".rept 100\n add %0, %0, #1\n .endr" : "+r"(x));
        const std::chrono::duration<double, std::nano> dt = std::chrono::steady_clock::now() - t0;
        best_ns = std::min(best_ns, dt.count() / (kLoops * 100.0));
    }
    return 1.0 / best_ns;
}

}  // namespace hpc::experiments
