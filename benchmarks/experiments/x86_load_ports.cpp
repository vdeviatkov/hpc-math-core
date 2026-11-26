/**
 * @file x86_load_ports.cpp
 * @brief Vector loads per cycle from L1 on an x86 core: 128-, 256- and
 *        512-bit loads, and 512-bit loads mixed with FMAs.
 *
 *   ./build/benchmarks/x86_load_ports > load_ports.csv
 *
 * Each loop body issues 12 independent aligned loads from a 4 KiB buffer
 * (L1-resident) into 12 registers that nothing reads, unrolled so loop
 * overhead is 2 instructions per 12 loads. The "fma" rows add 12 register-
 * only FMAs per body, the micro-kernel's mix with B held in registers.
 * Cycles are measured with the core clock (core_clock.hpp); best of 5
 * runs of 50 M iterations. stdout gets CSV rows: width, loads_per_cycle,
 * fmas_per_cycle.
 */

#include "core_clock.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>

#if defined(__x86_64__) && defined(__AVX512F__)

namespace {

alignas(64) float g_buf[1024];

#define LOAD12(INSN, R)                                                                            \
    INSN " 0(%[p]), %%" R "0\n\t" INSN " 64(%[p]), %%" R "1\n\t" INSN " 128(%[p]), %%" R "2\n\t"    \
    INSN " 192(%[p]), %%" R "3\n\t" INSN " 256(%[p]), %%" R "4\n\t" INSN " 320(%[p]), %%" R "5\n\t" \
    INSN " 384(%[p]), %%" R "6\n\t" INSN " 448(%[p]), %%" R "7\n\t" INSN " 512(%[p]), %%" R "8\n\t" \
    INSN " 576(%[p]), %%" R "9\n\t" INSN " 640(%[p]), %%" R "10\n\t" INSN " 704(%[p]), %%" R "11\n\t"
#define FMA12                                                                                         \
    "vfmadd231ps %%zmm20, %%zmm21, %%zmm12\n\tvfmadd231ps %%zmm20, %%zmm21, %%zmm13\n\t"              \
    "vfmadd231ps %%zmm20, %%zmm21, %%zmm14\n\tvfmadd231ps %%zmm20, %%zmm21, %%zmm15\n\t"              \
    "vfmadd231ps %%zmm20, %%zmm21, %%zmm16\n\tvfmadd231ps %%zmm20, %%zmm21, %%zmm17\n\t"              \
    "vfmadd231ps %%zmm20, %%zmm21, %%zmm18\n\tvfmadd231ps %%zmm20, %%zmm21, %%zmm19\n\t"              \
    "vfmadd231ps %%zmm20, %%zmm21, %%zmm22\n\tvfmadd231ps %%zmm20, %%zmm21, %%zmm23\n\t"              \
    "vfmadd231ps %%zmm20, %%zmm21, %%zmm24\n\tvfmadd231ps %%zmm20, %%zmm21, %%zmm25\n\t"
#define CLOBBERS                                                                                       \
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11",  \
        "xmm12", "xmm13", "xmm14", "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21",     \
        "xmm22", "xmm23", "xmm24", "xmm25", "memory", "cc"

template <int Kind>
void loop(std::uint64_t iters) {
    const float* p = g_buf;
    if constexpr (Kind == 128)
        asm volatile("1:\n\t" LOAD12("vmovaps", "xmm") "dec %[n]\n\tjnz 1b" : [n] "+r"(iters) : [p] "r"(p) : CLOBBERS);
    else if constexpr (Kind == 256)
        asm volatile("1:\n\t" LOAD12("vmovaps", "ymm") "dec %[n]\n\tjnz 1b" : [n] "+r"(iters) : [p] "r"(p) : CLOBBERS);
    else if constexpr (Kind == 512)
        asm volatile("1:\n\t" LOAD12("vmovaps", "zmm") "dec %[n]\n\tjnz 1b" : [n] "+r"(iters) : [p] "r"(p) : CLOBBERS);
    else  // 512-bit loads + FMAs
        asm volatile("1:\n\t" LOAD12("vmovaps", "zmm") FMA12 "dec %[n]\n\tjnz 1b"
                     : [n] "+r"(iters)
                     : [p] "r"(p)
                     : CLOBBERS);
}

template <int Kind>
void measure(const char* name, double ghz) {
    constexpr std::uint64_t kIters = 50'000'000;
    loop<Kind>(kIters / 10);  // warm-up
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        loop<Kind>(kIters);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    const double cycles = best * ghz * 1e9;
    const double loads  = 12.0 * kIters / cycles;
    std::printf("%s,%.2f,%.2f\n", name, loads, Kind == 0 ? loads : 0.0);
    std::fflush(stdout);
}

}  // namespace

int main() {
    const double ghz = hpc::experiments::core_ghz();
    std::fprintf(stderr, "core clock: %.3f GHz\n", ghz);
    std::printf("kind,loads_per_cycle,fmas_per_cycle\n");
    measure<128>("128-bit loads", ghz);
    measure<256>("256-bit loads", ghz);
    measure<512>("512-bit loads", ghz);
    measure<0>("512-bit loads + FMAs", ghz);
}

#else
int main() {
    std::fprintf(stderr, "x86_load_ports needs x86-64 with AVX-512\n");
    return 0;
}
#endif
