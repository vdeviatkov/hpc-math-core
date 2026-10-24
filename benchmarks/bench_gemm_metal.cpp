/**
 * @file bench_gemm_metal.cpp
 * @brief Google Benchmark driver for the Metal GEMM kernels (gemm/metal.hpp).
 *
 *   ./build/benchmarks/bench_gemm_metal --benchmark_filter=<regex>
 *
 * Every benchmark, Metal<Kernel>/f32/N=<N>, times the GEMM alone, as
 * bench_gemm_cuda does: A, B and C are allocated and filled in GPU-visible
 * memory before timing (fp16 copies of A and B too, for the F16 rows). The
 * time is GPU time, the command buffer's GPUStartTime..GPUEndTime. Unlike
 * the CUDA driver, each iteration encodes `reps` GEMMs into one command
 * buffer, with reps chosen so a command buffer runs for at least ~5 ms: at
 * N=256 a single GEMM is shorter than the command buffer's own fixed cost.
 * The reported time is per command buffer; GFLOP/s counts all reps GEMMs. One untimed call precedes the
 * loop, so shader compilation is not counted.
 *
 * The "precision" counter is the bits of the multiply inputs (16 for the
 * F16 rows, which accumulate in fp32); "reps" the GEMMs per command buffer.
 *
 * MetalVariant/<function>/N=<N> rows run the tile variants of Levels 2-5
 * (metal_kernel_variants); MetalFmaPeak/<f32|f16> measures the arithmetic
 * peak the GEMM rows are compared against. Everything reports SKIPPED
 * without Metal.
 */

#include "gemm/metal.hpp"
#include "hpc/matrix.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <string>

namespace {

using namespace hpc::gemm;

void fill_random(hpc::Matrix<float>& M, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < M.size(); ++i)
        M.data()[i] = dist(rng);
}

bool skip(benchmark::State& state) {
    if (!metal_available()) {
        state.SkipWithMessage("Metal not available");
        return true;
    }
    return false;
}

/// Register <name>/f32/N=<n> for each size; run(gemm, reps) returns ms per GEMM.
void add(const std::string& name, std::function<double(MetalGemm&, int)> run, int precision_bits,
         std::initializer_list<std::size_t> sizes) {
    for (const std::size_t n : sizes) {
        benchmark::RegisterBenchmark(name + "/f32/N=" + std::to_string(n), [=](benchmark::State& state) {
            if (skip(state))
                return;
            hpc::Matrix<float> A(n, n), B(n, n);
            fill_random(A, 1);
            fill_random(B, 2);
            MetalGemm g(n, n, n);
            g.set_inputs(A.data(), B.data());
            const double once = run(g, 1);  // warm-up and estimate, untimed
            const int reps    = std::clamp(static_cast<int>(std::ceil(5.0 / std::max(once, 1e-3))), 1, 256);
            // One iteration is one command buffer of `reps` GEMMs: its time is
            // reported whole and the flop count scaled to match, so GFLOP/s is
            // per GEMM while Google Benchmark's own time column is per batch.
            for (auto _ : state)
                state.SetIterationTime(run(g, reps) * reps / 1e3);
            const double flops        = 2.0 * double(n) * double(n) * double(n) * reps;
            state.counters["GFLOP/s"] = benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                                                           benchmark::Counter::OneK::kIs1000);
            state.counters["precision"] = precision_bits;
            state.counters["reps"]      = reps;
        })
            ->UseManualTime()
            ->Unit(benchmark::kMillisecond);
    }
}

constexpr std::initializer_list<std::size_t> kSizes = {256, 1024, 2048, 4096, 8192};

void register_all() {
    auto kernel = [](MetalKernel k) { return [k](MetalGemm& g, int reps) { return g.run_ms(k, reps); }; };
    // clang-format off
    add("MetalNaive",     kernel(MetalKernel::kNaive),     32, kSizes);
    add("MetalTiled",     kernel(MetalKernel::kTiled),     32, kSizes);
    add("MetalRegTile",   kernel(MetalKernel::kRegTile),   32, kSizes);
    add("MetalSimd",      kernel(MetalKernel::kSimd),      32, kSizes);
    add("MetalSimdTg",    kernel(MetalKernel::kSimdTg),    32, kSizes);
    add("MetalSimdTgF16", kernel(MetalKernel::kSimdTgF16), 16, kSizes);
    add("MetalMps",       kernel(MetalKernel::kMps),       32, kSizes);
    add("MetalMpsF16",    kernel(MetalKernel::kMpsF16),    16, kSizes);
    // clang-format on

    for (const std::string& fn : metal_kernel_variants())
        add("MetalVariant/" + fn, [fn](MetalGemm& g, int reps) { return g.run_variant_ms(fn, reps); },
            fn.find("f16") != std::string::npos ? 16 : 32, {2048, 4096});

    for (const bool f16 : {false, true})
        benchmark::RegisterBenchmark(std::string("MetalFmaPeak/") + (f16 ? "f16" : "f32"),
                                     [f16](benchmark::State& state) {
                                         if (skip(state))
                                             return;
                                         double gflops = 0.0;
                                         for (auto _ : state)
                                             gflops = metal_fma_peak_gflops(f16);
                                         state.counters["GFLOP/s"] = gflops;
                                     })
            ->Iterations(1)
            ->Unit(benchmark::kMillisecond);
}

}  // namespace

int main(int argc, char** argv) {
    register_all();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::AddCustomContext("metal_device", metal_available() ? metal_device_name() : "none");
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
