/**
 * @file bench_gemm_lowp.cpp
 * @brief Google Benchmark driver for the low-precision CPU GEMMs, with the
 *        FP32 kernels of the same ISA alongside as the baseline.
 *
 *   ./build/benchmarks/bench_gemm_lowp --benchmark_filter=<regex>
 *
 * Benchmarks are <Kernel>/<format>/N=<N> on square matrices. Inputs are
 * converted to the format before timing (conversion is a one-off cost a
 * real workload pays when it stores weights or activations), so each
 * iteration is one call: packing plus the multiplication, as in bench_gemm.
 * "GFLOP/s" is 2·N³ / time for every format, INT8 included (one
 * multiply-add = 2 operations), so the rows compare directly.
 *
 * Only the families compiled for this target are registered (see the
 * HPC_HAS_* checks): SME2 on Apple M4, NEON with BF16 / I8MM / FP16 on
 * Armv8.6+, AVX-512 BF16 / VNNI on x86.
 */

#include "gemm/avx512.hpp"
#include "gemm/avx512_lowp.hpp"
#include "gemm/neon.hpp"
#include "gemm/neon_lowp.hpp"
#include "gemm/sme.hpp"
#include "gemm/sme_lowp.hpp"
#include "hpc/isa.hpp"
#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <random>
#include <string>

namespace {

using hpc::bf16;
using hpc::fp16;
using hpc::Matrix;
using namespace hpc::gemm;

template <typename T>
Matrix<T> random_matrix(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::uniform_int_distribution<int> di(-128, 127);
    Matrix<T> m(n, n);
    for (std::size_t i = 0; i < m.size(); ++i) {
        if constexpr (std::is_same_v<T, float>)
            m.data()[i] = d(rng);
        else if constexpr (std::is_same_v<T, std::int8_t>)
            m.data()[i] = static_cast<std::int8_t>(di(rng));
        else if constexpr (std::is_same_v<T, bf16>)
            m.data()[i] = hpc::to_bf16(d(rng));
        else
            m.data()[i] = hpc::to_fp16(d(rng));
    }
    return m;
}

template <typename In>
using Out = std::conditional_t<std::is_same_v<In, std::int8_t>, std::int32_t, float>;

template <typename In>
void add(const std::string& name, const char* format, void (*kernel)(const Matrix<In>&, const Matrix<In>&,
                                                                       Matrix<Out<In>>&)) {
    for (const std::size_t n : {256, 1024, 2048, 4096}) {
        benchmark::RegisterBenchmark(name + "/" + format + "/N=" + std::to_string(n),
                                     [=](benchmark::State& state) {
                                         const Matrix<In> A = random_matrix<In>(n, 1);
                                         const Matrix<In> B = random_matrix<In>(n, 2);
                                         Matrix<Out<In>> C(n, n);
                                         kernel(A, B, C);  // warm-up
                                         for (auto _ : state) {
                                             kernel(A, B, C);
                                             benchmark::DoNotOptimize(C.data());
                                             benchmark::ClobberMemory();
                                         }
                                         state.counters["GFLOP/s"] = benchmark::Counter(
                                             2.0 * double(n) * double(n) * double(n),
                                             benchmark::Counter::kIsIterationInvariantRate,
                                             benchmark::Counter::OneK::kIs1000);
                                     })
            ->Unit(benchmark::kMillisecond);
    }
}

void register_all() {
#if HPC_HAS_SME
    add<float>("Sme", "f32", gemm_sme<float>);
    add<fp16>("SmeWidening", "fp16", gemm_sme_widening<fp16>);
    add<bf16>("SmeWidening", "bf16", gemm_sme_widening<bf16>);
    add<std::int8_t>("SmeWidening", "int8", gemm_sme_widening<std::int8_t>);
#endif
#if HPC_HAS_NEON
    add<float>("NeonBlocked", "f32", gemm_neon_blocked<float>);
#endif
#if HPC_HAS_NEON_LOWP
    add<bf16>("NeonBfmmla", "bf16", gemm_neon_bf16);
    add<std::int8_t>("NeonSmmla", "int8", gemm_neon_i8);
    add<fp16>("NeonFmlal", "fp16", gemm_neon_f16);
    add<fp16>("NeonFp16Acc", "fp16", gemm_neon_f16_acc);
#endif
#if HPC_HAS_AVX512
    add<float>("Avx512Blocked", "f32", gemm_avx512_blocked<float>);
#endif
#if HPC_HAS_AVX512_LOWP
    add<bf16>("Avx512Dpbf16", "bf16", gemm_avx512_bf16);
    add<std::int8_t>("Avx512Vnni", "int8", gemm_avx512_i8);
#endif
}

}  // namespace

int main(int argc, char** argv) {
    register_all();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
