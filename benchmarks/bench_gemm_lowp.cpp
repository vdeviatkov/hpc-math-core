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
 * Avx512{Dpbf16,Vnni}Tile<R>x<V>[Kc<bytes>] rows (add_tiles) run the AVX-512
 * kernels with other register tiles and k blocks.
 *
 * Only the families compiled for this target are registered (see the
 * HPC_HAS_* checks): SME2 on Apple M4, NEON with BF16 / I8MM / FP16 on
 * Armv8.6+, AVX-512 BF16 / VNNI on x86.
 */

#include "gemm/simd.hpp"
#include "gemm/sme.hpp"
#include "hpc/isa.hpp"
#include "hpc/lowp.hpp"
#include "hpc/matrix.hpp"
#include "simd_params.hpp"
#include "sme_params.hpp"

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

/// real_time: rate against wall-clock time, for kernels that run several
/// threads (CPU time is the calling thread's only).
template <typename In>
void add(const std::string& name, const char* format,
         void (*kernel)(const Matrix<In>&, const Matrix<In>&, Matrix<Out<In>>&), bool real_time = false) {
    for (const std::size_t n : {256, 1024, 2048, 4000, 4096}) {
        auto* b = benchmark::RegisterBenchmark(name + "/" + format + "/N=" + std::to_string(n),
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
                                     });
        b->Unit(benchmark::kMillisecond);
        if (real_time)
            b->UseRealTime();
    }
}

void register_all() {
#if HPC_HAS_SME
    using hpc::bench::sme, hpc::bench::sme_mt;
    add<float>("Sme", "f32", sme);
    add<fp16>("SmeWidening", "fp16", sme);
    add<bf16>("SmeWidening", "bf16", sme);
    add<std::int8_t>("SmeWidening", "int8", sme);
    // Both SME units (SmeParams::Threads): docs/kernels/sme.md, "Several SME units".
    add<float>("SmeMt", "f32", sme_mt, true);
    add<fp16>("SmeWideningMt", "fp16", sme_mt, true);
    add<bf16>("SmeWideningMt", "bf16", sme_mt, true);
    add<std::int8_t>("SmeWideningMt", "int8", sme_mt, true);
#endif
#if HPC_HAS_NEON
    add<float>("NeonBlocked", "f32", hpc::bench::simd_blocked<Q>);
#endif
#if HPC_HAS_NEON_LOWP
    add<bf16>("NeonBfmmla", "bf16", hpc::bench::simd_lowp<QMmla, bf16>);
    add<std::int8_t>("NeonSmmla", "int8", hpc::bench::simd_lowp<QMmla, std::int8_t>);
    add<fp16>("NeonFmlal", "fp16", hpc::bench::simd_lowp<QFmlal, fp16>);
    add<fp16>("NeonFp16Acc", "fp16", hpc::bench::simd_lowp<QFmlaF16, fp16>);
#endif
#if HPC_HAS_AVX512
    add<float>("Avx512Blocked", "f32", hpc::bench::simd_blocked<Zmm>);
#endif
#if HPC_HAS_AVX512_LOWP
    add<bf16>("Avx512Dpbf16", "bf16", hpc::bench::simd_lowp<ZmmDot, bf16>);
    add<std::int8_t>("Avx512Vnni", "int8", hpc::bench::simd_lowp<ZmmDot, std::int8_t>);
#endif
}

/// gemm_simd_lowp<ZmmDot> with an R × V tile and a k block of KC bytes, at
/// the measured Mc and Nc.
template <typename In, int R, int V, std::size_t KC>
void zmm_dot_tile(const Matrix<In>& A, const Matrix<In>& B, Matrix<lowp_out_t<In>>& C) {
    using P = hpc::bench::LowpParams<ZmmDot>;
    gemm_simd_lowp<ZmmDot, R, V, P::Mc, KC / sizeof(In), P::Nc>(A, B, C);
}

/// Register-tile sweep of the AVX-512 kernels:
/// Avx512{Dpbf16,Vnni}Tile<R>x<V>[Kc<bytes>]/<format>/N=<n>. The measured best
/// is 8x2 with Kc = 1024 bytes; a strip of B is Kc · 16·V bytes, so the wider
/// tiles also run at Kc = 512 to keep it inside Zen 5's 48 KiB L1.
void add_tiles() {
#if HPC_HAS_AVX512_LOWP
#define HPC_AVX512_LOWP_TILE(R, V, KC, SUFFIX)                                                     \
    add<bf16>("Avx512Dpbf16Tile" #R "x" #V SUFFIX, "bf16", zmm_dot_tile<bf16, R, V, KC>);           \
    add<std::int8_t>("Avx512VnniTile" #R "x" #V SUFFIX, "int8", zmm_dot_tile<std::int8_t, R, V, KC>)
    // 32 ZMM registers: R·V accumulators + V for B + 1 broadcast.
    HPC_AVX512_LOWP_TILE(8, 2, 1024, ""); HPC_AVX512_LOWP_TILE(6, 2, 1024, "");
    HPC_AVX512_LOWP_TILE(4, 3, 1024, ""); HPC_AVX512_LOWP_TILE(16, 1, 1024, "");
    HPC_AVX512_LOWP_TILE(12, 2, 1024, ""); HPC_AVX512_LOWP_TILE(14, 2, 1024, "");
    HPC_AVX512_LOWP_TILE(6, 3, 1024, ""); HPC_AVX512_LOWP_TILE(8, 3, 1024, "");
    HPC_AVX512_LOWP_TILE(4, 4, 1024, ""); HPC_AVX512_LOWP_TILE(6, 4, 1024, "");
    HPC_AVX512_LOWP_TILE(8, 3, 512, "Kc512"); HPC_AVX512_LOWP_TILE(4, 4, 512, "Kc512");
    HPC_AVX512_LOWP_TILE(6, 4, 512, "Kc512");
#undef HPC_AVX512_LOWP_TILE
#endif
}

/// gemm_simd_lowp<Ops> with an R × V tile and Mc × KC × Nc blocks (KC in
/// input elements).
template <template <typename> class Ops, typename In, int R, int V, std::size_t MC, std::size_t KC, std::size_t NC>
void neon_lowp_tile(const Matrix<In>& A, const Matrix<In>& B, Matrix<lowp_out_t<In>>& C) {
    gemm_simd_lowp<Ops, R, V, MC, KC, NC>(A, B, C);
}

/// Tile and block sweeps of the NEON low-precision kernels:
///   Neon{Bfmmla,Smmla}Tile<R>x<V>   R row pairs × V column pairs of 2 × 2 accumulators
///   Neon{Fmlal,Fp16Acc}Tile4x<V>    4 rows (the lanes of A) × V accumulators
///   Neon<Kernel>Block<Mc>x<KcBytes>x<Nc>   the measured tile with other blocks
/// Shapes are those kTileFits allows in the 32 vector registers.
void add_neon_lowp_sweeps() {
#if HPC_HAS_NEON_LOWP
    using P = hpc::bench::LowpParams<QMmla>;
#define HPC_MMLA_TILE(R, V)                                                                                  \
    add<bf16>("NeonBfmmlaTile" #R "x" #V, "bf16", neon_lowp_tile<QMmla, bf16, R, V, P::Mc, P::Kc<bf16>, P::Nc>); \
    add<std::int8_t>("NeonSmmlaTile" #R "x" #V, "int8",                                                      \
                     neon_lowp_tile<QMmla, std::int8_t, R, V, P::Mc, P::Kc<std::int8_t>, P::Nc>)
    HPC_MMLA_TILE(2, 4); HPC_MMLA_TILE(4, 2); HPC_MMLA_TILE(3, 4); HPC_MMLA_TILE(4, 3);
    HPC_MMLA_TILE(4, 4); HPC_MMLA_TILE(5, 4); HPC_MMLA_TILE(4, 5); HPC_MMLA_TILE(3, 6);
    HPC_MMLA_TILE(6, 3); HPC_MMLA_TILE(2, 8); HPC_MMLA_TILE(8, 2);
#undef HPC_MMLA_TILE
    using F = hpc::bench::LowpParams<QFmlal>;
#define HPC_FMLA_TILE(V)                                                                                      \
    add<fp16>("NeonFmlalTile4x" #V, "fp16", neon_lowp_tile<QFmlal, fp16, 4, V, F::Mc, F::Kc<fp16>, F::Nc>);       \
    add<fp16>("NeonFp16AccTile4x" #V, "fp16", neon_lowp_tile<QFmlaF16, fp16, 4, V, F::Mc, F::Kc<fp16>, F::Nc>)
    HPC_FMLA_TILE(2); HPC_FMLA_TILE(4); HPC_FMLA_TILE(6);
#undef HPC_FMLA_TILE
    add<fp16>("NeonFp16AccTile4x3", "fp16", neon_lowp_tile<QFmlaF16, fp16, 4, 3, F::Mc, F::Kc<fp16>, F::Nc>);
    add<fp16>("NeonFp16AccTile4x5", "fp16", neon_lowp_tile<QFmlaF16, fp16, 4, 5, F::Mc, F::Kc<fp16>, F::Nc>);

    // Blocks at the measured tiles: Mc ∈ {128, 256, 512} × k block of
    // {512, 1024, 2048, 4096} bytes × Nc ∈ {256, 512, 1024}.
#define HPC_LOWP_BLOCK(MC, KB, NC)                                                                                \
    add<bf16>("NeonBfmmlaBlock" #MC "x" #KB "x" #NC, "bf16",                                                      \
              neon_lowp_tile<QMmla, bf16, P::R, P::V, MC, KB / sizeof(bf16), NC>);                                \
    add<std::int8_t>("NeonSmmlaBlock" #MC "x" #KB "x" #NC, "int8",                                                \
                     neon_lowp_tile<QMmla, std::int8_t, P::R, P::V, MC, KB, NC>);                                 \
    add<fp16>("NeonFmlalBlock" #MC "x" #KB "x" #NC, "fp16",                                                       \
              neon_lowp_tile<QFmlal, fp16, F::R, F::V, MC, KB / sizeof(fp16), NC>);                               \
    add<fp16>("NeonFp16AccBlock" #MC "x" #KB "x" #NC, "fp16",                                                     \
              neon_lowp_tile<QFmlaF16, fp16, F::R, F::V, MC, KB / sizeof(fp16), NC>)
#define HPC_LOWP_BLOCK_K(MC, NC) \
    HPC_LOWP_BLOCK(MC, 512, NC); HPC_LOWP_BLOCK(MC, 1024, NC); HPC_LOWP_BLOCK(MC, 2048, NC); HPC_LOWP_BLOCK(MC, 4096, NC)
    HPC_LOWP_BLOCK_K(128, 256); HPC_LOWP_BLOCK_K(128, 512); HPC_LOWP_BLOCK_K(128, 1024);
    HPC_LOWP_BLOCK_K(256, 256); HPC_LOWP_BLOCK_K(256, 512); HPC_LOWP_BLOCK_K(256, 1024);
    HPC_LOWP_BLOCK_K(512, 256); HPC_LOWP_BLOCK_K(512, 512); HPC_LOWP_BLOCK_K(512, 1024);
#undef HPC_LOWP_BLOCK_K
#undef HPC_LOWP_BLOCK
#endif
}

/// gemm_sme on a widening type with an R × V ZA shape and Mc × KG·G × Nc
/// blocks (KG in k groups).
template <typename In, int R, int V, std::size_t MC, std::size_t KG, std::size_t NC>
void sme_widening_tile(const Matrix<In>& A, const Matrix<In>& B, Matrix<sme_out_t<In>>& C) {
    gemm_sme<In, R, V, MC, KG * kSmeG<In>, NC>(A, B, C);
}

/// Shape and block sweeps of the SME widening kernels:
///   SmeWideningTile<R>x<V>/<format>          every 4-tile shape, FP32's blocks in k groups
///   SmeWideningBlock<Mc>x<KG>x<Nc>/<format>  2 × 2 and 1 × 4 with other blocks (KG k groups)
void add_sme_widening_sweeps() {
#if HPC_HAS_SME
#define HPC_SME_W(NAME, R, V, MC, KG, NC)                                                         \
    add<fp16>(NAME, "fp16", sme_widening_tile<fp16, R, V, MC, KG, NC>);                           \
    add<bf16>(NAME, "bf16", sme_widening_tile<bf16, R, V, MC, KG, NC>);                           \
    add<std::int8_t>(NAME, "int8", sme_widening_tile<std::int8_t, R, V, MC, KG, NC>)
    HPC_SME_W("SmeWideningTile1x4", 1, 4, 1024, 2048, 4096);
    HPC_SME_W("SmeWideningTile2x2", 2, 2, 1024, 2048, 4096);
    HPC_SME_W("SmeWideningTile4x1", 4, 1, 1024, 2048, 4096);
#define HPC_SME_WB(R, V, MC, KG, NC) \
    HPC_SME_W("SmeWideningBlock" #R "x" #V "_" #MC "x" #KG "x" #NC, R, V, MC, KG, NC)
#define HPC_SME_WB_ALL(R, V)                                                                         \
    HPC_SME_WB(R, V, 512, 1024, 4096); HPC_SME_WB(R, V, 512, 2048, 4096); HPC_SME_WB(R, V, 512, 4096, 4096); \
    HPC_SME_WB(R, V, 1024, 1024, 4096); HPC_SME_WB(R, V, 1024, 4096, 4096);                          \
    HPC_SME_WB(R, V, 2048, 1024, 4096); HPC_SME_WB(R, V, 2048, 2048, 4096);                          \
    HPC_SME_WB(R, V, 1024, 2048, 2048); HPC_SME_WB(R, V, 512, 2048, 2048)
    HPC_SME_WB_ALL(2, 2);
    HPC_SME_WB_ALL(1, 4);
#undef HPC_SME_WB_ALL
#undef HPC_SME_WB
#undef HPC_SME_W
#endif
}

}  // namespace

int main(int argc, char** argv) {
    register_all();
    add_tiles();
    add_neon_lowp_sweeps();
    add_sme_widening_sweeps();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
