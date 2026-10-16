/**
 * @file bench_gemm.cpp
 * @brief Google Benchmark driver for the CPU GEMM kernels.
 *
 *   ./build/benchmarks/bench_gemm --benchmark_filter=<regex>
 *
 * Every kernel runs on square N×N matrices, f64 and f32, and reports
 * GFLOP/s = 2·N³ / time. Benchmarks are named <Kernel>/<f64|f32>/N=<N>
 * (e.g. NeonBlocked/f32/N=1024); matrices are allocated and filled outside
 * the timed loop. Footprint of A + B + C: 96 KB at N=64, 24 MB at N=1024 and
 * 384 MB at N=4096 for f64; half that for f32.
 *
 * A family whose ISA is absent from the build is registered anyway and
 * reports SKIPPED: its kernels are declared `= delete` there, so the kernel
 * lambda is only instantiated when the family is available (time_gemm's
 * `if constexpr`). The full kernel list therefore always appears in the
 * output, and nothing times a substitute kernel.
 */

#include "gemm/amx.hpp"
#include "gemm/avx2.hpp"
#include "gemm/avx512.hpp"
#include "gemm/blocked.hpp"
#include "gemm/kleidiai.hpp"
#include "gemm/naive.hpp"
#include "gemm/neon.hpp"
#include "gemm/prefetch.hpp"
#include "gemm/reordered.hpp"
#include "gemm/sme.hpp"
#include "gemm/sve.hpp"
#include "hpc/isa.hpp"
#include "hpc/matrix.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using hpc::kHaveAmx;
using hpc::kHaveAvx2;
using hpc::kHaveAvx512;
using hpc::kHaveKleidiAI;
using hpc::kHaveNeon;
using hpc::kHaveSme;
using hpc::kHaveSve;

constexpr const char* kNoAvx2     = "AVX2 not available on this target";
constexpr const char* kNoAvx512   = "AVX-512 not available on this target";
constexpr const char* kNoNeon     = "NEON not available on this target";
constexpr const char* kNoSve      = "SVE not available on this target";
constexpr const char* kNoSme      = "SME2 not available (build with -DHPC_ENABLE_SME=ON on Apple M4+)";
constexpr const char* kNoAmx      = "Accelerate.framework not available (Apple only, HPC_ENABLE_AMX)";
constexpr const char* kNoKleidiAI = "KleidiAI not built (needs SME2 and HPC_ENABLE_KLEIDIAI=ON)";

template <typename T>
void fill_random(hpc::Matrix<T>& M, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(T{0}, T{1});
    for (std::size_t i = 0; i < M.size(); ++i)
        M.data()[i] = dist(rng);
}

template <typename T>
constexpr const char* type_name() {
    return sizeof(T) == 4 ? "f32" : "f64";
}

void add_gflops(benchmark::State& state, std::size_t N) {
    const double n            = static_cast<double>(N);
    state.counters["GFLOP/s"] = benchmark::Counter(2.0 * n * n * n,
                                                   benchmark::Counter::kIsIterationInvariantRate,
                                                   benchmark::Counter::OneK::kIs1000);
}

/// Extra counters for one benchmark; called with a value of the element type.
struct NoCounters {
    template <typename T>
    void operator()(benchmark::State&, T) const {}
};

/// Square sizes to register. N is a template parameter, as in a fixed-size
/// call site: the kernels are header templates inlined into time_gemm, and
/// the compiler specialises their loops for the constant N (worth up to ~3×
/// for gemm_blocked at N=256). The published numbers were measured this way.
template <std::size_t... Ns>
struct Sizes {};
constexpr Sizes<64, 256, 512, 1024, 4096> kSizes{};

/// Time kernel(A, B, C) on N×N matrices, or report SKIPPED when !Available.
template <typename T, std::size_t N, bool Available, typename Kernel, typename Counters>
void time_gemm(benchmark::State& state, const char* unavailable, [[maybe_unused]] Kernel kernel,
               [[maybe_unused]] Counters counters) {
    if constexpr (!Available) {
        state.SkipWithMessage(unavailable);
    } else {
        hpc::Matrix<T> A(N, N), B(N, N), C(N, N);
        fill_random(A, 1);
        fill_random(B, 2);
        for (auto _ : state) {
            kernel(A, B, C);
            benchmark::DoNotOptimize(C.data());
            benchmark::ClobberMemory();
        }
        add_gflops(state, N);
        counters(state, T{});
    }
}

/// Register <name>/<type>/N=<n> for every size, f64 (unless !F64) then f32.
/// `kernel` is a generic lambda (auto& A, auto& B, auto& C). `real_time`
/// times by wall clock, for libraries that use several threads.
template <bool Available, bool F64 = true, std::size_t... Ns, typename Kernel,
          typename Counters = NoCounters>
void add(const std::string& name, const char* unavailable, Kernel kernel, Sizes<Ns...>,
         Counters counters = {}, bool real_time = false) {
    auto add_type = [&]<typename T>(T) {
        auto add_size = [&]<std::size_t N>(std::integral_constant<std::size_t, N>) {
            auto* b = benchmark::RegisterBenchmark(
                name + "/" + type_name<T>() + "/N=" + std::to_string(N),
                [=](benchmark::State& s) {
                    time_gemm<T, N, Available>(s, unavailable, kernel, counters);
                });
            b->Unit(benchmark::kMicrosecond);
            if (real_time)
                b->UseRealTime();
        };
        (add_size(std::integral_constant<std::size_t, Ns>{}), ...);
    };
    if constexpr (F64)
        add_type(double{});
    add_type(float{});
}

#define HPC_KERNEL(fn) [](auto& A, auto& B, auto& C) { fn(A, B, C); }

/// Vector length in elements: "vl" for SVE, "svl" for SME's streaming length.
struct SveVl {
    template <typename T>
    void operator()([[maybe_unused]] benchmark::State& s, T) const {
#if HPC_HAS_SVE
        s.counters["vl"] = static_cast<double>(sizeof(T) == 4 ? svcntw() : svcntd());
#endif
    }
};
struct SmeSvl {
    template <typename T>
    void operator()([[maybe_unused]] benchmark::State& s, T) const {
#if HPC_HAS_SME
        s.counters["svl"] = static_cast<double>(sizeof(T) == 4 ? svcntsw() : svcntsd());
#endif
    }
};

// ---------------------------------------------------------------------------
// Software prefetch: <Family>BlockedPf<D>, distance D micro-kernel rows
// ahead, at N = 256, 512, 1024. Only the blocked kernels get a variant: the
// naive ones are bound by misses prefetch cannot hide, and the reordered
// ones stream data the hardware prefetcher already follows.
// ---------------------------------------------------------------------------

#define HPC_PF_KERNEL(fn)                                                \
    [](auto& A, auto& B, auto& C) {                                      \
        using T = typename std::remove_cvref_t<decltype(A)>::value_type; \
        hpc::gemm::fn<T, D>(A, B, C);                                    \
    }

template <std::size_t D>
void add_prefetch() {
    constexpr Sizes<256, 512, 1024> opt{};
    const auto pf = [](benchmark::State& s, auto) { s.counters["pf_dist"] = double(D); };
    const auto d  = std::to_string(D);
    add<true>("BlockedPf" + d, nullptr, HPC_PF_KERNEL(gemm_blocked_prefetch), opt, pf);
    add<kHaveAvx2>("Avx2BlockedPf" + d, kNoAvx2, HPC_PF_KERNEL(gemm_avx2_blocked_prefetch), opt, pf);
    add<kHaveAvx512>("Avx512BlockedPf" + d, kNoAvx512, HPC_PF_KERNEL(gemm_avx512_blocked_prefetch),
                     opt, pf);
    add<kHaveNeon>("NeonBlockedPf" + d, kNoNeon, HPC_PF_KERNEL(gemm_neon_blocked_prefetch), opt, pf);
    add<kHaveSve>("SveBlockedPf" + d, kNoSve, HPC_PF_KERNEL(gemm_sve_blocked_prefetch), opt, pf);
}

// ---------------------------------------------------------------------------
// SME B-stride experiment: BStride/<type>/N=<n>/ldb=<ldb>/{in_place,packed}
//
// gemm_sme_blocked (B read in place) vs gemm_sme (B packed), as a function
// of the byte distance between B rows. Both run sme_detail::blocked_driver;
// only the B source differs. B is stored in an N×ldb buffer of which the
// first N columns are used, B(k, j) = B[k·ldb + j]:
//
//   1. N around 2048 and 4096 with ldb = N: a collapse confined to the exact
//      powers of two points at the row stride, not at N.
//   2. The 4096×4096 problem with ldb = 4096, 4097, 4128, 4160: the work and
//      useful data stay the same; padding slightly increases the page footprint. ldb = 4097
//      is a misalignment control (32-float row segments no longer 128 B
//      aligned, so each spans two cache lines).
//
// The names avoid a "Sme" prefix so --benchmark_filter=Sme skips these long
// runs. Results: docs/benchmarks.md, "SME: B row stride". Run with
//   --benchmark_filter=BStride --benchmark_repetitions=3
//   --benchmark_enable_random_interleaving=true --benchmark_report_aggregates_only=true
// ---------------------------------------------------------------------------

#if HPC_HAS_SME
template <typename T, bool PackB>
void time_bstride(benchmark::State& state, std::size_t N, std::size_t ldb) {
    hpc::Matrix<T> A(N, N), B(N, ldb), C(N, N);
    fill_random(A, 1);
    fill_random(B, 2);
    for (auto _ : state) {
        hpc::gemm::sme_detail::blocked_driver<T, PackB>(A, B.data(), ldb, C);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    add_gflops(state, N);
    state.counters["stride_B"] = static_cast<double>(ldb * sizeof(T));  // bytes between B rows
}
#endif

template <typename T>
void add_bstride() {
    struct Cfg {
        std::size_t N, ldb;
    };
    constexpr Cfg cfgs[] = {{1984, 1984}, {2016, 2016}, {2048, 2048}, {2080, 2080},
                            {4000, 4000}, {4032, 4032}, {4064, 4064}, {4096, 4096},
                            {4128, 4128}, {4096, 4097}, {4096, 4128}, {4096, 4160}};
    for (const Cfg c : cfgs) {
        const std::string name = std::string("BStride/") + type_name<T>() + "/N=" +
                                 std::to_string(c.N) + "/ldb=" + std::to_string(c.ldb);
#if HPC_HAS_SME
        benchmark::RegisterBenchmark(name + "/in_place", time_bstride<T, false>, c.N, c.ldb)
            ->Unit(benchmark::kMillisecond);
        benchmark::RegisterBenchmark(name + "/packed", time_bstride<T, true>, c.N, c.ldb)
            ->Unit(benchmark::kMillisecond);
#else
        for (const char* variant : {"/in_place", "/packed"})
            benchmark::RegisterBenchmark(name + variant,
                                         [](benchmark::State& s) { s.SkipWithMessage(kNoSme); });
#endif
    }
}

void register_all() {
    using namespace hpc::gemm;

    add<true>("Naive", nullptr, HPC_KERNEL(gemm_naive), kSizes);
    add<true>("Reordered", nullptr, HPC_KERNEL(gemm_reordered), kSizes);
    add<true>("Blocked", nullptr, HPC_KERNEL(gemm_blocked), kSizes,
              [](benchmark::State& s, auto) { s.counters["tile"] = double(kDefaultTile); });

    add<kHaveAvx2>("Avx2Naive", kNoAvx2, HPC_KERNEL(gemm_avx2_naive), kSizes);
    add<kHaveAvx2>("Avx2Reordered", kNoAvx2, HPC_KERNEL(gemm_avx2_reordered), kSizes);
    add<kHaveAvx2>("Avx2Blocked", kNoAvx2, HPC_KERNEL(gemm_avx2_blocked), kSizes);
    add<kHaveAvx512>("Avx512Naive", kNoAvx512, HPC_KERNEL(gemm_avx512_naive), kSizes);
    add<kHaveAvx512>("Avx512Reordered", kNoAvx512, HPC_KERNEL(gemm_avx512_reordered), kSizes);
    add<kHaveAvx512>("Avx512Blocked", kNoAvx512, HPC_KERNEL(gemm_avx512_blocked), kSizes);
    add<kHaveNeon>("NeonNaive", kNoNeon, HPC_KERNEL(gemm_neon_naive), kSizes);
    add<kHaveNeon>("NeonReordered", kNoNeon, HPC_KERNEL(gemm_neon_reordered), kSizes);
    add<kHaveNeon>("NeonBlocked", kNoNeon, HPC_KERNEL(gemm_neon_blocked), kSizes);
    add<kHaveSve>("SveNaive", kNoSve, HPC_KERNEL(gemm_sve_naive), kSizes, SveVl{});
    add<kHaveSve>("SveReordered", kNoSve, HPC_KERNEL(gemm_sve_reordered), kSizes, SveVl{});
    add<kHaveSve>("SveBlocked", kNoSve, HPC_KERNEL(gemm_sve_blocked), kSizes, SveVl{});

    // The four SME steps; N=4000 is a non-power-of-two control for N=4096.
    constexpr Sizes<64, 256, 512, 1024, 2048, 4000, 4096> sme{};
    add<kHaveSme>("Sme1Tile", kNoSme, HPC_KERNEL(gemm_sme_1tile), sme, SmeSvl{});
    add<kHaveSme>("SmeTiles", kNoSme, HPC_KERNEL(gemm_sme_tiles), sme, SmeSvl{});
    add<kHaveSme>("SmeBlocked", kNoSme, HPC_KERNEL(gemm_sme_blocked), sme, SmeSvl{});
    add<kHaveSme>("Sme", kNoSme, HPC_KERNEL(gemm_sme), sme, SmeSvl{});

    // Accelerate may run on several threads, and Google Benchmark's default
    // CPU time counts only the calling thread, so it is timed by wall clock;
    // VECLIB_MAXIMUM_THREADS=1 restricts it to one core. KleidiAI is
    // single-threaded and f32-only; its packing is timed, as gemm_sme's is.
    constexpr Sizes<64, 256, 512, 1024, 2048, 4096> lib{};
    add<kHaveAmx>("Amx", kNoAmx, HPC_KERNEL(gemm_amx), lib, NoCounters{}, true);
    add<kHaveKleidiAI, false>("KleidiAI", kNoKleidiAI, HPC_KERNEL(gemm_kleidiai), lib);

    add_prefetch<2>();
    add_prefetch<4>();
    add_prefetch<8>();
    add_prefetch<16>();

    add_bstride<float>();
    add_bstride<double>();
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
