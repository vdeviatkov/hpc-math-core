/**
 * @file lowp_peak.cpp
 * @brief Latency and throughput of the low-precision multiply-accumulate
 *        instructions, next to the FP32 FMA of the same unit.
 *
 *   ./build/benchmarks/lowp_peak > lowp_peak.csv
 *
 * The method of fma_peak.cpp: C independent accumulators, each updated in
 * turn by the instruction on constant inputs, no memory traffic. One
 * accumulator gives the latency; enough of them give the issue rate.
 * Every row reports operations per instruction (a multiply-add counts 2,
 * for integers too), so the GOP/s column compares formats directly.
 *
 *   NEON (128-bit)   FMLA .4s (fp32), FMLA .8h (fp16 → fp16), FMLAL (fp16 →
 *                    fp32, lower half), BFDOT, BFMMLA (bf16 → fp32), SDOT,
 *                    SMMLA (int8 → int32)
 *   AVX-512          VFMADD231PS (fp32), VDPBF16PS (bf16 → fp32),
 *                    VPDPBUSD (u8 × s8 → int32)
 *   SME2 (streaming) FMOPA fp32, widening FMOPA fp16 / bf16 → fp32,
 *                    SMOPA int8 → int32, into 1 and 4 ZA tiles
 *
 * Only the instructions the build targets are compiled. Best of 5 runs;
 * cycles are of the core clock measured at the start (core_clock.hpp).
 */

#include "core_clock.hpp"
#include "hpc/isa.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

#if HPC_HAS_NEON
    #include <arm_neon.h>
#endif
#if HPC_HAS_SME
    #include <arm_sme.h>
#endif
#if defined(__x86_64__)
    #include <immintrin.h>
#endif

namespace {

double g_ghz = 0;

template <typename V>
inline void opaque(V& v) {
#if defined(__aarch64__)
    asm volatile("" : "+w"(v));
#else
    asm volatile("" : "+v"(v));
#endif
}

constexpr long kIters = 20'000'000;

#define HPC_ACC(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)

/// Seconds for kIters rounds of Op on each of C accumulators (best of 5).
template <typename Op, int C>
double run() {
    static_assert(C >= 1 && C <= 16);
    using V = typename Op::Acc;
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        typename Op::In a = Op::a(), b = Op::b();
        opaque(a);
        opaque(b);
#define HPC_DECL(n) V acc##n = Op::init(n);
        HPC_ACC(HPC_DECL)
#undef HPC_DECL
        const auto t0 = std::chrono::steady_clock::now();
        for (long i = 0; i < kIters; ++i) {
#define HPC_STEP(n)                         \
    if constexpr (C > n) {                  \
        acc##n = Op::step(acc##n, a, b);    \
        opaque(acc##n);                     \
    }
            HPC_ACC(HPC_STEP)
#undef HPC_STEP
        }
        const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
        best = std::min(best, dt.count());
#define HPC_SINK(n) opaque(acc##n);
        HPC_ACC(HPC_SINK)
#undef HPC_SINK
    }
    return best;
}

[[maybe_unused]] void row(const char* isa, const char* instr, int ops, int accumulators, double seconds, double instrs) {
    const double cycles = seconds * g_ghz * 1e9 / instrs;
    std::printf("%s,%s,%d,%d,%.3f,%.2f,%.1f\n", isa, instr, ops, accumulators, cycles, 1.0 / cycles,
                instrs * ops / seconds / 1e9);
}

template <typename Op, int... Cs>
void sweep() {
    ((row(Op::isa, Op::name, Op::ops, Cs, run<Op, Cs>(), double(kIters) * Cs)), ...);
}

#if HPC_HAS_NEON
struct NeonFmlaF32 {
    static constexpr const char *isa = "neon", *name = "fmla_4s_f32";
    static constexpr int ops = 8;
    using Acc = float32x4_t;
    using In  = float32x4_t;
    static In a() { return vdupq_n_f32(0.999f); }
    static In b() { return vdupq_n_f32(1e-6f); }
    static Acc init(int n) { return vdupq_n_f32(float(n)); }
    static Acc step(Acc c, In x, In y) { return vfmaq_f32(c, x, y); }
};
#endif

#if HPC_HAS_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_FML)
struct NeonFmlaF16 {
    static constexpr const char *isa = "neon", *name = "fmla_8h_f16";
    static constexpr int ops = 16;
    using Acc = float16x8_t;
    using In  = float16x8_t;
    static In a() { return vdupq_n_f16(0.999f); }
    static In b() { return vdupq_n_f16(0.001f); }
    static Acc init(int n) { return vdupq_n_f16(float(n)); }
    static Acc step(Acc c, In x, In y) { return vfmaq_f16(c, x, y); }
};
struct NeonFmlal {
    static constexpr const char *isa = "neon", *name = "fmlal_f16_f32";
    static constexpr int ops = 8;
    using Acc = float32x4_t;
    using In  = float16x8_t;
    static In a() { return vdupq_n_f16(0.999f); }
    static In b() { return vdupq_n_f16(0.001f); }
    static Acc init(int n) { return vdupq_n_f32(float(n)); }
    static Acc step(Acc c, In x, In y) { return vfmlalq_low_f16(c, x, y); }
};
#endif

#if HPC_HAS_NEON && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
struct NeonBfdot {
    static constexpr const char *isa = "neon", *name = "bfdot";
    static constexpr int ops = 16;
    using Acc = float32x4_t;
    using In  = bfloat16x8_t;
    static In a() { return vreinterpretq_bf16_u16(vdupq_n_u16(0x3f80)); }
    static In b() { return vreinterpretq_bf16_u16(vdupq_n_u16(0x3380)); }
    static Acc init(int n) { return vdupq_n_f32(float(n)); }
    static Acc step(Acc c, In x, In y) { return vbfdotq_f32(c, x, y); }
};
struct NeonBfmmla {
    static constexpr const char *isa = "neon", *name = "bfmmla";
    static constexpr int ops = 32;
    using Acc = float32x4_t;
    using In  = bfloat16x8_t;
    static In a() { return NeonBfdot::a(); }
    static In b() { return NeonBfdot::b(); }
    static Acc init(int n) { return vdupq_n_f32(float(n)); }
    static Acc step(Acc c, In x, In y) { return vbfmmlaq_f32(c, x, y); }
};
#endif

#if HPC_HAS_NEON && defined(__ARM_FEATURE_DOTPROD) && defined(__ARM_FEATURE_MATMUL_INT8)
struct NeonSdot {
    static constexpr const char *isa = "neon", *name = "sdot";
    static constexpr int ops = 32;
    using Acc = int32x4_t;
    using In  = int8x16_t;
    static In a() { return vdupq_n_s8(1); }
    static In b() { return vdupq_n_s8(1); }
    static Acc init(int n) { return vdupq_n_s32(n); }
    static Acc step(Acc c, In x, In y) { return vdotq_s32(c, x, y); }
};
struct NeonSmmla {
    static constexpr const char *isa = "neon", *name = "smmla";
    static constexpr int ops = 64;
    using Acc = int32x4_t;
    using In  = int8x16_t;
    static In a() { return vdupq_n_s8(1); }
    static In b() { return vdupq_n_s8(1); }
    static Acc init(int n) { return vdupq_n_s32(n); }
    static Acc step(Acc c, In x, In y) { return vmmlaq_s32(c, x, y); }
};
#endif

#if defined(__AVX512F__)
struct Avx512Fma {
    static constexpr const char *isa = "avx512", *name = "vfmadd231ps";
    static constexpr int ops = 32;
    using Acc = __m512;
    using In  = __m512;
    static In a() { return _mm512_set1_ps(0.999f); }
    static In b() { return _mm512_set1_ps(1e-6f); }
    static Acc init(int n) { return _mm512_set1_ps(float(n)); }
    static Acc step(Acc c, In x, In y) { return _mm512_fmadd_ps(x, y, c); }
};
#endif
#if defined(__AVX512BF16__)
struct Avx512Dpbf16 {
    static constexpr const char *isa = "avx512", *name = "vdpbf16ps";
    static constexpr int ops = 64;
    using Acc = __m512;
    using In  = __m512bh;
    static In a() { return (__m512bh)_mm512_set1_epi16(0x3f80); }
    static In b() { return (__m512bh)_mm512_set1_epi16(0x3380); }
    static Acc init(int n) { return _mm512_set1_ps(float(n)); }
    static Acc step(Acc c, In x, In y) { return _mm512_dpbf16_ps(c, x, y); }
};
#endif
#if defined(__AVX512VNNI__)
struct Avx512Dpbusd {
    static constexpr const char *isa = "avx512", *name = "vpdpbusd";
    static constexpr int ops = 128;
    using Acc = __m512i;
    using In  = __m512i;
    static In a() { return _mm512_set1_epi8(1); }
    static In b() { return _mm512_set1_epi8(1); }
    static Acc init(int n) { return _mm512_set1_epi32(n); }
    static Acc step(Acc c, In x, In y) { return _mm512_dpbusd_epi32(c, x, y); }
};
#endif

#if HPC_HAS_SME
// Outer products into ZA tiles 0 … T-1 in turn, as in sme_fmopa.cpp. The
// widening forms take vectors of 32 fp16 / bf16 or 64 int8 and still produce
// a 16 × 16 tile of 32-bit sums: 2 (16-bit) or 4 (int8) products per element.
constexpr long kSmeIters = 4'000'000;

template <int... Tiles>
__arm_locally_streaming __arm_new("za") void mopa_f32(long iters, float* out) {
    const svbool_t pg   = svptrue_b32();
    const svfloat32_t a = svdup_f32(1.0f), b = svdup_f32(1e-9f);
    for (long i = 0; i < iters; ++i)
        (svmopa_za32_f32_m(Tiles, pg, pg, a, b), ...);
    svst1_hor_za32(0, 0, pg, out);
}
template <int... Tiles>
__arm_locally_streaming __arm_new("za") void mopa_f16(long iters, float* out) {
    const svbool_t pg   = svptrue_b16();
    const svfloat16_t a = svdup_f16(1.0f), b = svdup_f16(0.001f);
    for (long i = 0; i < iters; ++i)
        (svmopa_za32_f16_m(Tiles, pg, pg, a, b), ...);
    svst1_hor_za32(0, 0, svptrue_b32(), out);
}
template <int... Tiles>
__arm_locally_streaming __arm_new("za") void mopa_bf16(long iters, float* out) {
    const svbool_t pg    = svptrue_b16();
    const svbfloat16_t a = svreinterpret_bf16_u16(svdup_u16(0x3f80));
    const svbfloat16_t b = svreinterpret_bf16_u16(svdup_u16(0x3380));
    for (long i = 0; i < iters; ++i)
        (svmopa_za32_bf16_m(Tiles, pg, pg, a, b), ...);
    svst1_hor_za32(0, 0, svptrue_b32(), out);
}
template <int... Tiles>
__arm_locally_streaming __arm_new("za") void mopa_s8(long iters, float* out) {
    const svbool_t pg = svptrue_b8();
    const svint8_t a = svdup_s8(1), b = svdup_s8(1);
    for (long i = 0; i < iters; ++i)
        (svmopa_za32_s8_m(Tiles, pg, pg, a, b), ...);
    svst1_hor_za32(0, 0, svptrue_b32(), out);
}

template <typename Fn>
void sme_row(const char* name, int ops, int tiles, Fn fn) {
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    row("sme", name, ops, tiles, best, double(kSmeIters) * tiles);
}
#endif

}  // namespace

int main() {
    g_ghz = hpc::experiments::core_ghz();
    std::fprintf(stderr, "core clock: %.3f GHz\n", g_ghz);
    std::printf("isa,instruction,ops_per_instruction,accumulators,cycles_per_instruction,"
                "instructions_per_cycle,gops\n");
#if HPC_HAS_NEON
    sweep<NeonFmlaF32, 1, 8, 16>();
#endif
#if HPC_HAS_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_FML)
    sweep<NeonFmlaF16, 1, 8, 16>();
    sweep<NeonFmlal, 1, 8, 16>();
#endif
#if HPC_HAS_NEON && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    sweep<NeonBfdot, 1, 8, 16>();
    sweep<NeonBfmmla, 1, 8, 16>();
#endif
#if HPC_HAS_NEON && defined(__ARM_FEATURE_DOTPROD) && defined(__ARM_FEATURE_MATMUL_INT8)
    sweep<NeonSdot, 1, 8, 16>();
    sweep<NeonSmmla, 1, 8, 16>();
#endif
#if defined(__AVX512F__)
    sweep<Avx512Fma, 1, 8, 16>();
#endif
#if defined(__AVX512BF16__)
    sweep<Avx512Dpbf16, 1, 8, 16>();
#endif
#if defined(__AVX512VNNI__)
    sweep<Avx512Dpbusd, 1, 8, 16>();
#endif
#if HPC_HAS_SME
    float out[64];
    sme_row("fmopa_f32", 512, 1, [&] { mopa_f32<0>(kSmeIters, out); });
    sme_row("fmopa_f32", 512, 4, [&] { mopa_f32<0, 1, 2, 3>(kSmeIters, out); });
    sme_row("fmopa_f16_f32", 1024, 1, [&] { mopa_f16<0>(kSmeIters, out); });
    sme_row("fmopa_f16_f32", 1024, 4, [&] { mopa_f16<0, 1, 2, 3>(kSmeIters, out); });
    sme_row("bfmopa_bf16_f32", 1024, 1, [&] { mopa_bf16<0>(kSmeIters, out); });
    sme_row("bfmopa_bf16_f32", 1024, 4, [&] { mopa_bf16<0, 1, 2, 3>(kSmeIters, out); });
    sme_row("smopa_s8_s32", 2048, 1, [&] { mopa_s8<0>(kSmeIters, out); });
    sme_row("smopa_s8_s32", 2048, 4, [&] { mopa_s8<0, 1, 2, 3>(kSmeIters, out); });
#endif
    std::fprintf(stderr, "core clock after: %.3f GHz\n", hpc::experiments::core_ghz());
}
