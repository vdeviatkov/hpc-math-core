#pragma once

/**
 * @file neon.hpp
 * @brief ARM NEON primitives for the generic SIMD kernels (simd.hpp).
 *
 *   gemm_simd_reordered<Q>(A, B, C);
 *   gemm_simd_blocked<Q, R, V, TileM, TileK, TileN>(A, B, C);
 *   gemm_simd_packed<Q, R, V, Mc, Kc, Nc>(A, B, C);
 *
 * AArch64 has 32 × 128-bit Q registers (4 f32 / 2 f64). Apple M4 P-core
 * (measured with fma_peak, docs/experiments/cpu-simd): up to 4 FMAs per
 * cycle, ~3-cycle latency.
 *   f32: 4 pipes × 4 lanes × 2 FLOP = 32 FLOP/cycle → ~143 GFLOP/s at 4.49 GHz
 *   f64: 4 pipes × 2 lanes × 2 FLOP = 16 FLOP/cycle →  ~72 GFLOP/s
 * NEON is half the width of AVX2, but twice the FMA pipes give the same
 * FLOP/cycle.
 *
 * HPC_HAS_NEON (hpc/isa.hpp) requires AArch64: Q<double> uses f64 lanes
 * and vfmaq_f64, which 32-bit ARMv7 NEON lacks. Q<T> is declared on every
 * target and defined only where HPC_HAS_NEON is 1.
 */

#include "hpc/isa.hpp"

#if HPC_HAS_NEON
    #include <arm_neon.h>
#endif

namespace hpc::gemm {

template <typename T> struct Q;

#if HPC_HAS_NEON

// load_a puts A(i, k) in lane 0 of a 64-bit D register (S) with a plain LDR,
// and fma multiplies a B vector by that lane (FMLA by element). Broadcasting
// A with LD1R (load and replicate), which Clang emits for vdupq + vfmaq,
// held tiles with many rows back on M4: with LDR the 6 × 4 tile is 23–30%
// faster in the full multiplication (docs/kernels/neon.md).
template <> struct Q<float> {
    using V = float32x4_t;
    using S = float32x2_t;
    static V load(const float* p) { return vld1q_f32(p); }
    static void store(float* p, V v) { vst1q_f32(p, v); }
    static S load_a(const float* p) {
#if defined(__GNUC__)
        // From intrinsics (vld1_lane, vset_lane) Clang 17 folds the load and
        // the by-element FMA back into LD1R, so the LDR is inline assembly.
        S x;
        asm("ldr %s0, %1" : "=w"(x) : "m"(*p));
        return x;
#else
        return vld1_dup_f32(p);
#endif
    }
    static V fma(S a, V b, V acc) { return vfmaq_lane_f32(acc, b, a, 0); }
};
template <> struct Q<double> {
    using V = float64x2_t;
    using S = float64x1_t;
    static V load(const double* p) { return vld1q_f64(p); }
    static void store(double* p, V v) { vst1q_f64(p, v); }
    static S load_a(const double* p) { return vld1_f64(p); }  // LDR d
    static V fma(S a, V b, V acc) { return vfmaq_lane_f64(acc, b, a, 0); }
};

#endif  // HPC_HAS_NEON

}  // namespace hpc::gemm
