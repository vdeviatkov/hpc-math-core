#pragma once
/**
 * @file metal.hpp
 * @brief Host interface to the Metal GEMM kernels (src/metal/gemm.metal,
 *        driven from src/metal/gemm_metal.mm).
 *
 * Six levels on the Apple GPU, each adding one technique, all f32 in and
 * out (Apple GPUs have no FP64):
 *
 *   0  kNaive      one thread per C(i,j), device memory only
 *   1  kTiled      16x16 threadgroup-memory tiles, one output per thread
 *   2  kRegTile    64x64 block tile, 8x4 outputs per thread, float4 loads,
 *                  A transposed in threadgroup memory
 *   3  kSimd       simdgroup_float8x8 multiply-accumulate, fragments loaded
 *                  straight from device memory; 2x2 simdgroups of 32x32
 *   4  kSimdTg     Level 3 with 64x16 / 16x64 tiles of A and B staged in
 *                  padded threadgroup memory
 *   5  kSimdTgF16  Level 4 with fp16 A and B (converted once, before timing,
 *                  as CudaWmmaPipelined does), fp32 accumulate
 *
 * kMps and kMpsF16 are the vendor reference, MPSMatrixMultiplication (f32,
 * and fp16 inputs with an fp32 result), as cuBLAS is for the CUDA kernels.
 *
 * Levels 2-5 need M and N multiples of 64 and K of 16; otherwise Level 1
 * runs instead (it handles any shape). On M1-M4 GPUs simdgroup matrices run
 * on the ordinary FP32 ALUs; there is no separate matrix unit, so fp16 does
 * not raise the arithmetic peak (see metal_fma_peak_gflops).
 *
 * Shaders are compiled from source when the first MetalGemm is created
 * (newLibraryWithSource), so no offline Metal toolchain is needed. Without
 * Metal (non-Apple builds) src/metal/gemm_metal_stub.cpp provides the same
 * API: metal_available() returns false and everything else throws.
 */

#include "hpc/matrix.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace hpc::gemm {

/// True when a Metal GPU is present (always false in the stub build).
bool metal_available() noexcept;

/// Name of the default Metal device, e.g. "Apple M4 Max".
std::string metal_device_name();

enum class MetalKernel { kNaive, kTiled, kRegTile, kSimd, kSimdTg, kSimdTgF16, kMps, kMpsF16 };

/// Tile variants of Levels 2-5 compiled into the library (gemm_reg_tile_*,
/// gemm_simd_*, gemm_simd_tg_*), for MetalGemm::run_variant_ms sweeps.
std::vector<std::string> metal_kernel_variants();

/// A, B and C resident in GPU-visible memory: what the benchmarks time, so
/// GPU numbers measure the multiplication alone, as for CUDA. A copy of A
/// and B in fp16 is kept for kSimdTgF16 and kMpsF16.
class MetalGemm {
public:
    MetalGemm(std::size_t M, std::size_t K, std::size_t N);
    ~MetalGemm();
    MetalGemm(const MetalGemm&)            = delete;
    MetalGemm& operator=(const MetalGemm&) = delete;

    /// Copy row-major A (M x K) and B (K x N) in, and their fp16 copies.
    void set_inputs(const float* A, const float* B);
    /// Copy row-major C (M x N) out.
    void get_output(float* C) const;

    /// GPU time per GEMM, in milliseconds: `reps` back-to-back GEMMs are
    /// encoded into one command buffer and its GPUStartTime..GPUEndTime span
    /// is divided by reps. Waits for the GPU.
    double run_ms(MetalKernel kernel, int reps = 1);
    /// The same for a tile variant by function name (metal_kernel_variants).
    double run_variant_ms(const std::string& function, int reps = 1);

    /// The kernel that actually runs for `kernel` at this shape (fallback).
    MetalKernel effective(MetalKernel kernel) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// C = A·B on the GPU, copying in and out (convenience for tests).
void gemm_metal(MetalKernel kernel, const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);

/// Measured FMA throughput of the GPU in GFLOP/s, f32 or f16: independent
/// 4-wide FMA chains on every thread of a large grid (gemm.metal, fma_peak).
double metal_fma_peak_gflops(bool f16);

}  // namespace hpc::gemm
