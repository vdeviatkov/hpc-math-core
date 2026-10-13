/**
 * @file bench_gemm_cuda.cpp
 * @brief Google Benchmark driver for the CUDA GEMM kernels (gemm/cuda.hpp).
 *
 *   ./build/benchmarks/bench_gemm_cuda --benchmark_filter=<regex>
 *
 * Two kinds of benchmark:
 *
 *   <Kernel>/<f64|f32>/N=<N>             end to end: every iteration runs
 *       cudaMalloc, host-to-device copies, the kernel and the copy back, as
 *       the Matrix-based entry points do. N = 64 … 4096; cuBLAS and Level 7
 *       also at 8192 and 16384.
 *   <Kernel>ComputeOnly/f32/N=<N>        only the GEMM call, on device buffers
 *       filled once outside the timed loop (fp16 kernels get their inputs
 *       converted there too). N = 4096, 8192, 16384. At large N the
 *       end-to-end rows mostly measure PCIe transfers; these do not.
 *
 * The "precision" counter is the bits of the multiply inputs: 16 for the
 * Tensor Core kernels (fp16 in, fp32 accumulate), so their GFLOP/s is not
 * like-for-like with the fp32 rows.
 *
 * Every benchmark reports SKIPPED without a CUDA device (the stub build has
 * none), and the Tensor Core / TF32 ones also on GPUs older than they need.
 * This file must compile without a CUDA toolkit, so it uses only the
 * host-side helpers declared in gemm/cuda.hpp, not <cuda_runtime.h>.
 */

#include "gemm/cuda.hpp"
#include "hpc/matrix.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <string>

namespace {

using namespace hpc::gemm;

template <typename T>
void fill_random(hpc::Matrix<T>& M, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(T{-1}, T{1});
    for (std::size_t i = 0; i < M.size(); ++i)
        M.data()[i] = dist(rng);
}

template <typename T>
const char* type_name() {
    return sizeof(T) == 4 ? "f32" : "f64";
}

/// Hardware a benchmark needs beyond a CUDA device.
enum class Needs { kDevice, kTensorCores, kAmpere };

/// Report SKIPPED and return true when the hardware is missing.
bool skip(benchmark::State& state, Needs needs) {
    if (cuda_device_count() == 0) {
        state.SkipWithMessage("No CUDA device available");
        return true;
    }
    if (needs == Needs::kTensorCores && !cuda_has_tensor_cores()) {
        state.SkipWithMessage("needs Tensor Cores (sm_70+)");
        return true;
    }
    if (needs == Needs::kAmpere && !cuda_has_ampere()) {
        state.SkipWithMessage("needs sm_80+");
        return true;
    }
    return false;
}

void add_counters(benchmark::State& state, std::size_t N, int precision_bits) {
    const double n            = static_cast<double>(N);
    state.counters["GFLOP/s"] = benchmark::Counter(2.0 * n * n * n,
                                                   benchmark::Counter::kIsIterationInvariantRate,
                                                   benchmark::Counter::OneK::kIs1000);
    state.counters["precision"] = precision_bits;
}

/// End to end: kernel(A, B, C) on host matrices, transfers included.
template <typename T, typename Kernel>
void add_end_to_end(const std::string& name, Kernel kernel, Needs needs, int precision_bits,
                    std::initializer_list<std::size_t> sizes) {
    for (const std::size_t n : sizes) {
        benchmark::RegisterBenchmark(
            name + "/" + type_name<T>() + "/N=" + std::to_string(n),
            [=](benchmark::State& state) {
                if (skip(state, needs))
                    return;
                hpc::Matrix<T> A(n, n), B(n, n), C(n, n);
                fill_random(A, 1);
                fill_random(B, 2);
                for (auto _ : state) {
                    kernel(A, B, C);
                    benchmark::DoNotOptimize(C.data());
                    benchmark::ClobberMemory();
                }
                add_counters(state, n, precision_bits);
            })
            ->Unit(n > 4096 ? benchmark::kMillisecond : benchmark::kMicrosecond);
    }
}

/// Every FMA level runs in f64 and f32; the Tensor Core levels in f32 only.
constexpr std::initializer_list<std::size_t> kSizes = {64, 256, 512, 1024, 4096};

template <typename Kernel>
void add_both_types(const std::string& name, Kernel kernel) {
    add_end_to_end<double>(name, kernel, Needs::kDevice, 64, kSizes);
    add_end_to_end<float>(name, kernel, Needs::kDevice, 32, kSizes);
}

/// Compute-only: kernel(dA, dB, dC, N) on device buffers staged once. With
/// Fp16 the inputs are converted to fp16 on the device before timing.
template <bool Fp16, typename Kernel>
void add_compute_only(const std::string& name, Kernel kernel, Needs needs) {
    for (const std::size_t n : {4096, 8192, 16384}) {
        benchmark::RegisterBenchmark(
            name + "ComputeOnly/f32/N=" + std::to_string(n),
            [=](benchmark::State& state) {
                if (skip(state, needs))
                    return;
                const std::size_t bytes32 = n * n * sizeof(float);
                const std::size_t bytes16 = n * n * 2;  // __half
                hpc::Matrix<float> A(n, n), B(n, n);
                fill_random(A, 1);
                fill_random(B, 2);
                void* dA = gemm_cuda_malloc(bytes32);
                void* dB = gemm_cuda_malloc(bytes32);
                auto* dC = static_cast<float*>(gemm_cuda_malloc(bytes32));
                gemm_cuda_memcpy_h2d(dA, A.data(), bytes32);
                gemm_cuda_memcpy_h2d(dB, B.data(), bytes32);
                if constexpr (Fp16) {
                    void* dA16 = gemm_cuda_malloc(bytes16);
                    void* dB16 = gemm_cuda_malloc(bytes16);
                    gemm_cuda_convert_f32_to_f16_device(static_cast<float*>(dA), dA16, int(n * n));
                    gemm_cuda_convert_f32_to_f16_device(static_cast<float*>(dB), dB16, int(n * n));
                    gemm_cuda_free(dA);
                    gemm_cuda_free(dB);
                    dA = dA16;
                    dB = dB16;
                }
                gemm_cuda_device_synchronize();
                for (auto _ : state) {
                    kernel(dA, dB, dC, int(n));
                    gemm_cuda_device_synchronize();
                    benchmark::ClobberMemory();
                }
                gemm_cuda_free(dA);
                gemm_cuda_free(dB);
                gemm_cuda_free(dC);
                add_counters(state, n, Fp16 ? 16 : 32);
            })
            ->Unit(benchmark::kMillisecond);
    }
}

void register_all() {
    // clang-format off
    add_both_types("CudaNaive",      [](auto& A, auto& B, auto& C) { gemm_cuda_naive(A, B, C); });
    add_both_types("CudaBlocked",    [](auto& A, auto& B, auto& C) { gemm_cuda_blocked(A, B, C); });
    add_both_types("CudaRegTile",    [](auto& A, auto& B, auto& C) { gemm_cuda_reg_tile(A, B, C); });
    add_both_types("CudaDoubleBuf",  [](auto& A, auto& B, auto& C) { gemm_cuda_double_buf(A, B, C); });
    add_both_types("CudaVectorized", [](auto& A, auto& B, auto& C) { gemm_cuda_vectorized(A, B, C); });
    add_end_to_end<float>("CudaWmma", gemm_cuda_wmma, Needs::kTensorCores, 16, kSizes);
    add_end_to_end<float>("CudaMmaLdmatrix", gemm_cuda_mma_ldmatrix, Needs::kAmpere, 16, kSizes);
    // Level 7 needs M, N multiples of 128: N=64 runs its fallback, gemm_cuda_wmma.
    add_end_to_end<float>("CudaWmmaPipelined", gemm_cuda_wmma_pipelined, Needs::kTensorCores, 16,
                          {64, 256, 512, 1024, 4096, 8192, 16384});

    add_end_to_end<double>("CudaCublas", gemm_cuda_cublas<double>, Needs::kDevice, 64, kSizes);
    add_end_to_end<float>("CudaCublas", gemm_cuda_cublas<float>, Needs::kDevice, 32,
                          {64, 256, 512, 1024, 4096, 8192, 16384});
    add_end_to_end<float>("CudaCublasTf32", gemm_cuda_cublas_tf32, Needs::kAmpere, 32,
                          {64, 256, 512, 1024, 4096, 8192, 16384});

    add_compute_only<false>("CudaCublas", [](void* a, void* b, float* c, int n) {
        gemm_cuda_cublas_device_f32(static_cast<float*>(a), static_cast<float*>(b), c, n, n, n); },
        Needs::kDevice);
    add_compute_only<false>("CudaCublasTf32", [](void* a, void* b, float* c, int n) {
        gemm_cuda_cublas_tf32_device(static_cast<float*>(a), static_cast<float*>(b), c, n, n, n); },
        Needs::kAmpere);
    add_compute_only<true>("CudaCublasFp16", [](void* a, void* b, float* c, int n) {
        gemm_cuda_cublas_fp16_device(a, b, c, n, n, n); }, Needs::kTensorCores);
    // No fallback on the device path: every N here is a multiple of 128.
    add_compute_only<true>("CudaWmmaPipelined", [](void* a, void* b, float* c, int n) {
        gemm_cuda_wmma_pipelined_device(a, b, c, n, n, n); }, Needs::kTensorCores);
    // clang-format on
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
