/**
 * @file bench_gemm_cuda.cpp
 * @brief Google Benchmark driver for the CUDA GEMM kernels (gemm/cuda.hpp).
 *
 *   ./build/benchmarks/bench_gemm_cuda --benchmark_filter=<regex>
 *
 * Every benchmark, <Kernel>/<f64|f32>/N=<N>, times the GEMM alone, as the
 * CPU benchmarks do: A, B and C are allocated and filled on the GPU before
 * timing, and each iteration is one call of the kernel's device entry point
 * (gemm_cuda_*_device). The time is GPU time, from CUDA events recorded
 * before and after the launch (gemm_cuda_time_ms), reported to Google
 * Benchmark as manual time. One untimed call precedes the loop, so one-off
 * costs such as creating the cuBLAS handle are not counted.
 *
 * CudaWmma and CudaMmaLdmatrix take fp32 inputs and convert each tile to
 * fp16 inside the kernel. CudaWmmaPipelined and CudaCublasFp16 take fp16
 * inputs, converted once before timing.
 *
 * The "precision" counter is the bits of the multiply inputs: 16 for the
 * Tensor Core kernels (fp16 in, fp32 accumulate), 32 for TF32 cuBLAS (its
 * inputs are fp32, rounded to TF32 by the Tensor Cores), so their GFLOP/s is
 * not like-for-like with the plain fp32 rows.
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
#include <memory>
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

/// Square N x N inputs on the GPU: T, or fp16 when Fp16 (the kernel then
/// receives void* buffers of halves). C is always T.
struct DeviceFree {
    void operator()(void* p) const noexcept { gemm_cuda_free(p); }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;

template <typename T, bool Fp16>
struct DeviceProblem {
    DevicePtr A, B, C;

    explicit DeviceProblem(std::size_t n) {
        hpc::Matrix<T> hA(n, n), hB(n, n);
        fill_random(hA, 1);
        fill_random(hB, 2);
        const std::size_t bytes = n * n * sizeof(T);
        A.reset(gemm_cuda_malloc(bytes));
        B.reset(gemm_cuda_malloc(bytes));
        C.reset(gemm_cuda_malloc(bytes));
        gemm_cuda_memcpy_h2d(A.get(), hA.data(), bytes);
        gemm_cuda_memcpy_h2d(B.get(), hB.data(), bytes);
        if constexpr (Fp16) {
            static_assert(sizeof(T) == 4, "fp16 inputs are converted from fp32");
            for (DevicePtr* buf : {&A, &B}) {
                DevicePtr half(gemm_cuda_malloc(n * n * 2));
                gemm_cuda_convert_f32_to_f16_device(static_cast<const float*>(buf->get()), half.get(),
                                                    static_cast<int>(n * n));
                gemm_cuda_device_synchronize();
                *buf = std::move(half);
            }
        }
        gemm_cuda_device_synchronize();
    }
    DeviceProblem(const DeviceProblem&)            = delete;
    DeviceProblem& operator=(const DeviceProblem&) = delete;
};

/// Register <name>/<type>/N=<n> for each size: launch(A, B, C, n) on device
/// buffers, timed with GPU events.
template <typename T, bool Fp16 = false, typename Launch>
void add(const std::string& name, Launch launch, Needs needs, int precision_bits,
         std::initializer_list<std::size_t> sizes) {
    for (const std::size_t n : sizes) {
        benchmark::RegisterBenchmark(
            name + "/" + type_name<T>() + "/N=" + std::to_string(n),
            [=](benchmark::State& state) {
                if (skip(state, needs))
                    return;
                DeviceProblem<T, Fp16> p(n);
                const int N = static_cast<int>(n);
                auto call   = [&] { launch(p.A.get(), p.B.get(), static_cast<T*>(p.C.get()), N); };
                gemm_cuda_time_ms(call);  // warm-up, untimed
                for (auto _ : state)
                    state.SetIterationTime(gemm_cuda_time_ms(call) / 1e3);
                const double flops        = 2.0 * double(n) * double(n) * double(n);
                state.counters["GFLOP/s"] = benchmark::Counter(
                    flops, benchmark::Counter::kIsIterationInvariantRate,
                    benchmark::Counter::OneK::kIs1000);
                state.counters["precision"] = precision_bits;
            })
            ->UseManualTime()
            ->Unit(benchmark::kMillisecond);
    }
}

constexpr std::initializer_list<std::size_t> kSizesF32 = {256, 1024, 4096, 8192, 16384};
constexpr std::initializer_list<std::size_t> kSizesF64 = {256, 1024, 4096, 8192};

/// An FMA level (Levels 0-4) in both precisions.
#define HPC_ADD_FMA(name, fn)                                                                   \
    add<double>(name, [](void* a, void* b, double* c, int n) {                                  \
        fn<double>(static_cast<const double*>(a), static_cast<const double*>(b), c, n, n, n); }, \
        Needs::kDevice, 64, kSizesF64);                                                         \
    add<float>(name, [](void* a, void* b, float* c, int n) {                                    \
        fn<float>(static_cast<const float*>(a), static_cast<const float*>(b), c, n, n, n); },    \
        Needs::kDevice, 32, kSizesF32)

void register_all() {
    // clang-format off
    HPC_ADD_FMA("CudaNaive",      gemm_cuda_naive_device);
    HPC_ADD_FMA("CudaBlocked",    gemm_cuda_blocked_device);
    HPC_ADD_FMA("CudaRegTile",    gemm_cuda_reg_tile_device);
    HPC_ADD_FMA("CudaDoubleBuf",  gemm_cuda_double_buf_device);
    HPC_ADD_FMA("CudaVectorized", gemm_cuda_vectorized_device);

    add<float>("CudaWmma", [](void* a, void* b, float* c, int n) {
        gemm_cuda_wmma_device(static_cast<const float*>(a), static_cast<const float*>(b), c, n, n, n); },
        Needs::kTensorCores, 16, kSizesF32);
    add<float>("CudaMmaLdmatrix", [](void* a, void* b, float* c, int n) {
        gemm_cuda_mma_ldmatrix_device(static_cast<const float*>(a), static_cast<const float*>(b), c, n, n, n); },
        Needs::kAmpere, 16, kSizesF32);
    // No fallback on the device path: every size here is a multiple of 128.
    add<float, true>("CudaWmmaPipelined", [](void* a, void* b, float* c, int n) {
        gemm_cuda_wmma_pipelined_device(a, b, c, n, n, n); },
        Needs::kTensorCores, 16, kSizesF32);

    HPC_ADD_FMA("CudaCublas",     gemm_cuda_cublas_device);
    add<float>("CudaCublasTf32", [](void* a, void* b, float* c, int n) {
        gemm_cuda_cublas_tf32_device(static_cast<const float*>(a), static_cast<const float*>(b), c, n, n, n); },
        Needs::kAmpere, 32, kSizesF32);
    add<float, true>("CudaCublasFp16", [](void* a, void* b, float* c, int n) {
        gemm_cuda_cublas_fp16_device(a, b, c, n, n, n); },
        Needs::kTensorCores, 16, kSizesF32);
    // clang-format on
}

#undef HPC_ADD_FMA

}  // namespace

int main(int argc, char** argv) {
    register_all();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
