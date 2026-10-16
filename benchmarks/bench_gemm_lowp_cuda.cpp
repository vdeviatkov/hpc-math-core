/**
 * @file bench_gemm_lowp_cuda.cpp
 * @brief Google Benchmark driver for the low-precision CUDA GEMMs
 *        (gemm/cuda_lowp.hpp) and their cuBLAS references.
 *
 *   ./build/benchmarks/bench_gemm_lowp_cuda --benchmark_filter=<regex>
 *
 * <Mma|Cublas>/<format>/N=<N> on square matrices. As in bench_gemm_cuda,
 * the inputs are generated in the format, copied to the GPU before timing,
 * and each iteration is one device-entry-point call timed with CUDA events
 * (GPU time, reported as manual time). B is stored transposed (N x K), the
 * layout both the kernel and cuBLASLt's FP8 path take. "GFLOP/s" is
 * 2·N³ / time for every format, INT8 included.
 *
 * "err" is the accuracy at the benchmarked size: after timing, 256 sampled
 * C(i,j) are recomputed on the host in double from the same inputs, and the
 * counter is max |C − C_ref| / Σ_k |A(i,k)·B(k,j)| over the samples (0 for
 * exact INT8 results). FP32 accumulation shows up as ~1e-7 or below, fp16
 * accumulation as ~1e-4 to 1e-3.
 *
 * Every benchmark reports SKIPPED without a CUDA device or when the device
 * lacks the format (sm_80+, FP8 sm_89+).
 */

#include "gemm/cuda.hpp"
#include "gemm/cuda_lowp.hpp"
#include "hpc/lowp.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace hpc::gemm;

struct DeviceFree {
    void operator()(void* p) const noexcept { gemm_cuda_free(p); }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;

int element_bytes(CudaLowp f) { return f == CudaLowp::int8 || f == CudaLowp::fp8_e4m3 ? 1 : 2; }
int output_bytes(CudaLowp f) { return f == CudaLowp::fp16_acc16 ? 2 : 4; }

/// Element i of a buffer of the format, as a float.
float value(CudaLowp f, const unsigned char* bytes, std::size_t i) {
    switch (f) {
        case CudaLowp::bf16: {
            hpc::bf16 v;
            std::memcpy(&v.bits, bytes + 2 * i, 2);
            return hpc::to_float(v);
        }
        case CudaLowp::fp16:
        case CudaLowp::fp16_acc16: {
            hpc::fp16 v;
            std::memcpy(&v.bits, bytes + 2 * i, 2);
            return hpc::to_float(v);
        }
        case CudaLowp::int8: return static_cast<float>(static_cast<std::int8_t>(bytes[i]));
        case CudaLowp::fp8_e4m3: return hpc::to_float(hpc::fp8_e4m3{bytes[i]});
    }
    return 0;
}

/// Largest sampled |C − C_ref| / Σ|ab| (see the file comment).
double sampled_error(CudaLowp f, const std::vector<unsigned char>& a, const std::vector<unsigned char>& bt,
                     const void* dC, std::size_t n) {
    std::vector<unsigned char> c(n * n * output_bytes(f));
    gemm_cuda_memcpy_d2h(c.data(), dC, c.size());
    std::mt19937 rng(3);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    double worst = 0;
    for (int s = 0; s < 256; ++s) {
        const std::size_t i = pick(rng), j = pick(rng);
        double ref = 0, abs_sum = 0;
        for (std::size_t k = 0; k < n; ++k) {
            const double p = double(value(f, a.data(), i * n + k)) * double(value(f, bt.data(), j * n + k));
            ref += p;
            abs_sum += std::fabs(p);
        }
        double got;
        if (f == CudaLowp::fp16_acc16) {
            hpc::fp16 h;
            std::memcpy(&h.bits, &c[2 * (i * n + j)], 2);
            got = hpc::to_float(h);
        } else if (f == CudaLowp::int8) {
            std::int32_t v;
            std::memcpy(&v, &c[4 * (i * n + j)], 4);
            got = v;
        } else {
            float v;
            std::memcpy(&v, &c[4 * (i * n + j)], 4);
            got = v;
        }
        if (abs_sum > 0)
            worst = std::max(worst, std::fabs(got - ref) / abs_sum);
    }
    return worst;
}

/// n x n values of the format, uniform in [-1, 1) (INT8: [-128, 127]), as raw bytes.
std::vector<unsigned char> random_bytes(CudaLowp f, std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::uniform_int_distribution<int> di(-128, 127);
    std::vector<unsigned char> out(n * n * element_bytes(f));
    for (std::size_t i = 0; i < n * n; ++i) {
        switch (f) {
            case CudaLowp::bf16: {
                const auto v = hpc::to_bf16(d(rng)).bits;
                std::memcpy(&out[2 * i], &v, 2);
                break;
            }
            case CudaLowp::fp16:
            case CudaLowp::fp16_acc16: {
                const auto v = hpc::to_fp16(d(rng)).bits;
                std::memcpy(&out[2 * i], &v, 2);
                break;
            }
            case CudaLowp::int8: out[i] = static_cast<unsigned char>(di(rng)); break;
            case CudaLowp::fp8_e4m3: out[i] = hpc::to_fp8_e4m3(d(rng)).bits; break;
        }
    }
    return out;
}

void add(const char* format_name, CudaLowp f, bool cublas) {
    for (const std::size_t n : {1024, 2048, 4096, 8192, 16384}) {
        const std::string name = std::string(cublas ? "Cublas/" : "Mma/") + format_name + "/N=" + std::to_string(n);
        benchmark::RegisterBenchmark(name, [=](benchmark::State& state) {
            if (cuda_device_count() == 0 || !cuda_lowp_supported(f)) {
                state.SkipWithMessage("needs a CUDA device with this format (sm_80+, FP8 sm_89+)");
                return;
            }
            const auto a = random_bytes(f, n, 1), bt = random_bytes(f, n, 2);
            DevicePtr A(gemm_cuda_malloc(a.size())), Bt(gemm_cuda_malloc(bt.size()));
            DevicePtr C(gemm_cuda_malloc(n * n * output_bytes(f)));
            gemm_cuda_memcpy_h2d(A.get(), a.data(), a.size());
            gemm_cuda_memcpy_h2d(Bt.get(), bt.data(), bt.size());
            const int N = static_cast<int>(n);
            auto call   = [&] {
                if (cublas)
                    gemm_cuda_cublas_lowp_device(f, A.get(), Bt.get(), C.get(), N, N, N);
                else
                    gemm_cuda_lowp_device(f, A.get(), Bt.get(), C.get(), N, N, N);
            };
            gemm_cuda_time_ms(call);  // warm-up: handle creation, heuristics, smem attribute
            for (auto _ : state)
                state.SetIterationTime(gemm_cuda_time_ms(call) / 1e3);
            state.counters["err"] = sampled_error(f, a, bt, C.get(), n);
            state.counters["GFLOP/s"] = benchmark::Counter(2.0 * double(n) * double(n) * double(n),
                                                           benchmark::Counter::kIsIterationInvariantRate,
                                                           benchmark::Counter::OneK::kIs1000);
        })
            ->UseManualTime()
            ->Unit(benchmark::kMillisecond);
    }
}

void register_all() {
    for (const bool cublas : {false, true}) {
        add("bf16", CudaLowp::bf16, cublas);
        add("fp16", CudaLowp::fp16, cublas);
        add("fp16_acc16", CudaLowp::fp16_acc16, cublas);
        add("int8", CudaLowp::int8, cublas);
        add("fp8_e4m3", CudaLowp::fp8_e4m3, cublas);
    }
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
