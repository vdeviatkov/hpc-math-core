/**
 * @file cuda_peak.cu
 * @brief The GPU's measured limits: FMA and Tensor Core peaks, DRAM
 *        bandwidth, the clock it runs at, and its resources per SM.
 *
 *   ./build/benchmarks/cuda_peak > peak.csv
 *
 * Every rate comes from a kernel that does nothing else, timed with CUDA
 * events, best of 5:
 *
 *   fma_f32 / fma_f64  each thread runs kChains independent FMA chains on
 *                      registers; no memory traffic.
 *   mma_f16            each warp issues mma.sync.m16n8k16 (fp16 in, fp32
 *                      accumulate, the Tensor Core path of the GEMM kernels)
 *                      on kChains independent accumulators held in registers.
 *   mma_<in>_<acc>     the same for each low-precision mma.sync of
 *                      src/cuda/gemm_lowp.cu: bf16, fp16 with an fp16
 *                      accumulator, int8, FP8 e4m3 with fp32 and fp16
 *                      accumulators, and (built for sm_120a) the block-scaled
 *                      FP8 form. Inputs are constants: the rate does not
 *                      depend on the values.
 *   copy               reads and writes 1 GiB with 16-byte loads and stores;
 *                      bandwidth counts both directions.
 *
 * The clock the SMs run at depends on the load (power and temperature
 * limits), so it is sampled externally under each load: `cuda_peak hold
 * <test> <seconds>` runs one test back to back for that long, while
 * nvidia-smi reads the SM clock (docs/experiments/cuda-machine/README.md).
 * Kernels launch exactly one wave, every SM filled to its thread limit.
 * stdout gets CSV rows, stderr the device properties.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(x)                                                                         \
    do {                                                                                 \
        cudaError_t e = (x);                                                             \
        if (e != cudaSuccess) {                                                          \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

namespace {

constexpr int kChains = 16;
constexpr int kIters  = 1 << 14;

template <typename T>
__global__ void fma_kernel(T* out) {
    T acc[kChains];
    #pragma unroll
    for (int c = 0; c < kChains; ++c)
        acc[c] = T(threadIdx.x + c);
    const T a = T(0.999), b = T(1e-6);
    for (int i = 0; i < kIters; ++i) {
        #pragma unroll
        for (int c = 0; c < kChains; ++c)
            acc[c] = fma(acc[c], a, b);
    }
    T sum = 0;
    #pragma unroll
    for (int c = 0; c < kChains; ++c)
        sum += acc[c];
    if (sum == T(-1))  // never true; keeps the chains observable
        out[threadIdx.x] = sum;
}

__global__ void mma_kernel(float* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned a0 = 0x3c003c00u, a1 = a0, a2 = a0, a3 = a0;  // fp16 1.0 pairs
    const unsigned b0 = 0x00010001u, b1 = b0;                    // tiny fp16 values
    // Distinct starting values: identical chains with identical inputs
    // would let the compiler merge them into one.
    float acc[kChains][4];
    #pragma unroll
    for (int c = 0; c < kChains; ++c)
        acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = float(c);
    for (int i = 0; i < kIters; ++i) {
        #pragma unroll
        for (int c = 0; c < kChains; ++c)
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[c][0]), "+f"(acc[c][1]), "+f"(acc[c][2]), "+f"(acc[c][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    float sum = 0;
    #pragma unroll
    for (int c = 0; c < kChains; ++c)
        sum += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
    if (sum == -1.f)
        out[threadIdx.x] = sum;
#else
    (void)out;
#endif
}

// One mma.sync per format, on 4 A registers, 2 B registers and an
// accumulator of kAcc registers (fp32 / int32, or fp16 pairs).
#define HPC_MMA4(INS, CON)                                                                        \
    asm volatile(INS " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                  \
                 : "+" CON(c[0]), "+" CON(c[1]), "+" CON(c[2]), "+" CON(c[3])                        \
                 : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b))
#define HPC_MMA2(INS)                                                                            \
    asm volatile(INS " {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"                              \
                 : "+r"(c[0]), "+r"(c[1])                                                         \
                 : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b))
#define HPC_F "f"
#define HPC_R "r"

enum class Mma { bf16_f32, f16_f16, s8_s32, e4m3_f32, e4m3_f16, e4m3_f32_block_scaled };

template <Mma kOp>
__device__ __forceinline__ void mma_op(float* c, unsigned a, unsigned b) {
    if constexpr (kOp == Mma::bf16_f32)
        HPC_MMA4("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", HPC_F);
    else if constexpr (kOp == Mma::e4m3_f32)
        HPC_MMA4("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", HPC_F);
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ == 1200
    else if constexpr (kOp == Mma::e4m3_f32_block_scaled) {
        const unsigned scale = 0x7f7f7f7fu;  // UE8M0 2^0
        const unsigned short sel = 0;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                     ".f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                     "%10, {%11,%11}, %10, {%11,%11};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b), "r"(scale), "h"(sel));
    }
#endif
}

template <Mma kOp>
__device__ __forceinline__ void mma_op(unsigned* c, unsigned a, unsigned b) {
    if constexpr (kOp == Mma::f16_f16)
        HPC_MMA2("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16");
    else if constexpr (kOp == Mma::e4m3_f16)
        HPC_MMA2("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16");
    else if constexpr (kOp == Mma::s8_s32)
        HPC_MMA4("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", HPC_R);
}

template <Mma kOp, typename Acc, int kAcc>
__global__ void mma_lowp_kernel(float* out, unsigned a, unsigned b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    Acc acc[kChains][kAcc];
    #pragma unroll
    for (int c = 0; c < kChains; ++c)
        #pragma unroll
        for (int r = 0; r < kAcc; ++r)
            acc[c][r] = Acc(c);
    for (int i = 0; i < kIters; ++i) {
        #pragma unroll
        for (int c = 0; c < kChains; ++c)
            mma_op<kOp>(acc[c], a, b);
    }
    Acc sum = 0;
    #pragma unroll
    for (int c = 0; c < kChains; ++c)
        sum += acc[c][0];
    if (sum == Acc(12345))
        out[threadIdx.x] = float(sum);
#else
    (void)out, (void)a, (void)b;
#endif
}

// 1 if this build has the block-scaled FP8 mma (an sm_120a build), else 0.
__global__ void block_scaled_probe(int* out) {
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ == 1200
    *out = 1;
#else
    *out = 0;
#endif
}

__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x)
        dst[i] = src[i];
}

/// Best-of-5 GPU time of `launch`, in ms.
template <typename F>
float best_ms(F launch) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    launch();  // warm-up
    CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        CHECK(cudaEventRecord(start));
        launch();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        best = std::min(best, ms);
    }
    CHECK(cudaGetLastError());
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    const int threads = 256;
    const int blocks  = p.multiProcessorCount * (p.maxThreadsPerMultiProcessor / threads);
    float* out;
    CHECK(cudaMalloc(&out, 1024 * sizeof(double)));
    const size_t copy_bytes = size_t(1) << 30, copy_n = copy_bytes / sizeof(float4);
    float4 *src, *dst;
    CHECK(cudaMalloc(&src, copy_bytes));
    CHECK(cudaMalloc(&dst, copy_bytes));
    CHECK(cudaMemset(src, 0, copy_bytes));

    auto fma32 = [&] { fma_kernel<float><<<blocks, threads>>>(out); };
    auto fma64 = [&] { fma_kernel<double><<<blocks, threads>>>(reinterpret_cast<double*>(out)); };
    auto mma   = [&] { mma_kernel<<<blocks, threads>>>(out); };
    auto copy  = [&] { copy_kernel<<<p.multiProcessorCount * 16, 256>>>(src, dst, copy_n); };

    if (argc == 4 && std::string(argv[1]) == "hold") {
        const std::string test = argv[2];
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(std::atoi(argv[3]));
        while (std::chrono::steady_clock::now() < until) {
            for (int i = 0; i < 20; ++i)
                test == "fma_f32" ? fma32() : test == "fma_f64" ? fma64() : test == "mma" ? mma() : copy();
            CHECK(cudaDeviceSynchronize());
        }
        return 0;
    }

    std::fprintf(stderr,
                 "%s, sm_%d%d: %d SMs, %d threads and %d registers per SM, "
                 "%zu KiB shared memory per SM (%zu KiB per block opt-in), "
                 "%d MiB L2, %.1f GiB memory on a %d-bit bus\n",
                 p.name, p.major, p.minor, p.multiProcessorCount, p.maxThreadsPerMultiProcessor,
                 p.regsPerMultiprocessor, p.sharedMemPerMultiprocessor / 1024,
                 p.sharedMemPerBlockOptin / 1024, p.l2CacheSize >> 20,
                 double(p.totalGlobalMem) / (1 << 30), p.memoryBusWidth);
    std::printf("test,value,unit\n");
    const double fmas = double(blocks) * threads * kIters * kChains;
    std::printf("fma_f32,%.1f,TFLOP/s\n", 2 * fmas / best_ms(fma32) / 1e9);
    std::printf("fma_f64,%.2f,TFLOP/s\n", 2 * fmas / best_ms(fma64) / 1e9);
    if (p.major >= 8) {  // 2*16*8*16 flops per mma.sync per warp
        const double mmas = double(blocks) * (threads / 32) * kIters * kChains;
        std::printf("mma_f16_f32acc,%.1f,TFLOP/s\n", mmas * 2 * 16 * 8 * 16 / best_ms(mma) / 1e9);
    }
    if (p.major * 10 + p.minor >= 89) {  // 2*16*8*K flops per mma.sync per warp, K = 16 or 32
        const double mmas = double(blocks) * (threads / 32) * kIters * kChains;
        auto row = [&](const char* name, int k, auto launch) {
            std::printf("%s,%.1f,TFLOP/s\n", name, mmas * 2 * 16 * 8 * k / best_ms(launch) / 1e9);
        };
        row("mma_bf16_f32acc", 16, [&] { mma_lowp_kernel<Mma::bf16_f32, float, 4><<<blocks, threads>>>(out, 0x3f803f80u, 0x00010001u); });
        row("mma_f16_f16acc", 16, [&] { mma_lowp_kernel<Mma::f16_f16, unsigned, 2><<<blocks, threads>>>(out, 0x3c003c00u, 0x00010001u); });
        row("mma_s8_s32acc", 32, [&] { mma_lowp_kernel<Mma::s8_s32, unsigned, 4><<<blocks, threads>>>(out, 0x01010101u, 0x01010101u); });
        row("mma_e4m3_f32acc", 32, [&] { mma_lowp_kernel<Mma::e4m3_f32, float, 4><<<blocks, threads>>>(out, 0x38383838u, 0x01010101u); });
        row("mma_e4m3_f16acc", 32, [&] { mma_lowp_kernel<Mma::e4m3_f16, unsigned, 2><<<blocks, threads>>>(out, 0x38383838u, 0x01010101u); });
        int block_scaled = 0;
        block_scaled_probe<<<1, 1>>>(reinterpret_cast<int*>(out));
        CHECK(cudaMemcpy(&block_scaled, out, sizeof(int), cudaMemcpyDeviceToHost));
        if (block_scaled)  // only an sm_120a build has the instruction
            row("mma_e4m3_f32acc_block_scaled", 32, [&] { mma_lowp_kernel<Mma::e4m3_f32_block_scaled, float, 4><<<blocks, threads>>>(out, 0x38383838u, 0x01010101u); });
    }
    std::printf("dram_copy,%.0f,GB/s\n", 2.0 * copy_bytes / best_ms(copy) / 1e6);
    CHECK(cudaFree(src));
    CHECK(cudaFree(dst));
    CHECK(cudaFree(out));
}
