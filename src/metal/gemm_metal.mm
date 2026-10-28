// Host side of the Metal GEMM kernels (interface and design: gemm/metal.hpp).
//
// One process-lifetime context holds the device, a command queue and the
// pipelines compiled from gemm.metal, whose source CMake embeds as
// kMetalSource (hpc_metal_source.inc). Buffers use shared storage: on Apple
// silicon the CPU and GPU address the same memory, so set_inputs and
// get_output are plain memcpys and nothing is copied at dispatch time.

#include "gemm/metal.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>

namespace hpc::gemm {

namespace {

#include "hpc_metal_source.inc"  // static const char kMetalSource[]

/// Threadgroup geometry of one compiled GEMM function.
struct Geometry {
    unsigned bm, bn, bk;  ///< C tile per threadgroup and k step; shapes must be multiples
    unsigned threads;     ///< threads per threadgroup
    bool f16;             ///< reads the fp16 copies of A and B
};

/// Every tiled kernel and variant in gemm.metal, with the sizes of its
/// template arguments there (HPC_REG_TILE, HPC_SIMD, HPC_SIMD_TG): the two
/// lists must agree, which the MetalVariants test checks. Levels 0 and 1 are
/// dispatched separately (any shape).
const std::map<std::string, Geometry>& geometries() {
    static const std::map<std::string, Geometry> table = {
        {"gemm_reg_tile", {64, 64, 16, 128, false}},
        {"gemm_reg_tile_128x128_8x8", {128, 128, 16, 256, false}},
        {"gemm_reg_tile_64x64_4x4", {64, 64, 16, 256, false}},
        {"gemm_reg_tile_128x64_8x8", {128, 64, 16, 128, false}},
        {"gemm_simd", {64, 64, 8, 128, false}},
        {"gemm_simd_2x2_16x32", {32, 64, 8, 128, false}},
        {"gemm_simd_2x2_32x16", {64, 32, 8, 128, false}},
        {"gemm_simd_tg", {64, 64, 16, 128, false}},
        {"gemm_simd_tg_2x2_32x32_k32", {64, 64, 32, 128, false}},
        {"gemm_simd_tg_2x4_32x32_k16", {64, 128, 16, 256, false}},
        {"gemm_simd_tg_4x2_32x32_k16", {128, 64, 16, 256, false}},
        {"gemm_simd_tg_2x2_64x32_k16", {128, 64, 16, 128, false}},
        {"gemm_simd_tg_f16", {64, 64, 16, 128, true}},
        {"gemm_simd_tg_f16_2x2_32x32_k32", {64, 64, 32, 128, true}},
        {"gemm_simd_tg_f16_2x4_32x32_k16", {64, 128, 16, 256, true}},
        {"gemm_simd_tg_f16_2x2_64x32_k16", {128, 64, 16, 128, true}},
    };
    return table;
}

/// The gemm.metal function of a hand-written level; nullptr for the MPS reference.
const char* function_name(MetalKernel k) {
    switch (k) {
    case MetalKernel::kNaive: return "gemm_naive";
    case MetalKernel::kTiled: return "gemm_tiled";
    case MetalKernel::kRegTile: return "gemm_reg_tile";
    case MetalKernel::kSimd: return "gemm_simd";
    case MetalKernel::kSimdTg: return "gemm_simd_tg";
    case MetalKernel::kSimdTgF16: return "gemm_simd_tg_f16";
    case MetalKernel::kMps:
    case MetalKernel::kMpsF16: return nullptr;
    }
    return nullptr;
}

/// Throw a runtime_error naming what failed, with the NSError's text if any.
[[noreturn]] void fail(const std::string& what, NSError* error = nil) {
    std::string msg = "Metal: " + what;
    if (error)
        msg += ": " + std::string(error.localizedDescription.UTF8String);
    throw std::runtime_error(msg);
}

/// Device, queue and compiled pipelines, created once.
struct Context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;
    std::mutex mutex;

    /// Open the default device and compile gemm.metal (fast-math).
    Context() {
        device = MTLCreateSystemDefaultDevice();
        if (!device)
            fail("no device");
        queue = [device newCommandQueue];
        MTLCompileOptions* options = [MTLCompileOptions new];
        // mathMode exists from the macOS 15 SDK and OS on; before that,
        // fast math is the default (fastMathEnabled defaults to YES).
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_15_0
        if (@available(macOS 15.0, *))
            options.mathMode = MTLMathModeFast;
#endif
        NSError* error = nil;
        library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                       options:options
                                         error:&error];
        if (!library)
            fail("compiling gemm.metal", error);
    }

    /// The pipeline of function `name`, built on first use and cached.
    id<MTLComputePipelineState> pipeline(const std::string& name) {
        std::lock_guard lock(mutex);
        auto it = pipelines.find(name);
        if (it != pipelines.end())
            return it->second;
        id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        if (!fn)
            fail("no function " + name);
        NSError* error = nil;
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pso)
            fail("pipeline " + name, error);
        return pipelines[name] = pso;
    }
};

/// The process-lifetime Context, created on first use.
Context& context() {
    static Context ctx;
    return ctx;
}

/// The GEMM shape as the kernels read it (gemm.metal's Dims).
struct Dims {
    uint32_t M, K, N;
};

/// Seconds of GPU time the command buffer took, after waiting for it.
double gpu_seconds(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError)
        fail("command buffer", cb.error);
    return cb.GPUEndTime - cb.GPUStartTime;
}

}  // namespace

bool metal_available() noexcept {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

std::string metal_device_name() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device ? std::string(device.name.UTF8String) : std::string();
    }
}

std::vector<std::string> metal_kernel_variants() {
    std::vector<std::string> names;
    for (const auto& [name, g] : geometries())
        names.push_back(name);
    return names;
}

struct MetalGemm::Impl {
    std::size_t M, K, N;
    id<MTLBuffer> A, B, C, A16, B16;
    MPSMatrixMultiplication* mps = nil;
    MPSMatrixMultiplication* mps16 = nil;

    /// True when the shape is a multiple of the kernel's tile and k step.
    bool fits(const Geometry& g) const {
        return M % g.bm == 0 && N % g.bn == 0 && K % g.bk == 0;
    }

    /// Nothing to launch: M or N is 0, or K is 0 after clearing C.
    bool empty_product() const {
        if (M == 0 || N == 0)
            return true;
        if (K != 0)
            return false;
        std::memset(C.contents, 0, M * N * 4);
        return true;
    }

    /// Encode `reps` dispatches of `name` into one compute encoder; the
    /// default serial dispatch type orders them. Returns ms per GEMM.
    double run_function(const std::string& name, int reps) {
        if (empty_product())
            return 0.0;
        Context& ctx = context();
        id<MTLComputePipelineState> pso = ctx.pipeline(name);
        const Dims d{uint32_t(M), uint32_t(K), uint32_t(N)};
        MTLSize grid, group;
        bool f16 = false;
        if (name == "gemm_naive" || name == "gemm_tiled") {
            group = MTLSizeMake(16, 16, 1);
            grid  = MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1);
        } else {
            const Geometry& g = geometries().at(name);
            if (!fits(g))
                fail(name + " needs M, N multiples of its tile and K of its k step");
            group = MTLSizeMake(g.threads, 1, 1);
            grid  = MTLSizeMake(N / g.bn, M / g.bm, 1);
            f16   = g.f16;
        }
        if (group.width * group.height > pso.maxTotalThreadsPerThreadgroup)
            fail(name + " needs more threads per threadgroup than the pipeline allows");
        id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:(f16 ? A16 : A) offset:0 atIndex:0];
        [enc setBuffer:(f16 ? B16 : B) offset:0 atIndex:1];
        [enc setBuffer:C offset:0 atIndex:2];
        [enc setBytes:&d length:sizeof d atIndex:3];
        for (int r = 0; r < reps; ++r)
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        return gpu_seconds(cb) * 1e3 / reps;
    }

    /// MPSMatrixMultiplication, f32 or fp16 inputs, `reps` times in one
    /// command buffer. Returns ms per GEMM.
    double run_mps(bool f16, int reps) {
        if (empty_product())
            return 0.0;
        Context& ctx = context();
        MPSMatrixMultiplication* __strong& op = f16 ? mps16 : mps;
        if (!op)
            op = [[MPSMatrixMultiplication alloc] initWithDevice:ctx.device
                                                   transposeLeft:NO
                                                  transposeRight:NO
                                                      resultRows:M
                                                   resultColumns:N
                                                 interiorColumns:K
                                                           alpha:1.0
                                                            beta:0.0];
        const MPSDataType in   = f16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        const std::size_t size = f16 ? 2 : 4;
        auto desc = [](std::size_t rows, std::size_t cols, std::size_t elem, MPSDataType type) {
            return [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                         columns:cols
                                                        rowBytes:cols * elem
                                                        dataType:type];
        };
        MPSMatrix* a = [[MPSMatrix alloc] initWithBuffer:(f16 ? A16 : A) descriptor:desc(M, K, size, in)];
        MPSMatrix* b = [[MPSMatrix alloc] initWithBuffer:(f16 ? B16 : B) descriptor:desc(K, N, size, in)];
        MPSMatrix* c = [[MPSMatrix alloc] initWithBuffer:C descriptor:desc(M, N, 4, MPSDataTypeFloat32)];
        id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
        for (int r = 0; r < reps; ++r)
            [op encodeToCommandBuffer:cb leftMatrix:a rightMatrix:b resultMatrix:c];
        return gpu_seconds(cb) * 1e3 / reps;
    }
};

MetalGemm::MetalGemm(std::size_t M, std::size_t K, std::size_t N) : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        Context& ctx = context();
        impl_->M = M;
        impl_->K = K;
        impl_->N = N;
        auto alloc = [&](std::size_t bytes) {
            id<MTLBuffer> buf = [ctx.device newBufferWithLength:std::max<std::size_t>(bytes, 16)
                                                        options:MTLResourceStorageModeShared];
            if (!buf)
                fail("allocating " + std::to_string(bytes) + " bytes");
            return buf;
        };
        impl_->A   = alloc(M * K * 4);
        impl_->B   = alloc(K * N * 4);
        impl_->C   = alloc(M * N * 4);
        impl_->A16 = alloc(M * K * 2);
        impl_->B16 = alloc(K * N * 2);
    }
}

MetalGemm::~MetalGemm() = default;

void MetalGemm::set_inputs(const float* A, const float* B) {
    Impl& p = *impl_;
    std::memcpy(p.A.contents, A, p.M * p.K * 4);
    std::memcpy(p.B.contents, B, p.K * p.N * 4);
    auto* a16 = static_cast<_Float16*>(p.A16.contents);
    auto* b16 = static_cast<_Float16*>(p.B16.contents);
    for (std::size_t i = 0; i < p.M * p.K; ++i)
        a16[i] = static_cast<_Float16>(A[i]);
    for (std::size_t i = 0; i < p.K * p.N; ++i)
        b16[i] = static_cast<_Float16>(B[i]);
}

void MetalGemm::get_output(float* C) const { std::memcpy(C, impl_->C.contents, impl_->M * impl_->N * 4); }

MetalKernel MetalGemm::effective(MetalKernel kernel) const noexcept {
    const char* name = function_name(kernel);
    if (!name || kernel == MetalKernel::kNaive || kernel == MetalKernel::kTiled)
        return kernel;
    return impl_->fits(geometries().at(name)) ? kernel : MetalKernel::kTiled;
}

double MetalGemm::run_ms(MetalKernel kernel, int reps) {
    @autoreleasepool {
        if (kernel == MetalKernel::kMps || kernel == MetalKernel::kMpsF16)
            return impl_->run_mps(kernel == MetalKernel::kMpsF16, reps);
        return impl_->run_function(function_name(effective(kernel)), reps);
    }
}

double MetalGemm::run_variant_ms(const std::string& function, int reps) {
    @autoreleasepool {
        return impl_->run_function(function, reps);
    }
}

void gemm_metal(MetalKernel kernel, const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    if (A.cols() != B.rows() || C.rows() != A.rows() || C.cols() != B.cols())
        throw std::invalid_argument("gemm_metal: shape mismatch");
    MetalGemm g(A.rows(), A.cols(), B.cols());
    g.set_inputs(A.data(), B.data());
    g.run_ms(kernel);
    g.get_output(C.data());
}

double metal_fma_peak_gflops(bool f16) {
    @autoreleasepool {
        Context& ctx = context();
        id<MTLComputePipelineState> pso = ctx.pipeline(f16 ? "fma_peak_f16" : "fma_peak_f32");
        // Enough threads to fill every core several times over; iterations
        // long enough that launch overhead is negligible (2^40 flops per run,
        // ~75 ms on an M4 Max).
        // Best of 10 after one warm-up run, so the GPU clock has ramped up.
        const uint32_t threads = 1u << 20, iters = 16384;
        const float seed       = 1.0f;
        id<MTLBuffer> out = [ctx.device newBufferWithLength:threads * 4 options:MTLResourceStorageModeShared];
        double best = 0.0;
        for (int run = 0; run < 11; ++run) {
            id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBytes:&iters length:sizeof iters atIndex:1];
            [enc setBytes:&seed length:sizeof seed atIndex:2];
            [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            const double s     = gpu_seconds(cb);
            const double flops = double(threads) * iters * 8 /*chains*/ * 4 /*lanes*/ * 2;
            if (run > 0)
                best = std::max(best, flops / s / 1e9);
        }
        return best;
    }
}

}  // namespace hpc::gemm
