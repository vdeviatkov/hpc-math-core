// Metal GEMM kernels, C = A·B, row-major, A M×K, B K×N, C M×N.
//
// Compiled at run time from this source (embedded into the binary by CMake,
// see src/metal/gemm_metal.mm), so the build needs no offline Metal
// toolchain. Host interface and the level-by-level design:
// src/gemm/metal.hpp.
//
//   0  gemm_naive            one thread per C(i,j), device memory only
//   1  gemm_tiled            16×16 threadgroup-memory tiles
//   2  gemm_reg_tile         64×64 block tile, 8×4 outputs per thread, float4 loads
//   3  gemm_simd             simdgroup_float8x8 MMA straight from device memory
//   4  gemm_simd_tg          simdgroup MMA fed from threadgroup memory
//   5  gemm_simd_tg_f16      Level 4 with fp16 A and B, fp32 accumulate
//
// Levels 2-5 assume M, N, K are multiples of their tiles; the host falls back
// to Level 1, which handles any size.

#include <metal_stdlib>
using namespace metal;

// Full unrolling keeps per-thread arrays (accumulators, fragments) in
// registers; a loop left rolled indexes them dynamically and spills them.
#define HPC_UNROLL _Pragma("clang loop unroll(full)")

struct Dims {
    uint M, K, N;
};

// ---------------------------------------------------------------------------
// Level 0: one thread per output, every operand read from device memory.
// ---------------------------------------------------------------------------
kernel void gemm_naive(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                       device float* C [[buffer(2)]], constant Dims& d [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= d.M || gid.x >= d.N)
        return;
    float acc = 0.0f;
    for (uint k = 0; k < d.K; ++k)
        acc = fma(A[gid.y * d.K + k], B[k * d.N + gid.x], acc);
    C[gid.y * d.N + gid.x] = acc;
}

// ---------------------------------------------------------------------------
// Level 1: 16×16 tiles of A and B staged in threadgroup memory, one output
// per thread. Bounds-checked, so it serves as the fallback for any shape.
// ---------------------------------------------------------------------------
constant constexpr uint kT1 = 16;

kernel void gemm_tiled(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                       device float* C [[buffer(2)]], constant Dims& d [[buffer(3)]],
                       uint2 tid [[thread_position_in_threadgroup]],
                       uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup float As[kT1][kT1];
    threadgroup float Bs[kT1][kT1];
    const uint row = tg.y * kT1 + tid.y;
    const uint col = tg.x * kT1 + tid.x;
    float acc      = 0.0f;
    for (uint k0 = 0; k0 < d.K; k0 += kT1) {
        As[tid.y][tid.x] = (row < d.M && k0 + tid.x < d.K) ? A[row * d.K + k0 + tid.x] : 0.0f;
        Bs[tid.y][tid.x] = (k0 + tid.y < d.K && col < d.N) ? B[(k0 + tid.y) * d.N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < kT1; ++k)
            acc = fma(As[tid.y][k], Bs[k][tid.x], acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < d.M && col < d.N)
        C[row * d.N + col] = acc;
}

// ---------------------------------------------------------------------------
// Level 2: BM×BN block tile per threadgroup, TM×TN outputs per thread.
// A is stored transposed in threadgroup memory (As[k][m]) so both operands
// of the inner product are contiguous float4 reads; every value read from
// threadgroup memory feeds TM or TN FMAs instead of one.
// ---------------------------------------------------------------------------
template <uint BM, uint BN, uint BK, uint TM, uint TN>
kernel void gemm_reg_tile(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                          device float* C [[buffer(2)]], constant Dims& d [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
    constexpr uint kThreads = (BM / TM) * (BN / TN);
    threadgroup float As[BK][BM];
    threadgroup float Bs[BK][BN];

    const uint tx = tid % (BN / TN);
    const uint ty = tid / (BN / TN);
    const uint m0 = tg.y * BM;
    const uint n0 = tg.x * BN;

    float acc[TM][TN] = {};
    for (uint k0 = 0; k0 < d.K; k0 += BK) {
        for (uint i = tid; i < BM * BK / 4; i += kThreads) {
            const uint r  = i / (BK / 4);
            const uint kq = (i % (BK / 4)) * 4;
            const float4 v = *reinterpret_cast<device const float4*>(A + (m0 + r) * d.K + k0 + kq);
            As[kq + 0][r] = v.x;
            As[kq + 1][r] = v.y;
            As[kq + 2][r] = v.z;
            As[kq + 3][r] = v.w;
        }
        for (uint i = tid; i < BK * BN / 4; i += kThreads) {
            const uint k = i / (BN / 4);
            const uint c = (i % (BN / 4)) * 4;
            *reinterpret_cast<threadgroup float4*>(&Bs[k][c]) =
                *reinterpret_cast<device const float4*>(B + (k0 + k) * d.N + n0 + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        HPC_UNROLL
        for (uint k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            HPC_UNROLL
            for (uint i = 0; i < TM; ++i)
                a[i] = As[k][ty * TM + i];
            HPC_UNROLL
            for (uint j = 0; j < TN; ++j)
                b[j] = Bs[k][tx * TN + j];
            HPC_UNROLL
            for (uint i = 0; i < TM; ++i)
                HPC_UNROLL
                for (uint j = 0; j < TN; ++j)
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    HPC_UNROLL
    for (uint i = 0; i < TM; ++i)
        HPC_UNROLL
        for (uint j = 0; j < TN; j += 4)
            *reinterpret_cast<device float4*>(C + (m0 + ty * TM + i) * d.N + n0 + tx * TN + j) =
                float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
}

template [[host_name("gemm_reg_tile")]] kernel void gemm_reg_tile<64, 64, 16, 8, 4>(
    device const float*, device const float*, device float*, constant Dims&, uint, uint2);
template [[host_name("gemm_reg_tile_128x128_8x8")]] kernel void gemm_reg_tile<128, 128, 16, 8, 8>(
    device const float*, device const float*, device float*, constant Dims&, uint, uint2);
template [[host_name("gemm_reg_tile_64x64_4x4")]] kernel void gemm_reg_tile<64, 64, 16, 4, 4>(
    device const float*, device const float*, device float*, constant Dims&, uint, uint2);
template [[host_name("gemm_reg_tile_128x64_8x8")]] kernel void gemm_reg_tile<128, 64, 16, 8, 8>(
    device const float*, device const float*, device float*, constant Dims&, uint, uint2);

// ---------------------------------------------------------------------------
// Level 3: simdgroup matrices. A simdgroup (32 threads) multiplies 8×8
// fragments with simdgroup_multiply_accumulate; each simdgroup owns an
// SM×SN block of C (SM/8 × SN/8 accumulator fragments) and loads A and B
// fragments straight from device memory, relying on the GPU's caches for
// reuse between simdgroups. A threadgroup is WM×WN simdgroups.
// ---------------------------------------------------------------------------
template <uint WM, uint WN, uint SM, uint SN>
kernel void gemm_simd(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                      device float* C [[buffer(2)]], constant Dims& d [[buffer(3)]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint2 tg [[threadgroup_position_in_grid]]) {
    constexpr uint FM = SM / 8, FN = SN / 8;
    const uint m0 = tg.y * (WM * SM) + (sg / WN) * SM;
    const uint n0 = tg.x * (WN * SN) + (sg % WN) * SN;

    simdgroup_float8x8 acc[FM][FN];
    HPC_UNROLL
    for (uint i = 0; i < FM; ++i)
        HPC_UNROLL
        for (uint j = 0; j < FN; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < d.K; k += 8) {
        simdgroup_float8x8 a[FM], b[FN];
        HPC_UNROLL
        for (uint i = 0; i < FM; ++i)
            simdgroup_load(a[i], A + (m0 + i * 8) * d.K + k, d.K);
        HPC_UNROLL
        for (uint j = 0; j < FN; ++j)
            simdgroup_load(b[j], B + k * d.N + n0 + j * 8, d.N);
        HPC_UNROLL
        for (uint i = 0; i < FM; ++i)
            HPC_UNROLL
            for (uint j = 0; j < FN; ++j)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    HPC_UNROLL
    for (uint i = 0; i < FM; ++i)
        HPC_UNROLL
        for (uint j = 0; j < FN; ++j)
            simdgroup_store(acc[i][j], C + (m0 + i * 8) * d.N + n0 + j * 8, d.N);
}

template [[host_name("gemm_simd")]] kernel void gemm_simd<2, 2, 32, 32>(
    device const float*, device const float*, device float*, constant Dims&, uint, uint2);
template [[host_name("gemm_simd_2x2_16x32")]] kernel void gemm_simd<2, 2, 16, 32>(
    device const float*, device const float*, device float*, constant Dims&, uint, uint2);
template [[host_name("gemm_simd_2x2_32x16")]] kernel void gemm_simd<2, 2, 32, 16>(
    device const float*, device const float*, device float*, constant Dims&, uint, uint2);

// ---------------------------------------------------------------------------
// Levels 4 and 5: Level 3 with BM×BK and BK×BN tiles of A and B staged in
// threadgroup memory first, so every element is read from device memory
// once per threadgroup instead of once per simdgroup. T is the input type
// (float, or half for Level 5); accumulation is always in float.
//
// Each tile row is padded by kPad elements, which shifts consecutive rows
// across threadgroup-memory banks for the column-strided fragment loads.
// ---------------------------------------------------------------------------
template <typename T, uint WM, uint WN, uint SM, uint SN, uint BK>
kernel void gemm_simd_tg(device const T* A [[buffer(0)]], device const T* B [[buffer(1)]],
                         device float* C [[buffer(2)]], constant Dims& d [[buffer(3)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint2 tg [[threadgroup_position_in_grid]]) {
    using vec4 = vec<T, 4>;
    constexpr uint BM = WM * SM, BN = WN * SN;
    constexpr uint FM = SM / 8, FN = SN / 8;
    constexpr uint kThreads = WM * WN * 32;
    constexpr uint kPad = 16 / sizeof(T);
    threadgroup T As[BM][BK + kPad];
    threadgroup T Bs[BK][BN + kPad];

    const uint m0 = tg.y * BM;
    const uint n0 = tg.x * BN;
    const uint sm = (sg / WN) * SM;
    const uint sn = (sg % WN) * SN;

    simdgroup_float8x8 acc[FM][FN];
    HPC_UNROLL
    for (uint i = 0; i < FM; ++i)
        HPC_UNROLL
        for (uint j = 0; j < FN; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < d.K; k0 += BK) {
        for (uint i = tid; i < BM * BK / 4; i += kThreads) {
            const uint r = i / (BK / 4), c = (i % (BK / 4)) * 4;
            *reinterpret_cast<threadgroup vec4*>(&As[r][c]) =
                *reinterpret_cast<device const vec4*>(A + (m0 + r) * d.K + k0 + c);
        }
        for (uint i = tid; i < BK * BN / 4; i += kThreads) {
            const uint r = i / (BN / 4), c = (i % (BN / 4)) * 4;
            *reinterpret_cast<threadgroup vec4*>(&Bs[r][c]) =
                *reinterpret_cast<device const vec4*>(B + (k0 + r) * d.N + n0 + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        HPC_UNROLL
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<T, 8, 8> a[FM], b[FN];
            HPC_UNROLL
            for (uint i = 0; i < FM; ++i)
                simdgroup_load(a[i], &As[sm + i * 8][kk], BK + kPad);
            HPC_UNROLL
            for (uint j = 0; j < FN; ++j)
                simdgroup_load(b[j], &Bs[kk][sn + j * 8], BN + kPad);
            HPC_UNROLL
            for (uint i = 0; i < FM; ++i)
                HPC_UNROLL
                for (uint j = 0; j < FN; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    HPC_UNROLL
    for (uint i = 0; i < FM; ++i)
        HPC_UNROLL
        for (uint j = 0; j < FN; ++j)
            simdgroup_store(acc[i][j], C + (m0 + sm + i * 8) * d.N + n0 + sn + j * 8, d.N);
}

#define HPC_SIMD_TG(name, T, WM, WN, SM, SN, BK)                                               \
    template [[host_name(name)]] kernel void gemm_simd_tg<T, WM, WN, SM, SN, BK>(              \
        device const T*, device const T*, device float*, constant Dims&, uint, uint, uint2);
HPC_SIMD_TG("gemm_simd_tg", float, 2, 2, 32, 32, 16)
HPC_SIMD_TG("gemm_simd_tg_2x2_32x32_k32", float, 2, 2, 32, 32, 32)
HPC_SIMD_TG("gemm_simd_tg_2x4_32x32_k16", float, 2, 4, 32, 32, 16)
HPC_SIMD_TG("gemm_simd_tg_4x2_32x32_k16", float, 4, 2, 32, 32, 16)
HPC_SIMD_TG("gemm_simd_tg_2x2_64x32_k16", float, 2, 2, 64, 32, 16)
HPC_SIMD_TG("gemm_simd_tg_f16", half, 2, 2, 32, 32, 16)
HPC_SIMD_TG("gemm_simd_tg_f16_2x2_32x32_k32", half, 2, 2, 32, 32, 32)
HPC_SIMD_TG("gemm_simd_tg_f16_2x4_32x32_k16", half, 2, 4, 32, 32, 16)
HPC_SIMD_TG("gemm_simd_tg_f16_2x2_64x32_k16", half, 2, 2, 64, 32, 16)
#undef HPC_SIMD_TG

// ---------------------------------------------------------------------------
// FMA peak: every thread runs kChains independent FMA chains of 4-wide
// vectors, so the ALUs never wait on a result. Throughput = threads ×
// iters × kChains × 4 lanes × 2 flops / GPU time. Seeded from `seed` and
// written out so the compiler cannot fold or drop the loop.
// ---------------------------------------------------------------------------
template <typename T>
kernel void fma_peak(device T* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                     constant float& seed [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    using vec4             = vec<T, 4>;
    constexpr uint kChains = 8;
    vec4 x[kChains];
    HPC_UNROLL
    for (uint c = 0; c < kChains; ++c)
        x[c] = vec4(T(seed) + T(c) * T(0.001) + T(gid & 7) * T(0.0001));
    const vec4 m = vec4(T(0.999)), a = vec4(T(0.0001));
    for (uint i = 0; i < iters; ++i)
        HPC_UNROLL
        for (uint c = 0; c < kChains; ++c)
            x[c] = fma(x[c], m, a);
    vec4 s = x[0];
    for (uint c = 1; c < kChains; ++c)
        s += x[c];
    out[gid] = s.x + s.y + s.z + s.w;
}

template [[host_name("fma_peak_f32")]] kernel void fma_peak<float>(device float*, constant uint&,
                                                                   constant float&, uint);
template [[host_name("fma_peak_f16")]] kernel void fma_peak<half>(device half*, constant uint&,
                                                                  constant float&, uint);
