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
// The file reads top to bottom: shared helpers, then the kernels, each with
// its tile variants (the host's geometry table in gemm_metal.mm lists the
// same names and sizes). Levels 2-5 assume M, N, K are multiples of their
// tiles; the host falls back to Level 1, which handles any size.

#include <metal_stdlib>
using namespace metal;

// Full unrolling keeps per-thread arrays (accumulators, fragments) in
// registers; a loop left rolled indexes them dynamically and spills them.
#define HPC_UNROLL _Pragma("clang loop unroll(full)")

/// The GEMM shape, passed by the host in buffer 3 (gemm_metal.mm's Dims).
struct Dims {
    uint M, K, N;
};

// Sizes fixed by Metal itself, not tuning knobs:
//
// kVec   widest Metal vector type (float4, half4; there is no float8). Tile
//        loads and stores move kVec elements at a time, so every tile edge
//        they walk along (BK, BN, TN) must be a multiple of kVec.
// kFrag  edge of a simdgroup matrix. simdgroup_matrix<T, 8, 8> is the only
//        shape Metal provides, so simdgroup tiles (SM, SN, BK) and the K
//        step are multiples of kFrag.
constant constexpr uint kVec  = 4;
constant constexpr uint kFrag = 8;

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Copy the ROWS×COLS tile at `src` (row stride ld) into threadgroup memory
/// at `dst` (row stride DST_LD), kVec elements per load and store, the
/// THREADS threads of the threadgroup taking consecutive vectors.
template <typename T, uint ROWS, uint COLS, uint DST_LD, uint THREADS>
inline void load_tile(threadgroup T* dst, device const T* src, uint ld, uint tid) {
    using vec4 = vec<T, kVec>;
    for (uint i = tid; i < ROWS * COLS / kVec; i += THREADS) {
        const uint r = i / (COLS / kVec), c = (i % (COLS / kVec)) * kVec;
        *reinterpret_cast<threadgroup vec4*>(dst + r * DST_LD + c) =
            *reinterpret_cast<device const vec4*>(src + r * ld + c);
    }
}

/// Set every accumulator fragment to zero.
template <uint FM, uint FN>
inline void zero_acc(thread simdgroup_float8x8 (&acc)[FM][FN]) {
    HPC_UNROLL
    for (uint i = 0; i < FM; ++i)
        HPC_UNROLL
        for (uint j = 0; j < FN; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, kFrag, kFrag>(0.0f);
}

/// One 8-deep k step of a simdgroup's (8·FM)×(8·FN) block: load FM
/// fragments of A going down from `a` (row stride lda) and FN fragments of
/// B going right from `b` (row stride ldb), then accumulate every product
/// acc[i][j] += a[i]·b[j]. A and B may be in device or threadgroup memory.
template <typename T, uint FM, uint FN, typename PtrA, typename PtrB>
inline void simd_mma(thread simdgroup_float8x8 (&acc)[FM][FN], PtrA a, uint lda, PtrB b, uint ldb) {
    simdgroup_matrix<T, kFrag, kFrag> fa[FM], fb[FN];
    HPC_UNROLL
    for (uint i = 0; i < FM; ++i)
        simdgroup_load(fa[i], a + i * kFrag * lda, lda);
    HPC_UNROLL
    for (uint j = 0; j < FN; ++j)
        simdgroup_load(fb[j], b + j * kFrag, ldb);
    HPC_UNROLL
    for (uint i = 0; i < FM; ++i)
        HPC_UNROLL
        for (uint j = 0; j < FN; ++j)
            simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
}

/// Store the accumulator fragments to C, whose block starts at `c` (row stride ldc).
template <uint FM, uint FN>
inline void store_acc(thread simdgroup_float8x8 (&acc)[FM][FN], device float* c, uint ldc) {
    HPC_UNROLL
    for (uint i = 0; i < FM; ++i)
        HPC_UNROLL
        for (uint j = 0; j < FN; ++j)
            simdgroup_store(acc[i][j], c + i * kFrag * ldc + j * kFrag, ldc);
}

// ===========================================================================
// Level 0: one thread per output, every operand read from device memory.
// ===========================================================================

/// C(gid.y, gid.x) = Σ_k A(gid.y, k)·B(k, gid.x).
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

// ===========================================================================
// Level 1: 16×16 tiles of A and B staged in threadgroup memory, one output
// per thread. Bounds-checked, so it serves as the fallback for any shape.
// ===========================================================================
constant constexpr uint kT1 = 16;

/// Level 0's one output per thread, with A and B staged through threadgroup memory.
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

// ===========================================================================
// Level 2: BM×BN block tile per threadgroup, TM×TN outputs per thread.
// A is stored transposed in threadgroup memory (As[k][m]) so both operands
// of the inner product are contiguous reads; every value read from
// threadgroup memory feeds TM or TN FMAs instead of one.
// ===========================================================================

/// One BM×BN tile of C, TM×TN outputs per thread, (BM/TM)·(BN/TN) threads.
template <uint BM, uint BN, uint BK, uint TM, uint TN>
kernel void gemm_reg_tile(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                          device float* C [[buffer(2)]], constant Dims& d [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
    constexpr uint kThreads = (BM / TM) * (BN / TN);
    threadgroup float As[BK][BM];  // transposed: As[k][m] = A(m, k)
    threadgroup float Bs[BK][BN];

    const uint tx = tid % (BN / TN);  // this thread's TM×TN block
    const uint ty = tid / (BN / TN);
    const uint m0 = tg.y * BM, n0 = tg.x * BN;

    float acc[TM][TN] = {};
    for (uint k0 = 0; k0 < d.K; k0 += BK) {
        // A: each float4 is kVec consecutive k of row r, scattered into kVec rows of As.
        for (uint i = tid; i < BM * BK / kVec; i += kThreads) {
            const uint r  = i / (BK / kVec);
            const uint kq = (i % (BK / kVec)) * kVec;
            const float4 v = *reinterpret_cast<device const float4*>(A + (m0 + r) * d.K + k0 + kq);
            HPC_UNROLL
            for (uint c = 0; c < kVec; ++c)
                As[kq + c][r] = v[c];
        }
        load_tile<float, BK, BN, BN, kThreads>(&Bs[0][0], B + k0 * d.N + n0, d.N, tid);
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
        for (uint j = 0; j < TN; j += kVec)
            *reinterpret_cast<device float4*>(C + (m0 + ty * TM + i) * d.N + n0 + tx * TN + j) =
                float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
}

#define HPC_REG_TILE(name, BM, BN, BK, TM, TN)                                         \
    template [[host_name(name)]] kernel void gemm_reg_tile<BM, BN, BK, TM, TN>(        \
        device const float*, device const float*, device float*, constant Dims&, uint, uint2);
HPC_REG_TILE("gemm_reg_tile", 64, 64, 16, 8, 4)
HPC_REG_TILE("gemm_reg_tile_128x128_8x8", 128, 128, 16, 8, 8)
HPC_REG_TILE("gemm_reg_tile_64x64_4x4", 64, 64, 16, 4, 4)
HPC_REG_TILE("gemm_reg_tile_128x64_8x8", 128, 64, 16, 8, 8)
#undef HPC_REG_TILE

// ===========================================================================
// Level 3: simdgroup matrices. A simdgroup (32 threads) multiplies 8×8
// fragments with simdgroup_multiply_accumulate; each simdgroup owns an
// SM×SN block of C (SM/8 × SN/8 accumulator fragments) and loads A and B
// fragments straight from device memory, relying on the GPU's caches for
// reuse between simdgroups. A threadgroup is WM×WN simdgroups.
// ===========================================================================

/// One (WM·SM)×(WN·SN) tile of C, an SM×SN block per simdgroup.
template <uint WM, uint WN, uint SM, uint SN>
kernel void gemm_simd(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                      device float* C [[buffer(2)]], constant Dims& d [[buffer(3)]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint2 tg [[threadgroup_position_in_grid]]) {
    const uint m0 = tg.y * (WM * SM) + (sg / WN) * SM;  // this simdgroup's block
    const uint n0 = tg.x * (WN * SN) + (sg % WN) * SN;

    simdgroup_float8x8 acc[SM / kFrag][SN / kFrag];
    zero_acc(acc);
    for (uint k = 0; k < d.K; k += kFrag)
        simd_mma<float>(acc, A + m0 * d.K + k, d.K, B + k * d.N + n0, d.N);
    store_acc(acc, C + m0 * d.N + n0, d.N);
}

#define HPC_SIMD(name, WM, WN, SM, SN)                                                 \
    template [[host_name(name)]] kernel void gemm_simd<WM, WN, SM, SN>(                \
        device const float*, device const float*, device float*, constant Dims&, uint, uint2);
HPC_SIMD("gemm_simd", 2, 2, 32, 32)
HPC_SIMD("gemm_simd_2x2_16x32", 2, 2, 16, 32)
HPC_SIMD("gemm_simd_2x2_32x16", 2, 2, 32, 16)
#undef HPC_SIMD

// ===========================================================================
// Levels 4 and 5: Level 3 with BM×BK and BK×BN tiles of A and B staged in
// threadgroup memory first, so every element is read from device memory
// once per threadgroup instead of once per simdgroup. T is the input type
// (float, or half for Level 5); accumulation is always in float.
//
// Each tile row is padded by kPad elements (16 bytes), which shifts
// consecutive rows across threadgroup-memory banks for the column-strided
// fragment loads.
// ===========================================================================

/// Level 3's simdgroup blocks, fed from threadgroup-memory tiles of A and B.
template <typename T, uint WM, uint WN, uint SM, uint SN, uint BK>
kernel void gemm_simd_tg(device const T* A [[buffer(0)]], device const T* B [[buffer(1)]],
                         device float* C [[buffer(2)]], constant Dims& d [[buffer(3)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint2 tg [[threadgroup_position_in_grid]]) {
    constexpr uint BM = WM * SM, BN = WN * SN;
    constexpr uint kThreads = WM * WN * 32;
    constexpr uint kPad = 16 / sizeof(T);
    threadgroup T As[BM][BK + kPad];
    threadgroup T Bs[BK][BN + kPad];

    const uint sm = (sg / WN) * SM;  // this simdgroup's block within the tile
    const uint sn = (sg % WN) * SN;
    A += tg.y * BM * d.K;
    B += tg.x * BN;
    C += tg.y * BM * d.N + tg.x * BN;

    simdgroup_float8x8 acc[SM / kFrag][SN / kFrag];
    zero_acc(acc);
    for (uint k0 = 0; k0 < d.K; k0 += BK) {
        load_tile<T, BM, BK, BK + kPad, kThreads>(&As[0][0], A + k0, d.K, tid);
        load_tile<T, BK, BN, BN + kPad, kThreads>(&Bs[0][0], B + k0 * d.N, d.N, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        HPC_UNROLL
        for (uint kk = 0; kk < BK; kk += kFrag)
            simd_mma<T>(acc, &As[sm][kk], BK + kPad, &Bs[kk][sn], BN + kPad);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    store_acc(acc, C + sm * d.N + sn, d.N);
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

// ===========================================================================
// FMA peak: every thread runs kChains independent FMA chains of kVec-wide
// vectors, so the ALUs never wait on a result. Throughput = threads ×
// iters × kChains × kVec lanes × 2 flops / GPU time. Seeded from `seed` and
// written out so the compiler cannot fold or drop the loop.
// ===========================================================================

/// `iters` rounds of kChains independent vector FMAs; writes their sum to out[gid].
template <typename T>
kernel void fma_peak(device T* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                     constant float& seed [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    using vec4             = vec<T, kVec>;
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
