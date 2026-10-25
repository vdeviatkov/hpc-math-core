#pragma once

/**
 * @file packed_driver.hpp
 * @brief The packed-B GEMM driver behind gemm_simd_packed (simd.hpp).
 *
 * Same micro-kernels as blocked_driver.hpp, different loop order and B
 * layout. GotoBLAS order: each Kc × Nc block of B is copied once into
 * contiguous strips NR = V·W columns wide, and that copy serves every row
 * of A:
 *
 *   for jc (Nc columns of B and C, rounded down to whole strips)
 *     for pc (Kc steps of k)
 *       pack B(pc.., jc..) into strips: strip s holds kc rows of NR values
 *       for ic (Mc rows of A and C)
 *         [PackA: copy A(ic.., pc..) into an mc × kc row-major buffer]
 *         for jr (one NR-column strip, read from L1)
 *           for ir (R rows at a time)
 *             Micro::run<R, V>: C(R × NR) += A(R × kc) · strip(kc × NR)
 *
 * Why pack B at all for FP32: row-major B already has the layout the
 * micro-kernel wants (row k of a strip is V contiguous vectors), so packing
 * is not needed for the instruction, as it is for the dot-product kernels
 * (avx512_lowp.hpp). It is for the caches: in place, row k and row k+1 of
 * a strip are N elements apart, and at N·sizeof(T) a multiple of 4 KiB all
 * kc rows fall in the same L1 set (and few L2 sets), so the strip is evicted
 * long before its reuse (docs/articles/sme2-cache-associativity.md). Packed,
 * the strip is kc·NR contiguous elements and uses every set.
 *
 * PackA does the same for A: in place, the R rows a micro-kernel streams
 * are K elements apart (16 KiB at N = 4096 FP32), so they share one L1 set
 * too. Packed, they are kc elements apart.
 *
 * The strip is zero-padded to NR columns, but a micro-kernel never computes
 * padding: the last strip of a block runs row_strip with its true width
 * (fewer vectors, then scalar columns), as blocked_gemm's edges do. Packing
 * is inside the call and is timed.
 */

#include "gemm/blocked_driver.hpp"
#include "hpc/matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>

namespace hpc::gemm::detail {

/// Copy kc × nc of B (row stride ldb) into NR-column strips, each kc rows of
/// NR contiguous elements; columns past nc are zero.
template <std::size_t NR, typename T>
inline void pack_b_strips(const T* B, std::size_t ldb, std::size_t kc, std::size_t nc, T* dst) noexcept {
    for (std::size_t j0 = 0; j0 < nc; j0 += NR) {
        const std::size_t cols = std::min(NR, nc - j0);
        T* strip               = dst + j0 * kc;
        for (std::size_t k = 0; k < kc; ++k) {
            std::memcpy(strip + k * NR, B + k * ldb + j0, cols * sizeof(T));
            if (cols < NR)
                std::memset(strip + k * NR + cols, 0, (NR - cols) * sizeof(T));
        }
    }
}

/// C (=) A · B with packed B (and A if PackA). See the file header.
template <typename Micro, int R, int V, std::size_t Mc, std::size_t Kc, std::size_t Nc, bool PackA,
          typename T>
void packed_gemm(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    static_assert(R >= 1 && V >= 1, "the register tile needs at least one row and one vector");
    constexpr std::size_t NR  = V * Micro::template kLanes<T>;
    constexpr std::size_t Ncs = Nc / NR * NR;  // whole strips (an 8 × 3 AVX-512 strip is 48 floats wide)
    static_assert(Ncs > 0, "Nc must hold at least one strip");
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    C.zero();
    if (M == 0 || N == 0 || K == 0)
        return;

    // Matrix gives a 64-byte aligned, zeroed buffer; one row is enough.
    Matrix<T> b_pack(1, std::min(Kc, K) * std::min(Ncs, (N + NR - 1) / NR * NR));
    Matrix<T> a_pack(PackA ? 1 : 0, std::min(Mc, M) * std::min(Kc, K));

    for (std::size_t jc = 0; jc < N; jc += Ncs) {
        const std::size_t nc = std::min(Ncs, N - jc);
        for (std::size_t pc = 0; pc < K; pc += Kc) {
            const std::size_t kc = std::min(Kc, K - pc);
            pack_b_strips<NR>(B.data() + pc * N + jc, N, kc, nc, b_pack.data());
            for (std::size_t ic = 0; ic < M; ic += Mc) {
                const std::size_t mc = std::min(Mc, M - ic);
                const T* a           = A.data() + ic * K + pc;
                std::size_t lda      = K;
                if constexpr (PackA) {
                    for (std::size_t i = 0; i < mc; ++i)
                        std::memcpy(a_pack.data() + i * kc, a + i * K, kc * sizeof(T));
                    a   = a_pack.data();
                    lda = kc;
                }
                for (std::size_t jr = 0; jr < nc; jr += NR) {
                    const T* strip = b_pack.data() + jr * kc;
                    T* c           = C.data() + ic * N + jc + jr;
                    const std::size_t cols = std::min(NR, nc - jr);
                    for (std::size_t ir = 0; ir < mc; ir += R) {
                        const int rows = static_cast<int>(std::min<std::size_t>(R, mc - ir));
                        row_strip<Micro, R, V>(rows, a + ir * lda, strip, c + ir * N, cols, lda, NR, N, kc);
                    }
                }
            }
        }
    }
}

}  // namespace hpc::gemm::detail
