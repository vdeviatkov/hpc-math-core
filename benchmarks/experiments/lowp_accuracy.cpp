/**
 * @file lowp_accuracy.cpp
 * @brief What each low-precision format costs in accuracy, for inputs that
 *        start as FP32.
 *
 *   ./build/benchmarks/lowp_accuracy > lowp_accuracy.csv
 *
 * A (M x K) and B (K x N) are FP32, uniform in [-1, 1) or standard normal.
 * The exact product (in double, from the FP32 inputs) is compared with the
 * product of the inputs rounded to each format, accumulated as the kernels
 * do. Reported: the relative Frobenius error ||C − C_exact|| / ||C_exact||.
 *
 *   fp32        FP32 inputs, FP32 accumulation in k order (the FP32 kernels)
 *   bf16, fp16  rounded inputs, exact products, FP32 accumulation
 *   fp16_acc16  rounded inputs, every partial sum rounded to fp16
 *               (FMLA .8h; mma.sync .f16 adds 16 products per instruction
 *               and presumably rounds less often, so this is the
 *               pessimistic end)
 *   fp8_e4m3    rounded inputs (values in [-1, 1) need no scaling), FP32
 *               accumulation
 *   int8        symmetric quantisation with one scale per row of A and per
 *               column of B (s = max|x| / 127, q = round(x / s)), exact
 *               INT32 accumulation, C = s_A · s_B · Σ q_A · q_B
 *
 * Input rounding dominates every format except fp16_acc16: the products of
 * rounded inputs are summed almost exactly, but the rounding of each input
 * (relative at most 2^-8 for bf16, 2^-11 for fp16, 2^-4 for E4M3) is
 * already done.
 */

#include "hpc/lowp.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using Mat = std::vector<float>;  // row-major

double rel_error(const std::vector<double>& c, const std::vector<double>& exact) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        num += (c[i] - exact[i]) * (c[i] - exact[i]);
        den += exact[i] * exact[i];
    }
    return std::sqrt(num / den);
}

template <typename Round>
Mat rounded(const Mat& x, Round r) {
    Mat out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        out[i] = r(x[i]);
    return out;
}

/// C = A · B with exact products and FP32 (or, with Acc16, fp16) partial sums in k order.
template <bool Acc16 = false>
std::vector<double> multiply(const Mat& A, const Mat& B, int M, int K, int N) {
    std::vector<double> C(static_cast<std::size_t>(M) * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];  // rounded inputs: two ≤11-bit significands multiply exactly in FP32
                if constexpr (Acc16)
                    acc = hpc::to_float(hpc::to_fp16(acc));
            }
            C[i * N + j] = acc;
        }
    return C;
}

std::vector<double> multiply_int8(const Mat& A, const Mat& B, int M, int K, int N) {
    std::vector<float> sa(M), sb(N);
    std::vector<int> qa(A.size()), qb(B.size());
    for (int i = 0; i < M; ++i) {
        float m = 0;
        for (int k = 0; k < K; ++k)
            m = std::max(m, std::fabs(A[i * K + k]));
        sa[i] = m / 127;
        for (int k = 0; k < K; ++k)
            qa[i * K + k] = static_cast<int>(std::nearbyint(A[i * K + k] / sa[i]));
    }
    for (int j = 0; j < N; ++j) {
        float m = 0;
        for (int k = 0; k < K; ++k)
            m = std::max(m, std::fabs(B[k * N + j]));
        sb[j] = m / 127;
        for (int k = 0; k < K; ++k)
            qb[k * N + j] = static_cast<int>(std::nearbyint(B[k * N + j] / sb[j]));
    }
    std::vector<double> C(static_cast<std::size_t>(M) * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            long long acc = 0;
            for (int k = 0; k < K; ++k)
                acc += static_cast<long long>(qa[i * K + k]) * qb[k * N + j];
            C[i * N + j] = double(sa[i]) * double(sb[j]) * double(acc);
        }
    return C;
}

}  // namespace

int main() {
    constexpr int M = 64, N = 64;
    std::printf("distribution,K,format,rel_frobenius_error\n");
    for (const bool normal : {false, true})
        for (const int K : {256, 1024, 4096, 16384}) {
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> uni(-1.f, 1.f);
            std::normal_distribution<float> gauss(0.f, 1.f);
            Mat A(static_cast<std::size_t>(M) * K), B(static_cast<std::size_t>(K) * N);
            for (auto& x : A)
                x = normal ? gauss(rng) : uni(rng);
            for (auto& x : B)
                x = normal ? gauss(rng) : uni(rng);

            std::vector<double> exact(static_cast<std::size_t>(M) * N);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j) {
                    double acc = 0;
                    for (int k = 0; k < K; ++k)
                        acc += double(A[i * K + k]) * double(B[k * N + j]);
                    exact[i * N + j] = acc;
                }

            const auto bf  = [](float x) { return hpc::to_float(hpc::to_bf16(x)); };
            const auto h   = [](float x) { return hpc::to_float(hpc::to_fp16(x)); };
            const auto f8  = [](float x) { return hpc::to_float(hpc::to_fp8_e4m3(x)); };
            const char* d  = normal ? "normal" : "uniform";
            auto row = [&](const char* fmt, const std::vector<double>& c) {
                std::printf("%s,%d,%s,%.3g\n", d, K, fmt, rel_error(c, exact));
            };
            row("fp32", multiply(A, B, M, K, N));
            row("bf16", multiply(rounded(A, bf), rounded(B, bf), M, K, N));
            row("fp16", multiply(rounded(A, h), rounded(B, h), M, K, N));
            row("fp16_acc16", multiply<true>(rounded(A, h), rounded(B, h), M, K, N));
            row("fp8_e4m3", multiply(rounded(A, f8), rounded(B, f8), M, K, N));
            row("int8", multiply_int8(A, B, M, K, N));
        }
}
