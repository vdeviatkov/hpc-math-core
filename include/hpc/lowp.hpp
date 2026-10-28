#pragma once

/**
 * @file lowp.hpp
 * @brief Storage types for low-precision GEMM inputs: bf16, fp16 and FP8 E4M3.
 *
 * Each is a bit pattern (16 or 8 bits) in a struct, so Matrix<bf16>,
 * Matrix<fp16> and Matrix<fp8_e4m3> work on every compiler and target; the kernels reinterpret
 * them as the hardware's own types (__bf16, __fp16, __nv_bfloat16, …).
 * INT8 inputs are plain std::int8_t.
 *
 *   bf16  bfloat16: 1 sign, 8 exponent, 7 mantissa bits. FP32's range,
 *         ~2–3 significant decimal digits.
 *   fp16  IEEE binary16: 1 sign, 5 exponent, 10 mantissa bits. Largest
 *         finite value 65,504; ~3–4 significant decimal digits.
 *   fp8_e4m3  OCP FP8 E4M3: 1 sign, 4 exponent, 3 mantissa bits, no
 *         infinities. Largest finite value 448; ~1 significant digit.
 *         Conversions saturate to ±448, as the GPUs' saturating converts do.
 *
 * Conversions from float round to nearest, ties to even, the rounding the
 * hardware conversion instructions use, so inputs converted here match
 * inputs converted on the device bit for bit.
 */

#include "hpc/matrix.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace hpc {

struct bf16 {
    std::uint16_t bits;
};
struct fp16 {
    std::uint16_t bits;
};
struct fp8_e4m3 {
    std::uint8_t bits;
};

namespace lowp_detail {
inline std::uint32_t bits_of(float x) noexcept {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}
inline float float_of(std::uint32_t u) noexcept {
    float x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}
}  // namespace lowp_detail

/// float → bfloat16, round to nearest even; NaN stays NaN.
inline bf16 to_bf16(float x) noexcept {
    const std::uint32_t u = lowp_detail::bits_of(x);
    if ((u & 0x7fffffffu) > 0x7f800000u)  // NaN: keep it quiet and non-zero
        return bf16{static_cast<std::uint16_t>((u >> 16) | 0x0040u)};
    const std::uint32_t rounding = 0x7fffu + ((u >> 16) & 1u);
    return bf16{static_cast<std::uint16_t>((u + rounding) >> 16)};
}

inline float to_float(bf16 h) noexcept {
    return lowp_detail::float_of(static_cast<std::uint32_t>(h.bits) << 16);
}

/// float → IEEE binary16, round to nearest even, with subnormals, overflow
/// to infinity and NaN.
inline fp16 to_fp16(float x) noexcept {
    const std::uint32_t u    = lowp_detail::bits_of(x);
    const std::uint32_t sign = (u >> 16) & 0x8000u;
    const std::uint32_t mag  = u & 0x7fffffffu;
    if (mag > 0x7f800000u)  // NaN
        return fp16{static_cast<std::uint16_t>(sign | 0x7e00u)};
    if (mag >= 0x477ff000u)  // ≥ 65,520 rounds to infinity
        return fp16{static_cast<std::uint16_t>(sign | 0x7c00u)};
    if (mag < 0x38800000u) {  // below 2^-14: subnormal or zero in fp16
        if (mag < 0x33000000u)  // below half of the smallest subnormal
            return fp16{static_cast<std::uint16_t>(sign)};
        const std::uint32_t exp      = mag >> 23;
        const std::uint32_t mant     = (mag & 0x7fffffu) | 0x800000u;
        const std::uint32_t shift    = 126u - exp;  // 14 … 24
        const std::uint32_t half     = 1u << (shift - 1);
        const std::uint32_t rem      = mant & ((1u << shift) - 1);
        std::uint32_t r              = mant >> shift;
        if (rem > half || (rem == half && (r & 1u)))
            ++r;
        return fp16{static_cast<std::uint16_t>(sign | r)};
    }
    // Normal: rebias the exponent (127 → 15) and round the mantissa 23 → 10 bits.
    const std::uint32_t rounded = mag + 0xfffu + ((mag >> 13) & 1u);
    return fp16{static_cast<std::uint16_t>(sign | ((rounded - 0x38000000u) >> 13))};
}

inline float to_float(fp16 h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h.bits & 0x8000u) << 16;
    const std::uint32_t exp  = (h.bits >> 10) & 0x1fu;
    const std::uint32_t mant = h.bits & 0x3ffu;
    if (exp == 0x1fu)  // infinity or NaN
        return lowp_detail::float_of(sign | 0x7f800000u | (mant << 13));
    if (exp == 0) {  // zero or subnormal: mant · 2^-24
        const float v = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -v : v;
    }
    return lowp_detail::float_of(sign | ((exp + 112u) << 23) | (mant << 13));
}

/// float → FP8 E4M3, round to nearest even, saturating to ±448; NaN → 0x7f.
inline fp8_e4m3 to_fp8_e4m3(float x) noexcept {
    const std::uint32_t u    = lowp_detail::bits_of(x);
    const std::uint8_t sign  = static_cast<std::uint8_t>((u >> 24) & 0x80u);
    const std::uint32_t mag  = u & 0x7fffffffu;
    if (mag > 0x7f800000u)
        return fp8_e4m3{static_cast<std::uint8_t>(sign | 0x7fu)};
    const float a = lowp_detail::float_of(mag);
    if (a >= 464.0f)  // halfway between 448 and the next (non-existent) step: saturate
        return fp8_e4m3{static_cast<std::uint8_t>(sign | 0x7eu)};
    if (a < 0x1p-6f) {  // subnormal range: multiples of 2^-9
        const float q = std::nearbyint(a * 512.0f);  // ties to even under the default mode
        return fp8_e4m3{static_cast<std::uint8_t>(sign | static_cast<std::uint8_t>(q))};
    }
    // Normal: exponent 1 … 15 (bias 7), 3 mantissa bits; round 23 → 3 bits.
    const std::uint32_t rounded = mag + 0x7ffffu + ((mag >> 20) & 1u);
    std::uint32_t e             = (rounded >> 23) - 120u;  // rebias 127 → 7
    std::uint32_t m             = (rounded >> 20) & 7u;
    if (e > 15 || (e == 15 && m == 7))  // above 448 after rounding
        e = 15, m = 6;
    return fp8_e4m3{static_cast<std::uint8_t>(sign | (e << 3) | m)};
}

inline float to_float(fp8_e4m3 h) noexcept {
    const std::uint32_t e = (h.bits >> 3) & 0xfu, m = h.bits & 7u;
    const bool neg        = (h.bits & 0x80u) != 0;
    if (e == 15 && m == 7)
        return lowp_detail::float_of(0x7fc00000u);  // NaN
    const float v = e == 0 ? std::ldexp(static_cast<float>(m), -9)
                           : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, static_cast<int>(e) - 7);
    return neg ? -v : v;
}

inline float to_float(std::int8_t v) noexcept { return static_cast<float>(v); }

/// Element-wise conversion of a float matrix to bf16, fp16 or fp8_e4m3.
template <typename L>
Matrix<L> convert(const Matrix<float>& A) {
    Matrix<L> out(A.rows(), A.cols());
    for (std::size_t i = 0; i < A.size(); ++i) {
        if constexpr (std::is_same_v<L, bf16>)
            out.data()[i] = to_bf16(A.data()[i]);
        else if constexpr (std::is_same_v<L, fp16>)
            out.data()[i] = to_fp16(A.data()[i]);
        else
            out.data()[i] = to_fp8_e4m3(A.data()[i]);
    }
    return out;
}

}  // namespace hpc
