#include "gemm/simd.hpp"
#include "gemm/avx512.hpp"
namespace hpc::gemm {
#define I(R,V) template void vector_micro<Zmm<float>, R, V, float>(const float*, const float*, float*, std::size_t, std::size_t, std::size_t, std::size_t) noexcept;
I(4,2) I(2,4) I(4,4) I(6,2) I(8,2) I(6,3) I(8,3) I(12,2) I(14,2) I(16,1)
}
