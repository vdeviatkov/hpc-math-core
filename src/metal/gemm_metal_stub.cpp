// Metal API for builds without Metal (non-Apple): metal_available() is
// false, so the tests and benchmarks report SKIPPED; everything else throws.

#include "gemm/metal.hpp"

#include <stdexcept>

namespace hpc::gemm {

namespace {
[[noreturn]] void no_metal() { throw std::runtime_error("Metal not available in this build"); }
}  // namespace

bool metal_available() noexcept { return false; }
std::string metal_device_name() { return {}; }
std::vector<std::string> metal_kernel_variants() { return {}; }

struct MetalGemm::Impl {};
MetalGemm::MetalGemm(std::size_t, std::size_t, std::size_t) { no_metal(); }
MetalGemm::~MetalGemm() = default;
void MetalGemm::set_inputs(const float*, const float*) { no_metal(); }
void MetalGemm::get_output(float*) const { no_metal(); }
double MetalGemm::run_ms(MetalKernel, int) { no_metal(); }
double MetalGemm::run_variant_ms(const std::string&, int) { no_metal(); }
MetalKernel MetalGemm::effective(MetalKernel kernel) const noexcept { return kernel; }

void gemm_metal(MetalKernel, const Matrix<float>&, const Matrix<float>&, Matrix<float>&) { no_metal(); }
double metal_fma_peak_gflops(bool) { no_metal(); }

}  // namespace hpc::gemm
