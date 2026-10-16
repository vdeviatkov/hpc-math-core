/**
 * Review-only harness: export allocation/setup and untimed kernel calls for
 * two namespace-isolated dylibs compiled from before/after header snapshots.
 * setup runs outside timing. macro_step allows both versions to read the
 * same allocations, separating address-layout effects from kernel code.
 */
#include "gemm/amx.hpp"
#include "gemm/blocked.hpp"
#include "gemm/neon.hpp"
#include "gemm/prefetch.hpp"
#include "gemm/sme.hpp"

#include <memory>
using namespace hpc;
using namespace hpc::gemm;
template <class T>
struct Problem {
    Matrix<T> a, b, c;
    Problem(int n) : a(n, n), b(n, n), c(n, n) {
        for (size_t i = 0; i < a.size(); ++i) {
            a.data()[i] = T(int(i % 17) - 8) / T(17);
            b.data()[i] = T(int(i % 23) - 11) / T(23);
        }
    }
};
static std::unique_ptr<Problem<float>> problem_f;
static std::unique_ptr<Problem<double>> problem_d;
extern "C" void setup(int bits, int n) {
    if (bits == 32)
        problem_f = std::make_unique<Problem<float>>(n);
    else
        problem_d = std::make_unique<Problem<double>>(n);
}
template <class T>
void run(int id, Problem<T>& p) {
    switch (id) {
        case 0:
            gemm_blocked(p.a, p.b, p.c);
            break;
        case 1:
            gemm_neon_blocked(p.a, p.b, p.c);
            break;
        case 2:
            gemm_blocked_prefetch(p.a, p.b, p.c);
            break;
        case 3:
            gemm_neon_blocked_prefetch(p.a, p.b, p.c);
            break;
        case 4:
            gemm_sme_1tile(p.a, p.b, p.c);
            break;
        case 5:
            gemm_sme_tiles(p.a, p.b, p.c);
            break;
        case 6:
            gemm_sme_blocked(p.a, p.b, p.c);
            break;
        case 7:
            gemm_sme(p.a, p.b, p.c);
            break;
        case 8:
            gemm_amx(p.a, p.b, p.c);
            break;
    }
}
extern "C" void step(int id, int bits) {
    if (bits == 32)
        run(id, *problem_f);
    else
        run(id, *problem_d);
}
extern "C" void macro_step(int single, int bits, void* ap, void* b, void* c, int n) {
    if (bits == 64) {
        if (single)
            sme_detail::one_tile_kernel<double>((double*)ap, (double*)b, (double*)c, n, n, n);
        else
            sme_detail::macro_kernel<double>((double*)ap, (double*)b, 1, n, (double*)c, n, 512, n,
                                             n, false);
    } else {
        if (single)
            sme_detail::one_tile_kernel<float>((float*)ap, (float*)b, (float*)c, n, n, n);
        else
            sme_detail::macro_kernel<float>((float*)ap, (float*)b, 1, n, (float*)c, n, 1024, n, n,
                                            false);
    }
}
