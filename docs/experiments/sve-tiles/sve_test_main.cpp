#include <arm_sve.h>
#include <sys/prctl.h>
#include <linux/prctl.h>
#include <cstdlib>
#include <cstdio>
#include <gtest/gtest.h>
[[gnu::noinline]] int vector_bits() { return int(svcntb() * 8); }
int main(int argc, char** argv) {
    const char* requested = std::getenv("SVE_TEST_BITS");
    const int bits = requested ? std::atoi(requested) : 128;
    if (prctl(PR_SVE_SET_VL, bits / 8, 0L, 0L, 0L) < 0) { std::perror("PR_SVE_SET_VL"); return 2; }
    const int actual = vector_bits();
    std::printf("Emulated SVE vector length: %d bits\n", actual);
    if (actual != bits) return 3;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
