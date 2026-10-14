// Single-thread microbenchmarks to split "slow FP/SIMD execution" from
// "slow memory access" (e.g. a VM-wide TSO memory model).
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define L1_FLOATS 4096 /* 16 KiB, fits in L1D */
static float buf[L1_FLOATS] __attribute__((aligned(64)));
static float buf2[L1_FLOATS] __attribute__((aligned(64)));

int main(void) {
    for (int i = 0; i < L1_FLOATS; i++) buf[i] = (float)(i % 7) * 0.001f;
    double t, sink = 0;

    // 1) Register-only NEON FMA: 16 independent chains, no memory traffic.
    {
        float32x4_t a = vdupq_n_f32(1.0000001f), b = vdupq_n_f32(0.9999999f);
        float32x4_t c[16];
        for (int j = 0; j < 16; j++) c[j] = vdupq_n_f32((float)j);
        long n = 100000000;
        t = now();
        for (long i = 0; i < n; i++)
            for (int j = 0; j < 16; j++) c[j] = vfmaq_f32(c[j], a, b);
        t = now() - t;
        for (int j = 0; j < 16; j++) sink += vgetq_lane_f32(c[j], 0);
        printf("fma_reg   %7.3f s  %6.1f GFLOP/s\n", t, n * 16.0 * 4 * 2 / t / 1e9);
    }
    // 2) NEON loads from L1 (sum), 8 accumulators.
    {
        long reps = 400000;
        float32x4_t acc[8];
        for (int j = 0; j < 8; j++) acc[j] = vdupq_n_f32(0);
        t = now();
        for (long r = 0; r < reps; r++)
            for (int i = 0; i < L1_FLOATS; i += 32)
                for (int j = 0; j < 8; j++) acc[j] = vaddq_f32(acc[j], vld1q_f32(buf + i + 4 * j));
        t = now() - t;
        for (int j = 0; j < 8; j++) sink += vgetq_lane_f32(acc[j], 0);
        printf("load_l1   %7.3f s  %6.1f GB/s\n", t, reps * (double)sizeof buf / t / 1e9);
    }
    // 3) NEON stores into L1.
    {
        long reps = 400000;
        float32x4_t v = vdupq_n_f32(1.5f);
        t = now();
        for (long r = 0; r < reps; r++) {
            v = vaddq_f32(v, vdupq_n_f32(1e-9f));
            for (int i = 0; i < L1_FLOATS; i += 4) vst1q_f32(buf2 + i, v);
            __asm__ volatile("" ::: "memory");
        }
        t = now() - t;
        sink += buf2[123];
        printf("store_l1  %7.3f s  %6.1f GB/s\n", t, reps * (double)sizeof buf2 / t / 1e9);
    }
    // 4) L1 copy (load + store), like GEMM packing.
    {
        long reps = 400000;
        t = now();
        for (long r = 0; r < reps; r++) {
            for (int i = 0; i < L1_FLOATS; i += 4) vst1q_f32(buf2 + i, vld1q_f32(buf + i));
            __asm__ volatile("" ::: "memory");
        }
        t = now() - t;
        sink += buf2[77];
        printf("copy_l1   %7.3f s  %6.1f GB/s\n", t, reps * (double)sizeof buf / t / 1e9);
    }
    // 5) Scalar integer work, no memory (control).
    {
        uint64_t x = 88172645463325252ull;
        long n = 1000000000;
        t = now();
        for (long i = 0; i < n; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; }
        t = now() - t;
        sink += (double)(x & 1);
        printf("int_reg   %7.3f s  %6.2f Gops/s\n", t, n / t / 1e9);
    }
    printf("(sink %g)\n", sink);
    return 0;
}
