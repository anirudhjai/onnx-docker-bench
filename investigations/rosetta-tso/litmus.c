// Message-passing litmus test. Writer stores data=k then flag=k; reader loads
// flag then data. Seeing data < flag means a store-store or load-load reordering,
// which ARM's weak memory model allows and TSO forbids. Relaxed atomics compile to
// plain ldr/str; signal fences only stop the compiler from reordering.
// "flag changes seen" counts how often the reader observed the writer make progress:
// a large number shows the two threads really ran concurrently, so zero reorderings
// is meaningful rather than an artifact of the threads taking turns on one CPU.
// Also times scalar 64-bit loads from L1 (to compare with NEON loads).
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define ITERS 200000000L
static _Atomic long data_ __attribute__((aligned(128)));
static _Atomic long flag_ __attribute__((aligned(128)));
static _Atomic int done;

static void *writer(void *arg) {
    (void)arg;
    for (long k = 1; k <= ITERS; k++) {
        atomic_store_explicit(&data_, k, memory_order_relaxed);
        atomic_signal_fence(memory_order_seq_cst);
        atomic_store_explicit(&flag_, k, memory_order_relaxed);
    }
    atomic_store_explicit(&done, 1, memory_order_relaxed);
    return NULL;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t ibuf[2048] __attribute__((aligned(64))); /* 16 KiB */

int main(void) {
    for (int i = 0; i < 2048; i++) ibuf[i] = i;
    long reps = 400000;
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    double t = now();
    for (long r = 0; r < reps; r++) {
        for (int i = 0; i < 2048; i += 4) { s0 += ibuf[i]; s1 += ibuf[i + 1]; s2 += ibuf[i + 2]; s3 += ibuf[i + 3]; }
        __asm__ volatile("" ::: "memory");
    }
    t = now() - t;
    printf("scalar_load_l1 %6.3f s  %6.1f GB/s  (sum %llu)\n", t, reps * (double)sizeof ibuf / t / 1e9,
           (unsigned long long)(s0 + s1 + s2 + s3));

    pthread_t w;
    pthread_create(&w, NULL, writer, NULL);
    long samples = 0, violations = 0, changes = 0, last = 0;
    while (!atomic_load_explicit(&done, memory_order_relaxed)) {
        long f = atomic_load_explicit(&flag_, memory_order_relaxed);
        atomic_signal_fence(memory_order_seq_cst);
        long d = atomic_load_explicit(&data_, memory_order_relaxed);
        samples++;
        if (d < f) violations++;
        if (f != last) { changes++; last = f; }
    }
    pthread_join(w, NULL);
    printf("MP litmus: %ld samples, %ld flag changes seen (concurrency), %ld reorderings (data < flag)\n", samples, changes, violations);
    return 0;
}
