"""Single-thread float32 1024x1024 matmul through numpy's BLAS; prints GFLOP/s.

Usage: OPENBLAS_NUM_THREADS=1 python matmul.py "<label>"
threadpoolctl (optional) reports which BLAS and kernel ("architecture") ran.
Kept exactly as used during the investigation.
"""
import os, sys, time, numpy as np
n = 1024
rng = np.random.default_rng(0)
A = rng.standard_normal((n, n), dtype=np.float32); B = rng.standard_normal((n, n), dtype=np.float32)
for _ in range(3): A @ B
t = []
for _ in range(20):
    t0 = time.perf_counter(); A @ B; t.append(time.perf_counter() - t0)
try:
    import threadpoolctl
    blas = [(d["internal_api"], d.get("architecture"), d.get("num_threads")) for d in threadpoolctl.threadpool_info()]
except ImportError:
    blas = "threadpoolctl n/a"
med = float(np.median(t))
print(f"{sys.argv[1]:42s} median {med*1e3:7.2f} ms  {2*n**3/med/1e9:7.1f} GFLOP/s  blas={blas or 'accelerate (not reported)'}")
