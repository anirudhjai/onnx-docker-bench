# Mac, Docker Desktop with Rosetta on (phase 1)

`phase1.csv` holds the four phase 1 rows: a macOS host baseline and three
containers at `--cpus=2` (ORT default threads; `--intra-threads 2`; default
threads with `--spin off`).

**Use these as evidence for `investigations/rosetta-tso/`, not for the main
study.** The containers ran under Docker Desktop's default "Use Rosetta for
x86_64/amd64 emulation on Apple Silicon", which runs the Linux VM with TSO-like
memory ordering. That made memory loads 11–18x slower and ORT ~6.6x slower than
with the setting off, so these container latencies are inflated, and the ratios
between configs differ too (3.3x default-vs-matched here, ~7x with Rosetta off).

Conditions: MacBook Pro 13" M1, Docker Desktop 29.6.1 (VM: 8 vCPU, 3.8 GiB,
kernel 6.12.76-linuxkit), ORT 1.31.0, Python 3.12.0 (host) / 3.12.15
(container), ResNet-50 fp32 224, batch 1, 20 warmup + 300 timed iterations,
measured 2026-10-14 08:16–08:25 UTC. The model is the dynamo export still in use
(sha256 `e05907d9fd93…`; this CSV predates the `model_sha256` column).

The CSV uses the phase 1 column layout: no `run_id`, raw timings,
baseline/session thread counts, cgroup throttling, load average or model hash.
Its `process_threads` includes 7 idle OpenBLAS threads in the containers (fixed
later by capping `OPENBLAS_NUM_THREADS`).
