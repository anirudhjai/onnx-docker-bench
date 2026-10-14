# Docker Desktop + Rosetta: why ORT ran ~6.6x slower in containers on an M1 Mac

**Result:** with Docker Desktop's "Use Rosetta for x86_64/amd64 emulation on
Apple Silicon" enabled (the default), native arm64 containers on this M1 ran
ResNet-50 ~6.6x slower than macOS at the same thread count. Inside the VM, memory
loads were 11–18x slower and memory accesses were TSO-ordered (no reorderings
observable). Turning the setting off restored ARM ordering and host-speed loads,
OpenBLAS and ORT.

Machine: MacBook Pro 13" M1 (MacBookPro17,1), Docker Desktop 29.6.1 (8 vCPU,
3.8 GiB VM, kernel 6.12.76-linuxkit), ORT 1.31.0, Python 3.12. Compilers: Apple
clang 21.0.0 on the host, Debian clang 19.1.7 in the container.

## Tests

| File | Measures | Why |
|---|---|---|
| `microbench.c` | Single thread: `fma_reg` NEON FMA in registers (no memory); `load_l1` NEON loads from a 16 KiB L1-resident buffer; `store_l1` NEON stores to L1; `copy_l1` load+store (like GEMM packing); `int_reg` scalar integer work | Splits "slow FP/SIMD execution" from "slow memory access" |
| `litmus.c` | `scalar_load_l1` (64-bit loads from L1), then a message-passing litmus test: writer stores `data=k` then `flag=k`; reader loads `flag` then `data` and counts `data < flag` | ARM's weak model allows that reordering; TSO forbids it. "flag changes seen" proves the threads ran concurrently |
| `matmul.py` | numpy 1024x1024 float32 matmul, single thread, GFLOP/s, with the BLAS kernel reported by threadpoolctl | Is the slowdown ORT-specific or any SIMD-heavy GEMM? |
| `../../bench/run.py` | ORT ResNet-50 latency | The original symptom |

## How to run

In a container (from the repo root, after `docker build -t onnx-docker-bench:latest .`):

```bash
docker run --rm -e OPENBLAS_NUM_THREADS=1 -v "$PWD/investigations/rosetta-tso:/w:ro" --entrypoint sh onnx-docker-bench:latest -c '
  apt-get update -qq && apt-get install -y -qq --no-install-recommends clang >/dev/null
  pip install -q threadpoolctl==3.6.0
  clang -O2 -o /tmp/microbench /w/microbench.c && /tmp/microbench
  clang -O2 -pthread -o /tmp/litmus /w/litmus.c && /tmp/litmus
  python /w/matmul.py "container / OpenBLAS"'
```

On the macOS host:

```bash
clang -O2 -o /tmp/microbench investigations/rosetta-tso/microbench.c && /tmp/microbench
clang -O2 -pthread -o /tmp/litmus investigations/rosetta-tso/litmus.c && /tmp/litmus
```

For the host matmul, use numpy's OpenBLAS build: the default macOS wheel uses
Apple Accelerate (AMX coprocessor, ~1.1 TFLOP/s), which is not comparable.

```bash
python3.12 -m venv /tmp/venv-openblas
/tmp/venv-openblas/bin/pip download numpy==2.4.6 --platform macosx_11_0_arm64 --python-version 3.12 --only-binary=:all: --no-deps -d /tmp/wheels
/tmp/venv-openblas/bin/pip install /tmp/wheels/numpy-2.4.6-cp312-cp312-macosx_11_0_arm64.whl threadpoolctl==3.6.0
OPENBLAS_NUM_THREADS=1 /tmp/venv-openblas/bin/python investigations/rosetta-tso/matmul.py "host / OpenBLAS"
```

ORT: `bench/run.py --intra-threads 1 --warmup 5 --iters 30` (container without
`--cpus`) and `--intra-threads 2` with defaults (container with `--cpus=2`).

## Before / after

"Before" = Rosetta setting on (Docker Desktop default); "after" = off, VM
restarted. Host = macOS native, re-measured alongside "after".

| Test | Before (Rosetta on) | After (Rosetta off) | Host |
|---|---:|---:|---:|
| Rosetta binfmt handlers in VM | `rosetta`, `rosetta-wrapper` | none | n/a |
| MP litmus: reorderings / samples | **0 / ~860M** (5 runs) | **~1.0–1.1B / ~2.0–2.2B per run** (~50%, 5 runs) | 104M / 315M (~33%) |
| `fma_reg` (GFLOP/s) | 100.1 | 100.3 | 88.2 |
| `int_reg` (Gops/s) | 0.53 | 0.53 | 0.53 |
| `store_l1` (GB/s) | 49.6 | 49.5 | 49.4 |
| `load_l1` NEON (GB/s) | **7.2** | **131.6** | 130.1 |
| `scalar_load_l1` (GB/s) | **4.8–6.1** | **66.0–66.3** | 66.5 |
| `copy_l1` (GB/s) | **5.1** | **77.4** | 89.0 (55.2 in an earlier run; noisy) |
| OpenBLAS SGEMM, 1 thread (GFLOP/s) | **13.5** | **96.3** | 94.2 |
| ORT ResNet-50, 1 thread, no limit, p50 (ms) | **655.0** | **99.8** | 100.0 |
| ORT `--cpus=2 --intra-threads 2`, p50 / p99 (ms) | **339.1 / 344.0** | **53.5 / 56.6** | 55.9 / 57.2 (intra=2) |

Notes:

- The litmus program and parameters are identical before and after (writer does
  2x10^8 updates). The reader takes 10–14x more samples "after" because its loop
  is load-bound and loads got ~12x faster. "Before" threads were concurrent:
  150–160M flag changes seen per run.
- `fma_reg` and `copy_l1` differ between host and container partly because of
  different compilers; compare each column's before/after, not across columns.
- "Before" numbers were captured from terminal output during the investigation;
  raw "after" outputs are in `logs/`.

## Verdict and caveats

Turning the Rosetta setting off flipped two things together: the VM went from
TSO-like ordering (zero reorderings) to ARM ordering, and load throughput, GEMM
and ORT went from ~7–18x slow to host speed. Register-only FP, stores and integer
code were unaffected in both states. That is what the hypothesis predicted:
VM-wide TSO mode (reportedly enabled when Rosetta is mapped into an Apple
Virtualization VM) is what slows load-bound code.

What this does not prove: the setting toggles more than the memory model
(Rosetta registration, the Rosetta share, the hypervisor configuration). We
observed TSO and the slowdown appearing and disappearing together; we did not
toggle TSO alone.

The size is surprising. Published measurements of Apple's TSO mode on native
code are small: Khayet & Morrison (ASPLOS '26) report execution times typically
within 3% of Arm mode, and cite earlier averages of 8.94% (SPEC CPU 2017 FP) and
22% (Geekbench); a secondhand summary of earlier work mentions ~2x on synthetic
tests. Load throughput 11–18x lower is far outside that. Either VM-wide TSO as
set up for Rosetta behaves very differently from macOS's per-process TSO (at
least on the first-generation M1), or something else that the setting enables
contributes. We can't separate those here.

Practical consequence: on Apple Silicon, benchmark arm64 containers with the
Rosetta setting **off**, or results will mostly measure this effect. Turning it
off makes Docker Desktop emulate amd64 images with QEMU instead (slower for amd64
images).

Sources: [ASPLOS '26 paper](https://www.cs.tau.ac.il/~mad/publications/asplos2026-TSO.pdf),
[Rosetta-in-VM TSO note](https://threedots.ovh/blog/2022/06/quick-look-at-rosetta-on-linux/),
[Finch: always-on TSO in Rosetta VMs](https://github.com/runfinch/finch/issues/218),
[HN discussion of TSO cost](https://hn.svelte.dev/item/45216327).
