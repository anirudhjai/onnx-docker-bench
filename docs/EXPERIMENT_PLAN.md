# Phase 2 experiment plan

Status: draft for review, nothing here has been run. Prices were checked on
2026-10-14 (instances.vantage.sh, us-east-1 on-demand Linux) and should be
re-checked before launching.

## Questions

1. How much latency does ONNX Runtime's default intra-op thread count cost when
   a container is limited with `docker run --cpus`? ORT sizes its pool from the
   CPUs it can see, not from the cgroup quota (see "Findings so far").
2. How much of that cost is thread spinning versus oversubscription itself?
3. Does the answer change with int8 dynamic quantization, input size, and CPU
   architecture (arm64 vs x86_64)?

## Target machines

| | arm64 | x86_64 |
|---|---|---|
| Instance | **c8g.2xlarge** | **c8a.2xlarge** |
| CPU | AWS Graviton4 (Neoverse V2) | AMD EPYC 9R45 (Zen 5, "Turin") |
| vCPU | 8 (1 vCPU = 1 physical core) | 8 (1 vCPU = 1 physical core, no SMT) |
| Memory | 16 GiB | 16 GiB |
| On-demand | $0.319/h | $0.431/h |

Why these:

- **Native Linux.** On the Mac, Docker Desktop's VM makes ORT ~6.6x slower
  regardless of `--cpus` (finding 2), which would swamp the effect we want.
- **No SMT on either.** On both, `--cpus=N` is N physical cores' worth of time.
  Intel c7i/c8i expose 2 vCPUs per core, so `--cpus=4` would be 2 cores' worth
  of hyperthreads: a confound we'd have to explain in every chart.
- **8 vCPUs, not 4.** The largest quota (4) is then half the machine, so ORT's
  default pool (8) oversubscribes the quota in *every* cell, and dockerd/the OS
  have spare cores outside the container's quota.
- Same size and generation on both arches; compute-optimized, non-burstable
  (no T-family CPU credits). Fallbacks if capacity is short: c7g/c7a (c7a also
  has no SMT; $0.411/h).
- On-demand rather than spot: a ~14 h run shouldn't be interruptible. Spot would
  roughly halve an already small bill if the runner is made resumable.

Same OS image on both (Ubuntu 24.04 LTS), Docker Engine (not Desktop),
cgroup v2. The Dockerfile pins `python:3.12-slim` by its multi-arch index digest,
so both machines build from the same source.

## Matrix

Fixed: ResNet-50, batch 1, `--opt-level all`, ORT 1.31.0, sequential execution
mode, the pinned image.

| Factor | Levels |
|---|---|
| `--cpus` | 1, 2, 4 |
| `--intra-threads` | 0 (ORT default = 8 here), matched (= cpus) |
| `--spin` | default (= on; ORT logs `allow_spinning: 1`), off |
| model | fp32, int8 dynamic-quantized |
| input | 224x224, 448x448 |

3 x 2 x 2 x 2 x 2 = 48 cells. **Drop 4 redundant cells**: at `--cpus=1` the
matched setting is `intra=1`, which creates no worker threads, so `spin` cannot
matter (the pilot runs one of these once to confirm). **44 cells per machine.**

int8 model: `onnxruntime.quantization.quantize_dynamic(weight_type=QInt8)` on the
fp32 export. It needs a new `scripts/quantize_model.py`, which must first strip
the exporter's `value_info`; otherwise ORT's quantizer fails shape inference
(`Inferred shape and existing shape differ in dimension 0: (2048) vs (1000)`).
Mac probe (native macOS, scratch): int8 is 35.2 vs 99.1 ms at 1 thread and
16.3 vs 32.4 ms at 4 threads; top-1 agrees with fp32, max relative logit error
2.4%. The graph gains 50 `DynamicQuantizeLinear` nodes, so activation
quantization is part of the timed latency. Static (calibrated QDQ) quantization
would be faster but is a different experiment.

**Is 44 too many?** No, I'd keep all of it: ~14 h per machine (both machines in
parallel) for ~$10 total (see Runtime). If wall time matters more than coverage:

| Option | Cells | h/machine (R=5) | Trade-off |
|---|---:|---:|---|
| A. Full (recommended) | 44 | ~13.9 | — |
| B. Drop int8 @ 448 | 33 | ~10.3 | Lose "does the int8 speedup hold at larger inputs?" |
| C. Drop 448 entirely | 22 | ~3.7 | Lose "is the oversubscription penalty fixed overhead or proportional to work?" |

448 costs 10.1 of the ~14 hours, but it's also what distinguishes a per-operator
synchronisation cost from one that scales with work, so I'd cut it last.

## Repeats and iterations

**Unit of replication is the run (one container launch), not the iteration.**
Iterations within a run are not independent: under a CFS quota, latency follows
the 100 ms enforcement period (on the Mac, `--spin off` produced two clusters,
300–340 ms and 370–410 ms, and the p50 flipped between them across runs).
Thread placement and memory layout also change per launch. So per-iteration
confidence intervals understate uncertainty; spread across runs is the honest
error bar.

**Iterations per run: 1000 timed, 50 warmup.** A sample p99 rests on the few
samples above it. Distribution-free 95% CIs for the true p99 from n iid samples
(order statistics, B ~ Bin(n, 0.99)):

| n | samples above p99 | 95% CI for p99 spans ranks |
|---:|---:|---|
| 300 (phase 1) | ~3 | p97.7 .. max (no upper bound below the max) |
| 1000 | ~10 | p98.3 .. p99.7 |
| 5000 (5 runs pooled) | ~50 | p98.7 .. p99.3 |

At 300 iterations, "p99" is the 3rd–4th worst sample, which isn't meaningful.
At 1000 it is bounded on both sides.

**Cap for pathological cells:** if a 1000-iteration run would exceed ~10 min of
timed loop (estimated from a pilot), use `max(300, floor(600 s / latency))`. That
affects ~5 of 44 cells, all where ORT defaults meet 1–2 CPUs with spinning on.
Effects there are 3–6x, so the coarser p99 doesn't change conclusions. `iters` is
already recorded in every row.

**Repeats: R = 5 per cell, interleaved.** The [min, max] of R run-level values is
a distribution-free CI for their median with coverage 1 - 2*0.5^R: 75% for R=3,
**93.8% for R=5**, 96.9% for R=6 (+20% runtime). Pooled raw timings (5 x 1000)
give the tight pooled p99 above.

A difference counts as resolved when the [min, max] ranges across runs don't
overlap. Otherwise we report it as unresolved, like intra=2 vs spin-off on the
Mac.

## Runtime and cost

Planning assumptions (anchored on Mac native measurements where possible):

- fp32 224 single-thread: 120 ms (M1 native: 99 ms). 448 = 4x. int8 = 0.4x fp32 (M1: 0.36x).
- Matched-thread efficiency at 2/4 threads: fp32 0.85/0.75 (M1: 0.77 at 4), int8 0.75/0.55 (M1: 0.54 at 4).
- ORT default (8 threads) vs matched, spin on: 6x / 3.3x / 1.8x at 1/2/4 CPUs.
  Spin off: 1.3x / 1.1x / 1.05x. Only the 2-CPU figures were measured (Mac);
  the others are guesses and dominate the uncertainty.
- 15 s per run for container start, session creation and pilot.

| Plan | h/machine | c8g.2xlarge | c8a.2xlarge |
|---|---:|---:|---:|
| 44 cells, R=5, capped iterations (recommended) | ~13.9 | ~$4.4 | ~$6.0 |
| 44 cells, R=6, capped | ~16.6 | ~$5.3 | ~$7.2 |
| 44 cells, R=5, uncapped 1000 iterations | ~18.1 | ~$5.8 | ~$7.8 |

Run both machines in parallel: ~14 h wall clock, plus a ~1 h pilot each (1 run of
every cell at 100 iterations) to measure real latencies and recompute iteration
caps before the main run. EBS (30 GB gp3) and data transfer are cents. Even if the
assumptions are 2x optimistic, the total stays under ~$25.

## Controls

- **Idle host.** Fresh instances; nothing else running. Disable
  `unattended-upgrades` and snap refresh for the duration. Run the matrix in
  tmux, don't work on the box while it runs, and record `/proc/loadavg`
  before each run.
- **Warmup.** 50 iterations after session creation, plus one discarded burn-in
  run per machine (pulls the image and warms the page cache for the model file).
- **Run order.** Blocked randomization: each of the R rounds runs all 44 cells in
  a fresh random permutation (seed recorded). Drift (thermal state, neighbor load,
  time of day) then spreads across cells instead of lining up with one factor.
- **One container per run, sequentially**, same image ID on every run of a machine.
- **Recorded once per machine** (`results/<machine>/machine.json`): instance type,
  AMI, region/AZ (IMDSv2), `lscpu`, CPU flags, kernel, Docker version, cgroup
  version, THP setting, NUMA layout, memory, image digest and ID, ORT build info,
  and the repo's git commit.
- **Recorded per run** (already in the CSV): run_id, versions, os_cpu_count,
  affinity, cgroup quota, baseline/process/session-added threads, iters, all
  stats; raw per-iteration timings via `--save-raw`.

Proposed additions (need approval before implementing):

1. **cgroup `cpu.stat` deltas over the timed loop** (`usage_usec`, `nr_periods`,
   `nr_throttled`, `throttled_usec`). This is the direct mechanism evidence: on
   the Mac it showed whether a slowdown was throttling or slow execution.
2. **`/proc/loadavg` at run start**, as a per-row noise covariate.
3. **Model file sha256** per row, tying each row to the exact fp32/int8 file.
4. **`scripts/run_matrix.py`**: builds the randomized schedule, runs the pilot,
   launches containers, and is resumable (skips run_ids already present).

## Analysis

Charts (per machine, faceted by model x input; log latency axis):

1. **Latency vs `--cpus`**: median p50 per config with [min, max]-across-runs
   whiskers; one series per (intra, spin).
2. **Oversubscription penalty heatmap**: p50(intra=0) / p50(matched), by cpus x
   spin, one panel per arch x model x input.
3. **Tail ratio** p99/p50 per cell: where spinning and throttling hurt tails.
4. **ECDFs from raw timings** for the four (intra, spin) configs at `--cpus=2`,
   which shows quota-period clustering directly.
5. **Scaling efficiency** of matched threads (throughput vs cpus), fp32 vs int8, per arch.
6. **int8 speedup** (fp32 p50 / int8 p50) per cell, arm64 vs x86_64.

Tables:

- Per-cell summary: median [min–max] of p50/p95/p99, pooled p99, std,
  session_threads_added, cgroup_cpus, iters, runs.
- "What to set" table: best (intra, spin) per `--cpus` per arch and model, with
  the speedup over ORT defaults.
- Environment table from `machine.json`.
- (If cpu.stat is added) throttled-period fraction per cell.

## Findings so far (Mac, exploratory)

M1 MacBook Pro, Docker Desktop 29.6.1 (8 vCPU, 3.8 GiB VM), ORT 1.31.0, ResNet-50
fp32 224, batch 1. Data: `results/phase1.csv` plus scratch diagnostics.

**1. ORT's default thread pool ignores the container's CPU quota.** With
`--cpus=2`, the container sees 8 CPUs (`os.cpu_count()` and affinity), so ORT
starts 8 threads (7 workers + the caller) against 2 CPUs of quota:

| config (`--cpus=2`) | p50 ms (runs) | p99 ms (runs) |
|---|---|---|
| ORT defaults (8 threads, spin on) | 1113, 1107 | 1422, 1293 |
| `--intra-threads 2` | 339.1, 339.4, 339.2 | 344, 349, 350 |
| 8 threads, `--spin off` | 372, 376, 323 | 394, 395, 390 |

Matching threads to the quota is 3.3x faster at p50 and 4.1x at p99; disabling
spinning recovers about 3x. Run-to-run p50 spread is 0.1–0.5% for the stable
configs, far below these effects. intra=2 vs spin-off is **not** resolved at p50:
spin-off is bimodal and its p50 jumps between clusters (323–376 ms). It is
resolved in the tail (p99 344–350 vs 390–395 ms) and in std (1–3 vs ~33 ms).
Caveats: absolute numbers are inflated ~6.6x by finding 2; one machine; 2–3
runs per config; the 1- and 4-CPU behaviour is untested.

**2. Docker Desktop's VM makes ORT ~6.6x slower on this Mac, independent of
`--cpus`.** At 1 thread: 99.8 ms native vs 655 ms in a container with no CPU
limit. What we established:

- Same ORT commit on both sides, identical optimized graph, and ~all of the
  extra time is in Conv/FusedConv. Not denormals; ORT 1.22.1/1.24.4/1.30.0
  behave the same.
- Not ORT-specific: OpenBLAS single-thread SGEMM (same library version and
  kernel) is 95.7 vs 13.5 GFLOP/s (7.1x).
- CPU feature flags match (dotprod, fp16 and fhm on both; no i8mm, bf16, SVE or SME).
- Microbenchmarks: register-only NEON FMA, stores, and scalar integer code run
  at full speed in the VM, but **loads from L1 are 11–18x slower** (NEON loads
  131.5 vs 7.2 GB/s; scalar loads 64.7 vs ~5 GB/s). GEMM is load-throughput-bound.
- A message-passing litmus test sees ~33% reordered observations natively and
  **0 in ~860M samples inside the VM** (threads verified concurrent): the VM's
  vCPUs enforce TSO-like ordering.
- Docker Desktop (defaults) registers Rosetta in the VM, and mapping Rosetta
  into an Apple Virtualization VM reportedly runs the whole VM in TSO mode.

Hypothesis (not yet causally confirmed): VM-wide TSO, enabled for Rosetta, costs
this M1 an order of magnitude in load throughput, and GEMM-heavy inference
inherits it. The confirming test is to turn off Docker Desktop's "Use Rosetta
for x86_64/amd64 emulation on Apple Silicon" and re-run the litmus and load
microbenchmarks. Caveats: the magnitude may be M1-specific, and other
hypervisors or newer chips may differ. Either way, Docker Desktop on a Mac is not
a stand-in for native Linux, which is why Phase 2 runs on cloud VMs.
