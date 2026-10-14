# Phase 2 experiment plan

Status: ready for review; the tooling (runner, int8 model, per-run metrics) is
built and was validated on the Mac, but nothing here has run in the cloud. Prices
were checked on 2026-10-14 (instances.vantage.sh, us-east-1 on-demand Linux) and
should be re-checked before launching.

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
- On-demand rather than spot. The runner is resumable, but a spot interruption
  also takes the instance, and with it any results not yet copied off. Spot
  would save only ~$6 in total.

Same OS image on both (Ubuntu 24.04 LTS), Docker Engine (not Desktop),
cgroup v2. The Dockerfile pins `python:3.12-slim` by its multi-arch index digest,
so both machines build from the same source.

## Matrix

Fixed: ResNet-50, batch 1, `--opt-level all`, ORT 1.31.0, sequential execution
mode, the pinned image.

**onnxruntime is frozen at 1.31.0** (decided 2026-10-14). It was the current
release when pinned (published 2026-10-09). The study compares configurations,
not ORT versions, so every stage on every machine has to run the same build:
an upgrade mid-study, even to a 1.31.x patch, would confound comparisons across
stages and with the Mac results. It is pinned in `requirements.txt` and the
Dockerfile's `ORT_VERSION`. `cloud/setup.sh` refuses to continue if the built
image reports a different version, and every row and `machine-*.json` records it.

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

int8 model: `scripts/quantize_model.py` writes `models/resnet50.int8.onnx` with
`onnxruntime.quantization.quantize_dynamic(weight_type=QInt8)`. It strips the
exporter's `value_info` first (the quantizer transposes the classifier weight but
keeps its stale shape annotation, then fails its own shape inference) and checks
fp32-vs-int8 argmax agreement on 32 fixed random inputs (32/32 on the Mac; the
weights are random, so this is a sanity check, not accuracy). Mac native: int8
is 35.2 vs 99.1 ms at 1 thread and 16.3 vs 32.4 ms at 4 threads. The graph gains
50 `DynamicQuantizeLinear` nodes, so activation quantization is part of the timed
latency. Static (calibrated QDQ) quantization would be faster but is a different
experiment.

**Is 44 too many?** No, I'd keep all of it, run in stages (below). If wall time
matters more than coverage:

| Option | Cells | h/machine (R=5) | Trade-off |
|---|---:|---:|---|
| A. Full (recommended) | 44 | ~17 | — |
| B. Drop int8 @ 448 | 33 | ~13 | Lose "does the int8 speedup hold at larger inputs?" |
| C. Drop 448 entirely | 22 | ~5 | Lose "is the oversubscription penalty fixed overhead or proportional to work?" |

448 costs ~12 of the ~17 hours, but it's also what distinguishes a per-operator
synchronisation cost from one that scales with work, so I'd cut it last. The
staged plan defers that decision until the 224 results are in.

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
timed loop (estimated from the pilot), use `max(300, floor(600 s / latency))`.
That affects ~6 of 44 cells, all where ORT defaults meet 1–2 CPUs. Effects there
are 7–11x on the Mac, so the coarser p99 doesn't change conclusions. The runner
applies the cap from pilot estimates only, so all repeats of a cell use the same
count; `iters` is recorded in every row.

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
- ORT default (8 threads) vs matched, spin on: 11x / 7.2x / 3x at 1/2/4 CPUs.
  The 1- and 2-CPU figures are from the Mac pilot with Rosetta off (fp32 224);
  the 4-CPU figure is a guess. Spin off: 1.3x / 1.1x / 1.05x (measured only at
  2 CPUs, and under TSO). These dominate the uncertainty.
- 15 s per run for container start, session creation and model hashing.

| Plan | h/machine | c8g.2xlarge | c8a.2xlarge |
|---|---:|---:|---:|
| 44 cells, R=5, capped iterations (recommended) | ~17 | ~$5.4 | ~$7.3 |
| 44 cells, R=6, capped | ~20 | ~$6.5 | ~$8.8 |
| 44 cells, R=5, uncapped 1000 iterations | ~28 | ~$8.8 | ~$11.9 |

By stage (R=5, capped): pilot ~1 h, input=224 ~5 h, input=448 ~12 h. The
slowest single run is fp32/448/cpus=1/default threads/spin on: ~5.3 s per
iteration at the 300-iteration floor, ~31 min per run. Both machines run in
parallel. EBS (30 GB gp3) and data transfer are cents. Even if the assumptions are
2x optimistic, the total stays under ~$35.

## Staged execution

Run each stage on both machines, and review before starting the next. The
runner skips runs already in the output CSV, so a stage can be interrupted and
resumed, and stages share one results file.

```bash
python3 scripts/run_matrix.py matrices/phase2-pilot.json --build                     # stage 0: pilot, ~1 h
python3 scripts/run_matrix.py matrices/phase2.json --stage input=224 \
    --estimates results/phase2/$(uname -m)/pilot.csv --dry-run                        # check order, caps, ETA
python3 scripts/run_matrix.py matrices/phase2.json --stage input=224 \
    --estimates results/phase2/$(uname -m)/pilot.csv                                  # stage 1: ~5 h
# review stage 1, then:
python3 scripts/run_matrix.py matrices/phase2.json --stage input=448 \
    --estimates results/phase2/$(uname -m)/pilot.csv                                  # stage 2: ~12 h
```

1. **Pilot** (`matrices/phase2-pilot.json`): every cell once at 100 iterations,
   including the cpus=1/matched/spin=off cell the main matrix drops. Check that
   session_threads_added matches intra (default = 7 added on 8 vCPUs), cgroup_cpus
   matches `--cpus`, nr_throttled is 0 without a quota, and the dropped cell
   matches its spin=default twin. Its latencies set the iteration caps and ETAs.
2. **All 224 cells** (110 runs). Cheaper, and enough to answer Q1–Q2 for both
   models.
3. **Review**: noise between repeats, loadavg, anything odd. Then decide whether
   448 is worth ~12 h, or cut to option B.
4. **448 cells** (110 runs).

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
- **Recorded per run** (in the CSV): run_id, label (`<cell_id>/r<repeat>`),
  model sha256, versions, os_cpu_count, affinity, cgroup quota, 1-min load
  average at start, baseline/process/session-added threads, iters, all stats,
  and cgroup throttling deltas across the timed loop (`cgroup_nr_periods`,
  `cgroup_nr_throttled`, `cgroup_throttled_usec`); raw per-iteration timings in
  `raw/<run_id>.csv`. `throttled_usec` is summed over CPUs, so it can exceed
  wall time; `nr_throttled / nr_periods` is the easier number to read.
- Inside a container, the load average is the host's (or Docker VM's), which is
  what we want as a noise covariate.
- Not built yet: the once-per-machine `machine.json` recorder above.

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
- Throttled-period fraction (`nr_throttled / nr_periods`) per cell.

## Findings so far (Mac, exploratory)

M1 MacBook Pro, Docker Desktop 29.6.1 (8 vCPU, 3.8 GiB VM), ORT 1.31.0, ResNet-50
fp32 224, batch 1.

**1. ORT's default thread pool ignores the container's CPU quota.** Inside a
container the process sees all 8 VM CPUs (`os.cpu_count()` and affinity), so ORT
starts 8 threads (7 workers + the caller) whatever `--cpus` says.

With Docker Desktop's Rosetta setting off (finding 2), from the runner validation
runs (`matrices/mac-pilot.json`: 50 iterations per run, 4 runs per cell, scratch):

| `--cpus` | ORT default (8 threads), p50 ms | threads = cpus, p50 ms | penalty |
|---|---|---|---:|
| 1 | 1099–1206 | 100–102 | ~11x |
| 2 | 317–406 | 53–54 | ~7x |

Every default-thread run was throttled in 100% of its CFS periods (e.g. 554/554);
matched runs were throttled in 22–76% of periods.

With Rosetta on (`results/mac-rosetta-on/phase1.csv`, `--cpus=2`, 300 iterations), the same
comparison was 1113 vs 339 ms (3.3x). Spinning off brought the 8-thread case to
372 ms there, but its latencies split into two clusters (p50 between 323 and 376
ms across runs; p99 390–395 vs 344–350 ms matched). Rosetta's slow kernels hid
part of the penalty. Spinning off has not been measured with Rosetta off.

Caveats: one machine; 50 iterations per run (p99 not meaningful); the Docker VM's
vCPUs are scheduled by macOS across 4 performance and 4 efficiency cores.

**2. Docker Desktop's Rosetta setting made ORT ~6.6x slower on this Mac.** With
"Use Rosetta for x86_64/amd64 emulation on Apple Silicon" on (the default), the
VM's vCPUs showed TSO-like memory ordering and loads ran 11–18x slower; turning
it off restored both. Details, sources and how to reproduce:
`investigations/rosetta-tso/README.md`.

| Test | Rosetta on | Rosetta off | Host |
|---|---:|---:|---:|
| Message-passing litmus: reorderings | 0 of ~860M samples | ~50% | ~33% |
| NEON loads from L1 (GB/s) | 7.2 | 131.6 | 130.1 |
| OpenBLAS SGEMM, 1 thread (GFLOP/s) | 13.5 | 96.3 | 94.2 |
| ORT, 1 thread, p50 (ms) | 655.0 | 99.8 | 100.0 |
| ORT `--cpus=2 --intra-threads 2`, p50 (ms) | 339.1 | 53.5 | 55.9 (host, 2 threads) |

Strictly, toggling the setting flipped the ordering behaviour and the slowdown
together; TSO was not toggled on its own. The size is far above published costs
of Apple's TSO mode (typically within 3%; 9–22% suite averages in earlier work),
so either VM-wide TSO behaves differently on the M1 or the setting changes more
than the memory model. Practical consequence: benchmark arm64 containers on Apple
Silicon with Rosetta off. Even then, Docker Desktop on a Mac is not a stand-in for
native Linux (a VM, with macOS scheduling its vCPUs), which is why Phase 2 runs on
cloud VMs.
