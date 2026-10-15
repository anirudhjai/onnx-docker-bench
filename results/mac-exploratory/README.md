# Mac exploratory runs (runner validation)

Two short runs of `matrices/mac-pilot.json` made on 2026-10-14 between 09:22 and
09:30 UTC to validate `scripts/run_matrix.py` (including a kill-and-resume test).
Same machine, Docker Desktop with the Rosetta setting off, ResNet-50 fp32 224,
`--cpus` 1 and 2, ORT default threads vs threads = `--cpus`, spinning at default,
**50 iterations per run**, 2 repeats per cell in each file. Raw timings in `raw/`.

**Not part of the v1 study.** They are kept because they show how much the most
oversubscribed configuration varied between sessions on the same Mac: fp32 at
`--cpus=1` with ORT's default threads took **1099–1206 ms** (p50) here, against
713–806 ms in the official run (`results/mac-v1/`) about 13 hours later. The
matched-thread cells agreed closely (100–102 ms here, 101.1–101.6 ms there).

Conditions differ from the official run and were not all recorded: these runs
predate the `power_source`, `battery_pct` and `cgroup_usage_usec` columns, the
OpenBLAS thread cap was already in place, and other apps were open. So the
difference can't be attributed to a specific cause.
