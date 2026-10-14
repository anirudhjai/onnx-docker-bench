"""Run a benchmark matrix in Docker: randomized interleaved repeats, resumable.

Usage:
    python3 scripts/run_matrix.py matrices/phase2.json [--stage input=224] \
        [--estimates results/phase2/pilot.csv] [--dry-run] [--build]

Each run is one `docker run` of the benchmark image that appends one row, labelled
"<cell_id>/r<repeat>", to the matrix's output CSV and saves raw timings. Runs whose
label is already in the CSV are skipped, so an interrupted matrix resumes where it
stopped. Standard library only, so it runs on a fresh machine without the venv.
"""

import argparse
import csv
import itertools
import json
import math
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACTORS = ("model", "input", "cpus", "intra", "spin")
CONTAINER_LABEL = "onnx-docker-bench.matrix"

current_container = None  # name of the container being run, for signal cleanup


def cell_id(cell: dict) -> str:
    return f"{cell['model']}-{cell['input']}-cpus{cell['cpus']}-intra{cell['intra']}-spin{cell['spin']}"


def run_label(cell: dict, repeat: int) -> str:
    return f"{cell_id(cell)}/r{repeat}"


def load_matrix(path: Path) -> dict:
    m = json.loads(path.read_text())
    missing = {"name", "image", "out", "seed", "repeats", "warmup", "iters", "models", "factors"} - m.keys()
    if missing:
        raise SystemExit(f"{path}: missing keys {sorted(missing)}")
    # "{arch}" keeps results from different machines apart (e.g. results/phase2/aarch64/).
    m["out"] = m["out"].format(arch=platform.machine())
    if set(m["factors"]) != set(FACTORS):
        raise SystemExit(f"{path}: factors must be exactly {FACTORS}")
    results_dir = (ROOT / "results").resolve()
    if not (ROOT / m["out"]).resolve().is_relative_to(results_dir):
        raise SystemExit(f"{path}: out must be under results/ (it is mounted into the container)")
    for name, model in m["models"].items():
        if not (ROOT / model).resolve().is_relative_to((ROOT / "models").resolve()):
            raise SystemExit(f"{path}: model {name} must be under models/ (it is mounted into the container)")
    return m


def build_schedule(m: dict) -> list[tuple[int, dict]]:
    """All (repeat, cell) runs: each repeat is a fresh seeded permutation of all cells.

    Built for the whole matrix before any --stage filter, so a stage's runs always
    come out in the same relative order.
    """
    cells = []
    for values in itertools.product(*(m["factors"][k] for k in FACTORS)):
        cell = dict(zip(FACTORS, values))
        excluded = any(all(str(cell[k]) == str(v) for k, v in ex.items()) for ex in m.get("exclude", []))
        if not excluded:
            cells.append(cell)
    rng = random.Random(m["seed"])
    schedule = []
    for repeat in range(1, m["repeats"] + 1):
        order = cells[:]
        rng.shuffle(order)
        schedule += [(repeat, cell) for cell in order]
    return schedule


def parse_stage(items: list[str]) -> dict[str, set[str]]:
    stage = {}
    for item in items:
        key, sep, values = item.partition("=")
        if not sep or key not in FACTORS:
            raise SystemExit(f"--stage {item!r}: expected KEY=V1[,V2...] with KEY in {FACTORS}")
        stage[key] = set(values.split(","))
    return stage


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("p50_ms")]


def mean_latencies(rows: list[dict]) -> defaultdict[str, list[float]]:
    """Each cell's per-run mean latencies (ms); the mean is what sets a run's duration."""
    by_cell = defaultdict(list)
    for r in rows:
        cid, sep, _ = r["label"].rpartition("/r")
        if sep:
            by_cell[cid].append(float(r["mean_ms"]))
    return by_cell


def iterations(m: dict, est_ms: float | None) -> int:
    """Plan rule: m["iters"], unless that would exceed max_timed_s of timed loop,
    then max(min_iters, max_timed_s / latency). Only pilot estimates are used, so
    every repeat of a cell gets the same count."""
    cap = m.get("iter_cap")
    if cap is None or est_ms is None:
        return m["iters"]
    return min(m["iters"], max(cap["min_iters"], math.floor(cap["max_timed_s"] * 1e3 / est_ms)))


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def host_power_args() -> list[str]:
    """On a macOS host, the power state at run start (`pmset -g batt`), passed into the
    container, which can't see it. Nothing elsewhere (the benchmark records n/a)."""
    if platform.system() != "Darwin":
        return []
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    source = "AC" if "'AC Power'" in out else "battery" if "'Battery Power'" in out else "n/a"
    pct = next((w.rstrip("%;") for w in out.split() if w.endswith("%;")), "n/a")
    return ["--power-source", source, "--battery-pct", pct]


def docker_command(m: dict, cell: dict, repeat: int, iters: int, name: str, extra: list[str]) -> list[str]:
    intra = cell["cpus"] if cell["intra"] == "matched" else int(cell["intra"])
    return [
        "docker", "run", "--rm", "--name", name, "--label", f"{CONTAINER_LABEL}={m['name']}",
        f"--cpus={cell['cpus']}",
        "-v", f"{ROOT / 'models'}:/app/models:ro",
        "-v", f"{ROOT / 'results'}:/app/results",
        m["image"],
        "--model", m["models"][cell["model"]],
        "--intra-threads", str(intra),
        "--spin", cell["spin"],
        "--height", str(cell["input"]),
        "--width", str(cell["input"]),
        "--warmup", str(m["warmup"]),
        "--iters", str(iters),
        "--label", run_label(cell, repeat),
        "--out", m["out"],
        "--save-raw",
        *extra,
    ]


class Log:
    def __init__(self, path: Path | None):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}"
        print(line, flush=True)
        if self.path:
            with self.path.open("a") as f:
                f.write(line + "\n")


def docker(*args: str) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True).stdout.strip()


def remove_leftover_containers(m: dict, log: Log) -> None:
    """A runner killed hard leaves its container running; it could still append a row
    after we resume. Remove such containers before reading the CSV."""
    ids = docker("ps", "-aq", "--filter", f"label={CONTAINER_LABEL}={m['name']}").split()
    if ids:
        log(f"removing {len(ids)} leftover container(s) from an interrupted runner: {' '.join(ids)}")
        docker("rm", "-f", *ids)


def drop_partial_last_line(path: Path, log: Log) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        log(f"{path}: dropping incomplete last line from an interrupted write: {data[cut:]!r}")
        path.write_bytes(data[:cut])


def on_signal(signum, frame) -> None:
    if current_container:
        subprocess.run(["docker", "kill", current_container], capture_output=True)
    print(f"\ninterrupted (signal {signum}); killed {current_container or 'no container'}; rerun to resume")
    sys.exit(128 + signum)


def main() -> None:
    global current_container
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matrix", type=Path, help="matrix definition (JSON)")
    parser.add_argument("--stage", action="append", default=[], metavar="KEY=V1[,V2]",
                        help=f"only run cells whose factor matches; KEY in {FACTORS}; repeatable")
    parser.add_argument("--estimates", type=Path, action="append", default=[],
                        help="pilot CSV(s) with rows for the same cell ids; sets iteration caps and ETAs")
    parser.add_argument("--dry-run", action="store_true", help="print the run order and estimated duration, run nothing")
    parser.add_argument("--build", action="store_true", help="docker build the image first")
    args = parser.parse_args()

    m = load_matrix(args.matrix)
    out = ROOT / m["out"]
    stage = parse_stage(args.stage)
    log = Log(None if args.dry_run else out.with_suffix(".log"))
    out.parent.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        try:
            docker("info")
        except (OSError, subprocess.CalledProcessError):
            raise SystemExit("Docker is not running")
        if args.build:
            subprocess.run(["docker", "build", "-t", m["image"], str(ROOT)], check=True)
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        remove_leftover_containers(m, log)
        drop_partial_last_line(out, log)

    rows = read_rows(out)
    done = Counter(r["label"] for r in rows)
    for label, n in done.items():
        if n > 1:
            log(f"WARNING: {label} appears {n} times in {out}")
    pilot = {}
    for path in args.estimates:
        pilot.update({cid: statistics.median(v) for cid, v in mean_latencies(read_rows(path)).items()})
    # ETAs prefer this matrix's own completed runs (updated as runs finish), then the
    # pilot. Iteration caps use the pilot only, so all repeats of a cell match.
    observed = mean_latencies(rows)
    eta_latency = {**pilot, **{cid: statistics.median(v) for cid, v in observed.items()}}

    schedule = [(r, c) for r, c in build_schedule(m) if all(str(c[k]) in v for k, v in stage.items())]
    todo = [(r, c) for r, c in schedule if run_label(c, r) not in done]

    def plan(repeat: int, cell: dict) -> tuple[int, float | None]:
        iters = iterations(m, pilot.get(cell_id(cell)))
        est = eta_latency.get(cell_id(cell))
        secs = None if est is None else m.get("run_overhead_s", 15) + (m["warmup"] + iters) * est / 1e3
        return iters, secs

    stage_desc = " ".join(args.stage) or "all cells"
    log(f"matrix {m['name']} ({args.matrix}), stage: {stage_desc}, seed {m['seed']}; "
        f"{len(schedule)} runs, {len(schedule) - len(todo)} already in {m['out']}, {len(todo)} to do")
    if not args.dry_run:
        log(f"image {m['image']} = {docker('image', 'inspect', '-f', '{{.Id}}', m['image'])}")

    if args.dry_run:
        known, unknown = 0.0, 0
        for i, (repeat, cell) in enumerate(schedule, 1):
            label = run_label(cell, repeat)
            iters, secs = plan(repeat, cell)
            status = "done" if label in done else "todo"
            if status == "todo":
                known, unknown = (known + secs, unknown) if secs is not None else (known, unknown + 1)
            capped = "*" if iters < m["iters"] else " "
            print(f"{i:4d}  {status}  {label:48s} iters={iters:5d}{capped} est={fmt_duration(secs) if secs else '?'}")
        if not todo:
            estimate = "nothing left to run"
        elif unknown < len(todo):
            estimate = f"estimated {fmt_duration(known)}"
        else:
            estimate = "no latency estimates yet (pass --estimates <pilot.csv>)"
        print(f"remaining: {len(todo)} runs, {estimate}"
              + (f" + {unknown} runs with no latency estimate" if 0 < unknown < len(todo) else "")
              + ("; * = iteration cap applied" if any(plan(r, c)[0] < m["iters"] for r, c in todo) else ""))
        return

    for path in {ROOT / m["models"][c["model"]] for _, c in todo}:
        if not path.exists():
            raise SystemExit(f"{path} not found (scripts/export_model.py / scripts/quantize_model.py)")

    failures = []
    for i, (repeat, cell) in enumerate(todo, 1):
        label = run_label(cell, repeat)
        iters, secs = plan(repeat, cell)
        remaining = [plan(r, c)[1] for r, c in todo[i - 1:]]
        unknown = sum(s is None for s in remaining)
        eta = fmt_duration(sum(s for s in remaining if s is not None)) + (f" + {unknown} unestimated" if unknown else "")
        power = host_power_args()
        log(f"[{i}/{len(todo)}] start {label} iters={iters}{' (capped)' if iters < m['iters'] else ''} "
            f"est={fmt_duration(secs) if secs else '?'}, remaining ~{eta}"
            + (f", power {power[1]} {power[3]}%" if power else ""))
        current_container = f"odb-{m['name']}-{cell_id(cell)}-r{repeat}"
        t0 = time.monotonic()
        proc = subprocess.run(docker_command(m, cell, repeat, iters, current_container, power),
                              cwd=ROOT, capture_output=True, text=True)
        current_container = None
        if proc.returncode != 0:
            failures.append(label)
            log(f"FAILED {label} (exit {proc.returncode}): {proc.stderr.strip()[-500:]}")
            continue
        row = next((r for r in read_rows(out) if r["label"] == label), None)
        if row is None:
            failures.append(label)
            log(f"FAILED {label}: container exited 0 but wrote no row")
            continue
        observed[cell_id(cell)].append(float(row["mean_ms"]))
        eta_latency[cell_id(cell)] = statistics.median(observed[cell_id(cell)])
        log(f"done {label} in {time.monotonic() - t0:.0f}s: p50 {row['p50_ms']} p99 {row['p99_ms']} ms, "
            f"session threads +{row['session_threads_added']}, "
            f"throttled {row['cgroup_nr_throttled']}/{row['cgroup_nr_periods']} periods, run {row['run_id']}")

    if failures:
        log(f"{len(failures)} run(s) failed and will be retried on the next invocation: {', '.join(failures)}")
        sys.exit(1)
    log(f"stage complete: all {len(schedule)} runs of '{stage_desc}' are in {m['out']}")


if __name__ == "__main__":
    main()
