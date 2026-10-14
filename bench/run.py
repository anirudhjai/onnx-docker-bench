"""Measure ONNX Runtime CPU inference latency for a single configuration.

Appends one row (latency stats + environment metadata) to a CSV and prints a summary.

Usage:
    python bench/run.py [--model models/resnet50.onnx] [--intra-threads 0] ... [--out results/results.csv]
"""

import argparse
import csv
import gc
import hashlib
import os
import platform
import secrets
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# Must be set before numpy/onnxruntime load. numpy's Linux wheels bundle OpenBLAS,
# which starts a thread per visible CPU at import; the benchmark never calls BLAS,
# so those threads only inflate the process thread count and make it incomparable
# between host and container. OMP_NUM_THREADS does the same for any OpenMP runtime
# (ORT's CPU EP doesn't use OpenMP). setdefault keeps an explicit override working.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

OPT_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
# Value for the session.intra_op.allow_spinning config entry; None = leave ORT's default.
SPIN = {"default": None, "on": "1", "off": "0"}
INPUT_SEED = 0

FIELDS = [
    "run_id",
    "timestamp_utc",
    "label",
    "model",
    "model_sha256",
    "system",
    "machine",
    "python_version",
    "ort_version",
    "os_cpu_count",
    "affinity_cpus",
    "cgroup_cpu_max",
    "cgroup_cpus",
    "in_container",
    "loadavg_1m",
    "intra_threads",
    "spin",
    "baseline_threads",
    "process_threads",
    "session_threads_added",
    "opt_level",
    "batch",
    "height",
    "width",
    "warmup",
    "iters",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "std_ms",
    "min_ms",
    "max_ms",
    "throughput_ips_p50",
    # cgroup CPU throttling counters, as deltas across the timed loop.
    "cgroup_nr_periods",
    "cgroup_nr_throttled",
    "cgroup_throttled_usec",
]
THROTTLE_KEYS = ("nr_periods", "nr_throttled", "throttled_usec")


def read_cgroup_cpu_max() -> str | None:
    """Return the CPU quota as "<quota> <period>" (cgroup v2 cpu.max format), or None.

    Reads the root of the cgroup hierarchy visible to this process, which inside a
    container with a private cgroup namespace (Docker's default on cgroup v2) is the
    container's own cgroup. cgroup v1 values are converted to the v2 format.
    """
    v2 = Path("/sys/fs/cgroup/cpu.max")
    if v2.exists():
        return v2.read_text().strip()
    for d in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
        quota, period = Path(d, "cpu.cfs_quota_us"), Path(d, "cpu.cfs_period_us")
        if quota.exists() and period.exists():
            q = quota.read_text().strip()
            return f"{'max' if q == '-1' else q} {period.read_text().strip()}"
    return None


def read_cgroup_throttling() -> dict[str, int] | None:
    """Return nr_periods, nr_throttled and throttled_usec from the cgroup's cpu.stat.

    Same cgroup as read_cgroup_cpu_max. cgroup v1 reports throttled_time in ns,
    converted here to usec. None when there are no throttling stats (no cgroup, or
    no CPU quota on this cgroup).
    """
    for path, time_key, ns_per_unit in (
        ("/sys/fs/cgroup/cpu.stat", "throttled_usec", 1),
        ("/sys/fs/cgroup/cpu/cpu.stat", "throttled_time", 1000),
        ("/sys/fs/cgroup/cpu,cpuacct/cpu.stat", "throttled_time", 1000),
    ):
        try:
            stat = dict(line.split() for line in Path(path).read_text().splitlines() if line.strip())
        except OSError:
            continue
        if "nr_periods" not in stat or time_key not in stat:
            return None
        return {
            "nr_periods": int(stat["nr_periods"]),
            "nr_throttled": int(stat["nr_throttled"]),
            "throttled_usec": int(stat[time_key]) // ns_per_unit,
        }
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_process_threads() -> int | str:
    """Number of OS threads in this process (all of them, not only ORT's pool)."""
    if os.path.isdir("/proc/self/task"):
        return len(os.listdir("/proc/self/task"))
    if platform.system() == "Darwin":
        # `ps -M` prints a header line followed by one line per thread.
        out = subprocess.run(
            ["ps", "-M", "-p", str(os.getpid())], capture_output=True, text=True, check=True
        ).stdout
        return len(out.strip().splitlines()) - 1
    return "n/a"


def in_container() -> bool:
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    try:
        text = Path("/proc/1/cgroup").read_text()
    except OSError:
        return False
    return any(m in text for m in ("docker", "containerd", "kubepods", "libpod"))


def write_raw(path: Path, ms: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "ms"])
        writer.writerows((i, f"{v:.4f}") for i, v in enumerate(ms))


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    if not is_new:
        with path.open(newline="") as f:
            header = next(csv.reader(f), None)
        if header != FIELDS:
            raise SystemExit(f"{path}: existing header differs from current columns; refusing to append")
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("models/resnet50.onnx"))
    parser.add_argument("--intra-threads", type=int, default=0, help="intra-op threads (0 = ORT default)")
    parser.add_argument(
        "--spin",
        choices=SPIN,
        default="default",
        help="session.intra_op.allow_spinning: on=1, off=0, default=leave unset",
    )
    parser.add_argument("--opt-level", choices=OPT_LEVELS, default="all")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, default=Path("results/results.csv"))
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="also write per-iteration latencies to <dir of --out>/raw/<run_id>.csv",
    )
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 0 or args.intra_threads < 0:
        parser.error("--iters must be >= 1; --warmup and --intra-threads must be >= 0")

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"
    # Machine load when the run starts (inside a container: the host/VM's load).
    loadavg_1m = os.getloadavg()[0]
    model_sha256 = sha256_file(args.model)

    # Threads that exist before ORT creates its pool (main thread + anything the
    # imports started).
    baseline_threads = count_process_threads()

    # Untimed setup: session creation, input generation, warmup.
    # Execution mode stays at ORT's default (sequential), so there is no inter-op pool.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.intra_threads
    opts.graph_optimization_level = OPT_LEVELS[args.opt_level]
    if SPIN[args.spin] is not None:
        opts.add_session_config_entry("session.intra_op.allow_spinning", SPIN[args.spin])
    sess = ort.InferenceSession(str(args.model), opts, providers=["CPUExecutionProvider"])
    # Some wheels (e.g. macOS) also ship CoreML; make sure nothing but the CPU EP runs.
    if sess.get_providers() != ["CPUExecutionProvider"]:
        raise SystemExit(f"unexpected execution providers: {sess.get_providers()}")

    shape = (args.batch, 3, args.height, args.width)
    x = np.random.default_rng(INPUT_SEED).standard_normal(shape, dtype=np.float32)
    feed = {sess.get_inputs()[0].name: x}

    for _ in range(args.warmup):
        sess.run(None, feed)
    process_threads = count_process_threads()
    if isinstance(process_threads, int) and isinstance(baseline_threads, int):
        session_threads_added = process_threads - baseline_threads
    else:
        session_threads_added = "n/a"

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    times = [0.0] * args.iters
    gc.collect()
    throttle_before = read_cgroup_throttling()
    gc.disable()  # keep collector pauses out of the timed loop
    try:
        for i in range(args.iters):
            t0 = time.perf_counter()
            sess.run(None, feed)
            times[i] = time.perf_counter() - t0
    finally:
        gc.enable()
    throttle_after = read_cgroup_throttling()
    if throttle_before is None or throttle_after is None:
        throttle = {k: "n/a" for k in THROTTLE_KEYS}
    else:
        throttle = {k: throttle_after[k] - throttle_before[k] for k in THROTTLE_KEYS}

    ms = np.asarray(times) * 1e3
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    stats = {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": ms.mean(),
        "std_ms": ms.std(ddof=1) if ms.size > 1 else 0.0,
        "min_ms": ms.min(),
        "max_ms": ms.max(),
        "throughput_ips_p50": args.batch / (p50 / 1e3),
    }

    cpu_max = read_cgroup_cpu_max()
    if cpu_max is None:
        cgroup_cpus = "n/a"
    else:
        quota, period = cpu_max.split()
        cgroup_cpus = "max" if quota == "max" else f"{int(quota) / int(period):.2f}"

    row = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "label": args.label,
        "model": str(args.model),
        "model_sha256": model_sha256,
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "ort_version": ort.__version__,
        "os_cpu_count": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "n/a",
        "cgroup_cpu_max": cpu_max or "n/a",
        "cgroup_cpus": cgroup_cpus,
        "in_container": in_container(),
        "loadavg_1m": f"{loadavg_1m:.2f}",
        "intra_threads": args.intra_threads,
        "spin": args.spin,
        "baseline_threads": baseline_threads,
        "process_threads": process_threads,
        "session_threads_added": session_threads_added,
        "opt_level": args.opt_level,
        "batch": args.batch,
        "height": args.height,
        "width": args.width,
        "warmup": args.warmup,
        "iters": args.iters,
        **{k: f"{v:.4f}" for k, v in stats.items()},
        **{f"cgroup_{k}": v for k, v in throttle.items()},
    }
    # Raw file first, so a summary row never points at a run_id without its raw data.
    raw_path = args.out.parent / "raw" / f"{run_id}.csv"
    if args.save_raw:
        write_raw(raw_path, ms)
    append_row(args.out, row)

    print(
        f"[{args.label or '-'}] run {run_id}  {args.model}  batch={args.batch} {args.height}x{args.width}  "
        f"intra={args.intra_threads} spin={args.spin} opt={args.opt_level}  "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(
        f"  threads: baseline {baseline_threads}, after warmup {process_threads} "
        f"(session added {session_threads_added})"
    )
    print(
        f"  latency ms: p50 {p50:.3f}  p95 {p95:.3f}  p99 {p99:.3f}  "
        f"mean {stats['mean_ms']:.3f}  std {stats['std_ms']:.3f}  "
        f"min {stats['min_ms']:.3f}  max {stats['max_ms']:.3f}"
    )
    print(f"  throughput: {stats['throughput_ips_p50']:.1f} images/s (at p50)")
    print(
        f"  cgroup during timed loop: {throttle['nr_periods']} periods, {throttle['nr_throttled']} throttled, "
        f"{throttle['throttled_usec']} us throttled"
    )
    print(
        f"  env: {row['system']} {row['machine']}, python {row['python_version']}, ort {row['ort_version']}, "
        f"cpu_count={row['os_cpu_count']}, affinity={row['affinity_cpus']}, "
        f"cgroup cpu.max={row['cgroup_cpu_max']} ({cgroup_cpus} cpus), container={row['in_container']}, "
        f"load1={row['loadavg_1m']}, model sha256={model_sha256[:12]}"
    )
    print(f"  appended to {args.out}" + (f"; raw timings in {raw_path}" if args.save_raw else ""))


if __name__ == "__main__":
    main()
