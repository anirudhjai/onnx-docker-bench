"""Figures and the summary table for the Mac v1 results.

Reads only committed data: results/mac-v1/results.csv (+ raw/<run_id>.csv) and
investigations/rosetta-tso/results.csv. Writes docs/figures/*.png and
docs/results_summary.md. Every number in a title or label is computed here.

Usage:
    .venv/bin/python scripts/analyze.py [--csv results/mac-v1/results.csv] [--figures docs/figures]
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "mac-v1" / "results.csv"  # raw timings: raw/ next to it
FIGURES = ROOT / "docs" / "figures"
ROSETTA = ROOT / "investigations" / "rosetta-tso" / "results.csv"
SUMMARY = ROOT / "docs" / "results_summary.md"

# Validated categorical slots (dataviz reference palette, light mode), one meaning each.
MATCHED = "#2a78d6"   # threads = --cpus
DEFAULT = "#eb6834"   # ORT default thread count
SPIN_OFF = "#1baf7a"  # default threads, spinning off
INT8 = "#4a3aa7"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

CELL = re.compile(r"(?P<model>fp32|int8)-(?P<input>\d+)-cpus(?P<cpus>\d+)-intra(?P<intra>\w+?)-spin(?P<spin>\w+)$")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def load_runs() -> list[dict]:
    with RESULTS_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.update(CELL.match(r["label"].split("/")[0]).groupdict())
        r["cell"] = r["label"].split("/")[0]
    return rows


def raw_ms(run_id: str) -> np.ndarray:
    with (RESULTS_CSV.parent / "raw" / f"{run_id}.csv").open(newline="") as f:
        return np.array([float(r["ms"]) for r in csv.DictReader(f)])


def summarize(rows: list[dict]) -> dict[str, dict]:
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    cells = {}
    for cell, runs in by_cell.items():
        def col(name):
            return [float(r[name]) for r in runs]
        throttled = [int(r["cgroup_nr_throttled"]) / int(r["cgroup_nr_periods"]) for r in runs
                     if r["cgroup_nr_periods"] not in ("", "n/a", "0")]
        # Container CPU time per inference (ms), all threads; absent in older CSVs.
        cpu_ms = [int(r["cgroup_usage_usec"]) / 1e3 / int(r["iters"]) for r in runs
                  if r.get("cgroup_usage_usec", "").isdigit()]
        pooled = np.concatenate([raw_ms(r["run_id"]) for r in runs])
        cells[cell] = {
            **{k: runs[0][k] for k in ("model", "input", "cpus", "intra", "spin")},
            "runs": len(runs),
            "iters": sorted({int(r["iters"]) for r in runs}),
            "p50": col("p50_ms"), "p99": col("p99_ms"), "mean": col("mean_ms"),
            "throttled": throttled,
            "cpu_ms": cpu_ms,
            "threads_added": sorted({r["session_threads_added"] for r in runs}),
            "pooled": pooled,
        }
    return cells


def med(xs):
    return statistics.median(xs)


def key(model, cpus, intra, spin):
    return f"{model}-224-cpus{cpus}-intra{intra}-spin{spin}"


def runs_text(cells) -> str:
    n = sorted({c["runs"] for c in cells.values()})
    return f"{n[0]} runs" if len(n) == 1 else f"{n[0]}–{n[-1]} runs"


def pct_range(values) -> str:
    lo, hi = round(min(values)), round(max(values))
    return f"{lo}%" if lo == hi else f"{lo}–{hi}%"


def plain_log(axis):
    """Log axis labelled 20, 50, 100, 200, ... instead of powers of ten."""
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def bar_label(ax, x, y, text, color=INK, dy=1.08, **kw):
    ax.text(x, y * dy, text, ha="center", va="bottom", fontsize=9, color=color, **kw)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {(FIGURES / name).relative_to(ROOT)}")


def fig1_headline(cells):
    cpus = [1, 2, 4]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(cpus))
    w = 0.3
    ratios = []
    for i, c in enumerate(cpus):
        d, m = cells[key("fp32", c, "0", "default")], cells[key("fp32", c, "matched", "default")]
        for dx, cell, color in ((-w / 2, d, DEFAULT), (w / 2, m, MATCHED)):
            p50 = med(cell["p50"])
            ax.bar(x[i] + dx, p50, width=w - 0.02, color=color, zorder=2)
            ax.vlines(x[i] + dx, min(cell["p50"]), max(cell["p50"]), color=INK, linewidth=1, zorder=3)
            ax.plot(x[i] + dx, med(cell["p99"]), marker="D", markersize=6, color=SURFACE,
                    markeredgecolor=INK, markeredgewidth=1.2, zorder=4)
        ratio = med(d["p50"]) / med(m["p50"])
        ratios.append(ratio)
        bar_label(ax, x[i] - w / 2, med(d["p99"]), f"{ratio:.1f}× slower", dy=1.15, fontweight="bold")
    ax.set_yscale("log")
    plain_log(ax.yaxis)
    ax.set_xticks(x, [f"--cpus={c}" for c in cpus])
    ax.set_ylabel("latency per image (ms, log scale)")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    ax.set_title(f"ORT's default thread count is {min(ratios):.1f}–{max(ratios):.1f}× slower than\n"
                 "matching threads to the container's CPU limit")
    handles = [plt.Rectangle((0, 0), 1, 1, color=DEFAULT), plt.Rectangle((0, 0), 1, 1, color=MATCHED),
               plt.Line2D([], [], marker="D", linestyle="", color=SURFACE, markeredgecolor=INK, markersize=6)]
    ax.legend(handles, ["ORT default threads (8 = visible CPUs), p50",
                        "intra_op_num_threads = --cpus, p50", "p99"], loc="upper right", fontsize=9)
    fig.text(0.01, -0.02, f"ResNet-50 fp32, 224×224, batch 1, spinning at ORT default. Bars: median p50 of {runs_text(cells)}; "
             "lines: min–max across runs.\nDocker Desktop 29.6.1 on an M1 MacBook Pro (VM: 8 vCPUs), Rosetta off. "
             "Data: results/mac-v1/results.csv", fontsize=8, color=INK_2, va="top")
    save(fig, "fig1_default_vs_matched.png")
    return ratios


def fig2_throttling(cells):
    cpus = [1, 2, 4]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    for ax, spin, title in ((axes[0], "default", "spinning on (ORT default)"), (axes[1], "off", "spinning off")):
        x = np.arange(len(cpus))
        w = 0.3
        for dx, intra, color, label in ((-w / 2, "0", DEFAULT, "ORT default threads"),
                                        (w / 2, "matched", MATCHED, "threads = --cpus")):
            for i, c in enumerate(cpus):
                cell = cells.get(key("fp32", c, intra, spin))
                if cell is None:  # cpus=1/matched/spin off is not in the matrix (intra=1 has no workers)
                    ax.text(x[i] + dx, 0.02, "n/a", ha="center", fontsize=8, color=INK_2)
                    continue
                share = 100 * med(cell["throttled"])
                ax.bar(x[i] + dx, share, width=w - 0.02, color=color, zorder=2)
                ax.vlines(x[i] + dx, 100 * min(cell["throttled"]), 100 * max(cell["throttled"]), color=INK, linewidth=1, zorder=3)
                bar_label(ax, x[i] + dx, max(share, 1), f"{share:.0f}%", dy=1.0)
        ax.set_xticks(x, [f"--cpus={c}" for c in cpus])
        ax.set_title(title, fontsize=10, fontweight="normal")
        ax.set_ylim(0, 112)
    axes[0].set_ylabel("CFS periods throttled (%)")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=DEFAULT), plt.Rectangle((0, 0), 1, 1, color=MATCHED)],
               ["ORT default threads (8)", "intra_op_num_threads = --cpus"], loc="upper right",
               bbox_to_anchor=(0.99, 0.94), ncol=2, fontsize=9)
    shares = {intra: [100 * med(c["throttled"]) for c in cells.values()
                      if c["model"] == "fp32" and (c["intra"] == "0") == (intra == "0")] for intra in ("0", "matched")}
    fig.suptitle(f"ORT's default threads were throttled in {pct_range(shares['0'])} of CPU-quota periods,\n"
                 f"threads matched to --cpus in {pct_range(shares['matched'])}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.03, "fp32, 224×224. nr_throttled / nr_periods from the container's cgroup cpu.stat over the "
             f"timed loop; median of {runs_text(cells)}, lines: min–max. Data: results/mac-v1/results.csv",
             fontsize=8, color=INK_2, va="top")
    fig.tight_layout()
    save(fig, "fig2_throttling.png")


def fig3_distributions(cells):
    configs = [(key("fp32", 2, "matched", "default"), MATCHED, "threads = --cpus (2)"),
               (key("fp32", 2, "0", "off"), SPIN_OFF, "ORT default threads, spinning off"),
               (key("fp32", 2, "0", "default"), DEFAULT, "ORT default threads (spinning on)")]
    allv = np.concatenate([cells[k]["pooled"] for k, _, _ in configs])
    # A handful of extreme iterations would stretch the axis; cap it and say so.
    x_max = min(allv.max() * 1.05, 1000.0)
    beyond = allv[allv > x_max]
    bins = np.geomspace(allv.min() * 0.95, x_max, 90)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.6), sharex=True)
    for i, (ax, (k, color, label)) in enumerate(zip(axes, configs)):
        v = cells[k]["pooled"]
        ax.hist(v, bins=bins, color=color, zorder=2)
        ax.set_xscale("log")
        # Label on the side of the panel away from its data.
        x_text, ha = (0.99, "right") if i < 2 else (0.01, "left")
        ax.text(x_text, 0.85, f"{label}\np50 {np.median(v):.0f} ms, p99 {np.percentile(v, 99):.0f} ms, n = {v.size}",
                transform=ax.transAxes, ha=ha, va="top", fontsize=9, color=INK)
        ax.set_ylabel("iterations")
    axes[-1].set_xlabel("latency per image (ms, log scale)")
    axes[-1].set_xlim(right=x_max)
    plain_log(axes[-1].xaxis)
    off = cells[key("fp32", 2, "0", "off")]["pooled"]
    gap_lo, gap_hi = largest_gap(off)
    axes[1].axvspan(gap_lo, gap_hi, color=GRID, zorder=1)
    low, high = off[off <= gap_lo], off[off >= gap_hi]
    bimodal = gap_hi / gap_lo >= 1.15 and min(low.size, high.size) >= 0.05 * off.size
    title = (f"At --cpus=2, turning spinning off splits latency into two clusters:\n"
             f"{low.size / off.size:.0%} of iterations near {np.median(low):.0f} ms, "
             f"{high.size / off.size:.0%} near {np.median(high):.0f} ms" if bimodal else
             "Latency distributions at --cpus=2")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.01, f"fp32, 224×224, all timed iterations of {runs_text(cells)} pooled. Shaded: the widest empty gap in the "
             f"spin-off distribution ({gap_lo:.0f}–{gap_hi:.0f} ms).\n"
             + (f" Not shown: {beyond.size} iterations above {x_max:.0f} ms (max {beyond.max():.0f} ms)." if beyond.size else "")
             + " Data: results/mac-v1/raw/",
             fontsize=8, color=INK_2, va="top")
    fig.tight_layout()
    save(fig, "fig3_distributions_cpus2.png")


def largest_gap(v: np.ndarray) -> tuple[float, float]:
    s = np.sort(v)
    lo, hi = np.percentile(s, 5), np.percentile(s, 95)
    inner = s[(s >= lo) & (s <= hi)]
    i = int(np.argmax(np.diff(np.log(inner))))
    return inner[i], inner[i + 1]


def fig4_int8(cells):
    cpus = [1, 2, 4]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(cpus))
    w = 0.3
    speedups = []
    for i, c in enumerate(cpus):
        f, q = cells[key("fp32", c, "matched", "default")], cells[key("int8", c, "matched", "default")]
        for dx, cell, color in ((-w / 2, f, MATCHED), (w / 2, q, INT8)):
            ax.bar(x[i] + dx, med(cell["p50"]), width=w - 0.02, color=color, zorder=2)
            ax.vlines(x[i] + dx, min(cell["p50"]), max(cell["p50"]), color=INK, linewidth=1, zorder=3)
        s = med(f["p50"]) / med(q["p50"])
        speedups.append(s)
        bar_label(ax, x[i] + w / 2, med(q["p50"]), f"{s:.1f}× faster", dy=1.04)
    ax.set_xticks(x, [f"--cpus={c}" for c in cpus])
    ax.set_ylabel("p50 latency per image (ms)")
    ax.set_title(f"int8 dynamic quantization is {min(speedups):.1f}–{max(speedups):.1f}× faster than fp32\n"
                 "with threads matched to --cpus")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=MATCHED), plt.Rectangle((0, 0), 1, 1, color=INT8)],
              ["fp32", "int8 (dynamic, QInt8 weights)"], loc="upper right", fontsize=9)
    fig.text(0.01, -0.02, f"224×224, intra_op_num_threads = --cpus, spinning at ORT default. Median p50 of {runs_text(cells)}; "
             "lines: min–max.\nRandom weights: this is a latency comparison, not an accuracy one. "
             "Data: results/mac-v1/results.csv", fontsize=8, color=INK_2, va="top")
    save(fig, "fig4_fp32_vs_int8.png")
    return speedups


def fig5_rosetta():
    with ROSETTA.open(newline="") as f:
        rows = {r["metric"]: r for r in csv.DictReader(f)}
    panels = [("ort_resnet50_1thread_p50", "ORT ResNet-50, 1 thread\np50 latency (ms, lower is better)"),
              ("l1_neon_load_bandwidth", "NEON loads from L1\nbandwidth (GB/s, higher is better)")]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for ax, (metric, ylabel) in zip(axes, panels):
        r = rows[metric]
        on, off, host = float(r["rosetta_on"]), float(r["rosetta_off"]), float(r["host"])
        ax.bar([0, 1], [on, off], width=0.5, color=[DEFAULT, MATCHED], zorder=2)
        ax.axhline(host, color=INK_2, linewidth=1, zorder=3)
        ax.text(1.42, host, f"macOS host\n{host:g}", va="center", fontsize=8, color=INK_2)
        for xi, v in ((0, on), (1, off)):
            ax.text(xi, v * 1.02, f"{v:g}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1], ["Rosetta on\n(Docker default)", "Rosetta off"])
        ax.set_xlim(-0.5, 1.9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(on, off, host) * 1.18)
    ort, load = rows["ort_resnet50_1thread_p50"], rows["l1_neon_load_bandwidth"]
    fig.suptitle(f"With Docker Desktop's Rosetta setting on, L1 loads were "
                 f"{float(load['rosetta_off']) / float(load['rosetta_on']):.0f}× slower and ORT "
                 f"{float(ort['rosetta_on']) / float(ort['rosetta_off']):.1f}× slower (M1)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.03, "Arm64 containers on an M1 MacBook Pro, Docker Desktop 29.6.1. The VM showed TSO-like "
             "memory ordering only with the setting on;\nthe setting was toggled as a whole, so TSO is the leading "
             "explanation, not a proven cause. Data: investigations/rosetta-tso/results.csv",
             fontsize=8, color=INK_2, va="top")
    fig.tight_layout()
    save(fig, "fig5_rosetta.png")


def summary_table(cells) -> str:
    def rng(xs, fmt="{:.1f}"):
        return f"{fmt.format(med(xs))} ({fmt.format(min(xs))}–{fmt.format(max(xs))})"
    order = sorted(cells.values(), key=lambda c: (c["model"], int(c["cpus"]), c["intra"] != "0", c["spin"]))
    lines = [
        "| model | --cpus | intra threads | spin | runs | iters/run | p50 ms, median (min–max) | "
        "p99 ms, median (min–max) | pooled p99 ms | periods throttled | CPU ms per inference | threads added |",
        "|---|---:|---|---|---:|---:|---|---|---:|---|---|---:|",
    ]
    for c in order:
        intra = c["cpus"] if c["intra"] == "matched" else "0 (default)"
        thr = rng([100 * t for t in c["throttled"]], "{:.0f}%") if c["throttled"] else "n/a"
        cpu = rng(c["cpu_ms"]) if c["cpu_ms"] else "n/a"
        lines.append(
            f"| {c['model']} | {c['cpus']} | {intra} | {c['spin']} | {c['runs']} | "
            f"{'/'.join(map(str, c['iters']))} | {rng(c['p50'])} | {rng(c['p99'])} | "
            f"{np.percentile(c['pooled'], 99):.1f} | {thr} | {cpu} | {'/'.join(c['threads_added'])} |"
        )
    return "\n".join(lines)


def checks(rows: list[dict], cells: dict) -> str:
    """Data-quality checks, as markdown: power source, unexpected thread counts, and the
    run-to-run spread of the most oversubscribed fp32 cell."""
    out = ["## Checks", ""]

    # Power: AC vs battery runs of the same cell, compared with that cell's repeat spread.
    sources = sorted({r.get("power_source", "n/a") for r in rows})
    pcts = [int(r["battery_pct"]) for r in rows if r.get("battery_pct", "").isdigit()]
    out.append(f"- **Power source at run start:** {', '.join(sources)} across all {len(rows)} runs"
               + (f"; battery {min(pcts)}–{max(pcts)}%." if pcts else "."))
    mixed = []
    for cell, c in cells.items():
        runs = [r for r in rows if r["cell"] == cell]
        by_src = defaultdict(list)
        for r in runs:
            by_src[r.get("power_source", "n/a")].append(float(r["p50_ms"]))
        if len(by_src) > 1:
            spread = max(c["p50"]) - min(c["p50"])
            mixed.append(f"  - {cell}: " + ", ".join(f"{s} p50 {[round(v, 1) for v in vs]}" for s, vs in by_src.items())
                         + f" (all-run spread {spread:.1f} ms)")
    if mixed:
        out.append("- **Cells with both AC and battery runs** (compare the gap with the spread):")
        out += mixed
    else:
        out.append("- No cell has runs on more than one power source, so AC vs battery can't be compared "
                   "within a cell in this data.")

    # Threads: ORT's pool adds intra-1 workers; the default is one per visible CPU.
    odd = []
    for r in rows:
        expected = (int(r["cgroup_cpus"].split(".")[0]) if r["intra"] == "matched" else int(r["os_cpu_count"])) - 1
        if r["session_threads_added"] != str(expected):
            odd.append(f"{r['label']} added {r['session_threads_added']} (expected {expected}), run {r['run_id']}")
    out.append(f"- **Session threads added:** {'all as expected' if not odd else str(len(odd)) + ' unexpected:'}")
    out += [f"  - {o}" for o in odd]

    # The most oversubscribed fp32 cell, run by run.
    worst = [r for r in rows if r["cell"] == key("fp32", 1, "0", "default")]
    out.append(f"- **{key('fp32', 1, '0', 'default')}, run by run** (p50 / p99 ms, load average and battery at start):")
    for r in sorted(worst, key=lambda r: r["label"]):
        out.append(f"  - {r['label'].split('/')[1]}: {float(r['p50_ms']):.0f} / {float(r['p99_ms']):.0f}, "
                   f"load {r['loadavg_1m']}, {r.get('power_source', 'n/a')} {r.get('battery_pct', 'n/a')}%, "
                   f"{r['iters']} iterations")
    p50 = [float(r["p50_ms"]) for r in worst]
    if p50:
        out.append(f"  - range {min(p50):.0f}–{max(p50):.0f} ms ({max(p50) / min(p50):.2f}× max/min)")
    return "\n".join(out)


def main() -> None:
    global RESULTS_CSV, FIGURES, SUMMARY
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=RESULTS_CSV, help="results CSV (raw timings in raw/ beside it)")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    RESULTS_CSV, FIGURES, SUMMARY = args.csv.resolve(), args.figures.resolve(), args.summary.resolve()

    rows = load_runs()
    cells = summarize(rows)
    ratios = fig1_headline(cells)
    fig2_throttling(cells)
    fig3_distributions(cells)
    speedups = fig4_int8(cells)
    fig5_rosetta()
    table = summary_table(cells)
    notes = checks(rows, cells)
    SUMMARY.write_text(
        "# Mac v1 results summary\n\n"
        "Generated by `scripts/analyze.py` from `results/mac-v1/results.csv` and `results/mac-v1/raw/`. "
        "Spread is across repeats (one container launch each); pooled p99 uses every timed iteration of "
        "every repeat. \"Periods throttled\" is `cgroup_nr_throttled / cgroup_nr_periods` over the timed "
        "loop. \"CPU ms per inference\" is `cgroup_usage_usec / iters`: the container's CPU time per image, "
        "all threads. \"Threads added\" is the OS threads ORT's session added (pool size − 1).\n\n"
        + table + "\n\n" + notes + "\n"
    )
    print(f"wrote {SUMMARY.relative_to(ROOT)}")
    print(f"default/matched p50 ratios (cpus 1,2,4): {', '.join(f'{r:.2f}' for r in ratios)}")
    print(f"int8 speedups at matched threads (cpus 1,2,4): {', '.join(f'{s:.2f}' for s in speedups)}")
    print(table)
    print(notes)


if __name__ == "__main__":
    main()
