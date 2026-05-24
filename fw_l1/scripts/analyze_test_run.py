"""Render an HTML report (with trend charts + summary) from a TestRunner output dir.

Expected input directory layout (produced by the Android TestRunner):

    <run_dir>/
        results.csv     # per-prompt verdicts
        resources.csv   # periodic memory/CPU snapshots
        summary.txt     # human-readable aggregate
        config.json     # test configuration

Usage:

    uv run --python 3.13 --with pandas --with matplotlib \
        python scripts/analyze_test_run.py /path/to/test_runs/20260513_001500

Output:

    <run_dir>/report.html   (self-contained, charts embedded as base64 PNG)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "Helvetica Neue, Arial, sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 110,
})

ACCENT = "#0D47A1"
GREEN = "#1B5E20"
AMBER = "#B26500"
RED = "#B3261E"
NEUTRAL = "#555555"

GATE_COLORS = {"ALLOW": "#0B57D0", "STRIP": "#B26500", "BLOCK": "#B3261E"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _png(fig) -> str:
    """Render `fig` to a base64 data-URI PNG and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _percentile(s: pd.Series, q: float) -> float:
    return float(np.percentile(s.values, q * 100)) if len(s) else 0.0


# ----------------------------------------------------------------------------
# chart builders
# ----------------------------------------------------------------------------
def chart_latency_over_time(df: pd.DataFrame) -> str:
    # Cold-start outliers (e.g. the OrtSession warm-up on prompt 0) sit
    # one or two orders of magnitude above steady-state latency and squash
    # the trend visualization. Exclude any point >10x p99 from BOTH the
    # scatter and the rolling mean so the chart shows the actual trend;
    # surface the excluded points in the title instead of hiding them.
    p99 = float(df["latency_ms"].quantile(0.99))
    threshold = max(p99 * 10, 200.0)
    keep = df["latency_ms"] <= threshold
    trend = df[keep]
    excluded = df[~keep]

    fig, ax = plt.subplots(figsize=(11, 3.8))
    for gate, sub in trend.groupby("gate"):
        ax.scatter(
            sub["idx"], sub["latency_ms"],
            s=12, alpha=0.55, color=GATE_COLORS.get(gate, NEUTRAL), label=gate,
        )
    window = max(5, len(trend) // 50)
    rolling = trend["latency_ms"].rolling(window=window, min_periods=1).mean()
    ax.plot(trend["idx"], rolling, color=ACCENT, linewidth=1.6,
            label=f"rolling mean (window = {window} prompts)")
    ax.set_xlabel("prompt index (test order)")
    ax.set_ylabel("gate latency (ms)")

    if len(excluded):
        notes = ", ".join(
            f"idx={int(r.idx)} → {int(r.latency_ms)} ms"
            for r in excluded.itertuples()
        )
        title = (
            f"Latency over time — per prompt + rolling mean   "
            f"(excluded {len(excluded)} cold-start outlier"
            f"{'s' if len(excluded) != 1 else ''}: {notes})"
        )
    else:
        title = "Latency over time — per prompt + rolling mean"
    ax.set_title(title, fontsize=11)
    # Legend below the plot so it never overlaps the data points or the
    # rolling-mean line.
    ax.legend(frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncols=4)
    return _png(fig)


def chart_latency_hist(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bins = np.linspace(0, df["latency_ms"].quantile(0.99), 40)
    ax.hist(df["latency_ms"], bins=bins, color=ACCENT, alpha=0.85)
    ax.set_xlabel("gate latency (ms)")
    ax.set_ylabel("prompt count")
    ax.set_title("Latency distribution")
    for q, color in [(0.50, GREEN), (0.95, AMBER), (0.99, RED)]:
        v = _percentile(df["latency_ms"], q)
        ax.axvline(v, color=color, linestyle="--", linewidth=1, alpha=0.85)
        ax.text(v, ax.get_ylim()[1] * 0.92, f" p{int(q * 100)}={v:.0f}ms",
                color=color, fontsize=8, va="top")
    return _png(fig)


def chart_memory_over_time(res: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(res["timestamp_ms"] / 1000, res["pss_total_mb"], label="PSS total",
            color=ACCENT, linewidth=1.8)
    ax.plot(res["timestamp_ms"] / 1000, res["pss_native_mb"], label="PSS native",
            color="#0B57D0", linestyle="--", linewidth=1.3)
    ax.plot(res["timestamp_ms"] / 1000, res["java_heap_mb"], label="Java heap",
            color=AMBER, linewidth=1.3)
    ax.plot(res["timestamp_ms"] / 1000, res["native_alloc_mb"], label="native alloc",
            color="#777777", linewidth=1.1, alpha=0.7)
    ax.set_xlabel("test elapsed (s)")
    ax.set_ylabel("MB")
    ax.set_title("Memory over time")
    ax.legend(frameon=False, ncols=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    return _png(fig)


def chart_cpu_over_time(res: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(res["timestamp_ms"] / 1000, res["cpu_percent"], color=ACCENT, linewidth=1.6)
    ax.fill_between(res["timestamp_ms"] / 1000, 0, res["cpu_percent"],
                    color=ACCENT, alpha=0.12)
    ax.set_xlabel("test elapsed (s)")
    ax.set_ylabel("CPU % (process, all cores)")
    ax.set_title("Process CPU over time")
    return _png(fig)


def chart_threads(res: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(res["timestamp_ms"] / 1000, res["thread_count"], color=ACCENT,
            linewidth=1.6, marker="o", markersize=3)
    ax.set_xlabel("test elapsed (s)")
    ax.set_ylabel("active thread count")
    ax.set_title("Threads over time")
    return _png(fig)


def chart_gate_distribution(df: pd.DataFrame) -> str:
    counts = df["gate"].value_counts().reindex(["ALLOW", "STRIP", "BLOCK"]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bars = ax.bar(counts.index, counts.values,
                  color=[GATE_COLORS[g] for g in counts.index])
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f" {v}\n({v / len(df) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("prompts")
    ax.set_title("Gate decisions across the run")
    return _png(fig)


def chart_per_source_accuracy(df: pd.DataFrame) -> str:
    df = df.copy()
    df["correct"] = np.where(
        df["expected_action"] == "allow",
        df["gate"] == "ALLOW",
        df["gate"] != "ALLOW",
    )
    by = df.groupby("source")["correct"].agg(["sum", "count"])
    by["accuracy"] = by["sum"] / by["count"] * 100
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bars = ax.bar(by.index, by["accuracy"], color=ACCENT, alpha=0.85)
    for bar, acc, count in zip(bars, by["accuracy"], by["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, acc,
                f" {acc:.1f}%\nn={int(count)}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("binary accuracy (%)")
    ax.set_title("Binary accuracy by source")
    return _png(fig)


def chart_confusion(df: pd.DataFrame) -> str:
    # Order both axes by permissiveness (allow -> strip -> block) so the
    # diagonal cells (top-left to bottom-right) represent the "correct"
    # gate decision for each expected_action. Off-diagonal cells above
    # the diagonal are over-restrictive (false positives); cells below
    # the diagonal are under-restrictive (false negatives).
    row_order = ["expected allow", "expected strip", "expected block"]
    col_order = ["ALLOW", "STRIP", "BLOCK"]

    matrix = pd.crosstab(
        df["expected_action"].replace({
            "allow": "expected allow",
            "strip": "expected strip",
            "block": "expected block",
        }),
        df["gate"],
        dropna=False,
    )
    for r in row_order:
        if r not in matrix.index:
            matrix.loc[r] = 0
    for c in col_order:
        if c not in matrix.columns:
            matrix[c] = 0
    matrix = matrix.reindex(row_order)[col_order].astype(int)

    # Margins so the reader can verify the matrix sums against the
    # "Gate decisions across the run" bar chart (column totals match it).
    row_totals = matrix.sum(axis=1).values
    col_totals = matrix.sum(axis=0).values

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    im = ax.imshow(matrix.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    # Encode each row's total directly in the y-tick label, e.g.
    # "expected allow (n=1,000)".
    ax.set_yticklabels([
        f"{lbl} (n={tot:,})" for lbl, tot in zip(matrix.index, row_totals)
    ])
    for (i, j), v in np.ndenumerate(matrix.values):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > matrix.values.max() / 2 else "black",
                fontsize=11, fontweight="bold")
    col_total_str = "  ·  ".join(
        f"{c}={t:,}" for c, t in zip(matrix.columns, col_totals)
    )
    ax.set_title(
        "Expected action × gate decision (diagonal = correct)\n"
        f"Column totals: {col_total_str}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.04)
    return _png(fig)


def chart_per_class(df: pd.DataFrame) -> str:
    classes = sorted(df["expected_category"].unique())
    pivot = pd.crosstab(df["expected_category"], df["gate"]).reindex(classes).fillna(0).astype(int)
    for col in ("ALLOW", "STRIP", "BLOCK"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["ALLOW", "STRIP", "BLOCK"]]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bottom = np.zeros(len(pivot))
    for col in pivot.columns:
        ax.bar(pivot.index, pivot[col], bottom=bottom,
               label=col, color=GATE_COLORS[col])
        bottom += pivot[col].values
    ax.set_ylabel("prompts")
    ax.set_title("Per-class gate breakdown")
    # Headroom so the legend can't crash into the tallest bar.
    ax.set_ylim(0, bottom.max() * 1.05)
    ax.legend(frameon=False, ncols=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    return _png(fig)


# ----------------------------------------------------------------------------
# HTML composition
# ----------------------------------------------------------------------------
@dataclass
class Stats:
    n: int
    elapsed_s: float
    throughput: float
    mean: float
    p50: float
    p95: float
    p99: float
    max: float
    allow: int
    strip: int
    block: int
    fp: int
    fn: int
    accuracy: float


def compute_stats(df: pd.DataFrame) -> Stats:
    n = len(df)
    elapsed_s = (df["timestamp_ms"].max() + df["latency_ms"].iloc[-1]) / 1000 if n else 0.0
    correct = np.where(
        df["expected_action"] == "allow",
        df["gate"] == "ALLOW",
        df["gate"] != "ALLOW",
    ).sum()
    return Stats(
        n=n,
        elapsed_s=elapsed_s,
        throughput=n / elapsed_s if elapsed_s > 0 else 0.0,
        mean=float(df["latency_ms"].mean()),
        p50=_percentile(df["latency_ms"], 0.50),
        p95=_percentile(df["latency_ms"], 0.95),
        p99=_percentile(df["latency_ms"], 0.99),
        max=float(df["latency_ms"].max()),
        allow=int((df["gate"] == "ALLOW").sum()),
        strip=int((df["gate"] == "STRIP").sum()),
        block=int((df["gate"] == "BLOCK").sum()),
        fp=int(((df["expected_action"] == "allow") & (df["gate"] != "ALLOW")).sum()),
        fn=int(((df["expected_action"] == "block") & (df["gate"] == "ALLOW")).sum()),
        accuracy=correct / n * 100 if n else 0.0,
    )


CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         color: #1F1F1F; max-width: 1100px; margin: 32px auto; padding: 0 24px; }
  h1 { color: #0D47A1; border-bottom: 2px solid #0D47A1; padding-bottom: 8px; }
  h2 { color: #0D47A1; margin-top: 36px; }
  .meta { color: #555; font-size: 14px; }
  .summary { background: #F5F7FB; border-left: 4px solid #0D47A1; padding: 14px 20px; margin: 16px 0 24px; border-radius: 4px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
  .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 18px 0; }
  .stat { background: #fff; border: 1px solid #E0E0E0; border-radius: 6px; padding: 12px 14px; }
  .stat .label { color: #777; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  .stat .value { font-size: 22px; font-weight: 600; color: #0D47A1; margin-top: 4px; }
  .stat .sub { color: #555; font-size: 12px; margin-top: 2px; }
  img.chart { max-width: 100%; border: 1px solid #EAEAEA; border-radius: 4px; margin: 8px 0 18px; }
  pre.summary-block { background: #F8F8F8; border: 1px solid #E0E0E0; border-radius: 4px;
                      padding: 12px 16px; font-size: 12px; overflow-x: auto; }
  table { border-collapse: collapse; margin: 8px 0 16px; width: 100%; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #EAEAEA; font-size: 13px; }
  th { background: #F5F7FB; color: #0D47A1; }
  .footer { color: #777; font-size: 12px; margin-top: 48px; padding-top: 18px;
            border-top: 1px solid #EAEAEA; }
  .verdict-ok { color: #1B5E20; font-weight: 600; }
  .verdict-warn { color: #B26500; font-weight: 600; }
</style>
"""


def render_html(run_dir: Path, df: pd.DataFrame, res: pd.DataFrame,
                summary_text: str, config: dict) -> str:
    s = compute_stats(df)

    charts = {
        "latency_over_time": chart_latency_over_time(df),
        "latency_hist": chart_latency_hist(df),
        "memory": chart_memory_over_time(res),
        "cpu": chart_cpu_over_time(res),
        "threads": chart_threads(res),
        "gate_dist": chart_gate_distribution(df),
        "per_source": chart_per_source_accuracy(df),
        "confusion": chart_confusion(df),
        "per_class": chart_per_class(df),
    }

    # Per-source rows
    df2 = df.copy()
    df2["correct"] = np.where(
        df2["expected_action"] == "allow",
        df2["gate"] == "ALLOW",
        df2["gate"] != "ALLOW",
    )
    per_src = df2.groupby("source").agg(
        n=("idx", "count"),
        correct=("correct", "sum"),
        allow=("gate", lambda x: (x == "ALLOW").sum()),
        strip=("gate", lambda x: (x == "STRIP").sum()),
        block=("gate", lambda x: (x == "BLOCK").sum()),
    )
    per_src["accuracy"] = per_src["correct"] / per_src["n"] * 100
    per_src_rows = "".join(
        f"<tr><td>{src}</td><td>{r.n}</td><td>{r.correct}</td>"
        f"<td>{r.accuracy:.1f}%</td><td>{r.allow}</td>"
        f"<td>{r.strip}</td><td>{r.block}</td></tr>"
        for src, r in per_src.iterrows()
    )

    pss_min = int(res["pss_total_mb"].min())
    pss_max = int(res["pss_total_mb"].max())
    pss_drift = pss_max - pss_min
    cpu_mean = float(res["cpu_percent"].mean())
    cpu_max = float(res["cpu_percent"].max())
    thread_max = int(res["thread_count"].max())
    thread_min = int(res["thread_count"].min())

    pss_verdict = ("verdict-ok" if pss_drift <= 30
                   else "verdict-warn")
    cpu_verdict = "verdict-ok" if cpu_max < 60 else "verdict-warn"

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FW-L1 test run — {run_dir.name}</title>
{CSS}
</head><body>

<h1>FW-L1 automated test run</h1>
<div class="meta">
  Run id: <strong>{run_dir.name}</strong> &nbsp;·&nbsp;
  N = {s.n} &nbsp;·&nbsp; elapsed {s.elapsed_s:.1f} s &nbsp;·&nbsp;
  throughput {s.throughput:.1f}/s
</div>

<div class="summary">
  <strong>Headline:</strong> binary accuracy <strong>{s.accuracy:.1f}%</strong>
  ({s.fp} false positive, {s.fn} false negative).
  Process stayed within <strong>{pss_min} – {pss_max} MB PSS</strong>
  (<span class="{pss_verdict}">{pss_drift} MB drift</span>),
  CPU mean <strong>{cpu_mean:.1f}%</strong>
  (<span class="{cpu_verdict}">peak {cpu_max:.1f}%</span>),
  thread count {thread_min}–{thread_max}.
</div>

<div class="stat-grid">
  <div class="stat"><div class="label">Prompts</div><div class="value">{s.n}</div>
       <div class="sub">benign / adversarial / compound</div></div>
  <div class="stat"><div class="label">Throughput</div><div class="value">{s.throughput:.1f}/s</div>
       <div class="sub">over {s.elapsed_s:.1f} s</div></div>
  <div class="stat"><div class="label">Latency p50</div><div class="value">{s.p50:.0f} ms</div>
       <div class="sub">mean {s.mean:.0f} · max {s.max:.0f}</div></div>
  <div class="stat"><div class="label">Latency p95</div><div class="value">{s.p95:.0f} ms</div>
       <div class="sub">p99 {s.p99:.0f} ms</div></div>
  <div class="stat"><div class="label">ALLOW</div><div class="value">{s.allow}</div>
       <div class="sub">{s.allow / s.n * 100:.1f}%</div></div>
  <div class="stat"><div class="label">STRIP</div><div class="value">{s.strip}</div>
       <div class="sub">{s.strip / s.n * 100:.1f}%</div></div>
  <div class="stat"><div class="label">BLOCK</div><div class="value">{s.block}</div>
       <div class="sub">{s.block / s.n * 100:.1f}%</div></div>
  <div class="stat"><div class="label">Accuracy</div><div class="value">{s.accuracy:.1f}%</div>
       <div class="sub">fp={s.fp} · fn={s.fn}</div></div>
</div>

<h2>Latency trends</h2>
<img class="chart" src="{charts['latency_over_time']}"/>
<div class="grid2">
  <img class="chart" src="{charts['latency_hist']}"/>
  <img class="chart" src="{charts['gate_dist']}"/>
</div>

<h2>Resource trends</h2>
<img class="chart" src="{charts['memory']}"/>
<img class="chart" src="{charts['cpu']}"/>
<div class="grid2">
  <img class="chart" src="{charts['threads']}"/>
  <img class="chart" src="{charts['confusion']}"/>
</div>

<h2>Classification breakdown</h2>
<img class="chart" src="{charts['per_class']}"/>
<img class="chart" src="{charts['per_source']}"/>

<h2>Per-source detail</h2>
<table>
  <tr><th>source</th><th>n</th><th>correct</th><th>accuracy</th>
      <th>ALLOW</th><th>STRIP</th><th>BLOCK</th></tr>
  {per_src_rows}
</table>

<h2>Test configuration</h2>
<pre class="summary-block">{json.dumps(config, indent=2)}</pre>

<h2>Raw summary (from on-device run)</h2>
<pre class="summary-block">{summary_text}</pre>

<div class="footer">
  Generated by <code>mobile-rag-firewall/fw_l1/scripts/analyze_test_run.py</code>.
  Source CSVs: <code>results.csv</code>, <code>resources.csv</code>.
  Caveats: emulator-class hardware will outperform low-end real devices;
  network /query calls are not exercised by this harness.
</div>

</body></html>
"""
    return html


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path,
                    help="Path to a TestRunner output dir (containing results.csv etc.)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="HTML output path (default: <run_dir>/report.html)")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")
    results_csv = run_dir / "results.csv"
    resources_csv = run_dir / "resources.csv"
    summary_txt = run_dir / "summary.txt"
    config_json = run_dir / "config.json"
    for f in (results_csv, resources_csv):
        if not f.exists():
            sys.exit(f"missing: {f}")

    df = pd.read_csv(results_csv)
    res = pd.read_csv(resources_csv)
    summary_text = summary_txt.read_text() if summary_txt.exists() else "(missing)"
    config = json.loads(config_json.read_text()) if config_json.exists() else {}

    html = render_html(run_dir, df, res, summary_text, config)
    out = args.output or (run_dir / "report.html")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html) // 1024} KB, {len(df)} prompts, {len(res)} resource samples)")


if __name__ == "__main__":
    main()
