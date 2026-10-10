#!/usr/bin/env python3
"""
scripts/analyze_baseline.py

Scientific analysis and visualization pipeline for SyntraQ baseline traffic experiments.
Uses Pandas to calculate descriptive statistics (mean, std, min, max) and Matplotlib
to produce publication-quality comparison charts across traffic scenarios:
- Low Traffic
- Medium Traffic
- High Traffic
- Rush Hour
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set modern, publication-ready visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
})

# Standard scenario ordering and curated palette
SCENARIO_ORDER = ["Low Traffic", "Medium Traffic", "High Traffic", "Rush Hour"]
SCENARIO_COLORS = {
    "Low Traffic": "#2b9348",     # Calm green
    "Medium Traffic": "#0077b6",  # Ocean blue
    "High Traffic": "#e85d04",    # Warning orange
    "Rush Hour": "#d00000",       # Alert red
}


def run_experiment_binary(duration_s=180.0, num_seeds=10, out_dir="data/experiments"):
    """Invokes the compiled C++ syntraq_experiments binary if CSV is missing."""
    bin_candidates = [
        Path("build/bin/syntraq_experiments.exe"),
        Path("build/bin/syntraq_experiments"),
        Path("build/syntraq_experiments.exe"),
        Path("build/syntraq_experiments"),
    ]
    binary = None
    for cand in bin_candidates:
        if cand.exists():
            binary = cand
            break

    if not binary:
        raise FileNotFoundError(
            f"Could not locate syntraq_experiments binary. Looked in: {[str(c) for c in bin_candidates]}"
        )

    cmd = [
        str(binary),
        "--duration", str(duration_s),
        "--num-seeds", str(num_seeds),
        "--out-dir", str(out_dir),
    ]
    print(f"[SyntraQ Analysis] Executing experiment suite: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def calculate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates mean, std, min, and max for each metric grouped by scenario."""
    metrics = [
        ("avg_speed_kmh", "Average Speed (km/h)"),
        ("avg_waiting_time_s", "Average Waiting Time (s)"),
        ("avg_travel_time_s", "Average Travel Time (s)"),
        ("max_queue_length", "Max Queue Length (veh)"),
        ("avg_queue_length", "Average Queue Length (veh)"),
        ("throughput_vph", "Throughput (veh/h)"),
        ("congestion_ratio", "Congestion Ratio"),
    ]

    records = []
    for sc in SCENARIO_ORDER:
        sub = df[df["scenario"] == sc]
        if sub.empty:
            continue
        n = len(sub)
        for col, label in metrics:
            if col not in sub.columns:
                continue
            series = sub[col].dropna()
            records.append({
                "scenario": sc,
                "metric": col,
                "metric_label": label,
                "count": n,
                "mean": series.mean(),
                "std": series.std(ddof=1) if len(series) > 1 else 0.0,
                "min": series.min(),
                "max": series.max(),
            })

    return pd.DataFrame(records)


def plot_speed_and_throughput(df: pd.DataFrame, stats: pd.DataFrame, out_path: Path):
    """Plots Speed and Throughput comparisons with standard deviation error bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]
    colors = [SCENARIO_COLORS.get(s, "#333333") for s in scenarios]
    x = np.arange(len(scenarios))
    bar_width = 0.55

    # ── Panel 1: Average Speed ─────────────────────────
    spd_stats = stats[stats["metric"] == "avg_speed_kmh"].set_index("scenario").reindex(scenarios)
    means = spd_stats["mean"].values
    stds = spd_stats["std"].values

    bars1 = ax1.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.88,
                    edgecolor="#222222", width=bar_width, zorder=3)
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title("Average Vehicle Travel Speed")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Label values
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                     xytext=(0, 0), textcoords="offset points",
                     ha="center", va="center", color="white", weight="bold", fontsize=10)

    # ── Panel 2: Throughput ────────────────────────────
    thr_stats = stats[stats["metric"] == "throughput_vph"].set_index("scenario").reindex(scenarios)
    means2 = thr_stats["mean"].values
    stds2 = thr_stats["std"].values

    bars2 = ax2.bar(x, means2, yerr=stds2, capsize=6, color=colors, alpha=0.88,
                    edgecolor="#222222", width=bar_width, zorder=3)
    ax2.set_ylabel("Throughput (veh / hour)")
    ax2.set_title("Network Throughput (Completed Vehicles/Hour)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{int(round(h))}",
                     xy=(bar.get_x() + bar.get_width() / 2, max(20.0, h / 2)),
                     xytext=(0, 0), textcoords="offset points",
                     ha="center", va="center", color="white", weight="bold", fontsize=10)

    fig.suptitle("SyntraQ Baseline Traffic Evaluation — Speed & Capacity", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


def plot_delay_and_waiting(df: pd.DataFrame, stats: pd.DataFrame, out_path: Path):
    """Plots Average Waiting Time vs Total Travel Time across scenarios."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]
    x = np.arange(len(scenarios))
    bar_width = 0.35

    wait_stats = stats[stats["metric"] == "avg_waiting_time_s"].set_index("scenario").reindex(scenarios)
    trav_stats = stats[stats["metric"] == "avg_travel_time_s"].set_index("scenario").reindex(scenarios)

    ax.bar(x - bar_width/2, wait_stats["mean"].values, yerr=wait_stats["std"].values,
           width=bar_width, capsize=5, label="Waiting Time (in Queue)", color="#d90429", alpha=0.85,
           edgecolor="#222222", zorder=3)
    ax.bar(x + bar_width/2, trav_stats["mean"].values, yerr=trav_stats["std"].values,
           width=bar_width, capsize=5, label="Total Trip Travel Time", color="#0077b6", alpha=0.85,
           edgecolor="#222222", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Duration (seconds)")
    ax.set_title("Average Waiting Time vs. Total Travel Time by Scenario")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


def plot_queues_and_congestion(df: pd.DataFrame, stats: pd.DataFrame, out_path: Path):
    """Plots queue statistics and fleet congestion percentage across scenarios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]
    x = np.arange(len(scenarios))
    bar_width = 0.35

    # ── Panel 1: Queue Lengths (Avg vs Peak Max) ───────
    avg_q = stats[stats["metric"] == "avg_queue_length"].set_index("scenario").reindex(scenarios)
    max_q = stats[stats["metric"] == "max_queue_length"].set_index("scenario").reindex(scenarios)

    ax1.bar(x - bar_width/2, avg_q["mean"].values, yerr=avg_q["std"].values, width=bar_width,
            capsize=5, label="Average Network Queue (Vehicles)", color="#f77f00", alpha=0.88,
            edgecolor="#222222", zorder=3)
    ax1.bar(x + bar_width/2, max_q["mean"].values, yerr=max_q["std"].values, width=bar_width,
            capsize=5, label="Peak Single-Lane Queue (Vehicles)", color="#d62828", alpha=0.88,
            edgecolor="#222222", zorder=3)

    ax1.set_ylabel("Queue Depth (vehicles)")
    ax1.set_title("Intersection Queue Formation")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend(frameon=True, loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # ── Panel 2: Congestion Ratio (%) ──────────────────
    colors = [SCENARIO_COLORS.get(s, "#333333") for s in scenarios]
    cong = stats[stats["metric"] == "congestion_ratio"].set_index("scenario").reindex(scenarios)
    cong_pct = cong["mean"].values * 100.0
    cong_err = cong["std"].values * 100.0

    bars = ax2.bar(x, cong_pct, yerr=cong_err, capsize=6, color=colors, alpha=0.88,
                   edgecolor="#222222", width=0.55, zorder=3)
    ax2.set_ylabel("Congestion Ratio (%)")
    ax2.set_title("Fleet Congestion Ratio (Queued / Active Fleet)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.set_ylim(0, max(25.0, (cong_pct + cong_err).max() * 1.25))
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, max(1.0, h / 2)),
                     xytext=(0, 0), textcoords="offset points",
                     ha="center", va="center", color="white", weight="bold", fontsize=10)

    fig.suptitle("SyntraQ Baseline Traffic Evaluation — Queuing & Congestion Dynamics", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


def plot_comprehensive_dashboard(df: pd.DataFrame, stats: pd.DataFrame, out_path: Path):
    """Produces a multi-panel publication dashboard summarizing all baseline benchmarks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]
    colors = [SCENARIO_COLORS.get(s, "#333333") for s in scenarios]
    x = np.arange(len(scenarios))
    w = 0.52

    # 1. Speed
    spd = stats[stats["metric"] == "avg_speed_kmh"].set_index("scenario").reindex(scenarios)
    axes[0, 0].bar(x, spd["mean"].values, yerr=spd["std"].values, capsize=5, color=colors,
                   alpha=0.88, edgecolor="#222", width=w, zorder=3)
    axes[0, 0].set_title("A. Average Speed (km/h)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(scenarios)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # 2. Delay
    wt = stats[stats["metric"] == "avg_waiting_time_s"].set_index("scenario").reindex(scenarios)
    axes[0, 1].bar(x, wt["mean"].values, yerr=wt["std"].values, capsize=5, color=colors,
                   alpha=0.88, edgecolor="#222", width=w, zorder=3)
    axes[0, 1].set_title("B. Average Waiting Time (s)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(scenarios)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # 3. Peak Queues
    mq = stats[stats["metric"] == "max_queue_length"].set_index("scenario").reindex(scenarios)
    axes[1, 0].bar(x, mq["mean"].values, yerr=mq["std"].values, capsize=5, color=colors,
                   alpha=0.88, edgecolor="#222", width=w, zorder=3)
    axes[1, 0].set_title("C. Peak Queue Length (vehicles)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(scenarios)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # 4. Throughput
    thr = stats[stats["metric"] == "throughput_vph"].set_index("scenario").reindex(scenarios)
    axes[1, 1].bar(x, thr["mean"].values, yerr=thr["std"].values, capsize=5, color=colors,
                   alpha=0.88, edgecolor="#222", width=w, zorder=3)
    axes[1, 1].set_title("D. System Throughput (veh/h)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(scenarios)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    fig.suptitle("SyntraQ Fixed-Time Baseline Controller Benchmark Dashboard", fontsize=16, y=1.00)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SyntraQ baseline traffic experiments.")
    parser.add_argument("--runs-csv", default="data/experiments/baseline_runs.csv",
                        help="Path to raw simulation runs CSV.")
    parser.add_argument("--stats-csv", default="data/experiments/baseline_summary_stats.csv",
                        help="Path to export calculated summary statistics CSV.")
    parser.add_argument("--charts-dir", default="data/experiments/charts",
                        help="Directory to save generated chart PNGs.")
    parser.add_argument("--duration", type=float, default=180.0,
                        help="Duration in seconds per run if executing experiments.")
    parser.add_argument("--num-seeds", type=int, default=10,
                        help="Number of random seeds per scenario if executing experiments.")
    parser.add_argument("--run", action="store_true",
                        help="Force running experiments before analysis.")
    args = parser.parse_args()

    runs_csv = Path(args.runs_csv)
    stats_csv = Path(args.stats_csv)
    charts_dir = Path(args.charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.run or not runs_csv.exists():
        print(f"[SyntraQ Analysis] Runs CSV not found at '{runs_csv}'. Running experiment runner...")
        run_experiment_binary(duration_s=args.duration, num_seeds=args.num_seeds, out_dir=str(runs_csv.parent))

    print(f"[SyntraQ Analysis] Loading experimental data from: {runs_csv}")
    df = pd.read_csv(runs_csv)
    print(f"[SyntraQ Analysis] Loaded {len(df)} simulation runs across {df['scenario'].nunique()} scenarios.")

    # Compute statistical aggregates
    stats = calculate_summary_statistics(df)
    stats.to_csv(stats_csv, index=False)
    print(f"[SyntraQ Analysis] Exported statistical aggregates to: {stats_csv}")

    # Display clean table
    print("\n" + "=" * 80)
    print("                     BASELINE EXPERIMENTAL SUMMARY STATISTICS                  ")
    print("=" * 80)
    display_cols = ["scenario", "metric_label", "mean", "std", "min", "max"]
    for sc in SCENARIO_ORDER:
        sub = stats[stats["scenario"] == sc]
        if sub.empty:
            continue
        print(f"\n--- {sc} (N={sub['count'].iloc[0]}) ---")
        formatted = sub[display_cols].copy()
        formatted["mean ± std"] = formatted.apply(lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1)
        formatted["[min, max]"] = formatted.apply(lambda r: f"[{r['min']:.2f}, {r['max']:.2f}]", axis=1)
        print(formatted[["metric_label", "mean ± std", "[min, max]"]].to_string(index=False))

    # Generate charts
    print("\n[SyntraQ Analysis] Generating baseline charts...")
    plot_speed_and_throughput(df, stats, charts_dir / "speed_and_throughput.png")
    plot_delay_and_waiting(df, stats, charts_dir / "waiting_and_travel_time.png")
    plot_queues_and_congestion(df, stats, charts_dir / "queues_and_congestion.png")
    plot_comprehensive_dashboard(df, stats, charts_dir / "baseline_metrics_dashboard.png")

    print("\n[SyntraQ Analysis] Baseline experiment analysis complete.")


if __name__ == "__main__":
    main()
