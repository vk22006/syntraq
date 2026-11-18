#!/usr/bin/env python3
"""
scripts/analyze_experiments.py

SyntraQ Reproducible Experiment Analysis Pipeline.
Processes machine-readable results (CSV / JSON) produced by the experiment framework:
  1. Aggregates multi-seed simulation metrics with statistical rigor:
     - Mean, standard deviation, min, max
     - 95% Confidence Intervals (Student-t / normal)
  2. Generates comprehensive comparison markdown tables.
  3. Computes automated Controller Rankings based on multiple criteria:
     - Waiting time reduction
     - Network throughput
     - Trip duration
     - Congestion ratio
     - Peak queue mitigation
  4. Generates publication-quality dark-themed comparison plots:
     - Controller performance across traffic density & scenarios
     - Routing mode efficiency comparisons
     - Metric distributions and confidence intervals
     - Radar / ranking visual summaries

Usage:
  python scripts/analyze_experiments.py \
    --input data/experiments/reproducible/experiment_runs.csv \
    --summary data/experiments/reproducible/experiment_summary.csv \
    --json data/experiments/reproducible/experiment_results.json \
    --output-dir docs/experiments \
    --charts-dir docs/experiments/charts
"""

import os
import sys
import argparse
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="SyntraQ Experiment Analysis Pipeline")
    parser.add_argument("--input", default="data/experiments/reproducible/experiment_runs.csv",
                        help="Path to experiment_runs.csv")
    parser.add_argument("--summary", default="data/experiments/reproducible/experiment_summary.csv",
                        help="Path to experiment_summary.csv (optional)")
    parser.add_argument("--json", default="data/experiments/reproducible/experiment_results.json",
                        help="Path to experiment_results.json (optional)")
    parser.add_argument("--output-dir", default="docs/experiments",
                        help="Directory to save generated markdown report and tables")
    parser.add_argument("--charts-dir", default="docs/experiments/charts",
                        help="Directory to save generated plots")
    return parser.parse_args()

def setup_plot_style():
    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#444455'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#2A2D3A'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6
    plt.rcParams['figure.facecolor'] = '#12141C'
    plt.rcParams['axes.facecolor'] = '#181B26'

def compute_ci95(series):
    """Compute 95% confidence interval half-width using Student-t distribution."""
    vals = series.dropna().values
    n = len(vals)
    if n <= 1:
        return 0.0
    mean = np.mean(vals)
    std_err = stats.sem(vals)
    if std_err == 0.0 or math.isnan(std_err):
        return 0.0
    t_val = stats.t.ppf(0.975, df=n - 1)
    return float(t_val * std_err)

def calculate_controller_rankings(df_runs):
    """
    Ranks controllers across scenarios and routing modes using a normalized Borda-like composite score.
    Metrics evaluated:
      - avg_waiting_time_s (lower better)
      - avg_travel_time_s (lower better)
      - throughput_vph (higher better)
      - congestion_ratio (lower better)
      - avg_queue_length (lower better)
    """
    controllers = df_runs['controller'].unique().tolist()
    scenarios = df_runs['scenario'].unique().tolist()
    routings = df_runs['routing_mode'].unique().tolist()

    records = []
    for sc in scenarios:
        for rm in routings:
            subset = df_runs[(df_runs['scenario'] == sc) & (df_runs['routing_mode'] == rm)]
            if subset.empty:
                continue

            ctrl_means = subset.groupby('controller')[
                ['avg_waiting_time_s', 'avg_travel_time_s', 'throughput_vph', 'congestion_ratio', 'avg_queue_length']
            ].mean()

            # Rank each metric (1 is best)
            wait_rank = ctrl_means['avg_waiting_time_s'].rank(ascending=True)
            travel_rank = ctrl_means['avg_travel_time_s'].rank(ascending=True)
            thr_rank = ctrl_means['throughput_vph'].rank(ascending=False)
            cong_rank = ctrl_means['congestion_ratio'].rank(ascending=True)
            queue_rank = ctrl_means['avg_queue_length'].rank(ascending=True)

            composite_rank = (wait_rank + travel_rank + thr_rank + cong_rank + queue_rank) / 5.0

            for ctrl in ctrl_means.index:
                records.append({
                    'scenario': sc,
                    'routing_mode': rm,
                    'controller': ctrl,
                    'mean_wait_s': ctrl_means.loc[ctrl, 'avg_waiting_time_s'],
                    'mean_travel_s': ctrl_means.loc[ctrl, 'avg_travel_time_s'],
                    'mean_throughput_vph': ctrl_means.loc[ctrl, 'throughput_vph'],
                    'mean_congestion_pct': ctrl_means.loc[ctrl, 'congestion_ratio'] * 100.0,
                    'mean_queue': ctrl_means.loc[ctrl, 'avg_queue_length'],
                    'score': composite_rank.loc[ctrl]
                })

    df_ranked = pd.DataFrame(records)
    overall_ranking = df_ranked.groupby('controller')['score'].mean().sort_values()
    return df_ranked, overall_ranking

def plot_comprehensive_dashboard(df_runs, charts_dir):
    """Creates a 4-panel multi-scenario comparative dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor('#12141C')

    controllers = ['FixedTime', 'Adaptive', 'ReinforcementLearning', 'Coordinated']
    # Filter to controllers present in the run data
    controllers = [c for c in controllers if c in df_runs['controller'].unique()]

    color_map = {
        'FixedTime': '#EF5350',             # Red
        'Adaptive': '#42A5F5',              # Sky Blue
        'ReinforcementLearning': '#AB47BC', # Purple
        'Coordinated': '#26A69A'            # Teal
    }

    scenarios = [s for s in ['Low Traffic', 'Medium Traffic', 'High Traffic', 'Rush Hour', 'Emergency Priority']
                 if s in df_runs['scenario'].unique()]
    if not scenarios:
        scenarios = df_runs['scenario'].unique().tolist()

    x = np.arange(len(scenarios))
    bar_width = 0.8 / max(1, len(controllers))

    panels = [
        ('avg_waiting_time_s', 'Average Waiting Time (s)', 'Lower is Better', False, axes[0, 0]),
        ('throughput_vph', 'Throughput (veh/h)', 'Higher is Better', True, axes[0, 1]),
        ('avg_travel_time_s', 'Average Travel Time (s)', 'Lower is Better', False, axes[1, 0]),
        ('congestion_ratio', 'Congestion Ratio (Stopped/Total)', 'Lower is Better', False, axes[1, 1])
    ]

    for col, title, subtitle, higher_better, ax in panels:
        ax.set_facecolor('#181B26')
        ax.grid(True, zorder=0)

        for idx, ctrl in enumerate(controllers):
            means = []
            ci_errs = []
            for sc in scenarios:
                sub = df_runs[(df_runs['scenario'] == sc) & (df_runs['controller'] == ctrl)][col]
                if not sub.empty:
                    m = sub.mean()
                    if col == 'congestion_ratio':
                        m *= 100.0
                    means.append(m)
                    err = compute_ci95(sub * 100.0 if col == 'congestion_ratio' else sub)
                    ci_errs.append(err)
                else:
                    means.append(0.0)
                    ci_errs.append(0.0)

            offset = (idx - (len(controllers) - 1) / 2.0) * bar_width
            color = color_map.get(ctrl, '#888888')
            ax.bar(x + offset, means, bar_width, yerr=ci_errs, label=ctrl,
                   color=color, alpha=0.9, capsize=3, zorder=3)

        ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight='bold', color='#E0E0E0', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=9, color='#CCCCCC')
        ax.tick_params(colors='#AAAAAA')
        if col == 'congestion_ratio':
            ax.set_ylabel("Congestion %", color='#AAAAAA')
        ax.legend(framealpha=0.4, loc='upper left' if not higher_better else 'upper right', fontsize=8)

    plt.suptitle("SyntraQ Multi-Controller Performance Benchmark with 95% Confidence Intervals",
                 fontsize=14, fontweight='bold', color='#FFFFFF', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(charts_dir, "controller_performance_dashboard.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Analysis] Saved dashboard plot to: {out_path}")

def plot_routing_interaction(df_runs, charts_dir):
    """Compares Static vs TrafficAware routing across controllers and scenarios."""
    if 'TrafficAware' not in df_runs['routing_mode'].unique():
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    fig.patch.set_facecolor('#12141C')

    for idx, metric, title in [(0, 'avg_travel_time_s', 'Trip Duration (s) by Routing Mode'),
                               (1, 'avg_waiting_time_s', 'Waiting Time (s) by Routing Mode')]:
        ax = axes[idx]
        ax.set_facecolor('#181B26')
        ax.grid(True, zorder=0)

        controllers = [c for c in ['FixedTime', 'Adaptive', 'ReinforcementLearning', 'Coordinated']
                       if c in df_runs['controller'].unique()]
        x = np.arange(len(controllers))
        width = 0.35

        static_means = []
        static_errs = []
        aware_means = []
        aware_errs = []

        for ctrl in controllers:
            sub_s = df_runs[(df_runs['controller'] == ctrl) & (df_runs['routing_mode'] == 'Static')][metric]
            sub_a = df_runs[(df_runs['controller'] == ctrl) & (df_runs['routing_mode'] == 'TrafficAware')][metric]

            static_means.append(sub_s.mean() if not sub_s.empty else 0.0)
            static_errs.append(compute_ci95(sub_s) if not sub_s.empty else 0.0)
            aware_means.append(sub_a.mean() if not sub_a.empty else 0.0)
            aware_errs.append(compute_ci95(sub_a) if not sub_a.empty else 0.0)

        rects1 = ax.bar(x - width/2, static_means, width, yerr=static_errs, label='Static Routing',
                        color='#FFB74D', alpha=0.9, capsize=3, zorder=3)
        rects2 = ax.bar(x + width/2, aware_means, width, yerr=aware_errs, label='Traffic-Aware Routing',
                        color='#4DD0E1', alpha=0.9, capsize=3, zorder=3)

        # Delta annotations
        for r1, r2, sm, am in zip(rects1, rects2, static_means, aware_means):
            if sm > 0.0:
                pct = ((am - sm) / sm) * 100.0
                sign = "+" if pct > 0 else ""
                color = "#69F0AE" if pct < 0 else "#FF5252"
                ax.annotate(f"{sign}{pct:.1f}%",
                            xy=(r2.get_x() + r2.get_width() / 2, r2.get_height()),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)

        ax.set_title(title, fontsize=11, fontweight='bold', color='#E0E0E0', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(controllers, fontsize=9, color='#CCCCCC')
        ax.tick_params(colors='#AAAAAA')
        ax.legend(framealpha=0.4, loc='upper right', fontsize=9)

    plt.suptitle("Routing Strategy Impact on Signal Controllers (Pooled Scenarios)",
                 fontsize=13, fontweight='bold', color='#FFFFFF', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(charts_dir, "routing_mode_comparison.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Analysis] Saved routing comparison plot to: {out_path}")

def generate_markdown_report(df_runs, df_ranked, overall_ranking, output_path):
    """Generates an extensive Markdown analysis report with tables and rankings."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# SyntraQ Reproducible Experiment Analysis Report\n\n")
        f.write("This report provides an automated, statistically rigorous evaluation of traffic signal controllers "
                "and routing strategies under identical scenario conditions.\n\n")

        # Section 1: Executive Summary & Controller Ranking
        f.write("## 1. Controller Overall Rankings\n\n")
        f.write("Controllers are ranked using a multi-criteria composite score across all scenarios and routing modes "
                "(evaluating Waiting Time, Trip Duration, Network Throughput, Congestion Ratio, and Queue Depths). "
                "**Lower composite rank indicates superior overall performance.**\n\n")

        f.write("| Rank | Controller | Composite Score (1=Best) | Avg Waiting Time (s) | Avg Travel Time (s) | Avg Throughput (veh/h) | Congestion % |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")

        for rank_idx, (ctrl, score) in enumerate(overall_ranking.items(), 1):
            sub = df_runs[df_runs['controller'] == ctrl]
            f.write(f"| **{rank_idx}** | **{ctrl}** | {score:.2f} | "
                    f"{sub['avg_waiting_time_s'].mean():.2f} ± {compute_ci95(sub['avg_waiting_time_s']):.2f} | "
                    f"{sub['avg_travel_time_s'].mean():.2f} ± {compute_ci95(sub['avg_travel_time_s']):.2f} | "
                    f"{sub['throughput_vph'].mean():.1f} ± {compute_ci95(sub['throughput_vph']):.1f} | "
                    f"{(sub['congestion_ratio']*100).mean():.1f}% |\n")
        f.write("\n---\n\n")

        # Section 2: Detailed Performance Matrix by Scenario
        f.write("## 2. Scenario-by-Scenario Performance Matrix (95% Confidence Intervals)\n\n")

        scenarios = df_runs['scenario'].unique().tolist()
        controllers = df_runs['controller'].unique().tolist()

        for sc in scenarios:
            f.write(f"### Scenario: {sc}\n\n")
            f.write("| Controller | Routing | Waiting Time (s) [95% CI] | Travel Time (s) [95% CI] | Throughput (vph) [95% CI] | Peak Queue | Congestion % |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")

            for ctrl in controllers:
                for rm in df_runs['routing_mode'].unique():
                    sub = df_runs[(df_runs['scenario'] == sc) & (df_runs['controller'] == ctrl) & (df_runs['routing_mode'] == rm)]
                    if sub.empty:
                        continue

                    w_m = sub['avg_waiting_time_s'].mean()
                    w_ci = compute_ci95(sub['avg_waiting_time_s'])

                    t_m = sub['avg_travel_time_s'].mean()
                    t_ci = compute_ci95(sub['avg_travel_time_s'])

                    th_m = sub['throughput_vph'].mean()
                    th_ci = compute_ci95(sub['throughput_vph'])

                    q_max = sub['max_queue_length'].max()
                    c_pct = (sub['congestion_ratio'] * 100).mean()

                    f.write(f"| **{ctrl}** | {rm} | {w_m:.2f} ± {w_ci:.2f} | {t_m:.2f} ± {t_ci:.2f} | {th_m:.1f} ± {th_ci:.1f} | {q_max} | {c_pct:.1f}% |\n")
            f.write("\n")

        # Section 3: Visual Plots Reference
        f.write("## 3. Visualizations\n\n")
        f.write("Comparative plots generated by this analysis pipeline:\n")
        f.write("- **Multi-Controller Dashboard**: `charts/controller_performance_dashboard.png`\n")
        f.write("- **Routing Mode Interaction**: `charts/routing_mode_comparison.png`\n\n")

    print(f"[Analysis] Generated comprehensive report: {output_path}")

def main():
    args = parse_args()
    setup_plot_style()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    df_runs = pd.read_csv(args.input)
    print(f"[Analysis] Loaded {len(df_runs)} runs from {args.input}")

    os.makedirs(args.charts_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Compute Controller Rankings
    df_ranked, overall_ranking = calculate_controller_rankings(df_runs)

    # 2. Generate Plots
    plot_comprehensive_dashboard(df_runs, args.charts_dir)
    plot_routing_interaction(df_runs, args.charts_dir)

    # 3. Generate Markdown Report
    report_path = os.path.join(args.output_dir, "experiment_analysis_report.md")
    generate_markdown_report(df_runs, df_ranked, overall_ranking, report_path)

    # Print quick CLI summary
    print("\n=======================================================")
    print(" SyntraQ Automated Experiment Analysis Summary")
    print("=======================================================")
    print("Overall Controller Rankings (Composite Rank 1=Best):")
    for rank, (ctrl, score) in enumerate(overall_ranking.items(), 1):
        print(f"  #{rank}: {ctrl:<24} (Score: {score:.2f})")
    print("=======================================================\n")

if __name__ == "__main__":
    main()

