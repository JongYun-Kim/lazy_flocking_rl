#!/usr/bin/env python3
"""
Regenerate Figures 3 (bar chart) and 8 (box plot) with L2 metric, 1000 trials.

Usage (from repository root):
    docker run --rm \
      -v "$(pwd)/experiments":/workspace \
      py313 python /workspace/analysis/plot_paper_figures.py

Output:
    experiments/analysis/figures/cost_comparison_n20.png
    experiments/analysis/figures/scalability_boxplot.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent / "performance_benchmark" / "results"
FIG_DIR = SCRIPT_DIR / "figures"

COLORS = {
    "acs":       "#1f77b4",
    "heuristic": "#ff7f0e",
    "rl":        "#2ca02c",
}
LABELS = {"acs": "ACS", "heuristic": "Heuristic", "rl": "RL"}
METHODS = ["acs", "heuristic", "rl"]
HATCHES = {"acs": None, "heuristic": "//", "rl": "\\\\"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Figure 3: Cost comparison bar chart (n=20)
# ============================================================
def plot_cost_comparison():
    data = {}
    for method in METHODS:
        d = load_json(BENCH_DIR / f"{method}.json")
        eps = d["results"]
        data[method] = {
            "J":   [-e["reward_L2"] for e in eps],
            "C":   [e["control_cost_L2"] for e in eps],
            "t_f": [e["convergence_time"] for e in eps],
        }

    metrics = ["J", "C", "t_f"]
    metric_labels = ["Total Cost", "Steering Control Cost", "Convergence Time"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, method in enumerate(METHODS):
        means = [np.mean(data[method][m]) for m in metrics]
        stds  = [np.std(data[method][m], ddof=1) for m in metrics]
        bars = ax.bar(x + i * width, means, width,
                      yerr=stds, capsize=4,
                      color=COLORS[method], hatch=HATCHES[method],
                      edgecolor="white", linewidth=0.5,
                      label=LABELS[method])

    ax.set_ylabel("Values", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.legend(prop={"weight": "bold"})
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("png",):
        out = FIG_DIR / f"cost_comparison_n20.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ============================================================
# Figure 8: Scalability box plot
# ============================================================
def plot_scalability_boxplot():
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    data = {}
    for method in METHODS:
        data[method] = {}
        for n in sizes:
            fpath = BENCH_DIR / "scalability" / f"{method}_n{n}.json"
            d = load_json(fpath)
            data[method][n] = [-e["reward_L2"] for e in d["results"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    positions_per_group = 3
    group_width = 0.6
    box_width = group_width / positions_per_group

    for gi, n in enumerate(sizes):
        for mi, method in enumerate(METHODS):
            pos = gi - group_width/2 + box_width/2 + mi * box_width
            vals = data[method][n]

            bp = ax.boxplot(
                vals, positions=[pos], widths=box_width * 0.8,
                patch_artist=True,
                showfliers=True, flierprops=dict(marker='.', markersize=2, alpha=0.4),
                whis=(5, 95),
                showmeans=True,
                meanprops=dict(marker='', linestyle='--', linewidth=1,
                               color=COLORS[method]),
                meanline=True,
                medianprops=dict(color="black", linewidth=1),
            )
            bp["boxes"][0].set_facecolor(COLORS[method])
            if HATCHES[method]:
                bp["boxes"][0].set_hatch(HATCHES[method])

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Number of Agents", fontweight="bold")
    ax.set_ylabel("Cost", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLORS[m], edgecolor="black",
              hatch=HATCHES[m], label=LABELS[m])
        for m in METHODS
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              prop={"weight": "bold"})

    fig.tight_layout()
    for ext in ("png",):
        out = FIG_DIR / f"scalability_boxplot.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_cost_comparison()
    plot_scalability_boxplot()
