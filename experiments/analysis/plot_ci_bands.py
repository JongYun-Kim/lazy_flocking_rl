#!/usr/bin/env python3
"""
Generate ensemble time-series figure for n=20 (1,000 episodes).

Two panels:
  (a) Spatial Entropy sigma_p(t)
  (b) Order Parameter Phi(t)

Each panel shows ACS / Heuristic / RL with mean line + IQR shading.
Colors: Tableau blue (ACS), orange (Heuristic), green (RL).

Usage (from repository root):
    docker run --rm \
      -v "$(pwd)/experiments":/workspace \
      py313 python /workspace/analysis/plot_ci_bands.py

Output:
    experiments/analysis/figures/ensemble_n20.pdf
    experiments/analysis/figures/ensemble_n20.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data" / "comment29_analysis.json"
FIG_DIR = SCRIPT_DIR / "figures"

COLORS = {
    "acs":       "#1f77b4",  # Tableau blue
    "heuristic": "#ff7f0e",  # Tableau orange
    "rl":        "#2ca02c",  # Tableau green
}
LABELS = {"acs": "ACS", "heuristic": "Heuristic", "rl": "RL"}
METHODS = ["acs", "heuristic", "rl"]
IQR_ALPHA = 0.2


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH) as f:
        data = json.load(f)
    ts = data["n20_timeseries"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Panel (a): Spatial Entropy ---
    ax = axes[0]
    for method in METHODS:
        t = ts[method]["time_s"]
        sp = ts[method]["sigma_p"]
        ax.plot(t, sp["mean"], color=COLORS[method], linewidth=1.5)
        ax.fill_between(t, sp["q25"], sp["q75"],
                        color=COLORS[method], alpha=IQR_ALPHA)
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel(r"$\mathbf{\sigma_p}$ (m)", fontweight="bold")
    ax.set_title("(a) Spatial Entropy", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.5)

    legend_handles = [
        Patch(facecolor=COLORS[m], edgecolor=COLORS[m], alpha=IQR_ALPHA,
              linewidth=1.5, label=LABELS[m])
        for m in METHODS
    ]
    for m, h in zip(METHODS, legend_handles):
        h.set_edgecolor(COLORS[m])
    lines = [plt.Line2D([0], [0], color=COLORS[m], linewidth=1.5) for m in METHODS]
    combined = [
        (line, patch) for line, patch in zip(lines, legend_handles)
    ]
    ax.legend(
        [c for c in combined],
        [LABELS[m] for m in METHODS],
        handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None, pad=0)},
        loc="upper right", framealpha=0.9,
        prop={"weight": "bold"},
    )

    # --- Panel (b): Order Parameter ---
    ax = axes[1]
    for method in METHODS:
        t = ts[method]["time_s"]
        phi = ts[method]["phi"]
        ax.plot(t, phi["mean"], color=COLORS[method], linewidth=1.5)
        ax.fill_between(t, phi["q25"], phi["q75"],
                        color=COLORS[method], alpha=IQR_ALPHA)
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel(r"$\mathbf{\Phi}$", fontweight="bold")
    ax.set_title("(b) Order Parameter", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.5)

    ax.legend(
        [c for c in combined],
        [LABELS[m] for m in METHODS],
        handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None, pad=0)},
        loc="lower right", framealpha=0.9,
        prop={"weight": "bold"},
    )

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"ensemble_n20.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
