#!/usr/bin/env python3
"""
Plot spatial entropy and velocity entropy for a single illustrative episode
(seed 868, full-horizon, n=20). Matches the style of the existing paper figures.

Usage (from repository root):
    docker run --rm \
      -v "$(pwd)/experiments":/workspace \
      py313 python /workspace/analysis/plot_single_episode_entropy.py

Output:
    experiments/analysis/figures/single_episode_entropy_seed868.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
TRAJ_DIR = SCRIPT_DIR.parent / "performance_benchmark" / "results" / "trajectories"
FIG_DIR = SCRIPT_DIR / "figures"

COLORS = {
    "acs": "#1f77b4",
    "rl":  "#2ca02c",
}
DT = 0.1  # seconds per step
V_CRUISE = 15.0


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    acs = load_json(TRAJ_DIR / "acs_full_seed868.json")["results"][0]
    rl  = load_json(TRAJ_DIR / "rl_full_seed868.json")["results"][0]

    n_steps = len(acs["spatial_entropy"])
    time = [i * DT for i in range(n_steps)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Panel (a): Spatial Entropy ---
    ax = axes[0]
    ax.plot(time, acs["spatial_entropy"], color=COLORS["acs"],
            linewidth=1.5, label="ACS")
    ax.plot(time, rl["spatial_entropy"], color=COLORS["rl"],
            linewidth=1.5, linestyle="--", label="RL")
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel(r"$\sigma_{\mathbf{p}}$ (m)", fontweight="bold")
    ax.set_title("(a) Spatial Entropy", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(prop={"weight": "bold"})
    ax.grid(True, alpha=0.5)

    # --- Panel (b): Velocity Entropy ---
    ax = axes[1]
    ax.plot(time, acs["velocity_entropy"], color=COLORS["acs"],
            linewidth=1.5, label="ACS")
    ax.plot(time, rl["velocity_entropy"], color=COLORS["rl"],
            linewidth=1.5, linestyle="--", label="RL")
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel(r"$\sigma_{\mathbf{v}}$ (m/s)", fontweight="bold")
    ax.set_title("(b) Velocity Entropy", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(prop={"weight": "bold"})
    ax.grid(True, alpha=0.5)

    fig.tight_layout()

    out = FIG_DIR / "single_episode_entropy_seed868.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
