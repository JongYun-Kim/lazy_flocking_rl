import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable


class CustomNormalize(Normalize):
    """
    Custom normalization class to map data values to a specific color range.
    """
    def __init__(self, data_min=0, data_max=1, color_min=0.025, color_max=0.8, clip=True):
        super().__init__(vmin=data_min, vmax=data_max, clip=clip)
        self.color_min = color_min
        self.color_max = color_max

    def __call__(self, value, clip=None):
        # Get the normalized value using the Normalize class (parent class)
        normalized = super().__call__(value, clip=clip)
        # Map the normalized value to the specified color range you want
        return self.color_min + (self.color_max - self.color_min) * normalized


# --- Configuration ---
# Specify the path to the pickle file you want to use.
file_path = "./data/cl_n20_250703_190200.pkl"
# Specific time steps for snapshots.
snapshot_times = [1, 70, 200, 398]
# Algorithms to process.
algos = ["ACS", "RL"]


def plot_snapshot(ax, positions, headings, laziness_values, cmap, norm, arrow_len=12, edge_color='blue', arrow_color='red'):
    # Plot agent positions, colored by laziness
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=laziness_values,
        cmap=cmap,
        norm=norm,
        s=50,
        edgecolors=edge_color,
        linewidths=1,
        zorder=3
    )

    # Plot heading vectors
    us = np.cos(headings) * arrow_len
    vs = np.sin(headings) * arrow_len
    ax.quiver(positions[:, 0], positions[:, 1], us, vs,
              scale=0.5, scale_units='xy', width=0.01, color=arrow_color, zorder=4)
    return scatter


# --- Data Loading ---
try:
    with open(file_path, "rb") as f:
        results = pickle.load(f)
        seeds = pickle.load(f)
        env_config = pickle.load(f)
        time_taken = pickle.load(f)
    print(f"Data loaded successfully from: {file_path}")
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- Directory Setup ---
base_plot_dir = "plots"
data_file_name = os.path.splitext(os.path.basename(file_path))[0]
output_dir = os.path.join(base_plot_dir, data_file_name, "snapshots_all_v2")
os.makedirs(output_dir, exist_ok=True)

print(f"Plots will be saved in: {output_dir}")

# --- Plotting Snapshots ---
num_episodes = len(results[algos[0]]["state_hists"])
num_agents_max = results[algos[0]]["state_hists"][0].shape[1]
num_snapshots = len(snapshot_times)

# Setup colormap and normalization for laziness
cmap = plt.cm.hot_r
norm = CustomNormalize(data_min=0, data_max=1, color_min=0.02, color_max=0.72)

# Set y-axis limits for each algorithm
y_limits = {
    "ACS": (-125, 160),
    "RL": (-125, 520),
}
# Fixed x-axis limits and padding
x_min_fixed, x_max_fixed = -135, 155
padding = 8

for ep in range(num_episodes):
    fig, axs = plt.subplots(len(algos), num_snapshots, figsize=(16, 15))
    fig.patch.set_facecolor('white')

    for row, algo in enumerate(algos):
        state_hist = results[algo]["state_hists"][ep]
        laziness_hist = results[algo]["laziness_hists"][ep]
        algo_ymin, algo_ymax = y_limits[algo]

        for col, t in enumerate(snapshot_times):
            ax = axs[row, col]
            ax.set_facecolor('darkgray')

            # Plot trajectories up to the current time step, colored by laziness
            for agent_idx in range(num_agents_max):
                points = state_hist[:t + 1, agent_idx, :2].reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                laziness_segments = laziness_hist[:t, agent_idx]

                lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.7)
                lc.set_array(laziness_segments)
                lc.set_linewidth(1)
                ax.add_collection(lc)

            # Get data for the current time step
            positions = state_hist[t, :, :2]
            headings = state_hist[t, :, 4]
            laziness_at_t = laziness_hist[t, :]

            # Define edge colors for each algorithm
            algo_edge_colors = {
                "ACS": 'C0',  # matplotlib default blue
                "RL": 'C2',  # matplotlib default green
            }
            # Plot the snapshot
            scatter = plot_snapshot(
                ax,
                positions,
                headings,
                laziness_at_t,
                cmap,
                norm,
                edge_color=algo_edge_colors.get(algo, 'blue'),  # get edge color for each algorithm
                arrow_color=algo_edge_colors.get(algo, 'red'),
            )

            if row == 0:
                ax.set_title(f"{t / 10:g} s" if col != 0 else "0 s", fontsize=16)
            if col == 0:
                ax.set_ylabel(algo, fontsize=20, fontweight='bold')
                ax.tick_params(axis='y', labelsize=16)
            else:
                # Hide y-axis tick labels for inner columns
                ax.tick_params(labelleft=False)

            # Hide x-axis tick labels for all but the bottom row
            if row < len(algos) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(axis='x', labelsize=16)

            ax.grid(True)
            ax.set_xlim(x_min_fixed - padding, x_max_fixed + padding)
            ax.set_ylim(algo_ymin - padding, algo_ymax + padding)
            ax.set_aspect('equal', adjustable='box')

    plt.subplots_adjust(wspace=0.1, hspace=-0.28)

    # Get the position of the axes
    bottom_ax_pos = axs[-1, -1].get_position()
    top_ax_pos = axs[0, -1].get_position()

    cbar_ax = fig.add_axes([0.92, bottom_ax_pos.y0, 0.02, top_ax_pos.y1 - bottom_ax_pos.y0])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Laziness", fontsize=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)

    # Save the figure
    output_file = os.path.join(output_dir, f"ep_{ep + 1}_all_snapshots_v8.png")
    fig.savefig(output_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved combined snapshot for Episode {ep + 1} at {output_file}")

print("\nAll snapshots have been saved successfully.")
