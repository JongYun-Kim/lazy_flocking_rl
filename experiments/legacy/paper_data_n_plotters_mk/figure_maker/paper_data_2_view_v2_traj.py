import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
# Plots
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
# Save files and load files
import pickle

seed = 868

##############################################
figwidth=6
figheight=6
figsize= (figwidth, figheight)

xlabel_fontsize=12
ylabel_fontsize=12

xtick_fontsize=10
ytick_fontsize=10

##############################################

num_agents = 20

# Load the data
with open("rl_vs_acs_seed_{}.pkl".format(seed), "rb") as f:
    data = pickle.load(f)

agent = data["agent"]
acs_agent = data["acs_agent"]
all_converged_time = max(agent["converged_time"], acs_agent["converged_time"])

cut_time = int((agent["converged_time"] + acs_agent["converged_time"]) / 2)+50 # < maxtime: 1000

print(agent["converged_time"])
print(acs_agent["converged_time"])
print(all_converged_time)
print(cut_time)


# Determine the limits for x and y axes to ensure all plots share the same range
x_min = min(agent["pos"][:cut_time, :, 0].min(), acs_agent["pos"][:cut_time, :, 0].min())
x_max = max(agent["pos"][:cut_time, :, 0].max(), acs_agent["pos"][:cut_time, :, 0].max())
y_min = min(agent["pos"][:cut_time, :, 1].min(), acs_agent["pos"][:cut_time, :, 1].min())
y_max = max(agent["pos"][:cut_time, :, 1].max(), acs_agent["pos"][:cut_time, :, 1].max())

# Calculate the maximum range to use for both x and y limits
max_range = max(x_max - x_min, y_max - y_min)

# Center the limits around the mean of the min and max
x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2

# Set the same range for both x and y
x_lim = (x_center - max_range / 2, x_center + max_range / 2)
y_lim = (y_center - max_range / 2, y_center + max_range / 2)


def colorline(ax, x, y, cmap, norm, linewidth=1, alpha=1.0):
    """
    Plot a colored line with coordinates x and y using a colormap.
    """
    # Create a continuous norm to map from data points to colors
    z = np.linspace(0.0, 1.0, len(x))

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax.add_collection(lc)

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format for LineCollection.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def plot_agents(ax, data, title, cut_time, num_agents, x_lim, y_lim):
    # Define a set of colors to cycle through, one for each agent
    color_cycle = plt.cm.viridis(np.linspace(0, 1, num_agents))
    
    for i in range(num_agents):
        x_traj = data["pos"][:cut_time, i, 0]
        y_traj = data["pos"][:cut_time, i, 1]
        cmap = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
        cmap.set_array([])
        colorline(ax, x_traj, y_traj, cmap=cmap.cmap, norm=cmap.norm, linewidth=2, alpha=0.8)

    # ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X(m)", fontsize=xlabel_fontsize, fontweight='bold')
    ax.set_ylabel("Y(m)", fontsize=ylabel_fontsize, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    # Manually set tick labels to be bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # set tick fontsize
    ax.tick_params(axis='x', labelsize=xtick_fontsize)
    ax.tick_params(axis='y', labelsize=ytick_fontsize)


# Use the default Matplotlib color cycle for algorithm colors
alg_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]

fig_rl, ax_rl = plt.subplots(figsize=figsize)
plot_agents(ax_rl, agent, "", cut_time, num_agents, x_lim, y_lim)

fig_acs, ax_acs = plt.subplots(figsize=figsize)
plot_agents(ax_acs, acs_agent, "", cut_time, num_agents, x_lim, y_lim)


# Saving figures
fig_rl.savefig("paper_rl_traj_seed_{}_v2".format(seed), dpi=300)
fig_acs.savefig("paper_acs_traj_seed_{}_v2".format(seed), dpi=300)

plt.show()  # Only needed if you want to display the plots in an interactive environment
