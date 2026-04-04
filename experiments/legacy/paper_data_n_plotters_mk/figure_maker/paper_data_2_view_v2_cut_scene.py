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

###############################

fig_width = 15
fig_height = 6

cut_time = 200 # < maxtime: 1000

scatter_size = 50
quiver_scale = 20

xtick_fontsize = 12
ytick_fontsize = 12
tick_to_bold = True

x_label_fontsize = 15
y_lael_fontsize = 15
label_to_bold = True

legend_fontsize = 15
legend_bold = True

###############################

seed = 868

num_agents = 20

# Load the data
with open("rl_vs_acs_seed_{}.pkl".format(seed), "rb") as f:
    data = pickle.load(f)

agent = data["agent"]
acs_agent = data["acs_agent"]
all_converged_time = max(agent["converged_time"], acs_agent["converged_time"])

print(agent["converged_time"])
print(acs_agent["converged_time"])
print(all_converged_time)
print(cut_time)

# Determine the limits for x and y axes to ensure all plots share the same range
x_min = agent["pos"][cut_time, :, 0].min()
x_max = agent["pos"][cut_time, :, 0].max()
y_min = agent["pos"][cut_time, :, 1].min()
y_max = agent["pos"][cut_time, :, 1].max()

acs_x_min = acs_agent["pos"][cut_time, :, 0].min()
acs_x_max = acs_agent["pos"][cut_time, :, 0].max()
acs_y_min = acs_agent["pos"][cut_time, :, 1].min()
acs_y_max = acs_agent["pos"][cut_time, :, 1].max()

# Calculate the maximum range to use for both x and y limits
max_range = max(x_max - x_min, y_max - y_min)
acs_max_range = max(acs_x_max - acs_x_min, acs_y_max - acs_y_min)

max_range = max(max_range, acs_max_range)

# Center the limits around the mean of the min and max
x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2

acs_x_center = (acs_x_max + acs_x_min) / 2
acs_y_center = (acs_y_max + acs_y_min) / 2

# Set the same range for both x and y
x_lim = (x_center - max_range / 2 - 20, x_center + max_range / 2 + 20)
y_lim = (y_center - max_range / 2 - 20, y_center + max_range / 2 + 20)

acs_x_lim = (acs_x_center - max_range / 2 - 20, acs_x_center + max_range / 2 + 20)
acs_y_lim = (acs_y_center - max_range / 2 - 20, acs_y_center + max_range / 2 + 20)

pos = agent["pos"]
acs_pos = acs_agent["pos"]

heading = agent["heading"]
acs_heading = acs_agent["heading"]

fig, axs = plt.subplots(1, 2,figsize=(fig_width, fig_height))  # Creates a figure and two subplots

# Use the default Matplotlib color cycle for algorithm colors
alg_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
# Plot for the second method (acs_agent)
acs_u, acs_v = np.cos(acs_heading[cut_time, :]), np.sin(acs_heading[cut_time, :])  # Convert headings to direction vectors
axs[0].quiver(acs_pos[cut_time, :, 0], acs_pos[cut_time, :, 1], acs_u, acs_v, color='k', scale=quiver_scale)  # Quiver plot for headings
axs[0].scatter(acs_pos[cut_time, :, 0], acs_pos[cut_time, :, 1], color=alg_colors[0], s=scatter_size, label="ACS")  # Scatter plot for positions
axs[0].set_xlim(acs_x_lim)
axs[0].set_ylim(acs_y_lim)
# axs[0].set_title("ACS")
axs[0].set_aspect('equal', 'box')
axs[0].tick_params(axis='x',  labelsize=xtick_fontsize)
axs[0].tick_params(axis='y',  labelsize=ytick_fontsize)
axs[0].legend(loc='best', fontsize=legend_fontsize, frameon=True)

if tick_to_bold:
    for xtics in axs[0].get_xticklabels():
        xtics.set_fontweight('bold')
    for ytics in axs[0].get_yticklabels():
        ytics.set_fontweight('bold')

if label_to_bold:
    axs[0].set_xlabel("X(m)", fontsize=x_label_fontsize, fontweight='bold')
    axs[0].set_ylabel("Y(m)", fontsize=y_lael_fontsize, fontweight='bold')
else:
    axs[0].set_xlabel("X(m)", fontsize=x_label_fontsize)
    axs[0].set_ylabel("Y(m)", fontsize=y_lael_fontsize)

if legend_bold:
    for label in axs[0].get_legend().get_texts():
        label.set_fontweight('bold')

# Plot for the first method (agent)
u, v = np.cos(heading[cut_time, :]), np.sin(heading[cut_time, :])  # Convert headings to direction vectors
axs[1].quiver(pos[cut_time, :, 0], pos[cut_time, :, 1], u, v, color='k', scale=quiver_scale)  # Quiver plot for headings
axs[1].scatter(pos[cut_time, :, 0], pos[cut_time, :, 1], color=alg_colors[2], s=scatter_size, label="RL")  # Scatter plot for positions
axs[1].set_xlim(x_lim)
axs[1].set_ylim(y_lim)
# axs[1].set_title("RL")
axs[1].set_aspect('equal', 'box')
axs[1].tick_params(axis='x',  labelsize=xtick_fontsize)
axs[1].tick_params(axis='y',  labelsize=ytick_fontsize)
axs[1].legend(loc='best', fontsize=legend_fontsize, frameon=True)

if tick_to_bold:
    for xtics in axs[1].get_xticklabels():
        xtics.set_fontweight('bold')
    for ytics in axs[1].get_yticklabels():
        ytics.set_fontweight('bold')

if label_to_bold:
    axs[1].set_xlabel("X(m)", fontsize=x_label_fontsize, fontweight='bold')
    axs[1].set_ylabel("Y(m)", fontsize=y_lael_fontsize, fontweight='bold')
else:
    axs[1].set_xlabel("X(m)", fontsize=x_label_fontsize)
    axs[1].set_ylabel("Y(m)", fontsize=y_lael_fontsize)

if legend_bold:
    for label in axs[1].get_legend().get_texts():
        label.set_fontweight('bold')

# save the figure
plt.savefig("rl_vs_acs_seed_{}_{}step.png".format(seed, cut_time), bbox_inches='tight')

plt.tight_layout()
plt.show()
