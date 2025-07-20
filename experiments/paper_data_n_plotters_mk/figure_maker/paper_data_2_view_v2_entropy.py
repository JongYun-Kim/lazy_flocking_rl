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

#########################################################
fig_width = 15
fig_height = 6

x_tick_fontsize = 15
y_tick_fontsize = 15
tick_to_bold = False

x_label_fontsize = 15
x_label_bold = True

legend_fontsize = 15
legend_bold = False
#########################################################

num_agents = 20

# Load the data
with open("rl_vs_acs_seed_{}.pkl".format(seed), "rb") as f:
    data = pickle.load(f)

agent = data["agent"]
acs_agent = data["acs_agent"]
all_converged_time = max(agent["converged_time"], acs_agent["converged_time"])

# Use the default Matplotlib color cycle for algorithm colors
alg_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]

# Function to improve plot aesthetics
def beautify_plot(ax, xlabel, legend_loc='best'):
    if x_label_bold:
        ax.set_xlabel(xlabel, fontsize=x_label_fontsize, fontweight='bold')
    else:
        ax.set_xlabel(xlabel, fontsize=x_label_fontsize)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=True)
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='x', labelsize=x_tick_fontsize)
    ax.tick_params(axis='y', labelsize=y_tick_fontsize)

    # Set the fontweight of tick labels based on tick_to_bold variable
    tick_fontweight = 'bold' if tick_to_bold else 'normal'
    for xtics in ax.get_xticklabels():
        xtics.set_fontweight(tick_fontweight)
    for ytics in ax.get_yticklabels():
        ytics.set_fontweight(tick_fontweight)

    if legend_bold:
        for label in ax.get_legend().get_texts():
            label.set_fontweight('bold')

    ax.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.7)


# Prepare data
agent, acs_agent = data["agent"], data["acs_agent"]
all_converged_time = max(agent["converged_time"], acs_agent["converged_time"])
alg_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plotting function
def plot_performance(metric, save_name):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    time_range = np.array(range(all_converged_time))
    ax.plot(time_range, acs_agent[metric][:all_converged_time], label="ACS", linewidth=2, color=alg_colors[0])
    ax.plot(time_range, agent[metric][:all_converged_time], label="RL", linestyle="--", linewidth=2, color=alg_colors[2])
    ax.axvline(x=agent["converged_time"], color='red', linestyle=':', label='RL Converged', linewidth=1.8)

    beautify_plot(ax, xlabel="Time Step")
    fig.savefig(f"{save_name}_seed_{seed}.png", dpi=300)

# Generate and save plots
plot_performance("pos_hist", "paper_pos_entropy")
plot_performance("vel_hist", "paper_vel_entropy")

plt.show()