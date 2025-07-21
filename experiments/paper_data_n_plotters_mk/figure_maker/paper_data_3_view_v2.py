import os
import sys
import inspect

import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.patches import Patch  # For custom legend handles

########################
# Figure/style settings
fig_width = 16
fig_height = 9.6

xlabel_fontsize = 20
ylabel_fontsize = 20
label_bold = True

xtick_fontsize = 16
ytick_fontsize = 16
tick_bold = True

legend_fontsize = 20
legend_bold = True
#######################

# Add parent directory to path if needed
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Load data
with open("../paper_data_3_v2_max-time-2k.pkl", "rb") as f:
    data = pickle.load(f)
    acs_results       = data["acs"]        # {agent_num: n_exp x 2 [reward, time]}
    heuristic_results = data["heuristic"]
    rl_results        = data["rl"]

# List of agent counts to plot
num_agent_list = [8, 16, 32, 64, 128, 256, 512, 1024]

# Compute total cost = -reward
acs_total       = {n: -acs_results[n][:, 0]       for n in num_agent_list}
heuristic_total = {n: -heuristic_results[n][:, 0] for n in num_agent_list}
rl_total        = {n: -rl_results[n][:, 0]        for n in num_agent_list}

# Compute convergence time (scaled) and control cost separately
acs_time         = {n: acs_results[n][:, 1] * 0.1      for n in num_agent_list}
heuristic_time   = {n: heuristic_results[n][:, 1] * 0.1 for n in num_agent_list}
rl_time          = {n: rl_results[n][:, 1] * 0.1        for n in num_agent_list}

acs_control      = {n: acs_total[n] - acs_time[n]       for n in num_agent_list}
heuristic_control= {n: heuristic_total[n] - heuristic_time[n] for n in num_agent_list}
rl_control       = {n: rl_total[n] - rl_time[n]         for n in num_agent_list}

# Common boxplot style settings
alg_colors     = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
hatch_patterns = ['', '//', '\\']
line_props     = dict(color="black", linewidth=2.0, linestyle='-')
mean_props     = dict(linestyle=':', linewidth=2.0, color='black')

# Helper to set axis labels and ticks
def style_ax(ax, ylabel):
    # X-ticks
    ax.set_xticks([i*3 + 0.7 for i in range(len(num_agent_list))])
    ax.set_xticklabels(num_agent_list, fontsize=14)

    # Axis labels
    weight = "bold" if label_bold else "normal"
    ax.set_xlabel("Number of Agents", fontsize=xlabel_fontsize, fontweight=weight)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontweight=weight)

    # Tick label fonts
    tw = "bold" if tick_bold else "normal"
    plt.xticks(fontsize=xtick_fontsize, fontweight=tw)
    plt.yticks(fontsize=ytick_fontsize, fontweight=tw)

    # Grid
    ax.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

# 1) Plot Total Cost
fig_total, ax_total = plt.subplots(figsize=(fig_width, fig_height))
for i, n in enumerate(num_agent_list):
    group_data = [acs_total[n], heuristic_total[n], rl_total[n]]
    positions = [i*3, i*3 + 0.6, i*3 + 1.2]
    for j, arr in enumerate(group_data):
        bplot = ax_total.boxplot(
            arr,
            positions=[positions[j]],
            widths=0.5,
            patch_artist=True,
            medianprops=line_props,
            flierprops=dict(marker='o',
                            markerfacecolor=alg_colors[j],
                            markersize=5,
                            linestyle='none',
                            markeredgecolor='none',
                            alpha=0.5),
            meanline=True,
            meanprops=mean_props,
            showmeans=True,
            showcaps=True,
            whis=[5, 95]
        )
        for patch in bplot['boxes']:
            patch.set_facecolor(alg_colors[j])
            patch.set_hatch(hatch_patterns[j])

# Legend
legend_handles = [
    Patch(facecolor=c, hatch=h, label=l)
    for c, h, l in zip(alg_colors, hatch_patterns, ["ACS", "Heuristic", "RL"])
]
ax_total.legend(handles=legend_handles, loc="upper left", fontsize=legend_fontsize)
if legend_bold:
    for text in ax_total.get_legend().get_texts():
        text.set_fontweight('bold')

style_ax(ax_total, ylabel="Cost")
##  REMOVE THIS LINE IF YOU WANT TO SHOW ALL DATA
ax_total.set_ylim(top=900)
##
plt.tight_layout()
plt.show()
fig_total.savefig('paper_data_3_total_cost.png', dpi=300)

# 2) Plot Control Cost
fig_ctrl, ax_ctrl = plt.subplots(figsize=(fig_width, fig_height))
for i, n in enumerate(num_agent_list):
    group_data = [acs_control[n], heuristic_control[n], rl_control[n]]
    positions = [i*3, i*3 + 0.6, i*3 + 1.2]
    for j, arr in enumerate(group_data):
        bplot = ax_ctrl.boxplot(
            arr,
            positions=[positions[j]],
            widths=0.5,
            patch_artist=True,
            medianprops=line_props,
            flierprops=dict(marker='o',
                            markerfacecolor=alg_colors[j],
                            markersize=5,
                            linestyle='none',
                            markeredgecolor='none',
                            alpha=0.5),
            meanline=True,
            meanprops=mean_props,
            showmeans=True,
            showcaps=True,
            whis=[5, 95]
        )
        for patch in bplot['boxes']:
            patch.set_facecolor(alg_colors[j])
            patch.set_hatch(hatch_patterns[j])

ax_ctrl.legend(handles=legend_handles, loc="upper left", fontsize=legend_fontsize)
if legend_bold:
    for text in ax_ctrl.get_legend().get_texts():
        text.set_fontweight('bold')

style_ax(ax_ctrl, ylabel="Control Cost")
## Remove outliers from y-axis
ax_ctrl.set_ylim(top=700)
##
plt.tight_layout()
plt.show()
fig_ctrl.savefig('paper_data_3_control_cost.png', dpi=300)

# 3) Plot Convergence Time
fig_time, ax_time = plt.subplots(figsize=(fig_width, fig_height))
for i, n in enumerate(num_agent_list):
    group_data = [acs_time[n], heuristic_time[n], rl_time[n]]
    positions = [i*3, i*3 + 0.6, i*3 + 1.2]
    for j, arr in enumerate(group_data):
        bplot = ax_time.boxplot(
            arr,
            positions=[positions[j]],
            widths=0.5,
            patch_artist=True,
            medianprops=line_props,
            flierprops=dict(marker='o',
                            markerfacecolor=alg_colors[j],
                            markersize=5,
                            linestyle='none',
                            markeredgecolor='none',
                            alpha=0.5),
            meanline=True,
            meanprops=mean_props,
            showmeans=True,
            showcaps=True,
            whis=[5, 95]
        )
        for patch in bplot['boxes']:
            patch.set_facecolor(alg_colors[j])
            patch.set_hatch(hatch_patterns[j])

# Move legend to lower right
ax_time.legend(handles=legend_handles, loc="upper left", fontsize=legend_fontsize)
if legend_bold:
    for text in ax_time.get_legend().get_texts():
        text.set_fontweight('bold')

style_ax(ax_time, ylabel="Convergence Time (s)")
plt.tight_layout()
plt.show()
fig_time.savefig('paper_data_3_convergence_time.png', dpi=300)
