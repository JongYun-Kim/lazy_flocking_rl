import os
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.patches import Patch  # For custom legend handles

########################
fig_width = 16
fig_height = 8

xlabel_fontsize = 16
ylabel_fontsize = 16
label_bold = True

xtick_fontsize = 12
ytick_fontsize = 12
tick_bold = True

legend_fontsize = 14
legend_bold = True
#######################



# Load data
with open("../paper_data_3.pkl", "rb") as f:
    data = pickle.load(f)
    acs_results = data["acs"]  # {agent_num: n_exp x 2[reward, time]}
    heuristic_results = data["heuristic"]
    rl_results = data["rl"]

num_agent_list = [8,16,32,64,128,256,512,1024]

# Prepare data for plotting
acs_cost = {num_agents: -acs_results[num_agents][:, 0] for num_agents in num_agent_list}
heuristic_cost = {num_agents: -heuristic_results[num_agents][:, 0] for num_agents in num_agent_list}
rl_cost = {num_agents: -rl_results[num_agents][:, 0] for num_agents in num_agent_list}

# Setting up the plot with larger figure size for clarity
fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # Adjust as needed for your content

# Use the default Matplotlib color cycle for algorithm colors
alg_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]

# Horizontal line properties with increased line width
line_props = dict(color="black", linewidth=2.0, linestyle='-')
hatch_patterns = ['', '//', '\\'] 

# Mean line properties
mean_props = dict(linestyle=':', linewidth=2.0, color='black')

# Plotting the data with refined settings
for i, num_agents in enumerate(num_agent_list):
    data_to_plot = [acs_cost[num_agents]], heuristic_cost[num_agents], rl_cost[num_agents]
    # Reduced spacing for closer boxplots
    positions = [i*3, i*3+0.6, i*3+1.2]  # Adjust spacing here
    for j, data in enumerate(data_to_plot):
        # Flier properties with the same color as the box and alpha transparency
        flier_props = dict(marker='o', markerfacecolor=alg_colors[j], markersize=5, linestyle='none', markeredgecolor='none', alpha=0.5)
        # Plotting the boxplots with mean line
        bplot = ax.boxplot(data, positions=[positions[j]], widths=0.5, patch_artist=True, medianprops=line_props, flierprops=flier_props, meanline=True, meanprops=mean_props, showmeans=True, showcaps=True, whis=[5, 95])
        for patch in bplot['boxes']:
            patch.set_facecolor(alg_colors[j])
            patch.set_hatch(hatch_patterns[j])

# Customizing the plot for readability and clarity
ax.set_xticks([i*3+0.7 for i in range(len(num_agent_list))])  # Align x-ticks with the center boxplot of each group
ax.set_xticklabels(num_agent_list, fontsize=14)  # Increase font size

xlabel_fontweight = "bold" if label_bold else "normal"
ylabel_fontweight = "bold" if label_bold else "normal"

ax.set_xlabel("Number of Agents", fontsize=xlabel_fontsize, fontweight=xlabel_fontweight)
ax.set_ylabel("Cost", fontsize=ylabel_fontsize, fontweight=ylabel_fontweight)


xticks_fontweight = "bold" if tick_bold else "normal"
yticks_fontweight = "bold" if tick_bold else "normal"

plt.xticks(fontsize=xtick_fontsize, fontweight=xticks_fontweight)
plt.yticks(fontsize=ytick_fontsize, fontweight=yticks_fontweight)

# Update legend to include hatch patterns
legend_handles = [Patch(facecolor=color, hatch=hatch, label=label) for color, hatch, label in zip(alg_colors, hatch_patterns, ["ACS", "Heuristic", "RL"])]

ax.legend(handles=legend_handles, loc="upper left", fontsize=legend_fontsize)

if legend_bold:
    for label in ax.get_legend().get_texts():
        label.set_fontweight('bold')

# Adding grid lines for better readability
ax.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

# Ensure the plot is not cramped
plt.tight_layout()

# Displaying the plot
plt.show()

# Save the plot as a .png file
fig.savefig('paper_data_3.png', format='png', dpi=300)
