import pickle
import numpy as np
from matplotlib import pyplot as plt

fig_width = 10
fig_height = 6

xtick_fontsize = 15
ylabel_fontsize = 12
legend_fontsize = 14  # Added parameter for legend font size
ytick_fontsize = 12  # Added parameter for y-axis tick label font size

n_agent = 20

with open("paper_data_1.pkl", "rb") as data:
    data = pickle.load(data)
    acs_results = data["acs"][n_agent]  # n_exp x 2
    heuristic_results = data["heuristic"][n_agent]  # n_exp x 2
    rl_results = data["rl"][n_agent]  # n_exp x 2

acs_total_cost = -acs_results[:, 0]
acs_time = acs_results[:, 1] * 0.1
acs_control_cost = acs_total_cost - acs_time

heuristic_total_cost = -heuristic_results[:, 0]
heuristic_time = heuristic_results[:, 1] * 0.1
heuristic_control_cost = heuristic_total_cost - heuristic_time

rl_total_cost = -rl_results[:, 0]
rl_time = rl_results[:, 1] * 0.1
rl_control_cost = rl_total_cost - rl_time

# Categories
categories = ['Total Cost', 'Control Cost', 'Convergence Time']

# Data for each method
acs_data = [acs_total_cost.mean(), acs_control_cost.mean(), acs_time.mean()]
heuristic_data = [heuristic_total_cost.mean(), heuristic_control_cost.mean(), heuristic_time.mean()]
rl_data = [rl_total_cost.mean(), rl_control_cost.mean(), rl_time.mean()]

# Set up the bar width and positions
barWidth = 0.2
r1 = np.arange(len(categories))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create the figure and a single set of axes
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Plot the data
ax.bar(r1, acs_data, width=barWidth, label='ACS')
ax.bar(r2, heuristic_data, width=barWidth, label='Heuristic', hatch='//')
ax.bar(r3, rl_data, width=barWidth, label='RL', hatch='\\')

# Add labels, legend, and title
ax.set_xticks([r + barWidth for r in range(len(categories))])
ax.set_xticklabels(categories, fontsize=xtick_fontsize, fontweight='bold')
ax.set_ylabel('Values', fontsize=ylabel_fontsize, fontweight='bold')
# Adjusting y-axis tick label font size
ax.tick_params(axis='y', labelsize=ytick_fontsize)

# Adjusting legend font size
ax.legend(fontsize=legend_fontsize)
ax.grid(alpha=0.4)

plt.savefig('paper_cost_comparison_240226.png', format='png', dpi=300)

# Print mean and std of each method (Consider adjusting font size if displaying on figure)
print(f"ACS: mean: {acs_total_cost.mean()}, std: {acs_total_cost.std()}")
print(f"Heuristic: mean: {heuristic_total_cost.mean()}, std: {heuristic_total_cost.std()}")
print(f"RL: mean: {rl_total_cost.mean()}, std: {rl_total_cost.std()}")

plt.show()