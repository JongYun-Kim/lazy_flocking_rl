import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime


# Use a built-in style for a polished look. (Check your plt version)
plt.style.use("seaborn-v0_8-darkgrid")

# -------------------------------
# Data Loading
# -------------------------------

file_path = "./data/cl_n20_241118_162131.pkl"

# Load the saved data.
with open(file_path, "rb") as f:
    results = pickle.load(f)
    seeds = pickle.load(f)
    env_config = pickle.load(f)
    time_taken = pickle.load(f)

print("Data loaded successfully from:", file_path)
print("Simulation time taken:", time_taken)

# -------------------------------
# Trajectory Plotting: Agent Positions
# -------------------------------

algos = ["ACS", "Heuristic", "RL"]

# Note: state_hists is saved as a 3D array: (time_steps, num_agents, state_features)
num_episodes = len(results["ACS"]["state_hists"])
num_agents_max = results["ACS"]["state_hists"][0].shape[1]

# Loop over each episode to create and save a trajectory plot.
for ep in range(num_episodes):
    # Create a figure with one row per algorithm.
    fig, axs = plt.subplots(len(algos), 1, figsize=(8, 4 * len(algos)))
    # If there's only one algorithm, ensure axs is a list.
    if len(algos) == 1:
        axs = [axs]

    # Loop over each algorithm.
    for row, algo in enumerate(algos):
        # Get the state history for the current episode.
        state_hist = results[algo]["state_hists"][ep]  # shape: (time_steps, num_agents, state_features)
        # Create distinct colors for each agent using the viridis colormap.
        agent_colors = [cm.plasma(float(agent) / num_agents_max) for agent in range(num_agents_max)]

        ax = axs[row]
        for agent in range(num_agents_max):
            # The first element is x position and second is y position.
            x = state_hist[:, agent, 0]
            y = state_hist[:, agent, 1]
            ax.plot(x, y, color=agent_colors[agent], alpha=0.8, linewidth=1.5)
        ax.set_title(f"Trajectory - {algo}", fontsize=12)
        ax.set_xlabel("X Position", fontsize=10)
        ax.set_ylabel("Y Position", fontsize=10)
        ax.grid(True)
        # Set equal aspect ratio for better spatial visualization.
        ax.set_aspect("equal", adjustable="datalim")

    plt.suptitle(f"Episode {ep + 1} Agent Trajectories", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure as a separate file.
    today_str = datetime.now().strftime("%y%m%d")
    folder_name = f"{today_str}_A{num_agents_max}"
    output_dir = os.path.join("./plots", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"trj_ep_{ep + 1}.png")
    fig.savefig(output_file)
    plt.close(fig)
    print(f"Saved trajectory plot for Episode {ep + 1} as {output_file}")