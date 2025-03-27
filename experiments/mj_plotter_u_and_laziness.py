import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime


# Use a built-in matplotlib style for a cleaner look. (Check your plt version)
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
# Plotting: Control and Laziness Histories
# -------------------------------

algos = ["ACS", "Heuristic", "RL"]
# algos = ["ACS", "RL"]

num_episodes = len(results["ACS"]["control_hists"])
num_agents_max = results["ACS"]["control_hists"][0].shape[1]

# Loop over each episode to create and save a figure.
for ep in range(num_episodes):
    # Create a figure with one row per algorithm and two columns (control & laziness).
    fig, axs = plt.subplots(len(algos), 2, figsize=(14, 4 * len(algos)))

    # Loop over each algorithm (each row).
    for row, algo in enumerate(algos):
        # Generate distinct colors for each agent using a colormap.
        agent_colors = [cm.plasma(float(agent) / num_agents_max) for agent in range(num_agents_max)]

        # Retrieve histories for the current episode and algorithm.
        control_hist = results[algo]["control_hists"][ep]
        laziness_hist = results[algo]["laziness_hists"][ep]

        # Access the current row's axes.
        ax_control = axs[row, 0] if len(algos) > 1 else axs[0]
        ax_laziness = axs[row, 1] if len(algos) > 1 else axs[1]

        # --- Plot Control History ---
        for agent in range(num_agents_max):
            ax_control.plot(control_hist[:, agent],
                            color=agent_colors[agent],
                            alpha=0.8,
                            linewidth=1.5)
        ax_control.set_title(f"Control History - {algo}", fontsize=12)
        ax_control.set_xlabel("Time Step", fontsize=10)
        ax_control.set_ylabel("Control Input", fontsize=10)
        ax_control.grid(True)

        # --- Plot Laziness History (1 - Action) ---
        for agent in range(num_agents_max):
            ax_laziness.plot(laziness_hist[:, agent],
                             color=agent_colors[agent],
                             alpha=0.8,
                             linewidth=1.5)
        ax_laziness.set_title(f"Laziness History - {algo}", fontsize=12)
        ax_laziness.set_xlabel("Time Step", fontsize=10)
        ax_laziness.set_ylabel("Laziness", fontsize=10)
        ax_laziness.grid(True)

    # plt.suptitle(f"Episode {ep+1}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure as a separate file.
    today_str = datetime.now().strftime("%y%m%d")
    folder_name = f"{today_str}_A{num_agents_max}"
    output_dir = os.path.join("./plots", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"episode_{ep + 1}.png")
    fig.savefig(output_file)

    plt.close(fig)
    print(f"Saved figure for Episode {ep + 1} at {output_file}")