import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime
import numpy as np


# Use a built-in matplotlib style for a cleaner look. (Check your plt version)
# plt.style.use("seaborn-v0_8-darkgrid")
plt.style.use("default")

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

# algos = ["ACS", "Heuristic", "RL"]
algos = ["ACS", "RL"]

num_episodes = len(results["ACS"]["control_hists"])
num_agents_max = results["ACS"]["control_hists"][0].shape[1]

# Loop over each episode to create and save a figure.
for ep in range(num_episodes):
    if ep != 868:  # Only plot episode of the magic number 868 (for paper tbh).
        continue
    # Create a figure with one row per algorithm and two columns (control & laziness).
    fig, axs = plt.subplots(len(algos), 2, figsize=(14, 4 * len(algos)))

    # Loop over each algorithm (each row).
    for row, algo in enumerate(algos):
        # Generate distinct colors for each agent using a colormap.
        # agent_colors = [cm.plasma(float(agent) / num_agents_max) for agent in range(num_agents_max)]
        # 파란색(ACS) 또는 초록색(RL) 계열 설정
        # cmap = cm.Blues if algo == "ACS" else cm.Greens
        # agent_colors = [cmap((num_agents_max - agent) / (num_agents_max + 1)) for agent in range(num_agents_max)]
        # 1) matplotlib 기본 색상 사이클에서 파랑(0번)·초록(2번) 꺼내오기
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        base_color = default_colors[0] if algo == "ACS" else default_colors[2]
        # 2) 'base_color'와 흰색을 섞어주는 함수
        def shade_color(color, mix):
            rgb = np.array(mcolors.to_rgb(color))
            white = np.ones(3)
            return tuple(rgb * mix + white * (1 - mix))
        # 3) 에이전트마다 mix 비율을 조금씩 달리해서 리스트 생성
        agent_colors = [shade_color(base_color, 1 - (agent / (num_agents_max - 1)) * 0.7) for agent in range(num_agents_max)]

        # Retrieve histories for the current episode and algorithm.
        control_hist = results[algo]["control_hists"][ep]
        laziness_hist = results[algo]["laziness_hists"][ep]

        # Access the current row's axes.
        ax_control = axs[row, 0] if len(algos) > 1 else axs[0]
        ax_laziness = axs[row, 1] if len(algos) > 1 else axs[1]

        time_axis = np.linspace(0.1, 100.0, num=control_hist.shape[0])

        # --- Plot Control History ---
        # for agent in range(num_agents_max):
        for agent in reversed(range(num_agents_max)):
            ax_control.plot(time_axis, control_hist[:, agent], color=agent_colors[agent], alpha=1.0, linewidth=1.5)
        ax_control.set_title(f"Control Inputs Changes - {algo}", fontsize=14)
        ax_control.set_xlabel("Time (s)", fontsize=12)
        ax_control.set_ylabel("Control Input (rad/s)", fontsize=12)
        ax_control.grid(True)

        # --- Plot Laziness History (1 - Action) ---
        # for agent in range(num_agents_max):
        for agent in reversed(range(num_agents_max)):
            ax_laziness.plot(time_axis, laziness_hist[:, agent], color=agent_colors[agent], alpha=1.0, linewidth=1.5)
        ax_laziness.set_title(f"Laziness Changes - {algo}", fontsize=14)
        ax_laziness.set_xlabel("Time (s)", fontsize=12)
        ax_laziness.set_ylabel("Laziness", fontsize=12)
        ax_laziness.set_ylim(-0.05, 1)
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