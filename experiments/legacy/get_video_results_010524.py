#
import numpy as np
import copy
# Envs and models
from env.envs import LazyAgentsCentralized, LazyAgentsCentralizedPendReward, LazyAgentsWithDisturbance
from models.lazy_allocator import MyRLlibTorchWrapper, MyMLPModel
# RLlib from Ray
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
# Plots
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
# from matplotlib.collections import LineCollection
# import itertools
# Save files and load files
# import pickle
# import pandas as pd
# import os  # creates dirs
from datetime import datetime  # gets current date and time


if __name__ == "__main__":
    num_agents_max = 100
    num_agents_min = 100
    max_time_step = 2000
    env_config_heuristic = {
        "num_agents_max": num_agents_max,  # Maximum number of agents
        "num_agents_min": num_agents_min,  # Minimum number of agents
        # Optional parameters
        "speed": 15,  # Speed in m/s. Default is 15
        "max_turn_rate": 8/15,  # u_max: 8/speed
        # Env change test
        "inter_agent_strength": 5,  # CS strength: 5
        "communication_decay_rate": 1/3,  # beta; CS related must be < 1/2: 1/3
        "bonding_strength": 1,  # Cohesion strength: 1
        "initial_position_bound": 250,  # 250
        #
        "predefined_distance": 60,  # Predefined distance in meters. Default is 60
        # Tune the following parameters for your environment
        "std_pos_converged": 45,  # Standard position when converged. Default is 0.7*R
        "std_vel_converged": 0.15,  # 1.5,  # Standard velocity when converged. Default is 0.1
        "std_pos_rate_converged": 0.2, # 0.6,  # Standard position rate when converged. Default is 0.1
        "std_vel_rate_converged": 0.4,  # 0.8,  # Standard velocity rate when converged. Default is 0.2
        # TODO: SEE MAX TIME STEP CHANGED TO 2000 !!!!!!!!!!!!!!!!!!!!!!!
        "max_time_step": max_time_step,  # Maximum time steps. Default is 2000,
        "incomplete_episode_penalty": -0,  # Penalty for incomplete episode. Default is -600
        "normalize_obs": True,  # If True, the env will normalize the obs. Default: False\
        "use_fixed_horizon": False,  # If True, the env will use fixed horizon. Default: False
        "use_L2_norm": False,  # If True, the env will use L2 norm. Default: False
        # Heuristic policy
        "use_heuristics": True,  # If True, the env will use heuristic policy. Default: False
        "_use_fixed_lazy_idx": True,  # If True, the env will use fixed lazy idx. Default: True
        # For RLlib models
        "use_preprocessed_obs": True,  # If True, the env will return preprocessed obs. Default: True
        "use_mlp_settings": False,  # If True, flatten obs used without topology and padding. Default: False
        #                             Note: No padding applied to the MLP settings for now
        # Plot config
        "get_state_hist": True,  # If True, state_hist stored. Use this for plotting. Default: False
        # try to leave it empty in your config unless you explicitly want to plot
        # as it's gonna be False by default, use more memory, and slow down the training/evaluation
    }
    env_name = "lazy_env"
    env_class = LazyAgentsCentralized
    # env_class = LazyAgentsCentralizedPendReward  # should not be tested unless you know what you do with it
    # env_class = LazyAgentsWithDisturbance
    # env_config_heuristic["noise_scale"] = 1.2  # 0.1
    # env_config_heuristic["num_faulty_agents"] = 1

    # Register environment
    register_env(env_name, lambda cfg: env_class(cfg))

    # Get original env
    env = env_class(env_config_heuristic)

    # register your custom model
    # custom_model_config = {}
    model_name = "custom_model"
    ModelCatalog.register_custom_model(model_name, MyRLlibTorchWrapper)
    # model_name = "custom_model_mlp"
    # env_config["use_mlp_settings"] = True
    # ModelCatalog.register_custom_model(model_name, MyMLPModel)

    # Get nn policy
    # Get path
    base_path = "<LOAD_YOUR_PATH>"
    trial_path = base_path + "<LOAD_YOUR_TRIAL_PATH>"
    checkpoint_path = trial_path + "/checkpoint_<YOUR_CHECKPOINT_NUMBER>/policies/<TARGET_POLICY>"
  
    # Get policy from checkpoint
    policy = Policy.from_checkpoint(checkpoint_path)
    policy.model.eval()

    # Configure your experiment
    num_exp = 6

    # Lists to store env instances for each experiment _0912
    envs_org = []
    envs_fully_active = []
    envs_heuristic = []
    envs_nn = []
    #
    num_envs = 4  # (1) fully_active; (2) neural_network; (3) heuristic; (4) heuristic variant (adaptive lazy agent index)

    # Lists to store histories for each experiment
    episodic_reward_sums = np.zeros((num_exp, num_envs), dtype=np.float32)
    accumulated_reward_hists = np.zeros((num_exp, num_envs, max_time_step), dtype=np.float32)
    random_seeds = np.random.randint(0, 10000, size=num_exp)

    # Run experiments
    for exp in range(num_exp):
      
        # Get env
        env.seed(random_seeds[exp])
        obs = env.reset()
        num_agents = env.num_agents

        # Copy env
        env_fully_active = copy.deepcopy(env)
        env_heuristic = copy.deepcopy(env)
        env_nn = copy.deepcopy(env)
        env_heuristic_variant = copy.deepcopy(env)

        # Get results from fully active env
        done = False
        reward_sum_fully_active = 0
        action = np.ones(num_agents_max, dtype=np.float32)  # actions of the padded agents are ignored in the env
        while not done:
            _, reward, done, _ = env_fully_active.step(action)
            reward_sum_fully_active += reward
            accumulated_reward_hists[exp, 0, env_fully_active.time_step - 1] = reward_sum_fully_active

        # Get results from neural network env
        done = False
        reward_sum_nn = 0
        while not done:
            action = policy.compute_single_action(obs, explore=False)  # if stochastic model, test both explore=T/F
            obs, reward, done, _ = env_nn.step(action[0])
            # nn_action_hist[exp, env_nn.time_step - 1, :] = action[0]
            reward_sum_nn += reward
            accumulated_reward_hists[exp, 1, env_nn.time_step - 1] = reward_sum_nn

        # Get results from heuristic env
        done = False
        reward_sum_heuristic = 0
        while not done:
            action = env_heuristic.compute_heuristic_action()
            _, reward, done, _ = env_heuristic.step(action)
            # heu_action_hist[exp, env_heuristic.time_step - 1, :] = action
            reward_sum_heuristic += reward
            accumulated_reward_hists[exp, 2, env_heuristic.time_step - 1] = reward_sum_heuristic

        # Get results from heuristic variant env
        done = False
        reward_sum_heuristic_variant = 0
        while not done:
            action = env_heuristic_variant.compute_heuristic_action(mask=None, use_fixed_lazy=False)
            _, reward, done, _ = env_heuristic_variant.step(action)
            # heu_var_action_hist[exp, env_heuristic_variant.time_step - 1, :] = action
            reward_sum_heuristic_variant += reward
            accumulated_reward_hists[exp, 3, env_heuristic_variant.time_step - 1] = reward_sum_heuristic_variant

        # Store results
        episodic_reward_sums[exp, :] = [reward_sum_fully_active, reward_sum_heuristic, reward_sum_nn, reward_sum_heuristic_variant]

        # Step 1: Setup Figures for Four Subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Adjust figsize as needed
        plots = ['Fully Active', 'Lazy by Neural Network', 'Lazy by Heuristic', 'Lazy by Heuristic Variant']
        envs = [env_fully_active, env_nn, env_heuristic, env_heuristic_variant]

        # Prepare data structures for scatter and quiver plots for each environment
        scatters = []
        quivers = []
        trajectories = [[] for _ in range(4)]  # 4 lists for 4 environments

        # Prepare and initialize each subplot
        reward_texts = []
        for i, ax in enumerate(axs.flat):
            # Initialize a text object for displaying the accumulated reward
            reward_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', color='red')
            reward_texts.append(reward_text)

            # Get agent positions and velocities for each environment
            agent_positions = envs[i].state_hist[:envs[i].time_step, :, 0:2]
            agent_velocities = envs[i].state_hist[:envs[i].time_step, :, 2:4]

            ax.grid(True)
            ax.set_aspect('equal')
            ax.set_xlim(np.min(agent_positions[:, :, 0]), np.max(agent_positions[:, :, 0]))
            ax.set_ylim(np.min(agent_positions[:, :, 1]), np.max(agent_positions[:, :, 1]))
            ax.set_title(plots[i])

            # Initialize scatter and quiver plots for each environment
            scatter = ax.scatter(agent_positions[0, :, 0], agent_positions[0, :, 1])
            quiver = ax.quiver(agent_positions[0, :, 0], agent_positions[0, :, 1],
                               agent_velocities[0, :, 0], agent_velocities[0, :, 1])

            scatters.append(scatter)
            quivers.append(quiver)

            # Initialize trajectories for each agent in each environment
            for _ in range(agent_positions.shape[1]):  # Loop over agents
                trajectories[i].append(ax.plot([], [], color='gray', linewidth=0.5)[0])

        # Step 2: Update Function
        def update(frame):
            components = []
            for i in range(4):  # For each environment
                env = envs[i]
                if frame < env.time_step:  # Check if the environment is still active
                    agent_positions = env.state_hist[frame, :, 0:2]
                    agent_velocities = env.state_hist[frame, :, 2:4]
                else:  # If done, use the last available frame
                    agent_positions = env.state_hist[env.time_step - 1, :, 0:2]
                    agent_velocities = env.state_hist[env.time_step - 1, :, 2:4]

                # Update scatter (positions) with the highest zorder
                scatters[i].set_offsets(agent_positions)
                scatters[i].set_zorder(2)  # allows to be drawn in that order (smaller: bottom/first, larger: top/last)
                components.append(scatters[i])

                # Update velocities with a middle zorder
                quivers[i].set_offsets(agent_positions)
                quivers[i].set_UVC(agent_velocities[:, 0], agent_velocities[:, 1])
                quivers[i].set_zorder(3)
                components.append(quivers[i])

                # Update trajectories with a lower zorder
                for j, line in enumerate(trajectories[i]):
                    if frame < env.time_step:
                        traj_data = env.state_hist[:frame + 1, j, 0:2]
                    else:
                        traj_data = env.state_hist[:env.time_step, j, 0:2]
                    line.set_data(traj_data[:, 0], traj_data[:, 1])
                    line.set_zorder(1)
                    components.append(line)

                # Update the reward text for each subplot with the highest zorder
                if frame < env.time_step:
                    accumulated_reward = - accumulated_reward_hists[exp, i, frame]
                else:
                    accumulated_reward = - accumulated_reward_hists[exp, i, env.time_step - 1]
                reward_texts[i].set_text(f'Energy used: {accumulated_reward:.2f}')
                reward_texts[i].set_zorder(4)
                components.append(reward_texts[i])

            return components

        # Step 3: Create and Save Animation
        # Format the current date and time as YYMMDD_HHmm
        current_time = datetime.now().strftime("%y%m%d_%H%M")
        max_time_step_video = max(env.time_step for env in envs)  # Use the longest time step
        ani = FuncAnimation(fig, update, frames=max_time_step_video, interval=50, blit=True)
        ani.save(f'./videos/{num_agents_max}UAVs_experiment_{exp + 1}_{current_time}.mp4', writer='ffmpeg')

        print(f"Experiment {exp} is done.")

    print("All experiments are done.")
