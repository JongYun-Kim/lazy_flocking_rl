#
import numpy as np
import copy
# Envs and models
from env.envs import LazyAgentsCentralized, LazyAgentsCentralizedPendReward
from models.lazy_allocator import MyRLlibTorchWrapper, MyMLPModel
# RLlib from Ray
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
# Plots
import matplotlib.pyplot as plt
# Save files and load files
import pickle
import pandas as pd
import os  # creates dirs
from datetime import datetime  # gets current date and time


if __name__ == "__main__":
    num_agents_max = 25
    num_agents_min = 15
    max_time_step = 1000
    env_config_heuristic = {
        "num_agents_max": num_agents_max,  # Maximum number of agents
        "num_agents_min": num_agents_min,  # Minimum number of agents
        # Optional parameters
        "speed": 15,  # Speed in m/s. Default is 15
        "predefined_distance": 60,  # Predefined distance in meters. Default is 60
        # Tune the following parameters for your environment
        "std_pos_converged": 45,  # Standard position when converged. Default is 0.7*R
        "std_vel_converged": 0.1,  # Standard velocity when converged. Default is 0.1
        "std_pos_rate_converged": 0.1,  # Standard position rate when converged. Default is 0.1
        "std_vel_rate_converged": 0.2,  # Standard velocity rate when converged. Default is 0.2
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
    env_class = LazyAgentsCentralized  # want to see the performance in this env for comparison
    # env_class = LazyAgentsCentralizedPendReward
    register_env(env_name, lambda cfg: env_class(cfg))

    # Get original env
    env = env_class(env_config_heuristic)

    # register your custom model
    custom_model_config = {}
    model_name = "custom_model"
    ModelCatalog.register_custom_model(model_name, MyRLlibTorchWrapper)
    # model_name = "custom_model_mlp"
    # env_config["use_mlp_settings"] = True
    # ModelCatalog.register_custom_model(model_name, MyMLPModel)

    # Get nn policy
    # Get path
    base_path = "../../bk/bk_082623"
    trial_path = base_path + "../PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47"  # the first emergent behavior (chkp74)
    checkpoint_path = trial_path + "/checkpoint_000074/policies/default_policy"

    # Get policy from checkpoint
    policy = Policy.from_checkpoint(checkpoint_path)
    policy.model.eval()

    # Set experiment parameters
    num_exp = 2200
    reward_sums = np.zeros((num_exp, 4), dtype=np.float32)
    episode_lengths = np.zeros((num_exp, 4), dtype=np.float32)
    random_seeds = np.random.randint(0, 10000, size=num_exp)
    random_seeds_used = np.zeros((num_exp, 5), dtype=np.int32)
    num_agents_hist = np.zeros(num_exp, dtype=np.float32)

    # Lists to store env instances for each experiment _0912
    envs_org = []
    envs_fully_active = []
    envs_heuristic = []
    envs_nn = []
    envs_heuristic_variant = []

    for exp in range(num_exp):

        # Get env
        env.seed(random_seeds[exp])
        obs = env.reset()
        num_agents = env.num_agents
        num_agents_hist[exp] = num_agents

        # Copy env
        env_fully_active = copy.deepcopy(env)
        env_heuristic = copy.deepcopy(env)
        env_nn = copy.deepcopy(env)
        env_heuristic_variant = copy.deepcopy(env)
        # If you want to differentiate the randomness AFTER the initial conditions
        env_fully_active.seed(np.random.randint(0, 10000))
        env_heuristic.seed(np.random.randint(0, 10000))
        env_nn.seed(np.random.randint(0, 10000))
        env_heuristic_variant.seed(np.random.randint(0, 10000))

        # Get results from fully active env
        done = False
        reward_sum_fully_active = 0
        action = np.ones(num_agents_max, dtype=np.float32)  # actions of the padded agents are ignored in the env
        while not done:
            _, reward, done, _ = env_fully_active.step(action)
            reward_sum_fully_active += reward

        # Get results from heuristic env
        done = False
        reward_sum_heuristic = 0
        while not done:
            action = env_heuristic.compute_heuristic_action()
            _, reward, done, _ = env_heuristic.step(action)
            reward_sum_heuristic += reward

        # Get results from neural network env
        done = False
        reward_sum_nn = 0
        while not done:
            action = policy.compute_single_action(obs, explore=False)  # if stochastic model, test both explore=T/F
            obs, reward, done, _ = env_nn.step(action[0])
            reward_sum_nn += reward

        # Get results from heuristic variant env
        done = False
        reward_sum_heuristic_variant = 0
        while not done:
            action = env_heuristic_variant.compute_heuristic_action(mask=None, use_fixed_lazy=False)
            _, reward, done, _ = env_heuristic_variant.step(action)
            reward_sum_heuristic_variant += reward

        # Print results
        print("----------------------------------------------")
        print(f"Exp: {exp}                        | num_agents: {num_agents}")
        print(f"    ({exp}-1) fully_active:       r={reward_sum_fully_active:.2f}, t={env_fully_active.time_step}")
        print(f"    ({exp}-2) heuristic:          r={reward_sum_heuristic:.2f}, t={env_heuristic.time_step}")
        print(f"    ({exp}-3) nn:                 r={reward_sum_nn:.2f}, t={env_nn.time_step}")
        print(f"    ({exp}-4) heuristic_var:      r={reward_sum_heuristic_variant:.2f}, t={env_heuristic_variant.time_step}")
        print("----------------------------------------------\n")

        # Store results
        reward_sums[exp, 0] = reward_sum_fully_active
        reward_sums[exp, 1] = reward_sum_heuristic
        reward_sums[exp, 2] = reward_sum_nn
        reward_sums[exp, 3] = reward_sum_heuristic_variant
        episode_lengths[exp, 0] = env_fully_active.time_step
        episode_lengths[exp, 1] = env_heuristic.time_step
        episode_lengths[exp, 2] = env_nn.time_step
        episode_lengths[exp, 3] = env_heuristic_variant.time_step

        # Store environment instances _0912
        envs_org.append(copy.deepcopy(env))
        envs_fully_active.append(copy.deepcopy(env_fully_active))
        envs_heuristic.append(copy.deepcopy(env_heuristic))
        envs_nn.append(copy.deepcopy(env_nn))
        envs_heuristic_variant.append(copy.deepcopy(env_heuristic_variant))

        # Store random seeds used _0912
        random_seeds_used[exp, 0] = env.get_seed()
        random_seeds_used[exp, 1] = env_fully_active.get_seed()
        random_seeds_used[exp, 2] = env_heuristic.get_seed()
        random_seeds_used[exp, 3] = env_nn.get_seed()
        random_seeds_used[exp, 4] = env_heuristic_variant.get_seed()

        # print("one exp done")

    # Print average results
    print("==============================================")
    print(f"Average results (mean and std) over {num_exp} experiments:")
    print("--------------------Reward--------------------")
    print(f"    fully_active:          r={np.mean(reward_sums[:, 0]):.2f},   {np.std(reward_sums[:, 0]):.2f}")
    print(f"    heuristic:             r={np.mean(reward_sums[:, 1]):.2f},   {np.std(reward_sums[:, 1]):.2f}")
    print(f"    nn:                    r={np.mean(reward_sums[:, 2]):.2f},   {np.std(reward_sums[:, 2]):.2f}")
    print(f"    heuristic_var:         r={np.mean(reward_sums[:, 3]):.2f},   {np.std(reward_sums[:, 3]):.2f}")
    print("----------------------------------------------")
    print("---------------------Time---------------------")
    print(f"    fully_active:          t={np.mean(episode_lengths[:, 0]):.2f},   {np.std(episode_lengths[:, 0]):.2f}")
    print(f"    heuristic:             t={np.mean(episode_lengths[:, 1]):.2f},   {np.std(episode_lengths[:, 1]):.2f}")
    print(f"    nn:                    t={np.mean(episode_lengths[:, 2]):.2f},   {np.std(episode_lengths[:, 2]):.2f}")
    print(f"    heuristic_var:         t={np.mean(episode_lengths[:, 3]):.2f},   {np.std(episode_lengths[:, 3]):.2f}")
    print("==============================================\n")

    r_full = reward_sums[:, 0]
    r_heu = reward_sums[:, 1]
    r_nn = reward_sums[:, 2]
    r_heu_var = reward_sums[:, 3]
    t_full = episode_lengths[:, 0]
    t_heu = episode_lengths[:, 1]
    t_nn = episode_lengths[:, 2]
    t_heu_var = episode_lengths[:, 3]

    # Get current date and time
    current_directory = os.path.dirname(os.path.abspath(__file__))  # current directory where the script is located
    base_directory = os.path.dirname(current_directory)  # get the parent directory
    data_directory = os.path.join(base_directory, "data")  # data directory is in the base directory
    now = datetime.now()
    date_time = now.strftime("%y_%m_%d_%H%M")
    path_to_create = os.path.join(data_directory, date_time)

    # Create directory with date and time as name
    os.makedirs(path_to_create, exist_ok=True)
    file_path = os.path.join(path_to_create, "results.pkl")

    # # Save the results, seeds and environments _0912
    # np.save("results_reward_sums.npy", reward_sums)
    # np.save("results_episode_lengths.npy", episode_lengths)
    # np.save("num_agents_hist.npy", num_agents_hist)
    # np.save("random_seeds.npy", random_seeds)
    # Saving reward sums, episode lengths, num_agents_hist and random_seeds
    for name, data in zip(["results_reward_sums", "results_episode_lengths", "num_agents_hist", "random_seeds", "random_seeds_used"],
                          [reward_sums, episode_lengths, num_agents_hist, random_seeds, random_seeds_used]):
        np.save(os.path.join(path_to_create, f"{name}.npy"), data)
    # Also save what each column means, for easy viewing
    with open(os.path.join(path_to_create, "column_meanings.txt"), "w") as f:
        f.write("Column meanings:\n")
        f.write("0: reward_fully_active\n")
        f.write("1: reward_heuristic\n")
        f.write("2: reward_nn\n")
        f.write("3: reward_heuristic_variant\n")
        f.write("4: episode_length_fully_active\n")
        f.write("5: episode_length_heuristic\n")
        f.write("6: episode_length_nn\n")
        f.write("7: episode_length_heuristic_variant\n")
        f.write("8: num_agents\n")
        f.write("9: random_seeds\n")
        f.write("10: random_seed_used_org\n")
        f.write("11: random_seed_used_fully_active\n")
        f.write("12: random_seed_used_heuristic\n")
        f.write("13: random_seed_used_nn\n")
        f.write("14: random_seed_used_heuristic_variant\n")

    # Save environments with pickle _0912
    # with open("env_org.pkl", "wb") as f:
    #     pickle.dump(envs_org, f)
    # with open("envs_fully_active.pkl", "wb") as f:
    #     pickle.dump(envs_fully_active, f)
    # with open("envs_heuristic.pkl", "wb") as f:
    #     pickle.dump(envs_heuristic, f)
    # with open("envs_nn.pkl", "wb") as f:
    #     pickle.dump(envs_nn, f)
    # with open("envs_heuristic_variant.pkl", "wb") as f:
    #     pickle.dump(envs_heuristic_variant, f)
    # Save environments with pickle
    envs_names = ["envs_org", "envs_fully_active", "envs_heuristic", "envs_nn", "envs_heuristic_variant"]
    envs_data = [envs_org, envs_fully_active, envs_heuristic, envs_nn, envs_heuristic_variant]
    for name, data in zip(envs_names, envs_data):
        with open(os.path.join(path_to_create, f"{name}.pkl"), "wb") as f:
            pickle.dump(data, f)

    # Save results as CSV for easy viewing _0912
    results_df = pd.DataFrame({
        'reward_fully_active': reward_sums[:, 0],
        'reward_heuristic': reward_sums[:, 1],
        'reward_nn': reward_sums[:, 2],
        'reward_heuristic_variant': reward_sums[:, 3],
        'episode_length_fully_active': episode_lengths[:, 0],
        'episode_length_heuristic': episode_lengths[:, 1],
        'episode_length_nn': episode_lengths[:, 2],
        'episode_length_heuristic_variant': episode_lengths[:, 3],
        'num_agents': num_agents_hist,
        'random_seeds': random_seeds,
        'random_seed_used_org': random_seeds_used[:, 0],
        'random_seed_used_fully_active': random_seeds_used[:, 1],
        'random_seed_used_heuristic': random_seeds_used[:, 2],
        'random_seed_used_nn': random_seeds_used[:, 3],
        'random_seed_used_heuristic_variant': random_seeds_used[:, 4],
    })
    # Save results as CSV for easy viewing
    # results_df.to_csv("results.csv", index=False)  # _0912
    results_df.to_csv(os.path.join(path_to_create, "results.csv"), index=False)

    # Print the path where everything was saved
    print(f"\nData collection completed at", now.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Results are saved in directory: {path_to_create}")

    # Get the indices that would sort r_full
    sorted_indices = np.argsort(r_full)

    # Sort all arrays based on r_full's sorted indices
    r_full_sorted = r_full[sorted_indices]
    r_heu_sorted = r_heu[sorted_indices]
    r_nn_sorted = r_nn[sorted_indices]
    r_heu_var_sorted = r_heu_var[sorted_indices]

    # Generate the x-axis values (environments)
    environments = np.arange(1, num_exp+1)

    # Plot rewards for each model
    plt.figure(figsize=(12, 6))
    plt.plot(environments, r_full_sorted, label='Fully active', color='black')
    plt.plot(environments, r_heu_sorted, label='Heuristic policy', color='red')
    plt.plot(environments, r_nn_sorted, label='Neural network', color='blue')
    plt.plot(environments, r_heu_var_sorted, label='Heuristic policy variant', color='green')

    # Adding title and labels
    plt.title(f'Performance Comparison across {num_exp} Environments', fontsize=20)
    plt.xlabel('Environment Index (sorted by fully active case)', fontsize=17)
    plt.ylabel('Performance (episode reward)', fontsize=17)
    plt.legend(fontsize=15)

    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Done")

    plt.close()


