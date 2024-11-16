import numpy as np
import copy
# Envs and models
from env.envs import LazyAgentsCentralized, LazyAgentsCentralizedPendReward
from models.lazy_allocator import MyRLlibTorchWrapper, MyMLPModel
# RLlib from Ray
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
# Save files and load files
import pickle
import os  # creates dirs
from datetime import datetime  # gets current date and time


def get_action_from_algo(algo_, obs_, env_: LazyAgentsCentralized, policy_: Policy = None):
    if algo_ == "ACS":
        # ACS action logic
        action_ = env_.get_fully_active_action()
        return action_
    elif algo_ == "Heuristic":
        # Heuristic action logic
        action_ = env_.compute_heuristic_action()
        return action_
    elif algo_ == "RL":
        if policy_ is None:
            raise ValueError("Policy must be provided for RL algorithm.")
        # RL action logic
        action_ = policy_.compute_single_action(obs_, explore=False)[0]
        # clip the action in [0, 1]
        action_ = np.clip(action_, 0, 1)
        return action_
    else:
        raise ValueError("Invalid algorithm name.")


if __name__ == "__main__":
    # Env Configs
    num_agents_max = 20
    num_agents_min = 20
    max_time_step = 1000
    env_config = {
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
        "std_vel_converged": 0.1,  # 1.5,  # Standard velocity when converged. Default is 0.1
        "std_pos_rate_converged": 0.1, # 0.6,  # Standard position rate when converged. Default is 0.1
        "std_vel_rate_converged": 0.2,  # 0.8,  # Standard velocity rate when converged. Default is 0.2
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
    register_env(env_name, lambda cfg: env_class(cfg))
    env_org = env_class(env_config)  # Get a env instance

    # Model Settings
    model_name = "custom_model"
    ModelCatalog.register_custom_model(model_name, MyRLlibTorchWrapper)

    # Get Policy from the Checkpoint
    base_path = "../bk/bk_082623"
    trial_path = base_path + "/PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47"  # the first emergent behavior (chkp74)
    checkpoint_path = trial_path + "/checkpoint_000074/policies/default_policy"
    policy = Policy.from_checkpoint(checkpoint_path)
    policy.model.eval()

    # Experiment Settings
    num_episodes = 100
    save_files = False
    algos = ["ACS", "Heuristic", "RL"]

    # Seed Settings
    start_seed = 1
    seeds = np.arange(start_seed, start_seed + num_episodes)  # [start, +1, +2, ..., +num_episodes-1]

    # Results dictionary setup
    results = {algo: {"state_hists": [], "laziness_hists": [], "control_hists": [],
                      "episode_rewards": [], "episode_lengths": [],
                      "spatial_entropy_hists": [], "velocity_entropy_hists": []} for algo in algos
               }

    # Iterate over Episodes (Experiments)
    start_time = datetime.now()
    for episode in range(num_episodes):
        # Seed
        seed = int(seeds[episode])

        # Reset the Env
        env_org.seed(seed)
        obs = env_org.reset()
        action_hist = np.zeros((max_time_step, num_agents_max), dtype=np.float32)

        for algo in algos:
            env = copy.deepcopy(env_org)  # uses the same seed for each algo
            # Run the algorithm here (example)
            reward_sum = 0
            episode_length = None # Initialize episode length
            for t in range(max_time_step):
                # Action and state update logic would go here (pseudo-code)
                action = get_action_from_algo(algo, obs, env, policy)
                next_obs, reward, done, info = env.step(action)
                reward_sum += reward
                action_hist[t] = action

                if done and episode_length is None:
                    episode_length = t + 1
            if episode_length is None:
                episode_length = max_time_step

            # Append results for each algorithm and episode
            results[algo]["state_hists"].append(env.state_hist[:,:,0:5])
            results[algo]["laziness_hists"].append(1-action_hist)
            results[algo]["control_hists"].append(env.state_hist[:,:,5])
            results[algo]["episode_rewards"].append(reward_sum)
            results[algo]["episode_lengths"].append(episode_length)
            results[algo]["spatial_entropy_hists"].append(env.std_pos_hist)
            results[algo]["velocity_entropy_hists"].append(env.std_vel_hist)
        print(f"Episode {episode+1}/{num_episodes} done.")

    # Print the time taken
    end_time = datetime.now()
    print("\n    Time taken: ", end_time - start_time)
    # Now `results` contains organized data for each algorithm and episode


    # Save the Data
    # # 저장할 데이터: 알고리즘별(ACS, Heuristic, RL)로 나누어서 같이 저장
    # # # (1) state
    # # # (2) laziness (1-action)
    # # # (3) control inputs
    # # # (4) episode rewards
    # # # (5) episode lengths
    # # # (6) spatial entropy
    # # # (7) velocity entropy
    # # # (8) seeds
    # # # (9) settings:

    # Create a directory for saving the data at ./data dir (if not exists, create one)
    save_dir = "./data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the data (file name: cl_n{num_agents_max}_YYMMDD_HHMMSS.pkl)
    base_file_name = f"cl_n{num_agents_max}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    file_name = base_file_name + ".pkl"
    # Check if the file already exists; if so, change the file name '_1', '_2', ... at the end
    suffix = 1
    while os.path.exists(os.path.join(save_dir, file_name)):
        file_name = base_file_name + f"_{suffix}.pkl"
        suffix += 1
    with open(os.path.join(save_dir, file_name), "wb") as f:
        pickle.dump(results, f)
        pickle.dump(seeds, f)
        pickle.dump(env_config, f)
        print(f"Data saved at {os.path.join(save_dir, file_name)}")

    print("Data saved successfully at ", os.path.join(save_dir, file_name))
    print("Current Time: ", end_time)

    print("Done!")

