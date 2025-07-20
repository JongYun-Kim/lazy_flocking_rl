import os
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
# Envs and models
from envs_mj import LazyAgentsCentralized
from models.lazy_allocator import MyRLlibTorchWrapper
# RLlib from Ray
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import ray

from tqdm import tqdm

import time

import copy

import pickle


env_config = {
    "num_agents_max": None,  # Maximum number of agents
    "num_agents_min": None,  # Minimum number of agents
    # Optional parameters
    "speed": 15,  # Speed in m/s. Default is 15
    "predefined_distance": 60,  # Predefined distance in meters. Default is 60
    # Tune the following parameters for your environment
    "std_pos_converged": 45,  # Standard position when converged. Default is 0.7*R
    "std_vel_converged": 0.1,  # Standard velocity when converged. Default is 0.1
    "std_pos_rate_converged": 0.1,  # Standard position rate when converged. Default is 0.1
    "std_vel_rate_converged": 0.2,  # Standard velocity rate when converged. Default is 0.2
    "max_time_step": 1000,  # Maximum time steps. Default is 2000,
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


model_name = "custom_model"
ModelCatalog.register_custom_model(model_name, MyRLlibTorchWrapper)

base_path = "/server/Downloads/lazy_flocking_rl/bk/bk_082623"
trial_path = base_path + "/PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47"  # the first emergent behavior (chkp74)
checkpoint_path = trial_path + "/checkpoint_000074/policies/default_policy"

policy = Policy.from_checkpoint(checkpoint_path)
policy.model.eval()

with open("paper_data_3.pkl", "rb") as f:
    data = pickle.load(f)
    acs_results = data["acs"]  # {agent_num: n_exp x 2[reward, time]}
    heuristic_results = data["heuristic"]
    rl_results = data["rl"]

@ray.remote(num_cpus=1)
def test(seed, envs, n_agent):

    env_acs, env_heurstic, env_rl = envs
    env_rl.reset()

    # copy the state
    env_acs.agent_pos = copy.deepcopy(env_rl.agent_pos)
    env_acs.agent_vel = copy.deepcopy(env_rl.agent_vel)
    env_acs.agent_ang = copy.deepcopy(env_rl.agent_ang)
    env_acs.agent_omg = copy.deepcopy(env_rl.agent_omg)

    env_heurstic.agent_pos = copy.deepcopy(env_rl.agent_pos)
    env_heurstic.agent_vel = copy.deepcopy(env_rl.agent_vel)
    env_heurstic.agent_ang = copy.deepcopy(env_rl.agent_ang)
    env_heurstic.agent_omg = copy.deepcopy(env_rl.agent_omg)

    del env_rl

    ### ACS ###
    done = False
    reward_acs = 0
    t_acs = 0
    action = np.ones(n_agent, dtype=np.float32)
    env_acs.reset_MJ()
    while not done:
        _, reward, done, _ = env_acs.step(action)
        reward_acs += reward
        t_acs += 1
    acs_result = [reward_acs,t_acs]

    del env_acs
    
    ### Heuristic ###
    done = False
    reward_heuristic = 0
    t_heuristic = 0
    env_heurstic.reset_MJ()
    while not done:
        action = env_heurstic.compute_heuristic_action()
        _, reward, done, _ = env_heurstic.step(action)
        reward_heuristic += reward
        t_heuristic += 1
    heuristic_result = [reward_heuristic,t_heuristic]

    del env_heurstic

    data = {"acs": acs_result, "heuristic": heuristic_result}

    return data

ray.init(num_cpus=10,num_gpus=1)

################## Experiment Code ##################
num_agent_list = [1024]
num_exp = 100
seed = 0

for n_agent in num_agent_list:
    print(f"Number of agents: {n_agent}")

    if n_agent in acs_results:
        print("already done")
        print(f"Skipping {n_agent}...")
        continue

    acs_results[n_agent] = np.zeros((num_exp, 2)) # [reward, time]
    heuristic_results[n_agent] = np.zeros((num_exp, 2)) # [reward, time]
    rl_results[n_agent] = np.zeros((num_exp, 2)) # [reward, time]

    env_config["num_agents_max"] = n_agent
    env_config["num_agents_min"] = n_agent

    env_name = "lazy_env"
    env_class = LazyAgentsCentralized
    register_env(env_name, lambda cfg: env_class(cfg))

    env_acs = env_class(env_config)
    env_heurstic = env_class(env_config)
    env_rl = env_class(env_config)

    seed = range(num_exp)

    print("ACS and Heuristic...")

    works = [test.remote(s, [env_acs, env_heurstic, env_rl], n_agent) for s in seed]
    results = ray.get(works)

    for idx, result in enumerate(results):
        acs_results[n_agent][idx] = result["acs"]
        heuristic_results[n_agent][idx] = result["heuristic"]

    print("RL...")
    for exp in tqdm(seed):
        env_rl.seed(exp)
        obs = env_rl.reset()
        
        done = False
        reward_rl = 0
        t_rl = 0

        while not done:
            action = policy.compute_single_action(obs, explore=False)
            obs, reward, done, _ = env_rl.step(action[0])

            reward_rl += reward
            t_rl += 1      
        rl_results[n_agent][exp] = [reward_rl, t_rl]

    print(f"ACS: mean reward: {acs_results[n_agent][:, 0].mean()}, mean time: {acs_results[n_agent][:, 1].mean()}")
    print(f"Heuristic: mean reward: {heuristic_results[n_agent][:, 0].mean()}, mean time: {heuristic_results[n_agent][:, 1].mean()}")
    print(f"RL: mean reward: {rl_results[n_agent][:, 0].mean()}, mean time: {rl_results[n_agent][:, 1].mean()}")

    # print("Saving the results...")
    # # save the results in one file with pickle
    # results = {"acs": acs_results, "heuristic": heuristic_results, "rl": rl_results}
    # with open("paper_data_3.pkl", "wb") as f:
    #     pickle.dump(results, f)
