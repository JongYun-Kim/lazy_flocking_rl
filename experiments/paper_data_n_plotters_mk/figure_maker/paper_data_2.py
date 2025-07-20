import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
# Envs and models
from env.envs import LazyAgentsCentralized
from models.lazy_allocator import MyRLlibTorchWrapper
# RLlib from Ray
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
# Plots
import matplotlib.pyplot as plt
# Save files and load files
import pickle
import copy

num_agents_max = 20
num_agents_min = 20

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
env_name = "lazy_env"
env_class = LazyAgentsCentralized  # want to see the performance in this env for comparison
# env_class = LazyAgentsCentralizedPendReward
register_env(env_name, lambda cfg: env_class(cfg))


# register your custom model
custom_model_config = {}
model_name = "custom_model"
ModelCatalog.register_custom_model(model_name, MyRLlibTorchWrapper)

base_path = "/server/Downloads/lazy_flocking_rl/bk/bk_082623"
trial_path = base_path + "/PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47"  # the first emergent behavior (chkp74)
checkpoint_path = trial_path + "/checkpoint_000074/policies/default_policy"

policy = Policy.from_checkpoint(checkpoint_path)
policy.model.eval()

# seed = np.random.randint(0,1000)
seed = 219
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("seed: ", seed)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


# Get original env
env = env_class(env_config_heuristic)
env.seed(seed)
acs_env = env_class(env_config_heuristic)
acs_env.seed(seed)

### RL
obs = env.reset()
acs_env.agent_pos = copy.deepcopy(env.agent_pos)
acs_env.agent_vel = copy.deepcopy(env.agent_vel)
acs_env.agent_ang = copy.deepcopy(env.agent_ang)
acs_env.agent_omg = copy.deepcopy(env.agent_omg)
num_agents = env.num_agents

reward_list = []
cost_list = []
agent = {"pos": [], "vel": [], "heading": [], "heading_rate": [], "acs_u": [], "activness": [], "final_u": []}

reward_list.append(env.initial_rewards)
cost_list.append(env.target_reward)

agent["pos"].append(env.agent_pos.tolist()) # n_agent x 2
agent["vel"].append(env.agent_vel.tolist()) # n_agent x 2
agent["heading"].append(env.agent_ang.tolist()) # n_agent x 1
agent["heading_rate"].append(env.agent_omg.tolist()) # n_agent x 1
agent["acs_u"].append(env.u_fully_active.tolist()) # n_agent
agent["activness"].append(np.ones(num_agents).tolist()) # n_agent
agent["final_u"].append(env.u_lazy.tolist()) # n_agent

done = False
converged = False
max_time = env.max_time_step
for t in range(max_time):
    action = policy.compute_single_action(obs, explore=False)
    obs, reward, done, _ = env.step(action[0])

    # Get the reward
    reward_list.append(reward)
    cost_list.append(env.target_reward)

    agent["pos"].append(env.agent_pos.tolist()) # n_agent x 2
    agent["vel"].append(env.agent_vel.tolist()) # n_agent x 2
    agent["heading"].append(env.agent_ang.tolist()) # n_agent x 1
    agent["heading_rate"].append(env.agent_omg.tolist()) # n_agent x 1
    agent["acs_u"].append(env.u_fully_active.tolist()) # n_agent
    agent["activness"].append(action[0].tolist()) # n_agent
    agent["final_u"].append(env.u_lazy.tolist()) # n_agent
    
    if done and (not converged):
        agent["converged_time"] = t
        converged = True

# convert agent info to numpy
for key in agent.keys():
    agent[key] = np.array(agent[key])
agent["pos_hist"] = env.std_pos_hist
agent["vel_hist"] = env.std_vel_hist

### acs
acs_obs = acs_env.reset_MJ()
num_agents = acs_env.num_agents

acs_reward_list = []
acs_cost_list = []
acs_agent = {"pos": [], "vel": [], "heading": [], "heading_rate": [], "acs_u": [], "activness": [], "final_u": []}

acs_reward_list.append(acs_env.initial_rewards)
acs_cost_list.append(acs_env.target_reward)

acs_agent["pos"].append(acs_env.agent_pos.tolist()) # n_agent x 2
acs_agent["vel"].append(acs_env.agent_vel.tolist()) # n_agent x 2
acs_agent["heading"].append(acs_env.agent_ang.tolist()) # n_agent x 1
acs_agent["heading_rate"].append(acs_env.agent_omg.tolist()) # n_agent x 1
acs_agent["acs_u"].append(acs_env.u_fully_active.tolist()) # n_agent
acs_agent["activness"].append(np.ones(num_agents).tolist()) # n_agent
acs_agent["final_u"].append(acs_env.u_lazy.tolist()) # n_agent
    
done = False
converged = False
max_time = acs_env.max_time_step
for t in range(max_time):
    action = np.ones(num_agents, dtype=np.float32)
    acs_obs, reward, done, _ = acs_env.step(action)

    # Get the reward
    acs_reward_list.append(reward)
    acs_cost_list.append(acs_env.target_reward)

    acs_agent["pos"].append(acs_env.agent_pos.tolist()) # n_agent x 2
    acs_agent["vel"].append(acs_env.agent_vel.tolist()) # n_agent x 2
    acs_agent["heading"].append(acs_env.agent_ang.tolist()) # n_agent x 1
    acs_agent["heading_rate"].append(acs_env.agent_omg.tolist()) # n_agent x 1
    acs_agent["acs_u"].append(acs_env.u_fully_active.tolist()) # n_agent
    acs_agent["activness"].append(action.tolist()) # n_agent
    acs_agent["final_u"].append(acs_env.u_lazy.tolist()) # n_agent

    if done and (not converged):
        acs_agent["converged_time"] = t
        converged = True

# convert agent info to numpy
for key in acs_agent.keys():
    acs_agent[key] = np.array(acs_agent[key])
acs_agent["pos_hist"] = acs_env.std_pos_hist
acs_agent["vel_hist"] = acs_env.std_vel_hist


# plot the results
# plot pos log with scatter
fig, ax = plt.subplots(2,1)
for i in range(num_agents):
    ax[0].scatter(agent["pos"][:, i, 0], agent["pos"][:, i, 1], s=1)
    ax[1].scatter(acs_agent["pos"][:, i, 0], acs_agent["pos"][:, i, 1], s=1)
ax[0].set_title("RL")
ax[0].set_aspect('equal')
ax[1].set_title("ACS")
ax[1].set_aspect('equal')
fig.tight_layout()

# plot acs_u, activness, final_u in one figure
fig, ax = plt.subplots(3, 2)
for i in range(num_agents):
    ax[0, 0].plot(agent["acs_u"][:, i])
    ax[0, 1].plot(acs_agent["acs_u"][:, i])
    ax[1, 0].plot(agent["activness"][:, i])
    ax[1, 1].plot(acs_agent["activness"][:, i])
    ax[2, 0].plot(agent["final_u"][:, i])
    ax[2, 1].plot(acs_agent["final_u"][:, i])

ax[0, 0].set_title("acs_u")
ax[0, 1].set_title("acs_u")
ax[1, 0].set_title("activness")
ax[1, 1].set_title("activness")
ax[2, 0].set_title("final_u")
ax[2, 1].set_title("final_u")
fig.tight_layout()

# plot to compare std_pos_hist and std_vel_hist between RL and ACS
fig2, ax2 = plt.subplots(2,1)
# std pos hist
ax2[0].plot(range(max_time), agent["pos_hist"], label="RL")
ax2[0].plot(range(max_time), acs_agent["pos_hist"], label="ACS")
ax2[0].legend()
# std vel hist
ax2[1].plot(range(max_time), agent["vel_hist"], label="RL")
ax2[1].plot(range(max_time), acs_agent["vel_hist"], label="ACS")
ax2[1].legend()
fig2.tight_layout()

plt.show()

# Save the data in one file with pickle
data = {
    "agent": agent,
    "reward_list": reward_list,
    "cost_list": cost_list,
    "acs_agent": acs_agent,
    "acs_reward_list": acs_reward_list,
    "acs_cost_list": acs_cost_list,
}

file_name = "rl_vs_acs_seed_" + str(seed) + ".pkl"
with open(file_name, 'wb') as f:
    pickle.dump(data, f)