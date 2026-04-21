"""Shared configuration for the performance benchmark."""

import json
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
RESULTS_DIR = os.path.join(THIS_DIR, "results")

# Comparison 1: fixed 20 UAVs
NUM_AGENTS = 20
SEEDS = list(range(801, 901))  # 100 episodes

# Comparison 2: scalability sweep
SCALABILITY_AGENTS = [8, 16, 32, 64, 128, 256, 512, 1024]
SCALABILITY_SEEDS = list(range(1, 1001))  # 1000 episodes
PSO_SKIP_AGENTS = {512, 1024}

CHECKPOINT_PATH = os.path.join(
    WORKSPACE,
    "bk/bk_082623/"
    "PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47/"
    "checkpoint_000074/policies/default_policy",
)


def build_env_config(num_agents=NUM_AGENTS, max_time_step=1000,
                     use_heuristics=True):
    return {
        "num_agents_max": num_agents,
        "num_agents_min": num_agents,
        "speed": 15,
        "predefined_distance": 60,
        "std_pos_converged": 45,
        "std_vel_converged": 0.1,
        "std_pos_rate_converged": 0.1,
        "std_vel_rate_converged": 0.2,
        "max_time_step": max_time_step,
        "incomplete_episode_penalty": 0,
        "normalize_obs": True,
        "use_fixed_horizon": False,
        "_use_fixed_lazy_idx": True,
        "use_preprocessed_obs": True,
        "use_mlp_settings": False,
        "get_state_hist": False,
        "use_L2_norm": False,
        "use_heuristics": use_heuristics,
    }


def episode_metrics(env, reward_L1, reward_L2, last_info):
    """Extract standard metrics from a finished episode."""
    ct = env.time_step * env.dt
    return {
        "episode_length": int(env.time_step),
        "convergence_time": float(ct),
        "reward_L1": float(reward_L1),
        "reward_L2": float(reward_L2),
        "control_cost_L1": float(-reward_L1 - ct),
        "control_cost_L2": float(-reward_L2 - ct),
        "std_pos_last": float(last_info["std_pos"]),
        "std_vel_last": float(last_info["std_vel"]),
    }


def save_results(method, results, env_config, extra=None, subdir=None):
    out_dir = os.path.join(RESULTS_DIR, subdir) if subdir else RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{method.lower()}.json")

    doc = {
        "method": method,
        "num_agents": env_config["num_agents_max"],
        "num_episodes": len(results),
        "seeds": SEEDS,
        "env_config": env_config,
        "results": sorted(results, key=lambda x: x["seed"]),
    }
    if extra:
        doc.update(extra)
    with open(output_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Saved {len(results)} episodes → {output_path}")
    return output_path
