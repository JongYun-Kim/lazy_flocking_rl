"""Collect full-horizon (1000 steps) entropy trajectories for a single seed.

Unlike collect_trajectories.py which stops at convergence, this script
forces the episode to run for the full max_time_step by setting
use_fixed_horizon=True.

Usage:
    python -m experiments.performance_benchmark.collect_full_horizon
"""

import json
import os
import sys
import warnings

import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from experiments.performance_benchmark.config import CHECKPOINT_PATH, RESULTS_DIR, build_env_config

TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectories")
SEED = 868
MAX_TIME_STEP = 1000


def run_acs(seed, env_config):
    from env.envs import LazyAgentsCentralized

    env = LazyAgentsCentralized(env_config)
    env.seed(seed)
    env.reset()
    action = env.get_fully_active_action()

    done = False
    while not done:
        _, _, done, _ = env.step(action)

    T = env.time_step
    return {
        "seed": seed,
        "episode_length": T,
        "spatial_entropy": env.std_pos_hist[:T].tolist(),
        "velocity_entropy": env.std_vel_hist[:T].tolist(),
    }


def run_rl(seed, env_config):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ray.rllib.models import ModelCatalog
    from ray.rllib.policy.policy import Policy
    from ray.tune.registry import register_env
    from env.envs import LazyAgentsCentralized
    from models.lazy_allocator import MyRLlibTorchWrapper

    ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
    register_env("lazy_env", lambda cfg: LazyAgentsCentralized(cfg))
    policy = Policy.from_checkpoint(CHECKPOINT_PATH)
    policy.model.eval()

    env = LazyAgentsCentralized(env_config)
    env.seed(seed)
    obs = env.reset()

    done = False
    while not done:
        action = policy.compute_single_action(obs, explore=False)[0]
        action = np.clip(action, 0.0, 1.0)
        obs, _, done, _ = env.step(action)

    T = env.time_step
    return {
        "seed": seed,
        "episode_length": T,
        "spatial_entropy": env.std_pos_hist[:T].tolist(),
        "velocity_entropy": env.std_vel_hist[:T].tolist(),
    }


def main():
    os.makedirs(TRAJ_DIR, exist_ok=True)

    for method, use_heuristics in [("acs", False), ("rl", True)]:
        env_config = build_env_config(
            max_time_step=MAX_TIME_STEP,
            use_heuristics=use_heuristics,
        )
        env_config["use_fixed_horizon"] = True

        print(f"Running {method.upper()} seed={SEED} for {MAX_TIME_STEP} steps ...")
        if method == "acs":
            result = run_acs(SEED, env_config)
        else:
            result = run_rl(SEED, env_config)

        out_path = os.path.join(TRAJ_DIR, f"{method}_full_seed{SEED}.json")
        doc = {
            "method": method.upper(),
            "num_agents": env_config["num_agents_max"],
            "max_time_step": MAX_TIME_STEP,
            "use_fixed_horizon": True,
            "seeds": [SEED],
            "results": [result],
        }
        with open(out_path, "w") as f:
            json.dump(doc, f)
        print(f"  episode_length={result['episode_length']}  →  {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
