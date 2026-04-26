"""Collect per-timestep spatial/velocity entropy trajectories.

Saves the full std_pos and std_vel history for every episode, enabling
CI-band plots of entropy evolution over time.

Usage:
    python -m experiments.performance_benchmark.collect_trajectories --method acs
    python -m experiments.performance_benchmark.collect_trajectories --method heuristic
    python -m experiments.performance_benchmark.collect_trajectories --method rl
    python -m experiments.performance_benchmark.collect_trajectories --method pso
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import ray

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from experiments.performance_benchmark.config import (
    CHECKPOINT_PATH, RESULTS_DIR, SEEDS, WORKSPACE as WS, build_env_config,
)

TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectories")


@ray.remote
def _run_acs(seed, env_config, workspace):
    import sys
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
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


@ray.remote
def _run_heuristic(seed, env_config, workspace):
    import sys
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    from env.envs import LazyAgentsCentralized

    env = LazyAgentsCentralized(env_config)
    env.seed(seed)
    env.reset()

    done = False
    while not done:
        action = env.compute_heuristic_action()
        _, _, done, _ = env.step(action)

    T = env.time_step
    return {
        "seed": seed,
        "episode_length": T,
        "spatial_entropy": env.std_pos_hist[:T].tolist(),
        "velocity_entropy": env.std_vel_hist[:T].tolist(),
    }


@ray.remote
class _RLWorker:
    def __init__(self, checkpoint_path, workspace):
        import sys
        if workspace not in sys.path:
            sys.path.insert(0, workspace)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from ray.rllib.models import ModelCatalog
        from ray.rllib.policy.policy import Policy
        from ray.tune.registry import register_env
        from env.envs import LazyAgentsCentralized
        from models.lazy_allocator import MyRLlibTorchWrapper

        ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
        register_env("lazy_env", lambda cfg: LazyAgentsCentralized(cfg))
        self.policy = Policy.from_checkpoint(checkpoint_path)
        self.policy.model.eval()

    def run_episode(self, seed, env_config):
        import numpy as np
        from env.envs import LazyAgentsCentralized

        env = LazyAgentsCentralized(env_config)
        env.seed(seed)
        obs = env.reset()

        done = False
        while not done:
            action = self.policy.compute_single_action(obs, explore=False)[0]
            action = np.clip(action, 0.0, 1.0)
            obs, _, done, _ = env.step(action)

        T = env.time_step
        return {
            "seed": seed,
            "episode_length": T,
            "spatial_entropy": env.std_pos_hist[:T].tolist(),
            "velocity_entropy": env.std_vel_hist[:T].tolist(),
        }


@ray.remote
def _run_pso(seed, env_config, workspace, maxfe=None, seed_pso=False):
    import sys
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    import copy
    from env.envs import LazyAgentsCentralized
    from utils.metaheuristics import PSOActionOptimizer

    optimizer = PSOActionOptimizer()
    env = LazyAgentsCentralized(env_config)
    env.seed(seed)
    env.reset()

    optimal_action, cost, elapsed = optimizer.optimize(
        env, maxfe=maxfe, seed=seed if seed_pso else None,
    )

    env2 = LazyAgentsCentralized(env_config)
    env2.seed(seed)
    env2.reset()
    done = False
    while not done:
        _, _, done, _ = env2.step(optimal_action)

    T = env2.time_step
    return {
        "seed": seed,
        "episode_length": T,
        "pso_opt_time": float(elapsed),
        "spatial_entropy": env2.std_pos_hist[:T].tolist(),
        "velocity_entropy": env2.std_vel_hist[:T].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-timestep entropy trajectories")
    parser.add_argument("--method", required=True,
                        choices=["acs", "heuristic", "rl", "pso"])
    parser.add_argument("--num_cpus", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16,
                        help="number of Ray actors for RL (default: 16)")
    parser.add_argument("--maxfe", type=int, default=None,
                        help="PSO max function evaluations")
    parser.add_argument("--seed_pso", action="store_true",
                        help="deterministic PSO seed control")
    args = parser.parse_args()

    seeds = SEEDS
    use_heuristics = args.method in ("heuristic", "rl")
    env_config = build_env_config(use_heuristics=use_heuristics)
    if args.method == "pso":
        env_config["use_L2_norm"] = True

    ray.init(num_cpus=args.num_cpus)
    t0 = time.time()

    if args.method == "acs":
        futures = [_run_acs.remote(s, env_config, WS) for s in seeds]
        results = ray.get(futures)

    elif args.method == "heuristic":
        futures = [_run_heuristic.remote(s, env_config, WS) for s in seeds]
        results = ray.get(futures)

    elif args.method == "rl":
        workers = [_RLWorker.remote(CHECKPOINT_PATH, WS)
                   for _ in range(args.num_workers)]
        futures = []
        for i, seed in enumerate(seeds):
            futures.append(workers[i % args.num_workers]
                           .run_episode.remote(seed, env_config))
        results = ray.get(futures)

    elif args.method == "pso":
        futures = [_run_pso.remote(s, env_config, WS,
                                   args.maxfe, args.seed_pso)
                   for s in seeds]
        results = ray.get(futures)

    ray.shutdown()
    elapsed = time.time() - t0

    os.makedirs(TRAJ_DIR, exist_ok=True)
    out_path = os.path.join(TRAJ_DIR, f"{args.method}.json")
    doc = {
        "method": args.method.upper(),
        "num_agents": env_config["num_agents_max"],
        "num_episodes": len(results),
        "seeds": sorted({r["seed"] for r in results}),
        "results": sorted(results, key=lambda x: x["seed"]),
    }
    with open(out_path, "w") as f:
        json.dump(doc, f)
    print(f"Saved {len(results)} episodes → {out_path}")
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
