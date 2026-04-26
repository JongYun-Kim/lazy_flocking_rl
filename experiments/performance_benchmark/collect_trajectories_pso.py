"""Collect PSO per-timestep entropy trajectories and save optimal actions.

PSO uses Ray internally for particle evaluation, so episodes run sequentially.
Partial results are saved after every episode for resume support.

Usage:
    python -m experiments.performance_benchmark.collect_trajectories_pso --num_cpus 64 --seed_pso
    python -m experiments.performance_benchmark.collect_trajectories_pso --num_cpus 64 --seed_pso --resume
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import ray

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from env.envs import LazyAgentsCentralized
from experiments.performance_benchmark.config import (
    RESULTS_DIR, SEEDS, build_env_config,
)
from utils.metaheuristics import PSOActionOptimizer

TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectories")


def main():
    parser = argparse.ArgumentParser(
        description="Collect PSO trajectories and optimal actions")
    parser.add_argument("--num_cpus", type=int, default=64)
    parser.add_argument("--maxfe", type=int, default=None,
                        help="max function evaluations per PSO run (default: d*5000)")
    parser.add_argument("--seed_pso", action="store_true",
                        help="deterministic PSO seed control")
    parser.add_argument("--resume", action="store_true",
                        help="resume from partial file")
    args = parser.parse_args()

    os.makedirs(TRAJ_DIR, exist_ok=True)
    partial_path = os.path.join(TRAJ_DIR, "pso_partial.json")

    ray.init(num_cpus=args.num_cpus)
    optimizer = PSOActionOptimizer(num_cpus=args.num_cpus)

    env_config = build_env_config(use_heuristics=False)
    env_config["use_L2_norm"] = True

    trajectories = []
    actions = []
    done_seeds = set()

    if args.resume and os.path.exists(partial_path):
        with open(partial_path) as f:
            partial = json.load(f)
        trajectories = partial["trajectories"]
        actions = partial["actions"]
        done_seeds = {r["seed"] for r in trajectories}
        print(f"Resuming: {len(done_seeds)} episodes already done")

    t_total = time.time()
    for i, seed in enumerate(SEEDS):
        if seed in done_seeds:
            continue

        env = LazyAgentsCentralized(env_config)
        env.seed(seed)
        env.reset()

        optimal_action, cost, elapsed = optimizer.optimize(
            env, maxfe=args.maxfe,
            seed=seed if args.seed_pso else None,
        )

        env2 = LazyAgentsCentralized(env_config)
        env2.seed(seed)
        env2.reset()
        done = False
        while not done:
            _, _, done, _ = env2.step(optimal_action)

        T = env2.time_step
        trajectories.append({
            "seed": seed,
            "episode_length": T,
            "spatial_entropy": env2.std_pos_hist[:T].tolist(),
            "velocity_entropy": env2.std_vel_hist[:T].tolist(),
        })
        actions.append({
            "seed": seed,
            "optimal_action": optimal_action.tolist(),
            "cost": float(cost),
            "opt_time": float(elapsed),
        })

        print(
            f"[{len(trajectories)}/{len(SEEDS)}] seed={seed}  "
            f"opt={elapsed:.0f}s  steps={T}  cost={cost:.1f}"
        )

        with open(partial_path, "w") as f:
            json.dump({"trajectories": trajectories, "actions": actions}, f)

    ray.shutdown()

    # Save trajectory file
    traj_path = os.path.join(TRAJ_DIR, "pso.json")
    traj_doc = {
        "method": "PSO",
        "num_agents": env_config["num_agents_max"],
        "num_episodes": len(trajectories),
        "seeds": sorted({r["seed"] for r in trajectories}),
        "results": sorted(trajectories, key=lambda x: x["seed"]),
    }
    with open(traj_path, "w") as f:
        json.dump(traj_doc, f)
    print(f"Saved {len(trajectories)} trajectories → {traj_path}")

    # Save actions file
    actions_path = os.path.join(TRAJ_DIR, "pso_actions.json")
    actions_doc = {
        "method": "PSO",
        "num_agents": env_config["num_agents_max"],
        "num_episodes": len(actions),
        "seeds": sorted({a["seed"] for a in actions}),
        "env_config": env_config,
        "actions": sorted(actions, key=lambda x: x["seed"]),
    }
    with open(actions_path, "w") as f:
        json.dump(actions_doc, f, indent=2)
    print(f"Saved {len(actions)} actions → {actions_path}")

    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"Total wall-clock: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
