"""Collect PSO-optimised episodes.

Each episode: run SLPSO to find the optimal constant action, then evaluate it.
Progress is saved after every episode so partial results survive interruptions.
"""

import argparse
import json
import os
import sys
import time

import ray

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from env.envs import LazyAgentsCentralized
from experiments.performance_benchmark.config import (
    RESULTS_DIR, SEEDS, build_env_config, episode_metrics, save_results,
)
from utils.metaheuristics import PSOActionOptimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cpus", type=int, default=64)
    parser.add_argument("--maxfe", type=int, default=None,
                        help="max function evaluations per PSO run (default: d*5000)")
    parser.add_argument("--resume", action="store_true",
                        help="skip seeds already present in the partial file")
    parser.add_argument("--seed_pso", action="store_true",
                        help="propagate per-episode seed into PSO's internal RNG "
                             "(deterministic PSO). Default: off — preserves legacy "
                             "non-deterministic PSO runs for compatibility.")
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus)
    optimizer = PSOActionOptimizer(num_cpus=args.num_cpus)
    env_config = build_env_config(use_heuristics=False)
    env_config["use_L2_norm"] = True  # PSO optimises L2 episode reward

    partial_path = os.path.join(RESULTS_DIR, "pso_partial.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    done_seeds = set()
    if args.resume and os.path.exists(partial_path):
        with open(partial_path) as f:
            results = json.load(f)["results"]
        done_seeds = {r["seed"] for r in results}
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

        done = False
        r_L1, r_L2 = 0.0, 0.0
        while not done:
            _, _, done, info = env.step(optimal_action)
            r = info["original_rewards"]
            r_L1 += r[0]
            r_L2 += r[1]

        m = episode_metrics(env, r_L1, r_L2, info)
        m["seed"] = seed
        m["pso_opt_time"] = float(elapsed)
        m["pso_cost"] = float(cost)
        results.append(m)

        print(
            f"[{len(results)}/{len(SEEDS)}] seed={seed}  "
            f"opt={elapsed:.0f}s  steps={m['episode_length']}  "
            f"rL1={m['reward_L1']:.1f}"
        )

        with open(partial_path, "w") as f:
            json.dump({"results": results}, f, indent=2)

    ray.shutdown()

    save_results("PSO", results, env_config)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"Total wall-clock: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
