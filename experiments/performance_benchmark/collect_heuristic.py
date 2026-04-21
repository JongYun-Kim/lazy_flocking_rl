"""Collect Heuristic (sole-lazy) episodes with Ray parallelization."""

import argparse
import os
import sys
import time

import ray

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from experiments.performance_benchmark.config import (
    SEEDS, WORKSPACE as WS, build_env_config, save_results,
)


@ray.remote
def run_episode(seed, env_config, workspace):
    import sys
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    from env.envs import LazyAgentsCentralized
    from experiments.performance_benchmark.config import episode_metrics

    env = LazyAgentsCentralized(env_config)
    env.seed(seed)
    env.reset()

    done = False
    r_L1, r_L2 = 0.0, 0.0
    while not done:
        action = env.compute_heuristic_action()
        _, _, done, info = env.step(action)
        r = info["original_rewards"]
        r_L1 += r[0]
        r_L2 += r[1]

    m = episode_metrics(env, r_L1, r_L2, info)
    m["seed"] = seed
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cpus", type=int, default=64)
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus)
    env_config = build_env_config(use_heuristics=True)

    t0 = time.time()
    futures = [run_episode.remote(s, env_config, WS) for s in SEEDS]
    results = ray.get(futures)
    elapsed = time.time() - t0

    ray.shutdown()
    save_results("Heuristic", results, env_config)
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
