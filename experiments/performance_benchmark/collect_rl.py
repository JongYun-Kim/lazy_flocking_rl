"""Collect RL (trained PPO) episodes with Ray actors.

Each actor loads the checkpoint once and runs multiple episodes.
CPU-only: at ~2 ms/step the 100 episodes finish in seconds with 16 actors.
"""

import argparse
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
    CHECKPOINT_PATH, SEEDS, WORKSPACE as WS, build_env_config, save_results,
)


@ray.remote
class RLWorker:
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
        from experiments.performance_benchmark.config import episode_metrics

        env = LazyAgentsCentralized(env_config)
        env.seed(seed)
        obs = env.reset()

        done = False
        r_L1, r_L2 = 0.0, 0.0
        while not done:
            action = self.policy.compute_single_action(obs, explore=False)[0]
            action = np.clip(action, 0.0, 1.0)
            obs, _, done, info = env.step(action)
            r = info["original_rewards"]
            r_L1 += r[0]
            r_L2 += r[1]

        m = episode_metrics(env, r_L1, r_L2, info)
        m["seed"] = seed
        return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cpus", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus)
    env_config = build_env_config(use_heuristics=True)

    workers = [RLWorker.remote(CHECKPOINT_PATH, WS) for _ in range(args.num_workers)]

    t0 = time.time()
    futures = []
    for i, seed in enumerate(SEEDS):
        worker = workers[i % args.num_workers]
        futures.append(worker.run_episode.remote(seed, env_config))
    results = ray.get(futures)
    elapsed = time.time() - t0

    ray.shutdown()
    save_results("RL", results, env_config)
    print(f"Done in {elapsed:.1f}s ({args.num_workers} workers)")


if __name__ == "__main__":
    main()
