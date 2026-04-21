"""Scalability experiment (Comparison 2): sweep UAV counts for a given method.

Usage:
    python -m experiments.performance_benchmark.collect_scalability --method acs
    python -m experiments.performance_benchmark.collect_scalability --method heuristic
    python -m experiments.performance_benchmark.collect_scalability --method rl --checkpoint <path>
    python -m experiments.performance_benchmark.collect_scalability --method pso

Run each method sequentially (Ray resources overlap).
Results are saved per UAV count under results/scalability/.
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
    CHECKPOINT_PATH, PSO_SKIP_AGENTS, SCALABILITY_AGENTS, SEEDS,
    WORKSPACE as WS, build_env_config, episode_metrics, save_results,
)

MAX_TIME_STEP = 2000


# ---------------------------------------------------------------------------
# ACS / Heuristic — stateless Ray remote functions
# ---------------------------------------------------------------------------

@ray.remote
def _run_acs_ep(seed, env_config, workspace):
    import sys
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    from env.envs import LazyAgentsCentralized
    from experiments.performance_benchmark.config import episode_metrics

    env = LazyAgentsCentralized(env_config)
    env.seed(seed)
    env.reset()
    action = env.get_fully_active_action()

    done, r1, r2 = False, 0.0, 0.0
    while not done:
        _, _, done, info = env.step(action)
        r = info["original_rewards"]
        r1 += r[0]; r2 += r[1]

    m = episode_metrics(env, r1, r2, info)
    m["seed"] = seed
    return m


@ray.remote
def _run_heuristic_ep(seed, env_config, workspace):
    import sys
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    from env.envs import LazyAgentsCentralized
    from experiments.performance_benchmark.config import episode_metrics

    env = LazyAgentsCentralized(env_config)
    env.seed(seed)
    env.reset()

    done, r1, r2 = False, 0.0, 0.0
    while not done:
        action = env.compute_heuristic_action()
        _, _, done, info = env.step(action)
        r = info["original_rewards"]
        r1 += r[0]; r2 += r[1]

    m = episode_metrics(env, r1, r2, info)
    m["seed"] = seed
    return m


# ---------------------------------------------------------------------------
# RL — Ray actor (loads checkpoint once per worker)
# ---------------------------------------------------------------------------

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
        from experiments.performance_benchmark.config import episode_metrics

        env = LazyAgentsCentralized(env_config)
        env.seed(seed)
        obs = env.reset()

        done, r1, r2 = False, 0.0, 0.0
        while not done:
            action = self.policy.compute_single_action(obs, explore=False)[0]
            action = np.clip(action, 0.0, 1.0)
            obs, _, done, info = env.step(action)
            r = info["original_rewards"]
            r1 += r[0]; r2 += r[1]

        m = episode_metrics(env, r1, r2, info)
        m["seed"] = seed
        return m


# ---------------------------------------------------------------------------
# Collectors (call after ray.init)
# ---------------------------------------------------------------------------

def collect_acs(env_config, seeds):
    futures = [_run_acs_ep.remote(s, env_config, WS) for s in seeds]
    return ray.get(futures)


def collect_heuristic(env_config, seeds):
    futures = [_run_heuristic_ep.remote(s, env_config, WS) for s in seeds]
    return ray.get(futures)


def collect_rl(env_config, seeds, checkpoint, num_workers):
    workers = [_RLWorker.remote(checkpoint, WS) for _ in range(num_workers)]
    futures = []
    for i, seed in enumerate(seeds):
        futures.append(workers[i % num_workers].run_episode.remote(seed, env_config))
    results = ray.get(futures)
    del workers
    return results


def collect_pso(env_config, seeds, num_cpus, maxfe):
    from env.envs import LazyAgentsCentralized
    from utils.metaheuristics import PSOActionOptimizer

    optimizer = PSOActionOptimizer(num_cpus=num_cpus)
    results = []
    for i, seed in enumerate(seeds):
        env = LazyAgentsCentralized(env_config)
        env.seed(seed)
        env.reset()

        optimal_action, cost, elapsed = optimizer.optimize(env, maxfe=maxfe)

        done, r1, r2 = False, 0.0, 0.0
        while not done:
            _, _, done, info = env.step(optimal_action)
            r = info["original_rewards"]
            r1 += r[0]; r2 += r[1]

        m = episode_metrics(env, r1, r2, info)
        m["seed"] = seed
        m["pso_opt_time"] = float(elapsed)
        m["pso_cost"] = float(cost)
        results.append(m)

        print(f"    [{i+1}/{len(seeds)}] seed={seed}  "
              f"opt={elapsed:.0f}s  steps={m['episode_length']}  "
              f"rL1={m['reward_L1']:.1f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scalability experiment")
    parser.add_argument("--method", required=True,
                        choices=["acs", "heuristic", "rl", "pso"])
    parser.add_argument("--num_cpus", type=int, default=64)
    parser.add_argument("--agents", type=int, nargs="*", default=None,
                        help="UAV counts to evaluate (default: all scalability counts)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="RL checkpoint path")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="number of Ray actors for RL")
    parser.add_argument("--maxfe", type=int, default=None,
                        help="PSO max function evaluations")
    args = parser.parse_args()

    agent_counts = args.agents if args.agents else SCALABILITY_AGENTS

    ray.init(num_cpus=args.num_cpus)
    t_total = time.time()

    for n in agent_counts:
        if args.method == "pso" and n in PSO_SKIP_AGENTS:
            print(f"\n[{args.method.upper()} n={n}] skipped")
            continue

        use_heuristics = args.method in ("heuristic", "rl")
        env_config = build_env_config(
            num_agents=n, max_time_step=MAX_TIME_STEP,
            use_heuristics=use_heuristics,
        )
        if args.method == "pso":
            env_config["use_L2_norm"] = True

        print(f"\n[{args.method.upper()} n={n}] collecting {len(SEEDS)} episodes ...")
        t0 = time.time()

        if args.method == "acs":
            results = collect_acs(env_config, SEEDS)
        elif args.method == "heuristic":
            results = collect_heuristic(env_config, SEEDS)
        elif args.method == "rl":
            results = collect_rl(env_config, SEEDS, args.checkpoint,
                                 args.num_workers)
        elif args.method == "pso":
            results = collect_pso(env_config, SEEDS, args.num_cpus,
                                  args.maxfe)

        elapsed = time.time() - t0
        tag = f"{args.method}_n{n}"
        save_results(tag, results, env_config, subdir="scalability")
        print(f"  done in {elapsed:.1f}s")

    ray.shutdown()
    print(f"\nTotal wall-clock: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
