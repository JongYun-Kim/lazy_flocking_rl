"""Scalability experiment (Comparison 2): sweep UAV counts for a given method.

Usage:
    python -m experiments.performance_benchmark.collect_scalability --method acs
    python -m experiments.performance_benchmark.collect_scalability --method heuristic
    python -m experiments.performance_benchmark.collect_scalability --method rl
    python -m experiments.performance_benchmark.collect_scalability --method rl --device cuda
    python -m experiments.performance_benchmark.collect_scalability --method pso

Run each method sequentially (Ray resources overlap).
Results are saved per UAV count under results/scalability/.

RL notes:
    The trained checkpoint (num_agents_max=20) is used directly via the raw
    policy_network (LazinessAllocator), bypassing RLlib's fixed-size
    observation/action spaces. The transformer handles arbitrary sequence
    lengths natively. Pass --device cuda for GPU inference (recommended for
    n >= 64 based on profiling).
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
    CHECKPOINT_PATH, PSO_SKIP_AGENTS, SCALABILITY_AGENTS, SCALABILITY_SEEDS,
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
# RL — Ray actor using raw policy_network (works for any num_agents)
# ---------------------------------------------------------------------------

@ray.remote
class _RLWorker:
    """Runs the raw LazinessAllocator, bypassing RLlib's fixed-size spaces."""

    def __init__(self, checkpoint_path, device, workspace):
        import sys
        if workspace not in sys.path:
            sys.path.insert(0, workspace)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        import torch
        from ray.rllib.models import ModelCatalog
        from ray.rllib.policy.policy import Policy
        from ray.tune.registry import register_env
        from env.envs import LazyAgentsCentralized
        from models.lazy_allocator import MyRLlibTorchWrapper

        ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
        register_env("lazy_env", lambda cfg: LazyAgentsCentralized(cfg))

        policy = Policy.from_checkpoint(checkpoint_path)
        policy.model.eval()
        self.net = policy.model.policy_network
        self.device = torch.device(device)
        self.net.to(self.device)

    def run_episode(self, seed, env_config):
        import torch
        import numpy as np
        from env.envs import LazyAgentsCentralized
        from experiments.performance_benchmark.config import episode_metrics

        env = LazyAgentsCentralized(env_config)
        env.seed(seed)
        obs = env.reset()
        n = env_config["num_agents_max"]

        done, r1, r2 = False, 0.0, 0.0
        while not done:
            ae = torch.as_tensor(
                obs["agent_embeddings"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            pt = torch.as_tensor(
                obs["pad_tokens"], dtype=torch.int32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                out, _, _ = self.net({"agent_embeddings": ae, "pad_tokens": pt})
            half = out.shape[-1] // 2
            action = out[0, :half].clamp(0.0, 1.0).cpu().numpy()

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


def collect_rl(env_config, seeds, checkpoint, device, num_workers):
    if device == "cuda":
        import torch
        num_gpus = torch.cuda.device_count()
        gpu_per_worker = num_gpus / num_workers
    else:
        gpu_per_worker = 0
    workers = [
        _RLWorker.options(num_gpus=gpu_per_worker).remote(checkpoint, device, WS)
        for _ in range(num_workers)
    ]
    futures = []
    for i, seed in enumerate(seeds):
        futures.append(workers[i % num_workers].run_episode.remote(seed, env_config))
    results = ray.get(futures)
    del workers
    return results


def collect_pso(env_config, seeds, num_cpus, maxfe, partial_dir):
    from env.envs import LazyAgentsCentralized
    from utils.metaheuristics import PSOActionOptimizer

    n = env_config["num_agents_max"]
    partial_path = os.path.join(partial_dir, f"pso_n{n}_partial.json")

    # Resume support
    results = []
    done_seeds = set()
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            results = json.load(f)["results"]
        done_seeds = {r["seed"] for r in results}
        if done_seeds:
            print(f"    resuming: {len(done_seeds)} already done")

    optimizer = PSOActionOptimizer(num_cpus=num_cpus)
    for i, seed in enumerate(seeds):
        if seed in done_seeds:
            continue

        env = LazyAgentsCentralized(env_config)
        env.seed(seed)
        env.reset()

        optimal_action, cost, elapsed = optimizer.optimize(env, maxfe=maxfe)

        done = False
        r1, r2 = 0.0, 0.0
        while not done:
            _, _, done, info = env.step(optimal_action)
            r = info["original_rewards"]
            r1 += r[0]; r2 += r[1]

        m = episode_metrics(env, r1, r2, info)
        m["seed"] = seed
        m["pso_opt_time"] = float(elapsed)
        m["pso_cost"] = float(cost)
        results.append(m)

        print(f"    [{len(results)}/{len(seeds)}] seed={seed}  "
              f"opt={elapsed:.0f}s  steps={m['episode_length']}  "
              f"rL1={m['reward_L1']:.1f}")

        with open(partial_path, "w") as f:
            json.dump({"results": results}, f, indent=2)

    # Clean up partial
    if os.path.exists(partial_path):
        os.remove(partial_path)
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
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="RL inference device (cuda recommended for n>=64)")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="number of Ray actors for RL (default matches --num_cpus)")
    parser.add_argument("--maxfe", type=int, default=None,
                        help="PSO max function evaluations (default: d*5000)")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="limit episodes per scale (default: all seeds)")
    args = parser.parse_args()

    agent_counts = args.agents if args.agents else SCALABILITY_AGENTS
    seeds = SCALABILITY_SEEDS[:args.num_episodes] if args.num_episodes else SCALABILITY_SEEDS

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

        print(f"\n[{args.method.upper()} n={n}] {len(seeds)} episodes ...")
        t0 = time.time()

        if args.method == "acs":
            results = collect_acs(env_config, seeds)
        elif args.method == "heuristic":
            results = collect_heuristic(env_config, seeds)
        elif args.method == "rl":
            results = collect_rl(env_config, seeds, args.checkpoint,
                                 args.device, args.num_workers)
        elif args.method == "pso":
            from experiments.performance_benchmark.config import RESULTS_DIR
            partial_dir = os.path.join(RESULTS_DIR, "scalability")
            os.makedirs(partial_dir, exist_ok=True)
            results = collect_pso(env_config, seeds, args.num_cpus,
                                  args.maxfe, partial_dir)

        elapsed = time.time() - t0
        tag = f"{args.method}_n{n}"
        save_results(tag, results, env_config, subdir="scalability")
        print(f"  done in {elapsed:.1f}s")

    ray.shutdown()
    print(f"\nTotal wall-clock: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
