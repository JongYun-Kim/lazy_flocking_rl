"""
Compare PPO checkpoint 74 against the ACS (fully-active) baseline on
``LazyAgentsCentralized`` with paired seeds.

Eval setup
----------
* Env: ``LazyAgentsCentralized`` (fully-connected, centralized obs).
* ``use_fixed_horizon = False`` -- episodes terminate on convergence.
* ``use_L2_norm = True``        -- per-step reward = -(control_cost_L2 + rho*dt),
  so the episode reward sum penalises the control term in L2 (not L1).
* Convergence thresholds from the checkpoint's ``params.json``:
  ``std_pos_converged=45``, ``std_vel_converged=0.1``,
  ``std_pos_rate_converged=0.1``, ``std_vel_rate_converged=0.2``.
* ``num_agents = 20``, ``max_time_step = 1000``, ``num_episodes = 200``.
* Paired seeds: episode ``k`` uses the same seed for policy and ACS.

Per-step control cost is recovered from ``reward`` via
``control_cost_L2 = -reward - rho * dt`` (holds whenever
``incomplete_episode_penalty = 0``, which is the case here).

Usage
-----
    python -m experiments.checkpoint_test_260416.eval_ckpt74_vs_acs \
        --num_episodes 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import List

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from env.envs import LazyAgentsCentralized  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_CHECKPOINT = os.path.join(
    WORKSPACE,
    "bk/bk_082623/PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47/"
    "checkpoint_000074/policies/default_policy",
)


def make_env_config(num_agents: int, max_steps: int) -> dict:
    # Thresholds match the checkpoint's params.json.
    return {
        "num_agents_max": num_agents,
        "num_agents_min": num_agents,
        "speed": 15,
        "predefined_distance": 60,
        "std_pos_converged": 45.0,
        "std_vel_converged": 0.1,
        "std_pos_rate_converged": 0.1,
        "std_vel_rate_converged": 0.2,
        "max_time_step": max_steps,
        "incomplete_episode_penalty": 0,
        "normalize_obs": True,
        "use_fixed_horizon": False,  # terminate on convergence
        "use_L2_norm": True,         # reward uses L2 control cost
        "use_preprocessed_obs": True,
        "auto_step": False,
    }


def run_episode(env: LazyAgentsCentralized, max_steps: int, *, mode: str, policy=None) -> dict:
    """Run one episode. ``mode`` is ``"acs"`` or ``"policy"``.

    Tracks reward sum, length, and per-step control cost (L2).
    """

    rho = env.rho
    dt = env.dt
    time_cost_per_step = rho * dt

    obs = env.reset()
    reward_sum = 0.0
    control_cost_sum = 0.0
    length = max_steps
    for t in range(max_steps):
        if mode == "acs":
            action = env.get_fully_active_action()
        elif mode == "policy":
            action = policy.compute_single_action(obs, explore=False)[0]
            action = np.clip(action, 0.0, 1.0)
        else:
            raise ValueError(f"unknown mode: {mode}")

        obs, reward, done, _ = env.step(action)
        reward_sum += float(reward)
        # reward = -(control_cost_L2 + rho*dt)  =>  control_cost = -reward - rho*dt
        control_cost_sum += float(-reward - time_cost_per_step)
        if done:
            length = t + 1
            break

    final_std_pos = float(env.std_pos_hist[length - 1])
    final_std_vel = float(env.std_vel_hist[length - 1])
    converged = length < max_steps  # env only terminates early on convergence

    return {
        "reward_sum": float(reward_sum),
        "length": int(length),
        "control_cost_sum": float(control_cost_sum),
        "final_std_pos": final_std_pos,
        "final_std_vel": final_std_vel,
        "converged": bool(converged),
    }


def aggregate(episodes: List[dict]) -> dict:
    r = np.array([e["reward_sum"] for e in episodes], dtype=np.float64)
    l = np.array([e["length"] for e in episodes], dtype=np.float64)
    c = np.array([e["control_cost_sum"] for e in episodes], dtype=np.float64)
    fp = np.array([e["final_std_pos"] for e in episodes], dtype=np.float64)
    fv = np.array([e["final_std_vel"] for e in episodes], dtype=np.float64)
    conv = np.array([e["converged"] for e in episodes], dtype=bool)
    return {
        "num_episodes": int(len(episodes)),
        "reward_mean": float(r.mean()),
        "reward_std": float(r.std()),
        "length_mean": float(l.mean()),
        "length_std": float(l.std()),
        "control_cost_mean": float(c.mean()),
        "control_cost_std": float(c.std()),
        "final_std_pos_mean": float(fp.mean()),
        "final_std_vel_mean": float(fv.mean()),
        "converged_rate": float(conv.mean()),
        "num_converged": int(conv.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(THIS_DIR, "results", "ckpt74_vs_acs.json"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    from ray.rllib.policy.policy import Policy
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env
    from models.lazy_allocator import MyRLlibTorchWrapper

    ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
    register_env("lazy_env", lambda cfg: LazyAgentsCentralized(cfg))

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    policy = Policy.from_checkpoint(args.checkpoint)
    policy.model.eval()
    print("Checkpoint loaded.")

    env_config = make_env_config(args.num_agents, args.max_steps)
    print(
        f"\nEvaluating {args.num_episodes} paired episodes, "
        f"num_agents={args.num_agents}, max_steps={args.max_steps}, "
        f"base_seed={args.base_seed}"
    )
    print(
        "(use_fixed_horizon=False, use_L2_norm=True; "
        "episodes end on std-based convergence)"
    )

    policy_eps: List[dict] = []
    acs_eps: List[dict] = []
    t0 = time.time()
    log_every = max(1, args.num_episodes // 20)

    for ep in range(args.num_episodes):
        seed = args.base_seed + ep

        env_p = LazyAgentsCentralized(env_config)
        env_p.seed(seed)
        policy_eps.append(run_episode(env_p, args.max_steps, mode="policy", policy=policy))

        env_a = LazyAgentsCentralized(env_config)
        env_a.seed(seed)
        acs_eps.append(run_episode(env_a, args.max_steps, mode="acs"))

        if (ep + 1) % log_every == 0 or ep == args.num_episodes - 1:
            elapsed = time.time() - t0
            print(
                f"  [{ep+1:4d}/{args.num_episodes}] "
                f"policy R={policy_eps[-1]['reward_sum']:7.2f} L={policy_eps[-1]['length']:4d}  "
                f"acs R={acs_eps[-1]['reward_sum']:7.2f} L={acs_eps[-1]['length']:4d}  "
                f"({elapsed:.1f}s)"
            )

    elapsed = time.time() - t0

    policy_agg = aggregate(policy_eps)
    acs_agg = aggregate(acs_eps)

    # Paired differences (policy - acs) per episode.
    r_p = np.array([e["reward_sum"] for e in policy_eps])
    r_a = np.array([e["reward_sum"] for e in acs_eps])
    l_p = np.array([e["length"] for e in policy_eps])
    l_a = np.array([e["length"] for e in acs_eps])
    c_p = np.array([e["control_cost_sum"] for e in policy_eps])
    c_a = np.array([e["control_cost_sum"] for e in acs_eps])
    paired = {
        "reward_diff_mean": float((r_p - r_a).mean()),
        "reward_diff_std": float((r_p - r_a).std()),
        "length_diff_mean": float((l_p - l_a).mean()),
        "control_cost_diff_mean": float((c_p - c_a).mean()),
        "policy_beats_acs_reward_rate": float((r_p > r_a).mean()),
    }

    sep = "-" * 78
    print(sep)
    print(f"{'metric':<30s} {'policy(ckpt74)':>18s} {'ACS':>18s}")
    print(sep)
    print(f"{'episode reward (mean)':<30s} {policy_agg['reward_mean']:18.3f} {acs_agg['reward_mean']:18.3f}")
    print(f"{'episode reward (std)':<30s} {policy_agg['reward_std']:18.3f} {acs_agg['reward_std']:18.3f}")
    print(f"{'episode length (mean)':<30s} {policy_agg['length_mean']:18.2f} {acs_agg['length_mean']:18.2f}")
    print(f"{'episode length (std)':<30s} {policy_agg['length_std']:18.2f} {acs_agg['length_std']:18.2f}")
    print(f"{'control cost L2 (mean)':<30s} {policy_agg['control_cost_mean']:18.4f} {acs_agg['control_cost_mean']:18.4f}")
    print(f"{'control cost L2 (std)':<30s} {policy_agg['control_cost_std']:18.4f} {acs_agg['control_cost_std']:18.4f}")
    print(f"{'converged rate':<30s} {policy_agg['converged_rate']:18.3f} {acs_agg['converged_rate']:18.3f}")
    print(f"{'final std_pos (mean)':<30s} {policy_agg['final_std_pos_mean']:18.3f} {acs_agg['final_std_pos_mean']:18.3f}")
    print(f"{'final std_vel (mean)':<30s} {policy_agg['final_std_vel_mean']:18.3f} {acs_agg['final_std_vel_mean']:18.3f}")
    print(sep)
    print(f"paired  reward diff mean (policy - acs): {paired['reward_diff_mean']:+.3f} "
          f"(std {paired['reward_diff_std']:.3f}, policy>ACS in {100*paired['policy_beats_acs_reward_rate']:.1f}% of eps)")
    print(f"paired  length diff mean (policy - acs): {paired['length_diff_mean']:+.2f}")
    print(f"paired  ctrl-cost diff mean (policy - acs): {paired['control_cost_diff_mean']:+.4f}")
    print(f"Total elapsed: {elapsed:.1f}s")

    out_doc = {
        "args": vars(args),
        "env_config": env_config,
        "policy": policy_agg,
        "acs": acs_agg,
        "paired": paired,
        "policy_episodes": policy_eps,
        "acs_episodes": acs_eps,
    }
    with open(args.output, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
