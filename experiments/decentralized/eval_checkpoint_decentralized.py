"""
Evaluate the centralized checkpoint with *decentralized* (per-agent) inference.

The previous eval script (``eval_checkpoint_networked.py``) gave the policy a
**centralized** observation — every agent saw the full swarm state — even on
sparse topologies.  That tests whether the policy tolerates different ACS
dynamics, but the policy still has global information.

This script tests the checkpoint properly: each agent observes **only its
topology neighbors** (via ``build_per_agent_central_obs``).  The per-agent
observations are batched into a single forward pass, and each agent's own
action is extracted from its row in the batch output.

Three modes are evaluated per topology on the same seed:

* **decentralized** — per-agent local observation, batched forward pass.
* **centralized**   — original global observation, single forward pass
  (included for direct comparison of information loss).
* **acs**           — fully-active ACS baseline (no learned policy).

The FC topology serves as a sanity check: decentralized and centralized modes
must produce identical trajectories because all agents see all other agents.

Usage
-----
    python -m experiments.decentralized.eval_checkpoint_decentralized \
        --num_episodes 30 \
        --output experiments/decentralized/results/checkpoint_eval_decentralized.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Iterable, List, Optional

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from experiments.decentralized.env_networked import NetworkedLazyAgents  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_CHECKPOINT = os.path.join(
    WORKSPACE,
    "bk/bk_082623/PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47/"
    "checkpoint_000074/policies/default_policy",
)

DEFAULT_TOPOLOGIES = [
    {"label": "fully_connected", "topology": "fully_connected"},
    {"label": "star", "topology": "star"},
    {"label": "wheel", "topology": "wheel"},
    {"label": "binary_tree", "topology": "binary_tree"},
    {"label": "line", "topology": "line"},
    {"label": "ring_k1", "topology": {"type": "ring", "k": 1}},
    {"label": "ring_k2", "topology": {"type": "ring", "k": 2}},
    {"label": "ring_k3", "topology": {"type": "ring", "k": 3}},
    {"label": "mst", "topology": "mst"},
    {"label": "delaunay", "topology": "delaunay"},
    {"label": "knn_k5", "topology": {"type": "k_nearest", "k": 5}},
    {"label": "knn_k10", "topology": {"type": "k_nearest", "k": 10}},
    {"label": "knn_k15", "topology": {"type": "k_nearest", "k": 15}},
    {"label": "disk_R60", "topology": {"type": "disk", "comm_range": 60.0}},
    {"label": "disk_R120", "topology": {"type": "disk", "comm_range": 120.0}},
    {"label": "er_p0.2", "topology": {"type": "er_random", "p": 0.2}},
    {"label": "er_p0.3", "topology": {"type": "er_random", "p": 0.3}},
    {"label": "er_p0.5", "topology": {"type": "er_random", "p": 0.5}},
]


# ---------------------------------------------------------------------- #
# Threshold helpers (reused from the networked eval)
# ---------------------------------------------------------------------- #

def _round_thresholds(thresholds: dict, pos_mul: float, vel_mul: float) -> dict:
    if not thresholds:
        return {}
    return {
        "std_pos_converged": float(thresholds["std_pos_converged"] * pos_mul),
        "std_vel_converged": float(thresholds["std_vel_converged"] * vel_mul),
    }


def load_threshold_db(path: Optional[str]) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for label, info in data.get("topologies", {}).items():
        prop = info.get("proposed_thresholds", {})
        if prop:
            out[label] = prop
    return out


# ---------------------------------------------------------------------- #
# Per-agent (decentralized) action
# ---------------------------------------------------------------------- #

def decentralized_action(env: NetworkedLazyAgents, policy) -> np.ndarray:
    """Compute one joint action using per-agent local observations."""

    per_agent = env.build_per_agent_central_obs()
    live_idx = per_agent["live_indices"]
    N_live = len(live_idx)
    N_max = env.num_agents_max

    if N_live == 0:
        return np.zeros(N_max, dtype=np.float32)

    obs_batch = {
        "agent_embeddings": per_agent["agent_embeddings"],
        "pad_tokens": per_agent["pad_tokens"],
    }
    actions_batch, _, _ = policy.compute_actions(obs_batch, explore=False)

    action = np.zeros(N_max, dtype=np.float32)
    for b, i in enumerate(live_idx):
        action[i] = actions_batch[b, i]
    return np.clip(action, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# Episode runner
# ---------------------------------------------------------------------- #

def run_episode(
    env: NetworkedLazyAgents,
    max_steps: int,
    *,
    mode: str,
    policy=None,
    alignment_threshold: float = 0.95,
) -> dict:
    """Run one episode.

    Parameters
    ----------
    mode : {"acs", "centralized", "decentralized"}
    """

    obs = env.reset()
    reward_sum = 0.0
    length = max_steps
    for t in range(max_steps):
        if mode == "acs":
            action = env.get_fully_active_action()
        elif mode == "centralized":
            action, _, _ = policy.compute_single_action(obs, explore=False)
            action = np.clip(action, 0.0, 1.0)
        elif mode == "decentralized":
            action = decentralized_action(env, policy)
        else:
            raise ValueError(f"unknown mode: {mode}")

        obs, reward, done, info = env.step(action)
        reward_sum += reward
        if done:
            length = t + 1
            break

    return {
        "reward_sum": float(reward_sum),
        "length": int(length),
        "final_std_pos": float(env.std_pos_hist[length - 1]),
        "final_std_vel": float(env.std_vel_hist[length - 1]),
        "num_network_components": int(env.num_network_components()),
        "is_network_connected": bool(env.is_network_connected()),
        "num_aligned_components": int(
            env.num_aligned_components(alignment_threshold=alignment_threshold)
        ),
        "order_parameter": float(env.alignment_order_parameter()),
        "is_aligned_single_component": bool(
            env.is_aligned_single_component(alignment_threshold=alignment_threshold)
        ),
        "num_position_clusters": int(env.num_position_clusters()),
        "is_single_position_cluster": bool(env.is_single_flock()),
    }


def aggregate(episodes: List[dict]) -> dict:
    if not episodes:
        return {}

    def arr(key, dtype=np.float64):
        return np.array([e[key] for e in episodes], dtype=dtype)

    rewards = arr("reward_sum")
    lengths = arr("length")
    pos = arr("final_std_pos")
    vel = arr("final_std_vel")
    n_comp = arr("num_network_components", dtype=int)
    is_conn = arr("is_network_connected").astype(bool)
    n_aligned = arr("num_aligned_components", dtype=int)
    is_aligned = arr("is_aligned_single_component").astype(bool)
    op = arr("order_parameter")
    n_pos = arr("num_position_clusters", dtype=int)
    is_single_pos = arr("is_single_position_cluster").astype(bool)

    return {
        "num_episodes": len(episodes),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
        "single_flock_rate": float(is_conn.mean()),
        "mean_num_network_components": float(n_comp.mean()),
        "max_num_network_components": int(n_comp.max()),
        "aligned_single_component_rate": float(is_aligned.mean()),
        "mean_num_aligned_components": float(n_aligned.mean()),
        "order_parameter_mean": float(op.mean()),
        "order_parameter_std": float(op.std()),
        "single_position_cluster_rate": float(is_single_pos.mean()),
        "mean_num_position_clusters": float(n_pos.mean()),
        "max_num_position_clusters": int(n_pos.max()),
        "final_std_pos_mean": float(pos.mean()),
        "final_std_pos_std": float(pos.std()),
        "final_std_vel_mean": float(vel.mean()),
        "final_std_vel_std": float(vel.std()),
    }


# ---------------------------------------------------------------------- #
# Topology evaluator
# ---------------------------------------------------------------------- #

def make_env_config(num_agents, max_steps, topology_spec, thresholds):
    return {
        "num_agents_max": num_agents,
        "num_agents_min": num_agents,
        "speed": 15,
        "predefined_distance": 60,
        "std_pos_rate_converged": 0.1,
        "std_vel_rate_converged": 0.2,
        "std_pos_converged": float(thresholds.get("std_pos_converged", 45.0)),
        "std_vel_converged": float(thresholds.get("std_vel_converged", 0.1)),
        "max_time_step": max_steps,
        "incomplete_episode_penalty": 0,
        "normalize_obs": True,
        "use_fixed_horizon": True,
        "use_L2_norm": False,
        "use_preprocessed_obs": True,
        "obs_mode": "central",
        "network_topology": topology_spec,
    }


def evaluate_topology(
    *,
    policy,
    topology_spec,
    thresholds: dict,
    num_episodes: int,
    num_agents: int,
    max_steps: int,
    base_seed: int,
    alignment_threshold: float,
) -> dict:
    env_config = make_env_config(num_agents, max_steps, topology_spec, thresholds)

    dec_episodes: List[dict] = []
    cen_episodes: List[dict] = []
    acs_episodes: List[dict] = []

    for ep in range(num_episodes):
        seed = base_seed + ep

        env_dec = NetworkedLazyAgents(env_config)
        env_dec.seed(seed)
        dec_episodes.append(
            run_episode(env_dec, max_steps, mode="decentralized",
                        policy=policy, alignment_threshold=alignment_threshold)
        )

        env_cen = NetworkedLazyAgents(env_config)
        env_cen.seed(seed)
        cen_episodes.append(
            run_episode(env_cen, max_steps, mode="centralized",
                        policy=policy, alignment_threshold=alignment_threshold)
        )

        env_acs = NetworkedLazyAgents(env_config)
        env_acs.seed(seed)
        acs_episodes.append(
            run_episode(env_acs, max_steps, mode="acs",
                        alignment_threshold=alignment_threshold)
        )

    return {
        "topology_spec": topology_spec,
        "thresholds": thresholds,
        "decentralized": aggregate(dec_episodes),
        "centralized": aggregate(cen_episodes),
        "acs": aggregate(acs_episodes),
        "decentralized_episodes": dec_episodes,
        "centralized_episodes": cen_episodes,
        "acs_episodes": acs_episodes,
    }


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--base_seed", type=int, default=2000)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--acs_sweep", type=str,
        default=os.path.join(THIS_DIR, "results", "acs_topology_sweep.json"),
    )
    parser.add_argument("--pos_offset_mul", type=float, default=1.10)
    parser.add_argument("--vel_offset_mul", type=float, default=1.50)
    parser.add_argument("--alignment_threshold", type=float, default=0.95)
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(THIS_DIR, "results", "checkpoint_eval_decentralized.json"),
    )
    parser.add_argument(
        "--topologies", nargs="*", default=None,
        help="optional list of labels to restrict the run to",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    from ray.rllib.policy.policy import Policy
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env
    from models.lazy_allocator import MyRLlibTorchWrapper

    ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
    register_env("networked_lazy_env", lambda cfg: NetworkedLazyAgents(cfg))

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    policy = Policy.from_checkpoint(args.checkpoint)
    policy.model.eval()
    print("Checkpoint loaded.")

    topology_thresholds = load_threshold_db(args.acs_sweep)
    print(f"Loaded thresholds for {len(topology_thresholds)} topologies")

    if args.topologies:
        selected = [e for e in DEFAULT_TOPOLOGIES if e["label"] in args.topologies]
        if not selected:
            raise ValueError(f"No matching topologies for {args.topologies}")
    else:
        selected = list(DEFAULT_TOPOLOGIES)

    print(
        f"\nEvaluating on {len(selected)} topologies, "
        f"{args.num_episodes} episodes each, max_steps={args.max_steps}"
    )

    hdr = (
        f"{'topology':>16s} |"
        f" {'dec R':>8s} {'cen R':>8s} {'acs R':>8s}"
        f" | {'d-a':>6s} {'c-a':>6s}"
        f" | {'d 1f%':>5s} {'c 1f%':>5s} {'a 1f%':>5s}"
        f" | {'d op':>5s} {'c op':>5s} {'a op':>5s}"
        f" | {'d pos':>6s} {'c pos':>6s} {'a pos':>6s}"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(
        "(dec=decentralized, cen=centralized, a=ACS; "
        "d-a/c-a=reward gap vs ACS; 1f%=single-flock; op=order param; pos=final std_pos)"
    )
    print(sep)

    results = {}
    t0 = time.time()
    for entry in selected:
        label = entry["label"]
        topo = entry["topology"]
        base_thresh = topology_thresholds.get(label, {})
        thresh = _round_thresholds(base_thresh, args.pos_offset_mul, args.vel_offset_mul)
        if not thresh:
            thresh = {"std_pos_converged": 45.0, "std_vel_converged": 0.1}

        t_topo = time.time()
        out = evaluate_topology(
            policy=policy,
            topology_spec=topo,
            thresholds=thresh,
            num_episodes=args.num_episodes,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            base_seed=args.base_seed,
            alignment_threshold=args.alignment_threshold,
        )
        t_topo = time.time() - t_topo

        d = out["decentralized"]
        c = out["centralized"]
        a = out["acs"]
        d_gap = d["reward_mean"] - a["reward_mean"]
        c_gap = c["reward_mean"] - a["reward_mean"]

        print(
            f"{label:>16s} |"
            f" {d['reward_mean']:8.1f} {c['reward_mean']:8.1f} {a['reward_mean']:8.1f}"
            f" | {d_gap:+6.0f} {c_gap:+6.0f}"
            f" | {100*d['single_flock_rate']:4.0f}% {100*c['single_flock_rate']:4.0f}% {100*a['single_flock_rate']:4.0f}%"
            f" | {d['order_parameter_mean']:5.3f} {c['order_parameter_mean']:5.3f} {a['order_parameter_mean']:5.3f}"
            f" | {d['final_std_pos_mean']:6.1f} {c['final_std_pos_mean']:6.1f} {a['final_std_pos_mean']:6.1f}"
            f"  ({t_topo:.0f}s)"
        )
        results[label] = out

    elapsed = time.time() - t0
    print(sep)
    print(f"Total elapsed: {elapsed:.1f}s")

    out_doc = {"args": vars(args), "topologies": results}
    with open(args.output, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
