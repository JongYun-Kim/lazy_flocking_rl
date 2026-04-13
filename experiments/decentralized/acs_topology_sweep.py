"""
ACS-only sweep across communication topologies.

This script runs the Active Cohesion-Separation (ACS) control law -- i.e. the
fully-active baseline policy implemented inside the env's ``get_u`` -- on the
new :class:`NetworkedLazyAgents` env, varying the network topology.

For each topology candidate it asks one *primary* question and a few
*auxiliary* questions:

  1. **Primary -- single-flock rate.** Does the swarm end the episode as a
     single connected network component? "Single flock" here means *exactly*
     "all live agents are reachable from each other in the communication
     topology graph, in any number of hops". No alignment requirement.
     Static topologies such as ring, star, fully_connected, line, wheel,
     binary_tree, mst, delaunay are *trivially* a single flock by this
     definition. Dynamic ones (k_nearest, disk) and probabilistic ones (ER
     random) may or may not satisfy it on a given episode.

  2. **Auxiliary -- aligned-single-component rate.** Same as the primary, but
     additionally requires that the polar order parameter
     ``|mean(v_i)| / speed`` exceeds ``alignment_threshold``. This catches
     ring-like topologies that are 1 component but rotate rather than fly
     coherently.

  3. **Auxiliary -- spatial cluster count.** As a sanity check, the script
     also reports how many spatial clusters there are (agents within
     ``cluster_threshold`` form an undirected graph; clusters = connected
     components of that graph).

For trials that converge to a single network flock the final spatial entropy
``std_pos`` and velocity entropy ``std_vel`` are recorded; the proposed
``std_pos_converged`` / ``std_vel_converged`` are derived from this
distribution and plugged into the downstream ``eval_checkpoint_networked.py``.

Usage
-----
    python -m experiments.decentralized.acs_topology_sweep \\
        --num_episodes 30 --num_agents 20 --max_steps 1500 \\
        --output experiments/decentralized/results/acs_topology_sweep.json

The script is intentionally CPU-only and serial; with 20 agents and 30
episodes per topology it finishes in a few minutes on a modern machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

# Make sure the workspace root is on sys.path when run as a script.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from experiments.decentralized.env_networked import NetworkedLazyAgents  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------- #
# Topologies to sweep
# ---------------------------------------------------------------------- #
DEFAULT_TOPOLOGIES = [
    # Static, always-connected by construction.
    {"label": "fully_connected", "topology": "fully_connected"},
    {"label": "star", "topology": "star"},
    {"label": "wheel", "topology": "wheel"},
    {"label": "binary_tree", "topology": "binary_tree"},
    {"label": "line", "topology": "line"},
    {"label": "ring_k1", "topology": {"type": "ring", "k": 1}},
    {"label": "ring_k2", "topology": {"type": "ring", "k": 2}},
    {"label": "ring_k3", "topology": {"type": "ring", "k": 3}},

    # Dynamic, always-connected by construction.
    {"label": "mst", "topology": "mst"},
    {"label": "delaunay", "topology": "delaunay"},

    # Dynamic, may split. (k as requested by user.)
    {"label": "knn_k5", "topology": {"type": "k_nearest", "k": 5}},
    {"label": "knn_k10", "topology": {"type": "k_nearest", "k": 10}},
    {"label": "knn_k15", "topology": {"type": "k_nearest", "k": 15}},

    # Disk graphs (kept for the "becomes effectively FC at convergence" check).
    {"label": "disk_R60", "topology": {"type": "disk", "comm_range": 60.0}},
    {"label": "disk_R120", "topology": {"type": "disk", "comm_range": 120.0}},

    # Probabilistic (always included as exception even if not always 1 component).
    {"label": "er_p0.2", "topology": {"type": "er_random", "p": 0.2}},
    {"label": "er_p0.3", "topology": {"type": "er_random", "p": 0.3}},
    {"label": "er_p0.5", "topology": {"type": "er_random", "p": 0.5}},
]


# Plain-language descriptions, surfaced in the report.
TOPOLOGY_DESCRIPTIONS = {
    "fully_connected": (
        "Every pair of agents is connected. Each agent's ACS step averages "
        "over the entire swarm. This is the topology the existing checkpoint "
        "was trained on. *Static, always 1 component.*"
    ),
    "star": (
        "One designated *center* agent (the lowest-index live agent) is "
        "connected to every other agent; the leaves only see the center. "
        "*Static, always 1 component.*"
    ),
    "wheel": (
        "Hub + outer ring hybrid. Index 0 is the hub, connected to every "
        "other agent; the outer agents (1..n-1) form a ring among themselves. "
        "Combines star (centralized averaging) with ring (local index "
        "structure). *Static, always 1 component.*"
    ),
    "binary_tree": (
        "Heap-style binary tree: parent of node i (for i ≥ 1) is (i-1)//2, "
        "children are 2i+1 and 2i+2. Hierarchical structure with depth ~log₂ n. "
        "*Static, always 1 component.*"
    ),
    "line": (
        "Path graph: ring_k1 minus the wraparound edge. Each agent connects "
        "only to its index-adjacent neighbors. The two endpoints have degree 1. "
        "*Static, always 1 component.*"
    ),
    "ring_k1": (
        "1-D circular ring of width 1: each agent connects to its left and "
        "right index neighbor (degree 2 + self). *Static, always 1 component.*"
    ),
    "ring_k2": (
        "Ring of width 2: each agent connects to its 2 nearest left and 2 "
        "nearest right index neighbors (degree 4 + self). *Static, always 1 component.*"
    ),
    "ring_k3": (
        "Ring of width 3: each agent connects to its 3 nearest left and 3 "
        "nearest right index neighbors (degree 6 + self). *Static, always 1 component.*"
    ),
    "mst": (
        "Minimum spanning tree of the live agents' pairwise-distance graph, "
        "rebuilt every step. Has exactly n−1 edges (the sparsest possible "
        "connected graph). *Dynamic, always 1 component.*"
    ),
    "delaunay": (
        "Delaunay triangulation of the live agents' positions, rebuilt every "
        "step. Planar, ≤ 3n−6 edges, captures local spatial neighbors. "
        "*Dynamic, always 1 component (for non-degenerate point sets).*"
    ),
    "knn_k5": (
        "Each agent connects to its 5 spatially nearest other agents; the "
        "graph is symmetrized (edge if either side picked the other). "
        "*Dynamic, may split into multiple components.*"
    ),
    "knn_k10": "Same as `knn_k5` but k=10. *Dynamic.*",
    "knn_k15": "Same as `knn_k5` but k=15. *Dynamic.*",
    "disk_R60": (
        "Disk graph: two agents are connected iff their pairwise distance is "
        "≤ 60 m (= predefined ACS distance R). *Dynamic.* Note that R=60 < "
        "the converged-flock diameter (~2R = 120), so this remains sparse "
        "even after convergence -- but it usually fails to bond at "
        "initialisation."
    ),
    "disk_R120": (
        "Disk graph with communication range 120 m (= 2 * R). *Dynamic.* "
        "**Caveat: at convergence the swarm sits in a disk of diameter ≈ 2R, "
        "so all pairs are within range and this becomes effectively fully "
        "connected.** Kept here as a baseline for that observation."
    ),
    "er_p0.2": (
        "Erdős–Rényi random graph with edge probability 0.2, resampled per "
        "episode (kept fixed within an episode). Symmetric. *Probabilistic.*"
    ),
    "er_p0.3": "Erdős–Rényi random graph with edge probability 0.3. *Probabilistic.*",
    "er_p0.5": "Erdős–Rényi random graph with edge probability 0.5. *Probabilistic.*",
}

# Topology labels treated as "probabilistic exceptions": always included in
# the eval default set, regardless of their measured single-flock rate.
PROBABILISTIC_TOPOLOGIES = ("er_p0.2", "er_p0.3", "er_p0.5")


# ---------------------------------------------------------------------- #
# Episode runner
# ---------------------------------------------------------------------- #

def run_acs_episode(
    env: NetworkedLazyAgents,
    max_steps: int,
    cluster_threshold: float,
    alignment_threshold: float,
) -> dict:
    """Run one ACS-only episode and return episode statistics.

    Records both the *primary* metric (network is one connected component, no
    alignment requirement) and the *auxiliary* metrics (alignment-augmented
    flock count, polar order parameter, spatial cluster count).
    """

    obs = env.reset()
    reward_sum = 0.0
    length = max_steps
    for t in range(max_steps):
        action = env.get_fully_active_action()
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        if done:
            length = t + 1
            break

    final_std_pos = float(env.std_pos_hist[length - 1])
    final_std_vel = float(env.std_vel_hist[length - 1])

    # Primary: network connectivity only.
    n_components = env.num_network_components()
    is_network_connected = env.is_network_connected()

    # Auxiliary: alignment-augmented.
    n_aligned_components = env.num_aligned_components(alignment_threshold=alignment_threshold)
    overall_order_param = env.alignment_order_parameter()
    is_aligned_single_component = env.is_aligned_single_component(
        alignment_threshold=alignment_threshold
    )

    # Auxiliary: spatial clustering.
    is_single_pos = env.is_single_flock(distance_threshold=cluster_threshold)
    n_pos_clusters = env.num_position_clusters(distance_threshold=cluster_threshold)

    return {
        "reward_sum": reward_sum,
        "length": length,
        "final_std_pos": final_std_pos,
        "final_std_vel": final_std_vel,
        # primary -- connectivity only
        "num_network_components": int(n_components),
        "is_network_connected": bool(is_network_connected),
        # auxiliary -- alignment-augmented
        "num_aligned_components": int(n_aligned_components),
        "order_parameter": float(overall_order_param),
        "is_aligned_single_component": bool(is_aligned_single_component),
        # auxiliary -- spatial
        "num_position_clusters": int(n_pos_clusters),
        "is_single_position_cluster": bool(is_single_pos),
    }


def evaluate_topology(
    topology_spec,
    num_episodes: int,
    num_agents: int,
    max_steps: int,
    base_seed: int,
    cluster_threshold: float,
    alignment_threshold: float,
) -> dict:
    """Run ``num_episodes`` ACS rollouts on a topology and aggregate stats."""

    base_env_config = {
        "num_agents_max": num_agents,
        "num_agents_min": num_agents,
        "speed": 15,
        "predefined_distance": 60,
        # Use generous convergence thresholds so the dynamics actually have a
        # chance to terminate naturally; we'll re-tighten in step 4.
        "std_pos_converged": 1e9,
        "std_vel_converged": 1e9,
        "std_pos_rate_converged": 1e9,
        "std_vel_rate_converged": 1e9,
        "max_time_step": max_steps,
        "incomplete_episode_penalty": 0,
        "normalize_obs": False,
        "use_fixed_horizon": True,  # always run the full horizon
        "use_L2_norm": False,
        "use_preprocessed_obs": True,
        "obs_mode": "central",
        "network_topology": topology_spec,
    }

    episodes = []
    for ep in range(num_episodes):
        env = NetworkedLazyAgents(base_env_config)
        env.seed(base_seed + ep)
        ep_stats = run_acs_episode(
            env,
            max_steps,
            cluster_threshold=cluster_threshold,
            alignment_threshold=alignment_threshold,
        )
        episodes.append(ep_stats)

    arr = lambda key: np.array([e[key] for e in episodes], dtype=np.float64)
    n_total = len(episodes)

    def _stats(values):
        if values.size == 0:
            return {k: float("nan") for k in ("mean", "std", "min", "max")}
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    # Primary: network connectivity only.
    is_connected = arr("is_network_connected").astype(bool)
    n_connected = int(is_connected.sum())
    single_flock_rate = n_connected / n_total if n_total > 0 else 0.0

    n_components_arr = arr("num_network_components").astype(int)

    # Distribution of final stats over the episodes that ended with the
    # network being one connected component (= ended as a single flock).
    sp_connected = arr("final_std_pos")[is_connected]
    sv_connected = arr("final_std_vel")[is_connected]
    op_connected = arr("order_parameter")[is_connected]

    # Auxiliary: alignment-augmented.
    is_aligned_single = arr("is_aligned_single_component").astype(bool)
    aligned_single_component_rate = (
        float(is_aligned_single.mean()) if n_total > 0 else 0.0
    )
    n_aligned_components_arr = arr("num_aligned_components").astype(int)

    # Auxiliary: spatial clustering.
    is_single_pos = arr("is_single_position_cluster").astype(bool)
    n_single_pos = int(is_single_pos.sum())
    single_position_cluster_rate = n_single_pos / n_total if n_total > 0 else 0.0
    pos_cluster_arr = arr("num_position_clusters").astype(int)

    return {
        "topology_spec": topology_spec,
        "num_episodes": n_total,

        # primary -- connectivity-only
        "single_flock_rate": single_flock_rate,
        "num_single_flock": n_connected,
        "mean_num_network_components": float(n_components_arr.mean()),
        "max_num_network_components": int(n_components_arr.max()),
        "single_flock_pos": _stats(sp_connected),
        "single_flock_vel": _stats(sv_connected),
        "single_flock_order_parameter": _stats(op_connected),

        # auxiliary -- alignment
        "aligned_single_component_rate": aligned_single_component_rate,
        "mean_num_aligned_components": float(n_aligned_components_arr.mean()),
        "max_num_aligned_components": int(n_aligned_components_arr.max()),
        "order_parameter_overall": _stats(arr("order_parameter")),

        # auxiliary -- spatial
        "single_position_cluster_rate": single_position_cluster_rate,
        "num_single_position_cluster": n_single_pos,
        "mean_num_position_clusters": float(pos_cluster_arr.mean()),
        "max_num_position_clusters": int(pos_cluster_arr.max()),

        "all_episodes": episodes,
    }


# ---------------------------------------------------------------------- #
# Convergence threshold proposal
# ---------------------------------------------------------------------- #

def propose_convergence_thresholds(
    summary: dict, pos_offset: float = 1.5, vel_offset: float = 2.0, min_pos: float = 5.0
) -> dict:
    """Build relaxed ``std_pos_converged`` / ``std_vel_converged`` from a single
    topology summary.

    Uses the *single-flock* distribution: the runs whose final adjacency was
    one connected component (no alignment requirement). Empty if no single
    flock was observed.
    """

    if summary.get("num_single_flock", 0) == 0:
        return {}

    pos_mean = summary["single_flock_pos"]["mean"]
    pos_std = max(summary["single_flock_pos"]["std"], 1e-3)
    vel_mean = summary["single_flock_vel"]["mean"]
    vel_std = max(summary["single_flock_vel"]["std"], 1e-3)

    return {
        "std_pos_converged": float(max(pos_mean + pos_offset * pos_std, min_pos)),
        "std_vel_converged": float(max(vel_mean + vel_offset * vel_std, 0.05)),
    }


# ---------------------------------------------------------------------- #
# CLI entry-point
# ---------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--base_seed", type=int, default=1000)
    parser.add_argument(
        "--single_flock_rate_threshold",
        type=float,
        default=0.9,
        help="topologies with single-flock rate >= this are flagged as 'reliable' "
        "(probabilistic topologies are *also* always flagged as exceptions)",
    )
    parser.add_argument(
        "--cluster_threshold",
        type=float,
        default=120.0,
        help="distance under which two agents are in the same spatial cluster "
        "(auxiliary metric, default 2*R = 120m)",
    )
    parser.add_argument(
        "--alignment_threshold",
        type=float,
        default=0.95,
        help="polar order parameter threshold for the auxiliary "
        "alignment-augmented metric (default 0.95)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(THIS_DIR, "results", "acs_topology_sweep.json"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(
        f"ACS topology sweep | episodes={args.num_episodes}, agents={args.num_agents}, "
        f"steps={args.max_steps}, base_seed={args.base_seed}"
    )
    print("-" * 80)

    results = {}
    t0 = time.time()
    for entry in DEFAULT_TOPOLOGIES:
        label = entry["label"]
        topo = entry["topology"]
        t_topo_start = time.time()
        summary = evaluate_topology(
            topo,
            num_episodes=args.num_episodes,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            base_seed=args.base_seed,
            cluster_threshold=args.cluster_threshold,
            alignment_threshold=args.alignment_threshold,
        )
        thresholds = propose_convergence_thresholds(summary)
        elapsed = time.time() - t_topo_start
        ncomp = summary["mean_num_network_components"]
        nalig = summary["mean_num_aligned_components"]
        npos = summary["mean_num_position_clusters"]
        sf_pos = summary["single_flock_pos"]
        sf_vel = summary["single_flock_vel"]
        op = summary["order_parameter_overall"]
        print(
            f"{label:>16s} |"
            f" 1-flock={summary['single_flock_rate']:.2f}"
            f" ({summary['num_single_flock']}/{summary['num_episodes']})"
            f" | aligned-1c={summary['aligned_single_component_rate']:.2f}"
            f" | mean #comp={ncomp:4.2f}"
            f" #aligned={nalig:4.2f}"
            f" #posK={npos:5.2f}"
            f" | order={op['mean']:.3f}"
            f" | conv pos {sf_pos['mean']:6.2f}±{sf_pos['std']:5.2f}"
            f" vel {sf_vel['mean']:6.3f}±{sf_vel['std']:6.3f}"
            f" | {elapsed:5.1f}s"
        )
        results[label] = {
            "summary": summary,
            "proposed_thresholds": thresholds,
        }

    elapsed = time.time() - t0
    print("-" * 80)
    print(f"Total elapsed: {elapsed:.1f}s")

    # Identify topologies that reliably remain a single network component,
    # plus the probabilistic exceptions.
    reliable = []
    for label, info in results.items():
        rate = info["summary"]["single_flock_rate"]
        if rate >= args.single_flock_rate_threshold or label in PROBABILISTIC_TOPOLOGIES:
            reliable.append(label)
    print(
        f"\nReliable single-flock topologies "
        f"(rate >= {args.single_flock_rate_threshold} or probabilistic exception):"
    )
    for label in reliable:
        rate = results[label]["summary"]["single_flock_rate"]
        tag = "exception" if (
            label in PROBABILISTIC_TOPOLOGIES and rate < args.single_flock_rate_threshold
        ) else "reliable"
        print(f"  - {label} ({tag}): thresholds = {results[label]['proposed_thresholds']}")

    out_doc = {
        "args": vars(args),
        "topologies": results,
        "reliable_single_flock": reliable,
        "probabilistic_topologies": list(PROBABILISTIC_TOPOLOGIES),
        "descriptions": TOPOLOGY_DESCRIPTIONS,
    }
    with open(args.output, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
