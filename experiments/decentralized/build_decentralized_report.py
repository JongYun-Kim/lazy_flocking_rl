"""
Post-processes eval_checkpoint_decentralized.py JSON into a markdown report.

The report compares three inference modes per topology:
  - decentralized (per-agent local observation)
  - centralized   (global observation, existing eval)
  - ACS baseline  (no learned policy)

Usage
-----
    python -m experiments.decentralized.build_decentralized_report \
        --acs experiments/decentralized/results/acs_topology_sweep.json \
        --eval experiments/decentralized/results/checkpoint_eval_decentralized.json \
        --output experiments/decentralized/results/REPORT_decentralized.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _fmt(value, fmt: str) -> str:
    if value is None:
        return "-"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "-"
        return format(value, fmt)
    except (TypeError, ValueError):
        return "-"


def _get(d: dict, key: str, default=float("nan")):
    return d.get(key, default)


def write_header(lines: list) -> None:
    lines.append("# Decentralized checkpoint evaluation report")
    lines.append("")
    lines.append(
        "This report compares the centralized checkpoint evaluated in three modes:\n"
        "- **Decentralized**: each agent observes only its topology neighbors "
        "(per-agent local frame, batched forward pass).\n"
        "- **Centralized**: all agents observe the full swarm (global frame, "
        "single forward pass) — the prior evaluation method.\n"
        "- **ACS**: fully-active ACS baseline (no learned policy).\n"
    )
    lines.append(
        "The FC topology is a sanity check: decentralized and centralized must "
        "be identical because every agent sees every other agent."
    )
    lines.append("")


TOPOLOGY_DESCRIPTIONS = {
    "fully_connected": "Every agent connects to every other. N(N-1)/2 edges. Baseline topology used for training checkpoint 74.",
    "star": "Hub (index 0) connects to all others; leaves connect only to hub. 19 edges for 20 agents.",
    "wheel": "Hub connects to all outer agents + outer agents form a ring. Each outer agent sees hub + 2 ring neighbors = 3 connections.",
    "binary_tree": "Heap-style binary tree: parent of i (i>=1) is (i-1)//2. Root sees 2 children; internal nodes see parent + 2 children; leaves see parent only.",
    "line": "Path graph 0-1-2-...-19. Ends see 1 neighbor; middle agents see 2. Ring without wraparound.",
    "ring_k1": "Circular ring k=1: each agent connects to 1 neighbor on each side. 2 connections per agent.",
    "ring_k2": "Ring k=2: each agent connects to 2 neighbors on each side. 4 connections per agent.",
    "ring_k3": "Ring k=3: each agent connects to 3 neighbors on each side. 6 connections per agent.",
    "mst": "Minimum spanning tree of pairwise distances. Dynamic (rebuilt each step). Always connected, N-1 edges, avg degree ~2.",
    "delaunay": "Delaunay triangulation of positions. Dynamic. Always connected, typically ~3N edges, avg degree ~6.",
    "knn_k5": "k-nearest neighbors (k=5), symmetrized. Dynamic. May disconnect if clusters separate.",
    "knn_k10": "k-nearest neighbors (k=10), symmetrized. Dynamic. Dense enough to stay connected at convergence.",
    "knn_k15": "k-nearest neighbors (k=15), symmetrized. Dynamic. Nearly FC (15/19 possible neighbors).",
    "disk_R60": "Agents within 60m connected. Dynamic. Too sparse — swarm disperses.",
    "disk_R120": "Agents within 120m connected. Dynamic. Effectively FC at ACS convergence but policy-induced spread breaks it.",
    "er_p0.2": "Erdos-Renyi, edge prob 0.2. Static per episode. Expected degree 3.8. Connected ~70%.",
    "er_p0.3": "Erdos-Renyi, edge prob 0.3. Static per episode. Expected degree 5.7. Connected ~97%.",
    "er_p0.5": "Erdos-Renyi, edge prob 0.5. Static per episode. Expected degree 9.5. Connected ~100%.",
}


def write_topology_descriptions(lines: list) -> None:
    lines.append("## Network topology descriptions")
    lines.append("")
    lines.append("| topology | description |")
    lines.append("|---|---|")
    for label, desc in TOPOLOGY_DESCRIPTIONS.items():
        lines.append(f"| `{label}` | {desc} |")
    lines.append("")


def write_metric_defs(lines: list) -> None:
    lines.append("## Metric definitions")
    lines.append("")
    lines.append(
        "- **reward**: cumulative episode reward over the fixed horizon.\n"
        "- **1f%** (single flock rate): fraction of episodes where all agents "
        "form one connected component at the last step.\n"
        "- **op** (order parameter): `|mean(v_i)| / speed` — heading alignment "
        "(0 = random, 1 = perfectly aligned).\n"
        "- **pos** (final std_pos): `sqrt(Var(x) + Var(y))` at last step — "
        "spatial tightness.\n"
        "- **d-a**: reward gap between decentralized policy and ACS.\n"
        "- **c-a**: reward gap between centralized policy and ACS.\n"
        "- **d-c**: reward gap between decentralized and centralized — "
        "quantifies the cost of limiting information to local neighbors."
    )
    lines.append("")


def write_main_table(eval_data: dict, lines: list) -> None:
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| topology | dec R | cen R | acs R | d-a | c-a | d-c "
        "| dec 1f% | cen 1f% | acs 1f% "
        "| dec op | cen op | acs op "
        "| dec pos | cen pos | acs pos |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:"
        "|---:|---:|---:"
        "|---:|---:|---:"
        "|---:|---:|---:|"
    )
    for label, t in eval_data["topologies"].items():
        d = t["decentralized"]
        c = t["centralized"]
        a = t["acs"]
        d_a = d["reward_mean"] - a["reward_mean"]
        c_a = c["reward_mean"] - a["reward_mean"]
        d_c = d["reward_mean"] - c["reward_mean"]
        lines.append(
            f"| `{label}` "
            f"| {d['reward_mean']:.1f} "
            f"| {c['reward_mean']:.1f} "
            f"| {a['reward_mean']:.1f} "
            f"| {d_a:+.0f} "
            f"| {c_a:+.0f} "
            f"| {d_c:+.0f} "
            f"| {100*d['single_flock_rate']:.0f}% "
            f"| {100*c['single_flock_rate']:.0f}% "
            f"| {100*a['single_flock_rate']:.0f}% "
            f"| {d['order_parameter_mean']:.3f} "
            f"| {c['order_parameter_mean']:.3f} "
            f"| {a['order_parameter_mean']:.3f} "
            f"| {d['final_std_pos_mean']:.1f} "
            f"| {c['final_std_pos_mean']:.1f} "
            f"| {a['final_std_pos_mean']:.1f} |"
        )
    lines.append("")


def write_info_loss_table(eval_data: dict, lines: list) -> None:
    lines.append("## Information loss analysis (decentralized vs centralized)")
    lines.append("")
    lines.append(
        "This table isolates the effect of restricting each agent's observation "
        "to its local neighbors. Negative d-c means local info hurts; zero means "
        "no difference (FC or very dense topologies)."
    )
    lines.append("")
    lines.append(
        "| topology | d-c reward | dec op | cen op | op drop "
        "| dec pos | cen pos | pos increase |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, t in eval_data["topologies"].items():
        d = t["decentralized"]
        c = t["centralized"]
        d_c = d["reward_mean"] - c["reward_mean"]
        op_drop = d["order_parameter_mean"] - c["order_parameter_mean"]
        pos_inc = d["final_std_pos_mean"] - c["final_std_pos_mean"]
        lines.append(
            f"| `{label}` "
            f"| {d_c:+.1f} "
            f"| {d['order_parameter_mean']:.3f} "
            f"| {c['order_parameter_mean']:.3f} "
            f"| {op_drop:+.3f} "
            f"| {d['final_std_pos_mean']:.1f} "
            f"| {c['final_std_pos_mean']:.1f} "
            f"| {pos_inc:+.1f} |"
        )
    lines.append("")


def write_takeaways(eval_data: dict, lines: list) -> None:
    lines.append("## Takeaways")
    lines.append("")

    dec_beats_acs = []
    dec_loses_acs = []
    big_info_loss = []

    for label, t in eval_data["topologies"].items():
        d = t["decentralized"]
        c = t["centralized"]
        a = t["acs"]
        d_a = d["reward_mean"] - a["reward_mean"]
        d_c = d["reward_mean"] - c["reward_mean"]

        if d_a > 10 and d["single_flock_rate"] >= a["single_flock_rate"] - 0.05:
            dec_beats_acs.append((label, d_a, d_c, d, c, a))
        elif d_a < -10 or d["single_flock_rate"] < a["single_flock_rate"] - 0.10:
            dec_loses_acs.append((label, d_a, d_c, d, c, a))

        if d_c < -20:
            big_info_loss.append((label, d_c, d, c))

    if dec_beats_acs:
        lines.append(
            "**Decentralized policy beats ACS** (reward gap > +10, comparable flock rate):"
        )
        for label, d_a, d_c, d, c, a in dec_beats_acs:
            lines.append(
                f"- `{label}`: dec-acs **{d_a:+.0f}**, "
                f"info loss (dec-cen) **{d_c:+.0f}**, "
                f"dec op **{d['order_parameter_mean']:.3f}**"
            )
        lines.append("")

    if dec_loses_acs:
        lines.append(
            "**Decentralized policy fails vs ACS** (worse reward or flock breakup):"
        )
        for label, d_a, d_c, d, c, a in dec_loses_acs:
            lines.append(
                f"- `{label}`: dec-acs **{d_a:+.0f}**, "
                f"dec 1f% **{100*d['single_flock_rate']:.0f}%** "
                f"vs acs **{100*a['single_flock_rate']:.0f}%**"
            )
        lines.append("")

    if big_info_loss:
        lines.append(
            "**Significant information loss** (dec-cen reward gap < -20):"
        )
        for label, d_c, d, c in big_info_loss:
            lines.append(
                f"- `{label}`: dec-cen **{d_c:+.0f}**, "
                f"dec op **{d['order_parameter_mean']:.3f}** "
                f"vs cen op **{c['order_parameter_mean']:.3f}**"
            )
        lines.append("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval", type=str,
        default=os.path.join(THIS_DIR, "results", "checkpoint_eval_decentralized.json"),
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(THIS_DIR, "results", "REPORT_decentralized.md"),
    )
    args = parser.parse_args()

    if not os.path.exists(args.eval):
        raise FileNotFoundError(f"Eval file not found: {args.eval}")

    with open(args.eval) as f:
        eval_data = json.load(f)

    lines: list = []
    write_header(lines)
    write_topology_descriptions(lines)
    write_metric_defs(lines)
    write_main_table(eval_data, lines)
    write_info_loss_table(eval_data, lines)
    write_takeaways(eval_data, lines)

    eval_args = eval_data.get("args", {})
    lines.append("---")
    lines.append(
        f"*Generated from `eval_checkpoint_decentralized.py` — "
        f"{eval_args.get('num_episodes', '?')} episodes, "
        f"max_steps={eval_args.get('max_steps', '?')}, "
        f"num_agents={eval_args.get('num_agents', '?')}*"
    )
    lines.append("")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
