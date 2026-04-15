"""
Post-processes eval_checkpoint_decentralized.py JSON into a markdown report.

The report compares two inference modes per topology:
  - decentralized (per-agent local observation)
  - ACS baseline  (no learned policy)

For every topology we report:
  * convergence rate and convergence-step distribution (per mode),
  * metrics aggregated over all episodes,
  * metrics aggregated separately over converged and non-converged episodes.

Usage
-----
    python -m experiments.decentralized.build_decentralized_report \
        --eval experiments/decentralized/results/checkpoint_eval_decentralized.json \
        --output experiments/decentralized/results/REPORT_decentralized.md
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _fmt_float(value, fmt: str) -> str:
    if value is None:
        return "-"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "-"
        return format(value, fmt)
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value) -> str:
    if value is None:
        return "-"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "-"
        return f"{100 * value:.0f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_int(value) -> str:
    if value is None:
        return "-"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "-"
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def write_header(lines: list, eval_args: dict) -> None:
    lines.append("# Decentralized checkpoint evaluation report")
    lines.append("")
    lines.append(
        "This report compares the centralized-trained checkpoint deployed with\n"
        "*decentralized* per-agent observation against the fully-active ACS\n"
        "baseline on a range of communication topologies."
    )
    lines.append("")
    lines.append(
        f"**Convergence criterion.** An episode is *converged* if there exists "
        f"a trailing window of `{eval_args.get('conv_window', '?')}` steps in "
        f"which (a) connectivity held (single component every step), "
        f"(b) `std_pos` varied by less than "
        f"`{eval_args.get('conv_pos_rate', '?')}` m, and (c) the polar order "
        f"parameter varied by less than `{eval_args.get('conv_op_rate', '?')}`."
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
        "- **R**: mean cumulative episode reward (higher = better).\n"
        "- **d-a**: reward gap between decentralized policy and ACS "
        "(positive = policy beats ACS).\n"
        "- **cv%**: fraction of episodes that meet the convergence criterion.\n"
        "- **p50 / p90** (conv step): median / 90-th percentile of the "
        "first-converged step across converged episodes.\n"
        "- **1f%**: fraction of episodes whose final step is a single network "
        "component (\"single flock\").\n"
        "- **conn%**: mean fraction of steps per episode in which the network "
        "was a single component.\n"
        "- **op**: polar order parameter at last step `|mean(v_i)| / speed` "
        "(0 = random, 1 = aligned).\n"
        "- **pos**: `sqrt(Var(x) + Var(y))` at last step — spatial tightness."
    )
    lines.append("")


def write_convergence_table(eval_data: dict, lines: list) -> None:
    lines.append("## Convergence budget")
    lines.append("")
    lines.append(
        "| topology | dec cv% | acs cv% "
        "| dec p50 | dec p90 | dec max "
        "| acs p50 | acs p90 | acs max |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, t in eval_data["topologies"].items():
        d = t["decentralized"]
        a = t["acs"]
        lines.append(
            f"| `{label}` "
            f"| {_fmt_pct(d['converged_rate'])} "
            f"| {_fmt_pct(a['converged_rate'])} "
            f"| {_fmt_int(d.get('conv_step_p50'))} "
            f"| {_fmt_int(d.get('conv_step_p90'))} "
            f"| {_fmt_int(d.get('conv_step_max'))} "
            f"| {_fmt_int(a.get('conv_step_p50'))} "
            f"| {_fmt_int(a.get('conv_step_p90'))} "
            f"| {_fmt_int(a.get('conv_step_max'))} |"
        )
    lines.append("")


def write_overall_table(eval_data: dict, lines: list) -> None:
    lines.append("## Overall results (all episodes)")
    lines.append("")
    lines.append(
        "| topology | dec R | acs R | d-a "
        "| dec 1f% | acs 1f% "
        "| dec conn% | acs conn% "
        "| dec op | acs op "
        "| dec pos | acs pos |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, t in eval_data["topologies"].items():
        d = t["decentralized"]["all"]
        a = t["acs"]["all"]
        d_a = d["reward_mean"] - a["reward_mean"]
        lines.append(
            f"| `{label}` "
            f"| {d['reward_mean']:.1f} "
            f"| {a['reward_mean']:.1f} "
            f"| {d_a:+.0f} "
            f"| {_fmt_pct(d['single_flock_rate'])} "
            f"| {_fmt_pct(a['single_flock_rate'])} "
            f"| {_fmt_pct(d['conn_rate_mean'])} "
            f"| {_fmt_pct(a['conn_rate_mean'])} "
            f"| {d['final_order_parameter_mean']:.3f} "
            f"| {a['final_order_parameter_mean']:.3f} "
            f"| {d['final_std_pos_mean']:.1f} "
            f"| {a['final_std_pos_mean']:.1f} |"
        )
    lines.append("")


def write_split_table(eval_data: dict, lines: list, subset_key: str, title: str) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        "| topology | n dec | n acs | dec R | acs R | d-a "
        "| dec op | acs op | dec pos | acs pos |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, t in eval_data["topologies"].items():
        d = t["decentralized"][subset_key]
        a = t["acs"][subset_key]
        if d.get("count", 0) == 0 and a.get("count", 0) == 0:
            lines.append(
                f"| `{label}` | 0 | 0 | - | - | - | - | - | - | - |"
            )
            continue
        if d.get("count", 0) == 0:
            d_r = d_op = d_pos = float("nan")
        else:
            d_r = d["reward_mean"]
            d_op = d["final_order_parameter_mean"]
            d_pos = d["final_std_pos_mean"]
        if a.get("count", 0) == 0:
            a_r = a_op = a_pos = float("nan")
        else:
            a_r = a["reward_mean"]
            a_op = a["final_order_parameter_mean"]
            a_pos = a["final_std_pos_mean"]
        if not math.isnan(d_r) and not math.isnan(a_r):
            d_a_txt = f"{d_r - a_r:+.0f}"
        else:
            d_a_txt = "-"
        lines.append(
            f"| `{label}` "
            f"| {d.get('count', 0)} "
            f"| {a.get('count', 0)} "
            f"| {_fmt_float(d_r, '.1f')} "
            f"| {_fmt_float(a_r, '.1f')} "
            f"| {d_a_txt} "
            f"| {_fmt_float(d_op, '.3f')} "
            f"| {_fmt_float(a_op, '.3f')} "
            f"| {_fmt_float(d_pos, '.1f')} "
            f"| {_fmt_float(a_pos, '.1f')} |"
        )
    lines.append("")


def write_takeaways(eval_data: dict, lines: list) -> None:
    lines.append("## Takeaways")
    lines.append("")

    dec_beats = []
    dec_loses = []
    mixed_conv = []

    for label, t in eval_data["topologies"].items():
        d = t["decentralized"]
        a = t["acs"]
        d_all = d["all"]
        a_all = a["all"]
        d_a_all = d_all["reward_mean"] - a_all["reward_mean"]

        # Conv-conditioned dec-acs on converged subset
        d_conv = d["converged"]
        a_conv = a["converged"]
        if d_conv.get("count", 0) > 0 and a_conv.get("count", 0) > 0:
            d_a_conv = d_conv["reward_mean"] - a_conv["reward_mean"]
        else:
            d_a_conv = None

        if d_a_all > 10:
            dec_beats.append((label, d_a_all, d_a_conv,
                              d["converged_rate"], a["converged_rate"]))
        elif d_a_all < -10:
            dec_loses.append((label, d_a_all, d_a_conv,
                              d["converged_rate"], a["converged_rate"]))

        if abs(d["converged_rate"] - a["converged_rate"]) > 0.15:
            mixed_conv.append((label, d["converged_rate"], a["converged_rate"]))

    if dec_beats:
        lines.append(
            "**Decentralized policy beats ACS overall** (reward gap > +10):"
        )
        for label, d_a, d_a_conv, d_cv, a_cv in dec_beats:
            conv_txt = (f", conv-only d-a **{d_a_conv:+.0f}**"
                        if d_a_conv is not None else "")
            lines.append(
                f"- `{label}`: overall d-a **{d_a:+.0f}**{conv_txt}, "
                f"dec cv% **{100*d_cv:.0f}%** vs acs cv% **{100*a_cv:.0f}%**"
            )
        lines.append("")

    if dec_loses:
        lines.append(
            "**Decentralized policy fails vs ACS overall** (reward gap < -10):"
        )
        for label, d_a, d_a_conv, d_cv, a_cv in dec_loses:
            conv_txt = (f", conv-only d-a **{d_a_conv:+.0f}**"
                        if d_a_conv is not None else "")
            lines.append(
                f"- `{label}`: overall d-a **{d_a:+.0f}**{conv_txt}, "
                f"dec cv% **{100*d_cv:.0f}%** vs acs cv% **{100*a_cv:.0f}%**"
            )
        lines.append("")

    if mixed_conv:
        lines.append(
            "**Convergence rate differs significantly between dec and ACS** "
            "(|Δcv%| > 15 points):"
        )
        for label, d_cv, a_cv in mixed_conv:
            lines.append(
                f"- `{label}`: dec cv% **{100*d_cv:.0f}%** "
                f"vs acs cv% **{100*a_cv:.0f}%** "
                f"(Δ {100*(d_cv-a_cv):+.0f} pts)"
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

    eval_args = eval_data.get("args", {})

    lines: list = []
    write_header(lines, eval_args)
    write_topology_descriptions(lines)
    write_metric_defs(lines)
    write_convergence_table(eval_data, lines)
    write_overall_table(eval_data, lines)
    write_split_table(eval_data, lines, "converged",
                      "Converged-only metrics")
    write_split_table(eval_data, lines, "not_converged",
                      "Non-converged-only metrics")
    write_takeaways(eval_data, lines)

    lines.append("---")
    lines.append(
        f"*Generated from `eval_checkpoint_decentralized.py` — "
        f"{eval_args.get('num_episodes', '?')} episodes, "
        f"max_steps={eval_args.get('max_steps', '?')}, "
        f"num_agents={eval_args.get('num_agents', '?')}, "
        f"conv window={eval_args.get('conv_window', '?')}, "
        f"pos_rate={eval_args.get('conv_pos_rate', '?')}, "
        f"op_rate={eval_args.get('conv_op_rate', '?')}*"
    )
    lines.append("")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
