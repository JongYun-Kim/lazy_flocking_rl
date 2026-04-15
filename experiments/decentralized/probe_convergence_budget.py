"""
Probe ACS convergence budget per topology.

Runs ACS-only rollouts at a long horizon, records per-step connectivity and
polar order parameter together with the env's existing ``std_pos_hist`` and
``std_vel_hist``, and reports per-topology convergence time distribution. Used
to pick the ``max_steps`` for the main decentralized eval.

Convergence criterion (post-hoc)
--------------------------------
The earliest step ``t`` (with ``t >= window - 1``) at which every step in the
trailing window ``[t - window + 1, t]`` satisfies:

  * ``is_network_connected`` is ``True``;
  * ``max(std_pos[window]) - min(std_pos[window]) < pos_rate``;
  * ``max(order_parameter[window]) - min(order_parameter[window]) < op_rate``.

If no such ``t`` exists, the episode did not converge under this criterion.

Usage
-----
    python -m experiments.decentralized.probe_convergence_budget \
        --num_episodes 15 --max_steps 3000 \
        --output experiments/decentralized/results/convergence_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from experiments.decentralized.env_networked import NetworkedLazyAgents  # noqa: E402
from experiments.decentralized.eval_checkpoint_decentralized import (  # noqa: E402
    DEFAULT_TOPOLOGIES,
    load_threshold_db,
    make_env_config,
    _round_thresholds,
)


# Max_steps candidates to report "would-have-converged-by" rates for.
CHECKPOINT_STEPS = (500, 1000, 1500, 2000, 2500, 3000)


def compute_convergence_step(
    std_pos: np.ndarray,
    op: np.ndarray,
    conn: np.ndarray,
    *,
    window: int,
    pos_rate: float,
    op_rate: float,
) -> int | None:
    T = len(std_pos)
    if T < window:
        return None
    for t in range(window - 1, T):
        lo, hi = t - window + 1, t + 1
        if not conn[lo:hi].all():
            continue
        sp = std_pos[lo:hi]
        if (sp.max() - sp.min()) >= pos_rate:
            continue
        op_w = op[lo:hi]
        if (op_w.max() - op_w.min()) >= op_rate:
            continue
        return int(t)
    return None


def run_acs_episode_with_traces(env: NetworkedLazyAgents, max_steps: int) -> dict:
    env.reset()
    op_hist = np.zeros(max_steps, dtype=np.float32)
    conn_hist = np.zeros(max_steps, dtype=bool)
    length = max_steps
    for t in range(max_steps):
        action = env.get_fully_active_action()
        _, _, done, _ = env.step(action)
        op_hist[t] = env.alignment_order_parameter()
        conn_hist[t] = env.is_network_connected()
        if done:
            length = t + 1
            break
    return {
        "length": length,
        "std_pos": np.array(env.std_pos_hist[:length], dtype=np.float32),
        "std_vel": np.array(env.std_vel_hist[:length], dtype=np.float32),
        "op": op_hist[:length],
        "conn": conn_hist[:length],
    }


def _pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=15)
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--base_seed", type=int, default=5000)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--pos_rate", type=float, default=2.0,
                        help="trailing-window max-min threshold for std_pos")
    parser.add_argument("--op_rate", type=float, default=0.05,
                        help="trailing-window max-min threshold for order parameter")
    parser.add_argument(
        "--acs_sweep", type=str,
        default=os.path.join(THIS_DIR, "results", "acs_topology_sweep.json"),
    )
    parser.add_argument("--pos_offset_mul", type=float, default=1.10)
    parser.add_argument("--vel_offset_mul", type=float, default=1.50)
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(THIS_DIR, "results", "convergence_probe.json"),
    )
    parser.add_argument(
        "--topologies", nargs="*", default=None,
        help="optional list of labels to restrict the run to",
    )
    args = parser.parse_args()

    topology_thresholds = load_threshold_db(args.acs_sweep)

    if args.topologies:
        selected = [e for e in DEFAULT_TOPOLOGIES if e["label"] in args.topologies]
    else:
        selected = list(DEFAULT_TOPOLOGIES)

    print(
        f"Probe | window={args.window}, pos_rate={args.pos_rate}, op_rate={args.op_rate}, "
        f"max_steps={args.max_steps}, episodes={args.num_episodes}"
    )
    hdr = (
        f"{'topology':>16s} | {'conv%':>6s} | "
        f"{'p50':>6s} {'p90':>6s} {'max':>6s} | "
        f"{'conn%':>6s} | " +
        " ".join(f"{'<'+str(s):>6s}" for s in CHECKPOINT_STEPS)
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print("(conv% = fraction of episodes converged under criterion; "
          "p50/p90/max = convergence step stats over converged; "
          "conn% = mean fraction of steps with single component; "
          "<S = conv% if we had cut the horizon at S)")
    print(sep)

    results: dict = {}
    t0 = time.time()
    for entry in selected:
        label = entry["label"]
        topo = entry["topology"]
        base_thresh = topology_thresholds.get(label, {})
        thresh = _round_thresholds(base_thresh, args.pos_offset_mul, args.vel_offset_mul)
        if not thresh:
            thresh = {"std_pos_converged": 45.0, "std_vel_converged": 0.1}
        env_config = make_env_config(args.num_agents, args.max_steps, topo, thresh)

        ep_stats: list[dict] = []
        t_topo = time.time()
        for ep in range(args.num_episodes):
            env = NetworkedLazyAgents(env_config)
            env.seed(args.base_seed + ep)
            tr = run_acs_episode_with_traces(env, args.max_steps)

            conv_step = compute_convergence_step(
                tr["std_pos"], tr["op"], tr["conn"],
                window=args.window, pos_rate=args.pos_rate, op_rate=args.op_rate,
            )
            ep_stats.append({
                "length": tr["length"],
                "conv_step": conv_step,
                "conn_rate": float(tr["conn"].mean()),
                "final_conn": bool(tr["conn"][-1]),
                "final_std_pos": float(tr["std_pos"][-1]),
                "final_std_vel": float(tr["std_vel"][-1]),
                "final_op": float(tr["op"][-1]),
                "op_mean": float(tr["op"].mean()),
                "std_pos_mean": float(tr["std_pos"].mean()),
            })
        t_topo = time.time() - t_topo

        conv_steps = np.array(
            [e["conv_step"] for e in ep_stats if e["conv_step"] is not None],
            dtype=np.int64,
        )
        conv_rate_final = len(conv_steps) / args.num_episodes
        conn_rate_mean = float(np.mean([e["conn_rate"] for e in ep_stats]))

        # conv% if we had cut horizon at S -- count episodes whose conv_step <= S-1.
        by_step: dict = {}
        for S in CHECKPOINT_STEPS:
            n = sum(1 for e in ep_stats
                    if e["conv_step"] is not None and e["conv_step"] <= S - 1)
            by_step[str(S)] = n / args.num_episodes

        row = (
            f"{label:>16s} | {100*conv_rate_final:>5.0f}% | "
            f"{_pct(conv_steps, 50):>6.0f} {_pct(conv_steps, 90):>6.0f} "
            f"{float(conv_steps.max()) if conv_steps.size else float('nan'):>6.0f} | "
            f"{100*conn_rate_mean:>5.0f}% | " +
            " ".join(f"{100*by_step[str(s)]:>5.0f}%" for s in CHECKPOINT_STEPS) +
            f"  ({t_topo:.0f}s)"
        )
        print(row)

        results[label] = {
            "num_episodes": args.num_episodes,
            "conv_rate_final": conv_rate_final,
            "conn_rate_mean": conn_rate_mean,
            "conv_step_p50": _pct(conv_steps, 50),
            "conv_step_p90": _pct(conv_steps, 90),
            "conv_step_max": float(conv_steps.max()) if conv_steps.size else float("nan"),
            "conv_rate_by_step": by_step,
            "episodes": ep_stats,
        }

    elapsed = time.time() - t0
    print(sep)
    print(f"Total elapsed: {elapsed:.0f}s")

    out_doc = {"args": vars(args), "topologies": results}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
