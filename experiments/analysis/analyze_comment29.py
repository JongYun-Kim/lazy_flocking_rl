#!/usr/bin/env python3
"""
Compute statistics for Comment 2.9 response.

Reads benchmark results from ../performance_benchmark/results/ and produces
a single JSON file with:
  - n=20 summary statistics + pairwise tests
  - n=20 ensemble time-series (sigma_p, sigma_v, Phi) with 95% CI
  - Scalability analysis (n=8..1024): convergence rates, swarm integrity,
    cost decomposition, Heuristic-vs-RL tests

Usage (from repository root):
    docker run --rm \
      -v "$(pwd)/experiments":/workspace \
      py313 python /workspace/analysis/analyze_comment29.py

Output:
    experiments/analysis/data/comment29_analysis.json
"""
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats

# --------------- paths (relative to this script) ---------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "performance_benchmark" / "results"
OUTPUT_DIR = SCRIPT_DIR / "data"

# --------------- constants ---------------
V_CRUISE = 15.0   # m/s
DT = 0.1          # seconds per step
MAX_STEP_N20 = 1000    # 100 s
MAX_STEP_SCALE = 2000  # 200 s

METHODS = ["acs", "heuristic", "rl"]
SCALE_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]


def compute_phi(std_vel, v=V_CRUISE):
    ratio = (std_vel / v) ** 2
    return math.sqrt(max(0.0, 1.0 - ratio))


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def paired_ttest(a, b):
    a, b = np.array(a), np.array(b)
    t_stat, p_val = stats.ttest_rel(a, b)
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    return {
        "mean_diff": mean_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "n": len(a),
    }


def mann_whitney(a, b):
    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    cliffs_delta = (2 * u_stat / (len(a) * len(b))) - 1
    return {
        "u_stat": float(u_stat),
        "p_value": float(p_val),
        "cliffs_delta": float(cliffs_delta),
    }


def summary_stats(values):
    arr = np.array(values, dtype=float)
    n = len(arr)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    se = s / np.sqrt(n)
    return {
        "mean": m, "std": s,
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "q05": float(np.percentile(arr, 5)),
        "q95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": n,
        "ci95_low": m - 1.96 * se,
        "ci95_high": m + 1.96 * se,
    }


def _extract_metrics(episodes):
    return {
        "J": [-e["reward_L2"] for e in episodes],
        "C": [e["control_cost_L2"] for e in episodes],
        "t_f": [e["convergence_time"] for e in episodes],
        "std_pos_last": [e["std_pos_last"] for e in episodes],
        "std_vel_last": [e["std_vel_last"] for e in episodes],
        "phi": [compute_phi(e["std_vel_last"]) for e in episodes],
        "episode_length": [e["episode_length"] for e in episodes],
        "seed": [e["seed"] for e in episodes],
    }


# ============================================================
# PART 1: n=20 Benchmark
# ============================================================
def analyze_n20():
    print("=== Part 1: n=20 Benchmark ===")
    results = {}
    metrics = {}
    for method in METHODS:
        d = load_json(DATA_DIR / f"{method}.json")
        metrics[method] = _extract_metrics(d["results"])

    results["summary"] = {
        method: {k: summary_stats(metrics[method][k])
                 for k in ["J", "C", "t_f", "std_pos_last", "std_vel_last", "phi"]}
        for method in METHODS
    }

    results["convergence_rate"] = {}
    for method in METHODS:
        lengths = metrics[method]["episode_length"]
        conv = sum(1 for l in lengths if l < MAX_STEP_N20)
        results["convergence_rate"][method] = {
            "converged": conv, "total": len(lengths), "rate": conv / len(lengths),
        }

    assert metrics["acs"]["seed"] == metrics["heuristic"]["seed"] == metrics["rl"]["seed"]
    pairs = [("heuristic", "acs"), ("rl", "acs"), ("heuristic", "rl")]
    results["statistical_tests"] = {}
    for mk in ["J", "C", "t_f", "phi"]:
        results["statistical_tests"][mk] = {
            f"{a}_vs_{b}": {
                "paired_ttest": paired_ttest(metrics[a][mk], metrics[b][mk]),
                "mann_whitney": mann_whitney(metrics[a][mk], metrics[b][mk]),
            }
            for a, b in pairs
        }
    return results


# ============================================================
# PART 2: n=20 Ensemble Time-Series (CI bands)
# ============================================================
def analyze_trajectories():
    print("=== Part 2: Ensemble Time-Series ===")
    results = {}
    MAX_T = MAX_STEP_N20
    DOWNSAMPLE = 5  # every 0.5 s

    for method in METHODS:
        print(f"  Loading trajectories/{method}.json ...")
        traj = load_json(DATA_DIR / "trajectories" / f"{method}.json")
        episodes = traj["results"]
        n_eps = len(episodes)

        sp_matrix = np.zeros((n_eps, MAX_T))
        sv_matrix = np.zeros((n_eps, MAX_T))
        for i, ep in enumerate(episodes):
            sp, sv = ep["spatial_entropy"], ep["velocity_entropy"]
            ep_len = len(sp)
            if ep_len >= MAX_T:
                sp_matrix[i, :] = sp[:MAX_T]
                sv_matrix[i, :] = sv[:MAX_T]
            else:
                sp_matrix[i, :ep_len] = sp
                sv_matrix[i, :ep_len] = sv
                sp_matrix[i, ep_len:] = sp[-1]
                sv_matrix[i, ep_len:] = sv[-1]

        phi_matrix = np.sqrt(np.maximum(0.0, 1.0 - (sv_matrix / V_CRUISE) ** 2))

        time_idx = list(range(0, MAX_T, DOWNSAMPLE))
        time_s = [t * DT for t in time_idx]

        mr = {"time_s": time_s}
        for name, matrix in [("sigma_p", sp_matrix), ("sigma_v", sv_matrix), ("phi", phi_matrix)]:
            sampled = matrix[:, time_idx]
            means = np.mean(sampled, axis=0)
            stds = np.std(sampled, axis=0, ddof=1)
            ci = 1.96 * stds / np.sqrt(n_eps)
            mr[name] = {
                "mean": means.tolist(),
                "ci_low": (means - ci).tolist(),
                "ci_high": (means + ci).tolist(),
                "q25": np.percentile(sampled, 25, axis=0).tolist(),
                "q75": np.percentile(sampled, 75, axis=0).tolist(),
            }

        ep_lengths = np.array([len(ep["spatial_entropy"]) for ep in episodes])
        mr["n_still_running"] = [int(np.sum(ep_lengths > t)) for t in time_idx]
        results[method] = mr
        print(f"  {method}: {n_eps} episodes done")
    return results


# ============================================================
# PART 3: Scalability
# ============================================================
def analyze_scalability():
    print("=== Part 3: Scalability ===")
    results = {}
    for n in SCALE_SIZES:
        nk = f"n{n}"
        results[nk] = {}
        method_metrics = {}
        for method in METHODS:
            fpath = DATA_DIR / "scalability" / f"{method}_n{n}.json"
            if not fpath.exists():
                continue
            d = load_json(fpath)
            method_metrics[method] = _extract_metrics(d["results"])
            print(f"  {method}_n{n}: {len(d['results'])} episodes")

        results[nk]["summary"] = {
            method: {k: summary_stats(method_metrics[method][k])
                     for k in ["J", "C", "t_f", "std_pos_last", "std_vel_last", "phi"]}
            for method in method_metrics
        }

        results[nk]["convergence_rate"] = {}
        for method in method_metrics:
            lengths = method_metrics[method]["episode_length"]
            conv = sum(1 for l in lengths if l < MAX_STEP_SCALE)
            results[nk]["convergence_rate"][method] = {
                "converged": conv, "total": len(lengths), "rate": conv / len(lengths),
            }

        results[nk]["converged_integrity"] = {}
        for method in method_metrics:
            mask = [el < MAX_STEP_SCALE for el in method_metrics[method]["episode_length"]]
            conv_sp = [v for v, m in zip(method_metrics[method]["std_pos_last"], mask) if m]
            conv_sv = [v for v, m in zip(method_metrics[method]["std_vel_last"], mask) if m]
            conv_phi = [compute_phi(v) for v, m in zip(method_metrics[method]["std_vel_last"], mask) if m]
            if conv_sp:
                results[nk]["converged_integrity"][method] = {
                    "std_pos_last": summary_stats(conv_sp),
                    "std_vel_last": summary_stats(conv_sv),
                    "phi": summary_stats(conv_phi),
                    "n_converged": len(conv_sp),
                }

        if "heuristic" in method_metrics and "rl" in method_metrics:
            if method_metrics["heuristic"]["seed"] == method_metrics["rl"]["seed"]:
                results[nk]["heuristic_vs_rl"] = {
                    mk: {
                        "paired_ttest": paired_ttest(
                            method_metrics["heuristic"][mk], method_metrics["rl"][mk]),
                        "mann_whitney": mann_whitney(
                            method_metrics["heuristic"][mk], method_metrics["rl"][mk]),
                    }
                    for mk in ["J", "C", "t_f", "phi"]
                }
    return results


# ============================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "n20_benchmark": analyze_n20(),
        "n20_timeseries": analyze_trajectories(),
        "scalability": analyze_scalability(),
    }
    opath = OUTPUT_DIR / "comment29_analysis.json"
    with open(opath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== Done. {opath.stat().st_size / 1024:.1f} KB => {opath} ===")
