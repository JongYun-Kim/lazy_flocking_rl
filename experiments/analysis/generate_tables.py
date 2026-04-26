#!/usr/bin/env python3
"""
Generate summary tables from Comment 2.9 analysis data.

Outputs CSV files for:
  1. n=20 pairwise statistical tests
  2. Scalability convergence rates + swarm integrity
  3. Scalability cost decomposition
  4. Scalability Heuristic-vs-RL statistical tests

Usage (from repository root):
    docker run --rm \
      -v "$(pwd)/experiments":/workspace \
      py313 python /workspace/analysis/generate_tables.py

Output:
    experiments/analysis/data/table_*.csv
"""
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data" / "comment29_analysis.json"
OUT_DIR = SCRIPT_DIR / "data"

METHODS = ["acs", "heuristic", "rl"]
SCALE_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved {path}")


def main():
    with open(DATA_PATH) as f:
        d = json.load(f)

    # ---- Table 1: n=20 pairwise statistical tests ----
    header = ["Metric", "Comparison", "n", "mean_diff", "t", "p_value", "Cohen_d",
              "MW_p_value", "Cliff_delta"]
    rows = []
    for mk in ["J", "C", "t_f", "phi"]:
        for comp, tests in d["n20_benchmark"]["statistical_tests"][mk].items():
            pt = tests["paired_ttest"]
            mw = tests["mann_whitney"]
            rows.append([
                mk, comp, pt["n"],
                f"{pt['mean_diff']:.4f}", f"{pt['t_stat']:.2f}",
                f"{pt['p_value']:.2e}", f"{pt['cohens_d']:.4f}",
                f"{mw['p_value']:.2e}", f"{mw['cliffs_delta']:.4f}",
            ])
    write_csv(OUT_DIR / "table_n20_statistical_tests.csv", header, rows)

    # ---- Table 2: Scalability convergence + swarm integrity ----
    header = ["n", "Method", "conv_rate(%)", "conv_count", "total",
              "sp_last_mean", "sp_last_std", "sp_last_max",
              "sv_last_mean", "phi_mean", "phi_std"]
    rows = []
    for n in SCALE_SIZES:
        nk = f"n{n}"
        for method in METHODS:
            cr = d["scalability"][nk]["convergence_rate"][method]
            ci = d["scalability"][nk]["converged_integrity"].get(method, {})
            sp = ci.get("std_pos_last", {})
            sv = ci.get("std_vel_last", {})
            phi = ci.get("phi", {})
            rows.append([
                n, method.upper(),
                f"{cr['rate']*100:.1f}", cr["converged"], cr["total"],
                f"{sp.get('mean',''):.2f}" if sp else "",
                f"{sp.get('std',''):.2f}" if sp else "",
                f"{sp.get('max',''):.2f}" if sp else "",
                f"{sv.get('mean',''):.4f}" if sv else "",
                f"{phi.get('mean',''):.6f}" if phi else "",
                f"{phi.get('std',''):.6f}" if phi else "",
            ])
    write_csv(OUT_DIR / "table_scalability_integrity.csv", header, rows)

    # ---- Table 3: Scalability cost decomposition ----
    header = ["n", "Method", "J_mean", "J_std", "C_mean", "C_std",
              "t_f_mean", "t_f_std", "conv_rate(%)"]
    rows = []
    for n in SCALE_SIZES:
        nk = f"n{n}"
        for method in METHODS:
            s = d["scalability"][nk]["summary"][method]
            cr = d["scalability"][nk]["convergence_rate"][method]
            rows.append([
                n, method.upper(),
                f"{s['J']['mean']:.1f}", f"{s['J']['std']:.1f}",
                f"{s['C']['mean']:.1f}", f"{s['C']['std']:.1f}",
                f"{s['t_f']['mean']:.1f}", f"{s['t_f']['std']:.1f}",
                f"{cr['rate']*100:.1f}",
            ])
    write_csv(OUT_DIR / "table_scalability_cost_decomposition.csv", header, rows)

    # ---- Table 4: Scalability Heuristic vs RL tests ----
    header = ["n", "Metric", "mean_diff", "t", "p_value", "Cohen_d"]
    rows = []
    for n in SCALE_SIZES:
        nk = f"n{n}"
        hvr = d["scalability"][nk].get("heuristic_vs_rl", {})
        for mk in ["J", "C", "t_f", "phi"]:
            if mk in hvr:
                pt = hvr[mk]["paired_ttest"]
                rows.append([
                    n, mk,
                    f"{pt['mean_diff']:.2f}", f"{pt['t_stat']:.2f}",
                    f"{pt['p_value']:.2e}", f"{pt['cohens_d']:.4f}",
                ])
    write_csv(OUT_DIR / "table_scalability_heuristic_vs_rl.csv", header, rows)

    # ---- Print concise summary for main text use ----
    print("\n" + "=" * 60)
    print("CONCISE SUMMARY FOR MAIN TEXT (large n only)")
    print("=" * 60)
    print(f"\n{'n':>6} | {'ACS':>10} | {'Heur':>10} | {'RL':>10} | RL sp_last (conv)")
    print("-" * 62)
    for n in [256, 512, 1024]:
        nk = f"n{n}"
        parts = []
        for method in METHODS:
            cr = d["scalability"][nk]["convergence_rate"][method]
            parts.append(f"{cr['rate']*100:.1f}%")
        ci = d["scalability"][nk]["converged_integrity"]["rl"]["std_pos_last"]
        print(f"{n:>6} | {parts[0]:>10} | {parts[1]:>10} | {parts[2]:>10} | "
              f"{ci['mean']:.1f}+/-{ci['std']:.1f} [max {ci['max']:.1f}]")

    print("\nHeuristic vs RL crossover (J):")
    for n in SCALE_SIZES:
        nk = f"n{n}"
        hvr = d["scalability"][nk].get("heuristic_vs_rl", {})
        pt = hvr.get("J", {}).get("paired_ttest", {})
        if pt:
            direction = "Heur<RL" if pt["mean_diff"] < 0 else "RL<Heur"
            print(f"  n={n:>4}: diff={pt['mean_diff']:>+8.2f}, d={pt['cohens_d']:>+7.4f}, "
                  f"p={pt['p_value']:.2e} ({direction})")


if __name__ == "__main__":
    main()
