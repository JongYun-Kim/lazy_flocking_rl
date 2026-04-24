#!/usr/bin/env bash
set -e

LOG="/workspace/experiments/performance_benchmark/run_commands_log.txt"
echo "=== Benchmark re-run started: $(date) ===" | tee "$LOG"

run_cmd() {
    local label="$1"; shift
    echo "" | tee -a "$LOG"
    echo "[$label] START  $(date)" | tee -a "$LOG"
    echo "[$label] CMD: $*" | tee -a "$LOG"
    local t0=$(date +%s)
    "$@" 2>&1 | tail -5 | tee -a "$LOG"
    local t1=$(date +%s)
    echo "[$label] DONE   $(date)  ($(( t1 - t0 ))s)" | tee -a "$LOG"
}

cd /workspace

# --- Comparison 1: 20 UAVs, 100 episodes ---
run_cmd "C1-ACS"       python -m experiments.performance_benchmark.collect_acs --num_cpus 64
run_cmd "C1-Heuristic" python -m experiments.performance_benchmark.collect_heuristic --num_cpus 64
run_cmd "C1-RL"        python -m experiments.performance_benchmark.collect_rl --num_cpus 64 --num_workers 16
run_cmd "C1-PSO"       python -m experiments.performance_benchmark.collect_pso --num_cpus 64 --seed_pso

# --- Comparison 2: scalability sweep ---
run_cmd "C2-ACS"       python -m experiments.performance_benchmark.collect_scalability --method acs --num_cpus 64
run_cmd "C2-Heuristic" python -m experiments.performance_benchmark.collect_scalability --method heuristic --num_cpus 64
run_cmd "C2-RL"        python -m experiments.performance_benchmark.collect_scalability --method rl --device cuda --num_cpus 64 --num_workers 64
run_cmd "C2-PSO"       python -m experiments.performance_benchmark.collect_scalability --method pso --num_cpus 64 --num_episodes 30 --seed_pso

echo "" | tee -a "$LOG"
echo "=== ALL DONE: $(date) ===" | tee -a "$LOG"
