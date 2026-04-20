"""
Per-time-step computation-time benchmark: trained Transformer PPO vs heuristics.

Measures wall-clock time per action-selection call on :class:`LazyAgentsCentralized`
for four policies:

- ``ACS``             -- ``env.get_fully_active_action()``  (fully-active baseline)
- ``Heuristic``       -- ``env.compute_heuristic_action()`` (sole-lazy heuristic)
- ``RL``              -- trained PPO policy via ``policy.compute_single_action``
                         (includes RLlib's preprocessing, batch-building, and
                         action-distribution sampling/clipping)
- ``RL_pure_forward`` -- raw ``policy.model.policy_network(obs_tensors)`` call
                         only, with obs pre-tensorised on the target device
                         outside the timer (strips almost all RLlib overhead;
                         this is the "NN-only" compute cost)

Only the action-selection call is timed. The subsequent ``env.step(action)`` is
still executed so the agents move and the observation distribution is
representative, but its time is not counted.

Per-sample timing uses ``time.perf_counter_ns()``.

Notes
-----
- By default the trained policy is pinned to CPU (``--device cpu``) because the
  heuristics run on CPU; this is the apples-to-apples comparison for
  "computational load". Pass ``--device cuda`` to measure GPU inference instead
  (still synchronised before stopping the clock, but **only for the RL
  variants** -- see below).
- Cost attribution: ``torch.cuda.synchronize()`` is called around the timed
  call *only* for ``RL`` and ``RL_pure_forward``. ``ACS`` and ``Heuristic``
  are pure NumPy and never touch the GPU in any deployment path, so charging
  them sync overhead would misattribute cost that only exists for the NN
  policy. Their timings therefore look identical whether ``--device`` is
  ``cpu`` or ``cuda``.
- A warmup pass (``--warmup_steps``) is run per policy before timing to avoid
  first-call import/JIT effects.
- Number of timed samples per policy = ``num_rollouts * steps_per_rollout``,
  which defaults to 3 * 2000 = 6000 per policy. The env is reset between
  rollouts with ``base_seed + rollout_idx``; if the episode terminates early,
  we immediately reset within the same rollout so we always collect exactly
  ``steps_per_rollout`` samples.
- Model parameter counts and (when ``--device cuda``) GPU memory usage are
  collected once per run and stored under ``model_info`` in the output JSON.

Usage
-----
    python -m experiments.compute_benchmark.benchmark \
        --output experiments/compute_benchmark/results/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from typing import Dict, List

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

warnings.filterwarnings("ignore", category=DeprecationWarning)


DEFAULT_CHECKPOINT = os.path.join(
    WORKSPACE,
    "bk/bk_082623/PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47/"
    "checkpoint_000074/policies/default_policy",
)

ALGOS = ["ACS", "Heuristic", "RL", "RL_pure_forward"]


def build_env_config(num_agents: int, max_time_step: int) -> dict:
    return {
        "num_agents_max": num_agents,
        "num_agents_min": num_agents,
        "speed": 15,
        "max_turn_rate": 8 / 15,
        "inter_agent_strength": 5,
        "communication_decay_rate": 1 / 3,
        "bonding_strength": 1,
        "initial_position_bound": 250,
        "predefined_distance": 60,
        "std_pos_converged": 45,
        "std_vel_converged": 0.1,
        "std_pos_rate_converged": 0.1,
        "std_vel_rate_converged": 0.2,
        "max_time_step": max_time_step,
        "incomplete_episode_penalty": 0,
        "normalize_obs": True,
        "use_fixed_horizon": False,
        "use_L2_norm": False,
        "use_heuristics": True,
        "_use_fixed_lazy_idx": True,
        "use_preprocessed_obs": True,
        "get_state_hist": False,
    }


def get_action(algo: str, obs, env, policy):
    if algo == "ACS":
        return env.get_fully_active_action()
    if algo == "Heuristic":
        return env.compute_heuristic_action()
    if algo == "RL":
        action = policy.compute_single_action(obs, explore=False)[0]
        return np.clip(action, 0.0, 1.0)
    raise ValueError(f"Unknown algo: {algo}")


def pure_forward_action_of(policy, device: str):
    """Return a callable that runs only the raw NN forward pass and returns an action.

    The numpy->torch obs conversion and the mean/std post-processing are done
    *outside* the ``fwd_only`` timer via a two-phase structure: the caller
    builds the tensor dict, calls ``fwd_only`` (the timed portion) which just
    executes ``policy_network(...)``, and then finalises the action after the
    clock stops.
    """

    import torch

    dev = torch.device(device)
    net = policy.model.policy_network  # LazinessAllocator
    half = None  # action dimension, resolved from first output

    def tensorise(obs):
        ae = torch.as_tensor(obs["agent_embeddings"], dtype=torch.float32, device=dev).unsqueeze(0)
        pt = torch.as_tensor(obs["pad_tokens"], dtype=torch.int32, device=dev).unsqueeze(0)
        return {"agent_embeddings": ae, "pad_tokens": pt}

    def fwd_only(td):
        with torch.inference_mode():
            out, _, _ = net(td)
        return out

    def finalise(out):
        nonlocal half
        if half is None:
            half = out.shape[-1] // 2
        mean = out[0, :half]
        return mean.detach().clamp_(0.0, 1.0).cpu().numpy()

    return tensorise, fwd_only, finalise


def force_policy_device(policy, device: str):
    """Move a loaded RLlib torch policy onto a specific device."""
    import torch

    torch_device = torch.device(device)
    policy.device = torch_device
    if hasattr(policy, "model") and policy.model is not None:
        policy.model.to(torch_device)
    if hasattr(policy, "model_gpu_towers"):
        policy.model_gpu_towers = []


def cuda_sync_if_needed(device: str):
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def time_policy(
    algo: str,
    env,
    policy,
    num_rollouts: int,
    steps_per_rollout: int,
    warmup_steps: int,
    base_seed: int,
    device: str,
) -> Dict[str, object]:
    """Run warmup + timed rollouts for one algorithm.

    Returns a dict with the per-sample times in nanoseconds and summary stats.

    For ``RL_pure_forward``, only the raw ``policy_network`` forward pass is
    timed (the numpy->tensor obs conversion and the mean extraction happen
    outside the ``perf_counter_ns`` window).
    """

    pure_fwd = algo == "RL_pure_forward"
    tensorise = fwd_only = finalise = None
    if pure_fwd:
        tensorise, fwd_only, finalise = pure_forward_action_of(policy, device)

    # Only the RL variants actually launch GPU kernels. ACS / Heuristic are
    # pure NumPy and never touch the GPU in any deployment path, so charging
    # them ``torch.cuda.synchronize()`` overhead would misattribute cost that
    # only exists for the NN policy. We therefore sync *only* for RL runs.
    uses_gpu = pure_fwd or algo == "RL"
    needs_sync = uses_gpu and device.startswith("cuda")
    sync = cuda_sync_if_needed if needs_sync else (lambda _d: None)

    def step_action(obs):
        """Take one timed action; return (time_ns, action)."""
        if pure_fwd:
            td = tensorise(obs)
            sync(device)
            t0 = time.perf_counter_ns()
            out = fwd_only(td)
            sync(device)
            t1 = time.perf_counter_ns()
            action = finalise(out)
        else:
            sync(device)
            t0 = time.perf_counter_ns()
            action = get_action(algo, obs, env, policy)
            sync(device)
            t1 = time.perf_counter_ns()
        return t1 - t0, action

    # ---- Warmup ----
    env.seed(base_seed - 1)
    obs = env.reset()
    for _ in range(warmup_steps):
        _, action = step_action(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    # ---- Timed rollouts ----
    total_samples = num_rollouts * steps_per_rollout
    times_ns = np.empty(total_samples, dtype=np.int64)
    idx = 0
    resets = 0  # counts forced resets inside a rollout due to early termination
    for rollout_idx in range(num_rollouts):
        env.seed(base_seed + rollout_idx)
        obs = env.reset()
        for _ in range(steps_per_rollout):
            dt_ns, action = step_action(obs)
            times_ns[idx] = dt_ns
            idx += 1
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
                resets += 1

    times_us = times_ns / 1000.0  # microseconds
    return {
        "num_samples": int(total_samples),
        "mean_us": float(times_us.mean()),
        "std_us": float(times_us.std(ddof=1)),
        "median_us": float(np.median(times_us)),
        "p05_us": float(np.percentile(times_us, 5)),
        "p95_us": float(np.percentile(times_us, 95)),
        "p99_us": float(np.percentile(times_us, 99)),
        "min_us": float(times_us.min()),
        "max_us": float(times_us.max()),
        "times_us": times_us.tolist(),
        "inner_resets": int(resets),
    }


def time_pso(
    env_config: dict,
    num_rollouts: int,
    base_seed: int,
    num_cpus: int,
) -> Dict[str, object]:
    """Run PSO optimization and measure wall-clock time.

    PSO finds a single constant action per episode, so timing is
    per-optimization. Per-step equivalent = optimization_time / episode_steps.
    """
    from env.envs import LazyAgentsCentralized
    from utils.metaheuristics import PSOActionOptimizer

    optimizer = PSOActionOptimizer(num_cpus=num_cpus)

    optimization_times_s: List[float] = []
    episode_steps_list: List[int] = []
    per_step_times_us: List[float] = []
    costs: List[float] = []

    for i in range(num_rollouts):
        env = LazyAgentsCentralized(env_config)
        env.seed(base_seed + i)
        env.reset()

        optimal_action, cost, elapsed = optimizer.optimize(env)

        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(optimal_action)
            steps += 1

        optimization_times_s.append(elapsed)
        episode_steps_list.append(steps)
        per_step_us = elapsed / steps * 1e6
        per_step_times_us.append(per_step_us)
        costs.append(float(cost))

        print(
            f"  rollout {i}: opt_time={elapsed:.1f}s  "
            f"steps={steps}  per_step={per_step_us:.0f}us  cost={cost:.2f}"
        )

    optimizer.shutdown()

    ps = np.array(per_step_times_us)
    ot = np.array(optimization_times_s)
    es = np.array(episode_steps_list, dtype=int)

    return {
        "num_samples": num_rollouts,
        "mean_us": float(ps.mean()),
        "std_us": float(ps.std(ddof=1)) if num_rollouts > 1 else 0.0,
        "median_us": float(np.median(ps)),
        "min_us": float(ps.min()),
        "max_us": float(ps.max()),
        "per_step_times_us": ps.tolist(),
        "optimization_time_s": {
            "mean": float(ot.mean()),
            "std": float(ot.std(ddof=1)) if num_rollouts > 1 else 0.0,
            "min": float(ot.min()),
            "max": float(ot.max()),
            "values": ot.tolist(),
        },
        "episode_steps": {
            "mean": float(es.mean()),
            "min": int(es.min()),
            "max": int(es.max()),
            "values": es.tolist(),
        },
        "costs": costs,
        "num_cpus": num_cpus,
    }


def run_benchmark(args: argparse.Namespace) -> Dict[str, object]:
    algos_to_run = args.algos if args.algos else ALGOS
    needs_rl = any(a in algos_to_run for a in ["RL", "RL_pure_forward"])
    needs_shared_env = any(
        a in algos_to_run for a in ["ACS", "Heuristic", "RL", "RL_pure_forward"]
    )

    # Control device visibility *before* torch/ray import chain spins up.
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    env = None
    policy = None
    model_info = {}
    env_config = build_env_config(args.num_agents, args.max_time_step)

    if needs_shared_env:
        # Deferred imports so CUDA_VISIBLE_DEVICES applies.
        import torch  # noqa: F401
        from ray.rllib.models import ModelCatalog
        from ray.tune.registry import register_env

        # Torch's default on a big-core box (here: 36) oversubscribes a B=1
        # forward and the measurement gets dominated by thread-pool wakeup
        # jitter.  Pick an explicit, sensible count and record it.
        if args.torch_threads is not None and args.torch_threads > 0:
            torch.set_num_threads(args.torch_threads)
            try:
                torch.set_num_interop_threads(args.torch_threads)
            except RuntimeError:
                pass

        from env.envs import LazyAgentsCentralized
        from models.lazy_allocator import MyRLlibTorchWrapper

        ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
        register_env("lazy_env", lambda cfg: LazyAgentsCentralized(cfg))
        env = LazyAgentsCentralized(env_config)

    if needs_rl:
        from ray.rllib.policy.policy import Policy

        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        print(f"Loading policy from {args.checkpoint}")
        policy = Policy.from_checkpoint(args.checkpoint)
        policy.model.eval()
        force_policy_device(policy, args.device)
        print(f"Policy device: {next(policy.model.parameters()).device}")

    # Environment info for provenance.
    host_info = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "num_cpus": os.cpu_count(),
    }
    if needs_shared_env:
        import torch
        host_info.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        })
    if policy is not None:
        host_info["device"] = str(next(policy.model.parameters()).device)
    try:
        host_info["cpu_model"] = _read_first_cpu_model()
    except Exception:
        host_info["cpu_model"] = None

    if needs_rl:
        model_info = collect_model_info(policy, args.num_agents)
        print(
            f"Model: total={model_info['total_params']:,} params "
            f"(policy_net={model_info['policy_network_params']:,}, "
            f"value_branch={model_info['value_branch_params']:,}, "
            f"value_network={model_info['value_network_params']:,})"
        )
        print(f"       state_dict ≈ {model_info['state_dict_mb']:.3f} MiB "
              f"({model_info['dtype']})")
        if "cuda_weights_mb" in model_info:
            print(
                f"       GPU weights={model_info['cuda_weights_mb']:.3f} MiB, "
                f"peak-inference={model_info['cuda_peak_inference_mb']:.3f} MiB"
            )

    results: Dict[str, object] = {}
    for algo in algos_to_run:
        if algo == "PSO":
            print(
                f"\n[PSO] rollouts={args.num_rollouts}  num_cpus={args.num_cpus}"
            )
            t_start = time.time()
            summary = time_pso(
                env_config=env_config,
                num_rollouts=args.num_rollouts,
                base_seed=args.base_seed,
                num_cpus=args.num_cpus,
            )
            wall_s = time.time() - t_start
            summary["wall_clock_s"] = wall_s
            results[algo] = summary
            opt_t = summary["optimization_time_s"]
            ep_s = summary["episode_steps"]
            print(
                f"  per_step: mean={summary['mean_us']:.0f}us  "
                f"opt_time: mean={opt_t['mean']:.1f}s  "
                f"steps: mean={ep_s['mean']:.0f}"
            )
        else:
            print(
                f"\n[{algo}] warmup={args.warmup_steps}  "
                f"rollouts={args.num_rollouts} x steps={args.steps_per_rollout} "
                f"(n={args.num_rollouts*args.steps_per_rollout})"
            )
            t_start = time.time()
            summary = time_policy(
                algo=algo,
                env=env,
                policy=policy,
                num_rollouts=args.num_rollouts,
                steps_per_rollout=args.steps_per_rollout,
                warmup_steps=args.warmup_steps,
                base_seed=args.base_seed,
                device=args.device,
            )
            wall_s = time.time() - t_start
            summary["wall_clock_s"] = wall_s
            results[algo] = summary
            print(
                f"  mean={summary['mean_us']:.2f}us  median={summary['median_us']:.2f}us  "
                f"p95={summary['p95_us']:.2f}us  p99={summary['p99_us']:.2f}us  "
                f"std={summary['std_us']:.2f}us  n={summary['num_samples']}"
            )

    return {
        "args": vars(args),
        "env_config": env_config,
        "host_info": host_info,
        "model_info": model_info,
        "algorithms": results,
    }


def collect_model_info(policy, num_agents: int) -> Dict[str, object]:
    """Count parameters and (on CUDA) measure weight + peak-inference memory.

    - ``total_params`` is every parameter in ``policy.model`` (policy network +
      value branch).
    - ``policy_network_params`` is only the transformer that produces actions;
      it's what ``RL_pure_forward`` actually runs.
    - On CUDA we also report:
        * ``weights_mb``   -- bytes resident for the model's parameters
          (+ buffers) right after ``to(cuda)``.
        * ``peak_inference_mb`` -- peak allocated during ten B=1 forwards
          through ``policy_network`` (weights + activations + workspace).
    """

    import torch

    model = policy.model
    policy_net = model.policy_network
    value_branch = getattr(model, "value_branch", None)
    value_network = getattr(model, "value_network", None)

    def num_params(module) -> int:
        return sum(p.numel() for p in module.parameters()) if module is not None else 0

    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    info: Dict[str, object] = {
        "total_params": num_params(model),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "policy_network_params": num_params(policy_net),
        "value_branch_params": num_params(value_branch),
        "value_network_params": num_params(value_network),
        "param_bytes": int(param_bytes),
        "buffer_bytes": int(buffer_bytes),
        "param_mb": param_bytes / (1024 ** 2),
        "state_dict_mb": (param_bytes + buffer_bytes) / (1024 ** 2),
        "dtype": str(next(policy.model.parameters()).dtype),
    }

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        # Measure the weights-only delta: move to CPU first (so the GPU side
        # is clean of this model's tensors), take a baseline, then move back
        # and read the delta. Works whether or not the policy was already on
        # CUDA when collect_model_info was called.
        orig_device = next(model.parameters()).device
        model.to("cpu")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base_alloc = torch.cuda.memory_allocated(dev)
        model.to(dev)
        torch.cuda.synchronize()
        weights_alloc = torch.cuda.memory_allocated(dev) - base_alloc

        torch.cuda.reset_peak_memory_stats(dev)
        ae = torch.zeros(1, num_agents, 6, dtype=torch.float32, device=dev)
        pt = torch.zeros(1, num_agents, dtype=torch.int32, device=dev)
        td = {"agent_embeddings": ae, "pad_tokens": pt}
        with torch.inference_mode():
            for _ in range(10):
                _ = policy_net(td)
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated(dev)

        info["cuda_weights_mb"] = weights_alloc / (1024 ** 2)
        info["cuda_peak_inference_mb"] = peak_alloc / (1024 ** 2)
        info["cuda_activation_mb"] = max(0.0, (peak_alloc - weights_alloc) / (1024 ** 2))

        # restore to the originally-pinned device so the timed benchmark runs
        # on whatever --device the user asked for.
        model.to(orig_device)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return info


def _read_first_cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--max_time_step", type=int, default=2000)
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--steps_per_rollout", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--base_seed", type=int, default=4242)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=4,
        help="torch.set_num_threads/set_num_interop_threads value; "
        "the default 4 avoids the oversubscription that torch's default "
        "(= #physical cores) causes on a small B=1 forward",
    )
    parser.add_argument(
        "--algos",
        nargs="*",
        default=None,
        help="subset of algorithms to benchmark; PSO is opt-in "
        "(default: ACS Heuristic RL RL_pure_forward)",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=None,
        help="number of CPUs for Ray (PSO parallelisation); None = auto-detect",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(THIS_DIR, "results", "benchmark.json"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    doc = run_benchmark(args)
    with open(args.output, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
