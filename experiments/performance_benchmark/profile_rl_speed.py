"""Profile RL inference and env.step() time across agent counts.

Measures:
- CPU forward (B=1)
- GPU forward (B=1)
- GPU CUDA graph (B=1)
- GPU batched forward (B=32, B=64)
- env.step() time (CPU, no model)

Usage:
    python -m experiments.performance_benchmark.profile_rl_speed
"""

import copy
import os
import sys
import time
import warnings

import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

warnings.filterwarnings("ignore", category=DeprecationWarning)

AGENT_COUNTS = [8, 16, 20, 32, 64, 128, 256, 512, 1024]
NUM_ITERS = 200
WARMUP = 50

DEFAULT_CHECKPOINT = os.path.join(
    WORKSPACE,
    "bk/bk_082623/"
    "PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47/"
    "checkpoint_000074/policies/default_policy",
)


def load_policy_network():
    import torch
    from ray.rllib.models import ModelCatalog
    from ray.rllib.policy.policy import Policy
    from ray.tune.registry import register_env
    from env.envs import LazyAgentsCentralized
    from models.lazy_allocator import MyRLlibTorchWrapper

    ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
    register_env("lazy_env", lambda cfg: LazyAgentsCentralized(cfg))

    policy = Policy.from_checkpoint(DEFAULT_CHECKPOINT)
    policy.model.eval()
    net = policy.model.policy_network
    return net


def profile_forward(net, n, device, num_iters=NUM_ITERS, warmup=WARMUP,
                    batch_sizes=(1,)):
    import torch
    results = {}
    dev = torch.device(device)
    net_dev = copy.deepcopy(net).to(dev).eval()

    for B in batch_sizes:
        ae = torch.randn(B, n, 6, dtype=torch.float32, device=dev)
        pt = torch.zeros(B, n, dtype=torch.int32, device=dev)
        td = {"agent_embeddings": ae, "pad_tokens": pt}

        with torch.no_grad():
            for _ in range(warmup):
                net_dev(td)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(num_iters):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                net_dev(td)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        times_us = np.array(times) * 1e6
        results[B] = {
            "batch_total_us": float(times_us.mean()),
            "per_sample_us": float(times_us.mean() / B),
            "std_us": float(times_us.std()),
        }

    del net_dev
    if device == "cuda":
        import torch as _t
        _t.cuda.empty_cache()
    return results


def profile_cudagraph(net, n, num_iters=NUM_ITERS, warmup=WARMUP):
    import torch
    dev = torch.device("cuda")
    net_g = copy.deepcopy(net).to(dev).eval()

    static_ae = torch.zeros(1, n, 6, dtype=torch.float32, device=dev)
    static_pt = torch.zeros(1, n, dtype=torch.int32, device=dev)
    static_td = {"agent_embeddings": static_ae, "pad_tokens": static_pt}

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(5):
                net_g(static_td)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            net_g(static_td)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times_us = np.array(times) * 1e6
    del net_g, g
    torch.cuda.empty_cache()
    return {
        "per_sample_us": float(times_us.mean()),
        "std_us": float(times_us.std()),
    }


def profile_env_step(n, num_iters=NUM_ITERS):
    from env.envs import LazyAgentsCentralized
    config = {
        "num_agents_max": n, "num_agents_min": n,
        "speed": 15, "predefined_distance": 60,
        "std_pos_converged": 45, "std_vel_converged": 0.1,
        "std_pos_rate_converged": 0.1, "std_vel_rate_converged": 0.2,
        "max_time_step": 2000, "incomplete_episode_penalty": 0,
        "normalize_obs": True, "use_fixed_horizon": False,
        "_use_fixed_lazy_idx": True, "use_preprocessed_obs": True,
        "use_mlp_settings": False, "get_state_hist": False,
        "use_L2_norm": False, "use_heuristics": False,
    }
    env = LazyAgentsCentralized(config)
    env.seed(42)
    env.reset()
    action = np.ones(n, dtype=np.float32)

    # Warmup
    for _ in range(min(50, num_iters)):
        _, _, done, _ = env.step(action)
        if done:
            env.reset()

    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        _, _, done, _ = env.step(action)
        times.append(time.perf_counter() - t0)
        if done:
            env.reset()

    times_us = np.array(times) * 1e6
    return {"mean_us": float(times_us.mean()), "std_us": float(times_us.std())}


def main():
    import torch
    torch.set_num_threads(4)

    print("Loading model...")
    net = load_policy_network()
    net.cpu().eval()

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA — GPU columns will be skipped")

    batch_sizes_gpu = (1, 32, 64)

    # Header
    print(f"\n{'n':>6} | {'CPU B=1':>10} | ", end="")
    if has_cuda:
        print(f"{'GPU B=1':>10} | {'CUDAg B=1':>10} | {'GPU B=32':>10} | {'GPU B=64':>10} | ", end="")
    print(f"{'env.step':>10} |")
    print("-" * (80 if not has_cuda else 120))

    all_results = {}
    for n in AGENT_COUNTS:
        row = {"n": n}

        # CPU B=1
        cpu_res = profile_forward(net, n, "cpu", batch_sizes=(1,))
        row["cpu_b1_us"] = cpu_res[1]["per_sample_us"]

        if has_cuda:
            # GPU B=1, B=32, B=64
            gpu_res = profile_forward(net, n, "cuda", batch_sizes=batch_sizes_gpu)
            row["gpu_b1_us"] = gpu_res[1]["per_sample_us"]
            row["gpu_b32_us"] = gpu_res[32]["per_sample_us"]
            row["gpu_b64_us"] = gpu_res[64]["per_sample_us"]

            # CUDA graph B=1
            cg_res = profile_cudagraph(net, n)
            row["cudagraph_b1_us"] = cg_res["per_sample_us"]

        # env.step
        env_res = profile_env_step(n, num_iters=min(NUM_ITERS, max(50, 1000 // max(1, n // 64))))
        row["env_step_us"] = env_res["mean_us"]

        # Print
        print(f"{n:>6} | {row['cpu_b1_us']:>8.0f}us | ", end="")
        if has_cuda:
            print(f"{row['gpu_b1_us']:>8.0f}us | {row['cudagraph_b1_us']:>8.0f}us | "
                  f"{row['gpu_b32_us']:>8.0f}us | {row['gpu_b64_us']:>8.0f}us | ", end="")
        print(f"{row['env_step_us']:>8.0f}us |")

        all_results[n] = row

    # Summary
    if has_cuda:
        print("\n=== CPU vs GPU crossover (B=1) ===")
        for n, r in all_results.items():
            faster = "GPU" if r["gpu_b1_us"] < r["cpu_b1_us"] else "CPU"
            ratio = r["cpu_b1_us"] / r["gpu_b1_us"] if faster == "GPU" else r["gpu_b1_us"] / r["cpu_b1_us"]
            print(f"  n={n:>4}: {faster} wins ({ratio:.1f}x)")

        print("\n=== RL forward vs env.step (GPU B=64 per-sample) ===")
        for n, r in all_results.items():
            rl = r["gpu_b64_us"]
            env = r["env_step_us"]
            pct_rl = rl / (rl + env) * 100
            print(f"  n={n:>4}: RL={rl:.0f}us  env={env:.0f}us  → RL is {pct_rl:.0f}% of total step")

    # Save
    import json
    out_path = os.path.join(os.path.dirname(__file__), "results", "rl_speed_profile.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
