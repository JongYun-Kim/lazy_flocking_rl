"""
CUDA-graph-optimised benchmark for the trained Transformer policy.

Standalone from ``benchmark.py``: this file only touches the
``RL_pure_forward``-style path (no RLlib wrapping) and measures
how much the per-step GPU cost at B=1 can be pushed down with
``torch.cuda.CUDAGraph`` capture + replay, optionally combined with
half- or bfloat16-precision weights.

Same methodology as ``benchmark.py`` for comparability:
- 200 warmup steps per variant (not counted)
- 3 rollouts × 2000 timed steps = 6000 samples per variant
- Same env (LazyAgentsCentralized, 20 agents) and same seeding
  (base_seed=4242; rollout i uses seed 4242+i)
- Only the action-selection call is timed. For the graph variants the
  timed block is strictly ``g.replay()`` with ``torch.cuda.synchronize``
  brackets; numpy->device obs copy and tensor->action post-processing are
  *outside* the timer.

Variants
--------
- ``baseline_gpu_fp32``    -- no graph, identical to the existing
                              ``RL_pure_forward`` GPU row; included as a
                              reference on the same hardware state.
- ``cudagraph_fp32``       -- graph capture/replay, fp32 weights.
- ``cudagraph_fp16``       -- graph capture/replay, half-precision weights.
- ``cudagraph_bf16``       -- graph capture/replay, bfloat16 weights (skipped
                              if the GPU reports no bf16 support).

Usage
-----
    python -m experiments.compute_benchmark.benchmark_cudagraph \
        --output experiments/compute_benchmark/results/benchmark_cudagraph.json
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


# --------------------------------------------------------------------- #
# Forward-pass backends
# --------------------------------------------------------------------- #

class _BaselineForward:
    """No-graph baseline: identical to benchmark.py's RL_pure_forward path."""

    name = "baseline_gpu_fp32"

    def __init__(self, policy_network, num_agents: int, dtype, device):
        import torch

        self.net = policy_network
        self.device = device
        self.dtype = dtype
        self.num_agents = num_agents
        self.half_dim = None  # action size, resolved on first call
        self._last_out = None

    def prepare(self, obs):
        import torch

        ae = torch.as_tensor(
            obs["agent_embeddings"], dtype=self.dtype, device=self.device
        ).unsqueeze(0)
        pt = torch.as_tensor(
            obs["pad_tokens"], dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        self._td = {"agent_embeddings": ae, "pad_tokens": pt}

    def timed_forward(self):
        import torch

        with torch.no_grad():
            out, _, _ = self.net(self._td)
        self._last_out = out

    def finalise(self):
        if self.half_dim is None:
            self.half_dim = self._last_out.shape[-1] // 2
        mean = self._last_out[0, : self.half_dim].float().clamp(0.0, 1.0)
        return mean.detach().cpu().numpy()


class _CUDAGraphForward:
    """Graph-captured forward. Inputs live in static buffers; replay only."""

    def __init__(self, policy_network, num_agents: int, dtype, device, name: str):
        import torch

        self.name = name
        self.net = policy_network
        self.device = device
        self.dtype = dtype
        self.num_agents = num_agents

        self.static_ae = torch.zeros(1, num_agents, 6, dtype=dtype, device=device)
        self.static_pt = torch.zeros(1, num_agents, dtype=torch.int32, device=device)
        self.static_obs = {"agent_embeddings": self.static_ae, "pad_tokens": self.static_pt}

        # Warmup on a side stream, then join before capture -- recommended
        # pattern for torch.cuda.CUDAGraph. Using the default stream also
        # works for a model this small, but this is the safer dance.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(5):
                    out, _, _ = self.net(self.static_obs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(self.graph):
                out, _, _ = self.net(self.static_obs)
        self.static_out = out  # filled on each replay
        torch.cuda.synchronize()

        self.half_dim = self.static_out.shape[-1] // 2

    def prepare(self, obs):
        import torch

        ae_src = torch.from_numpy(obs["agent_embeddings"]).unsqueeze(0).to(self.dtype)
        pt_src = torch.from_numpy(obs["pad_tokens"].astype(np.int32)).unsqueeze(0)
        self.static_ae.copy_(ae_src)
        self.static_pt.copy_(pt_src)

    def timed_forward(self):
        self.graph.replay()

    def finalise(self):
        mean = self.static_out[0, : self.half_dim].float().clamp(0.0, 1.0)
        return mean.detach().cpu().numpy()


# --------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------- #

def _sanity_check(baseline, graph_fwd, obs, atol: float, rtol: float) -> dict:
    """Compare baseline and graph-replay outputs on one obs; report max abs/rel error."""

    import torch

    baseline.prepare(obs)
    baseline.timed_forward()
    ref = baseline._last_out[0].detach().float().cpu().numpy()

    graph_fwd.prepare(obs)
    graph_fwd.timed_forward()
    torch.cuda.synchronize()
    got = graph_fwd.static_out[0].detach().float().cpu().numpy()

    max_abs = float(np.abs(ref - got).max())
    denom = np.maximum(np.abs(ref), 1e-6)
    max_rel = float((np.abs(ref - got) / denom).max())
    ok = (max_abs <= atol) or (max_rel <= rtol)
    return {"ok": bool(ok), "max_abs": max_abs, "max_rel": max_rel, "atol": atol, "rtol": rtol}


# --------------------------------------------------------------------- #
# Timing loop (same shape as benchmark.py)
# --------------------------------------------------------------------- #

def time_variant(
    forwarder,
    env,
    *,
    num_rollouts: int,
    steps_per_rollout: int,
    warmup_steps: int,
    base_seed: int,
) -> Dict[str, object]:
    import torch

    def one_step(obs):
        forwarder.prepare(obs)  # outside timer: H2D copy / tensor build
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        forwarder.timed_forward()  # kernel launches + (graph replay for graph variants)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        action = forwarder.finalise()  # outside timer: GPU->CPU, numpy
        return t1 - t0, action

    # Warmup
    env.seed(base_seed - 1)
    obs = env.reset()
    for _ in range(warmup_steps):
        _, action = one_step(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    # Timed rollouts
    total = num_rollouts * steps_per_rollout
    times_ns = np.empty(total, dtype=np.int64)
    idx = 0
    resets = 0
    for rollout_idx in range(num_rollouts):
        env.seed(base_seed + rollout_idx)
        obs = env.reset()
        for _ in range(steps_per_rollout):
            dt_ns, action = one_step(obs)
            times_ns[idx] = dt_ns
            idx += 1
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
                resets += 1

    times_us = times_ns / 1000.0
    return {
        "num_samples": int(total),
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


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #

def _patch_get_context_for_dtype(net, dtype) -> int:
    """Patch ``LazinessAllocator.get_context_node`` to not force-cast to fp32.

    The checked-in method computes the mean count via ``.float()``, which
    promotes the resulting embedding back to fp32 and then trips the next
    LayerNorm (which has fp16 weights). Replace ``.float()`` with ``.to(dtype)``
    so the whole path stays in the target dtype.
    """

    import torch

    def patched_get_context_node(self, embeddings, pad_tokens, use_embeddings_mask=True, debug=False):
        if use_embeddings_mask:
            mask = pad_tokens.unsqueeze(-1).expand_as(embeddings)
            embeddings_masked = torch.where(mask == 1, embeddings, torch.zeros_like(embeddings))
            embeddings_sum = torch.sum(embeddings_masked, dim=1, keepdim=True)
            embeddings_count = torch.sum((mask == 0), dim=1, keepdim=True).to(embeddings.dtype)
            return embeddings_sum / embeddings_count
        return torch.mean(embeddings, dim=1, keepdim=True)

    net.get_context_node = patched_get_context_node.__get__(net, type(net))
    return 1


def _patch_mha_for_dtype(module, dtype) -> int:
    """Monkey-patch each MultiHeadAttentionLayer to use a dtype-safe masked_fill value.

    The project's attention uses ``masked_fill(mask == 0, -1e9)`` which overflows
    fp16 (max ~65504). Replace ``-1e9`` with the largest safely-representable
    negative in the target dtype, without editing the model source.

    Returns the number of patched modules.
    """

    import math
    import torch
    import torch.nn.functional as F
    from models.transformer_modules.multi_head_attention_layer import MultiHeadAttentionLayer

    if dtype == torch.float16:
        neg_large = -6.0e4  # fp16 finfo.max ~= 6.55e4
    else:
        neg_large = -1.0e9

    def patched(self, query, key, value, mask):
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, neg_large)
        attention_prob = F.softmax(attention_score, dim=-1)
        attention_prob = self.dropout(attention_prob)
        return torch.matmul(attention_prob, value)

    count = 0
    for m in module.modules():
        if isinstance(m, MultiHeadAttentionLayer):
            m.calculate_attention = patched.__get__(m, type(m))
            count += 1
    return count


def _copy_model(policy, dtype):
    """Return a deep-copied policy_network on CUDA with the requested dtype.

    For half-precision dtypes the attention mask fill value is patched to a
    representable negative so softmax doesn't blow up.
    """

    import copy
    import torch

    net = copy.deepcopy(policy.model.policy_network).to(dtype).to("cuda").eval()
    if dtype in (torch.float16,):
        n_patched = _patch_mha_for_dtype(net, dtype)
        print(f"  patched {n_patched} MHA layer(s) for dtype={dtype}")
    if dtype in (torch.float16, torch.bfloat16):
        _patch_get_context_for_dtype(net, dtype)
        print("  patched get_context_node to avoid fp32 up-cast")
    return net


def run(args: argparse.Namespace) -> Dict[str, object]:
    import torch
    from ray.rllib.models import ModelCatalog
    from ray.rllib.policy.policy import Policy
    from ray.tune.registry import register_env

    from env.envs import LazyAgentsCentralized
    from models.lazy_allocator import MyRLlibTorchWrapper

    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(args.torch_threads)
        except RuntimeError:
            pass

    torch.backends.cudnn.benchmark = True  # small wins on matmul-heavy paths

    ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
    register_env("lazy_env", lambda cfg: LazyAgentsCentralized(cfg))

    env_config = build_env_config(args.num_agents, args.max_time_step)
    env = LazyAgentsCentralized(env_config)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    print(f"Loading policy from {args.checkpoint}")
    policy = Policy.from_checkpoint(args.checkpoint)
    policy.model.eval()
    # Keep the original policy.model on CPU; for each variant we deep-copy the
    # policy_network subtree onto CUDA with the right dtype.
    policy.model.to("cpu")

    # What variants to try
    variants_spec = []
    variants_spec.append(("baseline_gpu_fp32", "baseline", torch.float32))
    variants_spec.append(("cudagraph_fp32", "graph", torch.float32))
    variants_spec.append(("cudagraph_fp16", "graph", torch.float16))
    if torch.cuda.is_bf16_supported():
        variants_spec.append(("cudagraph_bf16", "graph", torch.bfloat16))
    else:
        print("bf16 not supported on this GPU; skipping cudagraph_bf16.")

    # Baseline forward (kept fp32, no graph) -- also used to validate numerics
    baseline_net = _copy_model(policy, torch.float32)
    baseline = _BaselineForward(baseline_net, args.num_agents, torch.float32, "cuda")

    # Run an obs through env.reset once for sanity checks
    env.seed(0)
    sanity_obs = env.reset()

    results: Dict[str, object] = {}
    numerics: Dict[str, object] = {}
    for name, kind, dtype in variants_spec:
        print(f"\n=== {name} ===")

        if kind == "baseline":
            forwarder = baseline
            numerics[name] = {"ok": True, "max_abs": 0.0, "max_rel": 0.0, "note": "reference"}
        else:
            net = _copy_model(policy, dtype)
            forwarder = _CUDAGraphForward(net, args.num_agents, dtype, "cuda", name=name)
            # Numerics sanity: compare one forward vs baseline
            atol = 1e-4 if dtype == torch.float32 else (5e-2 if dtype == torch.float16 else 5e-2)
            rtol = 1e-4 if dtype == torch.float32 else (5e-2 if dtype == torch.float16 else 5e-2)
            s = _sanity_check(baseline, forwarder, sanity_obs, atol=atol, rtol=rtol)
            numerics[name] = s
            print(f"  numerics: max_abs={s['max_abs']:.2e}  max_rel={s['max_rel']:.2e}  ok={s['ok']}")

        t0 = time.time()
        summary = time_variant(
            forwarder,
            env,
            num_rollouts=args.num_rollouts,
            steps_per_rollout=args.steps_per_rollout,
            warmup_steps=args.warmup_steps,
            base_seed=args.base_seed,
        )
        summary["wall_clock_s"] = time.time() - t0
        results[name] = summary
        print(
            f"  mean={summary['mean_us']:.2f}us  median={summary['median_us']:.2f}us  "
            f"std={summary['std_us']:.2f}us  p95={summary['p95_us']:.2f}us  "
            f"p99={summary['p99_us']:.2f}us  min={summary['min_us']:.2f}us  "
            f"max={summary['max_us']:.2f}us  n={summary['num_samples']}"
        )

        # Free the variant-local copy before building the next one (keeps peak
        # GPU usage low on the 4x-Ada host).
        if kind != "baseline":
            del forwarder
            torch.cuda.empty_cache()

    host_info = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available() else None,
        "num_cpus": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }

    return {
        "args": vars(args),
        "env_config": env_config,
        "host_info": host_info,
        "numerics": numerics,
        "algorithms": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--max_time_step", type=int, default=2000)
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--steps_per_rollout", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--base_seed", type=int, default=4242)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(THIS_DIR, "results", "benchmark_cudagraph.json"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    doc = run(args)
    with open(args.output, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
