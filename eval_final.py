# Final evaluation: All MLP ablations (v2 + shared) vs Transformer vs ACS
# Proper eval: LazyAgentsCentralized, use_fixed_horizon=False, 100 seeded episodes
import numpy as np
import copy
import glob
import os
from env.envs import LazyAgentsCentralized
from models.lazy_allocator import MyRLlibTorchWrapper, MyMLPModel
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from datetime import datetime

max_time_step = 1000
env_config_base = {
    "num_agents_max": 20, "num_agents_min": 20,
    "speed": 15, "max_turn_rate": 8/15,
    "inter_agent_strength": 5, "communication_decay_rate": 1/3,
    "bonding_strength": 1, "initial_position_bound": 250,
    "predefined_distance": 60,
    "std_pos_converged": 45, "std_vel_converged": 0.1,
    "std_pos_rate_converged": 0.1, "std_vel_rate_converged": 0.2,
    "max_time_step": max_time_step, "incomplete_episode_penalty": 0,
    "normalize_obs": True, "use_fixed_horizon": False,
    "use_L2_norm": False, "use_heuristics": True,
    "_use_fixed_lazy_idx": True, "use_preprocessed_obs": True,
    "get_state_hist": False,
}

V2_DIR = "/home/flocking/ray_results/mlp_ablation_v2_fixed_eval"
SH_DIR = "/home/flocking/ray_results/mlp_ablation_v2_shared"
v2_dirs = sorted(glob.glob(f"{V2_DIR}/PPO_*"))
sh_dirs = sorted(glob.glob(f"{SH_DIR}/PPO_*"))

MODELS = {
    "Transformer": {
        "ckpt": "bk/bk_082623/PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47/checkpoint_000074/policies/default_policy",
        "mlp": False, "share": True, "params_label": "875K",
    },
    # v2: share_layers=False
    "MLP_base": {
        "ckpt": os.path.join(v2_dirs[0], "checkpoint_000188/policies/default_policy"),
        "mlp": True, "share": False, "params_label": "292K",
    },
    "MLP_determ": {
        "ckpt": os.path.join(v2_dirs[1], "checkpoint_000204/policies/default_policy"),
        "mlp": True, "share": False, "params_label": "292K",
    },
    "MLP_hi_lr": {
        "ckpt": os.path.join(v2_dirs[2], "checkpoint_000197/policies/default_policy"),
        "mlp": True, "share": False, "params_label": "292K",
    },
    "MLP_lr_sched": {
        "ckpt": os.path.join(v2_dirs[3], "checkpoint_000234/policies/default_policy"),
        "mlp": True, "share": False, "params_label": "292K",
    },
    "MLP_entropy": {
        "ckpt": os.path.join(v2_dirs[4], "checkpoint_000257/policies/default_policy"),
        "mlp": True, "share": False, "params_label": "292K",
    },
    "MLP_wide": {
        "ckpt": os.path.join(v2_dirs[5], "checkpoint_000291/policies/default_policy"),
        "mlp": True, "share": False, "params_label": "1044K",
    },
    "MLP_wide_deep": {
        "ckpt": os.path.join(v2_dirs[6], "checkpoint_000208/policies/default_policy"),
        "mlp": True, "share": False, "params_label": "1569K",
    },
    # shared: share_layers=True
    "MLP_sh_base": {
        "ckpt": os.path.join(sh_dirs[0], "checkpoint_000300/policies/default_policy"),
        "mlp": True, "share": True, "params_label": "168K",
    },
    "MLP_sh_wide": {
        "ckpt": os.path.join(sh_dirs[1], "checkpoint_000263/policies/default_policy"),
        "mlp": True, "share": True, "params_label": "598K",
    },
    "MLP_sh_wd": {
        "ckpt": os.path.join(sh_dirs[2], "checkpoint_000200/policies/default_policy"),
        "mlp": True, "share": True, "params_label": "836K",
    },
}


def run_episode(policy, env, max_steps):
    obs = env.reset()
    reward_sum = 0.0
    length = max_steps
    for t in range(max_steps):
        action = policy.compute_single_action(obs, explore=False)[0]
        action = np.clip(action, 0, 1)
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        if done:
            length = t + 1
            break
    return reward_sum, length


def run_acs_episode(env, max_steps):
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
    return reward_sum, length


if __name__ == "__main__":
    ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
    ModelCatalog.register_custom_model("custom_model_mlp", MyMLPModel)
    register_env("lazy_env", lambda cfg: LazyAgentsCentralized(cfg))

    # Load policies and count params
    policies = {}
    param_counts = {}
    for name, info in MODELS.items():
        print(f"Loading {name}...")
        p = Policy.from_checkpoint(info["ckpt"])
        p.model.eval()
        policies[name] = p
        param_counts[name] = sum(pr.numel() for pr in p.model.parameters())

    env_tf = LazyAgentsCentralized({**env_config_base, "use_mlp_settings": False})
    env_mlp = LazyAgentsCentralized({**env_config_base, "use_mlp_settings": True})

    num_episodes = 100
    seeds = np.arange(1, 1 + num_episodes)
    all_names = list(MODELS.keys()) + ["ACS"]
    results = {n: {"rewards": [], "lengths": []} for n in all_names}

    print(f"\nRunning {num_episodes} episodes for {len(all_names)} models...")
    start = datetime.now()

    for ep in range(num_episodes):
        seed = int(seeds[ep])
        env_tf.seed(seed); env_tf.reset()
        env_mlp.seed(seed); env_mlp.reset()

        for name, info in MODELS.items():
            base_env = env_mlp if info["mlp"] else env_tf
            env_copy = copy.deepcopy(base_env)
            r, l = run_episode(policies[name], env_copy, max_time_step)
            results[name]["rewards"].append(r)
            results[name]["lengths"].append(l)

        env_copy = copy.deepcopy(env_tf)
        r, l = run_acs_episode(env_copy, max_time_step)
        results["ACS"]["rewards"].append(r)
        results["ACS"]["lengths"].append(l)

        if (ep + 1) % 20 == 0:
            elapsed = (datetime.now() - start).total_seconds()
            eta = elapsed / (ep + 1) * (num_episodes - ep - 1)
            print(f"  {ep+1}/{num_episodes} done  (ETA: {eta:.0f}s)")

    print(f"\nDone in {datetime.now() - start}")

    # --- Report ---
    print("\n" + "=" * 100)
    print("FINAL EVALUATION — MLP Ablation Study (v2 + shared)")
    print(f"Environment: LazyAgentsCentralized | use_fixed_horizon=False | Episodes: {num_episodes}")
    print("=" * 100)

    tf_r = np.array(results["Transformer"]["rewards"])

    header = f"{'Model':<18s} {'Shared':>6s} {'Params':>8s} {'Reward':>8s} {'Gap':>8s} {'±std':>7s} {'Conv%':>6s} {'ConvLen':>8s} {'EpLen':>8s}"
    print(header)
    print("-" * 100)

    for name in all_names:
        rews = np.array(results[name]["rewards"])
        lens = np.array(results[name]["lengths"])
        conv = lens < max_time_step
        conv_len = f"{lens[conv].mean():.0f}" if conv.any() else "—"
        gap = f"{rews.mean() - tf_r.mean():+.1f}" if name != "Transformer" else "—"

        if name in MODELS:
            shared = "Yes" if MODELS[name]["share"] else "No"
            params = f"{param_counts[name]:,}"
        else:
            shared = "—"
            params = "—"

        print(f"{name:<18s} {shared:>6s} {params:>8s} {rews.mean():>8.1f} {gap:>8s} {rews.std():>7.1f} "
              f"{100*conv.mean():>5.0f}% {conv_len:>8s} {lens.mean():>8.1f}")

    print("=" * 100)
