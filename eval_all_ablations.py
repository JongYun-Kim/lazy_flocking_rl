# Full evaluation: All MLP ablations vs Transformer vs ACS
# Environment: LazyAgentsCentralized, 100 episodes, seeded
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

# --- Env config (matching existing eval methodology) ---
max_time_step = 1000
env_config_base = {
    "num_agents_max": 20,
    "num_agents_min": 20,
    "speed": 15,
    "max_turn_rate": 8/15,
    "inter_agent_strength": 5,
    "communication_decay_rate": 1/3,
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

# --- Checkpoints to evaluate ---
ABLATION_BASE = "/home/flocking/ray_results/mlp_ablation_v2"
trial_dirs = sorted(glob.glob(f"{ABLATION_BASE}/PPO_lazy_env_train_3c3fb_*"))

MODELS = {
    "Transformer": {
        "checkpoint": (
            "bk/bk_082623/"
            "PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47/"
            "checkpoint_000074/policies/default_policy"
        ),
        "mlp": False,
    },
    "MLP_baseline": {
        "checkpoint": (
            "/home/flocking/ray_results/mlp_ablation/"
            "PPO_lazy_env_train_d494b_00000_0_2026-04-02_03-34-18/"
            "checkpoint_000031/policies/default_policy"
        ),
        "mlp": True,
    },
    "MLP_deterministic": {
        "checkpoint": os.path.join(trial_dirs[0], "checkpoint_000272/policies/default_policy"),
        "mlp": True,
    },
    "MLP_higher_lr": {
        "checkpoint": os.path.join(trial_dirs[1], "checkpoint_000226/policies/default_policy"),
        "mlp": True,
    },
    "MLP_lr_schedule": {
        "checkpoint": os.path.join(trial_dirs[2], "checkpoint_000079/policies/default_policy"),
        "mlp": True,
    },
    "MLP_entropy": {
        "checkpoint": os.path.join(trial_dirs[3], "checkpoint_000032/policies/default_policy"),
        "mlp": True,
    },
    "MLP_wide": {
        "checkpoint": os.path.join(trial_dirs[4], "checkpoint_000210/policies/default_policy"),
        "mlp": True,
    },
    "MLP_wide_deep": {
        "checkpoint": os.path.join(trial_dirs[5], "checkpoint_000250/policies/default_policy"),
        "mlp": True,
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

    # Load all policies
    policies = {}
    for name, info in MODELS.items():
        print(f"Loading {name}...")
        p = Policy.from_checkpoint(info["checkpoint"])
        p.model.eval()
        policies[name] = p

    # Create envs
    env_tf = LazyAgentsCentralized({**env_config_base, "use_mlp_settings": False})
    env_mlp = LazyAgentsCentralized({**env_config_base, "use_mlp_settings": True})

    num_episodes = 100
    seeds = np.arange(1, 1 + num_episodes)

    # Results storage
    all_names = list(MODELS.keys()) + ["ACS"]
    results = {n: {"rewards": [], "lengths": []} for n in all_names}

    print(f"\nRunning {num_episodes} episodes for {len(all_names)} models...")
    start = datetime.now()

    for ep in range(num_episodes):
        seed = int(seeds[ep])

        # Seed and reset base envs
        env_tf.seed(seed); env_tf.reset()
        env_mlp.seed(seed); env_mlp.reset()

        for name, info in MODELS.items():
            base_env = env_mlp if info["mlp"] else env_tf
            env_copy = copy.deepcopy(base_env)
            r, l = run_episode(policies[name], env_copy, max_time_step)
            results[name]["rewards"].append(r)
            results[name]["lengths"].append(l)

        # ACS
        env_copy = copy.deepcopy(env_tf)
        r, l = run_acs_episode(env_copy, max_time_step)
        results["ACS"]["rewards"].append(r)
        results["ACS"]["lengths"].append(l)

        if (ep + 1) % 20 == 0:
            elapsed = (datetime.now() - start).total_seconds()
            eta = elapsed / (ep + 1) * (num_episodes - ep - 1)
            print(f"  {ep+1}/{num_episodes} done  (ETA: {eta:.0f}s)")

    elapsed = datetime.now() - start
    print(f"\nDone in {elapsed}")

    # --- Report ---
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS — MLP Ablation Study")
    print(f"Environment: LazyAgentsCentralized | Episodes: {num_episodes} | Seeded: 1-{num_episodes}")
    print("=" * 80)

    header = f"{'Model':<20s} {'Reward':>12s} {'±std':>8s} {'Length':>10s} {'±std':>8s} {'Conv%':>7s} {'ConvLen':>10s}"
    print(header)
    print("-" * 80)

    tf_rewards = np.array(results["Transformer"]["rewards"])

    for name in all_names:
        rews = np.array(results[name]["rewards"])
        lens = np.array(results[name]["lengths"])
        conv = lens < max_time_step
        conv_len = lens[conv].mean() if conv.any() else float('nan')
        conv_len_std = lens[conv].std() if conv.any() else float('nan')

        gap = f"({rews.mean() - tf_rewards.mean():+.1f})" if name != "Transformer" else ""
        print(f"{name:<20s} {rews.mean():>8.1f} {gap:>4s} {rews.std():>7.1f} "
              f"{lens.mean():>9.1f} {lens.std():>7.1f} "
              f"{100*conv.mean():>6.0f}% {conv_len:>9.1f}")

    print("=" * 80)
