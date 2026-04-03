# Evaluation: MLP (Exp 0, checkpoint 31) vs Transformer (checkpoint 74)
# Environment: LazyAgentsCentralized
# Metrics: episode reward, episode length (convergence time)
import numpy as np
import copy
from env.envs import LazyAgentsCentralized
from models.lazy_allocator import MyRLlibTorchWrapper, MyMLPModel
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from datetime import datetime

# --- Env config (matching existing eval methodology) ---
num_agents_max = 20
num_agents_min = 20
max_time_step = 1000

env_config_base = {
    "num_agents_max": num_agents_max,
    "num_agents_min": num_agents_min,
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

env_config_transformer = {**env_config_base, "use_mlp_settings": False}
env_config_mlp = {**env_config_base, "use_mlp_settings": True}


def evaluate_policy(policy, env, max_steps):
    """Run one episode and return (reward_sum, episode_length)."""
    obs = env.reset()
    reward_sum = 0.0
    episode_length = max_steps
    for t in range(max_steps):
        action = policy.compute_single_action(obs, explore=False)[0]
        action = np.clip(action, 0, 1)
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        if done:
            episode_length = t + 1
            break
    return reward_sum, episode_length


def evaluate_acs(env, max_steps):
    """Run one ACS (all-active) episode."""
    obs = env.reset()
    reward_sum = 0.0
    episode_length = max_steps
    for t in range(max_steps):
        action = env.get_fully_active_action()
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        if done:
            episode_length = t + 1
            break
    return reward_sum, episode_length


if __name__ == "__main__":
    # Register models
    ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
    ModelCatalog.register_custom_model("custom_model_mlp", MyMLPModel)
    register_env("lazy_env", lambda cfg: LazyAgentsCentralized(cfg))

    # Load policies
    tf_checkpoint = (
        "bk/bk_082623/"
        "PPO_lazy_env_36f5d_00000_0_use_deterministic_action_dist=True_2023-08-26_12-53-47/"
        "checkpoint_000074/policies/default_policy"
    )
    mlp_checkpoint = (
        "/home/flocking/ray_results/mlp_ablation/"
        "PPO_lazy_env_train_d494b_00000_0_2026-04-02_03-34-18/"
        "checkpoint_000031/policies/default_policy"
    )

    print("Loading Transformer policy...")
    tf_policy = Policy.from_checkpoint(tf_checkpoint)
    tf_policy.model.eval()

    print("Loading MLP policy...")
    mlp_policy = Policy.from_checkpoint(mlp_checkpoint)
    mlp_policy.model.eval()

    # Create envs
    env_tf = LazyAgentsCentralized(env_config_transformer)
    env_mlp = LazyAgentsCentralized(env_config_mlp)
    env_acs = LazyAgentsCentralized(env_config_transformer)

    # Evaluation
    num_episodes = 100
    seeds = np.arange(1, 1 + num_episodes)

    results = {
        "Transformer": {"rewards": [], "lengths": []},
        "MLP": {"rewards": [], "lengths": []},
        "ACS": {"rewards": [], "lengths": []},
    }

    print(f"\nRunning {num_episodes} episodes...")
    start = datetime.now()

    for ep in range(num_episodes):
        seed = int(seeds[ep])

        # Same seed for all — reset base envs then deepcopy
        env_tf.seed(seed)
        env_tf.reset()
        env_mlp.seed(seed)
        env_mlp.reset()
        env_acs.seed(seed)
        env_acs.reset()

        # Transformer
        env_t = copy.deepcopy(env_tf)
        r, l = evaluate_policy(tf_policy, env_t, max_time_step)
        results["Transformer"]["rewards"].append(r)
        results["Transformer"]["lengths"].append(l)

        # MLP
        env_m = copy.deepcopy(env_mlp)
        r, l = evaluate_policy(mlp_policy, env_m, max_time_step)
        results["MLP"]["rewards"].append(r)
        results["MLP"]["lengths"].append(l)

        # ACS baseline
        env_a = copy.deepcopy(env_acs)
        r, l = evaluate_acs(env_a, max_time_step)
        results["ACS"]["rewards"].append(r)
        results["ACS"]["lengths"].append(l)

        if (ep + 1) % 20 == 0:
            print(f"  {ep+1}/{num_episodes} done")

    elapsed = datetime.now() - start
    print(f"\nDone in {elapsed}")

    # --- Report ---
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS: MLP vs Transformer vs ACS (all-active)")
    print(f"Environment: LazyAgentsCentralized | Episodes: {num_episodes}")
    print("=" * 70)

    for name in ["Transformer", "MLP", "ACS"]:
        rews = np.array(results[name]["rewards"])
        lens = np.array(results[name]["lengths"])
        converged = lens < max_time_step
        print(f"\n--- {name} ---")
        print(f"  Episode Reward:  mean={rews.mean():.2f}  std={rews.std():.2f}  "
              f"min={rews.min():.2f}  max={rews.max():.2f}")
        print(f"  Episode Length:  mean={lens.mean():.1f}  std={lens.std():.1f}  "
              f"min={lens.min()}  max={lens.max()}")
        print(f"  Convergence:     {converged.sum()}/{num_episodes} "
              f"({100*converged.mean():.0f}%)")
        if converged.any():
            print(f"  Conv. Length:    mean={lens[converged].mean():.1f}  "
                  f"std={lens[converged].std():.1f}")

    # Summary comparison
    tf_r = np.array(results["Transformer"]["rewards"])
    mlp_r = np.array(results["MLP"]["rewards"])
    tf_l = np.array(results["Transformer"]["lengths"])
    mlp_l = np.array(results["MLP"]["lengths"])
    print(f"\n--- MLP vs Transformer ---")
    print(f"  Reward gap:  {mlp_r.mean() - tf_r.mean():+.2f} (MLP - TF)")
    print(f"  Length gap:  {mlp_l.mean() - tf_l.mean():+.1f} steps (MLP - TF)")
    print(f"  MLP wins on reward: {(mlp_r > tf_r).sum()}/{num_episodes}")
