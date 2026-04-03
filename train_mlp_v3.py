# MLP Ablation Study v3 — Final: 12 trials, 6 at a time
# Changes from v2:
#   - Eval env uses L1 (use_L2_norm=False) matching test eval
#   - 2 new deterministic trials with Transformer-matched param counts
#   - #5 wide and #9 shared_wide_deep also param-matched to Transformer
#   - Worker config matches Transformer training exactly
#   - Ray CPU capped at 48 → 6 trials run simultaneously
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from env.envs import LazyAgentsCentralized, LazyAgentsCentralizedPendReward
from models.lazy_allocator import MyMLPModel
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, episode, **kwargs):
        episode.user_data["L1_reward_sum"] = 0
        episode.user_data["L2_reward_sum"] = 0

    def on_episode_step(self, worker, episode, **kwargs):
        from_infos = episode.last_info_for()["original_rewards"]
        episode.user_data["L1_reward_sum"] += from_infos[0]
        episode.user_data["L2_reward_sum"] += from_infos[1]

    def on_episode_end(self, worker, episode, **kwargs):
        episode.custom_metrics["episode_L1_reward_sum"] = episode.user_data["L1_reward_sum"]
        episode.custom_metrics["episode_L2_reward_sum"] = episode.user_data["L2_reward_sum"]


# --- Env configs ---
ENV_CONFIG = {
    "num_agents_max": 20, "num_agents_min": 20,
    "speed": 15, "predefined_distance": 60,
    "std_pos_converged": 45, "std_vel_converged": 0.1,
    "std_pos_rate_converged": 0.1, "std_vel_rate_converged": 0.2,
    "max_time_step": 1000, "incomplete_episode_penalty": 0,
    "normalize_obs": True, "use_fixed_horizon": True,
    "use_L2_norm": True,  # Training env: L2
    "auto_step": False, "use_custom_ray": False,
    "use_preprocessed_obs": True, "use_mlp_settings": True,
    "w_vel": 0.2, "w_control": 0.02,
}

# Eval env: use_fixed_horizon=False + L1 (use_L2_norm=False)
EVAL_ENV_CONFIG = {**ENV_CONFIG, "use_fixed_horizon": False, "use_L2_norm": False}

# --- Schedules ---
LR_SCHEDULE = [[0, 1e-4], [1_000_000, 5e-5], [2_500_000, 1e-5], [4_200_000, 5e-6]]
ENTROPY_SCHEDULE = [[0, 0.01], [1_000_000, 0.005], [2_500_000, 0.001], [4_200_000, 0.0]]

# --- MLP model config helpers ---
def mlp_cfg(fc, val_fc, share, fixed_log_std=None):
    cfg = {
        "fc_sizes": fc, "fc_activation": "relu",
        "value_fc_sizes": val_fc, "value_fc_activation": "relu",
        "is_same_shape": False, "share_layers": share,
    }
    if fixed_log_std is not None:
        cfg["fixed_log_std"] = fixed_log_std
    return cfg

BASE_HP = {"lr": 3e-5, "lr_schedule": None, "entropy_coeff_schedule": None}
BEST_HP = {"lr": 1e-4, "lr_schedule": LR_SCHEDULE, "entropy_coeff_schedule": ENTROPY_SCHEDULE}

# --- 12 experiment variants ---
# Transformer: 874,497 params, share_layers=True, deterministic (log_std≈-10)
VARIANTS = [
    # share_layers=False (7)                                                                   # Params
    {"model_config": mlp_cfg([256]*3, [256,256,128], False),                  **BASE_HP},      #  0: baseline           292K
    {"model_config": mlp_cfg([256]*3, [256,256,128], False, -10.0),           **BASE_HP},      #  1: deterministic       292K
    {"model_config": mlp_cfg([256]*3, [256,256,128], False),     "lr": 1e-4,                   #  2: higher_lr          292K
     "lr_schedule": None, "entropy_coeff_schedule": None},
    {"model_config": mlp_cfg([256]*3, [256,256,128], False),     "lr": 1e-4,                   #  3: lr_schedule        292K
     "lr_schedule": LR_SCHEDULE, "entropy_coeff_schedule": None},
    {"model_config": mlp_cfg([256]*3, [256,256,128], False),                  **BEST_HP},      #  4: entropy            292K
    {"model_config": mlp_cfg([462]*3, [462,462,256], False),                  **BEST_HP},      #  5: wide_matched       872K ≈ TF
    {"model_config": mlp_cfg([512]*4, [512,512,512,256], False),              **BEST_HP},      #  6: wide_deep        1,569K
    # share_layers=True (3)
    {"model_config": mlp_cfg([256]*3, [256,256,128], True),                   **BASE_HP},      #  7: shared_base        168K
    {"model_config": mlp_cfg([512]*3, [512,512,256], True),                   **BEST_HP},      #  8: shared_wide        598K
    {"model_config": mlp_cfg([516]*4, [516,516,516,256], True),               **BEST_HP},      #  9: shared_wd_matched  874K ≈ TF
    # Param-matched deterministic (~875K ≈ Transformer)
    {"model_config": mlp_cfg([516]*4, [516,516,516,256], True, -10.0),        **BEST_HP},      # 10: sh_determ_match    874K ≈ TF
    {"model_config": mlp_cfg([462]*3, [462,462,256], False, -10.0),           **BEST_HP},      # 11: determ_match       872K ≈ TF
]


if __name__ == "__main__":
    # Cap CPU at 48 → 6 trials run simultaneously (7 workers + 1 eval = 8 CPUs/trial)
    ray.init(num_cpus=48)

    register_env("lazy_env_train", lambda cfg: LazyAgentsCentralizedPendReward(cfg))
    register_env("lazy_env_eval", lambda cfg: LazyAgentsCentralized(cfg))
    ModelCatalog.register_custom_model("custom_model_mlp", MyMLPModel)

    tune.run(
        "PPO",
        name="mlp_ablation_v3",
        stop={"training_iteration": 300},
        checkpoint_freq=1,
        keep_checkpoints_num=32,
        checkpoint_at_end=True,
        checkpoint_score_attr="evaluation/episode_reward_mean",
        config={
            "env": "lazy_env_train",
            "env_config": {
                **ENV_CONFIG,
                "_variant": tune.grid_search(list(range(len(VARIANTS)))),
            },
            "framework": "torch",
            "callbacks": MyCallbacks,
            "model": {
                "custom_model": "custom_model_mlp",
                "custom_model_config": tune.sample_from(
                    lambda spec: VARIANTS[spec.config["env_config"]["_variant"]]["model_config"]
                ),
            },
            "lr": tune.sample_from(
                lambda spec: VARIANTS[spec.config["env_config"]["_variant"]]["lr"]
            ),
            "lr_schedule": tune.sample_from(
                lambda spec: VARIANTS[spec.config["env_config"]["_variant"]]["lr_schedule"]
            ),
            "entropy_coeff_schedule": tune.sample_from(
                lambda spec: VARIANTS[spec.config["env_config"]["_variant"]]["entropy_coeff_schedule"]
            ),
            # --- Worker config matching Transformer exactly ---
            "num_gpus": 0.5,
            "num_workers": 7,
            "num_envs_per_worker": 2,
            "rollout_fragment_length": 1000,
            "train_batch_size": 14000,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 36,
            "batch_mode": "complete_episodes",
            "vf_loss_coeff": 0.1, "use_critic": True, "use_gae": True,
            "gamma": 0.992, "lambda": 0.96, "kl_coeff": 0,
            "clip_param": 0.25, "vf_clip_param": 20, "grad_clip": 40.0,
            "kl_target": 0.01,
            # --- Eval: L1, use_fixed_horizon=False, 100 episodes ---
            "evaluation_interval": 1,
            "evaluation_duration": 100,
            "evaluation_duration_unit": "episodes",
            "evaluation_num_workers": 1,
            "evaluation_config": {
                "env": "lazy_env_eval",
                "env_config": EVAL_ENV_CONFIG,
                "explore": False,
            },
        },
    )
