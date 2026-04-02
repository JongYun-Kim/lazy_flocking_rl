# MLP Ablation Study — All Additional Experiments (parallel via single tune.run)
#
# Experiments:
#   1. deterministic     — fixed log_std=-10 (matching Transformer approach), baseline HP
#   2. higher_lr         — lr=1e-4
#   3. lr_schedule       — lr 1e-4 → 5e-6
#   4. entropy           — lr schedule + entropy annealing
#   5. wide              — [512,512,512] + lr schedule + entropy
#   6. wide_deep         — [512,512,512,512] + lr schedule + entropy
#
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


# --- Env config (matching checkpoint 74) ---
ENV_CONFIG = {
    "num_agents_max": 20,
    "num_agents_min": 20,
    "speed": 15,
    "predefined_distance": 60,
    "std_pos_converged": 45,
    "std_vel_converged": 0.1,
    "std_pos_rate_converged": 0.1,
    "std_vel_rate_converged": 0.2,
    "max_time_step": 1000,
    "incomplete_episode_penalty": 0,
    "normalize_obs": True,
    "use_fixed_horizon": True,
    "use_L2_norm": True,
    "auto_step": False,
    "use_custom_ray": False,
    "use_preprocessed_obs": True,
    "use_mlp_settings": True,
    "w_vel": 0.2,
    "w_control": 0.02,
}

# --- MLP model configs ---
MLP_BASE = {
    "fc_sizes": [256, 256, 256],
    "fc_activation": "relu",
    "value_fc_sizes": [256, 256, 128],
    "value_fc_activation": "relu",
    "is_same_shape": False,
    "share_layers": False,
}

MLP_WIDE = {
    "fc_sizes": [512, 512, 512],
    "fc_activation": "relu",
    "value_fc_sizes": [512, 512, 256],
    "value_fc_activation": "relu",
    "is_same_shape": False,
    "share_layers": False,
}

MLP_WIDE_DEEP = {
    "fc_sizes": [512, 512, 512, 512],
    "fc_activation": "relu",
    "value_fc_sizes": [512, 512, 512, 256],
    "value_fc_activation": "relu",
    "is_same_shape": False,
    "share_layers": False,
}

# --- LR / entropy schedules ---
# 14000 samples/iter * 300 iters = 4.2M total timesteps
LR_SCHEDULE = [[0, 1e-4], [1_000_000, 5e-5], [2_500_000, 1e-5], [4_200_000, 5e-6]]
ENTROPY_SCHEDULE = [[0, 0.01], [1_000_000, 0.005], [2_500_000, 0.001], [4_200_000, 0.0]]

# --- Experiment variants (indexed 0-5) ---
VARIANTS = [
    # 0: deterministic (fixed log_std=-10)
    {
        "model_config": {**MLP_BASE, "fixed_log_std": -10.0},
        "lr": 3e-5,
        "lr_schedule": None,
        "entropy_coeff_schedule": None,
    },
    # 1: higher LR
    {
        "model_config": MLP_BASE,
        "lr": 1e-4,
        "lr_schedule": None,
        "entropy_coeff_schedule": None,
    },
    # 2: LR schedule
    {
        "model_config": MLP_BASE,
        "lr": 1e-4,
        "lr_schedule": LR_SCHEDULE,
        "entropy_coeff_schedule": None,
    },
    # 3: entropy + LR schedule
    {
        "model_config": MLP_BASE,
        "lr": 1e-4,
        "lr_schedule": LR_SCHEDULE,
        "entropy_coeff_schedule": ENTROPY_SCHEDULE,
    },
    # 4: wide [512,512,512] + best-effort HP
    {
        "model_config": MLP_WIDE,
        "lr": 1e-4,
        "lr_schedule": LR_SCHEDULE,
        "entropy_coeff_schedule": ENTROPY_SCHEDULE,
    },
    # 5: wide+deep [512,512,512,512] + best-effort HP
    {
        "model_config": MLP_WIDE_DEEP,
        "lr": 1e-4,
        "lr_schedule": LR_SCHEDULE,
        "entropy_coeff_schedule": ENTROPY_SCHEDULE,
    },
]

VARIANT_NAMES = [
    "deterministic", "higher_lr", "lr_schedule",
    "entropy", "wide", "wide_deep",
]


if __name__ == "__main__":
    register_env("lazy_env_train", lambda cfg: LazyAgentsCentralizedPendReward(cfg))
    register_env("lazy_env_eval", lambda cfg: LazyAgentsCentralized(cfg))
    ModelCatalog.register_custom_model("custom_model_mlp", MyMLPModel)

    tune.run(
        "PPO",
        name="mlp_ablation_v2",
        stop={"training_iteration": 300},
        checkpoint_freq=1,
        keep_checkpoints_num=32,
        checkpoint_at_end=True,
        checkpoint_score_attr="evaluation/episode_reward_mean",
        config={
            "env": "lazy_env_train",
            "env_config": {
                **ENV_CONFIG,
                # variant index lives inside env_config to avoid PPO config validation error
                "_variant": tune.grid_search(list(range(len(VARIANTS)))),
            },
            "framework": "torch",
            "callbacks": MyCallbacks,
            # --- Variant-dependent params resolved via sample_from ---
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
            # --- Fixed PPO config (from checkpoint 74) ---
            "num_gpus": 0.5,
            "num_workers": 5,
            "num_envs_per_worker": 2,
            "rollout_fragment_length": 1000,
            "train_batch_size": 10000,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 36,
            "batch_mode": "complete_episodes",
            "vf_loss_coeff": 0.1,
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.992,
            "lambda": 0.96,
            "kl_coeff": 0,
            "clip_param": 0.25,
            "vf_clip_param": 20,
            "grad_clip": 40.0,
            "kl_target": 0.01,
            # --- Evaluation ---
            "evaluation_interval": 1,
            "evaluation_duration": 10,
            "evaluation_duration_unit": "episodes",
            "evaluation_num_workers": 1,
            "evaluation_config": {
                "env": "lazy_env_eval",
                "explore": False,
            },
        },
    )
