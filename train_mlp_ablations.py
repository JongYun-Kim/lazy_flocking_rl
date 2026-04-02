# MLP Ablation Study — Additional Experiments
# Exp 1: "Deterministic" MLP (fixed log_std=-10, matching Transformer's approach)
# Exp 2: Higher LR (1e-4) — MLP has fewer params, may benefit from larger steps
# Exp 3: LR schedule (1e-4 → 1e-5) — warm start then fine-tune
# Exp 4: Entropy bonus with schedule — encourage exploration early, anneal to 0
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


# Shared env config (matching checkpoint 74)
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

# Shared PPO config (from checkpoint 74)
BASE_PPO_CONFIG = {
    "env": "lazy_env_train",
    "env_config": ENV_CONFIG,
    "framework": "torch",
    "callbacks": MyCallbacks,
    "num_gpus": 0.5,
    "num_workers": 7,
    "num_envs_per_worker": 2,
    "rollout_fragment_length": 1000,
    "train_batch_size": 14000,
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
    # Evaluation on base env
    "evaluation_interval": 1,
    "evaluation_duration": 10,
    "evaluation_duration_unit": "episodes",
    "evaluation_num_workers": 1,
    "evaluation_config": {
        "env": "lazy_env_eval",
        "explore": False,
    },
}

# MLP model configs per experiment
MLP_BASE = {
    "fc_sizes": [256, 256, 256],
    "fc_activation": "relu",
    "value_fc_sizes": [256, 256, 128],
    "value_fc_activation": "relu",
    "is_same_shape": False,
    "share_layers": False,
}

EXPERIMENTS = {
    # Exp 1: Fixed log_std=-10 (matching Transformer's "deterministic" mode)
    "mlp_deterministic": {
        "model_config": {**MLP_BASE, "fixed_log_std": -10.0},
        "lr": 3e-5,
    },
    # Exp 2: Higher LR — MLP is simpler, can often handle larger learning rates
    "mlp_higher_lr": {
        "model_config": MLP_BASE,
        "lr": 1e-4,
    },
    # Exp 3: LR schedule — start high for fast initial learning, decay for fine-tuning
    #   14000 samples/iter * 300 iters = 4.2M total timesteps
    "mlp_lr_schedule": {
        "model_config": MLP_BASE,
        "lr": 1e-4,
        "lr_schedule": [
            [0, 1e-4],
            [1_000_000, 5e-5],
            [2_500_000, 1e-5],
            [4_200_000, 5e-6],
        ],
    },
    # Exp 4: Entropy bonus with annealing — helps MLP explore before converging
    "mlp_entropy": {
        "model_config": MLP_BASE,
        "lr": 1e-4,
        "lr_schedule": [
            [0, 1e-4],
            [1_000_000, 5e-5],
            [2_500_000, 1e-5],
            [4_200_000, 5e-6],
        ],
        "entropy_coeff_schedule": [
            [0, 0.01],
            [1_000_000, 0.005],
            [2_500_000, 0.001],
            [4_200_000, 0.0],
        ],
    },
}


def run_experiment(name, exp_cfg):
    config = {**BASE_PPO_CONFIG}
    config["model"] = {
        "custom_model": "custom_model_mlp",
        "custom_model_config": exp_cfg["model_config"],
    }
    config["lr"] = exp_cfg["lr"]
    if "lr_schedule" in exp_cfg:
        config["lr_schedule"] = exp_cfg["lr_schedule"]
    if "entropy_coeff_schedule" in exp_cfg:
        config["entropy_coeff_schedule"] = exp_cfg["entropy_coeff_schedule"]

    tune.run(
        "PPO",
        name=f"mlp_ablation_{name}",
        stop={"training_iteration": 300},
        checkpoint_freq=1,
        keep_checkpoints_num=32,
        checkpoint_at_end=True,
        checkpoint_score_attr="evaluation/episode_reward_mean",
        config=config,
    )


if __name__ == "__main__":
    register_env("lazy_env_train", lambda cfg: LazyAgentsCentralizedPendReward(cfg))
    register_env("lazy_env_eval", lambda cfg: LazyAgentsCentralized(cfg))
    ModelCatalog.register_custom_model("custom_model_mlp", MyMLPModel)

    for exp_name, exp_cfg in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Starting experiment: {exp_name}")
        print(f"{'='*60}\n")
        run_experiment(exp_name, exp_cfg)
