# MLP Ablation Study: Train MLP model with same hyperparameters as Transformer checkpoint 74
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


if __name__ == "__main__":

    # Environment config matching checkpoint 74's params.json
    env_config = {
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
        # PendReward weights
        "w_vel": 0.2,
        "w_control": 0.02,
    }

    # Register training env (PendReward) and eval env (base)
    register_env("lazy_env_train", lambda cfg: LazyAgentsCentralizedPendReward(cfg))
    register_env("lazy_env_eval", lambda cfg: LazyAgentsCentralized(cfg))

    # Register MLP model
    model_name_mlp = "custom_model_mlp"
    ModelCatalog.register_custom_model(model_name_mlp, MyMLPModel)

    custom_model_config_mlp = {
        "fc_sizes": [256, 256, 256],
        "fc_activation": "relu",
        "value_fc_sizes": [256, 256, 128],
        "value_fc_activation": "relu",
        "is_same_shape": False,
        "share_layers": False,
    }

    # Train with PPO — hyperparameters match checkpoint 74 exactly
    tune.run(
        "PPO",
        name="mlp_ablation",
        stop={"training_iteration": 300},
        checkpoint_freq=1,
        keep_checkpoints_num=32,
        checkpoint_at_end=True,
        checkpoint_score_attr="evaluation/episode_reward_mean",
        config={
            "env": "lazy_env_train",
            "env_config": env_config,
            "framework": "torch",
            "callbacks": MyCallbacks,
            "model": {
                "custom_model": model_name_mlp,
                "custom_model_config": custom_model_config_mlp,
            },
            # Resources
            "num_gpus": 0.5,
            "num_workers": 7,
            "num_envs_per_worker": 2,
            # PPO hyperparameters (from checkpoint 74)
            "rollout_fragment_length": 1000,
            "train_batch_size": 14000,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 36,
            "batch_mode": "complete_episodes",
            "lr": 3e-5,
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
            # Evaluation on base env (LazyAgentsCentralized)
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
