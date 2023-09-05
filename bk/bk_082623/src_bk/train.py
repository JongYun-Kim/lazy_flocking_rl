import ray
from ray import tune
from ray.rllib.models import ModelCatalog
# from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from env.envs import LazyAgentsCentralizedPendReward
from models.lazy_allocator import MyRLlibTorchWrapper, MyMLPModel
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

    # Debugging...
    # ray.init(local_mode=True)

    # register your custom environment
    env_config = {
        # Check the target
    }
    env_name = "lazy_env"
    register_env(env_name, lambda cfg: LazyAgentsCentralizedPendReward(cfg))

    # register your custom model
    custom_model_config_transformer = {
        # Check the target
    }
    model_name_transformer = "custom_model"
    ModelCatalog.register_custom_model(model_name_transformer, MyRLlibTorchWrapper)

    #
    # train your custom model with PPO
    tune.run(
        "PPO",
        name="I_am_a_lazy_guy",
        # stop={"episode_reward_mean": -101},
        checkpoint_freq=1,
        keep_checkpoints_num=12,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",  # better to have the custom one instead
        config={
            "env": env_name,
            "env_config": env_config,
            #
            "framework": "torch",
            #
            "callbacks": MyCallbacks,
            #
            "model": {
                "custom_model": model_name_used,  # registered str or type
                "custom_model_config": custom_model_config_used,
            },
            # Check the target
            # ...
            # ...
        },
    )
