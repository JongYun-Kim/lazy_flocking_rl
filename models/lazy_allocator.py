import copy

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2, ModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC, normc_initializer

import numpy as np
import gym
from gym.spaces import Tuple, Box
from typing import Dict, List, Union

from models.transformer_modules.token_embedding import LinearEmbedding
from models.transformer_modules.multi_head_attention_layer import MultiHeadAttentionLayer
from models.transformer_modules.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from models.transformer_modules.encoder_block import EncoderBlock
from models.transformer_modules.decoder_block import CustomDecoderBlock as DecoderBlock
from models.transformer_modules.encoder import Encoder
from models.transformer_modules.decoder import Decoder, DecoderPlaceholder
from models.transformer_modules.pointer_net import GaussianActionDistGenerator, GaussianActionDistPlaceholder
from models.transformer_modules.pointer_net import MeanGenerator, PointerPlaceholder, FakeMeanGenerator
from models.transformer_modules.pointer_net import GaussControlDistGenerator

import torch
import torch.nn as nn


class LazinessAllocator(nn.Module):
    def __init__(self, src_embed, encoder, decoder, generator):
        super().__init__()

        if isinstance(generator, (GaussianActionDistGenerator, GaussianActionDistPlaceholder)):
            self.use_deterministic_action_dist = False
        elif isinstance(generator, (MeanGenerator, PointerPlaceholder)):
            self.use_deterministic_action_dist = True
        elif isinstance(generator, (GaussControlDistGenerator, GaussianActionDistPlaceholder)):
            self.use_deterministic_action_dist = False
        else:
            raise ValueError("generator must be an instance of GaussianActionDistGenerator or MeanGenerator")

        self.src_embed = src_embed
        self.d_v = src_embed.in_features
        self.encoder = encoder
        self.decoder = decoder
        self.gaussian_action_dist_generator = generator if not self.use_deterministic_action_dist else None
        self.deterministic_action_dist_generator = generator if self.use_deterministic_action_dist else None

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)

    def forward(self, src_dict: Dict[str, TensorType]):
        agent_embeddings = src_dict["agent_embeddings"][:, :, :self.d_v]  # (batch_size, seq_len_src, d_subobs)
        pad_tokens = src_dict["pad_tokens"]  # (batch_size, seq_len_src)

        # Masks (no head dim — unsqueeze(1) added at call site for MHA broadcasting)
        src_mask = self.make_src_mask(pad_tokens)  # (batch_size, seq_len_src, seq_len_src)
        tgt_mask = None
        context_token = torch.zeros_like(pad_tokens[:, 0:1])  # (batch_size, 1)
        src_tgt_mask = self.make_src_tgt_mask(pad_tokens, context_token)  # (batch_size, 1, seq_len_src)

        encoder_out = self.encode(agent_embeddings, src_mask.unsqueeze(1))  # (batch_size, seq_len_src, d_embed)

        h_c_N = self.get_context_node(embeddings=encoder_out, pad_tokens=pad_tokens, use_embeddings_mask=True)
        decoder_out = self.decode(h_c_N, encoder_out, tgt_mask, src_tgt_mask.unsqueeze(1))  # (batch_size, 1, d_embed_context)

        if self.use_deterministic_action_dist:
            out = self.deterministic_action_dist_generator(query=decoder_out, key=encoder_out, mask=src_tgt_mask)
            out = out.squeeze(1)  # (batch_size, num_agents_max)
        else:
            # out_mean, out_std: (batch_size, 1, num_agents_max)
            out_mean, out_std = self.gaussian_action_dist_generator(query=decoder_out, key=encoder_out, mask=src_tgt_mask)
            out = torch.cat((out_mean.squeeze(1), out_std.squeeze(1)), dim=-1)  # (batch_size, 2 * num_agents_max)
            assert out.shape == (agent_embeddings.shape[0], 2 * agent_embeddings.shape[1])

        return out, decoder_out, h_c_N

    def get_context_node(self, embeddings, pad_tokens, use_embeddings_mask=True, debug=False):
        # embeddings: (batch_size, num_agents_max, data_size)
        # pad_tokens: (batch_size, num_agents_max); 1=padded, 0=real
        batch_size, num_agents_max, data_size = embeddings.shape

        if use_embeddings_mask:
            mask = pad_tokens.unsqueeze(-1).expand_as(embeddings)  # (batch_size, num_agents_max, data_size)

            # NOTE: keeps padded (mask==1) embeddings and zeros real ones — inverted from typical convention.
            # The trained checkpoint depends on this behavior; do not "fix" without retraining.
            embeddings_masked = torch.where(mask == 1, embeddings, torch.zeros_like(embeddings))

            embeddings_sum = torch.sum(embeddings_masked, dim=1, keepdim=True)  # (batch_size, 1, data_size)
            embeddings_count = torch.sum((mask == 0), dim=1, keepdim=True).float()  # (batch_size, 1, data_size)

            if debug:
                if torch.any(embeddings_count == 0):
                    raise ValueError("All agents are padded in at least one sample.")

            embeddings_avg = embeddings_sum / embeddings_count  # (batch_size, 1, data_size)
        else:
            embeddings_avg = torch.mean(embeddings, dim=1, keepdim=True)  # (batch_size, 1, data_size)

        return embeddings_avg  # (batch_size, 1, d_embed_context)

    def make_src_mask(self, src):
        return self.make_pad_mask(src, src)  # (batch_size, seq_len_src, seq_len_src)

    def make_src_tgt_mask(self, src, tgt):
        return self.make_pad_mask(tgt, src)  # (batch_size, seq_len_tgt, seq_len_src)

    def make_pad_mask(self, query, key, pad_idx=1, dim_check=False):
        # mask value: 1 where real (token != pad_idx), 0 where padded (token == pad_idx)
        # In MHA: mask 0 → attention_score -inf → no attention
        if dim_check:
            assert len(query.shape) == 2, "query must have 2 dimensions: (n_batch, query_seq_len)"
            assert len(key.shape) == 2, "key must have 2 dimensions: (n_batch, key_seq_len)"
            assert query.size(0) == key.size(0), "query and key must have the same batch size"

        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).repeat(1, query_seq_len, 1)  # (n_batch, query_seq_len, key_seq_len)
        query_mask = query.ne(pad_idx).unsqueeze(2).repeat(1, 1, key_seq_len)  # (n_batch, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        # !!! NO HEAD DIM — unsqueeze(1) must be applied at the call site before passing to MHA !!!
        return mask  # (n_batch, query_seq_len, key_seq_len)


class MyRLlibTorchWrapper(TorchModelV2, nn.Module):

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        if model_config is not None:
            cfg = model_config["custom_model_config"]
            share_layers = cfg["share_layers"] if "share_layers" in cfg else True
            if "d_subobs" not in cfg:
                raise ValueError("d_subobs must be specified in custom_model_config")
            d_subobs = cfg["d_subobs"]
            d_embed_input = cfg["d_embed_input"] if "d_embed_input" in cfg else 128
            d_embed_context = cfg["d_embed_context"] if "d_embed_context" in cfg else 128
            d_model = cfg["d_model"] if "d_model" in cfg else 128
            d_model_decoder = cfg["d_model_decoder"] if "d_model_decoder" in cfg else 128
            n_layers_encoder = cfg["n_layers_encoder"] if "n_layers_encoder" in cfg else 3
            n_layers_decoder = cfg["n_layers_decoder"] if "n_layers_decoder" in cfg else 2
            h = cfg["num_heads"] if "num_heads" in cfg else 8
            d_ff = cfg["d_ff"] if "d_ff" in cfg else 512
            d_ff_decoder = cfg["d_ff_decoder"] if "d_ff_decoder" in cfg else 512
            clip_action_mean = cfg["clip_action_mean"] if "clip_action_mean" in cfg else 1.0
            clip_action_log_std = cfg["clip_action_log_std"] if "clip_action_log_std" in cfg else 10.0
            dr_rate = cfg["dr_rate"] if "dr_rate" in cfg else 0
            norm_eps = cfg["norm_eps"] if "norm_eps" in cfg else 1e-5
            is_bias = cfg["is_bias"] if "is_bias" in cfg else True  # bias in MHA linear layers (W_q, W_k, W_v)
            use_residual_in_decoder = cfg["use_residual_in_decoder"] if "use_residual_in_decoder" in cfg else True
            use_FNN_in_decoder = cfg["use_FNN_in_decoder"] if "use_FNN_in_decoder" in cfg else True
            use_deterministic_action_dist = cfg["use_deterministic_action_dist"] \
                if "use_deterministic_action_dist" in cfg else False

            if "ignore_residual_in_decoder" in cfg:
                use_residual_in_decoder = not cfg["ignore_residual_in_decoder"]
                print("DeprecationWarning: ignore_residual_in_decoder is deprecated; use use_residual_in_decoder instead")
                use_FNN_in_decoder = use_residual_in_decoder
                print("use_FNN_in_decoder is set to use_residual_in_decoder as {}".format(use_residual_in_decoder))
            if use_residual_in_decoder != use_FNN_in_decoder:
                print(*["Warning: use_residual_in_decoder != use_FNN_in_decoder"] * 7, sep="\n")
            if n_layers_decoder >= 2 and not use_residual_in_decoder:
                print(*["Warning: multiple decoder blocks often require residual connections"] * 7, sep="\n")
        else:
            raise ValueError("model_config must be specified")

        input_embed = LinearEmbedding(
            d_env=d_subobs,
            d_embed=d_embed_input,
        )
        mha_encoder = MultiHeadAttentionLayer(
            d_model=d_model,
            h=h,
            q_fc=nn.Linear(d_embed_input, d_model, is_bias),
            kv_fc=nn.Linear(d_embed_input, d_model, is_bias),
            out_fc=nn.Linear(d_model, d_embed_input, is_bias),
            dr_rate=dr_rate,
        )
        position_ff_encoder = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed_input, d_ff),
            fc2=nn.Linear(d_ff, d_embed_input),
            dr_rate=dr_rate,
        )
        norm_encoder = nn.LayerNorm(d_embed_input, eps=norm_eps)
        mha_decoder = MultiHeadAttentionLayer(
            d_model=d_model_decoder,
            h=h,
            q_fc=nn.Linear(d_embed_context, d_model_decoder, is_bias),
            kv_fc=nn.Linear(d_embed_input, d_model_decoder, is_bias),
            out_fc=nn.Linear(d_model_decoder, d_embed_context, is_bias),
            dr_rate=dr_rate,
        )
        position_ff_decoder = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed_context, d_ff_decoder),
            fc2=nn.Linear(d_ff_decoder, d_embed_context),
            dr_rate=dr_rate,
        ) if use_FNN_in_decoder else None
        norm_decoder = nn.LayerNorm(d_embed_context, eps=norm_eps)

        encoder_block = EncoderBlock(
            self_attention=copy.deepcopy(mha_encoder),
            position_ff=copy.deepcopy(position_ff_encoder),
            norm=copy.deepcopy(norm_encoder),
            dr_rate=dr_rate,
        )
        decoder_block = DecoderBlock(
            self_attention=None,
            cross_attention=copy.deepcopy(mha_decoder),
            position_ff=position_ff_decoder,
            norm=copy.deepcopy(norm_decoder),
            dr_rate=dr_rate,
            efficient=not use_residual_in_decoder,
        )

        encoder = Encoder(
            encoder_block=encoder_block,
            n_layer=n_layers_encoder,
            norm=copy.deepcopy(norm_encoder),
        )
        decoder = Decoder(
            decoder_block=decoder_block,
            n_layer=n_layers_decoder,
            norm=copy.deepcopy(norm_decoder),
        )
        if use_deterministic_action_dist:
            # Gaussian with constant near-zero std (effectively deterministic)
            generator = FakeMeanGenerator(
                d_model=d_model,
                q_fc=nn.Linear(d_embed_context, d_model, is_bias),
                k_fc=nn.Linear(d_embed_input, d_model, is_bias),
                clip_value_mean=clip_action_mean,
                clip_value_std=clip_action_log_std,
                dr_rate=dr_rate,
            )
        else:
            generator = GaussianActionDistGenerator(
                d_model=d_model,
                q_fc=nn.Linear(d_embed_context, d_model, is_bias),
                k_fc=nn.Linear(d_embed_input, d_model, is_bias),
                clip_value_mean=clip_action_mean,
                clip_value_std=clip_action_log_std,
                dr_rate=dr_rate,
            )

        action_size = action_space.shape[0]
        assert num_outputs == 2 * action_size, "num_outputs must be 2 * action_size"

        self.policy_network = LazinessAllocator(
            src_embed=input_embed,
            encoder=encoder,
            decoder=decoder,
            generator=generator,
        )

        self.values = None
        self.share_layers = share_layers
        if not self.share_layers:
            self.value_network = LazinessAllocator(
                src_embed=copy.deepcopy(input_embed),
                encoder=copy.deepcopy(encoder),
                decoder=DecoderPlaceholder(),
                generator=GaussianActionDistPlaceholder(),
            )

        self.value_branch = nn.Sequential(
            nn.Linear(in_features=d_embed_context, out_features=d_embed_context),
            nn.ReLU(),
            nn.Linear(in_features=d_embed_context, out_features=1),
        )

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        obs = input_dict["obs"]
        self._validate_obs(obs)

        # x: (batch_size, 2 * action_size); h_c_N1: (batch_size, 1, d_embed_context)
        if self.share_layers:
            x, h_c_N1, h_c_N = self.policy_network(obs)
            self.values = h_c_N1.squeeze(1)  # (batch_size, d_embed_context)
        else:
            x, _, _ = self.policy_network(obs)
            self.values = self.value_network(obs)[2].squeeze(1)  # (batch_size, d_embed_context)

        return x, state

    def _validate_obs(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> None:

        agent_embeddings = obs["agent_embeddings"]
        pad_tokens = obs["pad_tokens"]
        num_agents_max = pad_tokens.shape[1]
        pad_mask = pad_tokens == 1

        assert isinstance(agent_embeddings, torch.Tensor), "agent_embeddings must be a torch.Tensor"
        assert agent_embeddings.ndim == 3, "agent_embeddings must be a 3D tensor"
        assert agent_embeddings.shape[1] == num_agents_max
        assert torch.all(agent_embeddings[pad_mask, :] == 0), "padded agent_embeddings must be 0"

        assert isinstance(pad_tokens, torch.Tensor), "pad_tokens must be a torch.Tensor"
        assert pad_tokens.ndim == 2, "pad_tokens must be a 2D tensor"
        assert pad_tokens.shape[1] == num_agents_max

    def value_function(self) -> TensorType:
        return self.value_branch(self.values).squeeze(-1)


class MyMLPModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        cfg = model_config["custom_model_config"]
        if "is_same_shape" in cfg:
            self.is_same_shape = cfg["is_same_shape"]
        else:
            self.is_same_shape = False
            print("is_same_shape not received!!")
            print("is_same_shape == False")
        if "fc_sizes" in cfg:
            self.fc_sizes = cfg["fc_sizes"]
        else:
            self.fc_sizes = [256, 256]
            print(f"fc_sizes param in custom_model_config has NOT been received!")
            print(f"It goes with: fc_sizes = {self.fc_sizes}")
        if "fc_activation" in cfg:
            self.fc_activation = cfg["fc_activation"]
        else:
            self.fc_activation = "relu"
        if "value_fc_sizes" in cfg:
            if self.is_same_shape:
                self.value_fc_sizes = self.fc_sizes.copy()
            else:
                self.value_fc_sizes = cfg["value_fc_sizes"]
        else:
            self.value_fc_sizes = [256, 256]
            print(f"value_fc_sizes param in custom_model_config has NOT been received!")
            print(f"It goes with: value_fc_sizes = {self.value_fc_sizes}")
        if "value_fc_activation" in cfg:
            self.value_fc_activation = cfg["value_fc_activation"]
        else:
            self.value_fc_activation = "relu"
        # Define shared_layers flag
        if "share_layers" in cfg:
            self.share_layers = cfg["share_layers"]
        else:
            self.share_layers = False
            print("share_layers not received!!")
            print("share_layers == False")
        # Define deterministic_action flag
        if "deterministic_action" in cfg:
            self.deterministic_action = cfg["deterministic_action"]
        else:
            self.deterministic_action = False
            print("deterministic_action not received!!")
            print("deterministic_action == False")
        # Fixed log_std: when set, log_std outputs are fixed to this value (e.g., -10)
        # This mimics the Transformer's "deterministic" mode which uses Gaussian with very low std
        self.fixed_log_std = cfg.get("fixed_log_std", None)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size

        self._logits = None
        self._features = None
        self._values = None

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        for size in self.fc_sizes:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=self.fc_activation,
                )
            )
            prev_layer_size = size
        self.fc_net = nn.Sequential(*layers)

        if self.share_layers:
            self.value_fc_net = None
        else:
            value_layers = []
            prev_value_layer_size = int(np.product(obs_space.shape))
            for size in self.value_fc_sizes:
                value_layers.append(
                    SlimFC(
                        in_size=prev_value_layer_size,
                        out_size=size,
                        initializer=normc_initializer(1.0),
                        activation_fn=self.value_fc_activation,
                    )
                )
                prev_value_layer_size = size
            self.value_fc_net = nn.Sequential(*value_layers)

        self.last_size = self.fc_sizes[-1]
        self.last_value_size = self.value_fc_sizes[-1] if not self.share_layers else self.last_size
        if self.deterministic_action:
            self.action_branch = nn.Linear(self.last_size, num_outputs)
        else:
            assert num_outputs % 2 == 0, "num_outputs must be even!"
            assert num_outputs == 2 * action_space.shape[0], "num_outputs must be 2 * action_space.shape[0] for Gaussian dist"
            mean_size = int(num_outputs/2)
            self.action_branch_mean = nn.Linear(self.last_size, mean_size)
            self.action_branch_logstd = nn.Linear(self.last_size, mean_size)
        self.value_branch = nn.Linear(self.last_value_size, 1)

    @override(ModelV2)
    def value_function(self):
        assert self._values is not None, "must call forward() first"
        out = self.value_branch(self._values).squeeze(1)
        return out

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        self._features = self.fc_net(obs)

        if self.share_layers:
            self._values = self._features
        else:
            self._values = self.value_fc_net(obs)

        if self.deterministic_action:
            self._logits = self.action_branch(self._features)
        else:
            action_means = self.action_branch_mean(self._features)
                # [0, 1.05] range via tanh
            upper_bound = 1.05
            action_means = torch.tanh(action_means) * (upper_bound/2) + (upper_bound/2)

            if self.fixed_log_std is not None:
                action_stds = self.fixed_log_std * torch.ones_like(action_means)
            else:
                log_std_min = -10
                log_std_max = -1.5
                action_stds = self.action_branch_logstd(self._features)
                action_stds = torch.tanh(action_stds) * (log_std_max - log_std_min) / 2 + (log_std_max + log_std_min) / 2
            self._logits = torch.cat([action_means, action_stds], dim=1)


        assert self._logits.shape == (obs.shape[0], self.num_outputs), "logits shape is not correct!"

        return self._logits, state



