"""
Parameter-shared ego-observation transformer for decentralized PPO.

The model is a thin RLlib wrapper around a transformer encoder/decoder pulled
from ``models.transformer_modules`` (the same building blocks the existing
centralized ``MyRLlibTorchWrapper`` uses). The wrapper

  1. unpacks an ego observation of shape ``(B, N, N, d_v)`` and a per-agent
     neighbor mask of shape ``(B, N, N)``,
  2. flattens the agent dimension into the batch dimension so the transformer
     sees a tensor of shape ``(B*N, N, d_v)``,
  3. runs encoder + a learnable ego-query decoder to produce a single
     ``(mean, log_std)`` distribution per agent,
  4. reshapes back to ``(B, 2*N)`` -- the same format Ray PPO expects when the
     env's action space is a ``Box(num_agents_max,)``.

This makes the model directly drop-in compatible with Ray ``PPO`` (treat the
joint laziness vector as the action; the value head is a state-value function
computed from a global pool of the per-agent encoder embeddings).

It is therefore an instance of the parameter-sharing MAPPO pattern (a single
shared policy applied to each agent's local observation, with a centralized
critic that sees the pooled embeddings).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from models.transformer_modules.token_embedding import LinearEmbedding
from models.transformer_modules.multi_head_attention_layer import MultiHeadAttentionLayer
from models.transformer_modules.position_wise_feed_forward_layer import (
    PositionWiseFeedForwardLayer,
)
from models.transformer_modules.encoder_block import EncoderBlock
from models.transformer_modules.decoder_block import CustomDecoderBlock as DecoderBlock
from models.transformer_modules.encoder import Encoder
from models.transformer_modules.decoder import Decoder


class EgoSharedTransformer(TorchModelV2, nn.Module):
    """Decentralized parameter-shared transformer for ego observations.

    Configuration template (custom_model_config)::

        {
            "d_subobs": 5,           # number of input feature dims to read from
                                     # agent_embeddings (must be <= d_v)
            "d_embed": 128,
            "d_model": 128,
            "n_layers_encoder": 2,
            "n_layers_decoder": 1,
            "num_heads": 8,
            "d_ff": 512,
            "dr_rate": 0.0,
            "norm_eps": 1e-5,
            "is_bias": False,
            "clip_action_mean": 1.05,         # mean clipped to [0, clip_action_mean]
            "clip_action_log_std": 10.0,      # log_std clipped to [-clip_action_log_std, -2]
            "use_deterministic_action_dist": False,
            "share_layers": True,             # share encoder/decoder between policy and value
            "value_hidden": 128,              # value head hidden size
        }
    """

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

        cfg = (model_config or {}).get("custom_model_config", {})
        self.d_subobs = int(cfg.get("d_subobs", 5))
        d_embed = int(cfg.get("d_embed", 128))
        d_model = int(cfg.get("d_model", 128))
        n_layers_encoder = int(cfg.get("n_layers_encoder", 2))
        n_layers_decoder = int(cfg.get("n_layers_decoder", 1))
        h = int(cfg.get("num_heads", 8))
        d_ff = int(cfg.get("d_ff", 512))
        dr_rate = float(cfg.get("dr_rate", 0.0))
        norm_eps = float(cfg.get("norm_eps", 1e-5))
        is_bias = bool(cfg.get("is_bias", False))

        self.clip_action_mean = float(cfg.get("clip_action_mean", 1.05))
        self.clip_action_log_std = float(cfg.get("clip_action_log_std", 10.0))
        self.use_deterministic_action_dist = bool(
            cfg.get("use_deterministic_action_dist", False)
        )
        self.share_layers = bool(cfg.get("share_layers", True))
        value_hidden = int(cfg.get("value_hidden", d_embed))

        # ----- Encoder ------------------------------------------------- #
        input_embed = LinearEmbedding(d_env=self.d_subobs, d_embed=d_embed)
        mha_encoder = MultiHeadAttentionLayer(
            d_model=d_model,
            h=h,
            q_fc=nn.Linear(d_embed, d_model, is_bias),
            kv_fc=nn.Linear(d_embed, d_model, is_bias),
            out_fc=nn.Linear(d_model, d_embed, is_bias),
            dr_rate=dr_rate,
        )
        position_ff_encoder = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed, d_ff),
            fc2=nn.Linear(d_ff, d_embed),
            dr_rate=dr_rate,
        )
        norm_encoder = nn.LayerNorm(d_embed, eps=norm_eps)

        encoder_block = EncoderBlock(
            self_attention=copy.deepcopy(mha_encoder),
            position_ff=copy.deepcopy(position_ff_encoder),
            norm=copy.deepcopy(norm_encoder),
            dr_rate=dr_rate,
        )
        self.encoder = Encoder(
            encoder_block=encoder_block,
            n_layer=n_layers_encoder,
            norm=copy.deepcopy(norm_encoder),
        )
        self.input_embed = input_embed

        # ----- Decoder (cross-attends a learnable ego query to neighbors) - #
        mha_decoder = MultiHeadAttentionLayer(
            d_model=d_model,
            h=h,
            q_fc=nn.Linear(d_embed, d_model, is_bias),
            kv_fc=nn.Linear(d_embed, d_model, is_bias),
            out_fc=nn.Linear(d_model, d_embed, is_bias),
            dr_rate=dr_rate,
        )
        position_ff_decoder = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed, d_ff),
            fc2=nn.Linear(d_ff, d_embed),
            dr_rate=dr_rate,
        )
        norm_decoder = nn.LayerNorm(d_embed, eps=norm_eps)
        decoder_block = DecoderBlock(
            self_attention=None,
            cross_attention=copy.deepcopy(mha_decoder),
            position_ff=copy.deepcopy(position_ff_decoder),
            norm=copy.deepcopy(norm_decoder),
            dr_rate=dr_rate,
            efficient=False,
        )
        self.decoder = Decoder(
            decoder_block=decoder_block,
            n_layer=n_layers_decoder,
            norm=copy.deepcopy(norm_decoder),
        )

        # Per-agent action head: takes the per-agent decoded embedding and
        # outputs (mean, log_std) for that single agent.
        self.action_head_mean = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 1),
        )
        self.action_head_log_std = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 1),
        )

        # Centralized value head: state-value from the pooled per-agent embeddings.
        self.value_head = nn.Sequential(
            nn.Linear(d_embed, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

        # Cached value features.
        self._value_features: torch.Tensor = None  # (B, d_embed)

        # ----- Sanity checks ------------------------------------------ #
        action_size = action_space.shape[0]
        assert num_outputs == 2 * action_size, (
            f"num_outputs must be 2 * action_size for a Gaussian dist "
            f"(got {num_outputs} vs {2*action_size})"
        )
        self._action_size = action_size

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ):
        obs = input_dict["obs"]
        ego_embeddings = obs["agent_embeddings"]  # (B, N, N, d_v)
        ego_neighbor_mask = obs["ego_neighbor_mask"]  # (B, N, N) -- 1=valid neighbor
        pad_tokens = obs["pad_tokens"]  # (B, N) -- 1=ego is padded

        if ego_embeddings.dim() != 4:
            raise ValueError(
                f"agent_embeddings must be a 4-D tensor (B, N, N, d_v); "
                f"got shape {tuple(ego_embeddings.shape)}"
            )

        B, N, _, d_v = ego_embeddings.shape
        assert N == self._action_size, (
            f"agent dimension N ({N}) must equal the action size ({self._action_size})"
        )
        if d_v < self.d_subobs:
            raise ValueError(
                f"d_v ({d_v}) is smaller than configured d_subobs ({self.d_subobs})"
            )

        # ----- Flatten the ego dimension into the batch dimension ----- #
        flat_embeddings = ego_embeddings[..., : self.d_subobs].reshape(
            B * N, N, self.d_subobs
        )
        flat_neighbor_mask = ego_neighbor_mask.reshape(B * N, N).to(torch.bool)

        # Convert to the parent transformer's pad-style tokens (1 == valid token,
        # i.e. attended; 0 == padded). The MHA layer ultimately compares to 0 to
        # mask out, so we feed in the *neighbor* mask directly as 1/0 ints.
        # Be careful: the existing make_pad_mask uses pad_idx=1 == padded. Here
        # we want valid==1, so we pass `make_neighbor_mask` style and build the
        # mha mask manually below.
        valid = flat_neighbor_mask  # (B*N, N) bool

        # Self-attention key mask: (B*N, 1, N)
        # The MultiHeadAttentionLayer multiplies attention scores by mask==0 -> -inf,
        # so we want True for the valid tokens.
        src_key_mask = valid.unsqueeze(1)  # (B*N, 1, N)
        src_self_mask = src_key_mask & valid.unsqueeze(2)  # (B*N, N, N)

        # ----- Encoder ------------------------------------------------ #
        embedded = self.input_embed(flat_embeddings)  # (B*N, N, d_embed)
        encoder_out = self.encoder(embedded, src_self_mask.unsqueeze(1))
        # encoder_out: (B*N, N, d_embed)

        # ----- Per-agent context = mean over neighbors ---------------- #
        valid_f = valid.unsqueeze(-1).float()  # (B*N, N, 1)
        # Avoid div-by-zero for fully masked rows (shouldn't happen because of
        # self-loops, but just in case).
        denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        context = (encoder_out * valid_f).sum(dim=1, keepdim=True) / denom
        # context: (B*N, 1, d_embed)

        # ----- Decoder ------------------------------------------------ #
        # The decoder block expects src_tgt_mask of shape (B*N, 1, 1, N) and
        # ignores tgt_mask in our CustomDecoderBlock.
        cross_mask = src_key_mask.unsqueeze(1)  # (B*N, 1, 1, N)
        decoded = self.decoder(context, encoder_out, None, cross_mask)
        # decoded: (B*N, 1, d_embed)
        ego_features = decoded.squeeze(1)  # (B*N, d_embed)

        # ----- Action distribution per agent -------------------------- #
        mean_raw = self.action_head_mean(ego_features)  # (B*N, 1)
        mean = (torch.tanh(mean_raw) + 1.0) * 0.5 * self.clip_action_mean

        if self.use_deterministic_action_dist:
            log_std = torch.full_like(mean, -self.clip_action_log_std)
        else:
            log_std_raw = self.action_head_log_std(ego_features)  # (B*N, 1)
            # Map to roughly [-clip_action_log_std, -2]
            u = 2.0
            log_std = (
                (self.clip_action_log_std - u)
                * ((torch.tanh(log_std_raw) - 1.0) / 2.0)
                - u
            )

        mean = mean.view(B, N)  # (B, N)
        log_std = log_std.view(B, N)

        # Mask out padded ego agents (set mean=0, log_std=-clip).
        live_mask = (pad_tokens == 0).to(mean.dtype)  # (B, N)
        mean = mean * live_mask
        log_std = log_std * live_mask + (-self.clip_action_log_std) * (1.0 - live_mask)

        logits = torch.cat([mean, log_std], dim=1)  # (B, 2*N)

        # ----- Value features (centralized critic) -------------------- #
        # Pool per-agent ego features into a single state vector. Live agents
        # only.
        ego_features_b = ego_features.view(B, N, -1)
        live_b = live_mask.unsqueeze(-1)  # (B, N, 1)
        denom_b = live_b.sum(dim=1).clamp_min(1.0)  # (B, 1)
        pooled = (ego_features_b * live_b).sum(dim=1) / denom_b
        self._value_features = pooled  # (B, d_embed)

        return logits, state

    def value_function(self) -> TensorType:
        if self._value_features is None:
            raise RuntimeError("value_function called before forward")
        return self.value_head(self._value_features).squeeze(-1)
