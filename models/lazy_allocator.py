# Ray and RLlib
import copy

# import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2, ModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import ModelConfigDict, TensorType  # for type hints and annotations in functions
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper # for custom action distribution
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC, normc_initializer

# Python modules
import numpy as np
import gym
from typing import Dict, List, Union

# Custom modules
from models.transformer_modules.token_embedding import LinearEmbedding
from models.transformer_modules.multi_head_attention_layer import MultiHeadAttentionLayer
from models.transformer_modules.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from models.transformer_modules.encoder_block import EncoderBlock
# from models.transformer_modules.decoder_block import DecoderBlock
from models.transformer_modules.decoder_block import CustomDecoderBlock as DecoderBlock
from models.transformer_modules.encoder import Encoder
from models.transformer_modules.decoder import Decoder, DecoderPlaceholder
from models.transformer_modules.pointer_net import GaussianActionDistGenerator, GaussianActionDistPlaceholder
from models.transformer_modules.pointer_net import MeanGenerator, PointerPlaceholder, FakeMeanGenerator
# For control env
from models.transformer_modules.pointer_net import GaussControlDistGenerator

# PyTorch's modules
# import torch.nn.functional as F
import torch
import torch.nn as nn
# torch, nn = try_import_torch()  # This is a wrapper for importing torch and torch.nn in a try-except block in rl-lib
# try_import_torch() makes my IDE confused about the type hints and annotations in functions; Particularly, __call__.


class LazinessAllocatorDiscrete(nn.Module):

    def __init__(
            self,
            src_embed,
            encoder,
            decoder,
            generator,
    ):
        super().__init__()

        self.src_embed = src_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        # TODO: Currently we don't support Null-Action output.
        #       If you want it, you need to add a learnable token as a virtual agent
        #       before or at the input of the pointer network.

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)

    def forward(self, src_dict: Dict[str, TensorType]):
        # Get data
        agent_embeddings = src_dict["agent_embeddings"]  # (batch_size, num_agents_max==seq_len_src, d_subobs)
        pad_tokens = src_dict["pad_tokens"]  # (batch_size, num_agents_max==seq_len_src)

        # Get masks
        # _Encoder masks
        # src_mask: shape (batch_size, seq_len_src, seq_len_src); Be careful: 'NO HEAD DIM' here!!
        src_mask = self.make_src_mask(pad_tokens)
        # _Decoder masks
        tgt_mask = None  # tgt_mask is not used in this implementation
        context_token = torch.zeros_like(pad_tokens[:, 0:1])  # (batch_size, 1); it's 2-D!!!
        src_tgt_mask = self.make_src_tgt_mask(pad_tokens, context_token)  # (batch_size, 1, seq_len_src)

        # Encoder
        # encoder_out: shape: (batch_size, src_seq_len, d_embed)
        # A set of agent embeddings encoded; permutation invariant
        # unsqueeze(1) has been applied to src_mask to add head dimension for broadcasting in the MHA layer
        encoder_out = self.encode(agent_embeddings, src_mask.unsqueeze(1))

        # Decoder
        # Get the context vector; here, we just average the encoder output
        # h_c_N: shape: (batch_size, 1, d_embed_context);  d_embed_context == 3 * d_embed (not true right now)
        h_c_N = self.get_context_node(embeddings=encoder_out, pad_tokens=pad_tokens, use_embeddings_mask=True)
        # decoder_out: (batch_size, tgt_seq_len, d_embed_context)
        # tgt_seq_len == 1 in our case
        decoder_out = self.decode(h_c_N, encoder_out, tgt_mask, src_tgt_mask.unsqueeze(1))  # h_c^(N+1)

        # Generators: query==decoder_out; key==encoder_out; return==mean/std
        if self.use_deterministic_action_dist:
            # out: (batch_size, tgt_seq_len, seq_len_src) == (batch_size, 1, num_agents_max)
            out = self.deterministic_action_dist_generator(query=decoder_out, key=encoder_out, mask=src_tgt_mask)
            # out: (batch_size, seq_len_src) == (batch_size, num_agents_max) == (batch_size, num_outputs)
            out = out.squeeze(1)  # (batch_size, num_outputs)
        else:
            # out_mean: (batch_size, tgt_seq_len, seq_len_src) == (batch_size, 1, num_agents_max)
            # out_std: (batch_size, tgt_seq_len, seq_len_src) == (batch_size, 1, num_agents_max)
            out_mean, out_std = self.gaussian_action_dist_generator(query=decoder_out, key=encoder_out,
                                                                    mask=src_tgt_mask)
            # Concatenate the mean and std along the last dimension (squeeze(1) on both tensors)
            # out: (batch_size, 2 * seq_len_src) == (batch_size, 2 * num_agents_max) == (batch_size, num_outputs)
            out = torch.cat((out_mean.squeeze(1), out_std.squeeze(1)), dim=-1)  # (batch_size, num_outputs)
            assert out.shape == (agent_embeddings.shape[0], 2 * agent_embeddings.shape[1]), \
                "The shape of the output tensor is not correct."  # TODO: remove this once the model is stable

        # Return
        # out: (batch_size, src_seq_len)
        return out, decoder_out, h_c_N


    def get_context_node(self):
        pass

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask  # (batch_size, seq_len_src, seq_len_src)

    def make_src_tgt_mask(self, src, tgt):
        # src: key/value; tgt: query
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask  # (batch_size, seq_len_tgt, seq_len_src)

    def make_pad_mask(self, query, key, pad_idx=1, dim_check=False):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        # If input_token==pad_idx, then the mask value is 0, else 1
        # In the MHA layer, (no attention) == (attention_score: -inf) == (mask value is 0) == (input_token==pad_idx)
        # WARNING: Choose pad_idx carefully, particularly about the data type (e.g. float, int, ...)

        # Check if the query and key have the same dimension
        if dim_check:
            assert len(query.shape) == 2, "query must have 2 dimensions: (n_batch, query_seq_len)"
            assert len(key.shape) == 2, "key must have 2 dimensions: (n_batch, key_seq_len)"
            assert query.size(0) == key.size(0), "query and key must have the same batch size"

        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1)  # (n_batch, 1, key_seq_len); on the same device as key
        key_mask = key_mask.repeat(1, query_seq_len, 1)  # (n_batch, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(2)  # (n_batch, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, key_seq_len)  # (n_batch, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask  # output shape: (n_batch, query_seq_len, key_seq_len)  # Keep in mind: 'NO HEADING DIM' here!!


class LazinessAllocator(nn.Module):
    # TODO: model output does not really agree with the action distribution
    #       consider using a clipping in the generators and proper masking
    def __init__(
        self,
        src_embed,
        encoder,
        decoder,
        generator,
    ):

        super().__init__()
        # nn.Module.__init__(self)

        # Temp: action distribution
        if isinstance(generator, (GaussianActionDistGenerator, GaussianActionDistPlaceholder)):
            self.use_deterministic_action_dist = False
        elif isinstance(generator, (MeanGenerator, PointerPlaceholder)):
            self.use_deterministic_action_dist = True
        elif isinstance(generator, (GaussControlDistGenerator, GaussianActionDistPlaceholder)):
            self.use_deterministic_action_dist = False
        else:
            raise ValueError("generator must be an instance of GaussianActionDistGenerator or MeanGenerator")

        # Define layers
        self.src_embed = src_embed
        self.d_v = src_embed.in_features
        # self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        # Action branch: Gaussian pointer network for Gaussian action distribution
        self.gaussian_action_dist_generator = generator if not self.use_deterministic_action_dist else None
        self.deterministic_action_dist_generator = generator if self.use_deterministic_action_dist else None

        # custom layers if needed
        #

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)

    def forward(
        self,
        src_dict: Dict[str, TensorType],
    ):
        # Get data
        agent_embeddings = src_dict["agent_embeddings"][:, :, :self.d_v]  # (batch_size, num_agents_max==seq_len_src, d_subobs)
        pad_tokens = src_dict["pad_tokens"]  # (batch_size, num_agents_max==seq_len_src)
        # net_topology = src_dict["net_topology"]  # (batch_size, num_agents_max==seq_len_src, seq_len_src)

        # Get masks
        # _Encoder masks
        # src_mask: shape (batch_size, seq_len_src, seq_len_src); Be careful: 'NO HEAD DIM' here!!
        src_mask = self.make_src_mask(pad_tokens)
        # _Decoder masks
        tgt_mask = None # tgt_mask is not used in this implementation
        context_token = torch.zeros_like(pad_tokens[:, 0:1])  # (batch_size, 1); it's 2-D!!!
        # tgt_mask (q_mask): context_token (batch_size, 1), src_mask (k/v_mask): pad_tokens (batch_size, seq_len_src)
        src_tgt_mask = self.make_src_tgt_mask(pad_tokens, context_token)  # (batch_size, 1, seq_len_src)

        # Encoder
        # encoder_out: shape: (batch_size, src_seq_len, d_embed)
        # A set of agent embeddings encoded; permutation invariant
        # unsqueeze(1) has been applied to src_mask to add head dimension for broadcasting in the MHA layer
        encoder_out = self.encode(agent_embeddings, src_mask.unsqueeze(1))

        # Decoder
        # Get the context vector; here, we just average the encoder output
        # h_c_N: shape: (batch_size, 1, d_embed_context);  d_embed_context == 3 * d_embed (not true right now)
        h_c_N = self.get_context_node(embeddings=encoder_out, pad_tokens=pad_tokens, use_embeddings_mask=True)
        # decoder_out: (batch_size, tgt_seq_len, d_embed_context)
        # tgt_seq_len == 1 in our case
        decoder_out = self.decode(h_c_N, encoder_out, tgt_mask, src_tgt_mask.unsqueeze(1))  # h_c^(N+1)

        # Generators: query==decoder_out; key==encoder_out; return==mean/std
        if self.use_deterministic_action_dist:
            # out: (batch_size, tgt_seq_len, seq_len_src) == (batch_size, 1, num_agents_max)
            out = self.deterministic_action_dist_generator(query=decoder_out, key=encoder_out, mask=src_tgt_mask)
            # out: (batch_size, seq_len_src) == (batch_size, num_agents_max) == (batch_size, num_outputs)
            out = out.squeeze(1)  # (batch_size, num_outputs)
        else:
            # out_mean: (batch_size, tgt_seq_len, seq_len_src) == (batch_size, 1, num_agents_max)
            # out_std: (batch_size, tgt_seq_len, seq_len_src) == (batch_size, 1, num_agents_max)
            out_mean, out_std = self.gaussian_action_dist_generator(query=decoder_out, key=encoder_out, mask=src_tgt_mask)
            # Concatenate the mean and std along the last dimension (squeeze(1) on both tensors)
            # out: (batch_size, 2 * seq_len_src) == (batch_size, 2 * num_agents_max) == (batch_size, num_outputs)
            out = torch.cat((out_mean.squeeze(1), out_std.squeeze(1)), dim=-1)  # (batch_size, num_outputs)
            assert out.shape == (agent_embeddings.shape[0], 2 * agent_embeddings.shape[1]), \
                "The shape of the output tensor is not correct."  # TODO: remove this once the model is stable

        # Return
        # out: (batch_size, src_seq_len)
        return out, decoder_out, h_c_N

    # @staticmethod
    def get_context_node(self, embeddings, pad_tokens, use_embeddings_mask=True, debug=False):
        # embeddings: shape (batch_size, num_agents_max==seq_len_src, data_size==d_embed_input)
        # pad_tokens: shape (batch_size, num_agents_max==seq_len_src)

        # Obtain batch_size, num_agents, data_size from embeddings
        batch_size, num_agents_max, data_size = embeddings.shape

        if use_embeddings_mask:
            # Expand the dimensions of pad_tokens to match the shape of embeddings
            mask = pad_tokens.unsqueeze(-1).expand_as(embeddings)  # (batch_size, num_agents_max, data_size)

            # Replace masked values with zero for the average computation
            # embeddings_masked: (batch_size, num_agents_max, data_size)
            embeddings_masked = torch.where(mask == 1, embeddings, torch.zeros_like(embeddings))

            # Compute the sum and count non-zero elements
            embeddings_sum = torch.sum(embeddings_masked, dim=1, keepdim=True)  # (batch_size, 1, data_size)
            embeddings_count = torch.sum((mask == 0), dim=1, keepdim=True).float()  # (batch_size, 1, data_size)

            # Check if there is any sample where all agents are padded
            if debug:
                if torch.any(embeddings_count == 0):
                    raise ValueError("All agents are padded in at least one sample.")

            # Compute the average embeddings, only for non-masked elements
            embeddings_avg = embeddings_sum / embeddings_count
        else:
            # Compute the average embeddings: shape (batch_size, 1, data_size)
            embeddings_avg = torch.mean(embeddings, dim=1, keepdim=True)  # num_agents_max dim is reduced

        # Construct context embedding: shape (batch_size, 1, d_embed_context)
        # The resulting tensor, h_c, will have shape (batch_size, 1, d_embed_context)
        # Concatenate the additional info to h_c, if you need more info for the context vector.
        h_c = embeddings_avg
        # This represents the graph embeddings.
        # It summarizes the information of all nodes in the graph.

        return h_c  # (batch_size, 1, d_embed_context)

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask  # (batch_size, seq_len_src, seq_len_src)

    # def make_tgt_mask_outdated(self, tgt):
    #     pad_mask = self.make_pad_mask(tgt, tgt)
    #     seq_mask = self.make_subsequent_mask(tgt, tgt)
    #     mask = pad_mask & seq_mask
    #     return pad_mask & seq_mask

    def make_src_tgt_mask(self, src, tgt):
        # src: key/value; tgt: query
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask  # (batch_size, seq_len_tgt, seq_len_src)

    def make_pad_mask(self, query, key, pad_idx=1, dim_check=False):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        # If input_token==pad_idx, then the mask value is 0, else 1
        # In the MHA layer, (no attention) == (attention_score: -inf) == (mask value is 0) == (input_token==pad_idx)
        # WARNING: Choose pad_idx carefully, particularly about the data type (e.g. float, int, ...)

        # Check if the query and key have the same dimension
        if dim_check:
            assert len(query.shape) == 2, "query must have 2 dimensions: (n_batch, query_seq_len)"
            assert len(key.shape) == 2, "key must have 2 dimensions: (n_batch, key_seq_len)"
            assert query.size(0) == key.size(0), "query and key must have the same batch size"

        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1)  # (n_batch, 1, key_seq_len); on the same device as key
        key_mask = key_mask.repeat(1, query_seq_len, 1)  # (n_batch, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(2)  # (n_batch, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, key_seq_len)  # (n_batch, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask  # output shape: (n_batch, query_seq_len, key_seq_len)  # Keep in mind: 'NO HEADING DIM' here!!

    # def make_subsequent_mask(self, query, key):
    #     query_seq_len, key_seq_len = query.size(1), key.size(1)
    #
    #     tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')  # lower triangle without diagonal
    #     mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
    #     return mask


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
        """
        # Configuration template
        # "custom_model_config": {
        #     "d_subobs": 6,
        #     "d_embed_input": 128,
        #     "d_embed_context": 128,
        #     "d_model": 128,
        #     "d_model_decoder": 128,
        #     "n_layers_encoder": 3,
        #     "n_layers_decoder": 2,
        #     "num_heads": 8,
        #     "d_ff": 512,
        #     "d_ff_decoder": 512,
        #     "clip_action_mean": 1.0,  # [0, clip_action_mean]
        #     "clip_action_log_std": 10.0,  # [-clip_action_log_std, -2]
        #     "dr_rate": 0.1,
        #     "norm_eps": 1e-5,
        #     "is_bias": True,
        #     "share_layers": True,
        #     "use_residual_in_decoder": True,
        #     "use_FNN_in_decoder": True,
        #     "use_deterministic_action_dist": False,
        # },
        """

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Get model config
        if model_config is not None:
            cfg = model_config["custom_model_config"]
            share_layers = cfg["share_layers"] if "share_layers" in cfg else True
            d_subobs = cfg["d_subobs"] if "d_subobs" in cfg else ValueError("d_subobs must be specified")
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
                # DeprecationWarning: ignore_residual_in_decoder is deprecated; use_residual_in_decoder is used instead
                print("DeprecationWarning: ignore_residual_in_decoder is deprecated; use use_residual_in_decoder instead")
                use_FNN_in_decoder = use_residual_in_decoder
                print("use_FNN_in_decoder is set to use_residual_in_decoder as {}".format(use_residual_in_decoder))
            if use_residual_in_decoder != use_FNN_in_decoder:
                # Warning: use_residual_in_decoder != use_FNN_in_decoder; but don't change it
                warning_text = "Warning: use_residual_in_decoder != use_FNN_in_decoder; but don't change it"
                for i in range(7):
                    print(("%"*i) + warning_text + ("%"*i))
            if n_layers_decoder >= 2 and not use_residual_in_decoder:
                # Warning: multiple decoder blocks often require residual connections
                warning_text = "Warning: multiple decoder blocks often require residual connections"
                for i in range(7):
                    print(("%"*i) + warning_text + ("%"*i))
        else:
            raise ValueError("model_config must be specified")

        # 1. Define layers

        # 1-1. Module Level: Encoder
        # Need an embedding layer for the input; 2->128 in the case of Kool2019
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
        # 1-2. Module Level: Decoder
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

        # 1-3. Block Level
        encoder_block = EncoderBlock(
            self_attention=copy.deepcopy(mha_encoder),
            position_ff=copy.deepcopy(position_ff_encoder),
            norm=copy.deepcopy(norm_encoder),
            dr_rate=dr_rate,
        )
        decoder_block = DecoderBlock(
            self_attention=None,  # No self-attention in the decoder_block in this case!
            cross_attention=copy.deepcopy(mha_decoder),
            position_ff=position_ff_decoder,  # No position-wise FFN in the decoder_block in most cases!
            norm=copy.deepcopy(norm_decoder),
            dr_rate=dr_rate,
            efficient=not use_residual_in_decoder,
        )

        # 1-4. Transformer Level (Encoder + Decoder + Generator)
        encoder = Encoder(
            encoder_block=encoder_block,
            n_layer=n_layers_encoder,
            norm=copy.deepcopy(norm_encoder),
        )
        decoder = Decoder(
            decoder_block=decoder_block,
            n_layer=n_layers_decoder,
            norm=copy.deepcopy(norm_decoder),
            # norm=nn.Identity(),
        )
        if use_deterministic_action_dist:
            # generator = MeanGenerator(
            #     d_model=d_model,
            #     q_fc=nn.Linear(d_embed_context, d_model, is_bias),
            #     k_fc=nn.Linear(d_embed_input, d_model, is_bias),
            #     clip_value=clip_action_mean,
            #     dr_rate=dr_rate,
            # )
            generator = FakeMeanGenerator(  # is actually a continuous Gaussian dist with very small std
                d_model=d_model,
                q_fc=nn.Linear(d_embed_context, d_model, is_bias),
                k_fc=nn.Linear(d_embed_input, d_model, is_bias),
                clip_value_mean=clip_action_mean,
                clip_value_std=clip_action_log_std,  # all log_std-s are set to this value
                dr_rate=dr_rate,
            )
        else:  # Gaussian action distribution
            generator = GaussianActionDistGenerator(
                d_model=d_model,
                q_fc=nn.Linear(d_embed_context, d_model, is_bias),
                k_fc=nn.Linear(d_embed_input, d_model, is_bias),
                clip_value_mean=clip_action_mean,
                clip_value_std=clip_action_log_std,
                dr_rate=dr_rate,
            )  # outputs a probability distribution over the input sequence

        action_size = action_space.shape[0]  # it gives d given that action_space is a Box of d dimensions
        # if use_deterministic_action_dist:
        #     assert num_outputs == action_size, "num_outputs must be action_size; use deterministic action distribution"
        # else:
        #     assert num_outputs == 2 * action_size, "num_outputs must be 2 * action_size"

        assert num_outputs == 2 * action_size, "num_outputs must be action_size; use deterministic action distribution"

        # 2. Define policy network
        self.policy_network = LazinessAllocator(
            src_embed=input_embed,
            encoder=encoder,
            decoder=decoder,
            generator=generator,
        )

        # 3. Define value network
        self.values = None
        self.share_layers = share_layers
        if not self.share_layers:
            self.value_network = LazinessAllocator(
                src_embed=copy.deepcopy(input_embed),
                encoder=copy.deepcopy(encoder),
                # decoder=copy.deepcopy(decoder),
                decoder=DecoderPlaceholder(),
                # TODO: try this although PPO uses state-value function
                #      self.values should use h_c_N instead of h_c_N1 with a different value branch
                #      If so, the decoder is not used in the value network
                # generator=copy.deepcopy(generator),
                generator=GaussianActionDistPlaceholder() #if not use_deterministic_action_dist else PointerPlaceholder(),
            )

        self.value_branch = nn.Sequential(
            nn.Linear(in_features=d_embed_context, out_features=d_embed_context),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(in_features=d_embed_context, out_features=1),  # state-value function
        )

        # self.action_mean_branch = nn.Sequential(
        #     nn.Linear(in_features=d_embed_context, out_features=d_embed_context),
        #     nn.ReLU(),
        #     # nn.Tanh(),
        #     nn.Linear(in_features=d_embed_context, out_features=1),  # action-value function
        # )

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        obs = input_dict["obs"]

        # Check validity of the data incoming from the environment
        # TODO: deactivate this for better performance once the model is stable in your trials
        self._validate_obs(obs)

        # x: (batch_size, 2 * action_size); action_size==num_UAVs==num_agents_max
        # h_c_N1: (batch_size, 1, d_embed_context)
        if self.share_layers:
            x, h_c_N1, h_c_N = self.policy_network(obs)  # x: mean and std of the Gaussian distribution
            # self.values = h_c_N.squeeze(1)  # (batch_size, d_embed_context)
            self.values = h_c_N1.squeeze(1)  # (batch_size, d_embed_context)  # TODO: try this but not sure
        else:
            x, _, _ = self.policy_network(obs)
            # input of the value branch is h_c_N instead of h_c_N1 in the case of shared layers
            self.values = self.value_network(obs)[2].squeeze(1)  # self.values: (batch_size, d_embed_context)


        # Check batch dimension size for debugging purposes
        # if x.shape[0] != 1:
        #     print(f"batch size = {x.shape[0]} != 1")
        #     print("Stop!")
        # else:
        #     print(f"batch size = {x.shape[0]} == 1")

        # Return
        return x, state

    def _validate_obs(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> None:

        agent_embeddings = obs["agent_embeddings"]  # (batch_size, num_agents_max, d_subobs)
        net_topology = obs["net_topology"]  # (batch_size, num_agents_max, num_agents_max)
        pad_tokens = obs["pad_tokens"]  # (batch_size, num_agents_max)

        num_agents_max = pad_tokens.shape[1]

        # Get mask to only see the padded tokens (pad_idx=1)
        pad_mask = pad_tokens == 1  # (batch_size, num_agents_max)

        # Check if agent_embeddings is a torch.Tensor
        assert isinstance(agent_embeddings, torch.Tensor), "agent_embeddings must be a torch.Tensor"
        # Check if agent_embeddings is a 3D tensor
        assert agent_embeddings.ndim == 3, "agent_embeddings must be a 3D tensor"
        assert agent_embeddings.shape[1] == num_agents_max, \
            "agent_embeddings.shape[1] in agent_embeddings must be equal to num_agents_max"
        # assert agent_embeddings.shape[2] == self.model_config["custom_model_config"]["d_subobs"], \
        #     "agent_embeddings.shape[2] in agent_embeddings must be equal to d_subobs"
        # Check if all the padded data is 0
        assert torch.all(agent_embeddings[pad_mask, :] == 0), "all the padded data in agent_embeddings must be 0"

        # Check if net_topology is a torch.Tensor
        assert isinstance(net_topology, torch.Tensor), "net_topology must be a torch.Tensor"
        # Check if net_topology is a 3D tensor
        assert net_topology.ndim == 3, "net_topology must be a 3D tensor"

        # Check if pad_tokens is a torch.Tensor
        assert isinstance(pad_tokens, torch.Tensor), "obs must be a torch.Tensor"
        # Check if pad_tokens is a 2D tensor
        assert pad_tokens.ndim == 2, "pad_tokens must be a 2D tensor"
        assert pad_tokens.shape[1] == num_agents_max, \
            "pad_tokens.shape[1] in pad_tokens must be equal to num_agents_max"

    def value_function(self) -> TensorType:
        out = self.value_branch(self.values).squeeze(-1)  # (batch_size,)
        # if out.shape[0] != 1 and out.shape[0] != 32:
        #     print(f"batch size = {out.shape[0]} != 1")
        #     print("Stop!")
        #     print("Debugging...")

        return out


class MyMLPModel(TorchModelV2, nn.Module):
    """
    MLP
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Get configuration for this custom model
        '''
        # Config example:
        custom_model_config = {
            "custom_model_config": {
                "fc_sizes": [128, 64],
                "fc_activation": "relu",
                "value_fc_sizes": [128, 64],
                "value_fc_activation": "relu",
                "is_same_shape": False,  # avoid using this; let it be False unless you know what you are doing
                "share_layers": False,
                "deterministic_action": False,
            }
        }
        '''
        cfg = model_config["custom_model_config"]
        #
        if "is_same_shape" in cfg:
            # TODO: this may cause some confusion...
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

        # Define observation size
        self.obs_size = get_preprocessor(obs_space)(obs_space).size  # (6 * num_task_max)

        # Initialize logits
        self._logits = None
        # Holds the current "base" output (before logits/value_out layer).
        self._features = None
        self._values = None

        # Build the Module from fcs + 2xfc (action + value outs).
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        assert prev_layer_size == 100, "prev_layer_size must be 100"  # TODO: remove this line later
        # prev_layer_size = 120  # input size of fc_net  # TODO: static input size used for now
        # Create layers and get fc_net
        for size in self.fc_sizes[:]:
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
            # Get VALUE fc layers
            value_layers = []
            prev_value_layer_size = int(np.product(obs_space.shape))
            # Create layers and get fc_net
            for size in self.value_fc_sizes[:]:
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

        # Get last layers
        self.last_size = self.fc_sizes[-1]
        self.last_value_size = self.value_fc_sizes[-1] if not self.share_layers else self.last_size
        # Policy network's last layer
        if self.deterministic_action:
            self.action_branch = nn.Linear(self.last_size, num_outputs)
        else:
            assert num_outputs % 2 == 0, "num_outputs must be even!"
            assert num_outputs==action_space.shape[0], "num_outputs must be equal to action_space.shape[0]"
            mean_size = int(num_outputs/2)  # we use constant variance for now
            self.action_branch_mean = nn.Linear(self.last_size, mean_size)
            self.action_branch_logstd = nn.Linear(self.last_size, mean_size)
        # Value network's last layer
        self.value_branch = nn.Linear(self.last_value_size, 1)

    @override(ModelV2)
    def value_function(self):
        assert self._values is not None, "must call forward() first"
        out = self.value_branch(self._values).squeeze(1)
        return out

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Fetch the observation
        # obs = input_dict["obs_flat"]
        obs = input_dict["obs"]  # the obs already has been flattened in the env

        # Forward pass through fc_net
        self._features = self.fc_net(obs)

        # If not sharing layers, forward pass through value_fc_net
        # Else, _values are the same as _features
        if self.share_layers:
            self._values = self._features
        else:
            self._values = self.value_fc_net(obs)

        # Calculate logits
        if self.deterministic_action:
            self._logits = self.action_branch(self._features)
        else:
            action_means = self.action_branch_mean(self._features)
            # place action_means in [0, 1.05] range using tanh
            upper_bound = 1.05
            action_means = torch.tanh(action_means) * (upper_bound/2) + (upper_bound/2)

            # action_stds = (-10) * torch.ones_like(action_means, dtype=action_means.dtype, requires_grad=False)
            log_std_min = -10
            log_std_max = -1.5
            action_stds = self.action_branch_logstd(self._features)
            action_stds = torch.tanh(action_stds) * (log_std_max - log_std_min) / 2 + (log_std_max + log_std_min) / 2
            self._logits = torch.cat([action_means, action_stds], dim=1)


        # Check logits shape (batch_size, num_outputs)
        assert self._logits.shape == (obs.shape[0], self.num_outputs), "logits shape is not correct!"

        # # Apply action masking
        # pad_tasks = input_dict["obs"]["pad_tokens"]  # (batch_size, num_task_max); 0: not padded, 1: padded
        # done_tasks = input_dict["obs"]["completion_tokens"]  # (batch_size, num_task_max); 0: not done, 1: done
        # # Get action mask
        # action_mask = torch.zeros_like(self._logits)
        # action_mask[pad_tasks == 1] = FLOAT_MIN
        # action_mask[done_tasks == 1] = FLOAT_MIN
        # Apply action mask
        # self._logits = self._logits + action_mask

        # JUST FOR DEBUGGING
        # if obs.shape[0] != 1:
        #     print(f"batch size = {obs.shape[0]} != 1")
        #     print("Stop!")
        # else:
        #     print(f"batch size = {obs.shape[0]} == 1")

        # Return logits and state
        return self._logits, state


class MyRLlibTorchWrapperControl(TorchModelV2, nn.Module):

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        """
        # Configuration template
        # "custom_model_config": {
        #     "d_subobs": 4,
        #     "d_embed_input": 128,
        #     "d_embed_context": 128,
        #     "d_model": 128,
        #     "d_model_decoder": 128,
        #     "n_layers_encoder": 3,
        #     "n_layers_decoder": 1,
        #     "num_heads": 8,
        #     "d_ff": 512,
        #     "d_ff_decoder": 512,
        #     "clip_action_control": 8/15,  # [-u_max, u_max]
        #     "clip_action_log_std": 10.0,  # [-clip_action_log_std, -2]
        #     "dr_rate": 0,
        #     "norm_eps": 1e-5,
        #     "is_bias": False,
        #     "share_layers": True,
        #     "use_residual_in_decoder": True,
        #     "use_FNN_in_decoder": True,
        #     "use_deterministic_action_dist": False,
        # },
        """

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Get model config
        if model_config is not None:
            cfg = model_config["custom_model_config"]
            share_layers = cfg["share_layers"] if "share_layers" in cfg else True
            d_subobs = cfg["d_subobs"] if "d_subobs" in cfg else ValueError("d_subobs must be specified")
            if d_subobs >= 6:
                ValueError("d_subobs must be less than 6 in the control model")
            d_embed_input = cfg["d_embed_input"] if "d_embed_input" in cfg else 128
            d_embed_context = cfg["d_embed_context"] if "d_embed_context" in cfg else 128
            d_model = cfg["d_model"] if "d_model" in cfg else 128
            d_model_decoder = cfg["d_model_decoder"] if "d_model_decoder" in cfg else 128
            n_layers_encoder = cfg["n_layers_encoder"] if "n_layers_encoder" in cfg else 3
            n_layers_decoder = cfg["n_layers_decoder"] if "n_layers_decoder" in cfg else 1
            h = cfg["num_heads"] if "num_heads" in cfg else 8
            d_ff = cfg["d_ff"] if "d_ff" in cfg else 512
            d_ff_decoder = cfg["d_ff_decoder"] if "d_ff_decoder" in cfg else 512
            clip_action_control = cfg["clip_action_control"] if "clip_action_control" in cfg else 8/15
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
                # DeprecationWarning: ignore_residual_in_decoder is deprecated; use_residual_in_decoder is used instead
                print("DeprecationWarning: ignore_residual_in_decoder is deprecated; use use_residual_in_decoder instead")
                use_FNN_in_decoder = use_residual_in_decoder
                print("use_FNN_in_decoder is set to use_residual_in_decoder as {}".format(use_residual_in_decoder))
            if use_residual_in_decoder != use_FNN_in_decoder:
                # Warning: use_residual_in_decoder != use_FNN_in_decoder; but don't change it
                warning_text = "Warning: use_residual_in_decoder != use_FNN_in_decoder; but don't change it"
                for i in range(7):
                    print(("%"*i) + warning_text + ("%"*i))
            if n_layers_decoder >= 2 and not use_residual_in_decoder:
                # Warning: multiple decoder blocks often require residual connections
                warning_text = "Warning: multiple decoder blocks often require residual connections"
                for i in range(7):
                    print(("%"*i) + warning_text + ("%"*i))
        else:
            raise ValueError("model_config must be specified")

        # 1. Define layers

        # 1-1. Module Level: Encoder
        # Need an embedding layer for the input; 2->128 in the case of Kool2019; 6 -> n in our case
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
        # 1-2. Module Level: Decoder
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

        # 1-3. Block Level
        encoder_block = EncoderBlock(
            self_attention=copy.deepcopy(mha_encoder),
            position_ff=copy.deepcopy(position_ff_encoder),
            norm=copy.deepcopy(norm_encoder),
            dr_rate=dr_rate,
        )
        decoder_block = DecoderBlock(
            self_attention=None,  # No self-attention in the decoder_block in this case!
            cross_attention=copy.deepcopy(mha_decoder),
            position_ff=position_ff_decoder,  # No position-wise FFN in the decoder_block in most cases!
            norm=copy.deepcopy(norm_decoder),
            dr_rate=dr_rate,
            efficient=not use_residual_in_decoder,
        )

        # 1-4. Transformer Level (Encoder + Decoder + Generator)
        encoder = Encoder(
            encoder_block=encoder_block,
            n_layer=n_layers_encoder,
            norm=copy.deepcopy(norm_encoder),
        )
        decoder = Decoder(
            decoder_block=decoder_block,
            n_layer=n_layers_decoder,
            norm=copy.deepcopy(norm_decoder),
            # norm=nn.Identity(),
        )
        generator = GaussControlDistGenerator(
            d_model=d_model,
            q_fc=nn.Linear(d_embed_context, d_model, is_bias),
            k_fc=nn.Linear(d_embed_input, d_model, is_bias),
            clip_value_control=clip_action_control,
            clip_value_std=clip_action_log_std,
            dr_rate=dr_rate,
            ignore_std=use_deterministic_action_dist,
        )

        action_size = action_space.shape[0]  # it gives d given that action_space is a Box of d dimensions
        assert num_outputs == 2 * action_size, "num_outputs must be action_size; use deterministic action distribution"

        # 2. Define policy network
        self.policy_network = LazinessAllocator(
            src_embed=input_embed,
            encoder=encoder,
            decoder=decoder,
            generator=generator,
        )

        # 3. Define value network
        self.values = None
        self.share_layers = share_layers
        if not self.share_layers:
            self.value_network = LazinessAllocator(
                src_embed=copy.deepcopy(input_embed),
                encoder=copy.deepcopy(encoder),
                decoder=DecoderPlaceholder(),
                generator=GaussianActionDistPlaceholder() #if not use_deterministic_action_dist else PointerPlaceholder(),
            )

        self.value_branch = nn.Sequential(
            nn.Linear(in_features=d_embed_context, out_features=d_embed_context),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(in_features=d_embed_context, out_features=1),  # state-value function
        )

        # self.action_mean_branch = nn.Sequential(
        #     nn.Linear(in_features=d_embed_context, out_features=d_embed_context),
        #     nn.ReLU(),
        #     # nn.Tanh(),
        #     nn.Linear(in_features=d_embed_context, out_features=action_size),  # action-value function
        # )

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        obs = input_dict["obs"]

        # Check validity of the data incoming from the environment
        self._validate_obs(obs)

        # x: (batch_size, 2 * action_size); action_size==num_UAVs==num_agents_max
        # h_c_N1: (batch_size, 1, d_embed_context)
        if self.share_layers:
            x, h_c_N1, h_c_N = self.policy_network(obs)  # x: mean and std of the Gaussian distribution
            self.values = h_c_N.squeeze(1)  # (batch_size, d_embed_context)
            # self.values = h_c_N1.squeeze(1)  # (batch_size, d_embed_context)  # TODO: try this but not sure
        else:
            x, _, _ = self.policy_network(obs)
            # input of the value branch is h_c_N instead of h_c_N1 in the case of shared layers
            self.values = self.value_network(obs)[2].squeeze(1)  # self.values: (batch_size, d_embed_context)

        # Check batch dimension size for debugging purposes
        # if x.shape[0] != 1:
        #     print(f"batch size = {x.shape[0]} != 1")
        #     print("Stop!")
        # else:
        #     print(f"batch size = {x.shape[0]} == 1")

        # Return
        return x, state

    def _validate_obs(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> None:

        agent_embeddings = obs["agent_embeddings"]  # (batch_size, num_agents_max, d_subobs)
        net_topology = obs["net_topology"]  # (batch_size, num_agents_max, num_agents_max)
        pad_tokens = obs["pad_tokens"]  # (batch_size, num_agents_max)

        num_agents_max = pad_tokens.shape[1]

        # Get mask to only see the padded tokens (pad_idx=1)
        pad_mask = pad_tokens == 1  # (batch_size, num_agents_max)

        # Check if agent_embeddings is a torch.Tensor
        assert isinstance(agent_embeddings, torch.Tensor), "agent_embeddings must be a torch.Tensor"
        # Check if agent_embeddings is a 3D tensor
        assert agent_embeddings.ndim == 3, "agent_embeddings must be a 3D tensor"
        assert agent_embeddings.shape[1] == num_agents_max, \
            "agent_embeddings.shape[1] in agent_embeddings must be equal to num_agents_max"
        # assert agent_embeddings.shape[2] == self.model_config["custom_model_config"]["d_subobs"], \
        #     "agent_embeddings.shape[2] in agent_embeddings must be equal to d_subobs"
        # Check if all the padded data is 0
        assert torch.all(agent_embeddings[pad_mask, :] == 0), "all the padded data in agent_embeddings must be 0"

        # Check if net_topology is a torch.Tensor
        assert isinstance(net_topology, torch.Tensor), "net_topology must be a torch.Tensor"
        # Check if net_topology is a 3D tensor
        assert net_topology.ndim == 3, "net_topology must be a 3D tensor"

        # Check if pad_tokens is a torch.Tensor
        assert isinstance(pad_tokens, torch.Tensor), "obs must be a torch.Tensor"
        # Check if pad_tokens is a 2D tensor
        assert pad_tokens.ndim == 2, "pad_tokens must be a 2D tensor"
        assert pad_tokens.shape[1] == num_agents_max, \
            "pad_tokens.shape[1] in pad_tokens must be equal to num_agents_max"

    def value_function(self) -> TensorType:
        out = self.value_branch(self.values).squeeze(-1)  # (batch_size,)
        # if out.shape[0] != 1 and out.shape[0] != 32:
        #     print(f"batch size = {out.shape[0]} != 1")
        #     print("Stop!")
        #     print("Debugging...")
        return out


