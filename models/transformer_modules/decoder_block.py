import copy
import warnings
import torch.nn as nn

from models.transformer_modules.residual_connection_layer import ResidualConnectionLayer, NoResidualButSameForward


class CustomDecoderBlock(nn.Module):

    def __init__(self, cross_attention, norm, self_attention=None, position_ff=None, dr_rate=0, efficient=False):
        super(CustomDecoderBlock, self).__init__()
        # Initialize ResidualConnectionLayers
        if norm is None:
            norm = nn.Identity()

        if efficient:
            self.residual2 = NoResidualButSameForward(norm=copy.deepcopy(norm))
            self.residual3 = NoResidualButSameForward(norm=copy.deepcopy(norm))
        else:
            self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
            self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

        # Initialize self-attention layer and cross-attention layer
        self.self_attention = self_attention
        if self.self_attention is not None:
            warnings.warn("[CustomDecoderBlock] self_attention is not None but is not used in forward()")
        if cross_attention is None:
            raise ValueError("cross_attention is None; you must use a cross attention in this implementation")
        self.cross_attention = cross_attention
        # Initialize position-wise feed-forward network
        self.position_ff = position_ff

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        # tgt: (batch_size, tgt_seq_len, d_embed_context)
        # encoder_out: (batch_size, src_seq_len, d_embed_input)
        # tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        # src_tgt_mask: (batch_size, 1, tgt_seq_len==1, src_seq_len)

        # MHA layer with query as the output of the first MHA layer
        # Shape: (batch_size, tgt_seq_len, d_model)
        tgt = self.residual2(tgt, lambda tgt: self.cross_attention(query=tgt, key=encoder_out, value=encoder_out,
                                                                   mask=src_tgt_mask))
        # Position-wise feed-forward network, applied only if include_ffn is True
        # Shape: (batch_size, tgt_seq_len, d_model)
        if self.position_ff is not None:
            tgt = self.residual3(tgt, self.position_ff)

        # Return the output tensor
        # Shape: (batch_size, tgt_seq_len==1, d_embed_context)
        return tgt
