import torch.nn as nn


class LinearEmbedding(nn.Module):

    def __init__(self, d_env, d_embed):
        super(LinearEmbedding, self).__init__()
        self.embedding = nn.Linear(d_env, d_embed)
        self.d_embed = d_embed
        self.in_features = d_env

    def forward(self,
                x  # shape: (batch_size, seq_len, d_env); d_env: dimension of ray environment observation to process
                ):
        out = self.embedding(x)
        return out  # shape: (batch_size, seq_len, d_embed)
