import torch.nn as nn


class ResidualConnectionLayer(nn.Module):

    def __init__(self, norm, dr_rate=0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, x, sub_layer):
        out = x
        out = self.norm(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x
        return out


class IdentityResidualLayer(nn.Module):
    """Residual layer that passes input through unchanged (identity skip connection)."""
    def __init__(self):
        super(IdentityResidualLayer, self).__init__()

    def forward(self, x, sub_layer, *args, **kwargs):
        return x + sub_layer(x, *args, **kwargs)


class NoResidualButSameForward(nn.Module):
    """
    It does not do a residual connection, but it does the same forward pass as the sub_layer.
    It also does the normalization if norm is not None or nn.Identity().
    """
    def __init__(self, norm):
        super(NoResidualButSameForward, self).__init__()
        self.norm = norm

    def forward(self, x, sub_layer, *args, **kwargs):
        # Just to align with the grammar of the placeholder where this class is replaced
        if self.norm is not None:
            x = self.norm(x)
        return sub_layer(x, *args, **kwargs)
