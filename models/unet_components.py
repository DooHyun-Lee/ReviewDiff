import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Covn1(dim, dim, 3, 2, 1)

    # half the horizon 
    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
    
    # double the horizon
    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    # modified conv1d with additional groupnorm and activation 
    # spatial dimension is maintained
    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8):
        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            act_fn
        )

    def forward(self, x):
        return self.block(x)

class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, mish), 
            Conv1dBlock(out_channels, out_channels, kernel_size, mish)
        ])

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        # input willl be embeded time from Unet
        self.time_mlp = nn.Sequential(
            act_fn, 
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1')
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
        if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        # x : [batch, inp_channel, horizon]
        # t : [batch, embed_dim]
        # ret : [batch, out_channel, horizon]
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)