import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from unet_components import SinusoidalPosEmb, Conv1dBlock, ResidualTemporalBlock, Downsample1d, Upsample1d
import einops


class TemporalUnet(nn.Module):
    def __init__(
            self, 
            horizon, 
            transition_dim,
            cond_dim, 
            dim=128,
            dim_mults = (1, 2, 4, 8),
            returns_conditon=True,
            condition_dropout=0.25,
            kernel_size=5, # maybe reduce kernel_size ? (horizon is much shorter, maybe not shorter than 3)
    ):
        super().__init__()

        # transition_dim : obs_dim
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        mish = True
        act_fn = nn.Mish()
        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_conditon
        self.condition_dropout = condition_dropout
        
        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout) 
            embed_dim = 2 * dim
        else:
            embed_dim = dim

        self.downs, self.ups = nn.ModuleList([]), nn.ModuleList([])
        num_resolutions = len(in_out)

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (num_resolutions -1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kenerl_size=kernel_size, mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        if not is_last:
            horizon = horizon // 2
        
        mid_dim = dim[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)

        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = i >= (num_resolutions-1)
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out*2, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1)
        )

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        # x : [batch, horizon, obs_dim]
        # loss : [batch, horizon]
        
        # change into [batch, obs_dim, horizon]
        x = einops.rearrange(x, 'b h t -> b t h')
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout: # train both conditional and unconditional via masking 
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout : # 100% without condition (for inference)
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)
        # [batch, horizon, obs_dim]
        x = einops.rearrange(x, 'b t h -> b h t')
        return x