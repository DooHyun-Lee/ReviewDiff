from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
import torch
import torch.nn as nn
import math

class Transformernet(nn.Module):
    def __init__(self,
        in_channels, # 128
        model_channels, #128
        out_channels, # 128 
        dropout=0.1,
        config = None,
        config_name = 'bert-base-uncased'
        ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout

        time_embed_dim = model_channels*4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size)
        )
        self.input_up_proj = nn.Sequential(
            nn.Linear(in_channels, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        # TODO : make it conditional for reward as well
        self.input_transformers = BertEncoder(config)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand(1, -1))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_down_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, out_channels)
        )

    def forward(self, x, timesteps):
        '''
        x : input of x_t [bsize, horizon, emb_dim] 
        timesteps : t [bsize]
        return : expected x_0 [bsize, horizon, emb_dim]
        '''
        def timestep_embedding(timesteps, dim, max_period=10000):
            '''
            timesteps : [bsize]
            return : positional encoding [bsize, dim]
            '''
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            ).to(device=timesteps.device)
            args = timesteps[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

        # data after input_proj + time emb + positional emb
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        # 1)transformer model
        # 2)output_proj
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        return h