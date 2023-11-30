import torch.nn as nn
import torch

class InverseDynamics(nn.Module):
    def __init__(self, emb_dim, dropout_rate, embedding_lookup):
        super().__init__()
        self.embedding_lookup = embedding_lookup
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        self.mlp = nn.Sequential(nn.Linear(2 * emb_dim, 4 * emb_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(4*emb_dim, 4*emb_dim),
                                 nn.ReLU(),
                                 nn.Linear(4*emb_dim, emb_dim))
        self.num_items = len(self.embedding_lookup)
        self.lookuptesnor = torch.tensor(embedding_lookup, dtype=torch.float32)
        self.lookuptesnor = nn.Parameter(self.lookuptesnor, requires_grad=False)
        
    def get_dot_products(self, output_emb):
        # output_emb : [batch_size, emb_dim]
        # lookuptensor : [N, emb_dim]
        return torch.matmul(output_emb, self.lookuptesnor.T) # [batch_size, N]

    def forward(self, x):
        # x : concatenated obs
        output_emb = self.mlp(x)
        return self.get_dot_products(output_emb) # [batch_size, N]