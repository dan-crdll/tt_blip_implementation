from torch import nn
import torch.nn.functional as F


class CrossAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_it = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_tm = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)


        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, z, z_i, z_m):
        z, _ = self.self_attn(z, z, z)
        z_it, _ = self.cross_attn_it(z, z_i, z_m)
        z_tm, _ = self.cross_attn_tm(z, z_m, z_m)
        z = self.mlp(z_it + z_tm)
        return z
