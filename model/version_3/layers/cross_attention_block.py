from torch import nn
import torch.nn.functional as F
import torch 


class CrossAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_it = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_tm = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.mlp_it = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

        self.mlp_tm = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, z, z_i, z_m):
        z_t, _ = self.self_attn(z, z, z)
        z_it, _ = self.cross_attn_it(z_t, z_i, z_i)
        z_it = self.mlp_it(z_it)

        z_tm, _ = self.cross_attn_tm(z_t, z_m, z_m)
        z_tm = self.mlp_tm(z_tm)

        z_total = 0.4 * z + 0.3 * z_it + 0.3 * z_tm

        z = self.mlp(z_total)
        return z
