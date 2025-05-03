from torch import nn
import torch.nn.functional as F
import torch 


class CrossAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_it = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_tm = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.weights = nn.Parameter(torch.randn((1, 3)), requires_grad=True)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, z, z_i, z_m):
        z, _ = self.self_attn(z, z, z)
        z_it, _ = self.cross_attn_it(z, z_i, z_i)
        z_tm, _ = self.cross_attn_tm(z, z_m, z_m)

        # Apply attention weights
        lambda_1 = 1 / (w[0] ** 2 + 1e-8)
        lambda_2 = 1 / (w[1] ** 2 + 1e-8)
        lambda_3 = 1 / (w[2] ** 2 + 1e-8)

        z_total = lambda_1 * z + lambda_2 * z_it + lambda_3 * z_tm

        z = self.mlp(z_total)
        return z
