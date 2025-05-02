from torch import nn
import torch.nn.functional as F


class CrossAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, z, conditioning):
        z, _ = self.self_attn(z, z, z)
        z, _ = self.cross_attn(z, conditioning, conditioning)
        z = self.mlp(z)
        return z
