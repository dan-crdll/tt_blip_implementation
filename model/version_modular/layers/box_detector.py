import torch 
from torch import nn 


class BoxDetector(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=1024):
        super().__init__()

        self.lpaa = nn.MultiheadAttention(embed_dim, 16, batch_first=True)
        self.agg_token = nn.Parameter(torch.randn(1, embed_dim), requires_grad=True)

        self.bbox_detector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, z):
        BSZ, *_ = z.shape
        agg_token = self.agg_token.repeat(BSZ, 1).unsqueeze(1)

        z = nn.functional.layer_norm(z, z.shape[-1:])
        agg_token = nn.functional.layer_norm(agg_token, agg_token.shape[-1:])

        agg_token, _ = self.lpaa(agg_token, z, z)
        agg_token = agg_token.squeeze(1)
        y = self.bbox_detector(agg_token)
        return y