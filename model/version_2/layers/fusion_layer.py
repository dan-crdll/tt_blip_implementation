import torch 
from torch import nn 

"""
In this layer the feature from different modalities are fused

Inputs:
    - Features extracted for the two modalities (z_i, z_t, z_it)

Layer Description:
    Models Involved:
        - Cross Attention Layers for I-IT and T-IT fusion and then concatenation

Outputs: 
    - Concatenated (BSZ x 768) and cross-attended features
"""

class FusionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_decoders=1):
        super().__init__()
        decoder_layer = CrossAttnEncoderLayer(embed_dim, num_heads, hidden_dim, batch_first=True)
        self.cross_attn = CrossAttnEncoder(decoder_layer, num_decoders)
        self.self_attn = CrossAttnEncoder(decoder_layer, num_decoders)

    def forward(self, z_i, z_t, z_it):        
        z_it_cls = self.cross_attn(z_i, z_t)
        z = torch.cat([z_it_cls, z_it], 1)

        z = self.self_attn(z, z)
        z = nn.functional.adaptive_avg_pool1d(z.permute(0,2,1), 1).squeeze()

        return z
    

"""
Definition of cross attention encoder layers and wrapper
"""
class CrossAttnEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, batch_first=True):
        super().__init__()

        self.multi_head_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v):
        z, _ = self.multi_head_attn(q, k, v)
        x = self.norm_1(q + z)
        z = self.mlp(x)
        x = self.norm_2(x + z)
        return x
    

class CrossAttnEncoder(nn.Module):
    def __init__(self, encoder_layer, num_encoders):
        super().__init__()

        self.encoders = nn.ModuleList([
            encoder_layer for _ in range(num_encoders)
        ])
    
    def forward(self, z_q, z_kv):
        y = z_kv

        for encoder in self.encoders:
            y = encoder(z_q, y, y)
        return y