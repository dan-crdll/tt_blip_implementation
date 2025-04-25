from torch import nn
import torch.nn.functional as F

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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v):
        z, _ = self.multi_head_attn(q, k, v)
        x = self.norm_1(k + z)
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


"""
Fusion Layer for feature concatenation which uses self and 
cross attention encoders and computes contrastive loss
"""
class FusionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_encoders=1, num_decoders=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, batch_first=True)
        decoder_layer = CrossAttnEncoderLayer(embed_dim, num_heads, hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoders)
        self.decoder_img = CrossAttnEncoder(decoder_layer, num_decoders)
        self.decoder_txt = CrossAttnEncoder(decoder_layer, num_decoders)

    def forward(self, z):
        z_i, z_t, z_m = z
        BSZ, _, N = z_i.shape
        
        z_m = self.encoder(z_m)
        
        z_i = self.decoder_img(z_m, z_i)
        z_t = self.decoder_txt(z_m, z_t)

        return (z_i, z_t)