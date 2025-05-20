from torch import nn
import torch.nn.functional as F
import torch 


class GatedUnit(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.s1 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        self.s2 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        self.t1 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh()
        )

    def forward(self, z_it, z_m):
        z_itm = torch.cat([z_it, z_m], -1)
        p = self.s1(z_itm)

        z_itm_f = torch.cat([z_it * p, z_m], -1)

        z_itm = self.s2(z_itm)
        z_itm_f = self.t1(z_itm_f) * z_itm 

        z_it = z_it * (1 - z_itm) + z_itm_f
        return z_it 


class CrossAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ln_t = nn.LayerNorm(embed_dim)

        self.cross_attn_it = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ln_it = nn.LayerNorm(embed_dim)

        self.cross_attn_tm = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ln_tm = nn.LayerNorm(embed_dim)

        self.ln = nn.LayerNorm(embed_dim)
        # self.gated_unit = GatedUnit(embed_dim)
        # self.weights = nn.Parameter(torch.ones((2, 1, 1, 768)), requires_grad=True)

        self.mlp_it = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        self.mlp_tm = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        self.l_i = nn.LayerNorm(embed_dim)
        self.l_m = nn.LayerNorm(embed_dim)

        # self.gru = nn.GRUCell(embed_dim, embed_dim)

        # self.mlp = nn.Sequential(
        #     nn.Linear(embed_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, embed_dim),
        #     nn.ReLU()
        # )

    def forward(self, z, z_i, z_m):
        z_t, _ = self.self_attn(z, z, z)
        z_t = self.ln_t(z_t + z)

        z_it, _ = self.cross_attn_it(z_t, z_i, z_i)
        z_it = self.ln_it(z_it + z_t)
        z_it = self.mlp_it(z_it) + z_it
        z_it = self.l_i(z_it)

        z_tm, _ = self.cross_attn_tm(z_t, z_m, z_m)
        z_tm = self.ln_tm(z_tm + z_t)
        z_tm = self.mlp_tm(z_tm) + z_tm
        z_tm = self.l_m(z_tm)

        BSZ, SEQ_LEN, EMBED_DIM = z_it.shape

        z_total = z_it + z_tm + z

        # z_total = 0.5 * z_it + 0.5 * z_tm
        # z_total = self.ln(z_total)

        # z = self.mlp(z_total)
        # z = z_total
        z = self.ln(z_total)
        return z


def create_fusion_layer():
    print("##### FUSION LAYER CONFIGURATION #####")

    embed_dim = 768
    num_heads = int(input("Cross attention heads: "))
    hidden_dim = int(input("Cross attention hidden dim: "))
    dropout = float(input("Cross attention dropout: "))
    num_blocks = int(input("Number of cross attention blocks: "))

    return nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_blocks)
        ])