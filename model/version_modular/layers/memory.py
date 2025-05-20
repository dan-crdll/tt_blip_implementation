import torch
from torch import nn

class Memory(nn.Module):
    def __init__(self, embed_dim=768, slots=128, T=3):
        super().__init__()
        self.slots = nn.Parameter(torch.zeros((1, slots, embed_dim)))
        nn.init.xavier_uniform_(self.slots)

        self.read_mha = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.write_mha = nn.MultiheadAttention(embed_dim, 1, batch_first=True)

        self.gru = nn.GRUCell(embed_dim, embed_dim)
        self.T = T

        self.ln_m = nn.LayerNorm(embed_dim)

    def forward(self, z):
        BSZ = z.size(0)
        memory = self.slots.repeat(BSZ, 1, 1)  # (B, S, D)
        _, SLOTS, EMBED_DIM = memory.shape

        for i in range(self.T):
            memory = self.ln_m(memory)
            updated_memory, _ = self.write_mha(memory, z, z)  # (B, S, D)

            memory = self.gru(
                updated_memory.reshape(-1, EMBED_DIM),
                memory.reshape(-1, EMBED_DIM)
            ).reshape(BSZ, SLOTS, EMBED_DIM)

        # Read: inputs attend to updated memory
        out, _ = self.read_mha(z, memory, memory)  # (B, L, D)

        return out

