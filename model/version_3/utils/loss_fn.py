import torch 
from torch import nn 
from torch.nn import functional as F


class InfoNCE(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp

    def forward(self, emb_a, emb_b):
        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)

        logits = emb_a @ emb_b.T / self.temp
        targets = torch.arange(emb_a.shape[0], device=emb_a.device)
        loss = F.cross_entropy(logits, targets)
        return loss 