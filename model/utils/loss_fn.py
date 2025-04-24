import torch 
from torch import nn 
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()

        self.temp = temp 

    def forward(self, emb_a, emb_b):
        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)

        logits = emb_a @ emb_b.t() / self.temp

        targets = torch.arange(emb_a.size(0)).to(emb_a.device)

        # Cross-entropy loss (A->B)
        loss_a2b = F.cross_entropy(logits, targets)

        # Cross-entropy loss (B->A)
        loss_b2a = F.cross_entropy(logits.t(), targets)

        # Loss totale
        loss = (loss_a2b + loss_b2a) / 2

        return loss