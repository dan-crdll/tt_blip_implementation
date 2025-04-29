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

        loss_a2b = F.cross_entropy(logits, targets)
        loss_b2a = F.cross_entropy(logits.t(), targets)

        loss = (loss_a2b + loss_b2a) / 2

        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred, true):
        pred = F.sigmoid(pred)
        p_t = torch.where(true == 1, pred, 1 - pred).to(pred.device)
        l = - (1 - p_t) ** self.gamma * torch.log(p_t)
        l = l.mean()
        return l
    
class ManipulationAwareContrastiveLoss(nn.Module):
    def __init__(self, temp):
        self.loss = ContrastiveLoss(temp)
    
    def forward(self, img_cls, txt_cls, blip_enc):
        l_vt = self.loss(img_cls, txt_cls)
        l_vb = self.loss(img_cls, blip_enc)
        l_tb = self.loss(txt_cls, blip_enc)

        l = 1/3 * (l_vt + l_vb + l_tb)
        return l