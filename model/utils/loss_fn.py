import torch 
from torch import nn 
import torch.nn.functional as F
from model.layers.feature_extraction import FeatureExtractionLayer
from collections import deque


class ContrastiveLoss(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp 

    def forward(self, query, key):
        """
        query: Tensor of shape (B, D)
        key:   Tensor of shape (B + N, D)
               where the first B entries are the true positives (aligned with query),
               and the remaining N are negatives.
        """
        B = query.size(0)
        K = key.size(0)


        # Normalize embeddings
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        # Compute similarity scores: (B, B+N)
        logits = torch.matmul(query, key.T) / self.temp

        # Targets: for each query[i], the correct key is at position i
        targets = torch.arange(B, device=query.device)

        # Cross-entropy loss between query-key similarities and targets
        loss = F.cross_entropy(logits, targets)

        return loss
        
    
class ManipulationAwareContrastiveLoss(nn.Module):
    def __init__(self, temp, momentum_encoder:FeatureExtractionLayer, m=0.9, K=100):
        super().__init__()
        self.loss = ContrastiveLoss(temp)
        self.momentum_encoder = momentum_encoder
        self.K = K
        
        self.queue_i = deque([])
        self.queue_t = deque([])
        self.queue_m = deque([])

    
    def forward(self, img_cls, txt_cls, blip_enc, parameters, batch):
        with torch.no_grad():
            z_i, z_t, z_m = self.momentum_encoder(*batch)

            if len(self.queue_i) > 0:
                prev_i = self.queue_i.pop()
                prev_t = self.queue_t.pop()
                prev_m = self.queue_m.pop()

                z_i = torch.vstack([z_i[:, 0], prev_i])
                z_t = torch.vstack([z_t[:, 0], prev_t])
                z_m = torch.vstack([z_m[:, 0], prev_m])
            else:
                z_i = z_i[:, 0]
                z_t = z_t[:, 0]
                z_m = z_m[:, 0]
        
        l_i2m = self.loss(img_cls, z_m)
        l_t2m = self.loss(txt_cls, z_m)

        l_i2t = self.loss(img_cls, z_t)
        l_t2i = self.loss(txt_cls, z_i)

        l_i2i = self.loss(img_cls, z_i)
        l_t2t = self.loss(txt_cls, z_t)

        loss = 1/6 * (l_i2m + l_t2m + l_i2t + l_t2i + l_i2i + l_t2t)

        self.queue_i.append(z_i)
        self.queue_t.append(z_m)
        self.queue_m.append(z_t)
        self.queue_i = self.queue_i[-self.K:]
        self.queue_t = self.queue_t[-self.K:]
        self.queue_m = self.queue_m[-self.K:]

        for i, param in enumerate(self.momentum_encoder.parameters()):
            param.data = param.data * self.m + parameters[i].data * (1 - self.m)

        return loss
