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
    
class MocoLoss(nn.Module):
    def __init__(self, momentum_encoder:nn.Module, momentum:float, queue_size, temp=1.0):
        super().__init__()

        self.momentum_encoder = momentum_encoder
        for params in self.momentum_encoder.parameters():
            params.requires_grad = False
        self.momentum_encoder.eval()

        self.queue_i = nn.Buffer(torch.zeros((queue_size, 768)))
        self.queue_t = nn.Buffer(torch.zeros((queue_size, 768)))

        self.first_idx = 0
        self.last_idx = 0
        self.momentum = momentum 
        self.infonce_loss = InfoNCE(temp)
        self.queue_size = queue_size

    def forward(self, pred, batch, model_parameters):
        text, image = batch 
        BSZ = len(text)

        p_i, p_t = pred 

        with torch.no_grad():
            (z_i, z_t), *_ = self.momentum_encoder(text, image)
            z_i_all = z_i[:, 0].detach()
            z_t_all = z_t[:, 0].detach()

            if self.last_idx != 0:
                for i in range(self.first_idx, self.last_idx):
                    z_i_all = torch.cat([z_i_all, self.queue_i[i % self.queue_size].unsqueeze(0)], 0)
                    z_t_all = torch.cat([z_t_all, self.queue_t[i % self.queue_size].unsqueeze(0)], 0)

        l_itm = self.infonce_loss(p_i, z_t_all)
        l_itm += self.infonce_loss(p_t, z_i_all)
        l_itm += self.infonce_loss(p_i, z_i_all)
        l_itm += self.infonce_loss(p_t, z_t_all)
        l_itm /= 4


        for i in range(self.first_idx, self.first_idx + BSZ):
            self.queue_i[i % self.queue_size] = z_i[i, 0].detach()
            self.queue_t[i % self.queue_size] = z_t[i, 0].detach()
        self.first_idx += BSZ
        self.last_idx += BSZ

        model_parameters = list(model_parameters)
        for i, param in enumerate(self.momentum_encoder.parameters()):
            param.data = self.momentum * param.data + (1 - self.momentum) * model_parameters[i].data
        self.momentum_encoder.eval()
        return l_itm


