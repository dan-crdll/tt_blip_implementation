import torch
from torch import nn
from torch.nn import functional as F

class InfoNCE(nn.Module):
    def __init__(self, temp=1.0, embed_dim=768, projected_dim=256):
        super().__init__()
        self.temp = temp
        self.projector_a = nn.Linear(embed_dim, projected_dim)
        self.projector_b = nn.Linear(embed_dim, projected_dim)

    def forward(self, emb_a, emb_b):
        # Project the embeddings
        emb_a = self.projector_a(emb_a)
        emb_b = self.projector_b(emb_b)
        # Normalize the embeddings
        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)
        # Compute the InfoNCE loss
        logits = emb_a @ emb_b.T / self.temp
        targets = torch.arange(emb_a.shape[0], device=emb_a.device)
        loss = F.cross_entropy(logits, targets)
        return loss

class MocoLoss(nn.Module):
    def __init__(self, momentum_encoder: nn.Module, momentum: float, queue_size, temp=1.0):
        super().__init__()

        self.momentum_encoder = momentum_encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        self.momentum_encoder.eval()

        self.register_buffer("queue_i", torch.zeros(queue_size, 768))
        self.register_buffer("queue_t", torch.zeros(queue_size, 768))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # points to the next position in the queue

        self.momentum = momentum
        self.infonce_loss = InfoNCE(temp)
        self.queue_size = queue_size

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_i, keys_t):
        batch_size = keys_i.shape[0]
        ptr = int(self.queue_ptr)
        len_to_end = self.queue_size - ptr
        # Insert at the pointer; if overflow, wrap around
        if batch_size <= len_to_end:
            self.queue_i[ptr:ptr+batch_size] = keys_i
            self.queue_t[ptr:ptr+batch_size] = keys_t
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # Fill to end, then wrap
            self.queue_i[ptr:] = keys_i[:len_to_end]
            self.queue_t[ptr:] = keys_t[:len_to_end]
            end = batch_size - len_to_end
            self.queue_i[:end] = keys_i[len_to_end:]
            self.queue_t[:end] = keys_t[len_to_end:]
            ptr = end
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def update_momentum(self, model, momentum):
        for m_param, param in zip(self.momentum_encoder.parameters(), model.parameters()):
            m_param.data = m_param.data * momentum + param.data * (1. - momentum)

    def forward(self, pred, batch, model_parameters):
        text, image = batch
        BSZ = len(text)
        p_i, p_t = pred

        device = p_i.device
        # Compute keys using momentum encoder
        with torch.no_grad():
            (z_i, z_t), *_ = self.momentum_encoder(text, image)
            keys_i = z_i[:, 0].detach()  # [BSZ, 768]
            keys_t = z_t[:, 0].detach()
            # Get all negatives from queue (detach to ensure not to track grads)
            queue_i = self.queue_i.detach()
            queue_t = self.queue_t.detach()

        # InfoNCE using current batch + queue for negatives
        # Positives from batch, negatives from queue
        all_t_embs = torch.cat([keys_t, queue_t], dim=0)
        all_i_embs = torch.cat([keys_i, queue_i], dim=0)

        # Predictors are p_i and p_t (from online encoder)
        # Contrastive pairs: (p_i, all_t_embs), (p_t, all_i_embs)
        l_itm = self.infonce_loss(p_i, all_t_embs)
        l_itm += self.infonce_loss(p_t, all_i_embs)
        l_itm += self.infonce_loss(p_i, all_i_embs)
        l_itm += self.infonce_loss(p_t, all_t_embs)
        l_itm /= 4

        # Update queue
        self._dequeue_and_enqueue(keys_i, keys_t)

        # Update momentum encoder (optional, do it outside if possible)
        with torch.no_grad():
            for m_param, param in zip(self.momentum_encoder.parameters(), model_parameters):
                m_param.data = m_param.data * self.momentum + param.data * (1. - self.momentum)
            self.momentum_encoder.eval()

        return l_itm