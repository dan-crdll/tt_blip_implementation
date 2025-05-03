import torch
from torch import nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """
    Standard InfoNCE loss for contrastive learning.
    Each sample is contrasted against all others in the batch.
    """
    def __init__(self, temp=1.0, embed_dim=384, projected_dim=256):
        super().__init__()
        self.temp = temp
        self.projector_a = nn.Linear(embed_dim, projected_dim)
        self.projector_b = nn.Linear(embed_dim, projected_dim)

    def forward(self, emb_a, emb_b):
        """
        Args:
            emb_a: [B, D]
            emb_b: [B, D] or [B+N, D] for negatives
        Returns:
            Scalar contrastive loss
        """
        # Project and normalize embeddings
        emb_a = F.normalize(self.projector_a(emb_a), dim=-1)
        emb_b = F.normalize(self.projector_b(emb_b), dim=-1)

        logits = emb_a @ emb_b.T / self.temp
        targets = torch.arange(emb_a.size(0), device=emb_a.device)
        return F.cross_entropy(logits, targets)

class MocoLoss(nn.Module):
    """
    Momentum Contrast (MoCo) loss with a fixed-size queue for negatives.
    """

    def __init__(self, momentum_encoder: nn.Module, momentum: float, queue_size: int, temp=1.0, embed_dim=384):
        super().__init__()
        self.momentum_encoder = momentum_encoder
        self.momentum = momentum
        self.queue_size = queue_size

        # Freeze momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        self.momentum_encoder.eval()

        # Buffers to hold the queue of negatives
        self.register_buffer("queue_i", torch.zeros(queue_size, embed_dim))
        self.register_buffer("queue_t", torch.zeros(queue_size, embed_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # circular buffer pointer

        self.infonce_loss = InfoNCE(temp=temp, embed_dim=embed_dim)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_i: torch.Tensor, keys_t: torch.Tensor):
        """
        Enqueue new keys and dequeue old ones in a circular fashion.
        """
        batch_size = keys_i.size(0)
        ptr = int(self.queue_ptr)

        self.queue_i[ptr:ptr + batch_size] = keys_i
        self.queue_t[ptr:ptr + batch_size] = keys_t

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def update_momentum_encoder(self, model_parameters):
        """
        Update momentum encoder parameters using exponential moving average (EMA).
        """
        for m_param, param in zip(self.momentum_encoder.parameters(), model_parameters):
            m_param.data = m_param.data * self.momentum + param.data * (1.0 - self.momentum)

    def forward(self, pred, batch, model_parameters, single_approach=False):
        """
        Computes the MoCo loss.
        Args:
            pred: tuple of (p_i, p_t) from online encoder
            batch: tuple of (text, image)
            model_parameters: parameters of the online encoder
            single_approach: whether to use only (p_i vs t) and (p_i vs i)
        Returns:
            Scalar MoCo loss
        """
        text, image = batch
        BSZ = len(image)
        p_i, p_t = pred  # online encoder output

        device = p_i.device

        with torch.no_grad():
            (z_i, z_t), *_ = self.momentum_encoder(text, image)
            keys_i = z_i[:, 0].detach()
            keys_t = z_t[:, 0].detach()

            queue_i = self.queue_i.detach()
            queue_t = self.queue_t.detach()

        # Contrastive targets: online predictions vs all momentum (queue + batch)
        all_i = torch.cat([keys_i, queue_i], dim=0)
        all_t = torch.cat([keys_t, queue_t], dim=0)

        loss = self.infonce_loss(p_i, all_t) + self.infonce_loss(p_i, all_i)

        if not single_approach:
            loss += self.infonce_loss(p_t, all_i)
            loss += self.infonce_loss(p_t, all_t)
            loss /= 4
        else:
            loss /= 2

        if not single_approach:
            self._dequeue_and_enqueue(keys_i, keys_t)
            self.update_momentum_encoder(model_parameters)

        return loss
