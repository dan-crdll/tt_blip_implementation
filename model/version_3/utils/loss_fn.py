import torch
from torch import nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """
    Standard InfoNCE loss for contrastive learning.
    Each sample is contrasted against all others in the batch.
    """
    def __init__(self, temp=1.0, embed_dim=768, projected_dim=256):
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


class ITMLoss(nn.Module):
    def __init__(self, temp, momentum, image_encoder, text_encoder, queue_size=4096, embed_dim=768):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        for params in self.image_encoder.parameters():
            params.requires_grad = False
        for params in self.text_encoder.parameters():
            params.requires_grad = False
        self.image_encoder.eval()
        self.text_encoder.eval()

        self.momentum = momentum
        self.temp = temp
        self.queue_size = queue_size

        # Buffers to hold the queue of negatives
        self.register_buffer("queue_i", torch.zeros(queue_size, embed_dim))
        self.register_buffer("queue_t", torch.zeros(queue_size, embed_dim))
        self.register_buffer("queue_ptr_i", torch.zeros(1, dtype=torch.long))  # circular buffer pointer
        self.register_buffer("queue_ptr_t", torch.zeros(1, dtype=torch.long))  # circular buffer pointer

        self.infonce_loss = InfoNCE(temp=temp, embed_dim=embed_dim)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_i: torch.Tensor, keys_t: torch.Tensor):
        """
        Enqueue new keys and dequeue old ones in a circular fashion.
        """
        batch_size_i = keys_i.size(0)
        ptr_i = int(self.queue_ptr_i)
        batch_size_t = keys_t.size(0)
        ptr_t = int(self.queue_ptr_t)

        self.queue_i[ptr_i:ptr_i + batch_size_i] = keys_i
        self.queue_t[ptr_t:ptr_t + batch_size_t] = keys_t

        ptr_i = (ptr_i + batch_size_i) % self.queue_size
        ptr_t = (ptr_t + batch_size_t) % self.queue_size
        self.queue_ptr_i[0] = ptr_i
        self.queue_ptr_t[0] = ptr_t

    def forward(self, img_cls, txt_cls, img, txt, img_encoder_params, text_encoder_params, orig, labels):
        orig_img, orig_txt = orig
        with torch.no_grad():
            z_i_m = self.image_encoder(img)[:, 0].detach()
            z_t_m = self.text_encoder(txt)[:, 0].detach()

            z_i_m_orig = self.image_encoder(orig_img)[:, 0].detach()
            z_t_m_orig = self.text_encoder(orig_txt)[:, 0].detach()

            queue_i = self.queue_i.detach()
            queue_t = self.queue_t.detach()

        fake_images = (labels[:, 0] + labels[:, 1] > 0)
        fake_texts = (labels[:, 2] + labels[:, 3] > 0) 

        fake_images = torch.argwhere(fake_images)
        fake_texts = torch.argwhere(fake_texts)

        all_i = torch.cat([z_i_m_orig, z_i_m[fake_images].squeeze(1), queue_i], dim=0)
        all_t = torch.cat([z_t_m_orig, z_t_m[fake_texts].squeeze(1), queue_t], dim=0)

        l_i2t = self.infonce_loss(img_cls, all_t)
        l_t2i = self.infonce_loss(txt_cls, all_i)
        l_i2i = self.infonce_loss(img_cls, all_i)
        l_t2t = self.infonce_loss(txt_cls, all_t)
        l_itm = (l_i2t + l_t2i + l_i2i + l_t2t) / 4
        
        # Update the queue
        self._dequeue_and_enqueue(z_i_m[fake_images], z_t_m[fake_texts])

        # Momentum encoder update
        img_params = list(img_encoder_params)
        txt_params = list(text_encoder_params)
        for idx, param in enumerate(self.image_encoder.parameters()):
            param.data = param.data * self.momentum + img_params[idx].data * (1.0 - self.momentum)
        for idx, param in enumerate(self.text_encoder.parameters()):
            param.data = param.data * self.momentum + txt_params[idx].data * (1.0 - self.momentum)

        return l_itm

class DistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x1, x2):
        x1 = self.avg_pool(x1.permute(0, 2, 1)).squeeze(-1)
        x2 = self.avg_pool(x2.permute(0, 2, 1)).squeeze(-1)
        loss = 1 - F.cosine_similarity(x1, x2).mean()
        return loss