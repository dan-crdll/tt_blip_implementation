import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp

    def forward(self, query, key):
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        logits = torch.matmul(query, key.T) / self.temp
        targets = torch.arange(query.size(0), device=query.device)
        return F.cross_entropy(logits, targets)


class ManipulationAwareContrastiveLoss(nn.Module):
    def __init__(self, temp, momentum_encoder, m=0.9, K=64):
        super().__init__()
        self.loss = ContrastiveLoss(temp)
        self.vit_momentum, self.bert_momentum, self.blip_momentum = momentum_encoder
        self.m = m
        self.K = K
        self.register_buffer("queue_i", None)
        self.register_buffer("queue_t", None)
        self.register_buffer("queue_m", None)
        self.queue_ptr = 0
        self.initialized = False

        # Freeze momentum encoder parameters
        for enc in (self.vit_momentum, self.bert_momentum, self.blip_momentum):
            for param in enc.parameters():
                param.requires_grad = False

    def _init_queues(self, dim, device):
        self.queue_i = torch.zeros(self.K, dim, device=device)
        self.queue_t = torch.zeros(self.K, dim, device=device)
        self.queue_m = torch.zeros(self.K, dim, device=device)
        self.initialized = True

    def _enqueue_dequeue(self, queue, new, ptr):
        B = new.size(0)
        K = self.K
        insert_pos = (ptr + torch.arange(B, device=new.device)) % K
        queue[insert_pos] = new
        return queue

    def forward(self, img_cls, txt_cls, blip_enc, parameters, batch):
        with torch.no_grad():
            blip_pixel_values, blip_input_ids, blip_attn_mask, vit_pixel_values, bert_input_ids, bert_attn_mask = batch

            z_i = self.vit_momentum(pixel_values=vit_pixel_values).last_hidden_state
            z_t = self.bert_momentum(input_ids=bert_input_ids.long(), attention_mask=bert_attn_mask).last_hidden_state
            z_m = self.blip_momentum(query_embeds=z_t, attention_mask=bert_attn_mask, encoder_hidden_states=z_i).last_hidden_state[:, 0, :]
            
            z_t = z_t[:, 0, :]
            z_i = z_i[:, 0, :]

            if not self.initialized:
                dim = z_i.size(-1)
                self._init_queues(dim, z_i.device)

            # Append to queue (and use old entries as negatives)
            prev_i = self.queue_i.clone().detach()
            prev_t = self.queue_t.clone().detach()
            prev_m = self.queue_m.clone().detach()

        z_i_all = torch.cat([z_i, prev_i], dim=0)
        z_t_all = torch.cat([z_t, prev_t], dim=0)
        z_m_all = torch.cat([z_m, prev_m], dim=0)

        l_i2m = self.loss(img_cls, z_m_all)
        l_t2m = self.loss(txt_cls, z_m_all)
        l_m2i = self.loss(blip_enc, z_i_all)
        l_m2t = self.loss(blip_enc, z_t_all)

        l_m2m = self.loss(blip_enc, z_m_all)

        loss = (l_i2m + l_t2m + l_m2i + l_m2t + l_m2m) / 5

        with torch.no_grad():
            B = z_i.size(0)
            self.queue_i = self._enqueue_dequeue(self.queue_i, z_i.detach(), self.queue_ptr)
            self.queue_t = self._enqueue_dequeue(self.queue_t, z_t.detach(), self.queue_ptr)
            self.queue_m = self._enqueue_dequeue(self.queue_m, z_m.detach(), self.queue_ptr)
            self.queue_ptr = (self.queue_ptr + B) % self.K

            parameters_vit, parameters_bert, parameters_blip = map(list, parameters)
            for i, param in enumerate(self.vit_momentum.parameters()):
                param.data = param.data * self.m + parameters_vit[i].data * (1 - self.m)
            for i, param in enumerate(self.bert_momentum.parameters()):
                param.data = param.data * self.m + parameters_bert[i].data * (1 - self.m)
            for i, param in enumerate(self.blip_momentum.parameters()):
                param.data = param.data * self.m + parameters_blip[i].data * (1 - self.m)

        return loss
    
class ITMContrastive(nn.Module):
    def __init__(self, temp, momentum_encoder, m=0.9, K=64):
        super().__init__()
        self.loss = ContrastiveLoss(temp)
        self.vit_momentum, self.bert_momentum = momentum_encoder
        self.m = m
        self.K = K
        self.register_buffer("queue_i", None)
        self.register_buffer("queue_t", None)
        self.queue_ptr = 0
        self.initialized = False

        # Freeze momentum encoder parameters
        for enc in (self.vit_momentum, self.bert_momentum):
            for param in enc.parameters():
                param.requires_grad = False

    def _init_queues(self, dim, device):
        self.queue_i = torch.zeros(self.K, dim, device=device)
        self.queue_t = torch.zeros(self.K, dim, device=device)
        self.initialized = True

    def _enqueue_dequeue(self, queue, new, ptr):
        B = new.size(0)
        K = self.K
        insert_pos = (ptr + torch.arange(B, device=new.device)) % K
        queue[insert_pos] = new
        return queue

    def forward(self, img_cls, txt_cls, parameters, batch):
        with torch.no_grad():
            blip_pixel_values, blip_input_ids, blip_attn_mask, vit_pixel_values, bert_input_ids, bert_attn_mask = batch

            z_i = self.vit_momentum(pixel_values=vit_pixel_values).last_hidden_state
            z_t = self.bert_momentum(input_ids=bert_input_ids.long(), attention_mask=bert_attn_mask).last_hidden_state
            
            z_t = z_t[:, 0, :]
            z_i = z_i[:, 0, :]

            if not self.initialized:
                dim = z_i.size(-1)
                self._init_queues(dim, z_i.device)

            # Append to queue (and use old entries as negatives)
            prev_i = self.queue_i.clone().detach()
            prev_t = self.queue_t.clone().detach()

        z_i_all = torch.cat([z_i, prev_i], dim=0)
        z_t_all = torch.cat([z_t, prev_t], dim=0)

        l_v2t = self.loss(img_cls, z_t_all)
        l_t2v = self.loss(txt_cls, z_i_all)
        loss = (l_t2v + l_v2t) / 2

        with torch.no_grad():
            B = z_i.size(0)
            self.queue_i = self._enqueue_dequeue(self.queue_i, z_i.detach(), self.queue_ptr)
            self.queue_t = self._enqueue_dequeue(self.queue_t, z_t.detach(), self.queue_ptr)
            self.queue_ptr = (self.queue_ptr + B) % self.K

            parameters_vit, parameters_bert = map(list, parameters)
            for i, param in enumerate(self.vit_momentum.parameters()):
                param.data = param.data * self.m + parameters_vit[i].data * (1 - self.m)
            for i, param in enumerate(self.bert_momentum.parameters()):
                param.data = param.data * self.m + parameters_bert[i].data * (1 - self.m)

        return loss
    