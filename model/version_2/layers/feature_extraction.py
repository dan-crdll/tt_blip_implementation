import torch 
from torch import nn 
from transformers import ViTModel, BertModel, Blip2QFormerModel
import copy

"""
In this layer the feature extaction pass is performed

Inputs:
    - Image
    - Text

Layer Description:
    Models Involved:
        - ViT (for image encoding)
        - BERT (for text encoding)
        - BLIP2QFormer (for multimodal encoding)

Outputs: 
    - Feature extracted for image (z_i)
    - Feature extracted for text (z_t)
    - Feature extracted for multi-modal elaboration (z_it)
"""

class FeatureExtractionLayer(nn.Module):
    def __init__(self, vit:ViTModel, bert:BertModel, qformer:Blip2QFormerModel, trainable):
        super().__init__()

        self.vit = vit 
        self.bert = bert 
        self.qformer = qformer

        self.momentum_vit = copy.deepcopy(vit) 
        self.momentum_bert = copy.deepcopy(bert)

        self.num_layer_params = 0

        self._initialize_finetuning(trainable)

    def _initialize_finetuning(self, trainable):
        """
        Function to freeze and unfreeze selected model parameters for finetuning
        """
        # ViT unfreezing last "trainable" layers
        trainable_layers = self.vit.encoder.layer[trainable:]
        for param in self.vit.parameters():
            param.requires_grad_(False)
            self.num_layer_params += param.data.flatten().shape[0]
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad_(True)

        # BERT unfreezing last "trainable" layers
        trainable_layers = self.bert.encoder.layer[trainable:]
        for param in self.vit.parameters():
            param.requires_grad_(False)
            self.num_layer_params += param.data.flatten().shape[0]
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad_(True)

        # Q-Former is fully trainable
        for param in self.qformer.parameters():
            param.requires_grad_(True)
            self.num_layer_params += param.data.flatten().shape[0]

        # Momentum encoders are set as not trainable
        for param in self.momentum_vit.parameters():
            param.requires_grad_(False)
            self.num_layer_params += param.data.flatten().shape[0]
        for param in self.momentum_bert.parameters():
            param.requires_grad_(False)
            self.num_layer_params += param.data.flatten().shape[0]

    def forward(self,
                blip_pixel_values, 
                blip_input_ids, 
                blip_attn_mask, 
                vit_pixel_values, 
                bert_input_ids, 
                bert_attn_mask
            ):
        # ViT Feature Extraction (BSZ x N x 768)
        z_i = self.vit(pixel_values=vit_pixel_values).last_hidden_state
        # BERT Feature Extraction (BSZ x M x 768)
        z_t = self.bert(input_ids=bert_input_ids.long(), attention_mask=bert_attn_mask).last_hidden_state

        with torch.no_grad():
            z_t_m = self.momentum_bert(input_ids=bert_input_ids.long(), attention_mask=bert_attn_mask).last_hidden_state
            z_i_m = self.momentum_vit(pixel_values=vit_pixel_values).last_hidden_state
        # Q-FORMER Feature Aggregation (BSZ x M x 768)
        z_it = self.qformer(query_embeds=z_t_m, attention_mask=bert_attn_mask, encoder_hidden_states=z_i_m).last_hidden_state

        # Aggiornamento parametri Momentum Encoder
        bert_params = list(self.bert.parameters())
        vit_params = list(self.vit.parameters())
        for i, param in enumerate(self.momentum_bert.parameters()):
            param.data = param.data * 0.999 + bert_params[i].data * 0.001
        for i, param in enumerate(self.momentum_vit.parameters()):
            param.data = param.data * 0.999 + vit_params[i].data * 0.001

        return z_i, z_t, z_it

