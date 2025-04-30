import torch 
from torch import nn
from transformers import ViTForImageClassification, BertModel, BlipForImageTextRetrieval
import torch.nn.functional as F


class FeatureExtractionLayer(nn.Module):
    def __init__(self, empty_img, empty_txt, empty_attn_mask, trainable):
        super().__init__()

        # Use of pretrained models from transformers
        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").vit
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")

        # Loading of BLIP models
        self.blip_img = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip_txt = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

        # Initialization of empty inputs for incomplete BLIP
        self.empty_img = nn.Parameter(empty_img, requires_grad=False)
        self.empty_txt = nn.Parameter(empty_txt, requires_grad=False)
        self.empty_attn_mask = nn.Parameter(empty_attn_mask, requires_grad=False)

        self.initialize_training_mode(trainable)

    def initialize_training_mode(self, trainable):
        """
        Initialize the training mode by setting the requires_grad attribute of the parameters
        """
        # for param in self.blip.parameters():
        #     param.requires_grad_(False)

        for param in self.blip_img.parameters():
            param.requires_grad_(False)
        
        for param in self.blip_txt.parameters():
            param.requires_grad_(False)
        
        # Initialization of training mode for ViT and BERT
        trainable_layers = self.vit.encoder.layer[trainable:]
        for param in self.vit.parameters():
            param.requires_grad_(False)
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad_(True)

        trainable_layers = self.bert.encoder.layer[trainable:]
        for param in self.vit.parameters():
            param.requires_grad_(False)
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad_(True)

        layer = self.blip.text_encoder.encoder.layer[trainable:]
        for param in self.blip.parameters():
            param.requires_grad = False
        for layer in trainable_layers:
            for param in layer.parameters():
                if not any(torch.equal(param, p) for p in layer.attention.parameters()):
                    param.requires_grad = True

        
                
    def forward(
                self, 
                blip_pixel_values, 
                blip_input_ids, 
                blip_attn_mask, 
                vit_pixel_values, 
                bert_input_ids, 
                bert_attn_mask
            ):
        BSZ = blip_pixel_values.size(0)
        
        # Repeat empty tensors to match the batch size
        empty_img = self.empty_img.repeat(BSZ, 1, 1, 1)
        empty_txt = self.empty_txt.repeat(BSZ, 1).long()
        empty_attn_mask = self.empty_attn_mask.repeat(BSZ, 1)

        # multi-modality feature extraction
        """
        ViT Last hidden state has dimension BSZ x 197 x 768
        """
        vit_encodings = self.vit(pixel_values=vit_pixel_values).last_hidden_state[:, 0].unsqueeze(1)

        """
        BERT Last hidden state has dimension BSZ x N + 1 x 768
        """
        bert_encodings = self.bert(input_ids=bert_input_ids.long(), attention_mask=bert_attn_mask).last_hidden_state[:, 0].unsqueeze(1)

        """
        BLIP encodings have dimension BSZ x 577 x 768
        """
        blip_i_encodings = self.blip_img(
            pixel_values=blip_pixel_values,
            input_ids=empty_txt.long(),
            attention_mask=empty_attn_mask
        ).last_hidden_state[:, 1:]
        blip_t_encodings = self.blip_txt(
            pixel_values=empty_img,
            input_ids=blip_input_ids.long(),
            attention_mask=blip_attn_mask
        ).last_hidden_state[:, 1:]
        blip_encodings = self.blip(
            pixel_values=blip_pixel_values,
            input_ids=blip_input_ids.long(),
            attention_mask=blip_attn_mask
        ).last_hidden_state

        # Feature concatenation 
        image_feature = torch.cat([vit_encodings, blip_i_encodings], 1)
        txt_feature = torch.cat([bert_encodings, blip_t_encodings], 1)
        multimodal_feature = blip_encodings
        # They all have dim BSZ x 577 x 768
        return image_feature, txt_feature, multimodal_feature

