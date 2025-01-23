import torch
from torch import nn 
import lightning as L 
from transformers import BertModel, ViTModel, BlipForConditionalGeneration


class FeatureExtractionLayer(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        self.bert.eval()
        self.vit.eval()
        self.blip.eval()

        self.bert.requires_grad_(False)
        self.vit.requires_grad_(False)
        self.blip.requires_grad_(False)

        ### SET LAST 3 LAYERS OF VIT AND BERT TO TRAIN ###
        self.vit.encoder.layer[9:].requires_grad_(True)
        self.vit.encoder.layer[9:].train()
        self.vit.pooler.requires_grad_(True)
        self.vit.pooler.train()

        self.bert.encoder.layer[9:].requires_grad_(True)
        self.bert.encoder.layer[9:].train()
        self.bert.pooler.requires_grad_(True)
        self.bert.pooler.train()

        ### DUMMY IMG AND TEXT FOR BLIP IMAGE/TEXT ONLY ###
        self.dummy_img = nn.Parameter(torch.zeros((1, 3, 384, 384)), requires_grad=False)
        self.dummy_txt = nn.Parameter(torch.zeros((1, 1), dtype=torch.int), requires_grad=False)
        self.dummy_attn = nn.Parameter(torch.zeros((1, 1), dtype=torch.int), requires_grad=False)

    def forward(self, vit_img, blip_img, blip_txt, blip_attn, bert_txt, bert_attn):
        BSZ, *_ = vit_img.shape

        # IMAGES
        vit_encodings = self.vit(vit_img).last_hidden_state # 197 x 768
        blip_img_encodings = self.blip(blip_img, self.dummy_txt.repeat(BSZ, 1), attention_mask=self.dummy_attn.repeat(BSZ, 1)).last_hidden_state    # 577 x 768
        zi = torch.cat([vit_encodings, blip_img_encodings], dim=1)  # 197 + 577 x 768


        # TEXT
        bert_encodings = self.bert(bert_txt, bert_attn).last_hidden_state   # ?? x 768
        blip_txt_encodings = self.blip(self.dummy_img.repeat(BSZ, 1, 1, 1), blip_txt, attention_mask=blip_attn).last_hidden_state   # 577 x 768
        zt = torch.cat([bert_encodings, blip_txt_encodings], dim=1) # 577 + ?? x 768


        # IMAGE / TEXT
        blip_img_txt_encodings = self.blip(blip_img, blip_txt, attention_mask=blip_attn).last_hidden_state  # 577 x 768
        zit = blip_img_txt_encodings

        return zi, zt, zit 