import torch 
from torch import nn
from transformers import AutoModelForImageClassification, AutoModelForTextEncoding, AutoModelForImageTextToText
import lightning as L
from torchmetrics import Accuracy


class FeatureExtractionLayer(nn.Module):
    def __init__(self, empty_img, empty_txt, empty_attn_mask, embed_dim):
        super().__init__()

        self.vit_s = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-small-patch16-224").vit
        self.vit_projector = nn.Linear(384, 768)
        self.bert = AutoModelForTextEncoding.from_pretrained("google-bert/bert-base-uncased")
        
        self.blip_img = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip_txt = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-itm-base-coco")

        self.empty_img = nn.Parameter(empty_img, requires_grad=False)
        self.empty_txt = nn.Parameter(empty_txt, requires_grad=False)
        self.empty_attn_mask = nn.Parameter(empty_attn_mask, requires_grad=False)

        self.projector_img = nn.LazyLinear(embed_dim)
        self.projector_txt = nn.LazyLinear(embed_dim)
        self.projector_multi = nn.LazyLinear(embed_dim)

        # Freeze all layers
        for param in self.vit_s.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.blip.parameters():
            param.requires_grad = False
        for param in self.blip_img.parameters():
            param.requires_grad = False
        for param in self.blip_txt.parameters():
            param.requires_grad = False

        # Unfreeze the last three layers
        for param in list(self.vit_s.parameters())[-3:]:
            param.requires_grad = True
        for param in list(self.bert.parameters())[-3:]:
            param.requires_grad = True
        for param in list(self.blip.parameters())[-3:]:
            param.requires_grad = True
        for param in list(self.blip_img.parameters())[-3:]:
            param.requires_grad = True
        for param in list(self.blip_txt.parameters())[-3:]:
            param.requires_grad = True

    def forward(self, blip_pixel_values, blip_input_ids, blip_attn_mask, 
                vit_pixel_values, bert_input_ids, bert_attn_mask):
        BSZ = blip_pixel_values.size(0)
        
        # Repeat empty tensors to match the batch size
        empty_img = self.empty_img.repeat(BSZ, 1, 1, 1)
        empty_txt = self.empty_txt.repeat(BSZ, 1)
        empty_attn_mask = self.empty_attn_mask.repeat(BSZ, 1)

        # multi-modality feature extraction
        vit_encodings = self.vit_s(pixel_values=vit_pixel_values).last_hidden_state
        vit_encodings = self.vit_projector(vit_encodings)

        bert_encodings = self.bert(input_ids=bert_input_ids, attention_mask=bert_attn_mask).last_hidden_state

        blip_i_encodings = self.blip_img(
            pixel_values=blip_pixel_values,
            input_ids=empty_txt,
            attention_mask=empty_attn_mask
        ).last_hidden_state
        blip_t_encodings = self.blip_txt(
            pixel_values=empty_img,
            input_ids=blip_input_ids,
            attention_mask=blip_attn_mask
        ).last_hidden_state
        blip_encodings = self.blip(
            pixel_values=blip_pixel_values,
            input_ids=blip_input_ids,
            attention_mask=blip_attn_mask
        ).last_hidden_state

        # feature concatenation
        image_feature = self.projector_img(torch.cat([vit_encodings, blip_i_encodings], 1))
        text_feature = self.projector_txt(torch.cat([bert_encodings, blip_t_encodings], 1))
        multimodal_feature = self.projector_multi(blip_encodings)
        return image_feature, text_feature, multimodal_feature


class FusionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) for _ in range(3)
        ])
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU()
            ) for _ in range(3)
        ])

    def forward(self, z):
        z = list(z)
        for i in range(3):
            z[i], _ = self.mha_layers[i](z[1], z[i], z[i])
            z[i] = self.mlp_layers[i](z[i])
        z = torch.cat(z, 1)
        return z # BSZ x N x EMBED_DIM
    
class ClassificationLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, z):
        BSZ, N, _ = z.shape
        z = nn.functional.avg_pool1d(z.permute(0, 2, 1), N, N).squeeze(-1)
        y = self.classifier(z).squeeze(-1)
        return y 


class TT_BLIP_Model(L.LightningModule):
    def __init__(self, empty_img, empty_txt, empty_attn_mask, embed_dim, num_heads):
        super().__init__()
        self.feature_extraction_layer = FeatureExtractionLayer(empty_img, empty_txt, empty_attn_mask, embed_dim)
        self.fusion_layer = FusionLayer(embed_dim, num_heads)
        self.classification_layer = ClassificationLayer(embed_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc_fn = Accuracy('binary')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    
    def forward(self, x):
        z = self.feature_extraction_layer(*x)
        z = self.fusion_layer(z)
        y = self.classification_layer(z)
        return y
    
    def training_step(self, batch):
        x, y = batch 
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        acc = self.acc_fn(pred, y)

        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss 
    
    def validation_step(self, batch):
        x, y = batch 
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        acc = self.acc_fn(pred, y)

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss 