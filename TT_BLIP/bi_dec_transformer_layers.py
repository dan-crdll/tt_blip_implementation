import torch 
from torch import nn
from transformers import ViTForImageClassification, BertModel, BlipForImageTextRetrieval
import lightning as L
from torchmetrics import Accuracy, F1Score, Precision, Recall


class FeatureExtractionLayer(nn.Module):
    def __init__(self, empty_img, empty_txt, empty_attn_mask, embed_dim, trainable):
        super().__init__()

        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").vit
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")      

        self.blip_img = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip_txt = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

        self.empty_img = nn.Parameter(empty_img, requires_grad=False)
        self.empty_txt = nn.Parameter(empty_txt, requires_grad=False)
        self.empty_attn_mask = nn.Parameter(empty_attn_mask, requires_grad=False)

        self.initialize_training_mode(trainable)

    def initialize_training_mode(self, trainable):
        trainable_layers = self.blip.text_encoder.encoder.layer[trainable:]
        for param in self.blip.parameters():
            param.requires_grad = False
        for layer in trainable_layers:
            for param in layer.parameters():
                if not any(torch.equal(param, p) for p in layer.attention.parameters()):
                    param.requires_grad = True

        trainable_layers = self.blip_img.text_encoder.encoder.layer[trainable:]
        for param in self.blip_img.parameters():
            param.requires_grad = False
        for layer in trainable_layers:
            for param in layer.parameters():
                if not any(torch.equal(param, p) for p in layer.attention.parameters()):
                    param.requires_grad = True
            
        trainable_layers = self.blip_txt.text_encoder.encoder.layer[trainable:]
        for param in self.blip_txt.parameters():
            param.requires_grad = False
        for layer in trainable_layers:
            for param in layer.parameters():
                if not any(torch.equal(param, p) for p in layer.attention.parameters()):
                    param.requires_grad = True

        trainable_layers = self.vit.encoder.layer[trainable:]
        for param in self.vit.parameters():
            param.requires_grad = False
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True

        trainable_layers = self.bert.encoder.layer[trainable:]
        for param in self.vit.parameters():
            param.requires_grad = False
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True
        
                

    def forward(self, blip_pixel_values, blip_input_ids, blip_attn_mask, 
                vit_pixel_values, bert_input_ids, bert_attn_mask):
        BSZ = blip_pixel_values.size(0)
        
        # Repeat empty tensors to match the batch size
        empty_img = self.empty_img.repeat(BSZ, 1, 1, 1)
        empty_txt = self.empty_txt.repeat(BSZ, 1)
        empty_attn_mask = self.empty_attn_mask.repeat(BSZ, 1)
        cls_multi = torch.zeros((BSZ, 1, 768)).to(blip_pixel_values.device)

        # multi-modality feature extraction
        """
        ViT Last hidden state has dimension BSZ x 197 x 768 , can try taking only 
        classifier token
        """
        vit_encodings = self.vit(pixel_values=vit_pixel_values).last_hidden_state[:, 0].unsqueeze(1)

        """
        BERT Last hidden state has dimension BSZ x N + 1 x 768, can try taking only classifier token
        """
        bert_encodings = self.bert(input_ids=bert_input_ids, attention_mask=bert_attn_mask).last_hidden_state[:, 0].unsqueeze(1)

        """
        BLIP encodings have dimension BSZ x 577 x 768
        """
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

        """
        Feature concatenation
        """
        image_feature = torch.cat([vit_encodings, blip_i_encodings], 1)
        txt_feature = torch.cat([bert_encodings, blip_t_encodings], 1)
        multimodal_feature = torch.cat([cls_multi, blip_encodings], 1)

        # They all have dim BSZ x 578 x 768 with cls token
        return image_feature, txt_feature, multimodal_feature


class FusionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_encoders=11, num_decoders=11):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads, hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoders)
        self.decoder_img = nn.TransformerDecoder(decoder_layer, num_decoders)
        self.decoder_txt = nn.TransformerDecoder(decoder_layer, num_decoders)

    def forward(self, z):
        z_i, z_t, z_m = z

        z_m = self.encoder(z_m)
        
        z_i = self.decoder_img(z_i, z_m)
        z_i = z_i[:, 0].unsqueeze(1)


        z_t = self.decoder_txt(z_t, z_m)
        z_t = z_t[:, 0].unsqueeze(1)

        z = torch.cat([z_i, z_t], 1)
        return z # BSZ x 2 x EMBED_DIM
    
class ClassificationLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048):
        super().__init__()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        z = self.global_pooling(z.permute(0, 2, 1)).squeeze(-1)
        y = self.classifier(z).squeeze(-1)
        return y 


class BiDec_Model(L.LightningModule):
    def __init__(self, empty_img, empty_txt, empty_attn_mask, embed_dim, num_heads, hidden_dim, trainable=-3):
        super().__init__()
        self.feature_extraction_layer = FeatureExtractionLayer(empty_img, empty_txt, empty_attn_mask, embed_dim, trainable)
        self.fusion_layer = FusionLayer(embed_dim, num_heads, hidden_dim)
        self.classification_layer = ClassificationLayer(embed_dim, hidden_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc_fn = Accuracy('binary')
        self.f1_fn = F1Score('binary')
        self.prec_fn = Precision('binary')
        self.recall_fn = Recall('binary')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)
        def lr_lambda(current_step):
            warmup_steps = int(0.05 * self.trainer.max_steps)
            if current_step < warmup_steps:
                return current_step / warmup_steps
            return 0.5 * (1 + torch.cos(torch.tensor(current_step - warmup_steps)/(self.trainer.max_steps - warmup_steps) * 3.1416))

        scheduler = {"scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval":"step"}
        return [optimizer], [scheduler]
    
    def forward(self, x):
        z = self.feature_extraction_layer(*x)
        z = self.fusion_layer(z)
        y = self.classification_layer(z)
        return y
    
    def training_step(self, batch):
        x, y = batch 
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)

        pred = nn.functional.sigmoid(pred)
        acc = self.acc_fn(pred, y)

        f1 = self.f1_fn(pred, y)
        prec = self.prec_fn(pred, y)
        rec = self.recall_fn(pred, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        self.log("train_prec", prec, on_epoch=True, on_step=False)
        self.log("train_rec", rec, on_epoch=True, on_step=False)
        self.log("train_f1", f1, on_epoch=True, on_step=False)
        return loss 
    
    def validation_step(self, batch):
        x, y = batch 
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)

        pred = nn.functional.sigmoid(pred)
        acc = self.acc_fn(pred, y)

        f1 = self.f1_fn(pred, y)
        prec = self.prec_fn(pred, y)
        rec = self.recall_fn(pred, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        self.log("val_prec", prec, on_epoch=True, on_step=False)
        self.log("val_rec", rec, on_epoch=True, on_step=False)
        self.log("val_f1", f1, on_epoch=True, on_step=False)
        return loss 
