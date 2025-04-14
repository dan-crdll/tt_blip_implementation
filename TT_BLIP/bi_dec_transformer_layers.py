import torch 
from torch import nn
from transformers import ViTForImageClassification, BertModel, BlipForImageTextRetrieval
import lightning as L
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAUROC, MultilabelF1Score, MultilabelAveragePrecision
import torch.nn.functional as F


class FeatureExtractionLayer(nn.Module):
    def __init__(self, empty_img, empty_txt, empty_attn_mask, trainable):
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
        for param in self.blip.parameters():
            param.requires_grad = False

        for param in self.blip_img.parameters():
            param.requires_grad = False
        
        for param in self.blip_txt.parameters():
            param.requires_grad = False
        
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

        # contrastive loss computation (cosine similarity)
        l = (1.0 - F.cosine_similarity(image_feature, txt_feature)).mean()

        # They all have dim BSZ x 578 x 768 with cls token
        return image_feature, txt_feature, multimodal_feature, l


"""
Definition of cross attention encoder layers and wrapper
"""
class CrossAttnEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, batch_first=True):
        super().__init__()

        self.multi_head_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v):
        z, _ = self.multi_head_attn(q, k, v)
        x = self.norm_1(k + z)
        z = self.mlp(x)
        x = self.norm_2(x + z)
        return x
    

class CrossAttnEncoder(nn.Module):
    def __init__(self, encoder_layer, num_encoders):
        super().__init__()

        self.encoders = nn.ModuleList([
            encoder_layer for _ in range(num_encoders)
        ])
    
    def forward(self, z_q, z_kv):
        y = z_kv

        for encoder in self.encoders:
            y = encoder(z_q, y, y)
        return y


"""
Fusion Layer for feature concatenation which uses self and 
cross attention encoders and computes contrastive loss
"""
class FusionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_encoders=11, num_decoders=11):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, batch_first=True)
        decoder_layer = CrossAttnEncoderLayer(embed_dim, num_heads, hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoders)
        self.decoder_img = CrossAttnEncoder(decoder_layer, num_decoders)
        self.decoder_txt = CrossAttnEncoder(decoder_layer, num_decoders)

    def forward(self, z):
        z_i, z_t, z_m = z
        BSZ, _, N = z_i.shape
        
        z_m = self.encoder(z_m)
        
        z_i = self.decoder_img(z_m, z_i)
        z_t = self.decoder_txt(z_m, z_t)

        # z_i = z_i[:, 0].unsqueeze(1)
        # z_t = z_t[:, 0].unsqueeze(1)

        l = (1.0 - F.cosine_similarity(z_i, z_t)).mean()


        return (z_i, z_t), l
    
"""
Classification Layer for binary (Real/Fake) classification
"""
class ClassificationLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048):
        super().__init__()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.bin_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self.multi_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 4),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, z):
        z_i, z_t = z
        z = torch.cat([z_i, z_t], 1).to(z_i.device)

        cls = self.avg_pool(z.permute(0, 2, 1)).squeeze(-1)

        y_bin = self.bin_classifier(cls).squeeze(-1)
        y_multi = self.multi_classifier(cls)
        return y_bin, y_multi 


"""
Complete architecture
"""
class BiDec_Model(L.LightningModule):
    def __init__(self, empty_img, empty_txt, empty_attn_mask, embed_dim, num_heads, hidden_dim, trainable=-3):
        super().__init__()
        self.feature_extraction_layer = FeatureExtractionLayer(empty_img, empty_txt, empty_attn_mask, trainable)
        self.fusion_layer = FusionLayer(embed_dim, num_heads, hidden_dim)
        self.classification_layer = ClassificationLayer(embed_dim, hidden_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc_fn_bin = Accuracy('binary')
        self.acc_fn_multi = Accuracy('multilabel', num_labels=4)
        self.f1_fn = F1Score('binary')
        self.prec_fn = Precision('binary')
        self.recall_fn = Recall('binary')
        self.auc_fn = BinaryAUROC()

        self.cf1 = MultilabelF1Score(4, average='macro')
        self.of1 = MultilabelF1Score(4, average='micro')
        self.mAP = MultilabelAveragePrecision(4)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
        def lr_lambda(current_step):
            warmup_steps = int(0.05 * self.trainer.max_steps)
            if current_step < warmup_steps:
                return current_step / warmup_steps
            return 0.5 * (1 + torch.cos(torch.tensor(current_step - warmup_steps)/(self.trainer.max_steps - warmup_steps) * 3.1416))

        scheduler = {"scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval":"step"}
        return [optimizer], [scheduler]
    
    def forward(self, x):
        z_i, z_t, z_m, c_loss_1 = self.feature_extraction_layer(*x)
        z, c_loss_2 = self.fusion_layer((z_i, z_t, z_m))
        y = self.classification_layer(z)
        return y, c_loss_1 + c_loss_2
    

    def training_step(self, batch):
        x, (y_bin, y_multi) = batch 
        (pred_bin, pred_multi), c_loss = self.forward(x)
        
        multi_loss = self.loss_fn(pred_multi, y_multi.float())
        bin_loss = self.loss_fn(pred_bin, y_bin.float())
        loss = c_loss + multi_loss + bin_loss

        # -- BINARY CLASSIFICATION --
        pred_bin = nn.functional.sigmoid(pred_bin)
        acc_bin = self.acc_fn_bin(pred_bin, y_bin.float())
        f1 = self.f1_fn(pred_bin, y_bin.float())
        auc = self.auc_fn(pred_bin, y_bin.float())

        self.log_dict(
            {
                'Train/loss_bin': bin_loss,
                'Train/acc_bin': acc_bin
            }, prog_bar=True, on_epoch=True, on_step=True
        )
        
        self.log_dict(
            {
                'Train/f1_bin': f1,
                'Train/auc_bin': auc
            }, on_step=False, on_epoch=True, prog_bar=True
        )

        # -- MULTILABEL CLASSIFICATION --
        pred_multi = nn.functional.sigmoid(pred_multi)
        cf1 = self.cf1(pred_multi, y_multi.float())
        of1 = self.of1(pred_multi, y_multi.float())
        mAP = self.mAP(pred_multi, y_multi.long())
        acc_multi = self.acc_fn_multi(pred_multi, y_multi.float())
        self.log_dict(
            {
                'Train/cf1_multi':cf1,
                'Train/of1_multi':of1,
                'Train/mAP_multi': mAP,
                'Train/acc_multi': acc_multi
            }, prog_bar=True, on_epoch=True, on_step=False
        )

        self.log('Train/loss_multi', multi_loss, prog_bar=True, on_epoch=False, on_step=True)
        self.log('Train/con_loss', c_loss, prog_bar=True, on_epoch=False, on_step=True)

        # -- GENERAL LOSS --
        self.log("Train/loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss 
    
    def validation_step(self, batch):
        x, (y_bin, y_multi) = batch 
        (pred_bin, pred_multi), c_loss = self.forward(x)
        
        multi_loss = self.loss_fn(pred_multi, y_multi.float())
        bin_loss = self.loss_fn(pred_bin, y_bin.float())
        loss = c_loss + multi_loss + bin_loss

        # -- BINARY CLASSIFICATION --
        pred_bin = nn.functional.sigmoid(pred_bin)
        acc_bin = self.acc_fn_bin(pred_bin, y_bin.float())
        f1 = self.f1_fn(pred_bin, y_bin.float())
        auc = self.auc_fn(pred_bin, y_bin.float())

        self.log_dict(
            {
                'Val/loss_bin': bin_loss,
                'Val/acc_bin': acc_bin
            }, prog_bar=True, on_epoch=True, on_step=False
        )
        
        self.log_dict(
            {
                'Val/f1_bin': f1,
                'Val/auc_bin': auc
            }, on_step=False, on_epoch=True, prog_bar=True
        )

        # -- MULTILABEL CLASSIFICATION --
        pred_multi = nn.functional.sigmoid(pred_multi)
        cf1 = self.cf1(pred_multi, y_multi.float())
        of1 = self.of1(pred_multi, y_multi.float())
        mAP = self.mAP(pred_multi, y_multi.long())
        acc_multi = self.acc_fn_multi(pred_multi, y_multi.float())
        self.log_dict(
            {
                'Val/cf1_multi':cf1,
                'Val/of1_multi':of1,
                'Val/mAP_multi': mAP,
                'Val/acc_multi': acc_multi
            }, prog_bar=True, on_epoch=True, on_step=False
        )

        self.log('Val/loss_multi', multi_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('Val/con_loss', c_loss, prog_bar=True, on_epoch=True, on_step=False)

        # -- GENERAL LOSS --
        self.log("Val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss 