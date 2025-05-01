import torch 
from torch import nn
import lightning as L
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAUROC, MultilabelF1Score, MultilabelAveragePrecision
from model.utils.loss_fn import ManipulationAwareContrastiveLoss
from model.version_2.layers.feature_extraction import FeatureExtractionLayer
from model.version_2.layers.fusion_layer import FusionLayer
from model.version_2.layers.classification_layer import ClassificationLayer
from model.version_2.layers.contrastive_loss import ManipulationAwareContrastiveLoss
from transformers import ViTModel, BertModel, Blip2QFormerConfig, Blip2QFormerModel
import copy

"""
Architecture to perform multi-modal deepfake detection.

Feature Extraction Layer:

Fusion Layer:

Classification Layer:
"""

class Model(L.LightningModule):
    def __init__(self, num_decoders, num_heads, hidden_dim, trainable=-3):
        super().__init__()

        # Feature Extraction Layer
        vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        bert_model = BertModel.from_pretrained('google-bert/bert-base-uncased')
        qformer_config = Blip2QFormerConfig.from_pretrained('Salesforce/blip2-opt-2.7b')
        qformer_config.encoder_hidden_size = 768
        qformer_model = Blip2QFormerModel(qformer_config)

        self.feature_extraction_layer = FeatureExtractionLayer(vit_model, bert_model, qformer_model, trainable)
        
        # Fusion Layer
        self.fusion_layer = FusionLayer(768, num_heads, hidden_dim, num_decoders)

        # Classification Layer
        self.classification_layer = ClassificationLayer(768)

        # Contrastive Loss
        self.contrastive_loss = ManipulationAwareContrastiveLoss(0.9, (
            copy.deepcopy(vit_model), copy.deepcopy(bert_model), copy.deepcopy(qformer_model)
            ))

        # Binary Metrics
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc_fn_bin = Accuracy('binary')
        self.f1_fn = F1Score('binary')
        self.prec_fn = Precision('binary')
        self.recall_fn = Recall('binary')
        self.auc_fn = BinaryAUROC()

        # Multilabel Metrics
        self.acc_fn_multi = Accuracy('multilabel', num_labels=4)
        self.cf1 = MultilabelF1Score(4, average='macro')
        self.of1 = MultilabelF1Score(4, average='micro')
        self.mAP = MultilabelAveragePrecision(4)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)

        def lr_lambda(current_step):
            warmup_steps = int(0.05 * self.trainer.max_steps)
            if current_step < warmup_steps:
                return current_step / warmup_steps
            return 0.5 * (1 + torch.cos(torch.tensor(current_step - warmup_steps)/(self.trainer.max_steps - warmup_steps) * 3.1416))

        scheduler = {"scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval":"step"}
        return [optimizer], [scheduler]
    
    def forward(self, x):
        z_i, z_t, z_it = self.feature_extraction_layer(*x)
        z = self.fusion_layer(z_i, z_t, z_it)
        y_bin, y_multi = self.classification_layer(z)

        # c_loss = self.contrastive_loss(z_i[:, 0], z_t[:, 0], z_it[:, 0], (
        #     self.feature_extraction_layer.vit.parameters(), 
        #     self.feature_extraction_layer.bert.parameters(),
        #     self.feature_extraction_layer.qformer.parameters()
        # ), x)
        return y_bin, y_multi, 0.0
    
    def step(self, split, batch):
        x, (y_bin, y_multi) = batch 

        pred_bin, pred_multi, c_loss = self.forward(x)

        bin_loss = self.loss_fn(pred_bin, y_bin.float())
        mask = (nn.functional.sigmoid(pred_bin) < 0.5)

        if mask.sum() > 0:
            multi_loss = self.loss_fn(pred_multi[mask], y_multi[mask].float())
            cls_loss = bin_loss + multi_loss
        else:
            cls_loss = bin_loss

        loss = c_loss + cls_loss

        # -- BINARY CLASSIFICATION --
        pred_bin = nn.functional.sigmoid(pred_bin)
        acc_bin = self.acc_fn_bin(pred_bin, y_bin.float())
        f1 = self.f1_fn(pred_bin, y_bin.float())
        auc = self.auc_fn(pred_bin, y_bin.float())

        self.log_dict(
            {
                f'{split}/loss_bin': bin_loss,
                f'{split}/acc_bin': acc_bin,
                f'{split}/contrastive_loss': c_loss
            }, prog_bar=True, on_epoch=True, on_step=True if split == 'Train' else False
        )
        
        self.log_dict(
            {
                f'{split}/f1_bin': f1,
                f'{split}/auc_bin': auc
            }, on_step=False, on_epoch=True, prog_bar=True
        )

        # -- MULTILABEL CLASSIFICATION --
        pred_multi = nn.functional.sigmoid(pred_multi)

        if mask.sum() > 0:
            cf1 = self.cf1(pred_multi[mask], y_multi[mask].float())
            of1 = self.of1(pred_multi[mask], y_multi[mask].float())
            mAP = self.mAP(pred_multi[mask], y_multi[mask].long())
            acc_multi = self.acc_fn_multi(pred_multi[mask], y_multi[mask].float())

            self.log_dict(
                {
                    f'{split}/cf1_multi': cf1,
                    f'{split}/of1_multi': of1,
                    f'{split}/mAP_multi': mAP,
                    f'{split}/acc_multi': acc_multi
                }, prog_bar=True, on_epoch=True, on_step=False
            )
        else:
            self.log_dict(
                {
                    f'{split}/cf1_multi': 0.0,
                    f'{split}/of1_multi': 0.0,
                    f'{split}/mAP_multi': 0.0,
                    f'{split}/acc_multi': 0.0
                }, prog_bar=True, on_epoch=True, on_step=False
            )

        # -- GENERAL LOSS --
        self.log(f"{split}/loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss 
    

    def training_step(self, batch):
        loss = self.step("Train", batch)
        return loss 
    
    def validation_step(self, batch):
        loss = self.step("Val", batch)
        return loss