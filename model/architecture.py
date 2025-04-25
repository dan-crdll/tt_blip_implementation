import torch 
from torch import nn
import lightning as L
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAUROC, MultilabelF1Score, MultilabelAveragePrecision
from model.utils.loss_fn import FocalLoss
from model.layers.classification import ClassificationLayer
from model.layers.feature_extraction import FeatureExtractionLayer
from model.layers.fusion import FusionLayer


"""
Complete architecture
"""
class Model(L.LightningModule):
    def __init__(self, empty_img, empty_txt, empty_attn_mask, embed_dim, num_heads, hidden_dim, trainable=-3, gamma=2):
        super().__init__()
        # Model Layers
        self.feature_extraction_layer = FeatureExtractionLayer(empty_img, empty_txt, empty_attn_mask, trainable)
        self.fusion_layer = FusionLayer(embed_dim, num_heads, hidden_dim, 1, 1)
        self.classification_layer = ClassificationLayer(embed_dim, hidden_dim)

        # Binary Metrics
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(gamma)
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
        z_i, z_t, z_m, c_loss_1 = self.feature_extraction_layer(*x)
        z = self.fusion_layer((z_i, z_t, z_m))
        y = self.classification_layer(z)
        return y, c_loss_1
    

    def training_step(self, batch):
        x, (y_bin, y_multi) = batch 
        (pred_bin, pred_multi), c_loss = self.forward(x)
        
        multi_loss = self.loss_fn(pred_multi, y_multi.float())
        bin_loss = self.loss_fn(pred_bin, y_bin.float()) + self.focal_loss(pred_bin, y_bin)
        loss = 0.2 * c_loss + 0.4 * multi_loss + 0.4 * bin_loss

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
        bin_loss = self.loss_fn(pred_bin, y_bin.float()) + self.focal_loss(pred_bin, y_bin)
        loss = 0.2 * c_loss + 0.4 * multi_loss + 0.4 * bin_loss

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