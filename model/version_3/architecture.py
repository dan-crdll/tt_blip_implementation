import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import copy

from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAUROC, MultilabelF1Score, MultilabelAveragePrecision
from model.version_3.utils.loss_fn import DistanceLoss
from model.version_3.layers.feature_extraction import FeatureExtraction
from model.version_3.layers.cross_attention_block import CrossAttnBlock
from model.version_3.utils.blip2_model import Blip2Model


class Model(L.LightningModule):
    def __init__(self, embed_dim, num_heads, hidden_dim, temp=1.0, momentum=0.9, queue_size=32):
        super().__init__()

        # -- Feature Extraction Modules --
        self.feature_extraction = FeatureExtraction('cuda', temp)
        self.multimodal_feature_extraction = Blip2Model("dandelin/vilt-b32-mlm")

        # -- Cross-Attention Fusion Layers --
        self.fusion_layer = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads, hidden_dim)
            for _ in range(6)
        ])

        # -- Classification Head --
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 5)  # 1 binary + 4 multilabel outputs
        )

        # -- Loss Functions --
        self.loss_fn = nn.BCEWithLogitsLoss()
        # self.moco_loss = MocoLoss(
        #     copy.deepcopy(self.feature_extraction),
        #     momentum=momentum,
        #     queue_size=queue_size,
        #     temp=temp
        # )

        # -- Log Variance for Uncertainty Weighting --
        self.log_var = nn.Parameter(torch.zeros(2))
        self.dist_loss = DistanceLoss()

        # -- Metrics (Validation Only) --
        self._init_metrics()

    def _init_metrics(self):
        # Binary classification metrics
        self.val_acc_bin = Accuracy(task='binary')
        self.val_f1_bin = F1Score(task='binary')
        self.val_auc_bin = BinaryAUROC()

        # Multilabel classification metrics (4 labels)
        self.val_acc_multi = Accuracy(task='multilabel', num_labels=4)
        self.val_cf1_multi = MultilabelF1Score(num_labels=4, average='macro')
        self.val_of1_multi = MultilabelF1Score(num_labels=4, average='micro')
        self.val_map_multi = MultilabelAveragePrecision(num_labels=4)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)

        return optimizer

    def forward(self, img, txt, orig, labels):
        # Unimodal features and contrastive loss
        (z_i, z_t), (z_vit, z_bert), contrastive_loss = self.feature_extraction(img, txt, orig, labels)

        # Multimodal features and auxiliary moco loss
        z_tm = self.multimodal_feature_extraction(img, txt)

        clip_distance_t = self.dist_loss(z_bert, z_tm)
        clip_distance_i = self.dist_loss(z_vit, z_tm)
        clip_distance = (clip_distance_t + clip_distance_i) / 2.0

        loss = contrastive_loss + 0.3 * clip_distance

        # Fusion via attention blocks
        z = z_t
        for layer in self.fusion_layer:
            z = layer(z, z_i, z_tm)
        z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), 1).squeeze(-1)

        # Classification
        y = self.classifier(z)
        return y, loss

    def total_loss(self, contrastive, classification):
        return 0.3 * contrastive + 0.7 * classification

    def _step(self, split, batch):
        img, txt, (y_bin, y_multi), orig = batch
        pred, c_loss = self(img, txt, orig, y_multi)

        pred_bin = pred[:, 0]
        bin_loss = self.loss_fn(pred_bin, y_bin.float())

        mask = (y_bin == 0)
        pred_multi = pred[:, 1:]

        if mask.sum() > 0:
            multi_loss = self.loss_fn(pred_multi[mask], y_multi[mask].float())
            cls_loss = (bin_loss + multi_loss) / 2.0
        else:
            cls_loss = bin_loss 

        loss = self.total_loss(c_loss, cls_loss)

        # Logging losses
        self.log(f"{split}/loss", loss, on_step=True if split == "Train" else False, on_epoch=True, prog_bar=True)
        self.log(f"{split}/loss_bin", bin_loss, on_step=False, on_epoch=True)
        self.log(f"{split}/loss_multi", multi_loss, on_step=False, on_epoch=True)
        self.log(f"{split}/contrastive_loss", c_loss, on_step=False, on_epoch=True)

        # Validation metrics
        if split == 'Val':
            pred_bin_sigmoid = torch.sigmoid(pred_bin)
            self.val_acc_bin.update(pred_bin_sigmoid, y_bin)
            self.val_f1_bin.update(pred_bin_sigmoid, y_bin)
            self.val_auc_bin.update(pred_bin_sigmoid, y_bin)

            if mask.sum() > 0:
                pred_multi_sigmoid = torch.sigmoid(pred_multi[mask])
                self.val_acc_multi.update(pred_multi_sigmoid, y_multi[mask])
                self.val_cf1_multi.update(pred_multi_sigmoid, y_multi[mask])
                self.val_of1_multi.update(pred_multi_sigmoid, y_multi[mask])
                self.val_map_multi.update(pred_multi_sigmoid, y_multi[mask].long())

        return loss

    def training_step(self, batch, batch_idx):
        return self._step("Train", batch)
    
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        return self._step("Val", batch)

    def on_validation_epoch_end(self):
        # Log binary metrics
        self.log("Val/acc_bin", self.val_acc_bin.compute(), prog_bar=True)
        self.log("Val/f1_bin", self.val_f1_bin.compute(), prog_bar=True)
        self.log("Val/auc_bin", self.val_auc_bin.compute(), prog_bar=True)

        # Log multilabel metrics
        self.log("Val/acc_multi", self.val_acc_multi.compute(), prog_bar=True)
        self.log("Val/cf1_multi", self.val_cf1_multi.compute(), prog_bar=True)
        self.log("Val/of1_multi", self.val_of1_multi.compute(), prog_bar=True)
        self.log("Val/mAP_multi", self.val_map_multi.compute(), prog_bar=True)

        # Reset all metrics
        self.val_acc_bin.reset()
        self.val_f1_bin.reset()
        self.val_auc_bin.reset()
        self.val_acc_multi.reset()
        self.val_cf1_multi.reset()
        self.val_of1_multi.reset()
        self.val_map_multi.reset()
