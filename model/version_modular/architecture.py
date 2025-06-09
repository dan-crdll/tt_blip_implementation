import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import copy

from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.classification import BinaryAUROC, MultilabelF1Score, MultilabelAveragePrecision
from model.version_3.utils.loss_fn import DistanceLoss, FocalLoss, AutomaticWeightedLoss
from model.version_3.layers.feature_extraction import FeatureExtraction
from model.version_3.layers.cross_attention_block import CrossAttnBlock
from model.version_3.layers.memory import Memory
from model.version_3.utils.blip2_model import Blip2Model
from model.version_modular.layers.box_detector import BoxDetector



class Model(L.LightningModule):
    def __init__(
        self, 
        feature_extraction_layer,
        fusion_layer,
        classifier_bin,
        classifier_multi,
        lr=1e-5, 
        epoch_tracker=None,
        use_blip=1,
        dataModule=None
        ):
        super().__init__()

        # self.automatic_optimization = False

        # -- Feature Extraction Modules --
        self.feature_extraction = feature_extraction_layer
        self.use_blip = use_blip
        if use_blip:
            self.multimodal_feature_extraction = Blip2Model("Salesforce/blip-itm-base-coco")
            # self.proj = nn.Linear(1024, 768)

        # -- Cross-Attention Fusion Layers --
        self.fusion_layer = fusion_layer

        # -- Classification Head --
        self.classifier_bin = classifier_bin

        self.classifier_multi = classifier_multi

        # -- Loss Functions --
        self.loss_fn_bin = nn.BCEWithLogitsLoss()
        self.loss_fn_multi = nn.BCEWithLogitsLoss()

        # -- Log Variance for Uncertainty Weighting --
        self.dist_loss = DistanceLoss()
        self.lr = lr

        # -- Metrics (Validation Only) --
        self._init_metrics()

        self.num = 0
        self.epoch_tracker = epoch_tracker

        self.data_module = dataModule

        self.box_detector = BoxDetector()

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
        self.iou = IntersectionOverUnion()
        self.iou50 = IntersectionOverUnion(iou_threshold=0.5)
        self.iou75 = IntersectionOverUnion(iou_threshold=0.75)
        self.weight_decay = 0.02

    def load_partial_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        model_state_dict = self.state_dict()
        
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state_dict and v.size() == model_state_dict[k].size()
        }
        print(f"Loaded weights: {list(filtered_state_dict.keys())}")
        
        model_state_dict.update(filtered_state_dict)
        self.load_state_dict(model_state_dict, strict=False)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)


        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn

                # # Skip task_weights parameters

                if isinstance(m, whitelist_weight_modules):
                    if pn.endswith("weight"):
                        decay.add(fpn)
                    else:
                        no_decay.add(fpn)
                elif isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {
            pn: p for pn, p in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay

        # Safety check
        assert len(inter_params) == 0, f"Parameters in both decay and no_decay sets: {inter_params}"

        # Add missing parameters to no_decay
        missing = param_dict.keys() - union_params
        if missing:
            print(f"[Info] Adding {len(missing)} uncategorized parameters to no_decay:\n{missing}")
            no_decay.update(missing)

        optimizer_grouped_parameters = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0}
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-7),
            "interval": "epoch",
            "frequency": 1,
            "monitor":"Val/loss"
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, img, txt, orig, labels, split='Train'):
        
        # Unimodal features and contrastive loss
        (z_i, z_t), contrastive_loss = self.feature_extraction(img, txt, orig, labels, split)

        # Multimodal features and auxiliary moco loss

        if self.use_blip:
            z_tm = self.multimodal_feature_extraction(img, txt)
            # z_tm = self.proj(z_tm)

            clip_distance_t = self.dist_loss(z_t, z_tm)
            clip_distance_i = self.dist_loss(z_i, z_tm)
            clip_distance = (clip_distance_t + clip_distance_i) / 2.0
        else:
            z_tm = None 
            clip_distance = 0


        loss = contrastive_loss + 0.3 * clip_distance

        # Fusion via attention blocks
        z = z_t
        for k, layer in enumerate(self.fusion_layer):
            z = layer(z, z_i, z_tm)

        bbox = self.box_detector(z[:, 1:])
        z = z[:, 0]

        # Classification
        y_bin = self.classifier_bin(z).squeeze(-1)
        y_multi = self.classifier_multi(z)
        return (y_bin, y_multi, bbox), loss, (z_i, z_t)

    def _iou(self, pred_boxes, target_boxes):
        # pred_boxes, target_boxes: [B, 4] in format (x1, y1, x2, y2)

        # Intersection
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        # Union
        area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_gt = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = area_pred + area_gt - inter + 1e-6

        return inter / union

    def iou_loss(self, pred_boxes, target_boxes):
        return 1 - self._iou(pred_boxes, target_boxes).mean()

    def _step(self, split, batch):
        img, txt, (y_bin, y_multi, bbox), orig = batch
        (pred_bin, pred_multi, pred_bbox), c_loss, (z_img_b, z_txt_b) = self(img, txt, orig, y_multi, split)

        # --- Standard Feed Forward Pass --- 
        bin_loss = self.loss_fn_bin(pred_bin, y_bin.float())
        multi_loss = self.loss_fn_multi(pred_multi, y_multi.float())
        pred_bbox = F.sigmoid(pred_bbox)

        bbox_loss = torch.norm(pred_bbox - bbox).mean() + self.iou_loss(pred_bbox, bbox)

        pred_bbox = [{
            "boxes": pred_bbox[i, :].unsqueeze(0),
            "labels": torch.tensor([0], device=self.device)
        } for i in range(pred_bbox.shape[0])]

        bbox = [{
            "boxes": bbox[i, :].unsqueeze(0),
            "labels": torch.tensor([0], device=self.device)
        } for i in range(bbox.shape[0])]

        total_loss = 0.1 * c_loss + bin_loss + multi_loss + bbox_loss
        # -----------------------------------

        # Log sulle loss
        self.log(f"{split}/loss", total_loss, on_step=True if split == "Train" else False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split}/loss_bin", bin_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{split}/loss_multi", multi_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{split}/contrastive_loss", c_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{split}/bbox_loss", bbox_loss, on_step=False, on_epoch=True, sync_dist=True)


        # Validation metrics
        if split == 'Val':
            pred_bin_sigmoid = torch.sigmoid(pred_bin)
            self.val_acc_bin.update(pred_bin_sigmoid, y_bin)
            self.val_f1_bin.update(pred_bin_sigmoid, y_bin)
            self.val_auc_bin.update(pred_bin_sigmoid, y_bin)

            pred_multi_sigmoid = torch.sigmoid(pred_multi)
            self.val_acc_multi.update(pred_multi_sigmoid, y_multi)
            self.val_cf1_multi.update(pred_multi_sigmoid, y_multi)
            self.val_of1_multi.update(pred_multi_sigmoid, y_multi)
            self.val_map_multi.update(pred_multi_sigmoid, y_multi.long())
            self.iou.update(pred_bbox, bbox)
            self.iou50.update(pred_bbox, bbox)
            self.iou75.update(pred_bbox, bbox)

        return total_loss

    def on_train_epoch_start(self):
        if self.data_module:
            self.data_module.current_epoch = self.current_epoch
        
        # Update epoch tracker if it exists
        if self.epoch_tracker:
            self.epoch_tracker.set(self.current_epoch)


    def training_step(self, batch, batch_idx):
        self.feature_extraction.train()
        loss = self._step("Train", batch)

        return loss
    
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        self.feature_extraction.eval()
        loss = self._step("Val", batch)

        return loss

    def on_validation_epoch_end(self):
        # Log binary metrics
        self.log("Val/acc_bin", self.val_acc_bin.compute(), prog_bar=True, sync_dist=True)
        self.log("Val/f1_bin", self.val_f1_bin.compute(), prog_bar=True, sync_dist=True)
        self.log("Val/auc_bin", self.val_auc_bin.compute(), prog_bar=True, sync_dist=True)

        # Log multilabel metrics
        self.log("Val/acc_multi", self.val_acc_multi.compute(), prog_bar=True, sync_dist=True)
        self.log("Val/cf1_multi", self.val_cf1_multi.compute(), prog_bar=True, sync_dist=True)
        self.log("Val/of1_multi", self.val_of1_multi.compute(), prog_bar=True, sync_dist=True)
        self.log("Val/mAP_multi", self.val_map_multi.compute(), prog_bar=True, sync_dist=True)

        # Log IoU
        self.log("Val/IoU", self.iou.compute()['iou'], prog_bar=True, sync_dist=True)
        self.log("Val/IoU50", self.iou50.compute()['iou'], prog_bar=True, sync_dist=True)
        self.log("Val/IoU75", self.iou75.compute()['iou'], prog_bar=True, sync_dist=True)

        # Reset all metrics
        self.val_acc_bin.reset()
        self.val_f1_bin.reset()
        self.val_auc_bin.reset()
        self.val_acc_multi.reset()
        self.val_cf1_multi.reset()
        self.val_of1_multi.reset()
        self.val_map_multi.reset()
        self.iou.reset()
        self.iou50.reset()
        self.iou75.reset()
