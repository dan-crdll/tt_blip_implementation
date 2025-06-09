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

        # -- Feature Extraction Modules --
        self.feature_extraction = feature_extraction_layer
        self.use_blip = use_blip
        if use_blip:
            self.multimodal_feature_extraction = Blip2Model("Salesforce/blip-itm-base-coco")

        # -- Cross-Attention Fusion Layers --
        self.fusion_layer = fusion_layer

        # -- Classification Head --
        self.classifier_bin = classifier_bin
        self.classifier_multi = classifier_multi

        # -- Loss Functions (precompiled) --
        self.loss_fn_bin = nn.BCEWithLogitsLoss(reduction='mean')
        self.loss_fn_multi = nn.BCEWithLogitsLoss(reduction='mean')

        # -- Log Variance for Uncertainty Weighting --
        self.dist_loss = DistanceLoss()
        self.lr = lr

        # -- Metrics (Validation Only) --
        self._init_metrics()

        self.num = 0
        self.epoch_tracker = epoch_tracker
        self.data_module = dataModule
        self.box_detector = BoxDetector()
        
        # Optimization: Cache weight decay parameters
        self.weight_decay = 0.02
        self._optimizer_param_groups = None

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
        
        # IoU metrics with optimized settings
        self.iou = IntersectionOverUnion()
        self.iou50 = IntersectionOverUnion(iou_threshold=0.5)
        self.iou75 = IntersectionOverUnion(iou_threshold=0.75)

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

    def _get_optimizer_param_groups(self):
        """Cache parameter groups for optimizer to avoid recomputing"""
        if self._optimizer_param_groups is not None:
            return self._optimizer_param_groups
            
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn

                if isinstance(m, whitelist_weight_modules):
                    if pn.endswith("weight"):
                        decay.add(fpn)
                    else:
                        no_decay.add(fpn)
                elif isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay

        # Safety check
        assert len(inter_params) == 0, f"Parameters in both decay and no_decay sets: {inter_params}"

        # Add missing parameters to no_decay
        missing = param_dict.keys() - union_params
        if missing:
            print(f"[Info] Adding {len(missing)} uncategorized parameters to no_decay:\n{missing}")
            no_decay.update(missing)

        self._optimizer_param_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0}
        ]
        
        return self._optimizer_param_groups

    def configure_optimizers(self):
        optimizer_grouped_parameters = self._get_optimizer_param_groups()

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8
            # Note: fused=True is incompatible with gradient clipping
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=5, eta_min=1e-7
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "Val/loss"
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @torch.jit.script_if_tracing
    def _iou_vectorized(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Optimized vectorized IoU computation"""
        # Intersection coordinates
        inter_coords = torch.stack([
            torch.max(pred_boxes[:, 0], target_boxes[:, 0]),  # x1
            torch.max(pred_boxes[:, 1], target_boxes[:, 1]),  # y1
            torch.min(pred_boxes[:, 2], target_boxes[:, 2]),  # x2
            torch.min(pred_boxes[:, 3], target_boxes[:, 3])   # y2
        ], dim=1)
        
        # Intersection area
        inter_wh = (inter_coords[:, 2:] - inter_coords[:, :2]).clamp(min=0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]

        # Box areas
        pred_wh = pred_boxes[:, 2:] - pred_boxes[:, :2]
        target_wh = target_boxes[:, 2:] - target_boxes[:, :2]
        
        area_pred = pred_wh[:, 0] * pred_wh[:, 1]
        area_target = target_wh[:, 0] * target_wh[:, 1]
        
        # Union area
        union = area_pred + area_target - inter + 1e-6

        return inter / union

    def iou_loss(self, pred_boxes, target_boxes):
        return 1 - self._iou_vectorized(pred_boxes, target_boxes).mean()

    def forward(self, img, txt, orig, labels, split='Train'):
        # Unimodal features and contrastive loss
        (z_i, z_t), contrastive_loss = self.feature_extraction(img, txt, orig, labels, split)

        # Multimodal features and auxiliary loss
        if self.use_blip:
            z_tm = self.multimodal_feature_extraction(img, txt)
            
            # Compute distance losses in parallel
            clip_distance = (
                self.dist_loss(z_t, z_tm) + 
                self.dist_loss(z_i, z_tm)
            ) * 0.5
        else:
            z_tm = None 
            clip_distance = 0

        loss = contrastive_loss + 0.3 * clip_distance

        # Fusion via attention blocks
        z = z_t
        for layer in self.fusion_layer:
            z, z_it = layer(z, z_i, z_tm)

        bbox = self.box_detector(z_it[:, 1:])
        z = z[:, 0]

        # Classification
        y_bin = self.classifier_bin(z).squeeze(-1)
        y_multi = self.classifier_multi(z)
        
        return (y_bin, y_multi), bbox, loss, (z_i, z_t)

    def _prepare_bbox_format(self, bbox_tensor):
        """Prepare bbox in required format for metrics"""
        return [{
            "boxes": bbox_tensor[i, :].unsqueeze(0),
            "labels": torch.tensor([0], device=self.device, dtype=torch.long)
        } for i in range(bbox_tensor.shape[0])]

    def _step(self, split, batch):
        img, txt, (y_bin, y_multi, bbox), orig = batch
        (pred_bin, pred_multi), pred_bbox, c_loss, (z_img_b, z_txt_b) = self(
            img, txt, orig, y_multi, split
        )

        # Compute losses in parallel where possible
        bin_loss = self.loss_fn_bin(pred_bin, y_bin.float())
        multi_loss = self.loss_fn_multi(pred_multi, y_multi.float())
        
        # Apply sigmoid once and reuse
        pred_bbox_sigmoid = torch.sigmoid(pred_bbox)
        
        # Optimized bbox loss computation
        bbox_loss = (
            F.mse_loss(pred_bbox_sigmoid, bbox, reduction='mean') + 
            self.iou_loss(pred_bbox_sigmoid, bbox)
        )

        total_loss = 0.1 * c_loss + bin_loss + multi_loss + bbox_loss

        # Efficient logging
        log_dict = {
            f"{split}/loss": total_loss,
            f"{split}/loss_bin": bin_loss,
            f"{split}/loss_multi": multi_loss,
            f"{split}/contrastive_loss": c_loss,
            f"{split}/bbox_loss": bbox_loss,
        }
        
        # Get batch size from appropriate input
        if isinstance(img, torch.Tensor):
            batch_size = img.size(0)
        elif isinstance(img, (list, tuple)):
            batch_size = len(img)
        else:
            batch_size = None
        
        self.log_dict(
            log_dict,
            on_step=(split == "Train"),
            on_epoch=True,
            prog_bar=(split == "Train"),  # Only show progress bar for training
            sync_dist=True,
            batch_size=batch_size
        )

        # Validation metrics (computed only when needed)
        if split == 'Val':
            with torch.no_grad():  # Ensure no gradients for validation metrics
                pred_bin_sigmoid = torch.sigmoid(pred_bin)
                pred_multi_sigmoid = torch.sigmoid(pred_multi)
                
                # Update binary metrics
                self.val_acc_bin.update(pred_bin_sigmoid, y_bin)
                self.val_f1_bin.update(pred_bin_sigmoid, y_bin)
                self.val_auc_bin.update(pred_bin_sigmoid, y_bin)

                # Update multilabel metrics
                self.val_acc_multi.update(pred_multi_sigmoid, y_multi)
                self.val_cf1_multi.update(pred_multi_sigmoid, y_multi)
                self.val_of1_multi.update(pred_multi_sigmoid, y_multi)
                self.val_map_multi.update(pred_multi_sigmoid, y_multi.long())
                
                # Update IoU metrics
                pred_bbox_format = self._prepare_bbox_format(pred_bbox_sigmoid)
                bbox_format = self._prepare_bbox_format(bbox)
                
                self.iou.update(pred_bbox_format, bbox_format)
                self.iou50.update(pred_bbox_format, bbox_format)
                self.iou75.update(pred_bbox_format, bbox_format)

        return total_loss

    def on_train_epoch_start(self):
        if self.data_module:
            self.data_module.current_epoch = self.current_epoch
        
        if self.epoch_tracker:
            self.epoch_tracker.set(self.current_epoch)

    def training_step(self, batch, batch_idx):
        self.feature_extraction.train()
        return self._step("Train", batch)

    def on_train_epoch_end(self):
        # More aggressive cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def validation_step(self, batch, batch_idx):
        self.feature_extraction.eval()
        return self._step("Val", batch)

    def on_validation_epoch_end(self):
        # Compute and log all metrics at once
        val_metrics = {
            "Val/acc_bin": self.val_acc_bin.compute(),
            "Val/f1_bin": self.val_f1_bin.compute(), 
            "Val/auc_bin": self.val_auc_bin.compute(),
            "Val/acc_multi": self.val_acc_multi.compute(),
            "Val/cf1_multi": self.val_cf1_multi.compute(),
            "Val/of1_multi": self.val_of1_multi.compute(),
            "Val/mAP_multi": self.val_map_multi.compute(),
            "Val/IoU": self.iou.compute()['iou'],
            "Val/IoU50": self.iou50.compute()['iou'],
            "Val/IoU75": self.iou75.compute()['iou'],
        }
        
        self.log_dict(val_metrics, prog_bar=True, sync_dist=True)

        # Reset all metrics at once
        metrics_to_reset = [
            self.val_acc_bin, self.val_f1_bin, self.val_auc_bin,
            self.val_acc_multi, self.val_cf1_multi, self.val_of1_multi, self.val_map_multi,
            self.iou, self.iou50, self.iou75
        ]
        
        for metric in metrics_to_reset:
            metric.reset()