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

import torch
import numpy as np
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def compute_eer(y_scores, y_true):
    # Assicurati di lavorare su cpu e numpy
    y_true = y_true.detach().float().cpu().numpy()
    y_scores = y_scores.detach().float().cpu().numpy()
    y_true = np.round(y_true).astype(int)
    
    # Controlli di validità
    if len(np.unique(y_true)) < 2:
        return 0.5  # Se tutti i label sono uguali, EER = 0.5
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    # Controllo per valori NaN o infiniti
    if np.any(np.isnan(fpr)) or np.any(np.isnan(tpr)) or np.any(np.isinf(fpr)) or np.any(np.isinf(tpr)):
        return 0.5
    
    # Rimuovi duplicati e ordina
    unique_fpr, unique_indices = np.unique(fpr, return_index=True)
    unique_tpr = tpr[unique_indices]
    
    # Se abbiamo troppo pochi punti unici
    if len(unique_fpr) < 2:
        return 0.5
    
    try:
        # Interpolazione con controllo del dominio
        interp_func = interp1d(unique_fpr, unique_tpr, kind='linear', 
                              bounds_error=False, fill_value=(unique_tpr[0], unique_tpr[-1]))
        
        # Trova EER usando brentq
        def eer_func(x):
            return 1.0 - x - interp_func(x)
        
        # Controlla che la funzione sia valida agli estremi
        if np.isnan(eer_func(0.0)) or np.isnan(eer_func(1.0)):
            # Fallback: calcola EER come minima distanza tra FPR e FNR
            eer_idx = np.argmin(np.abs(fpr - fnr))
            return (fpr[eer_idx] + fnr[eer_idx]) / 2.0
        
        eer = brentq(eer_func, 0., 1.)
        return eer
        
    except (ValueError, RuntimeError):
        # Fallback: calcola EER come minima distanza tra FPR e FNR
        eer_idx = np.argmin(np.abs(fpr - fnr))
        return (fpr[eer_idx] + fnr[eer_idx]) / 2.0


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
        dataModule=None,
        arcface=True
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
        if arcface:
            from model.version_modular.utils.arcface import ArcMarginProduct

            self.head = nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.Linear(1024, 768),
                nn.ReLU()
            )
            self.classifier_bin = ArcMarginProduct(768, 2)
        else:
            self.classifier_bin = classifier_bin
        self.classifier_multi = classifier_multi

        # -- Loss Functions (precompiled) --
        if arcface:
            self.loss_fn_bin = nn.CrossEntropyLoss()
        else:
            self.loss_fn_bin = nn.BCEWithLogitsLoss(reduction='mean')
        self.loss_fn_multi = nn.BCEWithLogitsLoss(reduction='mean')
        self.arcface = arcface
        # -- Log Variance for Uncertainty Weighting --
        self.dist_loss = DistanceLoss()
        self.lr = lr

        # -- Metrics (Validation Only) --
        self._init_metrics()

        self.num = 0
        self.epoch_tracker = epoch_tracker
        self.data_module = dataModule
        
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


    def forward(self, img, txt, orig, labels, bin_labels, split='Train'):
        # Unimodal features and contrastive loss
        (z_i, z_t), contrastive_loss = self.feature_extraction(img, txt, orig, labels, 'Val')

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

        z = z[:, 0]

        # Classification
        if self.arcface:
            z_bin = self.head(z)
            y_bin = self.classifier_bin(z_bin, bin_labels)
        else:
            y_bin = self.classifier_bin(z).squeeze(-1)
        y_multi = self.classifier_multi(z)
        
        return (y_bin, y_multi), loss, (z_i, z_t)

    def _step(self, split, batch):
        img, txt, (y_bin, y_multi), orig = batch
        (pred_bin, pred_multi), c_loss, (z_img_b, z_txt_b) = self(
            img, txt, orig, y_multi, y_bin, split
        )

        # Compute losses in parallel where possible
        bin_loss = self.loss_fn_bin(pred_bin, y_bin.long().squeeze(-1) if self.arcface else y_bin.float())
        multi_loss = self.loss_fn_multi(pred_multi, y_multi.float())
                
        total_loss = 0.1 * c_loss + bin_loss + multi_loss

        # Efficient logging
        log_dict = {
            f"{split}/loss": total_loss,
            f"{split}/loss_bin": bin_loss,
            f"{split}/loss_multi": multi_loss,
            f"{split}/contrastive_loss": c_loss
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

        pred_bin_sigmoid = torch.sigmoid(pred_bin) if not self.arcface else torch.argmax(torch.softmax(pred_bin, -1), -1).float().squeeze(-1)
        pred_multi_sigmoid = torch.sigmoid(pred_multi)
        self.log(f"{split}/EER", compute_eer(pred_bin_sigmoid, y_bin), on_step=(split=='Train'), on_epoch=True)


        # Validation metrics (computed only when needed)
        if split == 'Val':
            with torch.no_grad():  # Ensure no gradients for validation metrics                
                # Update binary metrics
                self.val_acc_bin.update(pred_bin_sigmoid, y_bin)
                self.val_f1_bin.update(pred_bin_sigmoid, y_bin)
                self.val_auc_bin.update(pred_bin_sigmoid, y_bin)

                # Update multilabel metrics
                self.val_acc_multi.update(pred_multi_sigmoid, y_multi)
                self.val_cf1_multi.update(pred_multi_sigmoid, y_multi)
                self.val_of1_multi.update(pred_multi_sigmoid, y_multi)
                self.val_map_multi.update(pred_multi_sigmoid, y_multi.long())
                
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
        }
        
        self.log_dict(val_metrics, prog_bar=True, sync_dist=True)

        # Reset all metrics at once
        metrics_to_reset = [
            self.val_acc_bin, self.val_f1_bin, self.val_auc_bin,
            self.val_acc_multi, self.val_cf1_multi, self.val_of1_multi, self.val_map_multi,
        ]
        
        for metric in metrics_to_reset:
            metric.reset()


