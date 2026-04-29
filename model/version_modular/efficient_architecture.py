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
from model.version_modular.utils.blip2_model import Blip2Model
from model.version_modular.utils.multimodal_models import SigClip, FlavaModelWrapper
from model.version_modular.layers.box_detector import BoxDetector

import numpy as np
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os



def compute_iou(pred_boxes, target_boxes, coverage_threshold=1.0, reduce=True):
    i_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    i_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    i_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    i_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    intersection = torch.clamp(i_x2 - i_x1, min=0) * torch.clamp(i_y2 - i_y1, min=0)

    pred_area = (torch.clamp(pred_boxes[:, 2] - pred_boxes[:, 0], min=0) * 
                 torch.clamp(pred_boxes[:, 3] - pred_boxes[:, 1], min=0))
    target_area = (torch.clamp(target_boxes[:, 2] - target_boxes[:, 0], min=0) * 
                   torch.clamp(target_boxes[:, 3] - target_boxes[:, 1], min=0))
    union = pred_area + target_area - intersection
    union = torch.clamp(union, min=1e-6)  # Evita divisione per zero
    
    # If either box has zero area, IoU = 0
    zero_area = (target_area == 0)
    
    iou = torch.ones_like(intersection)
    valid_mask = ~zero_area
    
    if valid_mask.any():
        iou[valid_mask] = intersection[valid_mask] / torch.clamp(union[valid_mask], min=1e-8)

    if coverage_threshold < 1.0:
        # Imposta IoU = 0 per boxes che non soddisfano la copertura minima
        coverage = intersection / torch.clamp(target_area, min=1e-8)
        low_coverage_mask = coverage < coverage_threshold
        iou[low_coverage_mask] = 0.0

    return iou.mean() if reduce else iou

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
        arcface=True,
        bbox_img=False,
        bbox_txt=False,
        multimodal_model = None
        ):
        super().__init__()

        # -- Feature Extraction Modules --
        self.feature_extraction = feature_extraction_layer
        self.use_blip = use_blip
        if use_blip:
            if multimodal_model == "flava":
                self.multimodal_feature_extraction = FlavaModelWrapper("facebook/flava-full")
            elif multimodal_model == "sigclip":
                self.multimodal_feature_extraction = SigClip("google/siglip2-base-patch32-256x")
            else:
                self.multimodal_feature_extraction = Blip2Model("Salesforce/blip-itm-base-coco")

        # -- Cross-Attention Fusion Layers --
        self.fusion_layer = fusion_layer
        self.bbox_img = bbox_img
        self.bbox_txt = bbox_txt
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

        # -- BOX DETECTION --
        if bbox_img:
            self.bbox_regressor = BoxDetector()
        if bbox_txt:
            pass    # TODO: WORD TOKEN DETECTOR

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
        
        # BBOX Img metrics
        self.val_iou = 0.0
        self.val_iou_95 = 0.0
        self.val_iou_50 = 0.0
        self.val_iou_25 = 0.0

        self.n = 0

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
            z, z_it, z_it_ = layer(z, z_i, z_tm, True)  # z_it o z_i ???
        
        if self.bbox_img:
            bbox_i = self.bbox_regressor(z_it_)
        else:
            bbox_i = None
        
        #TODO
        bbox_t = None
        z = z[:, 0]

        y_bin = self.classifier_bin(z).squeeze(-1)
        y_multi = self.classifier_multi(z)
        
        return (y_bin, y_multi), loss, (z_i, z_t), (bbox_i, bbox_t)

    def _step(self, split, batch, batch_idx=0):
        # Estrazione del batch
        if self.bbox_img and self.bbox_txt:
            img, txt, (y_bin, y_multi), orig, (bbox_img, bbox_txt) = batch
        else:
            if self.bbox_img:
                img, txt, (y_bin, y_multi), orig, bbox_img = batch
            if self.bbox_txt:
                img, txt, (y_bin, y_multi), orig, bbox_txt = batch
            if not self.bbox_img and not self.bbox_txt:
                img, txt, (y_bin, y_multi), orig = batch
        # -- Fine estrazione batch

        (pred_bin, pred_multi), c_loss, (z_img_b, z_txt_b), (p_bbox_i, p_bbox_t) = self(
            img, txt, orig, y_multi, y_bin, split
        )

        # Compute losses in parallel where possible
        bin_loss = self.loss_fn_bin(pred_bin, y_bin.long().squeeze(-1) if self.arcface else y_bin.float())
        multi_loss = self.loss_fn_multi(pred_multi, y_multi.float())
        
        bbox_loss = 0
        
        if self.bbox_img:
            iou = compute_iou(p_bbox_i[:, :4], bbox_img, 1.0, False)
            filtered = iou[iou != 1]
            if filtered.numel() == 0:
                filtered = torch.tensor([1.0], device=iou.device)
            
            pred_loss = self.loss_fn_bin(p_bbox_i[:, -1], (y_multi[:, 0] + y_multi[:, 1]).float())

            bbox_loss += F.mse_loss(p_bbox_i[:, :4], bbox_img) + (1 - filtered).mean() + pred_loss
        if self.bbox_txt:
            pass # TODO

        total_loss = 0.1 * c_loss + bin_loss + multi_loss + bbox_loss

        # Replace the problematic section in your _step method (around line 430-480)
        # The issue is that bbox coordinates might be tensors with multiple elements

        if batch_idx == 0:
            folder = f'./boxes_samples/{split}/{self.current_epoch}'
            os.makedirs(folder, exist_ok=True)

            # Salva le immagini con i bounding box per ogni elemento del batch
            batch_size = len(img)
            
            for i in range(batch_size):  # Per ogni immagine nel batch
                # img[i] è già una PIL Image
                img_pil = img[i]
                img_width, img_height = img_pil.size
                
                # Crea una figura con due subplot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Subplot 1: Immagine con bounding box originale
                ax1.imshow(img_pil)
                ax1.set_title(f'Original BBox - Sample {i}')
                ax1.axis('off')
                
                if self.bbox_img and i < len(bbox_img):
                    # Converte coordinate normalizzate in pixel
                    bbox_orig = bbox_img[i].detach().cpu()
                    # FIX: Use .item() to extract scalar values
                    x1, y1, x2, y2 = bbox_orig
                    x1_px, y1_px = int(x1.item() * img_width), int(y1.item() * img_height)
                    x2_px, y2_px = int(x2.item() * img_width), int(y2.item() * img_height)
                    
                    # Disegna il rettangolo originale (rosso)
                    from matplotlib.patches import Rectangle
                    rect_orig = Rectangle((x1_px, y1_px), x2_px - x1_px, y2_px - y1_px, 
                                        linewidth=2, edgecolor='red', facecolor='none', 
                                        label='Original')
                    ax1.add_patch(rect_orig)
                    ax1.legend()
                
                # Subplot 2: Immagine con bounding box predetto
                ax2.imshow(img_pil)
                ax2.set_title(f'Predicted BBox - Sample {i}')
                ax2.axis('off')
                
                if self.bbox_img and i < len(p_bbox_i):
                    # Converte coordinate normalizzate predette in pixel
                    # FIX: Use .item() for scalar extraction and add bounds checking
                    bbox_pred_confidence = p_bbox_i[i, -1].detach().cpu().item()
                    if bbox_pred_confidence < 0.5:
                        x1_px, y1_px, x2_px, y2_px = 0.0, 0.0, 0.0, 0.0
                    else:
                        bbox_pred = p_bbox_i[i, :4].detach().cpu()
                        x1, y1, x2, y2 = bbox_pred
                        x1_px, y1_px = int(x1.item() * img_width), int(y1.item() * img_height)
                        x2_px, y2_px = int(x2.item() * img_width), int(y2.item() * img_height)
                    
                    # Disegna il rettangolo predetto (blu)
                    rect_pred = Rectangle((x1_px, y1_px), x2_px - x1_px, y2_px - y1_px, 
                                        linewidth=2, edgecolor='blue', facecolor='none',
                                        label='Predicted')
                    ax2.add_patch(rect_pred)
                    ax2.legend()
                
                # Salva l'immagine
                plt.tight_layout()
                plt.savefig(f'{folder}/sample_{i}_bbox_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # Opzionalmente, salva anche una versione con entrambi i box sovrapposti
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                ax.imshow(img_pil)
                ax.set_title(f'Overlay Comparison - Sample {i}')
                ax.axis('off')
                
                if self.bbox_img and i < len(bbox_img) and i < len(p_bbox_i):
                    # Box originale (rosso)
                    bbox_orig = bbox_img[i].detach().cpu()
                    x1, y1, x2, y2 = bbox_orig
                    x1_px, y1_px = int(x1.item() * img_width), int(y1.item() * img_height)
                    x2_px, y2_px = int(x2.item() * img_width), int(y2.item() * img_height)
                    rect_orig = Rectangle((x1_px, y1_px), x2_px - x1_px, y2_px - y1_px, 
                                        linewidth=2, edgecolor='red', facecolor='none',
                                        label='Original')
                    ax.add_patch(rect_orig)
                    
                    # Box predetto (blu)
                    bbox_pred_confidence = p_bbox_i[i, -1].detach().cpu().item()
                    if bbox_pred_confidence >= 0.5:
                        bbox_pred = p_bbox_i[i, :4].detach().cpu()
                        x1, y1, x2, y2 = bbox_pred
                        x1_px, y1_px = int(x1.item() * img_width), int(y1.item() * img_height)
                        x2_px, y2_px = int(x2.item() * img_width), int(y2.item() * img_height)
                    else:
                        x1_px, y1_px, x2_px, y2_px = 0, 0, 0, 0
                        
                    rect_pred = Rectangle((x1_px, y1_px), x2_px - x1_px, y2_px - y1_px, 
                                        linewidth=2, edgecolor='blue', facecolor='none',
                                        label='Predicted')
                    ax.add_patch(rect_pred)
                    
                    ax.legend()
                
                plt.tight_layout()
                plt.savefig(f'{folder}/sample_{i}_bbox_overlay.png', dpi=150, bbox_inches='tight')
                plt.close()

        # Efficient logging
        log_dict = {
            f"{split}/loss": total_loss,
            f"{split}/loss_bin": bin_loss,
            f"{split}/loss_multi": multi_loss,
            f"{split}/contrastive_loss": c_loss,
            f"{split}/bbox_loss": bbox_loss
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

                # p_bbox_i = F.sigmoid(p_bbox_i)
                
                self.val_iou += compute_iou(p_bbox_i, bbox_img, 1.0)
                self.val_iou_95 += compute_iou(p_bbox_i, bbox_img, 0.95)
                self.val_iou_50 += compute_iou(p_bbox_i, bbox_img, 0.5)
                self.val_iou_25 += compute_iou(p_bbox_i, bbox_img, 0.25)
                self.n += 1
                
        return total_loss

    def on_train_epoch_start(self):
        if self.data_module:
            self.data_module.current_epoch = self.current_epoch
        
        if self.epoch_tracker:
            self.epoch_tracker.set(self.current_epoch)

    def training_step(self, batch, batch_idx):
        self.feature_extraction.train()
        return self._step("Train", batch, batch_idx)

    def on_train_epoch_end(self):
        # More aggressive cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.feature_extraction.eval()
        return self._step("Val", batch, batch_idx)

    def on_validation_epoch_end(self, dataloader_idx=0):
        # Compute and log all metrics at once
        val_metrics = {
            f"Val_{dataloader_idx}/acc_bin": self.val_acc_bin.compute(),
            f"Val_{dataloader_idx}/f1_bin": self.val_f1_bin.compute(), 
            f"Val_{dataloader_idx}/auc_bin": self.val_auc_bin.compute(),
            f"Val_{dataloader_idx}/acc_multi": self.val_acc_multi.compute(),
            f"Val_{dataloader_idx}/cf1_multi": self.val_cf1_multi.compute(),
            f"Val_{dataloader_idx}/of1_multi": self.val_of1_multi.compute(),
            f"Val_{dataloader_idx}/mAP_multi": self.val_map_multi.compute(),
            f"Val_{dataloader_idx}/IoU": self.val_iou / self.n if self.bbox_img else torch.tensor(0.0),
            f"Val_{dataloader_idx}/IoU_95": self.val_iou_95 / self.n if self.bbox_img else torch.tensor(0.0),
            f"Val_{dataloader_idx}/IoU_50": self.val_iou_50 / self.n if self.bbox_img else torch.tensor(0.0),
            f"Val_{dataloader_idx}/IoU_25": self.val_iou_25 / self.n if self.bbox_img else torch.tensor(0.0),
        }
        
        self.log_dict(val_metrics, prog_bar=True, sync_dist=True)

        # Reset all metrics at once
        metrics_to_reset = [
            self.val_acc_bin, self.val_f1_bin, self.val_auc_bin,
            self.val_acc_multi, self.val_cf1_multi, self.val_of1_multi, self.val_map_multi
        ]

        self.val_iou = 0
        self.val_iou_95 = 0
        self.val_iou_50 = 0
        self.val_iou_25 = 0
        self.n = 0
        
        for metric in metrics_to_reset:
            metric.reset()


