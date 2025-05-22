import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import copy

from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAUROC, MultilabelF1Score, MultilabelAveragePrecision
from model.version_3.utils.loss_fn import DistanceLoss, FocalLoss, AutomaticWeightedLoss
from model.version_3.layers.feature_extraction import FeatureExtraction
from model.version_3.layers.cross_attention_block import CrossAttnBlock
from model.version_3.layers.memory import Memory
from model.version_3.utils.blip2_model import Blip2Model


class Model(L.LightningModule):
    def __init__(
        self, 
        feature_extraction_layer,
        fusion_layer,
        classifier_bin,
        classifier_multi,
        lr=1e-5):
        super().__init__()

        # self.automatic_optimization = False

        # -- Feature Extraction Modules --
        self.feature_extraction = feature_extraction_layer
        self.multimodal_feature_extraction = Blip2Model("Salesforce/blip-itm-base-coco")

        # -- Cross-Attention Fusion Layers --
        self.fusion_layer = fusion_layer

        # self.memory = Memory(embed_dim)

        # self.root = nn.Sequential(
        #     nn.Linear(embed_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU()
        # )

        # self.awl = AutomaticWeightedLoss(3)

        # -- Classification Head --
        self.classifier_bin = classifier_bin

        self.classifier_multi = classifier_multi

        # -- Loss Functions --
        self.loss_fn_bin = nn.BCEWithLogitsLoss()
        self.loss_fn_multi = nn.BCEWithLogitsLoss()
        # self.loss_fn_bin = FocalLoss(alpha=0.45) # 0.337
        # self.loss_fn_multi = FocalLoss(alpha=torch.tensor([1-0.245, 1-0.29, 1-0.08, 1-0.19]), gamma=4)
        # self.moco_loss = MocoLoss(
        #     copy.deepcopy(self.feature_extraction),
        #     momentum=momentum,
        #     queue_size=queue_size,
        #     temp=temp
        # )

        # -- Log Variance for Uncertainty Weighting --
        self.dist_loss = DistanceLoss()
        self.lr = lr

        # -- Metrics (Validation Only) --
        self._init_metrics()

        # self.means = nn.Parameter(torch.zeros((3)), requires_grad=False)
        # self.m2 = nn.Parameter(torch.zeros((4)), requires_grad=False)
        self.num = 0

        # self.task_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=True)
        # self.initial_losses = None  # to be filled during the first training step
        # self.shared_params = [
        #     # p for p in list(self.root.parameters()) + 
        #     #         list(self.fusion_layer.parameters()) + 
        #     #         list(self.feature_extraction.parameters())
        #     # if p.requires_grad
        #     p for p in list(self.feature_extraction.parameters())
        #     if p.requires_grad
        # ]

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
        self.weight_decay = 1e-3

    def load_partial_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        model_state_dict = self.state_dict()
        
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state_dict and v.size() == model_state_dict[k].size()
        }
        print(f"Loaded weights: {list(filtered_state_dict.keys())}")
        
        # Carica i pesi compatibili
        model_state_dict.update(filtered_state_dict)
        self.load_state_dict(model_state_dict, strict=False)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # Gather names of task_weights parameters to exclude them from the first optimizer
        # task_weight_param_names = {name for name, _ in self.named_parameters() if _ is self.task_weights}

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn

                # # Skip task_weights parameters
                # if fpn in task_weight_param_names:
                #     continue

                if isinstance(m, whitelist_weight_modules):
                    if pn.endswith("weight"):
                        decay.add(fpn)
                    else:
                        no_decay.add(fpn)
                elif isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {
            pn: p for pn, p in self.named_parameters()
            #  if pn not in task_weight_param_names
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

        # optimizer_weights = torch.optim.Adam([self.task_weights], lr=1e-3)

        # return [{"optimizer": optimizer, "lr_scheduler": scheduler}, {"optimizer": optimizer_weights}]
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, img, txt, orig, labels, split='Train'):
        
        # Unimodal features and contrastive loss
        (z_i, z_t), contrastive_loss = self.feature_extraction(img, txt, orig, labels, split)
        # (z_i, z_t), (z_vit, z_bert) = self.feature_extraction(img, txt, orig, labels)
        # (z_i, z_t) = self.feature_extraction(img, txt, orig, labels)

        # Multimodal features and auxiliary moco loss

        z_tm = self.multimodal_feature_extraction(img, txt)

        clip_distance_t = self.dist_loss(z_t, z_tm)
        clip_distance_i = self.dist_loss(z_i, z_tm)
        clip_distance = (clip_distance_t + clip_distance_i) / 2.0

        # clip_distance = self.dist_loss(z_t[-1], z_i[-1])

        loss = contrastive_loss + 0.3 * clip_distance
        # loss = contrastive_loss

        # Fusion via attention blocks
        z = z_t
        for k, layer in enumerate(self.fusion_layer):
            z = layer(z, z_i, z_tm)

        z = z[:, 0]
        # z = self.root(z)
        # Classification
        y_bin = self.classifier_bin(z).squeeze(-1)
        y_multi = self.classifier_multi(z)
        return (y_bin, y_multi), loss, (z_i, z_t)

    def _step(self, split, batch):
        img, txt, (y_bin, y_multi), orig = batch
        (pred_bin, pred_multi), c_loss, (z_img_b, z_txt_b) = self(img, txt, orig, y_multi, split)

        # --- Standard Feed Forward Pass --- 
        bin_loss = self.loss_fn_bin(pred_bin, y_bin.float())
        multi_loss = self.loss_fn_multi(pred_multi, y_multi.float())
        # task_losses = torch.stack([bin_loss, multi_loss, c_loss])

        # if self.initial_losses is None:
        #     self.initial_losses = task_losses.detach()
        
        # weighted_losses = self.task_weights * task_losses
        # total_loss = weighted_losses.sum()
        # total_loss = self.awl(bin_loss, multi_loss, c_loss)

        total_loss = c_loss + bin_loss + multi_loss
        # -----------------------------------

        # gradnorm_loss = None
        # if split == 'Train':
        #     grads = []
        #     # --- Computation of Gw(i) ---
        #     for i in range(3):
        #         grad = torch.autograd.grad(
        #             outputs=(task_losses[i] * self.task_weights[i]),
        #             inputs=self.shared_params,
        #             retain_graph=True,
        #             allow_unused=True,
        #             create_graph=True
        #         )
        
        #         grad_norm = torch.norm(torch.stack([
        #             g.norm() for g in grad if g is not None
        #         ]), p=2)
        #         grads.append(grad_norm)
            # ----------------------------

            # --- Computation of Loss Ratios and Relative Inverse Training Rate ---
            # loss_ratios = (task_losses / self.initial_losses)
            # self.means.data = (self.means.data * self.num + loss_ratios.detach()) / (self.num + 1)
            # self.num = self.num + 1
            # inv_train_rates = loss_ratios / self.means.data
            # ----------------------------------------------------------------------

            # avg_grad = sum(grads) / len(grads)

            # --- Computation of L_gradNorm ---
            # gradnorm_loss = torch.sum(torch.abs(
            #     torch.stack(grads).to('cuda') - avg_grad.to('cuda') * inv_train_rates.to('cuda') ** 1.5
            # ))
            # ----------------------------------------------------------------------

        # Log sulle loss
        self.log(f"{split}/loss", total_loss, on_step=True if split == "Train" else False, on_epoch=True, prog_bar=True)
        self.log(f"{split}/loss_bin", bin_loss, on_step=False, on_epoch=True)
        self.log(f"{split}/loss_multi", multi_loss, on_step=False, on_epoch=True)
        self.log(f"{split}/contrastive_loss", c_loss, on_step=False, on_epoch=True)

        # # Logging pesi
        # if split == 'Train':
        #     for i, n in enumerate(['bin', 'multi', 'contrastive']):
        #         self.log(f"W/{n}_weight", self.task_weights[i], on_step=True, on_epoch=False)
        #     return total_loss, gradnorm_loss


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

        # if split == 'Train':
        #     with torch.no_grad():
        #         z_img_r = self.feature_extraction.feature_extractor_img(orig[0])
        #         z_txt_r = self.feature_extraction.feature_extractor_txt(orig[1])
        #         z_img_b = z_img_b.detach()
        #         z_txt_b = z_txt_b.detach()

        #         diff_it = self.dist_loss(z_img_b, z_txt_r)
        #         diff_ii = self.dist_loss(z_img_b, z_img_r)
        #         diff_ti = self.dist_loss(z_txt_b, z_img_r)
        #         diff_tt = self.dist_loss(z_txt_b, z_txt_r)

        return total_loss

    def training_step(self, batch, batch_idx):
        # torch.autograd.set_detect_anomaly(True)

        # optimizer_W, optimizer_w = self.optimizers()

        self.feature_extraction.train()
        # self.multimodal_feature_extraction.train()
        loss = self._step("Train", batch)
        # lambda_gradnorm = 1.0

        # loss = loss / 16
        # gradnorm_loss = gradnorm_loss / 16

        # if gradnorm_loss is not None:
        #     # total_loss = loss + lambda_gradnorm * gradnorm_loss
        #     self.log('Train/gradnorm_loss', gradnorm_loss, on_step=True, on_epoch=False, prog_bar=True)
        #     self.manual_backward(gradnorm_loss, retain_graph=True)

        #     if (batch_idx + 1) % 16 == 0:
        #         self.clip_gradients(optimizer_w, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        #         optimizer_w.step()
        #         optimizer_w.zero_grad()

        #         self.task_weights.data.copy_(self.task_weights / self.task_weights.sum())

        # self.manual_backward(loss)
        # if (batch_idx + 1) % 16 == 0:
        #     self.clip_gradients(optimizer_W, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        #     optimizer_W.step()
        #     optimizer_W.zero_grad()

        return loss
    
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        self.feature_extraction.eval()
        # self.multimodal_feature_extraction.eval()
        loss = self._step("Val", batch)
        # lambda_gradnorm = 0.1

        # if gradnorm_loss is not None:
        #     total_loss = loss + lambda_gradnorm * gradnorm_loss
        #     self.log('Val/gradnorm_loss', gradnorm_loss, on_step=True, on_epoch=False, prog_bar=True)
        # else:
        #     total_loss = loss
        return loss

    # def on_after_backward(self):
    #     with torch.no_grad():
    #         normed_weights = self.task_weights / self.task_weights.sum()
    #         self.task_weights.data.copy_(normed_weights.detach())

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
