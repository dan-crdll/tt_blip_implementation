from model.version_3.layers.feature_extraction import FeatureExtraction
from model.version_3.layers.cross_attention_block import CrossAttnBlock
from model.version_3.utils.loss_fn import MocoLoss
from torch import nn 
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAUROC, MultilabelF1Score, MultilabelAveragePrecision
import torch.nn.functional as F
import torch
import lightning as L
import copy


class Model(L.LightningModule):
    def __init__(self, embed_dim, num_heads, hidden_dim, temp=1.0, momentum=0.9, queue_size=32):
        super().__init__()

        self.feature_extraction = FeatureExtraction('cuda', temp)

        self.fusion_layer = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads, hidden_dim) 
            for i in range(6)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 5)
        )

        self.moco_loss = MocoLoss(copy.deepcopy(self.feature_extraction), momentum=momentum, queue_size=queue_size, temp=temp)

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
    
    def forward(self, img, txt):
        (z_i, z_t), contrastive_loss = self.feature_extraction(img, txt)
        loss = contrastive_loss + self.moco_loss((z_i[:, 0], z_t[:, 0]), (img, txt), self.feature_extraction.parameters())
        loss /= 2

        z = z_t
        for layer in self.fusion_layer:
            z = layer(z, z_i)
        z = z[:, 0]
        y = self.classifier(z)

        return y, loss
    
    def step(self, split, batch):
        images, texts, (y_bin, y_multi) = batch 

        pred, c_loss = self.forward(images, texts)
        pred_bin = pred[:, 0]

        bin_loss = self.loss_fn(pred_bin, y_bin.float())
        mask = (nn.functional.sigmoid(pred_bin) < 0.5)

        pred_multi = pred[:, 1:]
        if mask.sum() > 0:
            multi_loss = self.loss_fn(pred_multi[mask], y_multi[mask].float())
            cls_loss = bin_loss + multi_loss
        else:
            cls_loss = bin_loss
        # loss = 0.2 * c_loss + 0.4 * multi_loss + 0.4 * bin_loss

        loss = c_loss + cls_loss

        # -- BINARY CLASSIFICATION --
        pred_bin = nn.functional.sigmoid(pred_bin)
        acc_bin = self.acc_fn_bin(pred_bin, y_bin.float())
        f1 = self.f1_fn(pred_bin, y_bin.float())
        auc = self.auc_fn(pred_bin, y_bin.float())

        self.log_dict(
            {
                f'{split}/loss_bin': bin_loss.detach().item(),
                f'{split}/acc_bin': acc_bin.detach().item(),
                f'{split}/contrastive_loss': c_loss.detach().item()
            }, prog_bar=True, on_epoch=True, on_step=True if split == 'Train' else False
        )
        
        self.log_dict(
            {
                f'{split}/f1_bin': f1.detach().item(),
                f'{split}/auc_bin': auc.detach().item()
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
                    f'{split}/cf1_multi': cf1.detach().item(),
                    f'{split}/of1_multi': of1.detach().item(),
                    f'{split}/mAP_multi': mAP.detach().item(),
                    f'{split}/acc_multi': acc_multi.detach().item()
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
