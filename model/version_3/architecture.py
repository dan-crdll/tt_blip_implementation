from model.version_3.layers.feature_extraction import FeatureExtraction
from model.version_3.layers.cross_attention_block import CrossAttnBlock
from torch import nn 
import torch.nn.functional as F
import lightning as L


class Model(L.LightningModule):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()

        self.feature_extraction = FeatureExtraction(self.device)

        self.fusion_layer = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads, hidden_dim) 
            for i in range(6)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(5)
        )
    
    def forward(self, img, txt):
        (z_i, z), contrastive_loss = self.feature_extraction(img, txt)
        
        for layer in self.fusion_layer:
            z = layer(z, z_i)
        y = self.classifier(z)
        loss = contrastive_loss
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
