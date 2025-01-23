from torchmetrics import Accuracy
import lightning as L
from torch import nn 
import torch


class TT_Blip(L.LightningModule):
    def __init__(self, feature_extraction_layer, fusion_layer, clsf_layer):
        super().__init__()

        self.feature_extraction_layer = feature_extraction_layer
        self.fusion_layer = fusion_layer
        self.clsf_layer = clsf_layer

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy('binary')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-3, weight_decay=1e-5)
    
    def forward(self, vit_img, blip_img, blip_txt, blip_attn, bert_txt, bert_attn):
        zi, zt, zit = self.feature_extraction_layer(vit_img, blip_img, blip_txt, blip_attn, bert_txt, bert_attn)
        z = self.fusion_layer(zi, zt, zit)
        y = self.clsf_layer(z)
        return y 
    
    def training_step(self, batch):
        vi, bi, (bt, ba), (bet, bea), y = batch
        pred = self.forward(vi, bi, bt, ba, bet, bea)

        loss = self.loss_fn(pred, y)
        acc = self.accuracy(pred, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss 
    
    def validation_step(self, batch):
        vi, bi, (bt, ba), (bet, bea), y = batch
        pred = self.forward(vi, bi, bt, ba, bet, bea)

        loss = self.loss_fn(pred, y)
        acc = self.accuracy(pred, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss 