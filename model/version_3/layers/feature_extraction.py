from model.version_3.utils.blip2_model import Blip2Model
from model.version_3.utils.unimodal_encoders import ViT, BERT
from model.version_3.utils.loss_fn import InfoNCE
import torch 
from torch import nn 


class ImageFeatureExtraction(nn.Module):
    def __init__(self, device='cpu'):
        self.vit = ViT("google/vit-base-patch16-224", device, unfreeze_from_layer=9)
        self.blip = Blip2Model("Salesforce/blip2-itm-vit-g", device)

    def forward(self, x):
        z_vit = self.vit(x)
        z_blip = self.blip(image=x)

        cls = ((z_vit[:, 0] + z_blip[:, 0]) / 2).unsqueeze(1)
        tokens = torch.cat([z_vit[:, 1:], z_blip[:, 1:]], dim=1)

        z = torch.cat([cls, tokens], dim=1)
        return z, (z_vit[:, 0], z_blip[:, 0])


class TextFeatureExtraction(nn.Module):
    def __init__(self, device='cpu'):
        self.bert = BERT("google-bert/bert-base-uncased", device=device, unfreeze_from_layer=9)
        self.blip = Blip2Model("Salesforce/blip2-itm-vit-g", device)

    def forward(self, x):
        z_bert = self.bert(x)
        z_blip = self.blip(text=x)

        cls = ((z_bert[:, 0] + z_blip[:, 0]) / 2).unsqueeze(1)
        tokens = torch.cat([z_bert[:, 1:], z_blip[:, 1:]], dim=1)
        z = torch.cat([cls, tokens], dim=1)
        return z, (z_bert[:, 0], z_blip[:, 0])
    

class FeatureExtraction(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.feature_extractor_img = ImageFeatureExtraction(device) 
        self.feature_extractor_txt = TextFeatureExtraction(device)

        self.infonce_loss = InfoNCE(temp=1.0)

    def forward(self, img, txt):
        z_i, (cls_vit, cls_blip_i) = self.feature_extractor_img(img)
        z_t, (cls_bert, cls_blip_t) = self.feature_extractor_txt(txt)

        l_i2t = self.infonce_loss(cls_blip_i, cls_bert)
        l_t2i = self.infonce_loss(cls_blip_t, cls_vit)

        l_itm = (self.infonce_loss(z_i[:, 0], z_t[:, 0]) + self.infonce_loss(z_t[:, 0], z_i[:, 0])) / 2.0

        return (z_i, z_t), ((l_i2t + l_t2i) / 2.0 + l_itm) / 2.0
