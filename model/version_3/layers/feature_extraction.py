from model.version_3.utils.blip2_model import Blip2Model
from model.version_3.utils.unimodal_encoders import ViT, BERT
from model.version_3.utils.loss_fn import InfoNCE
import torch 
from torch import nn 


class ImageFeatureExtraction(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.vit = ViT("WinKawaks/vit-small-patch16-224", device, unfreeze_from_layer=9)
        self.blip = Blip2Model("Salesforce/blip2-itm-vit-g", device)
        self.vit_projector = nn.Linear(384, 768)

    def forward(self, x):
        z_vit = self.vit(x)
        z_vit = self.vit_projector(z_vit)
        
        z_blip = self.blip(image=x)

        cls = ((z_vit[:, 0] + z_blip[:, 0]) / 2).unsqueeze(1)
        tokens = torch.cat([z_vit[:, 1:], z_blip[:, 1:]], dim=1)

        z = torch.cat([cls, tokens], dim=1)
        return z, (z_vit[:, 0], z_blip[:, 0])


class TextFeatureExtraction(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.bert = BERT("distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device, unfreeze_from_layer=9)
        self.blip = Blip2Model("Salesforce/blip2-itm-vit-g", device)

    def forward(self, x):
        z_bert = self.bert(x)
        z_blip = self.blip(text=x)

        cls = ((z_bert[:, 0] + z_blip[:, 0]) / 2).unsqueeze(1)
        tokens = torch.cat([z_bert[:, 1:], z_blip[:, 1:]], dim=1)
        z = torch.cat([cls, tokens], dim=1)
        return z, (z_bert[:, 0], z_blip[:, 0])
    

class FeatureExtraction(nn.Module):
    def __init__(self, device='cuda', temp=1.0):
        super().__init__()

        self.feature_extractor_img = ImageFeatureExtraction(device) 
        self.feature_extractor_txt = TextFeatureExtraction(device)

        self.infonce_loss = InfoNCE(temp=temp)

    def forward(self, img, txt):
        z_i, (cls_vit, cls_blip_i) = self.feature_extractor_img(img)
        z_t, (cls_bert, cls_blip_t) = self.feature_extractor_txt(txt)

        l_i2t = self.infonce_loss(cls_blip_i, cls_bert)
        l_t2i = self.infonce_loss(cls_blip_t, cls_vit)

        return (z_i, z_t), (l_i2t + l_t2i) / 2.0
