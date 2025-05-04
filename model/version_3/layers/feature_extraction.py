from model.version_3.utils.blip2_model import Blip2Model
from model.version_3.utils.unimodal_encoders import ViT, TextEncoder
from model.version_3.utils.loss_fn import ITMLoss
import torch 
from torch import nn 
import copy


class ImageFeatureExtraction(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.vit = ViT("WinKawaks/vit-tiny-patch16-224", device, unfreeze_from_layer=9)
        self.blip = Blip2Model("dandelin/vilt-b32-mlm", device, frozen=False)

    def forward(self, x):
        z_vit = self.vit(x)

        z_blip = self.blip(image=x)

        cls = ((z_vit[:, 0] + z_blip[:, 0]) / 2).unsqueeze(1)
        tokens = torch.cat([z_vit[:, 1:], z_blip[:, 1:]], dim=1)

        z = torch.cat([cls, tokens], dim=1)
        return z, (z_vit, z_blip[:, 0])


class TextFeatureExtraction(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.text_encoder = TextEncoder("albert/albert-base-v2", device=device, unfreeze_from_layer=3)
        self.blip = Blip2Model("dandelin/vilt-b32-mlm", device, frozen=False)

    def forward(self, x):
        z_text = self.text_encoder(x)
        z_blip = self.blip(text=x)

        cls = ((z_text[:, 0] + z_blip[:, 0]) / 2).unsqueeze(1)
        tokens = torch.cat([z_text[:, 1:], z_blip[:, 1:]], dim=1)
        z = torch.cat([cls, tokens], dim=1)
        return z, (z_text, z_blip[:, 0])
    

class FeatureExtraction(nn.Module):
    def __init__(self, device='cuda', temp=1.0):
        super().__init__()

        self.feature_extractor_img = ImageFeatureExtraction(device) 
        self.feature_extractor_txt = TextFeatureExtraction(device)

        self.itm_loss = ITMLoss(
            temp=temp, 
            momentum=0.999, 
            image_encoder=copy.deepcopy(self.feature_extractor_img.vit),
            text_encoder=copy.deepcopy(self.feature_extractor_txt.text_encoder)
        )

    def forward(self, img, txt, orig, labels):
        z_i, (z_vit, cls_blip_i) = self.feature_extractor_img(img)
        z_t, (z_bert, cls_blip_t) = self.feature_extractor_txt(txt)

        l_itm = self.itm_loss(
            z_i[:, 0], 
            z_t[:, 0], 
            img, 
            txt, 
            self.feature_extractor_img.vit.parameters(), 
            self.feature_extractor_txt.text_encoder.parameters(),
            orig, 
            labels
        )

        return (z_i, z_t), (z_vit, z_bert), l_itm
