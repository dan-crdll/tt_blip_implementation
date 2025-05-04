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
        self.blip = Blip2Model("openai/clip-vit-base-patch32", device)

    def forward(self, x):
        z_vit = self.vit(x)

        z_blip, _ = self.blip(image=x, text=["altered image" for _ in range(z_vit.shape[0])])

        cls = ((z_vit[:, 0] + z_blip[:, 0]) / 2).unsqueeze(1)
        tokens = torch.cat([z_vit[:, 1:], z_blip[:, 1:]], dim=1)

        z = torch.cat([cls, tokens], dim=1)
        return z, (z_vit[:, 0], z_blip[:, 0])


class TextFeatureExtraction(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.text_encoder = TextEncoder("albert/albert-base-v2", device=device, unfreeze_from_layer=3)
        self.blip = Blip2Model("openai/clip-vit-base-patch32", device)

    def forward(self, x):
        z_text = self.text_encoder(x)
        _, z_blip = self.blip(text=x)

        cls = ((z_text[:, 0] + z_blip[:, 0]) / 2).unsqueeze(1)
        tokens = torch.cat([z_text[:, 1:], z_blip[:, 1:]], dim=1)
        z = torch.cat([cls, tokens], dim=1)
        return z, (z_text[:, 0], z_blip[:, 0])
    

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

    def forward(self, img, txt):
        z_i, (cls_vit, cls_blip_i) = self.feature_extractor_img(img)
        z_t, (cls_bert, cls_blip_t) = self.feature_extractor_txt(txt)

        l_itm = self.itm_loss(
            z_i[:, 0], 
            z_t[:, 0], 
            img, 
            txt, 
            self.feature_extractor_img.vit.parameters(), 
            self.feature_extractor_txt.text_encoder.parameters()
        )

        return (z_i, z_t), l_itm
