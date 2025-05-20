from model.version_3.utils.blip2_model import Blip2Model
from model.version_3.utils.unimodal_encoders import ViT, TextEncoder
from model.version_3.utils.loss_fn import ITMLoss
import torch 
from torch import nn 
import copy


class ImageFeatureExtraction(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.vit = ViT("google/vit-base-patch16-224", device, unfreeze_from_layer=6)
        # self.proj = nn.Linear(1024, 768)
        # self.blip = Blip2Model("dandelin/vilt-b32-mlm", device, frozen=False)

    def forward(self, x):
        z_vit = self.vit(x)
        # z_vit = self.proj(z_vit[-1])
        # z_blip = self.blip(image=x)

        # z = torch.cat([z_vit, z_blip], dim=1)

        # return z, (z_vit, z_blip[:, 0])
        return z_vit


class TextFeatureExtraction(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.text_encoder = TextEncoder('microsoft/deberta-v3-base', device=device, unfreeze_from_layer=6)
        # self.blip = Blip2Model("dandelin/vilt-b32-mlm", device, frozen=False)

    def forward(self, x):
        z_text = self.text_encoder(x)
        # z_blip = self.blip(text=x)

        # z = torch.cat([z_text, z_blip], dim=1)
        # return z, (z_text, z_blip[:, 0])
        return z_text
    

class FeatureExtraction(nn.Module):
    def __init__(self, device='cuda', temp=1.0, queue_size=128, momentum=0.999):
        super().__init__()

        self.feature_extractor_img = ImageFeatureExtraction(device) 
        self.feature_extractor_txt = TextFeatureExtraction(device)

        self.itm_loss = ITMLoss(
            temp=temp, 
            momentum=momentum, 
            image_encoder=copy.deepcopy(self.feature_extractor_img),
            text_encoder=copy.deepcopy(self.feature_extractor_txt.text_encoder),
            queue_size=queue_size
        )

    def forward(self, img, txt, orig, labels, split):
        # z_i, (z_vit, cls_blip_i) = self.feature_extractor_img(img)
        # z_t, (z_bert, cls_blip_t) = self.feature_extractor_txt(txt)

        z_i = self.feature_extractor_img(img)
        z_t = self.feature_extractor_txt(txt)

        if split == 'Train':
            l_itm = self.itm_loss(
                z_i[-1][:, 0], 
                z_t[-1][:, 0], 
                img, 
                txt, 
                self.feature_extractor_img.parameters(), 
                self.feature_extractor_txt.text_encoder.parameters(),
                orig, 
                labels
            )
        else:
            l_itm = 0.0

        return (z_i, z_t), l_itm
        # return (z_i, z_t)


