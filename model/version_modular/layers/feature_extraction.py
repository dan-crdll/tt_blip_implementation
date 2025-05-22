from model.version_modular.utils.unimodal_encoders import ViT, TextEncoder
from model.version_modular.utils.loss_fn import ITMLoss
import torch 
from torch import nn 
import copy


class ImageFeatureExtraction(nn.Module):
    def __init__(self, hf_repo, unfreeze_from_layer, device='cuda', large=False):
        super().__init__()
        self.vit = ViT(hf_repo, device, unfreeze_from_layer)
        self.large = large 
        if large:
            self.proj = nn.Linear(1024, 768)

    def forward(self, x):
        z_vit = self.vit(x)
        if self.large:
            z_vit = self.proj(z_vit)

        return z_vit


class TextFeatureExtraction(nn.Module):
    def __init__(self, hf_repo, unfreeze_from_layer, device='cuda'):
        super().__init__()
        self.text_encoder = TextEncoder(hf_repo, device=device, unfreeze_from_layer=unfreeze_from_layer)

    def forward(self, x):
        z_text = self.text_encoder(x)

        return z_text
    

class FeatureExtraction(nn.Module):
    def __init__(self, hf_repo_vit, hf_repo_txt, unfrozen_vit, unfrozen_txt, large_vit, device='cuda', temp=0.07, queue_size=1024, momentum=0.999):
        super().__init__()

        self.feature_extractor_img = ImageFeatureExtraction(hf_repo_vit, unfrozen_vit, device, large_vit) 
        self.feature_extractor_txt = TextFeatureExtraction(hf_repo_txt, unfrozen_txt, device)

        self.itm_loss = ITMLoss(
            temp=temp, 
            momentum=momentum, 
            image_encoder=copy.deepcopy(self.feature_extractor_img),
            text_encoder=copy.deepcopy(self.feature_extractor_txt.text_encoder),
            queue_size=queue_size
        )

    def forward(self, img, txt, orig, labels, split):
        z_i = self.feature_extractor_img(img)
        z_t = self.feature_extractor_txt(txt)

        if split == 'Train':
            l_itm = self.itm_loss(
                z_i[:, 0], 
                z_t[:, 0], 
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



def create_feature_extraction():
    hf_repo_vit = ['google/vit-base-patch16-224', 'google/vit-large-patch16-224']
    hf_repo_txt = 'microsoft/deberta-v3-base'
    large_vit = False
    unfreeze_from_layer_vit = 0
    unfreeze_from_layer_txt = 0
    temp = 0.07
    queue_size = 1024
    momentum = 0.999

    print("##### FEATURE EXTRACTION LAYER CONFIGURATION #####")

    vit_choice = int(input("Use ViT Base (0) or ViT Large (1): "))
    hf_repo_vit = hf_repo_vit[vit_choice]
    if vit_choice == 1:
        large_vit=True

    unfreeze_from_layer_vit = int(input("Unfreeze from layer ViT (0-11): "))
    unfreeze_from_layer_txt = int(input("Unfreeze from layer DeBERTa (0-11): "))

    temp = float(input("Temperature for MAC Loss (0.04 - 0.07): "))
    queue_size = int(input("Queue size for MAC Loss (1024 - 4096): "))
    momentum = float(input("Momentum for MAC Loss encoders (0.99 - 0.999): "))

    return FeatureExtraction(
        hf_repo_vit=hf_repo_vit,
        hf_repo_txt=hf_repo_txt,
        unfrozen_vit=unfreeze_from_layer_vit,
        unfrozen_txt=unfreeze_from_layer_txt,
        large_vit=large_vit,
        temp=temp,
        queue_size=queue_size,
        momentum=momentum
    )