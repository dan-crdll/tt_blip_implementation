from transformers import ViTModel, ViTImageProcessor, DebertaV2Tokenizer, DebertaV2Model
from torch import nn

class ViT(nn.Module):
    def __init__(self, hf_repo, device='cpu', unfreeze_from_layer=0):
        super().__init__()
        self.device = device
        self.vit = ViTModel.from_pretrained(hf_repo, output_hidden_states=True)
        self.processor = ViTImageProcessor.from_pretrained(hf_repo)

        # Freeze all layers, then unfreeze from specified encoder block
        for param in self.vit.parameters():
            param.requires_grad = False
        for idx, block in enumerate(self.vit.encoder.layer):
            if idx >=10:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, image):
        inputs = self.processor(image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device, non_blocking=True)
        outputs = self.vit(pixel_values=pixel_values).hidden_states 
        
        # Take the average of the last three hidden states
        last_three = outputs[-3:]  # list of tensors
        avg_hidden = sum(last_three) / 3

        return avg_hidden, pixel_values   # (batch, seq_length, hidden_size)


class TextEncoder(nn.Module):
    def __init__(self, hf_repo, device='cpu', unfreeze_from_layer=0, n_layers=6):
        super().__init__()
        self.encoder = DebertaV2Model.from_pretrained(hf_repo, output_hidden_states=True)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(hf_repo, use_fast=False)
        self.n_layers = n_layers

        self.device = device

        for param in self.encoder.parameters():
           param.requires_grad = False
        for idx, block in enumerate(self.encoder.encoder.layer):
           if idx >= 10:
               for param in block.parameters():
                   param.requires_grad = True


    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(self.device, non_blocking=True)
        attention_mask = inputs['attention_mask'].to(self.device, non_blocking=True)
        z = self.encoder(input_ids=input_ids, attention_mask=attention_mask).hidden_states 

        last_three = z[-3:]  # list of tensors
        avg_hidden = sum(last_three) / 3

        return avg_hidden