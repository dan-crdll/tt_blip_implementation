from transformers import ViTModel, AutoModelForMaskedLM, ViTImageProcessor, AutoTokenizer
from torch import nn

class ViT(nn.Module):
    def __init__(self, hf_repo, device='cpu', unfreeze_from_layer=0):
        super().__init__()
        self.device = device
        self.vit = ViTModel.from_pretrained(hf_repo)
        self.processor = ViTImageProcessor.from_pretrained(hf_repo)

        self.projector = nn.Linear(192, 384)

        # Freeze all layers, then unfreeze from specified encoder block
        # for param in self.vit.parameters():
        #     param.requires_grad = False
        # for idx, block in enumerate(self.vit.encoder.layer):
        #     if idx >= unfreeze_from_layer:
        #         for param in block.parameters():
        #             param.requires_grad = True

    def forward(self, image):
        inputs = self.processor(image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device, non_blocking=True)
        outputs = self.vit(pixel_values=pixel_values).last_hidden_state
        outputs = self.projector(outputs)
        return outputs   # (batch, seq_length, hidden_size)


class TextEncoder(nn.Module):
    def __init__(self, hf_repo, device='cpu', unfreeze_from_layer=0, n_layers=6):
        super().__init__()
        self.encoder = AutoModelForMaskedLM.from_pretrained(hf_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo)
        self.n_layers = n_layers
        self.projector = nn.Linear(768, 384)

        self.device = device

        # for param in self.encoder.parameters():
        #    param.requires_grad = False
        # # for idx, block in enumerate(self.encoder.albert.encoder):
        # #    if idx >= unfreeze_from_layer:
        # #        for param in block.parameters():
        # #            param.requires_grad = True


    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device, non_blocking=True)
        attention_mask = inputs['attention_mask'].to(self.device, non_blocking=True)
        z = self.encoder.albert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        z = self.projector(z)
        return z