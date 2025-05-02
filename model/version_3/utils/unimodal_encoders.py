from transformers import ViTModel, BertModel, ViTImageProcessorFast, BertTokenizerFast
from torch import nn

class ViT(nn.Module):
    def __init__(self, hf_repo, device='cpu', unfreeze_from_layer=0):
        super().__init__()
        self.device = device
        self.vit = ViTModel.from_pretrained(hf_repo)
        self.processor = ViTImageProcessorFast.from_pretrained(hf_repo)
        self.vit.to(device)

        # Freeze all layers, then unfreeze from specified encoder block
        for param in self.vit.parameters():
            param.requires_grad = False
        for idx, block in enumerate(self.vit.encoder.layer):
            if idx >= unfreeze_from_layer:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, image):
        inputs = self.processor(image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device, non_blocking=True)
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state  # (batch, seq_length, hidden_size)


class BERT(nn.Module):
    def __init__(self, hf_repo, device='cpu', unfreeze_from_layer=0, n_layers=6):
        super().__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(hf_repo)
        self.tokenizer = BertTokenizerFast.from_pretrained(hf_repo)
        self.bert.to(device)
        self.n_layers = n_layers

        for param in self.bert.parameters():
            param.requires_grad = False
        for idx, block in enumerate(self.bert.encoder.layer):
            if idx >= unfreeze_from_layer and idx < unfreeze_from_layer + n_layers:
                for param in block.parameters():
                    param.requires_grad = True


    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device, non_blocking=True)
        attention_mask = inputs['attention_mask'].to(self.device, non_blocking=True)
        z = self.bert.embeddings(input_ids)
        for layer in self.bert.encoder.layer[:self.n_layers]:
            z = layer(z, attention_mask.float())[0]
        return z