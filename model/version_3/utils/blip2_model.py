from transformers import ViltProcessor, ViltModel
import torch
from torch import nn
from PIL import Image
import numpy as np
import lightning as L

class Blip2Model(nn.Module):
    def __init__(self, hf_repo, device='cuda', frozen=False):
        super().__init__()
        # Set device automatically if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = ViltProcessor.from_pretrained(hf_repo)
        self.model = ViltModel.from_pretrained(hf_repo)

        self.projector = nn.Linear(768, 384)
        
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, image=None, text=None, use_autocast=True):
        # Determine batch size
        batch_size = len(text) if text is not None else len(image)
        no_text = text is None
        # Robust default values
        if image is None:
            image = [Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 255) for _ in range(batch_size)]
        if text is None:
            text = [""] * batch_size

        # Preprocess
        processed = self.processor(text=text, images=image, return_tensors='pt', padding=True)

        x_img = processed['pixel_values'].to(self.device)
        x_txt = processed['input_ids'].to(self.device)
        x_attn_mask = processed['attention_mask'].to(self.device)
        token_type_ids = processed['token_type_ids'].to(self.device)
        pixel_mask = processed['pixel_mask'].to(self.device)
        z = self.model(
            pixel_values=x_img, 
            input_ids=x_txt, 
            attention_mask=x_attn_mask,
            token_type_ids=token_type_ids,
            pixel_mask=pixel_mask
          )

        z = z.last_hidden_state

        # Project features
        z = self.projector(z)

        return z

    def free_memory(self):
        """Free up CUDA memory after use."""
        del self.model
        torch.cuda.empty_cache()