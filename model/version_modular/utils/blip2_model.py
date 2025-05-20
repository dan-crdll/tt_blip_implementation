from transformers import BlipProcessor, BlipForImageTextRetrieval
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

        self.processor = BlipProcessor.from_pretrained(hf_repo)
        self.model = BlipForImageTextRetrieval.from_pretrained(hf_repo)

        for param in self.model.parameters():
            param.requires_grad_(False)
        self.eval()

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

        z = self.model(
            pixel_values=x_img,
            input_ids=x_txt,
            attention_mask=x_attn_mask,
          )

        return z.last_hidden_state

    def free_memory(self):
        """Free up CUDA memory after use."""
        del self.model
        torch.cuda.empty_cache()