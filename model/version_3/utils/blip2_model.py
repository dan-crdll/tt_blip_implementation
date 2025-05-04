from transformers import CLIPProcessor, CLIPModel
import torch
from torch import nn
from PIL import Image
import numpy as np

class Blip2Model(nn.Module):
    def __init__(self, hf_repo, device='cuda', frozen=False):
        super().__init__()
        # Set device automatically if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = CLIPProcessor.from_pretrained(hf_repo)
        self.model = CLIPModel.from_pretrained(hf_repo)

        self.vis_projector = nn.Linear(768, 384)
        self.txt_projector = nn.Linear(512, 384)
        
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
        processed = self.processor(text, image, return_tensors='pt', padding=True)

        x_img = processed['pixel_values'].to(self.device)
        x_txt = processed['input_ids'].to(self.device)
        x_attn_mask = processed['attention_mask'].to(self.device)

        z = self.model(pixel_values=x_img, input_ids=x_txt, attention_mask=x_attn_mask)

        z_img = z.vision_model_output.last_hidden_state
        z_txt = z.text_model_output.last_hidden_state

        # Project features
        z_img = self.vis_projector(z_img)
        z_txt = self.txt_projector(z_txt)

        return z_img, z_txt

    def free_memory(self):
        """Free up CUDA memory after use."""
        del self.model
        torch.cuda.empty_cache()