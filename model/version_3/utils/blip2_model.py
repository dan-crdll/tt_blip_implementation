from transformers import Blip2Processor, Blip2QFormerConfig, Blip2QFormerModel, ViTModel, ViTImageProcessor, AddedToken
import torch
from torch import nn
from PIL import Image
import numpy as np

class Blip2Model(nn.Module):
    def __init__(self, hf_repo, device=None, frozen=False):
        super().__init__()
        # Set device automatically if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load processor and model to the correct device
        self.processor = Blip2Processor.from_pretrained(hf_repo)
        config = Blip2QFormerConfig.from_pretrained(hf_repo)
        config.encoder_hidden_size = 768

        self.embedding = nn.Embedding(config.vocab_size, 768, padding_idx=0)
        self.pos_embedding = nn.Embedding(512, 768)
        
        self.model = Blip2QFormerModel(config)

        image_token = AddedToken("<image>", normalized=False, special=True)
        self.processor.tokenizer.add_tokens([image_token], special_tokens=True)

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

        for param in self.model.parameters():
            param.requires_grad_(True)

        self.vit.eval()
        for param in self.vit.parameters():
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
        processed = self.processor(image, text, return_tensors='pt', padding=True)
        processed_image = self.vit_processor(image, return_tensors='pt')
        x_img = processed_image['pixel_values'].to(self.device, non_blocking=True)
        x_txt = processed['input_ids'].to(self.device, non_blocking=True)
        x_attn_mask = processed['attention_mask'].to(self.device, non_blocking=True)

        z_img = self.vit(x_img)['last_hidden_state']

        # Image attention mask
        encoder_attn_mask = torch.ones((z_img.shape[0], z_img.shape[1]), device=z_img.device) \
            if image is not None else torch.zeros((z_img.shape[0], z_img.shape[1]), device=z_img.device)

        # Text embedding logic
        if no_text:
            z_txt = torch.zeros((z_img.shape[0], z_img.shape[1], 768), device=z_img.device)
            x_attn_mask = torch.zeros((z_img.shape[0], z_img.shape[1]), device=z_img.device)
        else:
            z_txt = self.embedding(x_txt)

        # QFormer
        z = self.model(
            encoder_hidden_states=z_img,
            query_embeds=z_txt,
            attention_mask=x_attn_mask,
            encoder_attention_mask=encoder_attn_mask
        )

        return z.last_hidden_state

    def free_memory(self):
        """Free up CUDA memory after use."""
        del self.model
        torch.cuda.empty_cache()