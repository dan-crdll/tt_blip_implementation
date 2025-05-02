from transformers import Blip2Processor, Blip2ForImageTextRetrieval, AddedToken
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
        self.model = Blip2ForImageTextRetrieval.from_pretrained(hf_repo)
        self.model.to(self.device)

        # Image token setup
        self.processor.num_query_tokens = self.model.config.num_query_tokens
        image_token = AddedToken("<image>", normalized=False, special=True)
        self.processor.tokenizer.add_tokens([image_token], special_tokens=True)

        self.model.resize_token_embeddings(len(self.processor.tokenizer), pad_to_multiple_of=64)
        self.model.config.image_token_index = len(self.processor.tokenizer) - 1

        # Freeze backbone, train adapters/Qformer if desired
        for param in self.model.parameters():
            param.requires_grad = False
        if not frozen:
            for param in self.model.qformer.parameters():
                param.requires_grad = True

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
        x_img = processed['pixel_values'].to(self.device, non_blocking=True)
        x_txt = processed['input_ids'].to(self.device, non_blocking=True)
        x_attn_mask = processed['attention_mask'].to(self.device, non_blocking=True)

        with torch.amp.autocast(enabled=use_autocast and self.device.startswith("cuda"), device_type=self.device):
            z_img = self.model.vision_model(x_img)['last_hidden_state']

            # Image attention mask
            encoder_attn_mask = torch.ones((z_img.shape[0], z_img.shape[1]), device=z_img.device) \
                if image is not None else torch.zeros((z_img.shape[0], z_img.shape[1]), device=z_img.device)

            # Text embedding logic
            if no_text:
                z_txt = torch.zeros((z_img.shape[0], z_img.shape[1], 768), device=z_img.device)
                x_attn_mask = torch.zeros((z_img.shape[0], z_img.shape[1]), device=z_img.device)
            else:
                z_txt = self.model.embeddings(x_txt, x_attn_mask)

            # QFormer
            z = self.model.qformer(
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