from transformers import AutoProcessor, AutoModel, SiglipModel, SiglipProcessor, FlavaModel, FlavaProcessor
import torch
from torch import nn
from PIL import Image
import numpy as np

class SigClip(nn.Module):
    def __init__(self, hf_repo, device=None, frozen=True):
        super().__init__()
        # Set device automatically if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = AutoProcessor.from_pretrained(hf_repo)
        self.model = AutoModel.from_pretrained(hf_repo)

        if frozen:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.eval()

        # Adapter to ensure output dimension is consistent (768)
        # Siglip base hidden size is usually 768
        self.hidden_size = self.model.config.vision_config.hidden_size
        self.adapter = nn.Linear(self.hidden_size, 768)

    def forward(self, image=None, text=None, use_autocast=True):
        # Determine batch size
        batch_size = len(text) if text is not None else (len(image) if image is not None else 1)

        # Robust default values
        if image is None:
            image = [Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255) for _ in range(batch_size)]
        if text is None:
            text = [""] * batch_size

        # Preprocess
        processed = self.processor(text=text, images=image, return_tensors='pt', padding=True).to(self.device)

        # Forward pass
        outputs = self.model(**processed)
        
        # SigLIP provides vision and text outputs separately. 
        # For a "multimodal" state like BLIP-2 Q-former, we return vision features.
        # Shape: (batch_size, seq_len, hidden_size)
        z = outputs.vision_model_output.last_hidden_state
        
        # Apply adapter to ensure output dimension is 768
        z = self.adapter(z)

        return z

    def free_memory(self):
        """Free up CUDA memory after use."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class FlavaModelWrapper(nn.Module):
    def __init__(self, hf_repo="facebook/flava-full", device=None, frozen=True):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = FlavaProcessor.from_pretrained(hf_repo)
        self.model = FlavaModel.from_pretrained(hf_repo)

        if frozen:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.eval()

        # Adapter to ensure output dimension is consistent (768)
        self.hidden_size = self.model.config.hidden_config.hidden_size
        self.adapter = nn.Linear(self.hidden_size, 768)

    def forward(self, image=None, text=None, use_autocast=True):
        # Determine batch size
        batch_size = len(text) if text is not None else (len(image) if image is not None else 1)

        # Robust default values
        if image is None:
            image = [Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255) for _ in range(batch_size)]
        if text is None:
            text = [""] * batch_size

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # multimodal_output shape: (batch_size, seq_len, hidden_size)
        z = outputs.multimodal_output
        
        # Apply adapter to ensure output dimension is 768
        z = self.adapter(z)

        return z

    def free_memory(self):
        """Free up CUDA memory after use."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
