import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
from typing import List, Union
import torchvision.transforms as T


class CLIPCLSExtractor(nn.Module):
    """Extract CLS token from CLIP's vision transformer."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14") -> None:
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        # Expose hidden dimension for downstream modules
        self.hidden_dim = self.model.config.hidden_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of image tensors."""
        # The CLIP processor is designed to work with PIL Images and handles its own transformations.
        # We convert our input tensors back to PIL Images for the processor.
        # do_rescale=False is important because our tensors are already in [0, 1].
        to_pil = T.ToPILImage()
        pil_images = [to_pil(img.cpu()) for img in images]

        # The device for the inputs should match the model's device.
        device = next(self.model.parameters()).device
        inputs = self.processor(images=pil_images, return_tensors="pt", do_rescale=False).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        # The last hidden state is the sequence of patch embeddings + CLS token
        # The CLS token is the first one in the sequence
        cls_token_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_token_embedding
