import torch
from transformers import CLIPModel, CLIPProcessor
from typing import List
from PIL import Image


class CLIPCLSExtractor:
    """Extract global semantic features from CLIP [CLS] token."""

    def __init__(self, model_id: str = "openai/clip-vit-large-patch14") -> None:
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()

    def __call__(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token
