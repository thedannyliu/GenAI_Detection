import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
from typing import List, Union, Optional, Dict
import torchvision.transforms as T

# Optional dependency: peft for LoRA fine-tuning
try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class CLIPCLSExtractor(nn.Module):
    """Extract CLS token from CLIP's vision transformer with optional LoRA."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", lora_cfg: Optional[Dict] = None) -> None:
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Inject LoRA adapters if requested
        self._lora_active = False
        if lora_cfg and lora_cfg.get("enable", False):
            if not _PEFT_AVAILABLE:
                raise RuntimeError("peft library is required for LoRA but not installed. pip install peft")
            # Build LoRAConfig with sensible defaults
            lora_config = LoraConfig(
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(lora_cfg.get("alpha", 16)),
                lora_dropout=float(lora_cfg.get("dropout", 0.1)),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            )
            self.model = get_peft_model(self.model, lora_config)
            self._lora_active = True

        # Determine if gradients are required (for conditional no_grad)
        self._requires_grad = any(p.requires_grad for p in self.model.parameters())
        
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
        pixel_values = inputs["pixel_values"] if isinstance(inputs, dict) else inputs.pixel_values

        # Run forward pass – use AMP autocast for faster inference on A100
        autocast_enabled = pixel_values.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=autocast_enabled):
            if self._lora_active:
                outputs = self.model.base_model(pixel_values=pixel_values)
            elif self._requires_grad:
                outputs = self.model(pixel_values=pixel_values)
            else:
                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values)
        # The last hidden state is the sequence of patch embeddings + CLS token
        # The CLS token is the first one in the sequence
        cls_token_embedding = outputs.last_hidden_state[:, 0, :].float()  # cast to fp32 for downstream MLPs
        return cls_token_embedding
