from typing import Optional, List, Dict, Any, Union, Tuple
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval
import numpy as np


class CLIPModelWrapper(nn.Module):
    """
    Wrapper for CLIP model for AI-generated image detection.
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        mode: str = "zero-shot",
        num_classes: int = 2
    ):
        """
        Initialize model.
        Args:
            model_name (str): Name of CLIP model to use
            mode (str): 'zero-shot' or 'fine-tune'
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.mode = mode
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        
        if mode == "fine-tune":
            # Add classification head
            self.classifier = nn.Linear(self.model.projection_dim, num_classes)
        else:
            self.classifier = None

    def forward(
        self,
        images: torch.Tensor,
        text_prompts: Optional[List[str]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Args:
            images (torch.Tensor): Input images
            text_prompts (Optional[List[str]]): Text prompts for zero-shot
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Model outputs
        """
        if self.mode == "zero-shot" and text_prompts is not None:
            # Zero-shot classification
            image_features = self.model.get_image_features(images)
            text_features = self.model.get_text_features(
                self.processor(text_prompts, return_tensors="pt", padding=True)["input_ids"].to(images.device)
            )
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            return similarity
        else:
            # Fine-tuned classification
            image_features = self.model.get_image_features(images)
            return self.classifier(image_features)


class BLIPModelWrapper(nn.Module):
    """
    Wrapper for BLIP model for AI-generated image detection.
    """
    def __init__(
        self,
        model_name: str = "Salesforce/blip-itm-base-coco",
        mode: str = "zero-shot",
        num_classes: int = 2
    ):
        """
        Initialize model.
        Args:
            model_name (str): Name of BLIP model to use
            mode (str): 'zero-shot' or 'fine-tune'
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.mode = mode
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name)
        
        if mode == "fine-tune":
            # Add classification head
            self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        else:
            self.classifier = None

    def forward(
        self,
        images: torch.Tensor,
        text_prompts: Optional[List[str]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Args:
            images (torch.Tensor): Input images
            text_prompts (Optional[List[str]]): Text prompts for zero-shot
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Model outputs
        """
        if self.mode == "zero-shot" and text_prompts is not None:
            # Zero-shot classification
            inputs = self.processor(
                images=images,
                text=text_prompts,
                return_tensors="pt",
                padding=True
            ).to(images.device)
            
            outputs = self.model(**inputs)
            similarity = outputs.itm_score.softmax(dim=-1)
            return similarity
        else:
            # Fine-tuned classification
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(images.device)
            
            outputs = self.model(**inputs)
            return self.classifier(outputs.last_hidden_state[:, 0, :])


class PromptTuningWrapper(nn.Module):
    """
    Wrapper for prompt tuning with CLIP.
    """
    def __init__(
        self,
        base_model: CLIPModelWrapper,
        num_prompt_tokens: int = 16,
        prompt_dim: int = 512
    ):
        """
        Initialize model.
        Args:
            base_model (CLIPModelWrapper): Base CLIP model
            num_prompt_tokens (int): Number of prompt tokens
            prompt_dim (int): Dimension of prompt embeddings
        """
        super().__init__()
        self.base_model = base_model
        self.num_prompt_tokens = num_prompt_tokens
        
        # Initialize learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_prompt_tokens, prompt_dim)
        )
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        text_prompts: List[str]
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            images (torch.Tensor): Input images
            text_prompts (List[str]): Text prompts
        Returns:
            torch.Tensor: Model outputs
        """
        # Get text embeddings
        text_inputs = self.base_model.processor(
            text_prompts,
            return_tensors="pt",
            padding=True
        ).to(images.device)
        
        # Add prompt embeddings
        text_embeddings = self.base_model.model.get_text_features(
            text_inputs["input_ids"]
        )
        text_embeddings = torch.cat([
            self.prompt_embeddings.expand(text_embeddings.size(0), -1, -1),
            text_embeddings
        ], dim=1)
        
        # Get image embeddings
        image_embeddings = self.base_model.model.get_image_features(images)
        
        # Compute similarity
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_embeddings @ text_embeddings.transpose(-2, -1)).softmax(dim=-1)
        return similarity 