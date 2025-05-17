import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple

class ResNet50Classifier(nn.Module):
    """
    ResNet-50 based classifier for AI-generated image detection.
    """
    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_backbone: bool = False
    ):
        """
        Initialize model.
        Args:
            pretrained (bool): Whether to use pretrained weights
            num_classes (int): Number of output classes
            freeze_backbone (bool): Whether to freeze backbone layers
        """
        super().__init__()
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, height, width]
        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes]
        """
        return self.backbone(x)

class ViTClassifier(nn.Module):
    """
    Vision Transformer based classifier for AI-generated image detection.
    """
    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_backbone: bool = False
    ):
        """
        Initialize model.
        Args:
            pretrained (bool): Whether to use pretrained weights
            num_classes (int): Number of output classes
            freeze_backbone (bool): Whether to freeze backbone layers
        """
        super().__init__()
        # Load pretrained ViT
        self.backbone = models.vit_b_16(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final layer
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, height, width]
        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes]
        """
        return self.backbone(x)

class LogisticRegressionOnFeatures(nn.Module):
    """
    Simple logistic regression on pre-extracted features.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2
    ):
        """
        Initialize model.
        Args:
            input_dim (int): Input feature dimension
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input features of shape [batch_size, input_dim]
        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes]
        """
        return self.classifier(x) 