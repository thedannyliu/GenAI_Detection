import torch
import torch.nn as nn
from typing import List, Optional
import torchvision.transforms as T

from .frequency_extractors import FrequencyFeatureExtractor
from .clip_semantics import CLIPCLSExtractor


class FrequencyOnlyDetector(nn.Module):
    """Classifier using ONLY frequency fingerprints."""

    def __init__(self, hidden_dim: int = 256, num_classes: int = 2, freq_methods: Optional[List[str]] = None):
        super().__init__()
        self.freq_extractor = FrequencyFeatureExtractor(methods=freq_methods)
        freq_dim = self.freq_extractor.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Expecting images as tensor batch on some device
        freq_feats = [self.freq_extractor(img.cpu()) for img in images]
        freq_feat = torch.stack(freq_feats, dim=0).to(images.device)
        logits = self.classifier(freq_feat)
        return logits


class SemanticOnlyDetector(nn.Module):
    """Classifier using ONLY CLIP CLS semantics."""

    def __init__(self, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.sem_extractor = CLIPCLSExtractor()
        sem_dim = self.sem_extractor.hidden_dim  # 1024 for ViT-L/14
        self.classifier = nn.Sequential(
            nn.Linear(sem_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sem_feat = self.sem_extractor(images)  # already on same device as model parameters
        logits = self.classifier(sem_feat)
        return logits 