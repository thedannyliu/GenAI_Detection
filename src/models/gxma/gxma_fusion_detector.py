import torch
import torch.nn as nn
from typing import List
from PIL import Image
import torchvision.transforms as T

from .frequency_extractors import FrequencyFeatureExtractor
from .clip_semantics import CLIPCLSExtractor


class CrossAttentionFusion(nn.Module):
    """Fuse frequency and semantic features via cross-attention."""

    def __init__(self, freq_dim: int, sem_dim: int, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.query_proj = nn.Linear(sem_dim, hidden_dim)
        self.key_proj = nn.Linear(freq_dim, hidden_dim)
        self.value_proj = nn.Linear(freq_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, freq_feat: torch.Tensor, sem_feat: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(sem_feat).unsqueeze(1)
        k = self.key_proj(freq_feat).unsqueeze(1)
        v = self.value_proj(freq_feat).unsqueeze(1)
        fused, _ = self.attn(q, k, v)
        return fused.squeeze(1)


class GXMAFusionDetector(nn.Module):
    """PoC detector combining frequency fingerprints and CLIP semantics."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.freq_extractor = FrequencyFeatureExtractor()
        self.sem_extractor = CLIPCLSExtractor()
        # frequency vector length: 128 + 64 + 64 = 256
        freq_dim = 256
        # CLIP ViT-L/14 has hidden size 768 for the vision encoder
        sem_dim = 768
        self.fusion = CrossAttentionFusion(freq_dim, sem_dim, hidden_dim, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        sem_feat = self.sem_extractor(images)
        to_tensor = T.ToTensor()
        freq_feats = [self.freq_extractor(to_tensor(img)) for img in images]
        freq_feat = torch.stack(freq_feats, dim=0)
        if torch.cuda.is_available():
            freq_feat = freq_feat.to(sem_feat.device)
        fused = self.fusion(freq_feat, sem_feat)
        logits = self.classifier(fused)
        return logits
