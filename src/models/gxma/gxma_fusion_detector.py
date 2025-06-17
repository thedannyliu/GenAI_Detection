import torch
import torch.nn as nn
from typing import List
from PIL import Image
import torchvision.transforms as T

from .frequency_extractors import (
    FrequencyFeatureExtractor,
    FrequencyFeatureExtractorSplit,
)
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


class CrossAttentionFusionParallel(nn.Module):
    """Parallel attention streams for each frequency expert."""

    def __init__(self, freq_dims: List[int], sem_dim: int, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.query_proj = nn.Linear(sem_dim, hidden_dim)

        self.key_proj_radial = nn.Linear(freq_dims[0], hidden_dim)
        self.value_proj_radial = nn.Linear(freq_dims[0], hidden_dim)
        self.key_proj_dct = nn.Linear(freq_dims[1], hidden_dim)
        self.value_proj_dct = nn.Linear(freq_dims[1], hidden_dim)
        self.key_proj_wavelet = nn.Linear(freq_dims[2], hidden_dim)
        self.value_proj_wavelet = nn.Linear(freq_dims[2], hidden_dim)

        self.attn_radial = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_dct = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_wavelet = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, freq_feats: List[torch.Tensor], sem_feat: torch.Tensor) -> torch.Tensor:
        f_radial, f_dct, f_wavelet = freq_feats
        q = self.query_proj(sem_feat).unsqueeze(1)

        k_r = self.key_proj_radial(f_radial).unsqueeze(1)
        v_r = self.value_proj_radial(f_radial).unsqueeze(1)
        out_r, _ = self.attn_radial(q, k_r, v_r)

        k_d = self.key_proj_dct(f_dct).unsqueeze(1)
        v_d = self.value_proj_dct(f_dct).unsqueeze(1)
        out_d, _ = self.attn_dct(q, k_d, v_d)

        k_w = self.key_proj_wavelet(f_wavelet).unsqueeze(1)
        v_w = self.value_proj_wavelet(f_wavelet).unsqueeze(1)
        out_w, _ = self.attn_wavelet(q, k_w, v_w)

        fused = out_r + out_d + out_w
        return fused.squeeze(1)


class GatedCrossAttentionFusion(nn.Module):
    """Hierarchical gating over parallel attention streams."""

    def __init__(self, freq_dims: List[int], sem_dim: int, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.parallel = CrossAttentionFusionParallel(freq_dims, sem_dim, hidden_dim, num_heads)
        self.gate_net = nn.Sequential(
            nn.Linear(sem_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, freq_feats: List[torch.Tensor], sem_feat: torch.Tensor) -> torch.Tensor:
        f_radial, f_dct, f_wavelet = freq_feats
        q_feat = self.parallel.query_proj(sem_feat)  # reuse query proj from parallel module

        # compute each attention output using same q
        out_r = self.parallel.attn_radial(q_feat.unsqueeze(1), self.parallel.key_proj_radial(f_radial).unsqueeze(1), self.parallel.value_proj_radial(f_radial).unsqueeze(1))[0].squeeze(1)
        out_d = self.parallel.attn_dct(q_feat.unsqueeze(1), self.parallel.key_proj_dct(f_dct).unsqueeze(1), self.parallel.value_proj_dct(f_dct).unsqueeze(1))[0].squeeze(1)
        out_w = self.parallel.attn_wavelet(q_feat.unsqueeze(1), self.parallel.key_proj_wavelet(f_wavelet).unsqueeze(1), self.parallel.value_proj_wavelet(f_wavelet).unsqueeze(1))[0].squeeze(1)

        gates = torch.softmax(self.gate_net(sem_feat), dim=-1)
        fused = gates[:, 0:1] * out_r + gates[:, 1:2] * out_d + gates[:, 2:3] * out_w
        return fused


class GXMAFusionDetector(nn.Module):
    """PoC detector combining frequency fingerprints and CLIP semantics."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_classes: int = 2,
        fusion_strategy: str = "concat",
    ) -> None:
        super().__init__()
        self.fusion_strategy = fusion_strategy

        if fusion_strategy == "concat":
            self.freq_extractor = FrequencyFeatureExtractor()
        else:
            self.freq_extractor = FrequencyFeatureExtractorSplit()

        self.sem_extractor = CLIPCLSExtractor()

        sem_dim = self.sem_extractor.hidden_dim

        if fusion_strategy == "concat":
            freq_dim = 256  # 128 + 64 + 64
            self.fusion = CrossAttentionFusion(freq_dim, sem_dim, hidden_dim, num_heads)
        elif fusion_strategy == "parallel":
            self.fusion = CrossAttentionFusionParallel([128, 64, 64], sem_dim, hidden_dim, num_heads)
        elif fusion_strategy == "gated":
            self.fusion = GatedCrossAttentionFusion([128, 64, 64], sem_dim, hidden_dim, num_heads)
        else:
            raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 'images' is a batch of tensors on the target device (e.g., cuda:1)
        sem_feat = self.sem_extractor(images)

        target_device = next(self.parameters()).device

        if self.fusion_strategy == "concat":
            freq_feats = [self.freq_extractor(img.cpu()) for img in images]
            freq_feat = torch.stack(freq_feats, dim=0).to(target_device)
            fused = self.fusion(freq_feat, sem_feat)
        else:
            freq_feats = [self.freq_extractor(img.cpu()) for img in images]
            f_radial = torch.stack([f[0] for f in freq_feats], dim=0).to(target_device)
            f_dct = torch.stack([f[1] for f in freq_feats], dim=0).to(target_device)
            f_wavelet = torch.stack([f[2] for f in freq_feats], dim=0).to(target_device)
            fused = self.fusion([f_radial, f_dct, f_wavelet], sem_feat)

        logits = self.classifier(fused)
        return logits
