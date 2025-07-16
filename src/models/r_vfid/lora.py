import math
from typing import Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA wrapper for `nn.Linear`.

    Formula:  y = x W^T + scale · x B A,  where B ∈ ℝ^{in×r}, A ∈ ℝ^{r×out}
    Original W is *frozen*; only A, B are trainable.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 4,
        lora_alpha: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")
        self.base = base_linear
        in_features, out_features = base_linear.in_features, base_linear.out_features
        self.r = r
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # LoRA parameters (A: down, B: up) – follow PEFT naming
        self.A = nn.Parameter(torch.randn(in_features, r) * 0.02)
        self.B = nn.Parameter(torch.randn(r, out_features) * 0.02)
        # Freeze base weight
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        lora_part = self.dropout(x) @ self.A @ self.B  # (B, *, out)
        return result + self.scaling * lora_part


class FrequencyExpert(nn.Module):
    """Frequency expert implemented as LoRA-adapted projection of ViT tokens.

    Idea: operate on token sequence; aggregate (mean) then apply LoRA-projected linear.
    This is a simplification vs fully injecting into every Transformer QKV.
    """

    def __init__(self, embed_dim: int = 768, r: int = 4, lora_alpha: int = 8):
        super().__init__()
        # Base linear frozen
        base_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lora_linear = LoRALinear(base_linear, r=r, lora_alpha=lora_alpha)
        nn.init.zeros_(self.lora_linear.base.weight)  # base weight zero out to rely on LoRA part

    def forward(self, vit_tokens: torch.Tensor) -> torch.Tensor:
        # vit_tokens: (B, T, d) — Average Pool as placeholder spectral aggregation
        pooled = vit_tokens.mean(dim=1)  # (B, d)
        return self.lora_linear(pooled) 