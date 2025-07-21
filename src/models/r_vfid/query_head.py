from __future__ import annotations

import torch
import torch.nn as nn

from .lora import LoRALinear


class HierarchicalSemanticQueryHead(nn.Module):
    """Generate query vector via frozen base MLP + additive LoRA corrections.

    After learning *T* tasks, the head contains:
      • frozen base MLP  H_base  (learned at pretraining or task-0)
      • list of LoRA adapters  [LoRA_i]  each trained on task *i* and then frozen
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2
        # Base MLP layers (frozen after init)
        self.base = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Initialize and freeze base params (they can be trained once if desired)
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA adapters list (empty at start)
        self.adapters: nn.ModuleList[LoRALinear] = nn.ModuleList()
        self.embed_dim = embed_dim

    # --------------------------------------------------
    def add_adapter(self, r: int = 4, lora_alpha: int = 8) -> None:
        """Add a new trainable LoRA adapter (will be the ONLY trainable part)."""
        # Base is an identity Linear to follow API (x @ I)
        base_linear = nn.Linear(self.embed_dim, self.embed_dim)
        nn.init.eye_(base_linear.weight)
        base_linear.requires_grad_(False)
        adapter = LoRALinear(base_linear, r=r, lora_alpha=lora_alpha)
        self.adapters.append(adapter)

    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,d)
        q = self.base(x)
        for adapter in self.adapters:
            q = q + adapter(x)
        return q 