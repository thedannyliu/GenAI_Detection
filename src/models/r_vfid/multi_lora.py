# pyright: reportMissingImports=false
"""Multi-expert LoRA injection utilities for ViT QKV.

Each attention QKV Linear is replaced by a `MultiLoRALinear`, which holds N distinct
`LoRALinear` experts.  `visual.set_expert(idx)` 會切換目前作用中的 expert，
使同一張 image 可以依序走多個 LoRA 路徑。
"""
from __future__ import annotations

import types
from typing import List

import torch.nn as nn

from .lora import LoRALinear


class MultiLoRALinear(nn.Module):
    """Wrapper that stores multiple LoRA experts and can switch at runtime."""

    def __init__(
        self,
        base_linear: nn.Linear,
        num_experts: int = 3,
        r: int = 4,
        lora_alpha: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")
        self.base = base_linear
        # freeze base weight
        for p in self.base.parameters():
            p.requires_grad = False
        self.experts = nn.ModuleList(
            [
                LoRALinear(base_linear, r=r, lora_alpha=lora_alpha, dropout=dropout)
                for _ in range(num_experts)
            ]
        )
        self.active_idx: int = 0

    # --------------------------------------------------
    def set_expert(self, idx: int) -> None:  # noqa: D401
        """切換目前使用的 expert 編號 (0-based)."""
        if idx < 0 or idx >= len(self.experts):
            raise ValueError(f"invalid expert idx {idx}")
        self.active_idx = idx
        # 控制梯度：只讓當前 expert 參數可訓練
        for i, exp in enumerate(self.experts):
            for p in exp.parameters():
                p.requires_grad = i == idx

    # --------------------------------------------------
    def forward(self, x):  # type: ignore[override]
        return self.experts[self.active_idx](x)

    # expose some helper attr for inspection
    @property
    def num_experts(self) -> int:  # noqa: D401
        return len(self.experts)


# -----------------------------------------------------------------------------
#    Injection util
# -----------------------------------------------------------------------------

def add_multi_lora_to_vit_qkv(
    clip_model: nn.Module,
    num_experts: int,
    r: int = 4,
    lora_alpha: int = 8,
) -> None:
    """Replace every QKV Linear inside ViT with MultiLoRALinear(num_experts).

    Also monkey-patch `visual.set_expert(idx)` to switch the active expert for
    **all** MultiLoRALinear children in one call.
    """
    visual = getattr(clip_model, "visual", None)
    if visual is None:
        return

    # find Transformer blocks container (best-effort)
    candidate_containers: List[nn.Module] = []
    for attr_name in ["blocks", "transformer", "resblocks", "trunk"]:
        if hasattr(visual, attr_name):
            candidate_containers.append(getattr(visual, attr_name))

    for container in candidate_containers:
        block_iter = (
            container
            if isinstance(container, (list, tuple, nn.ModuleList))
            else getattr(container, "resblocks", list(container.children()))
        )
        for block in block_iter:
            # attn module
            attn = None
            for name in ["attn", "attention"]:
                if hasattr(block, name):
                    attn = getattr(block, name)
            if attn is None:
                continue

            # unified qkv Linear
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                attn.qkv = MultiLoRALinear(attn.qkv, num_experts, r=r, lora_alpha=lora_alpha)
            else:
                # separate projections
                for proj_name in ["q_proj", "k_proj", "v_proj", "q", "k", "v"]:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), nn.Linear):
                        new_layer = MultiLoRALinear(
                            getattr(attn, proj_name), num_experts, r=r, lora_alpha=lora_alpha
                        )
                        setattr(attn, proj_name, new_layer)

    # ------------------------------------------------------------------
    # attach helper to visual for global switching
    # ------------------------------------------------------------------

    def _set_expert(self, idx: int):  # type: ignore[override]
        for m in self.modules():
            if isinstance(m, MultiLoRALinear):
                m.set_expert(idx)

    visual.set_expert = types.MethodType(_set_expert, visual)  # type: ignore[attr-defined]

    # 至此至少應有一個 MultiLoRALinear，如無則表 internal 結構變動 