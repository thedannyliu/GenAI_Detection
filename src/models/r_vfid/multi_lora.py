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

    # ---------------- Proxy attributes ----------------
    # Certain external modules (e.g., torch.nn.MultiheadAttention) directly access
    # `linear.weight` / `linear.bias` attributes instead of calling forward().
    # Provide read-only proxy to underlying *frozen* base weight so that such code
    # does not raise AttributeError.  LoRA 更新僅在 forward 過程中生效，
    # 但這些側路操作通常出現在推論前後的線性投影 (如 out_proj) 對梯度影響輕微。

    @property  # type: ignore[override]
    def weight(self):  # noqa: D401
        return self.base.weight

    @property  # type: ignore[override]
    def bias(self):  # noqa: D401
        return self.base.bias

    # --------------------------------------------------
    def add_expert(self, r: int = 4, lora_alpha: int = 8, dropout: float = 0.0) -> int:
        """Append a new LoRA expert branch and return its index."""
        new_exp = LoRALinear(self.base, r=r, lora_alpha=lora_alpha, dropout=dropout)
        # default requires_grad=True, older experts frozen by default
        for p in new_exp.parameters():
            p.requires_grad = True
        # freeze old experts
        for exp in self.experts:
            for p in exp.parameters():
                p.requires_grad = False
        self.experts.append(new_exp)
        return len(self.experts) - 1

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

    injected_cnt = 0
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
                injected_cnt += 1
            else:
                # separate projections
                for proj_name in ["q_proj", "k_proj", "v_proj", "q", "k", "v"]:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), nn.Linear):
                        new_layer = MultiLoRALinear(
                            getattr(attn, proj_name), num_experts, r=r, lora_alpha=lora_alpha
                        )
                        setattr(attn, proj_name, new_layer)
                        injected_cnt += 1

    # --------------------------------------------------------------
    # Secondary pass: wrap *all* Linear layers whose module path indicates
    # they belong to an attention block (attn/attention) – covers variant
    # naming schemes in open_clip.
    # --------------------------------------------------------------
    if injected_cnt == 0:
        for name, module in list(visual.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            # Heuristic: at least one of the parent segments mentions attention
            if any(seg in name for seg in ["attn", "attention", "q_proj", "k_proj", "v_proj", "qkv"]):
                parent_path = name.rsplit(".", 1)[0]
                parent = visual
                for attr in parent_path.split(".") if parent_path else []:
                    parent = getattr(parent, attr)
                attr_name = name.split(".")[-1]
                if isinstance(getattr(parent, attr_name), nn.Linear):
                    setattr(parent, attr_name, MultiLoRALinear(module, num_experts, r=r, lora_alpha=lora_alpha))
                    injected_cnt += 1
        # After batch wrapping, continue even if only some found – will still work with set_expert.

    # Final fallback removed; rely on attention-based search.

    # ------------------------------------------------------------------
    # attach helper to visual for global switching
    # ------------------------------------------------------------------

    def _set_expert(self, idx: int):  # type: ignore[override]
        for m in self.modules():
            if isinstance(m, MultiLoRALinear):
                m.set_expert(idx)

    visual.set_expert = types.MethodType(_set_expert, visual)  # type: ignore[attr-defined]

    # 至此至少應有一個 MultiLoRALinear，如無則表 internal 結構變動 