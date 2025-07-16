# pyright: reportMissingImports=false
import torch.nn as nn
from typing import Optional

from .lora import LoRALinear


def _wrap_linear_with_lora(
    linear: nn.Linear,
    r: int = 4,
    lora_alpha: int = 8,
) -> LoRALinear:
    """Return a LoRA-wrapped `nn.Linear`, freezing原權重。"""
    # 凍結原權重
    linear.weight.requires_grad = False
    if linear.bias is not None:
        linear.bias.requires_grad = False
    return LoRALinear(linear, r=r, lora_alpha=lora_alpha)


def add_lora_to_vit_qkv(
    clip_model: nn.Module,
    r: int = 4,
    lora_alpha: int = 8,
) -> None:
    """遍歷 `clip_model.visual` 中的所有 Transformer block，
    將具有 `qkv` (Linear) 欄位的 attention 模組改寫為 LoRA 版本。

    這個函式會 **in-place** 修改 `clip_model`，不回傳。
    若找不到符合條件的層，將跳過。
    """
    visual = getattr(clip_model, "visual", None)
    if visual is None:
        # open_clip 早期版本 model.visual -> VisionTransformer；若不存在則直接返回
        return

    # open_clip 的 ViT 命名大多為 visual.trunk 或 visual.transformer、visual.blocks
    # 嘗試多種路徑以保險起見。
    candidate_containers = []
    for attr_name in ["blocks", "transformer", "resblocks", "trunk"]:
        if hasattr(visual, attr_name):
            container = getattr(visual, attr_name)
            candidate_containers.append(container)

    for container in candidate_containers:
        # 決定 block 可迭代物件
        if isinstance(container, (list, tuple, nn.ModuleList)):
            block_iter = container  # 直接迭代
        else:
            if hasattr(container, "resblocks"):
                block_iter = container.resblocks  # VisionTransformer 內部 resblocks
            else:
                block_iter = list(container.children())  # fallback 可能仍為[]

        for block in block_iter:
            # 常見命名：block.attn  / block.attention
            attn = None
            if hasattr(block, "attn"):
                attn = getattr(block, "attn")
            elif hasattr(block, "attention"):
                attn = getattr(block, "attention")
            if attn is None:
                continue

            # ViT 常見：attn.qkv 為 Linear(embed, 3*embed)
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                attn.qkv = _wrap_linear_with_lora(attn.qkv, r=r, lora_alpha=lora_alpha)
            else:
                # 若分開 q_proj / k_proj / v_proj
                for proj_name in ["q_proj", "k_proj", "v_proj", "q", "k", "v"]:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), nn.Linear):
                        new_layer = _wrap_linear_with_lora(getattr(attn, proj_name), r=r, lora_alpha=lora_alpha)
                        setattr(attn, proj_name, new_layer) 