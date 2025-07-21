# pyright: reportMissingImports=false
from typing import Tuple

import torch
import torch.nn as nn


class MultiheadCrossAttentionRouter(nn.Module):
    """Router that maps (prompt_tokens, vit_tokens) → α over K experts via MHA.

    The Query comes from prompt tokens; Key/Value from vision tokens.
    After attention，平均池化 Query 方向，接全連線 softmax ⇒ α。
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 1,
        num_experts: int = 3,
        gating: str = "sigmoid",  # "softmax"|"sigmoid"
        normalize_sigmoid: bool = True,
    ):
        """Cross-attention router.

        Parameters
        ----------
        embed_dim : int
            dimension of token embeddings.
        num_heads : int
            MHA heads.
        num_experts : int
            number of experts K.
        gating : str, default "softmax"
            If "softmax" (default) the output α sums to 1 (single-choice style).
            If "sigmoid" the output passes through a sigmoid, allowing *multi-hot*
            activation.  When ``normalize_sigmoid`` is True, α will be L1-normalized
            so that its magnitude is comparable to softmax.
        normalize_sigmoid : bool, default True
            Whether to divide sigmoid activations by their sum (with ε) to keep the
            overall scale roughly constant.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, num_experts)
        self.num_experts = num_experts
        self.gating = gating
        self.normalize_sigmoid = normalize_sigmoid

    # --------------------------------------------------
    def add_expert(self, num_new: int = 1) -> None:
        """Expand output dimension to accommodate additional expert(s)."""
        if num_new <= 0:
            return
        old_weight = self.fc.weight.data
        old_bias = self.fc.bias.data
        out_dim, in_dim = old_weight.shape
        new_out_dim = out_dim + num_new
        new_fc = nn.Linear(in_dim, new_out_dim)
        # copy old params
        with torch.no_grad():
            new_fc.weight[:out_dim] = old_weight
            new_fc.bias[:out_dim] = old_bias
            # keep remaining params default init
        self.fc = new_fc
        self.num_experts += num_new

    def forward(self, prompt_tokens: torch.Tensor, vit_tokens: torch.Tensor) -> torch.Tensor:
        # prompt_tokens: (B, L', d)  serve as query
        # vit_tokens: (B, T, d)     serve as key/value
        attn_out, _ = self.attn(prompt_tokens, vit_tokens, vit_tokens, need_weights=False)
        pooled = attn_out.mean(dim=1)  # (B, d)
        logits = self.fc(pooled)  # (B, K)
        if self.gating == "softmax":
            alpha = logits.softmax(dim=-1)
        elif self.gating == "sigmoid":
            alpha = logits.sigmoid()
            if self.normalize_sigmoid:
                alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            raise ValueError(f"Unknown gating mode {self.gating}")
        return alpha 