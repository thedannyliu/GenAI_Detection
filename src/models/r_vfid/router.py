from typing import Tuple

import torch
import torch.nn as nn


class MultiheadCrossAttentionRouter(nn.Module):
    """Router that maps (prompt_tokens, vit_tokens) → α over K experts via MHA.

    The Query comes from prompt tokens; Key/Value from vision tokens.
    After attention，平均池化 Query 方向，接全連線 softmax ⇒ α。
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 1, num_experts: int = 3):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, num_experts)

    def forward(self, prompt_tokens: torch.Tensor, vit_tokens: torch.Tensor) -> torch.Tensor:
        # prompt_tokens: (B, L', d)  serve as query
        # vit_tokens: (B, T, d)     serve as key/value
        attn_out, _ = self.attn(prompt_tokens, vit_tokens, vit_tokens, need_weights=False)
        pooled = attn_out.mean(dim=1)  # (B, d)
        alpha = self.fc(pooled).softmax(dim=-1)  # (B, K)
        return alpha 