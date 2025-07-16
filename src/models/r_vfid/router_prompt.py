import torch
import torch.nn as nn


class RouterPromptPool(nn.Module):
    """Maintain a pool of learnable *read-only* router prompts.

    • Each domain/generator gets `prompt_length` tokens (learnable).
    • Existing prompts are **frozen** whenever新的 domain 被加入; 只微調最新 prompt。
    • Forward 時將全部 prompts 展平成 (L_total, d) 並 broadcast 到 batch。
    """

    def __init__(self, num_prompts: int = 1, prompt_length: int = 7, embed_dim: int = 768):
        super().__init__()
        self.prompt_length = prompt_length
        # Register as Parameter: (N, L, d)
        self.prompt_tokens = nn.Parameter(torch.randn(num_prompts, prompt_length, embed_dim) * 0.02)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def forward(self, batch_size: int) -> torch.Tensor:
        # Flatten prompts along prompt & token dim → (L_total, d)
        flat = self.prompt_tokens.reshape(-1, self.prompt_tokens.size(-1))  # (N*L, d)
        return flat.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, L_total, d)

    @torch.no_grad()
    def add_prompt(self):
        """Append a fresh prompt (L tokens) for a new domain; freeze old ones."""
        # Freeze existing
        self.prompt_tokens.requires_grad_(False)
        # Init new prompt
        new_prompt = torch.randn(1, self.prompt_length, self.prompt_tokens.size(-1), device=self.prompt_tokens.device) * 0.02
        # Concatenate
        updated = torch.cat([self.prompt_tokens, new_prompt], dim=0)
        # Re-register
        self.prompt_tokens = nn.Parameter(updated) 