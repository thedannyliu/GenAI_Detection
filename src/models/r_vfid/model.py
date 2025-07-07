import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .lora import FrequencyExpert
from .router import MultiheadCrossAttentionRouter
from .prompt_utils import PromptBuilder

try:
    import open_clip
except ImportError:
    open_clip = None  # type: ignore

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None  # type: ignore


class RouterPrompt(nn.Module):
    """Read-only prompt tokens (learnable) used as Query in Router.

    Parameters
    ----------
    prompt_length : int
        Number of tokens in a single prompt sentence (e.g. 7).
    num_sentences : int
        Number of sentences per image (e.g. 2 * c where c is top-c labels).
    embed_dim : int
        Embedding dimension (matches CLIP text encoder, typically 768).
    """

    def __init__(self, prompt_length: int = 7, num_sentences: int = 10, embed_dim: int = 768):
        super().__init__()
        self.prompt_tokens = nn.Parameter(
            torch.randn(1, num_sentences * prompt_length, embed_dim) * 0.02
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        # broadcast to batch
        return self.prompt_tokens.repeat(batch_size, 1, 1)  # (B, L', d)


class SimpleFrequencyExpert(nn.Module):
    """Stub implementation of a frequency expert.

    Real implementation should insert LoRA adapters into ViT blocks.
    Here we simply perform global average pooling over token dimension
    followed by a linear projection so that the interface matches.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, vit_tokens: torch.Tensor) -> torch.Tensor:
        # vit_tokens: (B, T, d)
        x = vit_tokens.mean(dim=1)  # (B, d)
        return self.proj(x)


class CrossAttentionRouter(nn.Module):
    """Single-head Cross-Attention that outputs gating weights α over K experts."""

    def __init__(self, embed_dim: int = 768, num_experts: int = 3):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, num_experts)
        self.num_experts = num_experts

    def forward(self, prompt_tokens: torch.Tensor, vit_tokens: torch.Tensor) -> torch.Tensor:
        # prompt_tokens: (B, L', d); vit_tokens: (B, T, d)
        q = self.q(prompt_tokens)  # (B, L', d)
        k = self.k(vit_tokens)     # (B, T, d)
        v = self.v(vit_tokens)     # (B, T, d)

        attn_scores = torch.einsum("bld,btd->blt", q, k) / (k.size(-1) ** 0.5)  # (B, L', T)
        attn_probs = attn_scores.softmax(dim=-1)  # (B, L', T)
        ctx = torch.einsum("blt,btd->bld", attn_probs, v)  # (B, L', d)

        # Aggregate over token dimension L' via mean, then map to α
        ctx_mean = ctx.mean(dim=1)  # (B, d)
        alpha_logits = self.out_proj(ctx_mean)  # (B, K)
        alpha = alpha_logits.softmax(dim=-1)
        return alpha  # (B, K)


class SEFusion(nn.Module):
    """Squeeze-and-Excite style recalibration over concatenated vector."""

    def __init__(self, in_dim: int = 1536, reduction: int = 4):
        super().__init__()
        hidden = in_dim // reduction
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2d)
        z = x.mean(dim=1, keepdim=True)  # (B, 1, 2d) — but mean over channel misplaced
        # use channel-wise squeeze: mean over feature dim
        z = x.mean(dim=-1, keepdim=True)  # (B, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))  # (B, 2d)
        return x * s


class RvfidModel(nn.Module):
    """Full R-VFiD pipeline (minimal skeleton)."""

    def __init__(
        self,
        num_experts: int = 3,
        prompt_length: int = 7,
        top_c: int = 5,
        embed_dim: int = 768,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        if open_clip is None:
            raise ImportError("open_clip_torch is required but not installed.")

        # 1. CLIP backbone (vision & text encoders)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        self.clip_model.eval()
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # 2. Prompt Builder for conditioned prompt tokens
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        # A minimal ImageNet class name list subset; should load full list in practice
        imagenet_stub = ["dog", "cat", "person", "car", "tree", "building", "sky", "flower"]
        self.prompt_builder = PromptBuilder(
            self.clip_model, tokenizer, imagenet_stub, top_c=top_c, prompt_length=prompt_length
        )

        # 3. Frequency Experts
        self.experts = nn.ModuleList([
            FrequencyExpert(embed_dim) for _ in range(num_experts)
        ])

        # 4. Router (Cross-Attention gating)
        self.router = MultiheadCrossAttentionRouter(embed_dim, num_heads=1, num_experts=num_experts)

        # 5. Fusion layer
        self.fusion = SEFusion(in_dim=embed_dim * 2)

        # 6. Classification Head (LoRA optional)
        self.classifier = nn.Linear(embed_dim * 2, 2)
        # Optionally wrap with LoRA via peft
        if LoraConfig is not None:
            lora_cfg = LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["weight"],  # placeholder
                lora_dropout=0.0,
                bias="none",
                task_type="SEQ_CLS",
            )
            try:
                self.classifier = get_peft_model(self.classifier, lora_cfg)
            except Exception:
                pass  # peft version mismatch; fall back to plain linear

    # -----------------------------------------------------
    # Utility helpers
    # -----------------------------------------------------
    @torch.no_grad()
    def _zero_shot_labels(self, images: torch.Tensor, c: int = 5) -> List[List[str]]:
        """Placeholder zero-shot top-c class prediction using CLIP.

        Returns list of class names per image. Current stub returns fixed labels.
        """
        batch_size = images.size(0)
        return [["object"] * c for _ in range(batch_size)]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.size(0)
        device = images.device

        # Step 0: normalise via clip_preprocess if images are PIL; assume tensor already.

        # Step 1: Generate conditioned Router Prompt embeddings
        prompt_tokens = self.prompt_builder.build(images).to(device)  # (B, L', d)

        # Step 2: Vision backbone — token sequence
        vit_tokens = self.clip_model.encode_image(images, output_hidden_states=False)
        # open_clip returns CLS-pooled vector by default; need hidden states — we use penultimate layer representation.
        if vit_tokens.ndim == 2:  # (B, d)
            # tile to fake token dim
            vit_tokens = vit_tokens.unsqueeze(1)  # (B, 1, d)
        V_cls = vit_tokens[:, 0, :]  # guard against token dimension

        # Step 3: Frequency Experts
        F_list = [expert(vit_tokens) for expert in self.experts]  # list of (B, d)
        F_stack = torch.stack(F_list, dim=1)  # (B, K, d)

        # Step 4: Router gating
        alpha = self.router(prompt_tokens, vit_tokens)  # (B, K)
        alpha_unsq = alpha.unsqueeze(-1)  # (B, K, 1)

        V_freq = (alpha_unsq * F_stack).sum(dim=1)  # (B, d)

        # Step 5: Fusion
        V_cat = torch.cat([V_cls, V_freq], dim=-1)  # (B, 2d)
        V_fuse = self.fusion(V_cat)

        # Step 6: Head
        logits = self.classifier(V_fuse)  # (B, 2)
        return logits


__all__: Tuple[str, ...] = ("RvfidModel",) 