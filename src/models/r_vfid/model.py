# pyright: reportMissingImports=false

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .frequency_expert import FrequencyExpert
from .router import MultiheadCrossAttentionRouter
from .multi_lora import add_multi_lora_to_vit_qkv
from .prompt_utils import PromptBuilder
from .router_prompt import RouterPromptPool

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
    """Channel-wise Squeeze-and-Excitation over feature vector (B, C)."""

    def __init__(self, in_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, in_dim // reduction)
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)
        s = self.fc2(F.relu(self.fc1(x))).sigmoid()  # (B, C)
        return x * s


class RvfidModel(nn.Module):
    """Full R-VFiD pipeline (minimal skeleton)."""

    def __init__(
        self,
        num_experts: int = 3,
        prompt_length: int = 7,
        top_c: int = 5,
        embed_dim: int | None = None,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        if open_clip is None:
            raise ImportError("open_clip_torch is required but not installed.")

        # 1. CLIP backbone (vision & text encoders)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.clip_model.eval()
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # 注入 LoRA 於 ViT QKV
        add_multi_lora_to_vit_qkv(self.clip_model, num_experts=num_experts, r=4, lora_alpha=8)

        # 依照實際 embed_dim 設定
        visual_dim = getattr(self.clip_model.visual, "width", None)
        if visual_dim is None:
            visual_dim = getattr(self.clip_model.visual, "embed_dim", 1024)
        self.embed_dim = embed_dim or visual_dim

        # Text token dim
        self.text_embed_dim: int = self.clip_model.token_embedding.embedding_dim  # typically 768

        # 若視覺 dim 與文字 dim 不一致，新增投影
        if self.text_embed_dim != self.embed_dim:
            self.prompt_proj = nn.Linear(self.text_embed_dim, self.embed_dim, bias=False)
        else:
            self.prompt_proj = nn.Identity()

        # 2a. Router Prompt Pool (learnable, read-only style)
        self.router_prompt_pool = RouterPromptPool(num_prompts=1, prompt_length=prompt_length, embed_dim=self.embed_dim)

        # 2b. Prompt Builder for conditioned prompt tokens
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        # A minimal ImageNet class name list subset; should load full list in practice
        imagenet_stub = ["dog", "cat", "person", "car", "tree", "building", "sky", "flower"]
        self.prompt_builder = PromptBuilder(
            self.clip_model, tokenizer, imagenet_stub, top_c=top_c, prompt_length=prompt_length
        )

        # 3. Frequency Experts
        modes_cycle = ["npr", "dncnn", "noiseprint"]
        self.experts = nn.ModuleList([
            FrequencyExpert(mode=modes_cycle[i % len(modes_cycle)], embed_dim=self.embed_dim, patch_level=True)  # type: ignore[arg-type]
            for i in range(num_experts)
        ])

        # 4. Router (Cross-Attention gating)
        self.router = MultiheadCrossAttentionRouter(self.embed_dim, num_heads=1, num_experts=num_experts)

        # 5. Fusion layer
        self.fusion = SEFusion(in_dim=self.embed_dim * 2)

        # 6. Classification Head (LoRA optional)
        self.classifier = nn.Linear(self.embed_dim * 2, 2)
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

        # Step 1: Router prompt (static + conditioned)
        static_prompt = self.router_prompt_pool(B).to(device)
        dynamic_prompt = self.prompt_builder.build(images).to(device)
        dynamic_prompt = self.prompt_proj(dynamic_prompt)
        prompt_tokens = torch.cat([static_prompt, dynamic_prompt], dim=1)  # (B, L_total, d)

        # --------------------
        # Semantic tokens per expert
        # --------------------

        tokens_per_expert = [expert(images) for expert in self.experts]  # (B,L,d) ×K
        F_list_vec = [tok.mean(dim=1) for tok in tokens_per_expert]
        F_stack = torch.stack(F_list_vec, dim=1)  # (B,K,d)

        # 先用 expert-0 取語意 tokens 給 Router
        self.clip_model.visual.set_expert(0)
        vit_tokens0 = self._get_visual_tokens(images)
        if vit_tokens0.size(-1) != self.embed_dim:
            if not hasattr(self, "_visual_proj"):
                self._visual_proj = nn.Linear(vit_tokens0.size(-1), self.embed_dim, bias=False).to(device)
            vit_tokens0 = self._visual_proj(vit_tokens0)

        # Router α
        alpha = self.router(prompt_tokens, vit_tokens0)  # (B,K)
        self.latest_alpha = alpha

        # ------------------------------------------
        # 迴圈各 expert → CLS token
        # ------------------------------------------
        V_cls_list = []
        tokens_stack = []
        for k in range(self.num_experts):
            self.clip_model.visual.set_expert(k)
            vit_tok_k = self._get_visual_tokens(images)
            if vit_tok_k.size(-1) != self.embed_dim:
                vit_tok_k = self._visual_proj(vit_tok_k)

            # fuse patch tokens from freq expert k
            patch_tok = tokens_per_expert[k]
            if patch_tok.shape[1] == vit_tok_k.shape[1] - 1:
                vit_tok_k = torch.cat([vit_tok_k[:, :1, :], vit_tok_k[:, 1:, :] + patch_tok], dim=1)

            V_cls_list.append(vit_tok_k[:, 0, :])
            tokens_stack.append(patch_tok)

        V_cls_stack = torch.stack(V_cls_list, dim=1)  # (B,K,d)
        alpha_unsq = alpha.unsqueeze(-1)
        V_cls = (alpha_unsq * V_cls_stack).sum(dim=1)  # (B,d)

        # Low-level fused vector
        V_freq = (alpha_unsq * F_stack).sum(dim=1)

        # Step 5: Fusion
        V_cat = torch.cat([V_cls, V_freq], dim=-1)
        V_fuse = self.fusion(V_cat)

        # Step 6: Head
        logits = self.classifier(V_fuse)  # (B, 2)
        # Store V_cls for InfoNCE
        self.latest_vcls = V_cls.detach()
        self.latest_prompt_tokens = prompt_tokens.detach()

        return logits

    # -----------------------------------------------------
    def _get_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return token序列 (B, 1+L, d) from CLIP visual."""
        with torch.no_grad():
            # open_clip 新版提供 visual.forward_features(images, return_all_tokens=True)
            visual = self.clip_model.visual
            tokens = None
            if hasattr(visual, "forward_features"):
                try:
                    tokens = visual.forward_features(images, return_all_tokens=True)
                    if isinstance(tokens, tuple):
                        tokens = tokens[0]
                except Exception:
                    tokens = None

            # open_clip<=0.2 may提供 visual.encode 或直接 __call__ 回 CLS 向量
            if tokens is None and hasattr(visual, "encode"):
                try:
                    tokens = visual.encode(images, return_all_tokens=True)
                except Exception:
                    tokens = None

            if tokens is None:
                # 最終 fallback：visual(images) 可能回 CLS (B,d)
                tokens = visual(images)
        # 保證形狀 (B, seq, d)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)  # (B,1,d)
        return tokens

    # ---------------- Loss Utilities ------------------
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute BCE + InfoNCE + Entropy losses per §3.6."""
        bce = nn.functional.binary_cross_entropy_with_logits(logits[:, 1], labels.float())

        # ---- InfoNCE ----
        # Anchor: per-sample mean router prompt (static part)   (B,d)
        B = labels.size(0)
        prompt_tokens = self.latest_prompt_tokens[:, : self.router_prompt_pool.prompt_tokens.numel() // self.router_prompt_pool.prompt_tokens.size(-1), :]  # static chunk length Ls
        anchor = prompt_tokens.view(B, -1, prompt_tokens.size(-1)).mean(dim=1)  # (B,d)

        v_sem = self.latest_vcls  # (B,d)
        positives = torch.cat([v_sem], dim=0)  # (B,d)
        logits_sim = (anchor @ positives.T) / 0.07  # (B,B)
        labels_nce = torch.arange(B, device=logits.device)
        info_nce = nn.functional.cross_entropy(logits_sim, labels_nce)

        # ---- Entropy ----
        alpha = self.latest_alpha  # (B, K)
        entropy = (-alpha * (alpha + 1e-8).log()).sum(dim=-1).mean()

        total = bce + 0.1 * info_nce + 0.01 * entropy
        return total

    # ---------------- Continual Learning API ------------------
    def add_domain_prompt_and_expert(self, mode: str = "npr", embed_dim: int = 768):
        """Add a new learnable router prompt & corresponding frequency expert."""
        self.router_prompt_pool.add_prompt()
        self.experts.append(FrequencyExpert(mode=mode, embed_dim=embed_dim))  # type: ignore[arg-type]


__all__: Tuple[str, ...] = ("RvfidModel",) 