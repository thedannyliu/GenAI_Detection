# pyright: reportMissingImports=false

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .frequency_expert import FrequencyExpert
from .router import MultiheadCrossAttentionRouter
from .multi_lora import add_multi_lora_to_vit_qkv
from .query_head import HierarchicalSemanticQueryHead

try:
    import open_clip
except ImportError:
    open_clip = None  # type: ignore

# ---------------- Dependency version checks ----------------
try:
    from packaging import version
except ImportError:
    version = None  # type: ignore

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
        gating_mode: str = "softmax",  # "softmax"|"sigmoid"
    ):
        super().__init__()
        self.num_experts = num_experts
        if open_clip is None:
            raise ImportError("open_clip_torch is required but not installed.")

        # Verify version requirements
        if version is not None and hasattr(open_clip, "__version__"):
            if version.parse(open_clip.__version__) < version.parse("2.20.0"):
                raise RuntimeError("R-VFiD requires open_clip_torch >= 2.20.0; found " + open_clip.__version__)
        if LoraConfig is None:
            raise RuntimeError("R-VFiD requires the 'peft' package for LoRA support. Please install peft>=0.5.0")

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

        # 2. Hierarchical Semantic Query Head
        self.query_head = HierarchicalSemanticQueryHead(embed_dim=self.embed_dim)

        # 3. Frequency Experts
        modes_cycle = ["npr", "dncnn", "noiseprint"]
        self.experts = nn.ModuleList([
            FrequencyExpert(mode=modes_cycle[i % len(modes_cycle)], embed_dim=self.embed_dim, patch_level=True)  # type: ignore[arg-type]
            for i in range(num_experts)
        ])

        # 4. Router (Cross-Attention gating)
        self.router = MultiheadCrossAttentionRouter(
            self.embed_dim, num_heads=1, num_experts=num_experts, gating=gating_mode
        )

        # Store gating_mode for reference
        self.gating_mode = gating_mode

        # UGRR hyperparameters
        self.ugrr_c: float = 10.0
        self.ugrr_hth: float = 1.0

        # 5. Fusion layer
        self.fusion = SEFusion(in_dim=self.embed_dim * 2)

        # Patch-level token fusion (concatenate -> proj)
        self.patch_fuse_proj = nn.Linear(self.embed_dim * 2, self.embed_dim, bias=False)

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
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.size(0)
        device = images.device

        # Step 1: Generate hierarchical semantic query from CLS token
        # Ensure expert-0 branch active when extracting semantic tokens
        self.clip_model.visual.set_expert(0)
        vit_tokens0 = self._get_visual_tokens(images)
        if vit_tokens0.size(-1) != self.embed_dim:
            if not hasattr(self, "_visual_proj"):
                self._visual_proj = nn.Linear(vit_tokens0.size(-1), self.embed_dim, bias=False).to(device)
            vit_tokens0 = self._visual_proj(vit_tokens0)

        # Extract CLS semantic embedding and build query
        v_cls_sem = vit_tokens0[:, 0, :]  # (B,d)
        q_final = self.query_head(v_cls_sem)  # (B,d)
        prompt_tokens = q_final.unsqueeze(1)  # (B,1,d)

        # Cache for loss terms / analysis
        self.latest_q = q_final.detach()

        # Router α
        alpha = self.router(prompt_tokens, vit_tokens0)  # (B,K)
        self.latest_alpha = alpha  # store for loss

        # ------------------------------------------
        # Compute each expert with α-scaled LoRA internally
        # ------------------------------------------

        V_cls_list = []
        F_vec_list = []
        for k in range(self.num_experts):
            a_k = alpha[:, k].unsqueeze(-1)  # (B,1)

            # --- ViT tokens with expert-k LoRA ---
            self.clip_model.visual.set_expert(k)
            vit_tok_k = self._get_visual_tokens(images)
            if vit_tok_k.size(-1) != self.embed_dim:
                vit_tok_k = self._visual_proj(vit_tok_k)

            # --- Frequency tokens ---
            patch_tok = self.experts[k](images)  # (B, L, d)
            if patch_tok.shape[1] == vit_tok_k.shape[1] - 1:
                concat_tok = torch.cat([vit_tok_k[:, 1:, :], patch_tok], dim=-1)
                fused_tok = self.patch_fuse_proj(concat_tok)
                vit_tok_k = torch.cat([vit_tok_k[:, :1, :], fused_tok], dim=1)

            # Scale inside expert (α) before aggregation
            vit_tok_k_scaled = vit_tok_k[:, 0, :] * a_k  # CLS token scaled (B,d)
            V_cls_list.append(vit_tok_k_scaled)

            F_vec = patch_tok.mean(dim=1) * a_k  # (B,d)
            F_vec_list.append(F_vec)

        # Sum directly (α already applied)
        V_cls = torch.stack(V_cls_list, dim=1).sum(dim=1)
        V_freq = torch.stack(F_vec_list, dim=1).sum(dim=1)

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
        # Anchor: query vector itself as semantic anchor
        B = labels.size(0)
        anchor = self.latest_q  # (B,d)

        v_sem = self.latest_vcls  # (B,d)
        positives = torch.cat([v_sem], dim=0)  # (B,d)
        logits_sim = (anchor @ positives.T) / 0.07  # (B,B)
        labels_nce = torch.arange(B, device=logits.device)
        info_nce = nn.functional.cross_entropy(logits_sim, labels_nce)

        # ---- UGRR Loss ----
        alpha = self.latest_alpha  # (B, K)
        entropy = (-alpha * (alpha + 1e-8).log()).sum(dim=-1)  # (B,)
        l2_sq = (alpha.pow(2).sum(dim=-1))  # (B,)
        beta = torch.sigmoid(self.ugrr_c * (entropy - self.ugrr_hth))  # (B,)
        ugrr = (beta * entropy - (1 - beta) * l2_sq).mean()

        total = bce + 0.1 * info_nce + 0.01 * ugrr
        return total

    # ---------------- Continual Learning API ------------------
    def add_domain_prompt_and_expert(self, mode: str = "npr", embed_dim: int = 768):
        """Add a new domain (prompt + frequency expert + LoRA branch).

        1. 在 RouterPromptPool 加入新 prompt (並凍結舊 prompt)。
        2. 為 ViT 中所有 MultiLoRALinear 增加一個 LoRA expert 分支並凍結舊分支。
        3. 增加對應 FrequencyExpert，凍結舊 expert 參數。
        4. 擴充 Router 的 fc 輸出層。
        """

        # 1) New Query Head adapter (LoRA correction)
        self.query_head.add_adapter(r=4, lora_alpha=8)

        # 2) ViT 追加 LoRA expert
        new_idx: int | None = None
        from .multi_lora import MultiLoRALinear  # local import to avoid cycle

        for m in self.clip_model.visual.modules():
            if isinstance(m, MultiLoRALinear):
                new_idx = m.add_expert(r=4, lora_alpha=8)

        # 3) 新 FrequencyExpert，並凍結舊 ones
        for old_exp in self.experts:
            for p in old_exp.parameters():
                p.requires_grad = False
        self.experts.append(FrequencyExpert(mode=mode, embed_dim=self.embed_dim, patch_level=True))  # type: ignore[arg-type]

        # 4) 擴充 Router 輸出
        self.router.add_expert(1)

        # 更新計數
        self.num_experts += 1

        # 驗證 new_idx 與 self.num_experts-1 一致（僅 debug）
        if new_idx is not None and new_idx != self.num_experts - 1:
            print(f"[Warning] MultiLoRA new expert idx {new_idx} inconsistent with model.num_experts {self.num_experts}")


__all__: Tuple[str, ...] = ("RvfidModel",) 