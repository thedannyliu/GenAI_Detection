import torch
import torch.nn as nn
from typing import List, Optional
from PIL import Image
import torchvision.transforms as T

from .frequency_extractors import FrequencyFeatureExtractor
from .clip_semantics import CLIPCLSExtractor


class CrossAttentionFusion(nn.Module):
    """Fuse frequency and semantic features via cross-attention."""

    def __init__(self, freq_dim: int, sem_dim: int, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.query_proj = nn.Linear(sem_dim, hidden_dim)
        self.key_proj = nn.Linear(freq_dim, hidden_dim)
        self.value_proj = nn.Linear(freq_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, freq_feat: torch.Tensor, sem_feat: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(sem_feat).unsqueeze(1)
        k = self.key_proj(freq_feat).unsqueeze(1)
        v = self.value_proj(freq_feat).unsqueeze(1)
        fused, _ = self.attn(q, k, v)
        return fused.squeeze(1)


# =========================== Strategy B: Parallel Attention Streams ===========================


class ParallelCrossAttentionFusion(nn.Module):
    """Parallel cross-attention streams – one dedicated attention module per frequency feature.

    The semantic feature serves as the unified Query.  Each frequency feature (e.g., radial FFT,
    DCT, Wavelet) has its own Key/Value projections and attention head.  The outputs from all
    streams are aggregated (summed by default) to obtain the final fused representation.
    """

    def __init__(
        self,
        freq_dims: dict,
        sem_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        aggregate: str = "sum",
    ) -> None:
        super().__init__()

        if aggregate not in {"sum", "concat"}:
            raise ValueError("aggregate must be 'sum' or 'concat'")

        self.aggregate = aggregate

        # Shared projection for Query (semantic feature)
        self.query_proj = nn.Linear(sem_dim, hidden_dim)

        # Per-method projections & attention modules
        self.key_projs = nn.ModuleDict()
        self.value_projs = nn.ModuleDict()
        self.attn_modules = nn.ModuleDict()

        for method, dim in freq_dims.items():
            self.key_projs[method] = nn.Linear(dim, hidden_dim)
            self.value_projs[method] = nn.Linear(dim, hidden_dim)
            self.attn_modules[method] = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
            )

    def forward(self, freq_feat_dict: dict, sem_feat: torch.Tensor) -> torch.Tensor:
        """Args
        -----
        freq_feat_dict : Dict[str, Tensor]
            Mapping from method name to frequency feature tensor with shape (B, dim_i).
        sem_feat : Tensor
            Semantic feature tensor with shape (B, sem_dim).
        Returns
        -------
        fused : Tensor
            Fused representation with shape (B, hidden_dim) when aggregate == 'sum', otherwise
            (B, hidden_dim * n_methods) for 'concat'.
        """

        q = self.query_proj(sem_feat).unsqueeze(1)  # (B, 1, H)

        outputs = []
        for method, feat in freq_feat_dict.items():
            k = self.key_projs[method](feat).unsqueeze(1)  # (B, 1, H)
            v = self.value_projs[method](feat).unsqueeze(1)
            out, _ = self.attn_modules[method](q, k, v)  # (B, 1, H)
            outputs.append(out)

        if self.aggregate == "sum":
            fused = torch.stack(outputs, dim=0).sum(dim=0).squeeze(1)  # (B, H)
        else:  # concat
            fused = torch.cat([o.squeeze(1) for o in outputs], dim=-1)  # (B, H * n_methods)

        return fused


# =========================== Strategy C: Parallel Streams + Meta-Gate ===========================


class HierarchicalGatedParallelFusion(nn.Module):
    """Parallel attention streams with a semantic-conditioned *Meta-Gate*.

    This extends ``ParallelCrossAttentionFusion`` by learning *softmax weights* (g1 … gN)
    conditioned on the **semantic feature**.  The weights modulate the contribution of
    each frequency expert enabling *context-aware* evidence aggregation as described in
    the GXMA proposal (Strategy C / Tier-2.5).
    """

    def __init__(
        self,
        freq_dims: dict,
        sem_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        gate_hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        # ---------- Parallel attention experts (one per frequency feature) ----------
        self.query_proj = nn.Linear(sem_dim, hidden_dim)
        self.key_projs = nn.ModuleDict()
        self.value_projs = nn.ModuleDict()
        self.attn_modules = nn.ModuleDict()

        for method, dim in freq_dims.items():
            self.key_projs[method] = nn.Linear(dim, hidden_dim)
            self.value_projs[method] = nn.Linear(dim, hidden_dim)
            self.attn_modules[method] = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
            )

        self.methods = list(freq_dims.keys())

        # ---------- Meta-Gate ----------
        self.gate_network = nn.Sequential(
            nn.Linear(sem_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, len(self.methods)),
        )

    def forward(self, freq_feat_dict: dict, sem_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args
        ----
        freq_feat_dict: mapping method → Tensor of shape (B, dim_i).
        sem_feat: semantic feature tensor of shape (B, sem_dim).
        Returns
        -------
        fused: Tensor of shape (B, hidden_dim) – weighted sum of expert outputs.
        """

        # 1. Shared Query projection
        q = self.query_proj(sem_feat).unsqueeze(1)  # (B, 1, H)

        # 2. Run each expert attention
        expert_outputs = []  # List[(B, 1, H)]
        for method in self.methods:
            feat = freq_feat_dict[method]
            k = self.key_projs[method](feat).unsqueeze(1)
            v = self.value_projs[method](feat).unsqueeze(1)
            out, _ = self.attn_modules[method](q, k, v)
            expert_outputs.append(out)  # (B, 1, H)

        # Stack -> (B, N, H)
        experts = torch.cat(expert_outputs, dim=1)  # (B, N_methods, H)

        # 3. Meta-Gate: produce softmax weights (B, N_methods)
        gate_logits = self.gate_network(sem_feat)  # (B, N_methods)
        gate_weights = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)  # (B, N_methods, 1)

        # 4. Weighted aggregation
        fused = torch.sum(experts * gate_weights, dim=1)  # (B, H)

        return fused


class GXMAFusionDetector(nn.Module):
    """PoC detector combining frequency fingerprints and CLIP semantics."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, num_classes: int = 2, freq_methods: Optional[List[str]] = None, model_cfg: Optional[dict] = None) -> None:
        super().__init__()

        if model_cfg is None:
            model_cfg = {}

        # Frequency & semantic extractors
        self.freq_extractor = FrequencyFeatureExtractor(methods=freq_methods)
        self.sem_extractor = CLIPCLSExtractor(lora_cfg=model_cfg.get("lora", None))

        # Determine feature dimensions
        sem_dim = self.sem_extractor.hidden_dim

        # Decide fusion strategy (single vs. parallel)
        self.fusion_strategy = model_cfg.get("fusion_strategy", "single").lower()

        if self.fusion_strategy == "single":
            freq_dim = self.freq_extractor.output_dim
            self.fusion = CrossAttentionFusion(freq_dim, sem_dim, hidden_dim, num_heads)
            fusion_output_dim = hidden_dim
        elif self.fusion_strategy == "parallel":
            # Build mapping: method -> dim
            freq_dims = {m: self.freq_extractor.METHOD_DIM[m] for m in self.freq_extractor.methods}
            self.fusion = ParallelCrossAttentionFusion(freq_dims, sem_dim, hidden_dim, num_heads, aggregate="sum")
            fusion_output_dim = hidden_dim  # 'sum' aggregation keeps dim size
        elif self.fusion_strategy in {"gated", "hierarchical", "hierarchical_gating"}:
            freq_dims = {m: self.freq_extractor.METHOD_DIM[m] for m in self.freq_extractor.methods}
            self.fusion = HierarchicalGatedParallelFusion(freq_dims, sem_dim, hidden_dim, num_heads)
            fusion_output_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported fusion_strategy: {self.fusion_strategy}")

        # Classifier head (2-layer MLP)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor, freq_feat_full: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            images: batch of image tensors (C,H,W) on device.
            freq_feat_full: optional pre-computed frequency feature tensor of shape (B, D_total).
        """
        # Semantic feature on device
        sem_feat = self.sem_extractor(images)

        # If frequency features not supplied, compute on-the-fly (original fallback)
        if freq_feat_full is None:
            freq_feats = [self.freq_extractor(img.cpu()) for img in images]
            freq_feat_full = torch.stack(freq_feats, dim=0)

        target_device = next(self.parameters()).device
        freq_feat_full = freq_feat_full.to(target_device)

        if self.fusion_strategy == "single":
            fused = self.fusion(freq_feat_full, sem_feat)
        elif self.fusion_strategy in {"parallel"}:  # simple parallel sum/concat
            freq_dict = {}
            start = 0
            for method in self.freq_extractor.methods:
                dim = self.freq_extractor.METHOD_DIM[method]
                freq_dict[method] = freq_feat_full[:, start : start + dim]
                start += dim
            fused = self.fusion(freq_dict, sem_feat)
        else:  # hierarchical / gated
            freq_dict = {}
            start = 0
            for method in self.freq_extractor.methods:
                dim = self.freq_extractor.METHOD_DIM[method]
                freq_dict[method] = freq_feat_full[:, start : start + dim]
                start += dim
            fused = self.fusion(freq_dict, sem_feat)
        logits = self.classifier(fused)
        return logits
