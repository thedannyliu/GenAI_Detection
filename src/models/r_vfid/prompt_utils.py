# pyright: reportMissingImports=false
import functools
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
except ImportError:
    open_clip = None  # type: ignore


class PromptBuilder:
    """Builds conditioned Router Prompt embeddings per image.

    Steps:
    1. Zero-shot CLIP rank → top-c class names.
    2. For each class, create templates (real/fake) and tokenize.
    3. Pass through CLIP token_embedding to get per-token embeddings; cut / pad to prompt_length.

    Result shape: (B, L', d)  where L' = 2c * prompt_length.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer,
        classnames: Optional[List[str]] = None,
        top_c: int = 5,
        prompt_length: int = 7,
        device: torch.device | str | None = None,
    ) -> None:
        if open_clip is None:
            raise ImportError("open_clip_torch is required for PromptBuilder.")
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        # -----------------------------
        # Load default ImageNet-1K label list if not provided
        # -----------------------------
        if classnames is None:
            # open_clip 自帶 1K 類別名稱 (英語) – 若版本差異導致缺失，退回簡易生成
            if hasattr(open_clip, "IMAGENET_CLASSNAMES"):
                classnames = list(open_clip.IMAGENET_CLASSNAMES)
            else:
                # Fallback: 以 'class_####' 代稱，避免程式中斷
                classnames = [f"class_{i:04d}" for i in range(1000)]

        self.classnames: List[str] = list(classnames)
        self.top_c = top_c
        self.prompt_len = prompt_length
        self.device = device or "cpu"
        # 延遲構建，以便於子類覆寫後再初始化
        self._build_text_features_cache()

    # -----------------------------------------------------
    # Public helpers for continual / dynamic class updates
    # -----------------------------------------------------
    def update_classnames(self, new_classes: List[str], rebuild: bool = True) -> None:
        """Append novel class names and optionally rebuild the text bank."""
        for name in new_classes:
            if name not in self.classnames:
                self.classnames.append(name)
        if rebuild:
            self._build_text_features_cache()

    def set_classnames(self, classnames: List[str]) -> None:
        """Overwrite existing class list and rebuild feature bank."""
        self.classnames = list(classnames)
        self._build_text_features_cache()

    def _build_text_features_cache(self):
        # Build cached text features for zero-shot classification (ImageNet style)
        templates = ["a photo of a {}"]
        all_prompts = [t.format(name.replace("_", " ")) for name in self.classnames for t in templates]
        with torch.no_grad():
            tokenized = self.tokenizer(all_prompts).to(self.device)
            text_feats = self.clip_model.encode_text(tokenized).float()
            text_feats = F.normalize(text_feats, dim=-1)
        self.text_feature_bank = text_feats  # shape (N_classes, d)

    @torch.no_grad()
    def _zero_shot_topc(self, image_feats: torch.Tensor) -> List[List[str]]:
        # image_feats: (B, d) unit-norm
        sims = image_feats @ self.text_feature_bank.T  # (B, N)
        topk = sims.topk(self.top_c, dim=-1).indices  # (B, c)
        batch_names: List[List[str]] = []
        for row in topk:
            names = [self.classnames[idx] for idx in row.tolist()]
            batch_names.append(names)
        return batch_names

    def build(self, images: torch.Tensor) -> torch.Tensor:
        B = images.size(0)
        device = images.device
        # Step 0: image feature
        with torch.no_grad():
            im_feats = self.clip_model.encode_image(images).float()
            im_feats = F.normalize(im_feats, dim=-1)
        batch_class_lists = self._zero_shot_topc(im_feats)

        # Build prompts per image
        prompts_per_image: List[torch.Tensor] = []
        for class_names in batch_class_lists:
            sentence_prompts = []
            for name in class_names:
                sentence_prompts.append(f"a real photo of a {name}")
                sentence_prompts.append(f"a fake photo of a {name}")
            tokenized = self.tokenizer(sentence_prompts).to(device)  # (2c, 77)
            with torch.no_grad():
                tok_embeds = self.clip_model.token_embedding(tokenized)  # (2c, 77, d)
            # take first prompt_length tokens for each sentence
            tok_embeds = tok_embeds[:, : self.prompt_len, :]
            tok_embeds = tok_embeds.reshape(-1, tok_embeds.size(-1))  # (2c*L, d)
            prompts_per_image.append(tok_embeds)

        prompt_tokens = torch.stack([p for p in prompts_per_image], dim=0)  # (B, L', d)
        return prompt_tokens 