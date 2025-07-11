from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRALinear


# ------------------------------------------------------------
#   Basic low-level filters (PyTorch convolution kernels)
# ------------------------------------------------------------

def _get_srm_kernel(device: torch.device) -> torch.Tensor:
    # 3×3 high-pass SRM kernel (simplified NoisePrint proxy)
    kernel = torch.tensor([
        [-1,  2, -1],
        [ 2, -4,  2],
        [-1,  2, -1],
    ], dtype=torch.float32, device=device) / 4.0
    kernel = kernel.view(1, 1, 3, 3)
    return kernel.repeat(3, 1, 1, 1)  # depth-wise for 3 channels


def _bicubic_upsample(x: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
    h, w = x.shape[-2:]
    return F.interpolate(x, size=(int(h * scale), int(w * scale)), mode="bicubic", align_corners=False)


def _lanczos_downsample(x: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    h, w = x.shape[-2:]
    return F.interpolate(x, size=(int(h * scale), int(w * scale)), mode="lanczos")


class _DnCNN(nn.Module):
    """Tiny 3-layer DnCNN-style denoiser (placeholder)."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1),
        )
        # Initialise as identity (so residual ≈ 0) – easier convergence
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------
#   FrequencyExpert Module
# ------------------------------------------------------------

class FrequencyExpert(nn.Module):
    """Produce low-level feature vector F_k (B,d) for R-VFiD.

    Parameters
    ----------
    mode : Literal["npr", "dncnn", "noiseprint"]
        Select which low-level transformation to apply.
    embed_dim : int
        Target dimension (CLIP d=768).
    r, lora_alpha : int
        LoRA config for projection head (maps pooled freq map → d).
    """

    def __init__(
        self,
        mode: Literal["npr", "dncnn", "noiseprint"],
        embed_dim: int = 768,
        r: int = 4,
        lora_alpha: int = 8,
    ) -> None:
        super().__init__()
        self.mode = mode
        # Projection (frozen base 1×1 conv implemented as Linear)
        base_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(base_linear.weight)  # rely on LoRA delta
        self.proj = LoRALinear(base_linear, r=r, lora_alpha=lora_alpha)

        if mode == "dncnn":
            self.denoiser = _DnCNN()
        elif mode == "noiseprint":
            # register fixed SRM kernel as conv
            kernel = _get_srm_kernel(torch.device("cpu"))
            self.register_buffer("srm_kernel", kernel, persistent=False)
        # npr does not need params

    # -----------------------------------------------------
    def _lowlevel_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "npr":
            up = _bicubic_upsample(x)
            down = _lanczos_downsample(up)
            residual = (x - down).abs()
            return residual
        elif self.mode == "dncnn":
            noise = x - self.denoiser(x)
            return noise.abs()
        else:  # noiseprint
            residual = F.conv2d(x, self.srm_kernel, padding=1, groups=3)
            return residual.abs()

    def forward(self, images: torch.Tensor, vit_tokens: torch.Tensor | None = None) -> torch.Tensor:
        # images: (B,3,H,W)
        B, _, H, W = images.shape
        device = images.device
        ll_feat = self._lowlevel_transform(images)  # (B,3,H,W)
        # Global average pool per channel then flatten concat
        pooled = F.adaptive_avg_pool2d(ll_feat, 1).view(B, -1)  # (B,3)
        # Project to embed_dim via LoRA linear (will broadcast)
        # Need input dim match base_linear.in_features (embed_dim). If 3 != 768, project via simple FC.
        if pooled.size(1) != self.proj.base.in_features:
            # simple linear pad
            pooled = F.pad(pooled, (0, self.proj.base.in_features - pooled.size(1)))
        out = self.proj(pooled)  # (B, d)
        return out 