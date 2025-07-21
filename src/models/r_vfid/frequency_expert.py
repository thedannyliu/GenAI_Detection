# pyright: reportMissingImports=false
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
    try:
        return F.interpolate(x, size=(int(h * scale), int(w * scale)), mode="lanczos")
    except NotImplementedError:
        # 某些 PyTorch 版本尚未支援 lanczos，退回 bicubic
        return F.interpolate(x, size=(int(h * scale), int(w * scale)), mode="bicubic", align_corners=False)


class _DnCNN(nn.Module):
    """Full DnCNN denoiser (Zhang et al., 2017).

    Default configuration follows the paper for colour images:
      • depth = 20 convolutional layers
      • 64 feature maps per layer
      • BatchNorm in all layers except the first & last

    The original DnCNN is trained to predict the *residual* (noise).  For ease
    of integration with ``FrequencyExpert`` we **return the denoised image**
    (i.e. ``x - residual``).  Downstream code then computes
    ``noise = x - denoised`` to obtain the residual magnitude, identical to
    the conventional formulation.
    """

    def __init__(
        self,
        channels: int = 3,
        depth: int = 20,
        features: int = 64,
        bias: bool = False,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        # 1) First layer (Conv + ReLU)
        layers.append(nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=bias))
        layers.append(nn.ReLU(inplace=True))

        # 2) Middle layers (Conv + BN + ReLU)
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=bias))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # 3) Last layer (Conv)
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=bias))

        self.net = nn.Sequential(*layers)

        # -------- Weight initialisation --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x)  # predicted noise
        denoised = x - residual
        return denoised


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
        patch_level: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.patch_level = patch_level
        # Projection (frozen base 1×1 conv implemented as Linear)
        base_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(base_linear.weight)  # rely on LoRA delta
        self.proj = LoRALinear(base_linear, r=r, lora_alpha=lora_alpha)

        if patch_level:
            # Patch embedding conv (same as ViT-L/14) to turn residual map → tokens
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=14, stride=14, bias=False)
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

        if self.patch_level:
            tokens = self.patch_embed(ll_feat)  # (B,d,H/14,W/14)
            tokens = tokens.flatten(2).transpose(1, 2)  # (B, L, d)
            return tokens  # patch-level tokens

        # ------ vector branch ------
        pooled = F.adaptive_avg_pool2d(ll_feat, 1).view(B, -1)  # (B,3)
        if pooled.size(1) != self.proj.base.in_features:
            pooled = F.pad(pooled, (0, self.proj.base.in_features - pooled.size(1)))
        out = self.proj(pooled)  # (B, d)
        return out 