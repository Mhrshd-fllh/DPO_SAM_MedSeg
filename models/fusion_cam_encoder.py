from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionArtifacts:
    saliency_hw: torch.Tensor        # [B,1,H,W] (original resized to image)
    saliency_e: torch.Tensor         # [B,1,He,We] (resized to embedding grid)
    gate: torch.Tensor              # [B,1,He,We]
    image_embeddings: torch.Tensor   # [B,C,He,We]
    fused_embeddings: torch.Tensor   # [B,C,He,We]


class CAMEncoderFusion(nn.Module):
    """
    Fuse CAM saliency (0..1) with SAM image encoder embeddings.

    Modes:
      - "residual_mul": E' = E * (1 + alpha * S)
      - "mul":         E' = E * S
      - "lerp":        E' = (1-beta)*E + beta*(E*S)  (beta in [0,1])
    """

    def __init__(self, mode: str = "residual_mul", alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        assert mode in {"residual_mul", "mul", "lerp"}
        self.mode = mode
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, image_embeddings: torch.Tensor, saliency: torch.Tensor) -> tuple[torch.Tensor, FusionArtifacts]:
        """
        image_embeddings: [B,C,He,We]
        saliency: either [B,H,W] or [B,1,H,W], float in 0..1
        """
        B, C, He, We = image_embeddings.shape

        if saliency.dim() == 3:
            sal = saliency.unsqueeze(1)  # [B,1,H,W]
        elif saliency.dim() == 4:
            sal = saliency
        else:
            raise ValueError(f"saliency must be [B,H,W] or [B,1,H,W], got {saliency.shape}")

        sal = sal.float().clamp(0.0, 1.0)

        # Resize saliency to embedding grid
        sal_e = F.interpolate(sal, size=(He, We), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

        if self.mode == "residual_mul":
            gate = (1.0 + self.alpha * sal_e)
            fused = image_embeddings * gate
        elif self.mode == "mul":
            gate = sal_e
            fused = image_embeddings * gate
        else:  # lerp
            gate = sal_e
            fused = (1.0 - self.beta) * image_embeddings + self.beta * (image_embeddings * gate)

        art = FusionArtifacts(
            saliency_hw=sal,           # still HW
            saliency_e=sal_e,
            gate=gate,
            image_embeddings=image_embeddings,
            fused_embeddings=fused,
        )
        return fused, art