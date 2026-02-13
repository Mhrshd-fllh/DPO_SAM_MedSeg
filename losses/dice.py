from __future__ import annotations
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,1,H,W]
        targets: [B,1,H,W] in {0,1}
        """
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1,2,3))
        den = (probs + targets).sum(dim=(1,2,3)) + self.eps
        dice = 1 - (num / den)
        return dice.mean()
