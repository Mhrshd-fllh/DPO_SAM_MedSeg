from __future__ import annotations
import torch
import torch.nn as nn
from losses.dice import DiceLoss
from losses.focal import BinaryFocalLoss


class DiceFocalCombo(nn.Module):
    def __init__(self, dice_w: float = 20.0, focal_w: float = 1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = BinaryFocalLoss()
        self.dice_w = dice_w
        self.focal_w = focal_w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_w * self.dice(logits, targets) + self.focal_w * self.focal(logits, targets)
