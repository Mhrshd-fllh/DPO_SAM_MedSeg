from __future__ import annotations
import torch


def dice_coeff(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    den = (preds + targets).sum(dim=(1,2,3)) + eps
    return (2 * inter / den).mean()
