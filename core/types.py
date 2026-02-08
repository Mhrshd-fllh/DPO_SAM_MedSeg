from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List
import torch


@dataclass(frozen=True)
class VisualPrompts:
    """
    SAM-style visual prompts:
      - boxes_xyxy: [B, 1, 4] float (x1,y1,x2,y2)
      - points_xy:  [B, K, 2] float (x,y)
      - points_labels: [B, K] int64 (1 pos, 0 neg)
    """
    boxes_xyxy: torch.Tensor
    points_xy: torch.Tensor
    points_labels: torch.Tensor


@dataclass(frozen=True)
class TextPrompts:
    """
    Text prompts per sample (already concatenated: VQA answer + GPT description).
    """
    text: List[str]


@dataclass(frozen=True)
class PromptBundle:
    visual: VisualPrompts
    text: TextPrompts


@dataclass(frozen=True)
class Batch:
    """
    Standard batch format throughout training/eval.
    """
    image: torch.Tensor                  # [B,3,H,W] float32
    mask: Optional[torch.Tensor]         # [B,1,H,W] float32 (None for unlabeled)
    meta: Dict                           # ids, original sizes, dataset name, etc.
