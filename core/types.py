from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import torch

@dataclass(frozen=True)
class VisualPromptArtifacts:
    tensors: Dict[str, Any]  # numpy arrays or torch tensors

@dataclass(frozen=True)
class VisualPrompts:
    boxes_xyxy: torch.Tensor
    points_xy: torch.Tensor
    points_labels: torch.Tensor
    artifacts: Optional[VisualPromptArtifacts] = None

@dataclass(frozen=True)
class TextPrompts:
    text: List[str]

@dataclass(frozen=True)
class PromptBundle:
    visual: VisualPrompts
    text: TextPrompts

@dataclass(frozen=True)
class Batch:
    image: torch.Tensor
    mask: Optional[torch.Tensor]
    meta: Dict
