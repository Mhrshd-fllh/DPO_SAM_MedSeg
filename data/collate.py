from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import torch


@dataclass
class Batch:
    image: torch.Tensor   # [B,3,H,W]
    mask: torch.Tensor    # [B,1,H,W]
    meta: Dict[str, List[Any]]


def collate_samples(samples) -> Batch:
    images = torch.stack([s.image for s in samples], dim=0)
    masks = torch.stack([s.mask for s in samples], dim=0)

    # meta: collect per-key lists
    meta: Dict[str, List[Any]] = {}
    for s in samples:
        for k, v in s.meta.items():
            meta.setdefault(k, []).append(v)

    return Batch(image=images, mask=masks, meta=meta)
