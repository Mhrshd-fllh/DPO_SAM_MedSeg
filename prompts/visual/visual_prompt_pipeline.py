from __future__ import annotations
from typing import List
import numpy as np
import torch

from core.types import VisualPrompts
from prompts.visual.densecrf import densecrf_refine
from prompts.visual.postprocess import (
    keep_largest_components,
    batch_mask_to_boxes,
    batch_sample_points,
)

class VisualPromptPipeline:
    """
    Paper-faithful pipeline: BiomedCLIP -> gScoreCAM -> DenseCRF -> CC -> box + points. :contentReference[oaicite:4]{index=4}
    """
    def __init__(
        self,
        saliency_fn,              # callable(images_torch, class_texts, clip_adapter)-> saliency [B,H,W]
        clip_adapter,             # BiomedCLIPAdapter
        num_points: int = 8,
        max_components: int = 1,
        crf_enabled: bool = True,
        crf_iters: int = 5,
        points_seed: int = 42,
        saliency_threshold: float = 0.5,
    ):
        self.saliency_fn = saliency_fn
        self.clip = clip_adapter
        self.num_points = num_points
        self.max_components = max_components
        self.crf_enabled = crf_enabled
        self.crf_iters = crf_iters
        self.points_seed = points_seed
        self.saliency_threshold = saliency_threshold

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, class_texts: List[str]) -> VisualPrompts:
        """
        images: [B,3,H,W] float32 in [0,1]
        class_texts: list[str] length B (e.g. "breast tumor")
        returns: VisualPrompts with pixel coords on resized images (H=W=256 typically)
        """
        B, _, H, W = images.shape

        # 1) gScoreCAM saliency using BiomedCLIP :contentReference[oaicite:5]{index=5}
        sal = self.saliency_fn(images, class_texts, self.clip)  # [B,H,W] float [0,1]

        # 2) DenseCRF refine to coarse mask :contentReference[oaicite:6]{index=6}
        imgs_u8 = (images.permute(0,2,3,1).cpu().numpy() * 255.0).clip(0,255).astype(np.uint8)

        masks = []
        for i in range(B):
            if self.crf_enabled:
                m = densecrf_refine(imgs_u8[i], sal[i], iters=self.crf_iters)
            else:
                m = (sal[i] > self.saliency_threshold).astype(np.uint8)
            masks.append(m)
        masks = np.stack(masks, axis=0)  # [B,H,W] uint8

        # 3) area constraint: keep largest CC(s) :contentReference[oaicite:7]{index=7}
        closed = np.stack([keep_largest_components(m, max_components=self.max_components) for m in masks], axis=0)

        # 4) box + random points from area :contentReference[oaicite:8]{index=8}
        boxes = batch_mask_to_boxes(closed)                      # [B,1,4]
        pts, lbl = batch_sample_points(closed, k=self.num_points, seed=self.points_seed)

        # to torch
        boxes_t = torch.from_numpy(boxes).to(images.device)
        pts_t   = torch.from_numpy(pts).to(images.device)
        lbl_t   = torch.from_numpy(lbl).to(images.device)

        return VisualPrompts(boxes_xyxy=boxes_t, points_xy=pts_t, points_labels=lbl_t)
