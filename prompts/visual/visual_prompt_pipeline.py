from __future__ import annotations
from typing import List, Any, Dict
import numpy as np
import torch

from core.types import VisualPrompts, VisualPromptArtifacts
from prompts.visual.densecrf import densecrf_refine
from prompts.visual.postprocess import (
    keep_largest_components,
    batch_mask_to_boxes,
    batch_sample_points,
)

class VisualPromptPipeline:
    def __init__(
        self,
        saliency_fn,
        clip_adapter,
        num_points: int = 8,
        max_components: int = 1,
        crf_enabled: bool = True,
        crf_iters: int = 5,
        points_seed: int = 42,
        saliency_threshold: float = 0.5,
        return_artifacts: bool = True,
    ):
        self.saliency_fn = saliency_fn
        self.clip = clip_adapter
        self.num_points = num_points
        self.max_components = max_components
        self.crf_enabled = crf_enabled
        self.crf_iters = crf_iters
        self.points_seed = points_seed
        self.saliency_threshold = saliency_threshold
        self.return_artifacts = return_artifacts

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, class_texts: List[str]) -> VisualPrompts:
        B, _, H, W = images.shape

        sal, acts = self.saliency_fn(images, class_texts, self.clip)  # sal: [B,H,W] np

        imgs_u8 = (images.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        masks_pre, masks_post = [], []
        for i in range(B):
            pre = (sal[i] > self.saliency_threshold).astype(np.uint8)
            masks_pre.append(pre)

            if self.crf_enabled:
                post = densecrf_refine(imgs_u8[i], sal[i], iters=self.crf_iters)
            else:
                post = pre
            masks_post.append(post)

        masks_pre = np.stack(masks_pre, axis=0)     # [B,H,W]
        masks_post = np.stack(masks_post, axis=0)   # [B,H,W]

        closed = np.stack([keep_largest_components(m, max_components=self.max_components) for m in masks_post], axis=0)

        boxes = batch_mask_to_boxes(closed)  # [B,1,4]
        pts, lbl = batch_sample_points(closed, k=self.num_points, seed=self.points_seed)

        boxes_t = torch.from_numpy(boxes).to(images.device)
        pts_t = torch.from_numpy(pts).to(images.device)
        lbl_t = torch.from_numpy(lbl).to(images.device)

        artifacts = None
        if self.return_artifacts:
            art: Dict[str, Any] = {
                "saliency": sal,              # np [B,H,W]
                "mask_pre": masks_pre,        # np [B,H,W]
                "mask_post": masks_post,      # np [B,H,W]
                "mask_cc": closed,            # np [B,H,W]
                "boxes": boxes,               # np [B,1,4]
                "points": pts,                # np [B,K,2]
                "point_labels": lbl,          # np [B,K]
            }
            art.update(acts)
            artifacts = VisualPromptArtifacts(tensors=art)

        return VisualPrompts(
            boxes_xyxy=boxes_t,
            points_xy=pts_t,
            points_labels=lbl_t,
            artifacts=artifacts,
        )
