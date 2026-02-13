from __future__ import annotations
import numpy as np
import torch

from core.types import VisualPrompts
from prompts.visual.postprocess import batch_mask_to_boxes, batch_sample_points


def build_visual_prompts_from_gt_masks(
    masks: torch.Tensor,  # [B,1,H,W] in {0,1}
    num_points: int = 8,
    seed: int = 42,
) -> VisualPrompts:
    m = masks.detach().cpu().numpy()
    m = (m[:, 0] > 0).astype(np.uint8)  # [B,H,W]

    boxes = batch_mask_to_boxes(m)  # [B,1,4]
    pts, lbl = batch_sample_points(m, k=num_points, seed=seed)

    device = masks.device
    return VisualPrompts(
        boxes_xyxy=torch.from_numpy(boxes).to(device),
        points_xy=torch.from_numpy(pts).to(device),
        points_labels=torch.from_numpy(lbl).to(device),
    )
