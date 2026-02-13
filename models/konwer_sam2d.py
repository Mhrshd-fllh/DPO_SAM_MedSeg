from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.types import VisualPrompts


@dataclass
class KonwerOutputs:
    mask_logits: torch.Tensor  # [B,1,H,W]
    extra: Dict[str, Any]


class KonwerSAM2D(nn.Module):
    """
    SAM-style wrapper:
      - image_encoder
      - prompt_encoder (boxes + points)
      - mask_decoder
    """
    def __init__(self, sam_model: nn.Module):
        super().__init__()
        self.sam = sam_model

        # expect SAM-like attributes
        for attr in ["image_encoder", "prompt_encoder", "mask_decoder"]:
            if not hasattr(self.sam, attr):
                raise AttributeError(f"SAM model missing attribute: {attr}")

    def forward(self, images: torch.Tensor, vp: VisualPrompts) -> KonwerOutputs:
        """
        images: [B,3,H,W] float in [0,1]
        vp.boxes_xyxy: [B,1,4]
        vp.points_xy:  [B,K,2]
        vp.points_labels: [B,K]
        """
        B, C, H, W = images.shape

        # SAM expects pixel range depending on implementation.
        # We'll assume images already normalized to [0,1] and let SAM handle.
        image_embeddings = self.sam.image_encoder(images)  # [B, ...]
        # Dense positional encoding
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(vp.points_xy, vp.points_labels),
            boxes=vp.boxes_xyxy,
            masks=None,
        )

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # low_res_masks: [B,1,h,w] (usually 256/4 etc)
        mask_logits = F.interpolate(low_res_masks, size=(H, W), mode="bilinear", align_corners=False)

        return KonwerOutputs(
            mask_logits=mask_logits,
            extra={"iou_pred": iou_predictions},
        )
