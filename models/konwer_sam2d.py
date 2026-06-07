from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.types import VisualPrompts, TextPrompts
from models.prompt_fuser import PromptFuser


@dataclass
class KonwerOutputs:
    mask_logits: torch.Tensor  # [B,1,H,W]
    extra: Dict[str, Any]


class KonwerSAM2D(nn.Module):
    """
    SAM-style wrapper with text prompt support:
      - image_encoder
      - prompt_encoder (boxes + points + text)
      - mask_decoder
    
    Text prompts are fused with visual prompts via PromptFuser.
    """
    def __init__(self, sam_model: nn.Module, text_encoder=None, fusion_mode: str = "concat"):
        super().__init__()
        self.sam = sam_model
        self.text_encoder = text_encoder
        self.fusion_mode = fusion_mode

        # expect SAM-like attributes
        for attr in ["image_encoder", "prompt_encoder", "mask_decoder"]:
            if not hasattr(self.sam, attr):
                raise AttributeError(f"SAM model missing attribute: {attr}")

        # Initialize text-visual fusion if text encoder provided
        self.prompt_fuser = None
        if text_encoder is not None:
            self.prompt_fuser = PromptFuser(
                text_dim=512,  # BiomedCLIP dimension
                prompt_dim=256,  # SAM prompt encoder dimension
                fusion_mode=fusion_mode,
            )

    def forward(
        self,
        images: torch.Tensor,
        vp: VisualPrompts,
        tp: Optional[TextPrompts] = None,
    ) -> KonwerOutputs:
        """
        images: [B,3,H,W] float in [0,1]
        vp.boxes_xyxy: [B,1,4]
        vp.points_xy:  [B,K,2]
        vp.points_labels: [B,K]
        tp: TextPrompts with text: List[str] of length B (optional)
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

        # Fuse text embeddings if provided
        if tp is not None and self.text_encoder is not None and self.prompt_fuser is not None:
            # Encode text -> [B, 512]
            text_embeddings = self.text_encoder.encode(tp.text)  
            
            # Pass BOTH sparse and dense tensors to allow dense_add/dense_mul modes
            sparse_embeddings, dense_embeddings = self.prompt_fuser(
                sparse_embeddings, 
                dense_embeddings, 
                text_embeddings
            )

        # Feed the updated dense prompt embeddings into the decoder
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings, # Fused text content integrated here
            multimask_output=False,
        )
        # low_res_masks: [B,1,h,w] (usually 256/4 etc)
        mask_logits = F.interpolate(low_res_masks, size=(H, W), mode="bilinear", align_corners=False)

        return KonwerOutputs(
            mask_logits=mask_logits,
            extra={"iou_pred": iou_predictions},
        )
