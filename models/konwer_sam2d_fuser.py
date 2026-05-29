from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.types import VisualPrompts
from models.fusion_cam_encoder import CAMEncoderFusion, FusionArtifacts


from dataclasses import dataclass
from typing import Optional
import torch
from models.fusion_cam_encoder import FusionArtifacts

@dataclass
class DualMaskOutputs:
    baseline_mask_logits: torch.Tensor   # [B,1,H,W]
    fused_mask_logits: torch.Tensor      # [B,1,H,W]
    combined_mask_logits: torch.Tensor   # [B,1,H,W]
    fusion_artifacts: Optional[FusionArtifacts]

class KonwerSAM2DFused(nn.Module):
    """
    Runs SAM-Med2D twice:
      1) baseline
      2) fused: image_embeddings gated by CAM saliency before mask decoder
    """

    def __init__(self, sam, fusion: CAMEncoderFusion, lambda_logits: float = 0.5):
        super().__init__()
        self.sam = sam
        self.fusion = fusion
        self.lambda_logits = float(lambda_logits)

    @torch.no_grad()
    def _encode_prompts(self, visual_prompts: VisualPrompts):
        """
        Returns (sparse_embeddings, dense_embeddings)
        This assumes SAM-like prompt_encoder signature.
        """
        # point/box tensors expected by SAM:
        points = (visual_prompts.points_xy, visual_prompts.points_labels)
        boxes = visual_prompts.boxes_xyxy

        # Some forks accept boxes=None; keep robust:
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
        )
        return sparse_embeddings, dense_embeddings

    def forward(self, images: torch.Tensor, visual_prompts: VisualPrompts) -> DualMaskOutputs:
        """
        images: [B,3,H,W] float
        visual_prompts: contains prompts + artifacts (cam saliency in artifacts)
        """
        B, _, H, W = images.shape

        # ----- image encoder -----
        image_embeddings = self.sam.image_encoder(images)  # [B,C,He,We]

        # ----- prompt encoder -----
        sparse_embeddings, dense_embeddings = self._encode_prompts(visual_prompts)

        # ----- baseline mask decoder -----
        low_res_masks_base, iou_preds_base = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # low_res_masks: [B,1,h',w'] -> upsample to H,W
        baseline_logits = F.interpolate(low_res_masks_base, size=(H, W), mode="bilinear", align_corners=False)

        # ----- fused branch -----
        fusion_artifacts = None
        fused_logits = baseline_logits  # fallback

        # extract saliency from visual_prompts artifacts if present
        saliency = None
        if getattr(visual_prompts, "artifacts", None) is not None:
            tens = visual_prompts.artifacts.tensors
            # in your pipeline the key is "saliency" (np [B,H,W])
            if "saliency" in tens:
                sal = tens["saliency"]
                if isinstance(sal, torch.Tensor):
                    saliency = sal
                else:
                    # numpy -> torch
                    saliency = torch.from_numpy(sal).to(images.device)

        if saliency is not None:
            fused_embeddings, fusion_artifacts = self.fusion(image_embeddings, saliency)

            low_res_masks_fused, iou_preds_fused = self.sam.mask_decoder(
                image_embeddings=fused_embeddings,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            fused_logits = F.interpolate(low_res_masks_fused, size=(H, W), mode="bilinear", align_corners=False)

        lam = self.lambda_logits
        combined_logits = (1.0 - lam) * baseline_logits + lam * fused_logits

        return DualMaskOutputs(
            baseline_mask_logits=baseline_logits,
            fused_mask_logits=fused_logits,
            combined_mask_logits=combined_logits,
            fusion_artifacts=fusion_artifacts,
        )