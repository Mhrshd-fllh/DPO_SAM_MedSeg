from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.types import VisualPrompts
# Import the new learnable module instead of the old static one
from models.red_mask_fuser import RedMaskSpatialFuser
from models.fusion_cam_encoder import FusionArtifacts


@dataclass
class DualMaskOutputs:
    baseline_mask_logits: torch.Tensor   # [B,1,H,W]
    fused_mask_logits: torch.Tensor      # [B,1,H,W]
    combined_mask_logits: torch.Tensor   # [B,1,H,W]
    fusion_artifacts: Optional[FusionArtifacts]


class KonwerSAM2DFused(nn.Module):
    """
    Runs SAM-Med2D with two parallel branches:
      1) Baseline branch: Standard SAM execution.
      2) Red-Mask Convolutional Fused branch: Dynamic spatial gating and feature injection
         using the coarse BiomedCLIP CAM masks to suppress false positives.
    """

    def __init__(self, sam, lambda_logits: float = 0.5):
        super().__init__()
        self.sam = sam
        self.lambda_logits = float(lambda_logits)
        
        # 🚀 Injecting the learnable convolutional fuser (channel dimension = 256)
        self.red_mask_fuser = RedMaskSpatialFuser(embed_dim=256)

    @torch.no_grad()
    def _encode_prompts(self, visual_prompts: VisualPrompts):
        """
        Returns (sparse_embeddings, dense_embeddings)
        Extracts point and box coordinates from visual prompt bundles.
        """
        points = (visual_prompts.points_xy, visual_prompts.points_labels)
        boxes = visual_prompts.boxes_xyxy

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
        )
        return sparse_embeddings, dense_embeddings

    def forward(self, images: torch.Tensor, visual_prompts: VisualPrompts) -> DualMaskOutputs:
        """
        Args:
            images: [B,3,H,W] float image tensors
            visual_prompts: Bundle containing prompts and artifacts (saliency masks inside)
        """
        B, _, H, W = images.shape

        # ----- 1. Image & Prompt Encoding -----
        image_embeddings = self.sam.image_encoder(images)  # [B, 256, He, We]
        He, We = image_embeddings.shape[-2:]
        
        sparse_embeddings, dense_embeddings = self._encode_prompts(visual_prompts)

        # ----- 2. Baseline Mask Decoder Branch -----
        low_res_masks_base, iou_preds_base = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        baseline_logits = F.interpolate(low_res_masks_base, size=(H, W), mode="bilinear", align_corners=False)

        # ----- 3. Red-Mask Fused Branch (Our Innovation) -----
        fusion_artifacts = None
        fused_logits = baseline_logits  # Fallback option if saliency is missing

        # Extract coarse CAM mask (saliency) from visual_prompts artifacts safely
        saliency = None
        if getattr(visual_prompts, "artifacts", None) is not None:
            tens = visual_prompts.artifacts.tensors
            if "saliency" in tens:
                sal = tens["saliency"]
                if isinstance(sal, torch.Tensor):
                    saliency = sal
                else:
                    saliency = torch.from_numpy(sal).to(images.device)

        # Execute our custom learnable spatial fusion block
        if saliency is not None:
            # Step A: Pass coarse mask through convolutional fuser to get weights and gating maps
            mask_feat, spatial_gate = self.red_mask_fuser(saliency, (He, We))
            
            # Step B: Apply Hadamard multiplication (gating) and add residual edge features
            fused_embeddings = (image_embeddings * spatial_gate) + mask_feat

            # Step C: Run mask decoder using the newly enriched features
            low_res_masks_fused, iou_preds_fused = self.sam.mask_decoder(
                image_embeddings=fused_embeddings,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            fused_logits = F.interpolate(low_res_masks_fused, size=(H, W), mode="bilinear", align_corners=False)

            # Keep the training artifacts logged so MLflow/Loss tracking doesn't break
            if saliency.dim() == 3:
                sal_hw = saliency.unsqueeze(1)
            else:
                sal_hw = saliency
                
            fusion_artifacts = FusionArtifacts(
                saliency_hw=sal_hw.float().clamp(0.0, 1.0),
                saliency_e=F.interpolate(sal_hw.float(), size=(He, We), mode="bilinear", align_corners=False).clamp(0.0, 1.0),
                gate=spatial_gate,
                image_embeddings=image_embeddings,
                fused_embeddings=fused_embeddings,
            )

        # ----- 4. Logits Combination -----
        lam = self.lambda_logits
        combined_logits = (1.0 - lam) * baseline_logits + lam * fused_logits

        return DualMaskOutputs(
            baseline_mask_logits=baseline_logits,
            fused_mask_logits=fused_logits,
            combined_mask_logits=combined_logits,
            fusion_artifacts=fusion_artifacts,
        )