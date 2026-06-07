from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


class PromptFuser(nn.Module):
    """
    Combines visual prompt embeddings with text embeddings.
    Supports both baseline sparse token concatenation and paper-faithful dense modulation.
    """

    def __init__(self, text_dim: int = 512, prompt_dim: int = 256, fusion_mode: str = "dense_add"):
        """
        text_dim: BiomedCLIP embedding dimension (usually 512)
        prompt_dim: SAM prompt encoder embedding dimension (usually 256)
        fusion_mode: "dense_add", "dense_mul", "concat", "weighted_sum", "attention"
        """
        super().__init__()
        self.text_dim = text_dim
        self.prompt_dim = prompt_dim
        self.fusion_mode = fusion_mode

        # Paper-Faithful Dense Projection Layers
        if "dense" in fusion_mode:
            self.text_to_dense = nn.Sequential(
                nn.Linear(text_dim, prompt_dim),
                nn.GELU()
            )
        
        # Experimental Sparse Tracking Options (Your custom branch styles)
        elif fusion_mode == "concat":
            self.proj = nn.Linear(text_dim + prompt_dim, prompt_dim)
        elif fusion_mode == "weighted_sum":
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.text_proj = nn.Linear(text_dim, prompt_dim)
        elif fusion_mode == "attention":
            self.text_to_prompt = nn.Linear(text_dim, prompt_dim)

    def forward(
        self,
        sparse_embeddings: Optional[torch.Tensor],
        dense_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sparse_embeddings: [B, N, D] from SAM prompt encoder (Can be None or sparse sequence)
            dense_embeddings: [B, D, H_e, W_e] Dense spatial mask feature embeddings from SAM
            text_embeddings: [B, D_text] encoded textual description vector from BiomedCLIP

        Returns:
            fused_sparse: [B, N_out, D] 
            fused_dense:  [B, D, H_e, W_e]
        """
        B, C, H_e, W_e = dense_embeddings.shape
        assert text_embeddings.shape[0] == B, "Batch size mismatch"

        # Initialize defaults to prevent breaking baseline passes
        fused_sparse = sparse_embeddings
        fused_dense = dense_embeddings

        # 1. 🌟 PAPER-FAITHFUL DENSE INJECTION MODES
        if self.fusion_mode == "dense_add":
            # Project text space [B, 512] -> [B, 256]
            text_feat = self.text_to_dense(text_embeddings) # [B, 256]
            # Unsqueeze into explicit spatial dimensions [B, 256, 1, 1] to broadcast element-wise
            text_spatial = text_feat.unsqueeze(-1).unsqueeze(-1) 
            fused_dense = dense_embeddings + text_spatial
            return fused_sparse, fused_dense

        elif self.fusion_mode == "dense_mul":
            text_feat = self.text_to_dense(text_embeddings)
            text_spatial = torch.sigmoid(text_feat.unsqueeze(-1).unsqueeze(-1))
            fused_dense = dense_embeddings * text_spatial
            return fused_sparse, fused_dense

        # 2. 🧪 EXPERIMENTAL SPARSE TOKEN CONFIGURATIONS (From your custom branch)
        if sparse_embeddings is None:
            return fused_sparse, fused_dense

        if self.fusion_mode == "concat":
            text_exp = text_embeddings.unsqueeze(1)
            avg_visual = sparse_embeddings.mean(dim=1)
            combined = torch.cat([text_exp.squeeze(1), avg_visual], dim=-1)
            fused_token = self.proj(combined).unsqueeze(1)
            fused_sparse = torch.cat([fused_token, sparse_embeddings], dim=1)

        elif self.fusion_mode == "weighted_sum":
            text_proj = self.text_proj(text_embeddings).unsqueeze(1)
            visual_avg = sparse_embeddings.mean(dim=1, keepdim=True)
            fused_token = (1.0 - self.alpha) * visual_avg + self.alpha * text_proj
            fused_sparse = torch.cat([fused_token, sparse_embeddings], dim=1)

        elif self.fusion_mode == "attention":
            text_proj = self.text_to_prompt(text_embeddings).unsqueeze(1)
            fused_sparse = torch.cat([text_proj, sparse_embeddings], dim=1)

        return fused_sparse, fused_dense