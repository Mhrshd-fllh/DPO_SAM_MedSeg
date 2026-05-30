from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class TextFusionArtifacts:
    text_embeddings: torch.Tensor  # [B, D]


class PromptFuser(nn.Module):
    """
    Combines visual prompt embeddings with text embeddings.
    Strategy: concatenate then project, or use attention-based fusion.
    """

    def __init__(self, text_dim: int = 512, prompt_dim: int = 256, fusion_mode: str = "concat"):
        """
        text_dim: BiomedCLIP embedding dimension (usually 512)
        prompt_dim: SAM prompt encoder embedding dimension (usually 256)
        fusion_mode: "concat" or "weighted_sum" or "attention"
        """
        super().__init__()
        self.text_dim = text_dim
        self.prompt_dim = prompt_dim
        self.fusion_mode = fusion_mode

        if fusion_mode == "concat":
            # Project concatenated [text_emb + prompt_emb] back to prompt_dim
            self.proj = nn.Linear(text_dim + prompt_dim, prompt_dim)
        elif fusion_mode == "weighted_sum":
            # Learnable weights for text vs visual
            self.alpha = nn.Parameter(torch.tensor(0.5))
            # Project text to match prompt_dim
            self.text_proj = nn.Linear(text_dim, prompt_dim)
        elif fusion_mode == "attention":
            # Simple cross-attention: text attends to visual prompts
            self.text_to_prompt = nn.Linear(text_dim, prompt_dim)

    def forward(
        self,
        sparse_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse text embeddings into prompt embeddings.

        Args:
            sparse_embeddings: [B, N, D] from SAM prompt encoder
            text_embeddings: [B, D] from text encoder

        Returns:
            fused_embeddings: [B, N, D] or [B, N+1, D] depending on mode
        """
        B, N, D = sparse_embeddings.shape
        assert text_embeddings.shape[0] == B, "Batch size mismatch"

        if self.fusion_mode == "concat":
            # Expand text to [B, 1, D_text]
            text_exp = text_embeddings.unsqueeze(1)  # [B, 1, 512]

            # Repeat sparse embeddings and concatenate
            sparse_exp = sparse_embeddings.view(B, -1)  # [B, N*D]

            # For each text token, concatenate with average of visual
            avg_visual = sparse_embeddings.mean(dim=1)  # [B, D]
            combined = torch.cat([text_exp.squeeze(1), avg_visual], dim=-1)  # [B, 512+256]
            fused_token = self.proj(combined).unsqueeze(1)  # [B, 1, D]

            # Return combined: text token + original visual
            fused = torch.cat([fused_token, sparse_embeddings], dim=1)  # [B, N+1, D]
            return fused

        elif self.fusion_mode == "weighted_sum":
            text_proj = self.text_proj(text_embeddings).unsqueeze(1)  # [B, 1, D]
            # Average visual prompts
            visual_avg = sparse_embeddings.mean(dim=1, keepdim=True)  # [B, 1, D]
            # Weighted sum
            fused = (1.0 - self.alpha) * visual_avg + self.alpha * text_proj
            return torch.cat([fused, sparse_embeddings], dim=1)  # [B, N+1, D]

        elif self.fusion_mode == "attention":
            # Simple: add text to sparse embeddings
            text_proj = self.text_to_prompt(text_embeddings).unsqueeze(1)  # [B, 1, D]
            fused = torch.cat([text_proj, sparse_embeddings], dim=1)  # [B, N+1, D]
            return fused

        return sparse_embeddings  # Fallback
