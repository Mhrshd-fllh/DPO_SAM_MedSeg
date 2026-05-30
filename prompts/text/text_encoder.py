from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn

from prompts.visual.load_biomedclip import encode_text


class TextEncoderAdapter:
    """
    Encodes text strings to BiomedCLIP embeddings.
    For use in SAM-style models with text prompts.
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        model: BiomedCLIP model (eval mode)
        tokenizer: BiomedCLIP tokenizer
        device: cuda or cpu
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of text strings to embeddings.

        Args:
            texts: List[str] of length B

        Returns:
            text_embeddings: [B, D] (normalized)
        """
        # Handle empty strings
        if not texts or all(t == "" for t in texts):
            # Return zero embeddings
            return torch.zeros(len(texts), 512, device=self.device)

        # Encode
        feats = encode_text(self.model, self.tokenizer, texts, self.device)
        return feats  # [B, D]
