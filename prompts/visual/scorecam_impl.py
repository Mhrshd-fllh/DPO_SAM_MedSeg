from __future__ import annotations
from typing import List
import numpy as np
import torch

class GScoreCAMImpl:
    """
    Wrapper that exposes:
      run(model, imgs_uint8_rgb, texts) -> saliency [B,H,W] float
    You will adapt the internals depending on the exact gScoreCAM API version.
    """
    def __init__(self, target_layer: str):
        self.target_layer = target_layer

    def run(self, clip_model, imgs_uint8_rgb: np.ndarray, texts: List[str]) -> np.ndarray:
        # TODO: implement with the gScoreCAM repo API.
        # The key idea:
        #  1) choose a target layer (self.target_layer)
        #  2) compute class score for the given text prompt (CLIP similarity)
        #  3) run gScoreCAM to get per-pixel saliency
        raise NotImplementedError(
            "Wire gScoreCAM repo here. If you tell me your gScoreCAM version / API entrypoint, "
            "I can adapt this function exactly to it."
        )
