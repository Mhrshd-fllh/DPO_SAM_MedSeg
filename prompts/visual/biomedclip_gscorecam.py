from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch

class BiomedCLIPAdapter:
    """
    Minimal interface:
      - encode_image(images) -> image_features
      - encode_text(texts)   -> text_features
    You can back this by open-clip-torch BiomedCLIP checkpoints.
    """
    def __init__(self, model, preprocess, tokenizer, device="cuda"):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device).eval()

    @torch.no_grad()
    def encode_image(self, images_uint8_rgb: np.ndarray) -> torch.Tensor:
        # images_uint8_rgb: [B,H,W,3] uint8
        imgs = []
        for i in range(images_uint8_rgb.shape[0]):
            x = self.preprocess(images_uint8_rgb[i]).unsqueeze(0)  # [1,3,h,w]
            imgs.append(x)
        x = torch.cat(imgs, dim=0).to(self.device)
        feats = self.model.encode_image(x)
        return feats

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        t = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(t)
        return feats


from prompts.visual.hooks import capture_activations

class GScoreCAMSaliency:
    def __init__(self, scorecam_impl, device="cuda", capture_layer: str | None = None):
        self.impl = scorecam_impl
        self.device = device
        self.capture_layer = capture_layer

    @torch.no_grad()
    def __call__(self, images_torch, class_texts, clip_adapter):
        imgs = (images_torch.permute(0,2,3,1).cpu().numpy() * 255.0).clip(0,255).astype("uint8")

        acts = {}
        if self.capture_layer:
            with capture_activations(clip_adapter.model, self.capture_layer, acts, "layer_out"):
                sal = self.impl.run(clip_adapter.model, imgs, class_texts)
        else:
            sal = self.impl.run(clip_adapter.model, imgs, class_texts)

        # normalize saliency ...
        # return both saliency + activations
        return sal, acts

