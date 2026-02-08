from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from prompts.visual.hooks import capture_activations


def _resolve_layer(model: nn.Module, layer_path: str) -> nn.Module:
    obj = model
    for part in layer_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


class BiomedCLIPAdapter:
    """
    Holds:
      - clip model (open_clip)
      - preprocess (PIL -> tensor normalized)
      - tokenizer
    """
    def __init__(self, model, preprocess, tokenizer, device="cuda"):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device).eval()

    def preprocess_uint8_rgb(self, imgs_uint8_rgb: np.ndarray) -> torch.Tensor:
        """
        imgs_uint8_rgb: [B,H,W,3] uint8 (RGB)
        returns: [B,3,224,224] float tensor (normalized, per CLIP preprocess)
        """
        ts = []
        for i in range(imgs_uint8_rgb.shape[0]):
            pil = Image.fromarray(imgs_uint8_rgb[i], mode="RGB")
            t = self.preprocess(pil).unsqueeze(0)  # [1,3,224,224]
            ts.append(t)
        x = torch.cat(ts, dim=0).to(self.device)
        return x


class _CLIPTextLogitsWrapper(nn.Module):
    """
    Wrap CLIP model to output logits over a fixed set of texts.
    This makes it compatible with CAM libraries expecting class logits.
    """
    def __init__(self, clip_model: nn.Module, text_tokens: torch.Tensor):
        super().__init__()
        self.clip = clip_model
        self.register_buffer("text_tokens", text_tokens)

    @torch.no_grad()
    def _text_features(self):
        txt = self.clip.encode_text(self.text_tokens)
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-6)
        return txt

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # image_tensor: [B,3,224,224] normalized
        img = self.clip.encode_image(image_tensor)
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-6)
        txt = self._text_features()  # [T,D]
        logits = img @ txt.T         # [B,T]
        return logits


def _vit_reshape_transform(x: torch.Tensor) -> torch.Tensor:
    """
    For ViT, CAM needs [B,C,H,W]. Many ViT blocks output [B, tokens, C].
    We'll drop cls token and reshape tokens into square map.
    """
    # x: [B, tokens, C]
    if x.dim() != 3:
        return x
    B, T, C = x.shape
    # assume first token is CLS
    if T <= 1:
        return x.transpose(1, 2).unsqueeze(-1)
    t = T - 1
    s = int(np.sqrt(t))
    if s * s != t:
        # fallback: treat as 1D map
        return x[:, 1:, :].transpose(1, 2).unsqueeze(-1)
    x = x[:, 1:, :]                 # [B, t, C]
    x = x.reshape(B, s, s, C).permute(0, 3, 1, 2).contiguous()  # [B,C,s,s]
    return x


class GScoreCAMSaliency:
    """
    Fully runnable ScoreCAM-based saliency generator for CLIP models.
    Returns:
      saliency: np.ndarray [B, H, W] in [0,1]
      acts: dict of optional captured activations (if capture_layer set)
    """
    def __init__(
        self,
        target_layer_path: str,
        capture_layer: Optional[str] = None,
        use_vit_reshape: bool = True,
    ):
        self.target_layer_path = target_layer_path
        self.capture_layer = capture_layer
        self.use_vit_reshape = use_vit_reshape

    @torch.no_grad()
    def __call__(self, images_torch: torch.Tensor, class_texts: List[str], clip_adapter: BiomedCLIPAdapter):
        """
        images_torch: [B,3,H,W] in [0,1] (typically 256x256 after our resize)
        class_texts: length B; we compute saliency per-sample to respect per-sample text.
        """
        from pytorch_grad_cam import ScoreCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        B, _, H, W = images_torch.shape
        imgs_u8 = (images_torch.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # resolve target layer once
        target_layer = _resolve_layer(clip_adapter.model, self.target_layer_path)

        saliency_all = []
        acts_all: Dict[str, Any] = {}

        for i in range(B):
            # Prepare single-sample CLIP input tensor for CAM
            x_cam = clip_adapter.preprocess_uint8_rgb(imgs_u8[i:i+1])  # [1,3,224,224]

            # Tokenize single text
            tokens = clip_adapter.tokenizer([class_texts[i]]).to(clip_adapter.device)

            # Wrap CLIP to output logits over 1 text
            wrapped = _CLIPTextLogitsWrapper(clip_adapter.model, tokens).to(clip_adapter.device).eval()

            # capture optional activation
            local_acts: Dict[str, Any] = {}
            if self.capture_layer:
                ctx = capture_activations(clip_adapter.model, self.capture_layer, local_acts, f"layer_out_{i:03d}")
            else:
                # dummy context manager
                class _Null:
                    def __enter__(self): return self
                    def __exit__(self, *a): return False
                ctx = _Null()

            # CAM compute
            with ctx:
                cam = ScoreCAM(
                    model=wrapped,
                    target_layers=[target_layer],
                    reshape_transform=_vit_reshape_transform if self.use_vit_reshape else None,
                )
                # target category 0 (only one text)
                grayscale_cam = cam(
                    input_tensor=x_cam,
                    targets=[ClassifierOutputTarget(0)],
                )  # np [1, 224, 224]

            cam_map = grayscale_cam[0].astype(np.float32)

            # Resize CAM back to our pipeline resolution (H,W)
            cam_map = np.clip(cam_map, 0.0, 1.0)
            cam_map = cv2_resize_float(cam_map, (W, H))

            # Normalize [0,1] robustly
            mn, mx = cam_map.min(), cam_map.max()
            cam_map = (cam_map - mn) / (mx - mn + 1e-6)

            saliency_all.append(cam_map)

            # merge captured activation
            for k, v in local_acts.items():
                acts_all[k] = v

        saliency = np.stack(saliency_all, axis=0)  # [B,H,W]
        return saliency, acts_all


def cv2_resize_float(x: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    import cv2
    return cv2.resize(x, size_wh, interpolation=cv2.INTER_LINEAR)
