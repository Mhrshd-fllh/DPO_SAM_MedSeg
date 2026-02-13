from __future__ import annotations
from typing import Optional
import os
import torch


def load_sam_model(
    checkpoint_path: str,
    model_type: str = "vit_b",
    device: Optional[str] = None,
):
    """
    Loads SAM-style model from segment_anything registry.
    This works if:
      - you installed SAM-Med2D (which provides segment_anything)
      OR
      - you installed Meta SAM (segment_anything)

    checkpoint_path: path to .pth
    model_type: one of registry keys, e.g. vit_b / vit_l / vit_h (depends on registry)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Provide a valid path in configs/train.yaml (sam.checkpoint)."
        )

    try:
        from segment_anything import sam_model_registry
    except Exception as e:
        raise ImportError(
            "segment_anything is not importable. Install SAM-Med2D or Meta SAM first."
        ) from e

    if model_type not in sam_model_registry:
        raise KeyError(
            f"model_type={model_type} not in sam_model_registry keys: {list(sam_model_registry.keys())}"
        )

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device).train()
    return sam
