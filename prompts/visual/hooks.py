from __future__ import annotations
from contextlib import contextmanager
from typing import Dict, Optional
import torch
import torch.nn as nn

@contextmanager
def capture_activations(model: nn.Module, layer_path: str, store: Dict[str, torch.Tensor], key: str):
    """
    layer_path مثل:
      - "visual.trunk.layer4"  (بسته به معماری)
      - "visual.transformer.resblocks.11"
    """
    layer = model
    for part in layer_path.split("."):
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)

    handle = None
    def hook_fn(_, __, output):
        store[key] = output.detach()

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()
