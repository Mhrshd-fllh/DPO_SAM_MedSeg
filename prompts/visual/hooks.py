from __future__ import annotations
from contextlib import contextmanager
from typing import Dict
import torch.nn as nn
import torch

def _resolve_layer(model: nn.Module, layer_path: str) -> nn.Module:
    obj = model
    for part in layer_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj

@contextmanager
def capture_activations(model: nn.Module, layer_path: str, store: Dict[str, torch.Tensor], key: str):
    layer = _resolve_layer(model, layer_path)
    handle = None

    def hook_fn(_, __, output):
        store[key] = output.detach()

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()
