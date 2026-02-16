from __future__ import annotations
from typing import Optional
import os
import torch


def load_sam_model(
    checkpoint_path: str,
    model_type: str = "vit_b",
    device: Optional[str] = None,
    strict: bool = True,
):
    """
    Supports multiple segment_anything registries:
      A) Meta SAM: builder(checkpoint=path)
      B) Forks:    builder()
      C) SAM-Med2D (this one): builder(args)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Provide a valid path in configs/train.yaml (sam.checkpoint)"
        )

    try:
        from segment_anything import sam_model_registry
    except Exception as e:
        raise ImportError(
            "segment_anything is not importable. Make sure PYTHONPATH includes SAM-Med2D repo "
            "or install Meta SAM."
        ) from e

    if model_type not in sam_model_registry:
        raise KeyError(
            f"model_type={model_type} not in sam_model_registry keys: {list(sam_model_registry.keys())}"
        )

    builder = sam_model_registry[model_type]

    # --- Case A: Meta SAM style (checkpoint kwarg) ---
    try:
        sam = builder(checkpoint=checkpoint_path)
        sam.to(device).train()
        return sam
    except TypeError:
        pass

    # We'll load weights manually after model construction
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        sd = state["model"]
    elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        sd = state["state_dict"]
    elif isinstance(state, dict):
        sd = state
    else:
        raise RuntimeError("Unsupported checkpoint format")

    def strip_any_prefix(k: str) -> str:
        for pfx in ["module.", "sam.", "model.", "net."]:
            if k.startswith(pfx):
                return k[len(pfx):]
        return k

    state_dict = {strip_any_prefix(k): v for k, v in sd.items()}
    some_keys = list(state_dict.keys())[:20]
    print("[ckpt] example keys:", some_keys)

    # --- Case B: builder() ---
    try:
        sam = builder()
    except TypeError as te:
        # --- Case C: builder(args) ---
        import inspect, re
        from types import SimpleNamespace

        # Extract "args.xxx" fields used inside builder
        src = inspect.getsource(builder)
        attrs = sorted(set(re.findall(r"args\.([A-Za-z_]\w*)", src)))

        args = SimpleNamespace()
        for a in attrs:
            # reasonable defaults
            if "ckpt" in a or "checkpoint" in a:
                setattr(args, a, None)
            elif a in ("device",):
                setattr(args, a, device)
            elif "img_size" in a or "image_size" in a:
                setattr(args, a, 256)  # SAM default
            elif "model_type" in a or "vit" in a:
                setattr(args, a, model_type)
            elif a.startswith("use_") or a.endswith("_enabled"):
                setattr(args, a, False)
            else:
                setattr(args, a, None)

        sam = builder(args)

    # manual load
    missing, unexpected = sam.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"[load_sam_model] loaded with strict=False")
        print(f"  missing keys: {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")

    sam.to(device).train()
    return sam
