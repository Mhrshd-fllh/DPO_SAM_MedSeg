from __future__ import annotations
import yaml
from typing import Dict, Any


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update dict `base` with `new`.
    """
    for k, v in new.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(*yaml_paths: str) -> Dict[str, Any]:
    """
    Load and merge multiple YAML config files.
    Later files override earlier ones.

    Example:
        load_config("default.yaml", "prompts.yaml", "datasets.yaml", "train.yaml")
    """
    cfg: Dict[str, Any] = {}

    for path in yaml_paths:
        if path is None:
            continue
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            if data is not None:
                cfg = _deep_update(cfg, data)

    return cfg
