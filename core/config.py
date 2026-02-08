from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    default_path: str,
    prompts_path: Optional[str] = None,
    datasets_path: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_yaml(default_path)
    if prompts_path:
        cfg = deep_update(cfg, {"prompts": load_yaml(prompts_path)})
    if datasets_path:
        cfg = deep_update(cfg, {"datasets": load_yaml(datasets_path)})
    return cfg
