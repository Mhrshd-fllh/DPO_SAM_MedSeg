from __future__ import annotations

from typing import Callable, Dict, Any


class Registry:
    def __init__(self):
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def deco(fn: Callable[..., Any]):
            if name in self._items:
                raise KeyError(f"Duplicate registry key: {name}")
            self._items[name] = fn
            return fn
        return deco

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise KeyError(f"Unknown registry key: {name}. Available: {list(self._items.keys())}")
        return self._items[name]

    def keys(self):
        return list(self._items.keys())


PROMPTS_REGISTRY = Registry()
MODELS_REGISTRY = Registry()
DATASETS_REGISTRY = Registry()
