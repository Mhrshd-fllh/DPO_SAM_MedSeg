from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import cv2
import numpy as np

@dataclass
class ResizeToSquare:
    size: int = 256

    def __call__(self, image: np.ndarray, mask: np.ndarray | None, meta: Dict[str, Any]):
        """
        image: HxWxC RGB uint8 or float32
        mask:  HxW   uint8/float or None
        meta:  dict (will be updated with original size)
        """
        h, w = image.shape[:2]
        meta = dict(meta)
        meta["orig_size"] = (h, w)
        meta["resized_size"] = (self.size, self.size)

        img = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            m = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        else:
            m = None
        return img, m, meta
