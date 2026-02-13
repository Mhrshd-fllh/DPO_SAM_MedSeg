from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass(frozen=True)
class Sample:
    image: torch.Tensor      # [3,H,W] float32 in [0,1]
    mask: torch.Tensor       # [1,H,W] float32 in {0,1}
    meta: Dict


class BUSIDataset(Dataset):
    """
    Expected structure:
      data/BUSI/{split}/images/*.png
      data/BUSI/{split}/masks/*.png   (same filenames as images)
    """
    def __init__(self, root: str, split: str = "train", image_size: int = 256):
        self.root = root
        self.split = split
        self.image_size = image_size

        img_dir = os.path.join(root, split, "images")
        msk_dir = os.path.join(root, split, "masks")

        self.img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        assert len(self.img_paths) > 0, f"No images found in {img_dir}"

        # masks are expected to match filenames
        self.msk_dir = msk_dir

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_png(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: str) -> Image.Image:
        # masks are binary-ish png; keep as L
        return Image.open(path).convert("L")

    def __getitem__(self, idx: int) -> Sample:
        ip = self.img_paths[idx]
        fn = os.path.basename(ip)
        mp = os.path.join(self.msk_dir, fn)
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Mask not found for image {fn}: expected {mp}")

        img = self._load_png(ip).resize((self.image_size, self.image_size))
        msk = self._load_mask(mp).resize((self.image_size, self.image_size))

        img_np = np.array(img, dtype=np.uint8)                  # HWC
        msk_np = np.array(msk, dtype=np.uint8)                  # HW

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # [3,H,W]
        msk_t = (torch.from_numpy(msk_np) > 0).float().unsqueeze(0)        # [1,H,W]

        meta = {"filename": fn}
        return Sample(image=img_t, mask=msk_t, meta=meta)
