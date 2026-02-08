from __future__ import annotations

import os
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core.types import Batch

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _list_images(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return files

class BUSIDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_dir: str = "images",
        mask_dir: str = "masks",
        transform=None,
        mask_foreground_threshold: int = 1,
        dataset_name: str = "busi",
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.mask_foreground_threshold = mask_foreground_threshold
        self.dataset_name = dataset_name

        self.img_dir = os.path.join(root, split, image_dir)
        self.msk_dir = os.path.join(root, split, mask_dir)

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(self.img_dir)
        if not os.path.isdir(self.msk_dir):
            raise FileNotFoundError(self.msk_dir)

        self.ids = _list_images(self.img_dir)
        if len(self.ids) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        # sanity check a couple pairs
        for fn in self.ids[:3]:
            mp = os.path.join(self.msk_dir, fn)
            if not os.path.exists(mp):
                raise FileNotFoundError(f"Mask not found for {fn}: {mp}")

    def __len__(self):
        return len(self.ids)

    def _read_rgb(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask(self, path: str) -> np.ndarray:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(path)
        # binarize
        m = (m >= self.mask_foreground_threshold).astype(np.uint8)
        return m

    def __getitem__(self, idx: int) -> Batch:
        fn = self.ids[idx]
        img_path = os.path.join(self.img_dir, fn)
        msk_path = os.path.join(self.msk_dir, fn)

        image = self._read_rgb(img_path)  # HWC uint8 RGB
        mask = self._read_mask(msk_path)  # HW uint8 {0,1}

        meta: Dict[str, Any] = {
            "id": fn,
            "dataset": self.dataset_name,
            "split": self.split,
            "image_path": img_path,
            "mask_path": msk_path,
        }

        if self.transform is not None:
            image, mask, meta = self.transform(image, mask, meta)

        # to tensors
        image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)  # [3,H,W]
        mask_t = torch.from_numpy(mask.astype(np.float32))[None, ...]                  # [1,H,W]

        return Batch(image=image_t, mask=mask_t, meta=meta)
