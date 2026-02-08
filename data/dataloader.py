from __future__ import annotations

from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader, Subset
import torch

from data.datasets.busi_dataset import BUSIDataset
from data.transforms.resize_sam_med2d import ResizeToSquare

def build_busi_datasets(cfg: Dict[str, Any]):
    ds_cfg = cfg["datasets"]["busi"]
    tr_cfg = cfg["train"]

    transform = ResizeToSquare(size=int(tr_cfg["img_size"]))

    train_ds = BUSIDataset(
        root=ds_cfg["root"],
        split=ds_cfg.get("train_split", "train"),
        image_dir=ds_cfg.get("image_dir", "images"),
        mask_dir=ds_cfg.get("mask_dir", "masks"),
        transform=transform,
        mask_foreground_threshold=int(ds_cfg.get("mask_foreground_threshold", 1)),
        dataset_name="busi",
    )

    test_ds = BUSIDataset(
        root=ds_cfg["root"],
        split=ds_cfg.get("test_split", "test"),
        image_dir=ds_cfg.get("image_dir", "images"),
        mask_dir=ds_cfg.get("mask_dir", "masks"),
        transform=transform,
        mask_foreground_threshold=int(ds_cfg.get("mask_foreground_threshold", 1)),
        dataset_name="busi",
    )

    return train_ds, test_ds

def labeled_subset(dataset, labeled_frac: float):
    n = len(dataset)
    k = max(1, int(n * labeled_frac))
    # deterministic subset (first k) — you can swap to random later
    idxs = list(range(k))
    return Subset(dataset, idxs)

def build_loaders(cfg: Dict[str, Any]):
    train_ds, test_ds = build_busi_datasets(cfg)

    labeled_frac = float(cfg["train"].get("labeled_frac", 1.0))
    train_lab = labeled_subset(train_ds, labeled_frac)

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])

    train_loader = DataLoader(train_lab, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    return train_loader, test_loader
