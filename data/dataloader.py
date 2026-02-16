from __future__ import annotations
from typing import Tuple
import torch
from torch.utils.data import DataLoader

from data.datasets.busi_dataset import BUSIDataset
from data.collate import collate_samples

def build_busi_loaders(cfg) -> Tuple[DataLoader, DataLoader]:
    root = cfg["datasets"]["busi"]["root"]
    image_size = int(cfg["datasets"]["busi"]["image_size"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])

    train_ds = BUSIDataset(root=root, split="train", image_size=image_size)
    test_ds = BUSIDataset(root=root, split="test", image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_samples,   # ✅
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_samples,   # ✅
    )
    return train_loader, test_loader
