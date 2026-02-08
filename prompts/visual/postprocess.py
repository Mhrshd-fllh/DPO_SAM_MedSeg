from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def keep_largest_components(mask: np.ndarray, max_components: int = 1) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    keep_ids = order[:max_components] + 1

    out = np.zeros_like(m, dtype=np.uint8)
    for cid in keep_ids:
        out[labels == cid] = 1
    return out


def mask_to_box_xyxy(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([0, 0, 1, 1], dtype=np.float32)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def sample_points_from_mask(mask: np.ndarray, k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        pts = np.zeros((k, 2), dtype=np.float32)
        lbl = np.zeros((k,), dtype=np.int64)
        return pts, lbl

    idx = rng.integers(0, len(xs), size=k)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    lbl = np.ones((k,), dtype=np.int64)
    return pts, lbl


def batch_mask_to_boxes(masks: np.ndarray) -> np.ndarray:
    boxes = []
    for i in range(masks.shape[0]):
        boxes.append(mask_to_box_xyxy(masks[i]))
    boxes = np.stack(boxes, axis=0)[:, None, :]  # [B,1,4]
    return boxes.astype(np.float32)


def batch_sample_points(masks: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pts_list, lbl_list = [], []
    for i in range(masks.shape[0]):
        pts, lbl = sample_points_from_mask(masks[i], k=k, rng=rng)
        pts_list.append(pts)
        lbl_list.append(lbl)
    return np.stack(pts_list, axis=0), np.stack(lbl_list, axis=0)
