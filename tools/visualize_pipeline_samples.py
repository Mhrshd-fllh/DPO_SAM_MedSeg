from __future__ import annotations

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from core.config import load_config
from data.datasets.busi_dataset import BUSIDataset
from data.collate import collate_samples
from torch.utils.data import DataLoader

from models.load_sam_med2d import load_sam_model
from models.konwer_sam2d import KonwerSAM2D

from prompts.visual.gt_visual_prompts import build_visual_prompts_from_gt_masks

# CAM visual pipeline (optional)
from prompts.visual.load_biomedclip import load_biomedclip
from prompts.visual.biomedclip_gscorecam import BiomedCLIPAdapter, GScoreCAMSaliency
from prompts.visual.visual_prompt_pipeline import VisualPromptPipeline


def _to_np_img(x: torch.Tensor) -> np.ndarray:
    """x: [3,H,W] float in [0,1] -> HWC uint8"""
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return x


def _to_np_mask01(x: torch.Tensor) -> np.ndarray:
    """x: [1,H,W] float/bool -> HW uint8(0/255)"""
    x = x.detach().cpu()
    if x.dim() == 3:
        x = x[0]
    x = (x > 0).byte().numpy() * 255
    return x


def _save_img(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, arr)


def _save_gray(path: str, arr_hw: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, arr_hw, cmap="gray", vmin=0, vmax=255)


def _overlay_mask(image_hwc: np.ndarray, mask_hw_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """mask_hw_01: 0/255"""
    img = image_hwc.astype(np.float32)
    m = (mask_hw_01.astype(np.float32) / 255.0)[..., None]
    # red overlay
    overlay = img.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 255.0 * m[..., 0], 0, 255)
    out = (1 - alpha) * img + alpha * overlay
    return out.astype(np.uint8)


def _draw_points_and_box(image_hwc: np.ndarray, boxes_xyxy: np.ndarray, points_xy: np.ndarray, points_lbl: np.ndarray):
    """
    Draw with matplotlib then return rendered RGB array.
    boxes_xyxy: [1,4] float
    points_xy: [K,2]
    points_lbl: [K] (1 pos / 0 neg)
    """
    H, W, _ = image_hwc.shape
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = plt.gca()
    ax.imshow(image_hwc)
    ax.axis("off")

    # box
    x1, y1, x2, y2 = boxes_xyxy[0]
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
    ax.add_patch(rect)

    # points
    pos = points_lbl > 0
    neg = points_lbl == 0
    if pos.any():
        ax.scatter(points_xy[pos, 0], points_xy[pos, 1], s=30, marker="o")
    if neg.any():
        ax.scatter(points_xy[neg, 0], points_xy[neg, 1], s=30, marker="x")

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    buf = buf[:, :, 1:]
    plt.close(fig)

    # ensure same size (matplotlib can slightly differ); resize if needed
    if buf.shape[:2] != (H, W):
        # nearest resize
        import cv2
        buf = cv2.resize(buf, (W, H), interpolation=cv2.INTER_NEAREST)
    return buf


def build_cam_pipeline(cfg, device: str) -> VisualPromptPipeline:
    clip_model, preprocess, tokenizer = load_biomedclip(device=device)
    clip_adapter = BiomedCLIPAdapter(model=clip_model, preprocess=preprocess, tokenizer=tokenizer, device=device)

    saliency_fn = GScoreCAMSaliency(
        target_layer_path=cfg["prompts"]["visual"]["cam"]["target_layer"],
        capture_layer=None,
        use_vit_reshape=bool(cfg["prompts"]["visual"]["cam"]["use_vit_reshape"]),
    )

    vp = VisualPromptPipeline(
        saliency_fn=saliency_fn,
        clip_adapter=clip_adapter,
        num_points=int(cfg["prompts"]["visual"]["num_points"]),
        max_components=int(cfg["prompts"]["visual"]["max_components"]),
        crf_enabled=bool(cfg["prompts"]["visual"]["crf"]["enabled"]),
        crf_iters=int(cfg["prompts"]["visual"]["crf"]["iters"]),
        points_seed=int(cfg["prompts"]["visual"]["points_seed"]),
        saliency_threshold=float(cfg["prompts"]["visual"]["saliency_threshold"]),
        return_artifacts=True,   # ✅ مهم برای ذخیره مراحل
    )
    return vp


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--datasets", default="configs/datasets.yaml")
    ap.add_argument("--train_cfg", default="configs/train.yaml")

    ap.add_argument("--ckpt", default=None, help="path to runs/stage1/best.pt or last.pt (optional)")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--num_samples", type=int, default=6)
    ap.add_argument("--out_dir", default="debug_out/vis_samples")
    args = ap.parse_args()

    cfg = load_config(args.config, args.prompts, args.datasets, args.train_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    root = cfg["datasets"]["busi"]["root"]
    image_size = int(cfg["datasets"]["busi"]["image_size"])
    ds = BUSIDataset(root=root, split=args.split, image_size=image_size)

    # loader (batch = num_samples)
    loader = DataLoader(ds, batch_size=args.num_samples, shuffle=True, num_workers=0, collate_fn=collate_samples)
    batch = next(iter(loader))

    images = batch.image.to(device)  # [B,3,H,W]
    gt = batch.mask.to(device)       # [B,1,H,W]
    B, _, H, W = images.shape

    # build SAM model
    sam = load_sam_model(
        checkpoint_path=cfg["sam"]["checkpoint"],
        model_type=cfg["sam"]["model_type"],
        device=device,
        strict=bool(cfg["sam"].get("strict", False)),
    )
    model = KonwerSAM2D(sam).to(device).eval()

    # optionally load trained weights (best.pt/last.pt)
    if args.ckpt is not None:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "model" in ck:
            model.load_state_dict(ck["model"], strict=False)
        else:
            model.load_state_dict(ck, strict=False)
        print(f"[viz] loaded model weights from: {args.ckpt}")

    prompt_source = cfg["train"].get("prompt_source", "gt")
    class_text = cfg["prompts"]["visual"].get("class_text", "breast tumor")

    cam_pipeline = None
    if prompt_source == "cam":
        cam_pipeline = build_cam_pipeline(cfg, device=device)

    os.makedirs(args.out_dir, exist_ok=True)

    # run
    if prompt_source == "gt":
        vp = build_visual_prompts_from_gt_masks(
            masks=gt,
            num_points=int(cfg["prompts"]["visual"]["num_points"]),
            seed=int(cfg["prompts"]["visual"]["points_seed"]),
        )
        artifacts = None
    else:
        class_texts = [class_text] * B
        vp = cam_pipeline(images, class_texts)
        artifacts = vp.artifacts  # defined when return_artifacts=True

    out = model(images, vp)
    logits = out.mask_logits                         # [B,1,H,W]
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()

    for i in range(B):
        sample_dir = os.path.join(args.out_dir, f"{args.split}_{i:02d}_{batch.meta['filename'][i]}")
        os.makedirs(sample_dir, exist_ok=True)

        img_np = _to_np_img(images[i])
        gt_np = _to_np_mask01(gt[i])
        pred_np = _to_np_mask01(pred[i])

        # 0) input & GT
        _save_img(os.path.join(sample_dir, "00_image.png"), img_np)
        _save_gray(os.path.join(sample_dir, "01_gt.png"), gt_np)
        _save_img(os.path.join(sample_dir, "02_gt_overlay.png"), _overlay_mask(img_np, gt_np))

        # 1) prompts visualization (box+points)
        box = vp.boxes_xyxy[i].detach().cpu().numpy()          # [1,4]
        pts = vp.points_xy[i].detach().cpu().numpy()           # [K,2]
        lbl = vp.points_labels[i].detach().cpu().numpy().astype(np.int32)  # [K]
        prompt_vis = _draw_points_and_box(img_np, box, pts, lbl)
        _save_img(os.path.join(sample_dir, "03_prompts_box_points.png"), prompt_vis)

        # 2) pred
        _save_gray(os.path.join(sample_dir, "10_pred.png"), pred_np)
        _save_img(os.path.join(sample_dir, "11_pred_overlay.png"), _overlay_mask(img_np, pred_np))

        # 3) prob heatmap
        prob_np = (probs[i, 0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        _save_gray(os.path.join(sample_dir, "12_prob.png"), prob_np)

        # 4) CAM artifacts (if available)
        if artifacts is not None:
            # artifacts typically include: saliency, pre_mask, post_mask, overlay, etc.
            # We'll save what exists.
            a = artifacts[i]

            # Save saliency if provided
            if "saliency" in a and a["saliency"] is not None:
                sal = a["saliency"]
                if isinstance(sal, torch.Tensor):
                    sal = sal.detach().cpu().numpy()
                # sal expected [H,W] float 0..1
                sal_u8 = (np.clip(sal, 0, 1) * 255).astype(np.uint8)
                _save_gray(os.path.join(sample_dir, "20_cam_saliency.png"), sal_u8)

            # pre/post masks (binary or 0..1)
            for key, out_name in [
                ("pre_mask", "21_cam_pre_mask.png"),
                ("post_mask", "22_cam_post_mask.png"),
                ("cc_mask", "23_cam_cc_mask.png"),
            ]:
                if key in a and a[key] is not None:
                    m = a[key]
                    if isinstance(m, torch.Tensor):
                        m = m.detach().cpu().numpy()
                    # handle 0..1 float or bool
                    m_u8 = (m > 0).astype(np.uint8) * 255 if m.max() <= 1.0 else (m > 0).astype(np.uint8) * 255
                    _save_gray(os.path.join(sample_dir, out_name), m_u8)

            # overlay if provided (already RGB)
            if "overlay" in a and a["overlay"] is not None:
                ov = a["overlay"]
                if isinstance(ov, torch.Tensor):
                    ov = ov.detach().cpu().numpy()
                # expect HWC uint8 or float
                if ov.dtype != np.uint8:
                    ov = (np.clip(ov, 0, 1) * 255).astype(np.uint8)
                _save_img(os.path.join(sample_dir, "24_cam_overlay.png"), ov)

        print(f"[saved] {sample_dir}")

    print(f"\nDone. Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
