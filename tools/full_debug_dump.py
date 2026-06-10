from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from core.config import load_config
from torch.utils.data import DataLoader

from data.datasets.busi_dataset import BUSIDataset
from data.collate import collate_samples

from models.load_sam_med2d import load_sam_model
from models.konwer_sam2d import KonwerSAM2D

# ❌ Removed CAMEncoderFusion as it is replaced by our learnable red_mask_fuser module
from models.konwer_sam2d_fuser import KonwerSAM2DFused

from prompts.visual.gt_visual_prompts import build_visual_prompts_from_gt_masks

# CAM pipeline pieces (only used when prompt_source == "cam")
from prompts.visual.load_biomedclip import load_biomedclip
from prompts.visual.biomedclip_gscorecam import BiomedCLIPAdapter, GScoreCAMSaliency
from prompts.visual.visual_prompt_pipeline import VisualPromptPipeline

# Text pipeline imports
from prompts.text.text_prompt_pipeline import TextPromptPipeline, TextPromptConfig
from prompts.text.text_encoder import TextEncoderAdapter


# -------------------------
# Utilities: saving + stats
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_uint8_img(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu().clamp(0, 1)
    if x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    elif x.shape[0] == 1:
        x = x.squeeze(0)
    return (x.numpy() * 255.0).astype(np.uint8)


def save_rgb(path: str, hwc_u8: np.ndarray):
    ensure_dir(os.path.dirname(path))
    plt.imsave(path, hwc_u8)


def save_gray(path: str, hw_u8: np.ndarray):
    ensure_dir(os.path.dirname(path))
    plt.imsave(path, hw_u8, cmap="gray", vmin=0, vmax=255)


def overlay_mask(img_u8: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = img_u8.astype(np.float32)
    m = (mask_u8.astype(np.float32) / 255.0)
    if m.ndim == 2:
        m = m[..., None]
    ov = img.copy()
    ov[..., 0] = np.clip(ov[..., 0] + 255.0 * m[..., 0], 0, 255)
    out = (1 - alpha) * img + alpha * ov
    return out.astype(np.uint8)


def tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    x_det = x.detach()
    d: Dict[str, Any] = {
        "shape": list(x_det.shape),
        "dtype": str(x_det.dtype),
        "device": str(x_det.device),
    }
    if x_det.numel() > 0 and x_det.dtype.is_floating_point:
        d.update(
            {
                "min": float(x_det.min().cpu()),
                "max": float(x_det.max().cpu()),
                "mean": float(x_det.mean().cpu()),
                "std": float(x_det.std().cpu()),
            }
        )
    return d


def array_stats(x: np.ndarray) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
    }
    if x.size > 0 and np.issubdtype(x.dtype, np.floating):
        d.update(
            {
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
            }
        )
    elif x.size > 0 and np.issubdtype(x.dtype, np.integer):
        d.update({"min": int(np.min(x)), "max": int(np.max(x))})
    return d


def save_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_tensor_pt(path: str, t: torch.Tensor):
    ensure_dir(os.path.dirname(path))
    torch.save(t.detach().cpu(), path)


def save_array_npy(path: str, a: np.ndarray):
    ensure_dir(os.path.dirname(path))
    np.save(path, a)


def mask01_to_u8(mask01: torch.Tensor) -> np.ndarray:
    m = mask01.detach().cpu()
    if m.dim() == 3:
        m = m[0]
    m = (m > 0).numpy().astype(np.uint8) * 255
    return m


def float_hw_to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def metrics_from_masks(pred01: torch.Tensor, gt01: torch.Tensor) -> Dict[str, float]:
    p = (pred01 > 0).float()
    g = (gt01 > 0).float()
    tp = (p * g).sum().item()
    fp = (p * (1 - g)).sum().item()
    fn = ((1 - p) * g).sum().item()

    eps = 1e-7
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    return {"dice": float(dice), "iou": float(iou), "precision": float(prec), "recall": float(rec)}


def error_map_u8(pred_u8: np.ndarray, gt_u8: np.ndarray) -> np.ndarray:
    p = pred_u8 > 0
    g = gt_u8 > 0
    tp = p & g
    fp = p & (~g)
    fn = (~p) & g
    H, W = pred_u8.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[tp, 1] = 255  # Green for True Positives
    out[fp, 0] = 255  # Red for False Positives (Over-segmentation)
    out[fn, 2] = 255  # Blue for False Negatives (Under-segmentation)
    return out


def build_cam_pipeline(cfg: Dict[str, Any], device: str) -> VisualPromptPipeline:
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
        return_artifacts=True,
    )
    return vp


def _dump_branch(
    sample_dir: str,
    tag: str,
    img_u8: np.ndarray,
    gt_u8: np.ndarray,
    logits_i: torch.Tensor,
    probs_i: torch.Tensor,
    pred_i: torch.Tensor,
):
    save_tensor_pt(os.path.join(sample_dir, f"{tag}_logits.pt"), logits_i)
    save_tensor_pt(os.path.join(sample_dir, f"{tag}_probs.pt"), probs_i)
    save_tensor_pt(os.path.join(sample_dir, f"{tag}_pred.pt"), pred_i)

    pred_u8 = mask01_to_u8(pred_i)
    prob_u8 = (probs_i[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

    save_gray(os.path.join(sample_dir, f"{tag}_prob.png"), prob_u8)
    save_gray(os.path.join(sample_dir, f"{tag}_pred.png"), pred_u8)
    save_rgb(os.path.join(sample_dir, f"{tag}_pred_overlay.png"), overlay_mask(img_u8, pred_u8))

    err_rgb = error_map_u8(pred_u8, gt_u8)
    save_rgb(os.path.join(sample_dir, f"{tag}_error_map.png"), err_rgb)

    m = metrics_from_masks(pred_i, torch.from_numpy((gt_u8 > 0).astype(np.float32)).unsqueeze(0).to(pred_i.device))
    save_json(os.path.join(sample_dir, f"{tag}_metrics.json"), m)


# -------------------------
# Main
# -------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--datasets", default="configs/datasets.yaml")
    ap.add_argument("--train_cfg", default="configs/train.yaml")

    ap.add_argument("--ckpt", default=None, help="optional model weights")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--out_dir", default="debug_out/full_pipeline")
    ap.add_argument(
        "--force_prompt_source",
        default=None,
        choices=[None, "gt", "cam"],
        help="override config train.prompt_source",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, args.prompts, args.datasets, args.train_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt_source = cfg.get("train", {}).get("prompt_source", "gt")
    if args.force_prompt_source is not None:
        prompt_source = args.force_prompt_source

    fusion_cfg = cfg.get("fusion", {})
    fusion_enabled = bool(fusion_cfg.get("enabled", True))
    lambda_logits = float(fusion_cfg.get("lambda_logits", 0.5))

    ensure_dir(args.out_dir)
    save_json(
        os.path.join(args.out_dir, "run_config.json"),
        {
            "device": device,
            "prompt_source": prompt_source,
            "ckpt": args.ckpt,
            "fusion": {
                "enabled": fusion_enabled,
                "lambda_logits": lambda_logits,
                "fuser_type": "LearnableRedMaskSpatialFuser" if fusion_enabled else "None"
            },
            "sam": cfg.get("sam", {}),
            "datasets": cfg.get("datasets", {}),
            "prompts_visual": cfg.get("prompts", {}).get("visual", {}),
            "train": cfg.get("train", {}),
        },
    )

    root = cfg["datasets"]["busi"]["root"]
    image_size = int(cfg["datasets"]["busi"]["image_size"])
    ds = BUSIDataset(root=root, split=args.split, image_size=image_size)

    loader = DataLoader(ds, batch_size=args.num_samples, shuffle=True, num_workers=0, collate_fn=collate_samples)
    batch = next(iter(loader))

    images = batch.image.to(device)  # [B,3,H,W]
    masks = batch.mask.to(device)   # [B,1,H,W]
    labels = getattr(batch, "label", None)
    B, _, H, W = images.shape

    # Initialize text pipeline components
    text_cfg = TextPromptConfig(
        vqa_enabled=bool(cfg["prompts"]["text"].get("vqa_enabled", True)),
        gpt_enabled=bool(cfg["prompts"]["text"].get("gpt_enabled", False)),
        gpt_model=cfg["prompts"]["text"].get("gpt_model", "gpt-4o-mini"),
    )
    text_pipeline = TextPromptPipeline(text_cfg, device=device)
    clip_model_t, _, tokenizer_t = load_biomedclip(device=device)
    text_encoder = TextEncoderAdapter(model=clip_model_t, tokenizer=tokenizer_t, device=device)

    # Build SAM
    sam = load_sam_model(
        checkpoint_path=cfg["sam"]["checkpoint"],
        model_type=cfg["sam"]["model_type"],
        device=device,
        strict=bool(cfg["sam"].get("strict", False)),
    )

    # ✅ Fixed: Instantiating the clean, unified learnable model structure
    if fusion_enabled:
        model = KonwerSAM2DFused(sam, lambda_logits=lambda_logits).to(device).eval()
    else:
        text_fusion_mode = cfg["train"].get("text_fusion_mode", "concat")
        model = KonwerSAM2D(sam, text_encoder=text_encoder, fusion_mode=text_fusion_mode).to(device).eval()

    if args.ckpt is not None:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(sd, strict=False)
        print(f"[debug] loaded model weights: {args.ckpt}")

    # Process Prompts
    artifacts_per_sample: Optional[List[Dict[str, Any]]] = None
    class_text = cfg["prompts"]["visual"].get("class_text", "breast tumor")

    if prompt_source == "gt":
        visual_prompts = build_visual_prompts_from_gt_masks(
            masks=masks,
            num_points=int(cfg["prompts"]["visual"]["num_points"]),
            seed=int(cfg["prompts"]["visual"]["points_seed"]),
        )
    else:
        cam_pipeline = build_cam_pipeline(cfg, device=device)
        class_texts = [class_text] * B
        visual_prompts = cam_pipeline(images, class_texts)

        if hasattr(cam_pipeline, "artifacts") and cam_pipeline.artifacts is not None:
            artifacts_per_sample = cam_pipeline.artifacts

    tp_data = text_pipeline(images, labels=labels)

    # Forward routing based on chosen model variant
    if fusion_enabled:
        out = model(images, visual_prompts=visual_prompts)
        logits_base = out.baseline_mask_logits
        logits_fused = out.fused_mask_logits
        logits_comb = out.combined_mask_logits
        fusion_artifacts = out.fusion_artifacts
    else:
        out = model(images, vp=visual_prompts, tp=tp_data)
        logits_base = out.mask_logits
        logits_fused = logits_comb = fusion_artifacts = None

    prob_base = torch.sigmoid(logits_base)
    pred_base = (prob_base > 0.5).float()

    if logits_fused is not None and logits_comb is not None:
        prob_fused = torch.sigmoid(logits_fused)
        pred_fused = (prob_fused > 0.5).float()
        prob_comb = torch.sigmoid(logits_comb)
        pred_comb = (prob_comb > 0.5).float()

    # Dump per sample loops
    for i in range(B):
        fn = "unknown"
        if "filename" in batch.meta and i < len(batch.meta["filename"]):
            fn = str(batch.meta["filename"][i])

        sample_dir = os.path.join(args.out_dir, f"{args.split}_{i:02d}_{fn}")
        ensure_dir(sample_dir)

        save_tensor_pt(os.path.join(sample_dir, "raw_image.pt"), images[i])
        save_tensor_pt(os.path.join(sample_dir, "raw_gt.pt"), masks[i])

        img_u8 = to_uint8_img(images[i])
        gt_u8 = mask01_to_u8(masks[i])

        save_rgb(os.path.join(sample_dir, "00_image.png"), img_u8)
        save_gray(os.path.join(sample_dir, "01_gt.png"), gt_u8)
        save_rgb(os.path.join(sample_dir, "02_gt_overlay.png"), overlay_mask(img_u8, gt_u8))

        box_np = visual_prompts.boxes_xyxy[i].detach().cpu().numpy()
        pts_np = visual_prompts.points_xy[i].detach().cpu().numpy()
        lbl_np = visual_prompts.points_labels[i].detach().cpu().numpy()

        save_array_npy(os.path.join(sample_dir, "prompt_box.npy"), box_np)
        save_array_npy(os.path.join(sample_dir, "prompt_points.npy"), pts_np)
        save_array_npy(os.path.join(sample_dir, "prompt_point_labels.npy"), lbl_np)

        current_text_prompt = tp_data[i] if isinstance(tp_data, list) and i < len(tp_data) else str(tp_data)

        prompt_txt = "\n".join(
            [
                f"prompt_source: {prompt_source}",
                f"class_text: {class_text}",
                f"text_prompt_string: {current_text_prompt}",
                f"box_xyxy: {box_np.tolist()}",
                f"points_xy: {pts_np.tolist()}",
                f"point_labels: {lbl_np.tolist()}",
            ]
        )
        save_text(os.path.join(sample_dir, "20_prompts.txt"), prompt_txt)

        # Render overlays
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = plt.gca()
        ax.imshow(img_u8)
        ax.axis("off")
        if box_np.size >= 4:
            x1, y1, x2, y2 = box_np[0][:4]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2, color="red")
            ax.add_patch(rect)
        pos = lbl_np > 0
        neg = lbl_np == 0
        if pos.any():
            ax.scatter(pts_np[pos, 0], pts_np[pos, 1], s=30, marker="o", color="green")
        if neg.any():
            ax.scatter(pts_np[neg, 0], pts_np[neg, 1], s=30, marker="x", color="blue")
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        
        rgba_buf = fig.canvas.buffer_rgba()
        buf = np.asarray(rgba_buf)[..., :3]
        plt.close(fig)
        save_rgb(os.path.join(sample_dir, "21_prompts_overlay.png"), buf)

        # Save CAM maps
        if prompt_source == "cam":
            a = None
            if artifacts_per_sample is not None and i < len(artifacts_per_sample):
                a = artifacts_per_sample[i]

            if a is None and getattr(visual_prompts, "artifacts", None) is not None:
                tens = visual_prompts.artifacts.tensors
                a = {}
                for k in ["saliency", "mask_pre", "mask_post", "mask_cc", "boxes", "points", "point_labels"]:
                    if k in tens:
                        v = tens[k]
                        try: a[k] = v[i]
                        except Exception: a[k] = v

            if a is not None:
                if "saliency" in a and a["saliency"] is not None:
                    sal = a["saliency"]
                    if isinstance(sal, torch.Tensor):
                        sal = sal.detach().cpu().numpy()
                    save_array_npy(os.path.join(sample_dir, "cam_saliency.npy"), sal)
                    save_gray(os.path.join(sample_dir, "30_cam_saliency.png"), float_hw_to_u8(sal))

                for key, name in [
                    ("mask_pre", "31_cam_mask_pre.png"),
                    ("mask_post", "32_cam_mask_post.png"),
                    ("mask_cc", "33_cam_mask_cc.png"),
                ]:
                    if key in a and a[key] is not None:
                        m = a[key]
                        if isinstance(m, torch.Tensor):
                            m = m.detach().cpu().numpy()
                        m_u8 = (m > 0).astype(np.uint8) * 255
                        save_array_npy(os.path.join(sample_dir, f"{key}.npy"), m)
                        save_gray(os.path.join(sample_dir, name), m_u8)
                        save_rgb(os.path.join(sample_dir, name.replace(".png", "_overlay.png")), overlay_mask(img_u8, m_u8))

        # Dump per-branch targets
        _dump_branch(
            sample_dir=sample_dir,
            tag="40_base",
            img_u8=img_u8,
            gt_u8=gt_u8,
            logits_i=logits_base[i],
            probs_i=prob_base[i],
            pred_i=pred_base[i],
        )

        if fusion_enabled and logits_fused is not None:
            _dump_branch(
                sample_dir=sample_dir,
                tag="50_fused",
                img_u8=img_u8,
                gt_u8=gt_u8,
                logits_i=logits_fused[i],
                probs_i=prob_fused[i],
                pred_i=pred_fused[i],
            )
            _dump_branch(
                sample_dir=sample_dir,
                tag="60_comb",
                img_u8=img_u8,
                gt_u8=gt_u8,
                logits_i=logits_comb[i],
                probs_i=prob_comb[i],
                pred_i=pred_comb[i],
            )

            # 🚀 NEW ADDITION: Detailed feature activation analytics for your new block!
            if fusion_artifacts is not None:
                try:
                    gate = fusion_artifacts.gate[i]
                    save_tensor_pt(os.path.join(sample_dir, "70_gate.pt"), gate)
                    gate_u8 = (gate[0].detach().cpu().clamp(0, 2).numpy() / 2.0 * 255).astype(np.uint8)
                    save_gray(os.path.join(sample_dir, "70_gate.png"), gate_u8)

                    sal_e = fusion_artifacts.saliency_e[i]
                    save_tensor_pt(os.path.join(sample_dir, "71_saliency_e.pt"), sal_e)
                    sal_e_u8 = (sal_e[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                    save_gray(os.path.join(sample_dir, "71_saliency_e.png"), sal_e_u8)
                    
                    # Log mean embedding activations to see suppression effects clearly
                    fused_embed = fusion_artifacts.fused_embeddings[i]
                    base_embed = fusion_artifacts.image_embeddings[i]
                    embed_diff = (fused_embed - base_embed).abs().mean(dim=0)
                    embed_diff_np = embed_diff.detach().cpu().numpy()
                    embed_diff_u8 = ((embed_diff_np - embed_diff_np.min()) / (embed_diff_np.max() - embed_diff_np.min() + 1e-8) * 255).astype(np.uint8)
                    save_gray(os.path.join(sample_dir, "72_embedding_impact_map.png"), embed_diff_u8)
                    
                except Exception as e:
                    save_text(os.path.join(sample_dir, "70_fusion_artifacts_error.txt"), str(e))

        # Generate summary mapping dictionaries
        summary: Dict[str, Any] = {
            "filename": fn,
            "prompt_source": prompt_source,
            "class_text": class_text,
            "text_prompt_string": str(current_text_prompt),
            "image": tensor_stats(images[i]),
            "gt": tensor_stats(masks[i]),
            "prompt": {
                "box_xyxy": box_np.tolist(),
                "points_xy": pts_np.tolist(),
                "point_labels": lbl_np.tolist(),
            },
            "base": {
                "logits": tensor_stats(logits_base[i]),
                "probs": tensor_stats(prob_base[i]),
                "pred": tensor_stats(pred_base[i]),
                "metrics": metrics_from_masks(pred_base[i], masks[i]),
            },
        }

        if fusion_enabled and logits_fused is not None:
            summary["fused"] = {
                "logits": tensor_stats(logits_fused[i]),
                "probs": tensor_stats(prob_fused[i]),
                "pred": tensor_stats(pred_fused[i]),
                "metrics": metrics_from_masks(pred_fused[i], masks[i]),
            }
            summary["combined"] = {
                "logits": tensor_stats(logits_comb[i]),
                "probs": tensor_stats(prob_comb[i]),
                "pred": tensor_stats(pred_comb[i]),
                "metrics": metrics_from_masks(pred_comb[i], masks[i]),
            }

        save_json(os.path.join(sample_dir, "summary.json"), summary)

        lines = [
            f"filename: {fn}",
            f"prompt_source: {prompt_source}",
            f"class_text: {class_text}",
            f"text_prompt: {current_text_prompt}",
            "",
            "== Shapes ==",
            f"image: {tuple(images[i].shape)} dtype={images[i].dtype} device={images[i].device}",
            f"gt:    {tuple(masks[i].shape)} dtype={masks[i].dtype}",
            "",
            "== Prompt ==",
            f"box_xyxy: {box_np.tolist()}",
            f"points_xy: {pts_np.tolist()}",
            f"point_labels: {lbl_np.tolist()}",
            "",
            "== Metrics ==",
        ]
        mb = metrics_from_masks(pred_base[i], masks[i])
        lines.append(f"[base] dice={mb['dice']:.4f} iou={mb['iou']:.4f} precision={mb['precision']:.4f} recall={mb['recall']:.4f}")
        if fusion_enabled and logits_fused is not None:
            mf = metrics_from_masks(pred_fused[i], masks[i])
            mc = metrics_from_masks(pred_comb[i], masks[i])
            lines.append(f"[fused] dice={mf['dice']:.4f} iou={mf['iou']:.4f} precision={mf['precision']:.4f} recall={mf['recall']:.4f}")
            lines.append(f"[comb ] dice={mc['dice']:.4f} iou={mc['iou']:.4f} precision={mc['precision']:.4f} recall={mc['recall']:.4f}")

        save_text(os.path.join(sample_dir, "summary.txt"), "\n".join(lines))
        print("[saved]", sample_dir)

    print("\nDone. All dumps in:", args.out_dir)


if __name__ == "__main__":
    main()