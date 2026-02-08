from __future__ import annotations
import os, argparse
import numpy as np
import cv2
import torch

from core.config import load_config
from data.dataloader import build_busi_datasets
from prompts.visual.load_biomedclip import load_biomedclip
from prompts.visual.biomedclip_gscorecam import BiomedCLIPAdapter, GScoreCAMSaliency
from prompts.visual.visual_prompt_pipeline import VisualPromptPipeline


def overlay_box_points(img_rgb: np.ndarray, box, points, labels):
    out = img_rgb.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (x, y), lab in zip(points, labels):
        c = (0, 0, 255) if int(lab) == 1 else (255, 0, 0)
        cv2.circle(out, (int(x), int(y)), 3, c, -1)
    return out

def heatmap_on_image(img_rgb: np.ndarray, heat: np.ndarray):
    h = (heat * 255).clip(0, 255).astype(np.uint8)
    h = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img_rgb, 0.65, h, 0.35, 0)
    return out

def save_mask(path, mask01):
    m = (mask01.astype(np.uint8) * 255)
    cv2.imwrite(path, m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--datasets", default="configs/datasets.yaml")
    ap.add_argument("--out_dir", default="debug_out/visual_prompts")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--class_text", default="breast tumor")
    ap.add_argument("--target_layer", default="visual.trunk.blocks.11", help="CAM target layer path")
    ap.add_argument("--capture_layer", default=None, help="optional activation capture layer path")
    ap.add_argument("--no_crf", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.config, args.prompts, args.datasets)

    train_ds, test_ds = build_busi_datasets(cfg)
    ds = train_ds if args.split == "train" else test_ds

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load BiomedCLIP
    model, preprocess, tokenizer = load_biomedclip(device=device)
    clip_adapter = BiomedCLIPAdapter(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)

    # Saliency generator (ScoreCAM-based)
    saliency_fn = GScoreCAMSaliency(
        target_layer_path=args.target_layer,
        capture_layer=args.capture_layer,
        use_vit_reshape=True,
    )

    pipeline = VisualPromptPipeline(
        saliency_fn=saliency_fn,
        clip_adapter=clip_adapter,
        num_points=int(cfg["prompts"]["visual"]["num_points"]),
        max_components=int(cfg["prompts"]["visual"]["max_components"]),
        crf_enabled=not args.no_crf,
        crf_iters=5,
        return_artifacts=True,
    )

    n = min(args.num_samples, len(ds))
    for i in range(n):
        b = ds[i]
        img_t = b.image.unsqueeze(0).to(device)

        vp = pipeline(img_t, [args.class_text])
        art = vp.artifacts.tensors

        img = (b.image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        sal = art["saliency"][0]
        pre = art["mask_pre"][0]
        post = art["mask_post"][0]
        cc = art["mask_cc"][0]
        box = art["boxes"][0, 0]
        pts = art["points"][0]
        lbl = art["point_labels"][0]

        cv2.imwrite(os.path.join(args.out_dir, f"{i:03d}_img.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.out_dir, f"{i:03d}_saliency.png"),
                    cv2.cvtColor(heatmap_on_image(img, sal), cv2.COLOR_RGB2BGR))
        save_mask(os.path.join(args.out_dir, f"{i:03d}_mask_pre.png"), pre)
        save_mask(os.path.join(args.out_dir, f"{i:03d}_mask_post.png"), post)
        save_mask(os.path.join(args.out_dir, f"{i:03d}_mask_cc.png"), cc)

        overlay = overlay_box_points(img, box, pts, lbl)
        cv2.imwrite(os.path.join(args.out_dir, f"{i:03d}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Saved {n} samples to: {args.out_dir}")

if __name__ == "__main__":
    main()
