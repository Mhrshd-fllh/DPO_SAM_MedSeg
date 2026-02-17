from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from core.config import load_config
from data.datasets.busi_dataset import BUSIDataset
from data.collate import collate_samples
from torch.utils.data import DataLoader

from prompts.visual.load_biomedclip import load_biomedclip
from prompts.visual.biomedclip_gscorecam import BiomedCLIPAdapter, GScoreCAMSaliency
from prompts.visual.visual_prompt_pipeline import VisualPromptPipeline


def save_gray(path: str, hw_u8: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, hw_u8, cmap="gray", vmin=0, vmax=255)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--datasets", default="configs/datasets.yaml")
    ap.add_argument("--train_cfg", default="configs/train.yaml")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--out_dir", default="debug_out/cam_only")
    args = ap.parse_args()

    cfg = load_config(args.config, args.prompts, args.datasets, args.train_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    root = cfg["datasets"]["busi"]["root"]
    image_size = int(cfg["datasets"]["busi"]["image_size"])
    ds = BUSIDataset(root=root, split=args.split, image_size=image_size)
    loader = DataLoader(ds, batch_size=args.num_samples, shuffle=True, num_workers=0, collate_fn=collate_samples)
    batch = next(iter(loader))
    images = batch.image.to(device)
    B, _, H, W = images.shape

    # build CAM pipeline
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

    class_text = cfg["prompts"]["visual"].get("class_text", "breast tumor")
    class_texts = [class_text] * B

    _ = vp(images, class_texts)

    # save saliency + masks
    for i in range(B):
        fn = batch.meta["filename"][i]
        sd = os.path.join(args.out_dir, f"{args.split}_{i:02d}_{fn}")
        os.makedirs(sd, exist_ok=True)

        a = vp.artifacts[i]
        sal = a["saliency"]
        if isinstance(sal, torch.Tensor):
            sal = sal.detach().cpu().numpy()
        sal_u8 = (np.clip(sal, 0, 1) * 255).astype(np.uint8)
        save_gray(os.path.join(sd, "saliency.png"), sal_u8)

        for k in ["mask_pre", "mask_post", "mask_cc"]:
            if k in a and a[k] is not None:
                m = a[k]
                if isinstance(m, torch.Tensor):
                    m = m.detach().cpu().numpy()
                m_u8 = (m > 0).astype(np.uint8) * 255
                save_gray(os.path.join(sd, f"{k}.png"), m_u8)

        print("[saved]", sd)

    print("Done:", args.out_dir)


if __name__ == "__main__":
    main()
