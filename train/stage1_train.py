from __future__ import annotations
import os
import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from core.config import load_config
from data.dataloader import build_busi_loaders
from models.load_sam_med2d import load_sam_model
from models.konwer_sam2d import KonwerSAM2D
from losses.combo import DiceFocalCombo
from eval.metrics import dice_coeff

from prompts.visual.load_biomedclip import load_biomedclip
from prompts.visual.biomedclip_gscorecam import BiomedCLIPAdapter, GScoreCAMSaliency
from prompts.visual.visual_prompt_pipeline import VisualPromptPipeline
from prompts.visual.gt_visual_prompts import build_visual_prompts_from_gt_masks


def freeze_image_encoder_if_needed(model: KonwerSAM2D, freeze: bool):
    if not freeze:
        return
    for p in model.sam.image_encoder.parameters():
        p.requires_grad = False


def build_cam_visual_pipeline(cfg, device: str) -> VisualPromptPipeline:
    # BiomedCLIP
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
        return_artifacts=False,   # training: no need artifacts
    )
    return vp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--datasets", default="configs/datasets.yaml")
    ap.add_argument("--train_cfg", default="configs/train.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config, args.prompts, args.datasets, args.train_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = build_busi_loaders(cfg)

    # SAM-Med2D / SAM
    sam = load_sam_model(
        checkpoint_path=cfg["sam"]["checkpoint"],
        model_type=cfg["sam"]["model_type"],
        device=device,
        strict=bool(cfg["sam"].get("strict", True)),
    )

    model = KonwerSAM2D(sam).to(device)
    freeze_image_encoder_if_needed(model, bool(cfg["train"]["freeze_image_encoder"]))

    # visual prompt source
    prompt_source = cfg["train"]["prompt_source"]  # "cam" or "gt"
    if prompt_source not in ("cam", "gt"):
        raise ValueError("train.prompt_source must be 'cam' or 'gt'")

    cam_pipeline = None
    if prompt_source == "cam":
        cam_pipeline = build_cam_visual_pipeline(cfg, device=device)

    crit = DiceFocalCombo(
        dice_w=float(cfg["train"]["loss"]["dice_w"]),
        focal_w=float(cfg["train"]["loss"]["focal_w"]),
    )

    opt = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    sch = StepLR(opt, step_size=int(cfg["train"]["lr_step"]), gamma=float(cfg["train"]["lr_gamma"]))

    epochs = int(cfg["train"]["epochs"])
    out_dir = cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    class_text = cfg["prompts"]["visual"]["class_text"]
    best_dice = -1.0

    for ep in range(1, epochs + 1):
        # -------- train --------
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            images = batch.image.to(device)
            masks = batch.mask.to(device)

            if prompt_source == "gt":
                vp = build_visual_prompts_from_gt_masks(
                    masks=masks,
                    num_points=int(cfg["prompts"]["visual"]["num_points"]),
                    seed=int(cfg["prompts"]["visual"]["points_seed"]),
                )
            else:
                class_texts = [class_text] * images.shape[0]
                vp = cam_pipeline(images, class_texts)

            out = model(images, vp)
            loss = crit(out.mask_logits, masks)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        sch.step()

        # -------- eval --------
        model.eval()
        dices = []
        with torch.no_grad():
            for batch in test_loader:
                images = batch.image.to(device)
                masks = batch.mask.to(device)

                if prompt_source == "gt":
                    vp = build_visual_prompts_from_gt_masks(
                        masks=masks,
                        num_points=int(cfg["prompts"]["visual"]["num_points"]),
                        seed=int(cfg["prompts"]["visual"]["points_seed"]),
                    )
                else:
                    class_texts = [class_text] * images.shape[0]
                    vp = cam_pipeline(images, class_texts)

                out = model(images, vp)
                d = dice_coeff(out.mask_logits, masks).item()
                dices.append(d)

        mean_dice = sum(dices) / max(1, len(dices))
        mean_loss = total_loss / max(1, len(train_loader))

        print(f"[Epoch {ep:02d}] loss={mean_loss:.4f}  val_dice={mean_dice:.4f}")

        # checkpoint
        ckpt = {
            "epoch": ep,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "cfg": cfg,
            "val_dice": mean_dice,
        }
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))

    print(f"Done. Best val_dice={best_dice:.4f}. Checkpoints in: {out_dir}")


if __name__ == "__main__":
    main()
