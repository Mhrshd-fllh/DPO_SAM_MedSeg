from __future__ import annotations
import os
import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from core.config import load_config
from data.dataloader import build_busi_loaders
from models.load_sam_med2d import load_sam_model
from models.konwer_sam2d import KonwerSAM2D
from models.konwer_sam2d_fuser import KonwerSAM2DFused           # Added imports
from models.fusion_cam_encoder import CAMEncoderFusion         # Added imports
from losses.combo import DiceFocalCombo
from eval.metrics import dice_coeff

from prompts.visual.load_biomedclip import load_biomedclip
from prompts.visual.biomedclip_gscorecam import BiomedCLIPAdapter, GScoreCAMSaliency
from prompts.visual.visual_prompt_pipeline import VisualPromptPipeline
from prompts.visual.gt_visual_prompts import build_visual_prompts_from_gt_masks
from prompts.text.text_prompt_pipeline import TextPromptPipeline, TextPromptConfig
from prompts.text.text_encoder import TextEncoderAdapter

def freeze_image_encoder_if_needed(model: torch.nn.Module, freeze: bool):
    if not freeze:
        return
    # Support both wrapped configurations
    target_model = model.sam if hasattr(model, "sam") else model
    for p in target_model.image_encoder.parameters():
        p.requires_grad = False


def build_cam_visual_pipeline(cfg, device: str) -> VisualPromptPipeline:
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
        return_artifacts=True,   # 🔥 Set to True so artifacts (saliency maps) are passed down the pipeline!
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

    # Initialize text prompt pipeline
    text_cfg = TextPromptConfig(
        question_template=cfg["prompts"]["text"].get("question_template", "Question: {}, Answer is:"),
        question=cfg["prompts"]["text"].get(
            "question",
            "What is the shape of breast tumor and where is it located?",
        ),
        vqa_enabled=bool(cfg["prompts"]["text"].get("vqa_enabled", True)),
        gpt_enabled=bool(cfg["prompts"]["text"].get("gpt_enabled", False)),
        vqa_model_id=cfg["prompts"]["text"].get("vqa_model_id", "Salesforce/blip-vqa-base"),
        gpt_model=cfg["prompts"]["text"].get("gpt_model", "gpt-4o-mini"),
    )
    text_pipeline = TextPromptPipeline(text_cfg, device=device)
    gpt_descriptions_by_label = text_pipeline.precompute_gpt_descriptions(
        list(getattr(train_loader.dataset, "labels", []))
        + list(getattr(test_loader.dataset, "labels", []))
    )

    # Load BiomedCLIP for text encoding
    clip_model, _, tokenizer = load_biomedclip(device=device)
    text_encoder = TextEncoderAdapter(model=clip_model, tokenizer=tokenizer, device=device)

    # SAM-Med2D / SAM
    sam = load_sam_model(
        checkpoint_path=cfg["sam"]["checkpoint"],
        model_type=cfg["sam"]["model_type"],
        device=device,
        strict=bool(cfg["sam"].get("strict", True)),
    )

    # 🔥 DYNAMIC SWITCH: Choose model architecture variant based on train.yaml switches
    use_visual_fuser = cfg.get("fusion", {}).get("enabled", False)
    
    if use_visual_fuser:
        print("Initializing visual gating branch: [KonwerSAM2DFused]")
        fusion_module = CAMEncoderFusion(
            mode=cfg["fusion"].get("mode", "residual_mul"),
            alpha=float(cfg["fusion"].get("alpha", 1.0)),
        )
        model = KonwerSAM2DFused(
            sam=sam,
            fusion=fusion_module,
            lambda_logits=float(cfg["fusion"].get("lambda_logits", 0.5))
        ).to(device)
    else:
        print("Initializing text baseline branch: [KonwerSAM2D]")
        fusion_mode = cfg["train"].get("text_fusion_mode", "concat")
        model = KonwerSAM2D(
            sam, 
            text_encoder=text_encoder,
            fusion_mode=fusion_mode
        ).to(device)

    freeze_image_encoder_if_needed(model, bool(cfg["train"]["freeze_image_encoder"]))

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

        for batch in tqdm(train_loader, desc=f"Epoch {ep:02d} [Train]", leave=False):
            images = batch.image.to(device)
            masks = batch.mask.to(device)
            labels = batch.label

            if prompt_source == "gt":
                vp = build_visual_prompts_from_gt_masks(
                    masks=masks,
                    num_points=int(cfg["prompts"]["visual"]["num_points"]),
                    seed=int(cfg["prompts"]["visual"]["points_seed"]),
                )
            else:
                class_texts = [class_text] * images.shape[0]
                vp = cam_pipeline(images, class_texts)

            # Generate text prompts
            tp = text_pipeline(
                images,
                labels=labels,
                gpt_descriptions_by_label=gpt_descriptions_by_label,
            )

            opt.zero_grad(set_to_none=True)

            # 🔥 DYNAMIC FORWARD & LOSS COMPUTATION BLOCK
            if use_visual_fuser:
                # KonwerSAM2DFused pass
                out = model(images, visual_prompts=vp)
                # Compute loss directly on the combined prediction map
                loss = crit(out.combined_mask_logits, masks)
                pred_logits = out.combined_mask_logits
            else:
                # KonwerSAM2D text variant pass
                out = model(images, vp=vp, tp=tp)
                loss = crit(out.mask_logits, masks)
                pred_logits = out.mask_logits

            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        sch.step()

        # -------- eval --------
        model.eval()
        dices = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {ep:02d} [Eval]", leave=False):
                images = batch.image.to(device)
                masks = batch.mask.to(device)
                labels = batch.label

                if prompt_source == "gt":
                    vp = build_visual_prompts_from_gt_masks(
                        masks=masks,
                        num_points=int(cfg["prompts"]["visual"]["num_points"]),
                        seed=int(cfg["prompts"]["visual"]["points_seed"]),
                    )
                else:
                    class_texts = [class_text] * images.shape[0]
                    vp = cam_pipeline(images, class_texts)

                tp = text_pipeline(
                    images,
                    labels=labels,
                    gpt_descriptions_by_label=gpt_descriptions_by_label,
                )

                if use_visual_fuser:
                    out = model(images, visual_prompts=vp)
                    pred_logits = out.combined_mask_logits
                else:
                    out = model(images, vp=vp, tp=tp)
                    pred_logits = out.mask_logits

                d = dice_coeff(pred_logits, masks).item()
                dices.append(d)

        mean_dice = sum(dices) / max(1, len(dices))
        mean_loss = total_loss / max(1, len(train_loader))

        print(f"[Epoch {ep:02d}] loss={mean_loss:.4f}  val_dice={mean_dice:.4f}")

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
