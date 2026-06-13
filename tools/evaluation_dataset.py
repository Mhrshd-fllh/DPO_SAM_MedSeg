from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import load_config
from data.datasets.busi_dataset import BUSIDataset
from data.collate import collate_samples

from models.load_sam_med2d import load_sam_model
from models.konwer_sam2d import KonwerSAM2D

# Fused model components
from models.fusion_cam_encoder import CAMEncoderFusion
from models.konwer_sam2d_fuser import KonwerSAM2DFused

from prompts.visual.gt_visual_prompts import build_visual_prompts_from_gt_masks

# CAM / Text pipeline components
from prompts.visual.load_biomedclip import load_biomedclip
from prompts.visual.biomedclip_gscorecam import BiomedCLIPAdapter, GScoreCAMSaliency
from prompts.visual.visual_prompt_pipeline import VisualPromptPipeline
from prompts.text.text_prompt_pipeline import TextPromptPipeline, TextPromptConfig
from prompts.text.text_encoder import TextEncoderAdapter


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def compute_batch_metrics(pred01: torch.Tensor, gt01: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Computes true positives, false positives, and false negatives per sample in the batch
    to allow exact dataset-wide aggregated metric calculation.
    """
    p = (pred01 > 0).float()
    g = (gt01 > 0).float()
    
    # Sum over spatial dimensions [H, W] to keep per-sample counts
    tp = (p * g).sum(dim=(-2, -1))
    fp = (p * (1 - g)).sum(dim=(-2, -1))
    fn = ((1 - p) * g).sum(dim=(-2, -1))
    
    return {"tp": tp, "fp": fp, "fn": fn}


def calculate_final_metrics(tp_list: List[float], fp_list: List[float], fn_list: List[float]) -> Dict[str, float]:
    """
    Computes the final precision, recall, dice, and IoU scores over the entire dataset split.
    """
    total_tp = sum(tp_list)
    total_fp = sum(fp_list)
    total_fn = sum(fn_list)
    
    eps = 1e-7
    dice = (2 * total_tp + eps) / (2 * total_tp + total_fp + total_fn + eps)
    iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    precision = (total_tp + eps) / (total_tp + total_fp + eps)
    recall = (total_tp + eps) / (total_tp + total_fn + eps)
    
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall)
    }


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
        return_artifacts=False,  # Turned off for speed during evaluation
    )
    return vp


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--datasets", default="configs/datasets.yaml")
    ap.add_argument("--train_cfg", default="configs/train.yaml")
    ap.add_argument("--ckpt", default=None, help="Path to model weights checkpoint")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_dir", default="evaluation_results")
    ap.add_argument("--out_name", default="test_metrics", help="Base name for txt/json output files")
    args = ap.parse_args()

    cfg = load_config(args.config, args.prompts, args.datasets, args.train_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt_source = cfg.get("train", {}).get("prompt_source", "gt")
    fusion_cfg = cfg.get("fusion", {})
    fusion_enabled = bool(fusion_cfg.get("enabled", True))
    fusion_mode = str(fusion_cfg.get("mode", "residual_mul"))
    fusion_alpha = float(fusion_cfg.get("alpha", 1.0))
    fusion_beta = float(fusion_cfg.get("beta", 0.5))
    lambda_logits = float(fusion_cfg.get("lambda_logits", 0.5))

    # Setup Dataset
    root = cfg["datasets"]["busi"]["root"]
    image_size = int(cfg["datasets"]["busi"]["image_size"])
    ds = BUSIDataset(root=root, split=args.split, image_size=image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_samples)

    # Initialize Prompt Pipelines
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
        getattr(ds, "labels", [])
    )
    clip_model_t, _, tokenizer_t = load_biomedclip(device=device)
    text_encoder = TextEncoderAdapter(model=clip_model_t, tokenizer=tokenizer_t, device=device)

    if prompt_source == "cam":
        cam_pipeline = build_cam_pipeline(cfg, device=device)
    else:
        cam_pipeline = None

    # Load SAM Model
    sam = load_sam_model(
        checkpoint_path=cfg["sam"]["checkpoint"],
        model_type=cfg["sam"]["model_type"],
        device=device,
        strict=bool(cfg["sam"].get("strict", False)),
    )

    # Load architecture variant block
    if fusion_enabled:
        fusion_mod = CAMEncoderFusion(mode=fusion_mode, alpha=fusion_alpha, beta=fusion_beta)
        model = KonwerSAM2DFused(sam, fusion=fusion_mod, lambda_logits=lambda_logits).to(device).eval()
    else:
        text_fusion_mode = cfg["train"].get("text_fusion_mode", "concat")
        model = KonwerSAM2D(sam, text_encoder=text_encoder, fusion_mode=text_fusion_mode).to(device).eval()

    if args.ckpt is not None:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(sd, strict=False)
        print(f"[Eval] Successfully loaded checkpoint: {args.ckpt}")
    else:
        print("[Eval] Warning: Evaluating using uncalibrated baseline initialization (no --ckpt specified).")

    # Storage lists for metric tracking arrays
    counts = {
        "base": {"tp": [], "fp": [], "fn": []},
        "fused": {"tp": [], "fp": [], "fn": []},
        "comb": {"tp": [], "fp": [], "fn": []}
    }

    class_text = cfg["prompts"]["visual"].get("class_text", "breast tumor")

    print(f"\n[Eval] Commencing evaluation over {len(ds)} items (Split: {args.split})...")
    for batch in tqdm(loader, desc="Evaluating Dataset"):
        images = batch.image.to(device)  # [B, 3, H, W]
        masks = batch.mask.to(device)   # [B, 1, H, W]
        labels = batch.label
        B = images.shape[0]

        # Resolve Visual Prompts
        if prompt_source == "gt":
            visual_prompts = build_visual_prompts_from_gt_masks(
                masks=masks,
                num_points=int(cfg["prompts"]["visual"]["num_points"]),
                seed=int(cfg["prompts"]["visual"]["points_seed"]),
            )
        else:
            class_texts = [class_text] * B
            visual_prompts = cam_pipeline(images, class_texts)

        # Resolve Text Prompts
        tp_data = text_pipeline(
            images,
            labels=labels,
            gpt_descriptions_by_label=gpt_descriptions_by_label,
        )

        # Model Routing & Forward Execution
        if fusion_enabled:
            out = model(images, visual_prompts=visual_prompts)
            logits_base = out.baseline_mask_logits
            logits_fused = out.fused_mask_logits
            logits_comb = out.combined_mask_logits
        else:
            out = model(images, vp=visual_prompts, tp=tp_data)
            logits_base = out.mask_logits
            logits_fused = logits_comb = None

        # Base Variant Metrics Calculation
        pred_base = (torch.sigmoid(logits_base) > 0.5).float()
        m_base = compute_batch_metrics(pred_base, masks)
        counts["base"]["tp"].extend(m_base["tp"].cpu().tolist())
        counts["base"]["fp"].extend(m_base["fp"].cpu().tolist())
        counts["base"]["fn"].extend(m_base["fn"].cpu().tolist())

        # Fused & Combined metrics computation if enabled
        if fusion_enabled and logits_fused is not None:
            pred_fused = (torch.sigmoid(logits_fused) > 0.5).float()
            m_fused = compute_batch_metrics(pred_fused, masks)
            counts["fused"]["tp"].extend(m_fused["tp"].cpu().tolist())
            counts["fused"]["fp"].extend(m_fused["fp"].cpu().tolist())
            counts["fused"]["fn"].extend(m_fused["fn"].cpu().tolist())

            pred_comb = (torch.sigmoid(logits_comb) > 0.5).float()
            m_comb = compute_batch_metrics(pred_comb, masks)
            counts["comb"]["tp"].extend(m_comb["tp"].cpu().tolist())
            counts["comb"]["fp"].extend(m_comb["fp"].cpu().tolist())
            counts["comb"]["fn"].extend(m_comb["fn"].cpu().tolist())

    # Compile Final Aggregate Metrics
    metrics_summary = {
        "dataset_split": args.split,
        "total_samples": len(ds),
        "prompt_source": prompt_source,
        "fusion_enabled": fusion_enabled,
        "base": calculate_final_metrics(counts["base"]["tp"], counts["base"]["fp"], counts["base"]["fn"])
    }

    if fusion_enabled:
        metrics_summary["fused"] = calculate_final_metrics(counts["fused"]["tp"], counts["fused"]["fp"], counts["fused"]["fn"])
        metrics_summary["combined"] = calculate_final_metrics(counts["comb"]["tp"], counts["comb"]["fp"], counts["comb"]["fn"])
        metrics_summary["fusion_configuration"] = {
            "mode": fusion_mode,
            "alpha": fusion_alpha,
            "beta": fusion_beta,
            "lambda_logits": lambda_logits
        }

    # Save Output Artifacts
    ensure_dir(args.out_dir)
    json_path = os.path.join(args.out_dir, f"{args.out_name}.json")
    txt_path = os.path.join(args.out_dir, f"{args.out_name}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Format plain text report summary
    report_lines = [
        "==================================================",
        f"        DATASET EVALUATION REPORT ({args.split.upper()} SPLIT)",
        "==================================================",
        f"Total Checked Samples : {metrics_summary['total_samples']}",
        f"Prompting Mechanism   : {metrics_summary['prompt_source'].upper()}",
        f"Visual Fusion Layer   : {'ENABLED' if fusion_enabled else 'DISABLED'}",
        "--------------------------------------------------",
        "Performance Metrics Summary:",
        "--------------------------------------------------"
    ]
    
    b = metrics_summary["base"]
    report_lines.append(f"[BASE BRANCH]     Dice: {b['dice']:.4f}  | IoU: {b['iou']:.4f}  | Precision: {b['precision']:.4f}  | Recall: {b['recall']:.4f}")
    
    if fusion_enabled:
        f_m = metrics_summary["fused"]
        c_m = metrics_summary["combined"]
        report_lines.append(f"[FUSED BRANCH]    Dice: {f_m['dice']:.4f}  | IoU: {f_m['iou']:.4f}  | Precision: {f_m['precision']:.4f}  | Recall: {f_m['recall']:.4f}")
        report_lines.append(f"[COMBINED OUTPUT] Dice: {c_m['dice']:.4f}  | IoU: {c_m['iou']:.4f}  | Precision: {c_m['precision']:.4f}  | Recall: {c_m['recall']:.4f}")
        report_lines.append("--------------------------------------------------")
        report_lines.append(f"Gating Configuration Parameters: Mode={fusion_mode}, Alpha={fusion_alpha}, Lambda={lambda_logits}")
    
    report_lines.append("==================================================")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[Success] Evaluation complete!")
    print(f"📊 JSON Metrics Matrix: {json_path}")
    print(f"📄 Text Summary Report: {txt_path}\n")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
