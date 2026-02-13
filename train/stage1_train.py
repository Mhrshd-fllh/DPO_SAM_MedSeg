from __future__ import annotations
import os
import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from core.config import load_config
from data.dataloader import build_loaders
from models.load_sam_med2d import load_sam_model
from models.konwer_sam2d import KonwerSAM2D
from losses.combo import DiceFocalCombo
from eval.metrics import dice_coeff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--datasets", default="configs/datasets.yaml")
    ap.add_argument("--train_cfg", default="configs/train.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config, args.prompts, args.datasets, args.train_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = build_loaders(cfg)

    sam = load_sam_model(
        checkpoint_path=cfg["sam"]["checkpoint"],
        model_type=cfg["sam"]["model_type"],
        device=device,
    )
    model = KonwerSAM2D(sam).to(device)

    # loss
    crit = DiceFocalCombo(dice_w=cfg["train"]["loss"]["dice_w"], focal_w=cfg["train"]["loss"]["focal_w"])

    # optimizer + scheduler
    opt = AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    sch = StepLR(opt, step_size=int(cfg["train"]["lr_step"]), gamma=float(cfg["train"]["lr_gamma"]))

    epochs = int(cfg["train"]["epochs"])
    out_dir = cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for batch in train_loader:
            images = batch.image.to(device)
            masks = batch.mask.to(device)

            # ⚠️ فعلاً visual prompts باید از pipeline ساخته بشه.
            # در این برنچ، فقط wiring مدل رو زدیم؛ در برنچ بعدی prompt builder رو وارد train می‌کنیم.
            raise RuntimeError(
                "Stage-1 train requires visual prompts. Next branch will integrate PromptBundleBuilder "
                "and VisualPromptPipeline into the training loop."
            )

        sch.step()

        # (eval placeholder)
        model.eval()
        with torch.no_grad():
            pass

        print(f"Epoch {ep}/{epochs} done.")

    print("Done.")


if __name__ == "__main__":
    main()
