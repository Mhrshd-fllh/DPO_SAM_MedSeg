from __future__ import annotations
import argparse
import numpy as np
import torch

from prompts.visual.load_biomedclip import load_biomedclip
from prompts.visual.biomedclip_gscorecam import BiomedCLIPAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contains", default="visual", help="only inspect modules whose name contains this")
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_biomedclip(device=device)
    clip = BiomedCLIPAdapter(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)

    # dummy input (random image) in the exact format that encode_image expects
    x = torch.randn(1, 3, 224, 224, device=device)

    stats = []

    hooks = []
    def hooker(name):
        def _h(_, __, out):
            # handle tuples/lists
            if isinstance(out, (tuple, list)):
                out0 = out[0]
            else:
                out0 = out
            if not torch.is_tensor(out0):
                return
            o = out0.detach()
            # we want spatial-ish activations for CAM: either [B,C,H,W] or [B,T,C]
            if o.dim() not in (3, 4):
                return
            # compute non-triviality
            m = float(o.mean().item())
            s = float(o.std().item())
            mx = float(o.abs().max().item())
            shape = tuple(o.shape)
            stats.append((name, shape, m, s, mx))
        return _h

    for name, module in clip.model.named_modules():
        if args.contains and args.contains not in name:
            continue
        hooks.append(module.register_forward_hook(hooker(name)))

    with torch.no_grad():
        _ = clip.model.encode_image(x)

    for h in hooks:
        h.remove()

    # sort: prefer higher std and non-zero max, and “more spatial” shapes
    def score(item):
        _, shape, _, s, mx = item
        spatial_bonus = 1.0
        if len(shape) == 4:
            spatial_bonus = 2.0
        return (spatial_bonus * s, mx)

    stats_sorted = sorted(stats, key=score, reverse=True)

    print("\nTop candidate target layers for CAM:")
    for i, (name, shape, m, s, mx) in enumerate(stats_sorted[:args.topk]):
        print(f"{i:02d}  {name:60s}  shape={shape}  mean={m:.4g}  std={s:.4g}  absmax={mx:.4g}")

    if len(stats_sorted) > 0:
        print("\nSuggested --target_layer:")
        print(stats_sorted[0][0])

if __name__ == "__main__":
    main()
