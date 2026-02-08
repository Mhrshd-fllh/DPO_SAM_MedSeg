import argparse
import torch
from prompts.visual.load_biomedclip import load_biomedclip

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contains", default="visual", help="substring filter")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = load_biomedclip(device=device)

    for name, _ in model.named_modules():
        if args.contains in name:
            print(name)

if __name__ == "__main__":
    main()
