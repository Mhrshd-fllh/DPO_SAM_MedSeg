from __future__ import annotations
import open_clip
import torch

def load_biomedclip(device: str = "cuda"):
    """
    Loads Microsoft BiomedCLIP via open_clip HF hub.
    Returns: (model, preprocess, tokenizer)
    """
    model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, preprocess = open_clip.create_model_from_pretrained(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    model = model.to(device).eval()
    return model, preprocess, tokenizer
