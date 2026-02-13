from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any

import torch
from PIL import Image

from core.types import TextPrompts
from prompts.text.vqa_medvint_adapter import HFVQAAdapter
from prompts.text.gpt4_adapter import OpenAIGPTAdapter


@dataclass
class TextPromptConfig:
    question_template: str = "Question: {}, Answer is:"
    question: str = "What is the shape of breast tumor and where is it located?"
    vqa_enabled: bool = True
    gpt_enabled: bool = False

    # VQA
    vqa_model_id: str = "Salesforce/blip-vqa-base"

    # GPT
    gpt_model: str = "gpt-4o-mini"


def _to_pil_list(images: torch.Tensor) -> List[Image.Image]:
    """
    images: [B,3,H,W] float in [0,1]
    """
    imgs = []
    x = images.detach().cpu()
    for i in range(x.shape[0]):
        im = x[i]
        im = (im * 255.0).clamp(0, 255).byte()
        im = im.permute(1, 2, 0).numpy()
        imgs.append(Image.fromarray(im))
    return imgs


class TextPromptPipeline:
    """
    Real pipeline:
      - VQA answer from an actual VQA model
      - GPT generic description from OpenAI (optional)
      - concatenate them
    """
    def __init__(self, cfg: TextPromptConfig, device: Optional[str] = None):
        self.cfg = cfg

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.vqa = None
        self.gpt = None

        if cfg.vqa_enabled:
            self.vqa = HFVQAAdapter(model_id=cfg.vqa_model_id, device=device)

        if cfg.gpt_enabled:
            self.gpt = OpenAIGPTAdapter(model=cfg.gpt_model)

    def __call__(self, images: torch.Tensor, labels: Optional[List[str]] = None) -> TextPrompts:
        """
        images: [B,3,H,W] float in [0,1]
        labels: optional list[str] length B (used by GPT)
        """
        B = images.shape[0]
        labels = labels or ["unknown"] * B

        # Build questions
        q = self.cfg.question_template.format(self.cfg.question)
        questions = [q for _ in range(B)]

        # VQA
        vqa_answers = [""] * B
        if self.cfg.vqa_enabled:
            pil_images = _to_pil_list(images)
            vqa_res = self.vqa.infer(images=pil_images, questions=questions)
            vqa_answers = [a.strip() for a in vqa_res.answers]

        # GPT
        gpt_descs = [""] * B
        if self.cfg.gpt_enabled:
            gpt_res = self.gpt.describe(labels)
            gpt_descs = [d.strip() for d in gpt_res.descriptions]

        # Concatenate
        out_text: List[str] = []
        for a, d in zip(vqa_answers, gpt_descs):
            if a and d:
                out_text.append(a + " " + d)
            elif a:
                out_text.append(a)
            elif d:
                out_text.append(d)
            else:
                out_text.append("")

        return TextPrompts(text=out_text)
