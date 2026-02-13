from __future__ import annotations
from typing import List, Optional

import torch

from core.types import PromptBundle
from prompts.visual.visual_prompt_pipeline import VisualPromptPipeline
from prompts.text.text_prompt_pipeline import TextPromptPipeline


class PromptBundleBuilder:
    def __init__(self, visual: VisualPromptPipeline, text: TextPromptPipeline):
        self.visual = visual
        self.text = text

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        class_texts: List[str],
        labels: Optional[List[str]] = None,
    ) -> PromptBundle:
        vp = self.visual(images, class_texts)
        tp = self.text(images=images, labels=labels)
        return PromptBundle(visual=vp, text=tp)
