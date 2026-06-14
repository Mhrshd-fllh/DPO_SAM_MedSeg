from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
from PIL import Image


@dataclass
class VQAResult:
    answers: List[str]
    scores: Optional[List[float]] = None
    raw_outputs: Optional[List[Any]] = None


class HFVQAAdapter:
    """
    Real VQA adapter using HuggingFace transformers pipeline("visual-question-answering").

    Default model is a runnable general VQA model. For paper-faithful medical VQA,
    change `model_id` to your MedVInT/PMC-VQA checkpoint that supports the same task.
    """
    def __init__(
        self,
        model_id: str = "Salesforce/blip-vqa-base",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        from transformers import pipeline  # transformers<5 recommended

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # HF pipeline uses device index for cuda, -1 for cpu
        device_idx = 0 if device.startswith("cuda") else -1

        self.model_id = model_id
        self.device = device

        # dtype handling
        if torch_dtype is None:
            if device.startswith("cuda"):
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        self.pipe = pipeline(
            task="image-text-to-text",
            model=model_id,
            device=device_idx,
        )

    @staticmethod
    def _extract_answer(out: Any) -> tuple[str, float]:
        """
        Different HF VQA/image-text pipelines return different keys.
        BLIP VQA with image-text-to-text commonly returns generated_text,
        while visual-question-answering returns answer/score.
        """
        if isinstance(out, list):
            if len(out) == 0:
                return "", 0.0
            out = out[0]

        if isinstance(out, str):
            return out.strip(), 0.0

        if not isinstance(out, dict):
            return str(out).strip(), 0.0

        answer = ""
        for key in ("answer", "generated_text", "text", "caption"):
            value = out.get(key)
            if value is not None:
                answer = str(value).strip()
                break

        score_value = out.get("score", 0.0)
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0

        return answer, score

    def infer(self, images: List[Union[Image.Image, torch.Tensor]], questions: List[str]) -> VQAResult:
        """
        images: list of PIL.Image (recommended). torch.Tensor can be converted externally if needed.
        questions: list[str] length B
        """
        if len(images) != len(questions):
            raise ValueError(f"images and questions must have same length. got {len(images)} vs {len(questions)}")

        answers: List[str] = []
        scores: List[float] = []
        raw_outputs: List[Any] = []

        for img, q in zip(images, questions):
            if isinstance(img, torch.Tensor):
                # expect CHW in [0,1] or [0,255]
                x = img.detach().cpu()
                if x.dim() != 3:
                    raise ValueError("Tensor image must be CHW")
                if x.shape[0] == 1:
                    x = x.repeat(3, 1, 1)
                if x.max() <= 1.0:
                    x = (x * 255.0).clamp(0, 255)
                x = x.byte().permute(1, 2, 0).numpy()
                img = Image.fromarray(x)

            out = self.pipe(images=img, text=q)
            raw_outputs.append(out)
            ans, sc = self._extract_answer(out)
            answers.append(ans)
            scores.append(sc)

        return VQAResult(answers=answers, scores=scores, raw_outputs=raw_outputs)
