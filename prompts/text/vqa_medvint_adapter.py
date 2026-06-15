from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from PIL import Image


@dataclass
class VQAResult:
    answers: List[str]
    scores: Optional[List[float]] = None


class MedVInTAdapter:
    """
    MedVInT (Medical Visual Instruction Tuned) VQA adapter.
    
    Loads a MedVInT model checkpoint and performs medical visual question answering.
    Uses the official HuggingFace checkpoint: xmcmic/MedVInT-TE
    
    Args:
        model_id: HuggingFace model ID or local checkpoint path
                 Default: "xmcmic/MedVInT-TE" (official MedVInT checkpoint)
        device: Device to load model on ("cuda" or "cpu")
        torch_dtype: Precision for model (float32, float16, bfloat16)
        max_new_tokens: Max tokens to generate in answer
    """
    def __init__(
        self,
        model_id: str = "xmcmic/MedVInT-TE",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 100,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens

        if torch_dtype is None:
            torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.torch_dtype = torch_dtype

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
            )
            self.model.eval()
            
            # Freeze all model parameters
            for param in self.model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MedVInT model from '{model_id}'. "
                f"Ensure the checkpoint exists and is compatible. Error: {e}"
            )

    def _process_image(self, img: Union[Image.Image, torch.Tensor]) -> Image.Image:
        """Convert tensor to PIL Image if needed."""
        if isinstance(img, torch.Tensor):
            x = img.detach().cpu()
            if x.dim() != 3:
                raise ValueError("Tensor image must be CHW")
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            if x.max() <= 1.0:
                x = (x * 255.0).clamp(0, 255)
            x = x.byte().permute(1, 2, 0).numpy()
            img = Image.fromarray(x)
        return img

    def infer(self, images: List[Union[Image.Image, torch.Tensor]], questions: List[str]) -> VQAResult:
        """
        Generate VQA answers using MedVInT.
        
        Args:
            images: List of PIL Images or torch tensors [C,H,W]
            questions: List of question strings
            
        Returns:
            VQAResult with answers and scores
        """
        if len(images) != len(questions):
            raise ValueError(f"images and questions must have same length. got {len(images)} vs {len(questions)}")

        answers: List[str] = []
        scores: List[float] = []

        with torch.no_grad():
            for img, question in zip(images, questions):
                img = self._process_image(img)
                
                prompt = f"<image>\n{question}"
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    top_p=0.95,
                    temperature=0.7,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                
                generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
                answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                score = 1.0 if answer else 0.0
                answers.append(answer)
                scores.append(score)

        return VQAResult(answers=answers, scores=scores)
