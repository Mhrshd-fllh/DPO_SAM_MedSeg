from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union, Any
import torch
from PIL import Image

@dataclass
class VQAResult:
    answers: List[str]
    scores: Optional[List[float]] = None
    raw_outputs: Optional[Any] = None  

class MedVInTAdapter:
    """
    True Image-to-Text Medical VQA Adapter using 'MILVLG/biomed-vlp-blip-large'.
    Fuses visual pixels and clinical questions natively without any subfolder bugs.
    """
    def __init__(
        self,
        model_id: str = "edgeun/blip-medical-vqa-rad",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 50,
    ):
        from transformers import BlipProcessor, BlipForQuestionAnswering

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype if torch_dtype else (torch.float16 if "cuda" in device else torch.float32)

        try:
            print(f"[📋 VQA VLM] Loading True Image-Text Model: {model_id}...")
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForQuestionAnswering.from_pretrained(
                model_id, 
                torch_dtype=self.torch_dtype
            ).to(device)
            
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
                
            print("🚀 [VQA Success] Medical Vision-Language Model is fully loaded!")
        except Exception as e:
            raise RuntimeError(f"Failed to load Blip Medical VLM model. Error: {e}")

    def _process_image(self, img: Union[Image.Image, torch.Tensor]) -> Image.Image:
        """Ensure image is a valid PIL Image for the BlipProcessor."""
        if isinstance(img, torch.Tensor):
            x = img.detach().cpu()
            if x.dim() != 3: 
                raise ValueError("Tensor image must be CHW")
            if x.shape[0] == 1: 
                x = x.repeat(3, 1, 1)
            if x.max() <= 1.0: 
                x = (x * 255.0).clamp(0, 255)
            img = Image.fromarray(x.byte().permute(1, 2, 0).numpy())
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        return img

    def infer(self, images: List[Union[Image.Image, torch.Tensor]], questions: List[str]) -> VQAResult:
        """
        Natively generates answers by looking at the ultrasound image and reading the question.
        """
        if len(images) != len(questions):
            raise ValueError(f"Images and questions length mismatch: {len(images)} vs {len(questions)}")

        answers: List[str] = []
        scores: List[float] = []
        all_raw_outputs = [] 

        with torch.no_grad():
            for img, question in zip(images, questions):
                pil_img = self._process_image(img)
                
                inputs = self.processor(
                    images=pil_img, 
                    text=question, 
                    return_tensors="pt"
                ).to(self.device).to(self.torch_dtype)
                
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens
                )
                
                all_raw_outputs.append(outputs) 
                
                answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
                
                if not answer:
                    answer = "inconclusive"
                    score = 0.0
                else:
                    score = 1.0
                    
                answers.append(answer)
                scores.append(score)

        return VQAResult(answers=answers, scores=scores, raw_outputs=all_raw_outputs)
